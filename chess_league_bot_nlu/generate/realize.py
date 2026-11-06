"""Ask the model to write messages for plans, with placeholders where times and arguments go.

The model never writes a time or an argument itself: it writes {T1} or
{player}, and code fills in the planned text afterwards, so every labelled span
is exact by construction.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

from ..records import opponent
from . import catalog, llm
from .command_plan import CommandPlan
from .errors import Unwritable
from .placeholders import PLACEHOLDER
from .plan import PlannedTime, Step, ThreadPlan
from .wording import written, written_clock, written_day, written_local

LIST = "{LIST}"
OPPONENT = "OPPONENT"


@dataclass
class Draft:
    """One message to write: its line in the plan, and what its placeholders stand for.

    Some of the text is laid out in code rather than by the writer: a list of times, one per
    line, where the writer put {LIST}, and times struck through in an edit, before the time
    that replaced them."""

    line: str
    values: dict[str, str]
    words: int
    allow_weekday: bool = False
    ids: dict[str, str] = field(default_factory=dict)
    listed: list[str] = field(default_factory=list)
    bullet: str = ""
    after_list: str = "\n"
    struck: dict[str, str] = field(default_factory=dict)
    optional: list[str] = field(default_factory=list)

    @property
    def placeholders(self) -> list[str]:
        """The placeholders the writer's text must use once each, once its list is laid out."""
        return [name for name in self.values if name not in self.struck.values() and name not in self.optional]

    def mentions(self, spans: dict[str, tuple[int, int]]) -> dict[str, tuple[int, int]]:
        """Filled spans by the id of the time each placeholder stands for."""
        return {self.ids.get(name, name): span for name, span in spans.items()}

    def laid_out(self, template: str) -> str:
        """The writer's text with its {LIST} replaced by the listed placeholders, one per line.
        Slips are mended first: an @ before {OPPONENT}, which is an @-mention already, and a
        placeholder written without its braces ("T4 is too late")."""
        template = template.replace("@{" + OPPONENT + "}", "{" + OPPONENT + "}")
        for name in self.values:
            bare = re.compile(rf"(?<![{{\w]){re.escape(name)}(?![}}\w])")
            if f"{{{name}}}" not in template and len(bare.findall(template)) == 1:
                template = bare.sub(f"{{{name}}}", template)
        if not self.listed or LIST not in template:
            return template
        before, _, after = template.partition(LIST)
        before = before.rstrip(" \t")
        after = re.sub(r"^[ \t.,;:!?]+", "", after)
        lines = "\n".join(f"{self.bullet}{{{name}}}" for name in self.listed)
        before = before + "\n" if before and not before.endswith("\n") else before
        after = self.after_list + after if after and not after.startswith("\n") else after
        return before + lines + after

    def edited(self, template: str) -> str:
        """The text with each struck-through time put before the time that replaced it."""
        for name, struck in self.struck.items():
            template = template.replace(f"{{{name}}}", f"~~{{{struck}}}~~ {{{name}}}", 1)
        return template


def describe(time: PlannedTime) -> str:
    days = " or ".join(f"{start:%a %-d %b}" for start, _ in time.slots())
    if time.end:
        return f"{days} {time.start:%H:%M}-{time.end:%H:%M} UTC"
    return f"{days} {time.start:%H:%M} UTC" + (" or later" if time.open else "")


def instruction(plan: ThreadPlan, step: Step, draft: Draft) -> str:
    beats = catalog.beats()
    names = {time.form or time.role: f"{{{time.placeholder}}}" for time in step.times if not time.struck_for}
    offered = [f"{{{time.placeholder}}}" for time in step.times if time.role == "offered"]
    note = f" ({step.note})" if step.note else ""
    if step.kind == "offer" and "day" in names:
        text = beats["offer_same_day"].format(day=names["day"], time=names["clock"], note=note)
    elif step.kind == "offer" and draft.listed:
        text = beats["offer_list"].format(count=len(draft.listed), note=note)
    elif step.kind == "offer":
        text = beats["offer"].format(times=", ".join(offered), note=note)
    elif step.kind == "accept":
        offer = plan.offer(step.accepts)
        if step.style in ("restate", "partial", "pick"):
            key = f"accept_{step.style}"
        else:
            key = "accept_bare_reference" if step.reference else "accept_bare"
        accepted = next((time for time in step.times if time.role == "accepted"), None)
        text = beats[key].format(
            time=f"{{{accepted.placeholder}}}" if accepted else "",
            offer=describe(offer),
            reference=f"'{step.reference}'",
            note=note,
        )
        rejected = [time for time in step.times if time.role == "other" and time.form != "local"]
        if rejected:
            text = beats["rejecting"].format(time=f"{{{rejected[0].placeholder}}}") + text
    elif step.kind == "retract":
        key = "retract_restate" if step.times else "retract"
        offer = plan.offer(step.retracts)
        text = beats[key].format(time=names.get("other", ""), offer=describe(offer), note=note)
    elif step.kind in ("withdraw", "mistaken_accept", "correct"):
        text = beats[step.kind].format(time=names.get("offered") or names.get("other"), note=note)
    else:
        text = beats[step.kind].format(note=step.note)
    if "local" in names:
        text += beats["local"].format(time=names["local"])
    return text


def step_draft(plan: ThreadPlan, step: Step, rng: random.Random, mention: bool) -> Draft:
    beats, layout, formats = catalog.beats(), catalog.layout(), catalog.time_formats()
    window = (plan.published, plan.deadline)
    values: dict[str, str] = {}
    previous: PlannedTime | None = None
    struck = {time.struck_for: time.placeholder for time in step.times if time.struck_for}
    writer_times = [time for time in step.times if not time.struck_for]
    offered = [time for time in writer_times if time.role == "offered"]
    listed = (
        step.kind == "offer"
        and len(offered) >= 2
        and len(offered) == len(writer_times)
        and (step.layout == "list" or (not step.layout and rng.random() < layout.list_chance))
    )
    for time in step.times:
        text = None
        if time.form == "day":
            text = written_day(time, rng)
        elif time.form == "local":
            text = written_local(time, rng)
        elif time.form == "clock" or (
            not listed
            and step.kind == "offer"
            and previous is not None
            and previous.role == time.role == "offered"
            and time.placeholder not in struck
            and same_day(previous, time)
            and rng.random() < formats.clock_only_chance
        ):
            context = values[previous.placeholder] if previous else ""
            text = written_clock(time, step.sent, window, rng, context)
            if text is None and time.form == "clock":
                raise Unwritable(f"no clock alone reads as {time}")
        values[time.placeholder] = text or written(time, step.sent, window, rng)
        if not time.struck_for:
            previous = time
    bullet = rng.choices(list(layout.bullets), list(layout.bullets.values()))[0]
    draft = Draft(
        line="",
        values=values,
        words=step.words,
        allow_weekday=step.reference_kind == "day",
        ids={time.placeholder: time.id for time in step.times},
        listed=[time.placeholder for time in offered] if listed else [],
        bullet=bullet,
        after_list="\n\n" if rng.random() < layout.blank_line_after_list else "\n",
        struck=struck,
    )
    # A time said again that labels nothing (a day kept, a time turned down or taken back, a
    # conversion) may be left out; so may an @-mention, which writers add unasked too.
    draft.optional += [
        time.placeholder
        for time in step.times
        if time.role == "other" and step.kind in ("offer", "accept", "retract") and not time.struck_for
    ]
    values[OPPONENT] = f"@**{plan.names[opponent(step.speaker)]}**"
    extra = ""
    if mention:
        extra += beats["mention"]
    else:
        draft.optional.append(OPPONENT)
    if step.words >= layout.paragraph_min_words and rng.random() < layout.paragraph_chance:
        extra += beats["paragraphs"]
    # Only offers say what their placeholders mean: shown a time to accept, the writer tends to copy it out.
    meanings = (
        ""
        if listed
        else "; ".join(f"{{{time.placeholder}}} = {describe(time)}" for time in offered if step.kind == "offer")
    )
    draft.line = beats["line"].format(
        speaker=step.speaker,
        persona=plan.personas[step.speaker].short,
        sent=f"{step.sent:%a %-d %b %H:%M}",
        words=step.words,
        instruction=instruction(plan, step, draft) + extra,
        meanings=f" [{meanings}]" if meanings else "",
    )
    return draft


def same_day(previous: PlannedTime, time: PlannedTime) -> bool:
    """Whether a time could borrow its day from the single time written just before it."""
    single = not previous.more_days and not time.more_days and not time.end and not time.open
    return single and previous.zone == time.zone and previous.local_start().date() == time.local_start().date()


def thread_prompt(plan: ThreadPlan, framing: str, rng: random.Random) -> tuple[str, str, list[Draft]]:
    prompts, layout = catalog.prompts("threads"), catalog.layout()
    drafts, spoken = [], set()
    for step in plan.steps:
        mention = step.speaker not in spoken and rng.random() < layout.mention_chance
        spoken.add(step.speaker)
        drafts.append(step_draft(plan, step, rng, mention))
    user = prompts.task.format(
        league=plan.league.description,
        round=plan.round,
        published=f"{plan.published:%A %-d %B %H:%M}",
        deadline=f"{plan.deadline:%A %-d %B %H:%M}",
        plan="\n".join(f"{index}. {draft.line}" for index, draft in enumerate(drafts, start=1)),
        count=len(drafts),
    )
    return prompts.framings[framing].system, user, drafts


def command_draft(plan: CommandPlan) -> Draft:
    settings = catalog.commands()
    what = plan.intent.description
    if plan.none_kind:
        what = settings.none_what.format(kind=plan.none_kind)
    placeholders = ", ".join(f"{{{label}}}" for label in plan.arguments)
    line = settings.line.format(
        what=what,
        event=settings.event.format(description=settings.event_descriptions[plan.event]) if plan.event else "",
        include=settings.include.format(placeholders=placeholders) if placeholders else settings.no_placeholders,
        angle=settings.angle.format(angle=plan.angle) if plan.angle else "",
        style=plan.style.description,
        words=plan.words,
    )
    return Draft(line, dict(plan.arguments), plan.words)


def command_prompt(plans: list[CommandPlan], framing: str) -> tuple[str, str, list[Draft]]:
    prompts = catalog.prompts("commands")
    drafts = [command_draft(plan) for plan in plans]
    lines = "\n".join(f"{index}. {draft.line}" for index, draft in enumerate(drafts, start=1))
    return prompts.framings[framing].system, prompts.task.format(count=len(drafts), lines=lines), drafts


def write(system: str, user: str, count: int, temperature: float) -> list[str]:
    reply = llm.json_from(llm.chat(system, user, temperature, max_tokens=1600))
    if not isinstance(reply, list) or len(reply) != count or not all(isinstance(text, str) for text in reply):
        raise llm.BadReply(f"expected a list of {count} strings")
    return [text.strip() for text in reply]


def fill(template: str, values: dict[str, str]) -> tuple[str, dict[str, tuple[int, int]]]:
    """Put the planned text in place of each placeholder, recording where it went."""
    text, spans, cursor = "", {}, 0
    for match in PLACEHOLDER.finditer(template):
        name = match.group(0)[1:-1]
        text += template[cursor : match.start()]
        start = len(text)
        text += values[name]
        spans[name] = (start, len(text))
        cursor = match.end()
    return text + template[cursor:], spans
