"""An independent reading of finished text, which must agree with the plan.

The reader never sees the plan. For threads it says which messages accept
which times, and which take a time back (an offer or an acceptance); for commands it classifies each message. Records it
disagrees with are dropped.

A second check, QA, does see the labels: it is asked whether each offer is
really an offer. It doesn't judge acceptances or withdrawals, which the reader
and the resolver already check, and which it gets wrong when they restate the
time ("tue 13:00" for "the 10th 1pm").
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from ..labels import INTENTS
from ..records import Command, Retraction, Thread, Withdrawal, parse_time
from ..resolve import clocks, resolve, weekday_of
from . import catalog, llm


@dataclass(frozen=True)
class Meaning:
    """When a planned time is, and the message it was mentioned in: enough to read a quote of it."""

    start: datetime
    sent: datetime
    window: tuple[datetime, datetime]


def same_time(quoted: str, written: list[str], meaning: Meaning | None = None, only_on_its_day: bool = False) -> bool:
    """Whether the reader quoted the planned time: one of the ways it was written, all of it and
    no other time with it ("thu 1" is not "thu 13:00"), or, given its `meaning`, any quote that
    reads as exactly that time ("21:00" for "21:00 tmrw"), with its day from the last way written:
    the accepting message's own, when it restates the time. With `only_on_its_day` (no other offer of
    that player's stands on its day), a quote with no clock that names the day counts too."""
    if only_on_its_day and meaning is not None and not clocks(quoted) and weekday_of(quoted) == meaning.start.weekday():
        return True  # the day alone, as a message that points at its offer by day says it ("wednesday evening")
    if meaning is not None and 1 <= len(clocks(quoted)) <= 2:
        found = resolve(quoted, meaning.sent, meaning.window, context=written[-1])
        one_time = len(clocks(quoted)) == 1 or (found and found[0].end is not None)
        if len(found) == 1 and found[0].start == meaning.start and one_time:
            return True
    for text in written:
        pattern = r"\s+".join(re.escape(part) for part in text.split())
        if re.search(rf"(?<![a-z0-9]){pattern}(?![0-9])", quoted, re.IGNORECASE) and len(clocks(quoted)) == len(
            clocks(text)
        ):
            return True
    return False


def thread_problems(thread: Thread, window: tuple[datetime, datetime]) -> list[str]:
    prompts = catalog.prompts("threads")
    numbered = "\n".join(
        f"{index}. {message.speaker}: {message.text}" for index, message in enumerate(thread.messages, start=1)
    )
    reply = llm.json_from(llm.chat(prompts.verify_system, prompts.verify.format(transcript=numbered), 0.0, 600))
    if not isinstance(reply, dict):
        return ["the reader's reply is not an object"]
    number = {message.id: index for index, message in enumerate(thread.messages, start=1)}
    mentions = {mention.id: mention for mention in thread.mentions}

    picks = {acceptance.offer: acceptance.time for acceptance in thread.acceptances}

    def meaning(item) -> Meaning:
        """The time an acceptance agreed to (or one taken back had), or an offer's first."""
        offer = mentions[item.offer]
        start = getattr(item, "time", None) or offer.times[0].start
        if isinstance(item, Retraction) or (isinstance(item, Withdrawal) and item.offer in picks):
            start = picks.get(item.offer) or start
        return Meaning(parse_time(start), parse_time(thread.message(offer.message).sent), window)

    def ways_written(offer_id: str, message_id: str) -> list[str]:
        """How the reader may quote the time: the offer as written, or as restated in the message
        (for taking back an acceptance, the agreed time it restates)."""
        offer = mentions[offer_id]
        retracting = any(item.message == message_id for item in thread.retractions) or any(
            item.message == message_id and item.offer in picks for item in thread.withdrawals
        )
        quoted = [thread.text_of(offer)]
        quoted += [
            thread.text_of(mention)
            for mention in thread.mentions
            if mention.message == message_id
            and (mention.accepts == offer_id or mention.repeats_of(offer) or (retracting and mention.role == "other"))
        ]
        return quoted

    withdrawn = {item.offer: number[item.message] for item in thread.withdrawals}

    def alone_on_its_day(item) -> bool:
        """Whether the item's offer is the only one its player has standing on its UTC day at that message."""
        offer = mentions[item.offer]
        player, at = thread.message(offer.message).speaker, number[item.message]
        day = parse_time(offer.times[0].start).date()
        same_day = [
            other
            for other in thread.mentions
            if other.role == "offered"
            and thread.message(other.message).speaker == player
            and number[other.message] < at
            and withdrawn.get(other.id, at + 1) >= at
            and any(parse_time(time.start).date() == day for time in other.times)
        ]
        return len(same_day) == 1

    too_late = {
        number[mention.message]
        for mention in thread.mentions
        if mention.role == "other"
        and any(mention.repeats_of(mentions[offer]) and number[mention.message] > at for offer, at in withdrawn.items())
    }
    found = []
    # Taking back an offer and taking back an acceptance are one mark, so they are one answer.
    for key, expected in (("accepted", thread.acceptances), ("withdrawn", [*thread.withdrawals, *thread.retractions])):
        given = reply.get(key)
        answers = [
            answer
            for answer in (given if isinstance(given, list) else [])
            if isinstance(answer, dict) and str(answer.get("message", "")).isdigit()
        ]
        by_message = {int(answer["message"]): str(answer.get("time", "")) for answer in answers}
        if key == "accepted":
            # Agreeing to a time already taken back reads, fairly, as accepting; it is planned as not.
            by_message = {message: time for message, time in by_message.items() if message not in too_late}
        wanted = {number[item.message]: ways_written(item.offer, item.message) for item in expected}
        if key != "accepted" and thread.plot == "reschedule":
            # Moving an agreed game reads, fairly, as taking the agreed time back.
            by_message = {message: time for message, time in by_message.items() if message in wanted}
        if set(by_message) != set(wanted):
            found.append(f"reader found {key} in messages {sorted(by_message)}, plan has {sorted(wanted)}")
            continue
        meanings = {number[item.message]: meaning(item) for item in expected}
        lone = {number[item.message]: alone_on_its_day(item) for item in expected}
        for message, written in wanted.items():
            if not same_time(by_message[message], written, meanings[message], lone[message]):
                found.append(f"reader says message {message} {key} {by_message[message]!r}, plan says {written[0]!r}")
    return found


def claims(thread: Thread) -> list[str]:
    number = {message.id: index for index, message in enumerate(thread.messages, start=1)}
    return [
        f'message {number[mention.message]}: {thread.message(mention.message).speaker} suggests "{thread.text_of(mention)}"'
        for mention in thread.mentions
        if mention.role == "offered"
    ]


def label_problems(thread: Thread) -> list[str]:
    listed = claims(thread)
    if not listed:
        return []
    prompts = catalog.prompts("threads")
    numbered = "\n".join(
        f"{index}. {message.speaker}: {message.text}" for index, message in enumerate(thread.messages, start=1)
    )
    prompt = prompts.qa.format(
        transcript=numbered, claims="\n".join(f"{index}. {claim}" for index, claim in enumerate(listed, start=1))
    )
    return disputed(listed, llm.json_from(llm.chat(prompts.verify_system, prompt, 0.0, 80 * len(listed) + 100)))


def disputed(listed: list[str], reply: list | dict) -> list[str]:
    """The claims QA calls false, or that it left unanswered."""
    verdicts = {
        answer.get("claim"): answer for answer in (reply if isinstance(reply, list) else []) if isinstance(answer, dict)
    }
    found = []
    for index, claim in enumerate(listed, start=1):
        answer = verdicts.get(index)
        if answer is None or not isinstance(answer.get("true"), bool):
            found.append(f"QA didn't answer: {claim}")
        elif not answer["true"]:
            found.append(f"QA says false: {claim} ({answer.get('why', '')})")
    return found


def command_problems(commands: list[Command]) -> list[list[str]]:
    """What a reader reading each message on its own finds wrong with its plan."""
    return [command_reading(command) for command in commands]


def command_reading(command: Command) -> list[str]:
    """Whether a reader, shown only this message, takes it to mean its planned intent and event,
    and, when it names two or more arguments, which value is which. A reader that sees another
    reading from the same group of look-alike commands rejects it; another reading outside the
    group is fair, since short messages often are a little open."""
    prompts, settings = catalog.prompts("commands"), catalog.commands()
    intents = "\n".join(
        f"- {key}: {intent.description}"
        + (f" ({intent.unlike})" if intent.unlike else "")
        + (f"; often like: {' / '.join(settings.reads_like[key])}" if key in settings.reads_like else "")
        for key, intent in INTENTS.items()
    )
    notifying = [intent for intent in INTENTS.values() if intent.events]
    labels = sorted({argument.label for argument in command.arguments})
    checked = len(command.arguments) >= 2
    prompt = prompts.verify.format(
        intents=intents,
        notify=" or ".join(intent.key for intent in notifying),
        events=", ".join(f'"{event}"' for intent in notifying for event in intent.events),
        arguments=ARGUMENT_QUESTION.format(labels=", ".join(labels)) if checked else "",
        argument_reply=', "arguments": {"label": "quoted text"}' if checked else "",
        message=command.text,
    )
    answer = llm.json_from(llm.chat(prompts.verify_system, prompt, 0.0, 300))
    if not isinstance(answer, dict):
        return ["the reader's reply is not an object"]
    found = []
    if answer.get("intent") != command.intent:
        found.append(f"reader says {answer.get('intent')!r}, plan says {command.intent!r}")
    elif INTENTS[command.intent].events and answer.get("event") != command.event:
        found.append(f"reader says event {answer.get('event')!r}, plan says {command.event!r}")
    siblings = {other for group in settings.contrast_groups if command.intent in group for other in group} - {
        command.intent
    }
    also = [other for other in answer.get("also") or [] if other in siblings]
    if also:
        found.append(f"reader says it could also mean {also}")
    if checked:
        quoted = answer.get("arguments") if isinstance(answer.get("arguments"), dict) else {}
        for argument in command.arguments:
            value = command.text[argument.start : argument.end]
            if squashed(value) not in squashed(str(quoted.get(argument.label, ""))):
                found.append(f"reader quotes {argument.label} as {quoted.get(argument.label)!r}, plan says {value!r}")
    return found


ARGUMENT_QUESTION = "The message names these, each once: {labels}. Quote the text that gives each one.\n\n"


def squashed(text: str) -> str:
    return re.sub(r"[@*\s]+", " ", text).strip().lower()
