"""Records from plans and the text written for them. Every label comes from the plan."""

from __future__ import annotations

from datetime import date

from .. import dates
from ..records import (
    Acceptance,
    Argument,
    Command,
    Mention,
    Message,
    Retraction,
    Thread,
    Utc,
    Withdrawal,
    format_time,
    monday_of,
)
from .command_plan import CommandPlan
from .errors import Rejected
from .plan import ThreadPlan


def thread_record(record_id: str, plan: ThreadPlan, texts: list[tuple[str, dict[str, tuple[int, int]]]]) -> Thread:
    messages, mentions, acceptances, withdrawals, retractions = [], [], [], [], []
    week = monday_of(plan.published.date())
    for index, (step, (text, spans)) in enumerate(zip(plan.steps, texts, strict=True), start=1):
        message_id = f"m{index}"
        text, template, spans = placed(
            text,
            spans,
            dates.round_days(plan.published.date(), plan.deadline.date(), step.sent.date()),
            week,
        )
        messages.append(Message(message_id, step.speaker, format_time(step.sent), text, template))
        for time in step.times:
            if time.id not in spans:
                continue  # an optional time the writer left out
            utc = [Utc(format_time(begin), format_time(end) if end else None, time.open) for begin, end in time.slots()]
            mentions.append(Mention(time.id, message_id, *spans[time.id], time.role, utc, time.accepts))
        if step.accepts:
            acceptances.append(Acceptance(message_id, step.accepts, format_time(step.pick) if step.pick else None))
        if step.withdraws:
            withdrawals.append(Withdrawal(message_id, step.withdraws))
        if step.retracts:
            retractions.append(Retraction(message_id, step.retracts))
    thread = Thread(
        id=record_id,
        messages=messages,
        mentions=mentions,
        acceptances=acceptances,
        withdrawals=withdrawals,
        retractions=retractions,
        plot=plan.plot,
        week=week,
    )
    thread.validate()
    return thread


def placed(
    text: str, spans: dict[str, tuple[int, int]], days: list[date], week: date
) -> tuple[str, str, dict[str, tuple[int, int]]]:
    """A written message (its times' dates as placeholders) as its text, its template with every
    date in it named by days after the week's Monday, and its times' spans in the text."""
    rendered = dates.rendered(text)
    spans = {name: (rendered.in_text(start), rendered.in_text(end)) for name, (start, end) in spans.items()}
    template = dates.templated(text, days)
    if left := dates.unplaced(template, days):
        raise Rejected(f"a date that can't be placed: {', '.join(left)}")
    return rendered.text, dates.relative(template, week), spans


def command_record(record_id: str, plan: CommandPlan, text: str, spans: dict[str, tuple[int, int]]) -> Command:
    command = Command(
        id=record_id,
        text=text,
        intent=plan.intent.key,
        arguments=[Argument(label, *spans[label]) for label in plan.arguments],
        event=plan.event,
        style=plan.style.name,
        none_kind=plan.none_kind.kind if plan.none_kind else None,
    )
    command.validate()
    return command
