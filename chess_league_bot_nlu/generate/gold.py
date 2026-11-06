"""Records from plans and the text written for them. Every label comes from the plan."""

from __future__ import annotations

from ..records import Acceptance, Argument, Command, Mention, Message, Retraction, Thread, Utc, Withdrawal, format_time
from .command_plan import CommandPlan
from .plan import ThreadPlan


def thread_record(
    record_id: str, plan: ThreadPlan, texts: list[tuple[str, dict[str, tuple[int, int]]]], provenance: dict
) -> Thread:
    messages, mentions, acceptances, withdrawals, retractions = [], [], [], [], []
    for index, (step, (text, spans)) in enumerate(zip(plan.steps, texts, strict=True), start=1):
        message_id = f"m{index}"
        messages.append(Message(message_id, step.speaker, format_time(step.sent), text))
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
    personas = {player: persona.description for player, persona in plan.personas.items()}
    thread = Thread(
        id=record_id,
        league=plan.league.name,
        pairing_published=format_time(plan.published),
        deadline=format_time(plan.deadline),
        messages=messages,
        mentions=mentions,
        acceptances=acceptances,
        withdrawals=withdrawals,
        retractions=retractions,
        provenance={"plot": plan.plot, "personas": personas, **provenance},
    )
    thread.validate()
    return thread


def command_record(
    record_id: str, plan: CommandPlan, text: str, spans: dict[str, tuple[int, int]], provenance: dict
) -> Command:
    details = {"style": plan.style.name}
    if plan.none_kind:
        details["none_kind"] = plan.none_kind
    if plan.angle:
        details["angle"] = plan.angle
    command = Command(
        id=record_id,
        text=text,
        intent=plan.intent.key,
        arguments=[Argument(label, *spans[label]) for label in plan.arguments],
        event=plan.event,
        provenance={**details, **provenance},
    )
    command.validate()
    return command
