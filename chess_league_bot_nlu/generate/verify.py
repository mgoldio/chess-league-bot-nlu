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
from ..records import Command, Retraction, Thread, parse_time
from ..resolve import clocks, resolve
from . import catalog, llm


@dataclass(frozen=True)
class Meaning:
    """When a planned time is, and the message it was mentioned in: enough to read a quote of it."""

    start: datetime
    sent: datetime
    window: tuple[datetime, datetime]


def same_time(quoted: str, written: list[str], meaning: Meaning | None = None) -> bool:
    """Whether the reader quoted the planned time: one of the ways it was written, all of it and
    no other time with it ("thu 1" is not "thu 13:00"), or, given its `meaning`, any quote that
    reads as exactly that time ("21:00" for "21:00 tmrw"), with its day from the last way written:
    the accepting message's own, when it restates the time."""
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


def thread_problems(thread: Thread) -> list[str]:
    prompts = catalog.prompts("threads")
    numbered = "\n".join(
        f"{index}. {message.speaker}: {message.text}" for index, message in enumerate(thread.messages, start=1)
    )
    reply = llm.json_from(llm.chat(prompts.verify_system, prompts.verify.format(transcript=numbered), 0.0, 600))
    if not isinstance(reply, dict):
        return ["the reader's reply is not an object"]
    number = {message.id: index for index, message in enumerate(thread.messages, start=1)}
    mentions = {mention.id: mention for mention in thread.mentions}
    window = (parse_time(thread.pairing_published), parse_time(thread.deadline))

    picks = {acceptance.offer: acceptance.time for acceptance in thread.acceptances}

    def meaning(item) -> Meaning:
        """The time an acceptance agreed to (or one taken back had), or an offer's first."""
        offer = mentions[item.offer]
        start = getattr(item, "time", None) or offer.times[0].start
        if isinstance(item, Retraction):
            start = picks.get(item.offer) or start
        return Meaning(parse_time(start), parse_time(thread.message(offer.message).sent), window)

    def ways_written(offer_id: str, message_id: str) -> list[str]:
        """How the reader may quote the time: the offer as written, or as restated in the message
        (for taking back an acceptance, the agreed time it restates)."""
        offer = mentions[offer_id]
        retracting = any(item.message == message_id for item in thread.retractions)
        quoted = [thread.text_of(offer)]
        quoted += [
            thread.text_of(mention)
            for mention in thread.mentions
            if mention.message == message_id
            and (mention.accepts == offer_id or mention.repeats_of(offer) or (retracting and mention.role == "other"))
        ]
        return quoted

    withdrawn = {item.offer: number[item.message] for item in thread.withdrawals}
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
        if key != "accepted" and thread.provenance.get("plot") == "reschedule":
            # Moving an agreed game reads, fairly, as taking the agreed time back.
            by_message = {message: time for message, time in by_message.items() if message in wanted}
        if set(by_message) != set(wanted):
            found.append(f"reader found {key} in messages {sorted(by_message)}, plan has {sorted(wanted)}")
            continue
        meanings = {number[item.message]: meaning(item) for item in expected}
        for message, written in wanted.items():
            if not same_time(by_message[message], written, meanings[message]):
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
    prompts = catalog.prompts("commands")
    intents = "\n".join(f"- {key}: {intent.description}" for key, intent in INTENTS.items() if key != "none")
    notifying = [intent for intent in INTENTS.values() if intent.events]
    events = ", ".join(f'"{event}"' for intent in notifying for event in intent.events)
    numbered = "\n".join(f"{index}. {command.text}" for index, command in enumerate(commands, start=1))
    notify = " or ".join(intent.key for intent in notifying)
    prompt = prompts.verify.format(intents=intents, notify=notify, events=events, messages=numbered)
    reply = llm.json_from(llm.chat(prompts.verify_system, prompt, 0.0, 120 * len(commands) + 100))
    if not isinstance(reply, list) or len(reply) != len(commands):
        return [["the reader's reply has the wrong length"] for _ in commands]
    problems = []
    for command, answer in zip(commands, reply, strict=True):
        answer = answer if isinstance(answer, dict) else {}
        found = []
        if answer.get("intent") != command.intent:
            found.append(f"reader says {answer.get('intent')!r}, plan says {command.intent!r}")
        elif command.event and answer.get("event") != command.event:
            found.append(f"reader says event {answer.get('event')!r}, plan says {command.event!r}")
        problems.append(found)
    return problems
