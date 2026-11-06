"""What a scheduling thread comes to after each message: the offers still standing, and what was accepted.

What the model reads from a thread, and what a record says, are the same kind of thing: times
written in messages, each with its role there (`Time`), and messages that accept or take back
an offer made earlier (`Event`). Both come to an outcome the same way. A time a player offers
stands until a later message of theirs takes it back. A later message of the other player's
accepting a standing offer accepts it; taking it back again from the player who accepted it
retracts that, and from the player who offered it withdraws the offer and the acceptance with it.
Anything else a message does with a time changes nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import records


@dataclass(frozen=True)
class Time:
    """A time written in a message, as characters of its text, and its role there."""

    message: str
    start: int
    end: int
    role: str


@dataclass(frozen=True)
class Event:
    """A message accepting, or taking back, an offer made in an earlier message."""

    message: str
    offer: Time
    kind: str


@dataclass
class Outcome:
    """The offers standing, those withdrawn, and the accepted ones, each with the time its
    acceptance restated, if it restated one ("sat 18:00 works")."""

    standing: set[Time] = field(default_factory=set)
    withdrawn: set[Time] = field(default_factory=set)
    accepted: dict[Time, Time | None] = field(default_factory=dict)


def gold(thread: records.Thread) -> tuple[list[Time], list[Event]]:
    times = {mention.id: Time(mention.message, mention.start, mention.end, mention.role) for mention in thread.mentions}
    events = [Event(acceptance.message, times[acceptance.offer], "accept") for acceptance in thread.acceptances]
    events += [
        Event(taken.message, times[taken.offer], "take back") for taken in [*thread.withdrawals, *thread.retractions]
    ]
    return list(times.values()), events


def outcome(messages: list[records.Message], times: list[Time], events: list[Event], upto: int) -> Outcome:
    """What the first `upto` messages come to."""
    speakers = {message.id: message.speaker for message in messages}
    result, accepters = Outcome(), {}
    for message in messages[:upto]:
        for time in times:
            if time.message == message.id and time.role == "offered":
                result.standing.add(time)
        for event in events:
            if event.message != message.id or event.offer not in result.standing:
                continue
            offerer = speakers[event.offer.message]
            if event.kind == "accept" and message.speaker != offerer and event.offer not in result.accepted:
                restated = [time for time in times if time.message == message.id and time.role == "accepted"]
                result.accepted[event.offer] = restated[0] if restated else None
                accepters[event.offer] = message.speaker
            elif event.kind == "take back" and message.speaker == offerer:
                result.standing.discard(event.offer)
                result.withdrawn.add(event.offer)
                result.accepted.pop(event.offer, None)
            elif event.kind == "take back" and accepters.get(event.offer) == message.speaker:
                result.accepted.pop(event.offer, None)
                accepters.pop(event.offer)
    return result
