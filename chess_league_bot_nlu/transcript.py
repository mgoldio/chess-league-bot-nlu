"""Scheduling threads as the model reads them, and the model's marks as rule facts.

The model reads the thread so far as plain "white: ..." / "black: ..." lines,
with no timestamps. A message keeps its own line breaks, so a list of times
or a second paragraph continues on lines with no speaker; only the start of a
message names one. (The model's word splitter reads line breaks as spaces.)
It marks three things: times a player offered, offered
times the other player accepted, and offered times taken back. Acceptances and
withdrawals are marked on the offer itself, so a reply like "deal 👍" still
points at a time. Taking back is one mark whoever does it: from the player who
offered the time it withdraws the offer, and from the player who accepted it
it retracts the acceptance, and the offer stands again. Who did what follows from which line a mark
sits in. A time offered on several days ("21:00 fri or sat") is one mark
and one offer per day; accepting it picks a day.

`Replay` feeds the thread to the model one message at a time, as the bot would,
and turns the marks into `rules` facts: an acceptance or withdrawal happens at
the message after which its mark first appears, and only if that message's
sender is the one who could have made it.
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from . import records, rules
from .labels import ACCEPTED, OFFERED, SCHEDULING, WITHDRAWN
from .records import opponent
from .resolve import (
    OPEN_START,
    Resolved,
    clocks,
    day_alternatives,
    last_date_phrase,
    resolve,
    weekday_of,
    weekdays_in,
    zone_offset,
)

MY_TIME = re.compile(r"\b(?:my (?:time|evening|morning)|local(?: time)?)\b", re.IGNORECASE)
YOUR_TIME = re.compile(r"\byour (?:time|evening|morning)\b", re.IGNORECASE)
DAY_AND_CLOCK = re.compile(r"[A-Za-z]+\.?\s+(?:at\s+|@\s*)?\d{1,2}(?:[:h.]\d{2})?\s*(?:am|pm|h)?", re.IGNORECASE)


@dataclass
class Transcript:
    text: str
    starts: dict[str, int] = field(default_factory=dict)
    messages: list[records.Message] = field(default_factory=list)

    def locate(self, start: int, end: int) -> tuple[records.Message, int, int] | None:
        """The message a transcript span falls in, and its offsets within that message."""
        for message in self.messages:
            offset = self.starts[message.id]
            if offset <= start and end <= offset + len(message.text):
                return message, start - offset, end - offset
        return None


def transcript(thread: records.Thread, upto: int | None = None) -> Transcript:
    script = Transcript("")
    for message in thread.messages[:upto]:
        line = f"{message.speaker}: "
        script.starts[message.id] = len(script.text) + len(line)
        script.messages.append(message)
        script.text += line + message.text + "\n"
    return script


def gold_spans(thread: records.Thread, script: Transcript) -> dict[str, list[dict]]:
    """The spans the model should mark in this transcript, in transcript offsets."""
    shown = set(script.starts)
    mentions = {mention.id: mention for mention in thread.mentions}
    spans: dict[str, list[dict]] = {label: [] for label in SCHEDULING}

    def add(label: str, mention: records.Mention) -> None:
        offset = script.starts[mention.message]
        span = {"text": thread.text_of(mention), "start": offset + mention.start, "end": offset + mention.end}
        if span not in spans[label]:
            spans[label].append(span)

    for mention in thread.mentions:
        if mention.message in shown and mention.role == "offered":
            add(OFFERED, mention)
        elif mention.message in shown and mention.role == "accepted":
            add(ACCEPTED, mention)
    for acceptance in thread.acceptances:
        if acceptance.message in shown:
            add(ACCEPTED, mentions[acceptance.offer])
    for withdrawal in [*thread.withdrawals, *thread.retractions]:
        if withdrawal.message in shown:
            add(WITHDRAWN, mentions[withdrawal.offer])
    return spans


def stated_zone(thread: records.Thread, message: records.Message, player: str) -> timedelta | None:
    """The zone a player last named, in this message or an earlier one."""
    for earlier in reversed(thread.messages[: thread.messages.index(message) + 1]):
        if earlier.speaker == player and (offset := zone_offset(earlier.text)) is not None:
            return offset
    return None


def day_context(thread: records.Thread, message: records.Message, start: int, end: int) -> str:
    """Where a bare clock time takes its day from: a day right after it ("20:00 thursday",
    "20:00 on thursday"), then the text before it, then earlier messages."""
    following = re.match(r"\s*(?:(?:on|this|next)\s+)?(\S+(?:\s+\d{1,2}\S*)?)", message.text[end:])
    if (
        following
        and (phrase := last_date_phrase(following.group(1)))
        and following.group(1).lower().startswith(phrase.lower())
    ):
        return phrase
    if phrase := last_date_phrase(message.text[:start]):
        return phrase
    for earlier in reversed(thread.messages[: thread.messages.index(message)]):
        if phrase := last_date_phrase(earlier.text):
            return phrase
    return ""


def resolve_span(thread: records.Thread, message: records.Message, start: int, end: int) -> list[Resolved]:
    """Resolve a marked time, using the thread for its zone and, for a bare clock time, its day."""
    text = message.text[start:end]
    window = (records.parse_time(thread.pairing_published), records.parse_time(thread.deadline))
    hint = None
    if MY_TIME.search(text):
        hint = stated_zone(thread, message, message.speaker)
    elif YOUR_TIME.search(text):
        hint = stated_zone(thread, message, opponent(message.speaker))
    found = resolve(text, records.parse_time(message.sent), window, hint, day_context(thread, message, start, end))
    if not found and weekday_of(text) is not None:
        found = restated(thread, message, text, window)
    return found


def restated(thread: records.Thread, message: records.Message, text: str, window) -> list[Resolved]:
    """A bare day ("thu?") repeating a time given earlier for that day."""
    weekday = weekday_of(text)
    for earlier in reversed(thread.messages[: thread.messages.index(message)]):
        for match in DAY_AND_CLOCK.finditer(earlier.text):
            if weekday_of(match.group(0)) != weekday:
                continue
            zone = stated_zone(thread, earlier, earlier.speaker)
            if found := resolve(match.group(0), records.parse_time(earlier.sent), window, zone):
                return found
    return []


@dataclass(frozen=True)
class Span:
    message: str
    start: int
    end: int


class Replay:
    """Rule facts built up from the model's marks, one message at a time."""

    def __init__(self, thread: records.Thread):
        self.thread = thread
        self.offers: dict[Span, list[rules.Offer]] = {}
        self.acceptances: dict[Span, rules.Accepted] = {}
        self.seen: set[tuple[str, Span]] = set()

    def new(self, label: str, found: list[tuple[records.Message, Span]]) -> list[tuple[records.Message, Span]]:
        """The marks that weren't there after the previous message: only they can be its doing."""
        fresh = [(message, span) for message, span in found if (label, span) not in self.seen]
        self.seen |= {(label, span) for _, span in found}
        return fresh

    def located(self, script: Transcript, values: list[dict]) -> list[tuple[records.Message, Span]]:
        found = []
        for value in values:
            place = script.locate(value["start"], value["end"])
            if place:
                message, start, end = place
                found.append((message, Span(message.id, start, end)))
        return found

    def step(self, upto: int, spans: dict[str, list[dict]]) -> None:
        """Take the marks the model made on the first `upto` messages."""
        script = transcript(self.thread, upto)
        latest = self.thread.messages[upto - 1]
        sent = records.parse_time(latest.sent)
        for message, span in self.located(script, spans.get(OFFERED, [])):
            if span not in self.offers:
                self.add_offer(message, span)
        for _, span in self.new(WITHDRAWN, self.located(script, spans.get(WITHDRAWN, []))):
            offer = self.on_offer(span)
            if offer and self.offers[offer][0].withdrawn is None and self.made_by(latest, offer, own=True):
                self.offers[offer] = [dataclasses.replace(each, withdrawn=sent) for each in self.offers[offer]]
            elif offer and self.retractable(offer, latest):
                self.acceptances[offer] = dataclasses.replace(self.acceptances[offer], retracted=sent)
        accepted = self.located(script, spans.get(ACCEPTED, []))
        for _, span in self.new(ACCEPTED, accepted):
            offer = self.on_offer(span)
            if offer and offer not in self.acceptances and self.made_by(latest, offer, own=False):
                time = self.pick(accepted, latest, offer)
                if choice := rules.chosen(self.offers[offer], time):
                    self.acceptances[offer] = rules.Accepted(latest.speaker, sent, choice.id, time)

    def add_offer(self, message: records.Message, span: Span) -> None:
        times = resolve_span(self.thread, message, span.start, span.end)
        if not times:
            return
        written = message.text[span.start : span.end]
        self.offers[span] = [
            rules.Offer(
                id=f"{span.message}:{span.start}" + (f"/{index}" if len(times) > 1 else ""),
                player=message.speaker,
                sent=records.parse_time(message.sent),
                start=time.start,
                end=time.end,
                open=time.end is None and bool(OPEN_START.search(written)),
            )
            for index, time in enumerate(times)
        ]

    def made_by(self, latest: records.Message, offer: Span, own: bool) -> bool:
        """Whether the latest message could withdraw (`own`) or accept this offer: a later message by the right player."""
        player = self.offers[offer][0].player
        return latest.id != offer.message and latest.speaker == (player if own else opponent(player))

    def retractable(self, offer: Span, latest: records.Message) -> bool:
        """Whether the latest message could take back an acceptance of this offer: a later one by the accepter."""
        acceptance = self.acceptances.get(offer)
        return (
            acceptance is not None
            and acceptance.retracted is None
            and acceptance.player == latest.speaker
            and acceptance.sent < records.parse_time(latest.sent)
        )

    def on_offer(self, span: Span) -> Span | None:
        for offer in self.offers:
            if offer.message == span.message and offer.start < span.end and span.start < offer.end:
                return offer
        return None

    def pick(
        self, accepted: list[tuple[records.Message, Span]], latest: records.Message, offer: Span
    ) -> datetime | None:
        """The time restated in the accepting message, for accepting a range or one of several days:
        the time it writes, or, when it names only a day ("sat works"), that day's time in the offer."""
        for message, span in accepted:
            if message.id != latest.id or self.on_offer(span) is not None:
                continue
            text = message.text[span.start : span.end]
            if clocks(text):
                times = resolve_span(self.thread, message, span.start, span.end)
                if len(times) == 1:
                    return times[0].start
            elif (weekday := weekday_of(text)) is not None:
                return self.on_day(offer, weekday)
        return None

    def on_day(self, offer: Span, weekday: int) -> datetime | None:
        """The start of an offer's time on the day it names as written."""
        written = self.thread.message(offer.message).text[offer.start : offer.end]
        days = day_alternatives(written) or weekdays_in(written)[:1]
        for (_, _, day), each in zip(days, self.offers[offer], strict=False):
            if day == weekday:
                return each.start
        return None

    def facts(self) -> tuple[list[rules.Offer], list[rules.Accepted]]:
        return [offer for offers in self.offers.values() for offer in offers], list(self.acceptances.values())


def within_window(thread: records.Thread, rule: rules.Rule) -> int:
    """How many messages fall inside the rule's window."""
    cutoff = records.parse_time(thread.pairing_published) + rule.window
    return sum(records.parse_time(message.sent) <= cutoff for message in thread.messages)
