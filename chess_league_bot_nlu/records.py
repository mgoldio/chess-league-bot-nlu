"""The dataset's record formats, validated on load.

Records hold what was said and what it does: which spans mention times or
arguments, and who accepted or withdrew what. A thread's times are stored as
days after the Monday of its pairing's week ("+3T19:00:00"), and the dates its
messages write are placeholders for those days (`dates`), so a thread is on no
particular week; loaded, it is put on one. Weekday names are written out, so it
only moves by whole weeks.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from datetime import time as clock_time

from . import dates
from .labels import INTENTS

PLAYERS = ("white", "black")
ROLES = ("offered", "accepted", "other")
# The Mondays a loaded thread is put on, when no week is asked for: the 52 weeks from this one.
FIRST_WEEK = date(2026, 1, 5)


def opponent(player: str) -> str:
    return "black" if player == "white" else "white"


class InvalidRecord(ValueError):
    pass


def parse_time(value: str) -> datetime:
    moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if moment.tzinfo is None:
        raise InvalidRecord(f"time without a zone: {value}")
    return moment.astimezone(UTC)


def format_time(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def monday_of(day: date) -> date:
    return day - timedelta(days=day.weekday())


def after(value: str, week: date) -> str:
    """A time as days after the week's Monday and the time of day, in UTC ("+3T19:00:00")."""
    moment = parse_time(value)
    return f"{(moment.date() - week).days:+d}T{moment:%H:%M:%S}"


def on(value: str, week: date) -> str:
    days, clock = value.split("T")
    return format_time(datetime.combine(week + timedelta(days=int(days)), clock_time.fromisoformat(clock), UTC))


def default_week(thread_id: str) -> date:
    """The week a thread is put on when no week is asked for, the same for a thread every time."""
    return FIRST_WEEK + timedelta(weeks=int(hashlib.sha256(thread_id.encode()).hexdigest(), 16) % 52)


@dataclass
class Message:
    """A message; `template` is its text with each date it writes a placeholder for the day (`dates`)."""

    id: str
    speaker: str
    sent: str
    text: str
    template: str | None = None

    def __post_init__(self):
        if self.template is None:
            self.template = self.text


@dataclass
class Utc:
    """When a mentioned time is, in UTC: a start, and an end for a range. A time offered
    on several days ("21:00 fri or sat") has one per day."""

    start: str
    end: str | None = None
    open: bool = False


@dataclass
class Mention:
    """A time mentioned in a message: where it is in the text, what role it plays, and when it is."""

    id: str
    message: str
    start: int
    end: int
    role: str
    times: list[Utc]
    accepts: str | None = None

    def repeats_of(self, other: Mention) -> bool:
        """Whether this mention names the same time as another, or one of its days."""
        return self.times == other.times or (len(self.times) == 1 and self.times[0] in other.times)


@dataclass
class Acceptance:
    """An acceptance of an offer; `time` is the pick, when the offer is a range or names several days."""

    message: str
    offer: str
    time: str | None = None


@dataclass
class Withdrawal:
    message: str
    offer: str


@dataclass
class Retraction:
    """A player takes back their own acceptance of an offer, which then stands again."""

    message: str
    offer: str


@dataclass
class Thread:
    """A scheduling thread on the week whose Monday is `week`; `plot` is the situation it was written for."""

    id: str
    messages: list[Message]
    mentions: list[Mention]
    acceptances: list[Acceptance] = field(default_factory=list)
    withdrawals: list[Withdrawal] = field(default_factory=list)
    retractions: list[Retraction] = field(default_factory=list)
    plot: str | None = None
    week: date = field(default=FIRST_WEEK, compare=False)

    @classmethod
    def from_dict(cls, data: dict, week: date | None = None) -> Thread:
        """The stored thread, put on the week of the Monday `week`, or on its own week when none is asked for."""
        week = week or default_week(data["id"])
        if week.weekday() != 0:
            raise ValueError(f"a week starts on a Monday, not {week:%A}")
        pieces = {message["id"]: dates.rendered(message["text"], week) for message in data["messages"]}

        def placed(mention: dict) -> Mention:
            message = pieces.get(mention["message"])
            if message is None:
                raise InvalidRecord(f"{data['id']}/{mention['id']}: unknown message {mention['message']}")
            try:
                start, end = message.in_text(mention["start"]), message.in_text(mention["end"])
            except ValueError as error:
                raise InvalidRecord(f"{data['id']}/{mention['id']}: {error}") from error
            times = [
                Utc(on(each["start"], week), each.get("end") and on(each["end"], week), each.get("open", False))
                for each in mention["times"]
            ]
            return Mention(
                mention["id"], mention["message"], start, end, mention["role"], times, mention.get("accepts")
            )

        thread = cls(
            id=data["id"],
            messages=[
                Message(
                    message["id"],
                    message["speaker"],
                    on(message["sent"], week),
                    pieces[message["id"]].text,
                    message["text"],
                )
                for message in data["messages"]
            ],
            mentions=[placed(mention) for mention in data["mentions"]],
            acceptances=[
                Acceptance(each["message"], each["offer"], each.get("time") and on(each["time"], week))
                for each in data.get("acceptances", [])
            ],
            withdrawals=[Withdrawal(**each) for each in data.get("withdrawals", [])],
            retractions=[Retraction(**each) for each in data.get("retractions", [])],
            plot=data.get("plot"),
            week=week,
        )
        thread.validate()
        return thread

    def to_dict(self) -> dict:
        """The stored thread: times as days after its week's Monday, and each message's template."""
        pieces = {message.id: dates.rendered(message.template, self.week) for message in self.messages}

        def moved(value: str | None) -> str | None:
            return after(value, self.week) if value else None

        stored = {
            "id": self.id,
            "messages": [
                {"id": message.id, "speaker": message.speaker, "sent": moved(message.sent), "text": message.template}
                for message in self.messages
            ],
            "mentions": [
                {
                    **asdict(mention),
                    "start": pieces[mention.message].in_template(mention.start),
                    "end": pieces[mention.message].in_template(mention.end),
                    "times": [
                        {"start": moved(each.start), "end": moved(each.end), "open": each.open}
                        for each in mention.times
                    ],
                }
                for mention in self.mentions
            ],
            "acceptances": [{**asdict(each), "time": moved(each.time)} for each in self.acceptances],
            "withdrawals": [asdict(each) for each in self.withdrawals],
            "retractions": [asdict(each) for each in self.retractions],
        }
        return {**stored, "plot": self.plot} if self.plot else stored

    def on(self, week: date) -> Thread:
        """The same thread on the week of the Monday `week`."""
        return Thread.from_dict(self.to_dict(), week)

    def moved(self, weeks: int) -> Thread:
        return self.on(self.week + timedelta(weeks=weeks))

    def message(self, message_id: str) -> Message:
        return next(message for message in self.messages if message.id == message_id)

    def text_of(self, mention: Mention) -> str:
        return self.message(mention.message).text[mention.start : mention.end]

    def validate(self) -> None:
        messages = {message.id: message for message in self.messages}
        if len(messages) != len(self.messages):
            raise InvalidRecord(f"{self.id}: duplicate message ids")
        sent = [parse_time(message.sent) for message in self.messages]
        if sent != sorted(sent):
            raise InvalidRecord(f"{self.id}: messages out of order")
        for message in self.messages:
            if message.speaker not in PLAYERS:
                raise InvalidRecord(f"{self.id}/{message.id}: speaker {message.speaker!r}")
        mentions = {mention.id: mention for mention in self.mentions}
        if len(mentions) != len(self.mentions):
            raise InvalidRecord(f"{self.id}: duplicate mention ids")
        for mention in self.mentions:
            message = messages.get(mention.message)
            if message is None:
                raise InvalidRecord(f"{self.id}/{mention.id}: unknown message {mention.message}")
            if not 0 <= mention.start < mention.end <= len(message.text):
                raise InvalidRecord(
                    f"{self.id}/{mention.id}: offsets {mention.start}-{mention.end} outside the message"
                )
            if mention.role not in ROLES:
                raise InvalidRecord(f"{self.id}/{mention.id}: role {mention.role!r}")
            if (mention.role == "accepted") != (mention.accepts is not None):
                raise InvalidRecord(
                    f"{self.id}/{mention.id}: accepted mentions, and only they, name the offer they accept"
                )
            if not mention.times:
                raise InvalidRecord(f"{self.id}/{mention.id}: no time")
            for time in mention.times:
                if time.end is not None and parse_time(time.end) <= parse_time(time.start):
                    raise InvalidRecord(f"{self.id}/{mention.id}: range ends before it starts")
        references = [(item.offer, item.message) for item in [*self.acceptances, *self.withdrawals, *self.retractions]]
        references += [(mention.accepts, mention.message) for mention in self.mentions if mention.accepts]
        for offer_id, message_id in references:
            offer = mentions.get(offer_id)
            if offer is None or offer.role != "offered":
                raise InvalidRecord(f"{self.id}: {message_id} refers to {offer_id}, which is not an offered time")
            if message_id not in messages:
                raise InvalidRecord(f"{self.id}: reference from unknown message {message_id}")
        for acceptance in self.acceptances:
            if acceptance.time is None and len(mentions[acceptance.offer].times) > 1:
                raise InvalidRecord(f"{self.id}: {acceptance.message} accepts {acceptance.offer} without picking a day")
            if acceptance.time and not any(
                picks(each, parse_time(acceptance.time)) for each in mentions[acceptance.offer].times
            ):
                raise InvalidRecord(
                    f"{self.id}: {acceptance.message} picks none of the times {acceptance.offer} offers"
                )
        order = {message.id: index for index, message in enumerate(self.messages)}
        for retraction in self.retractions:
            speaker = messages[retraction.message].speaker
            if not any(
                acceptance.offer == retraction.offer
                and messages[acceptance.message].speaker == speaker
                and order[acceptance.message] < order[retraction.message]
                for acceptance in self.acceptances
            ):
                raise InvalidRecord(f"{self.id}: {retraction.message} takes back an acceptance its sender never made")


def picks(offered: Utc, time: datetime) -> bool:
    """Whether a time is one an offer gives: its start, a time in its range, or any time after an open start."""
    start = parse_time(offered.start)
    if offered.end:
        return start <= time <= parse_time(offered.end)
    return time == start or (offered.open and time >= start)


@dataclass
class Argument:
    label: str
    start: int
    end: int


@dataclass
class Command:
    """A message to the bot; `style` is how it was asked to be written, and `none_kind`, for a
    message that is no command, what kind of message it is."""

    id: str
    text: str
    intent: str
    arguments: list[Argument] = field(default_factory=list)
    event: str | None = None
    style: str | None = None
    none_kind: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> Command:
        command = cls(
            id=data["id"],
            text=data["text"],
            intent=data["intent"],
            arguments=[Argument(**argument) for argument in data.get("arguments", [])],
            event=data.get("event"),
            style=data.get("style"),
            none_kind=data.get("none_kind"),
        )
        command.validate()
        return command

    def to_dict(self) -> dict:
        return {key: value for key, value in asdict(self).items() if value is not None}

    def validate(self) -> None:
        intent = INTENTS.get(self.intent)
        if intent is None:
            raise InvalidRecord(f"{self.id}: unknown intent {self.intent!r}")
        for argument in self.arguments:
            if not 0 <= argument.start < argument.end <= len(self.text):
                raise InvalidRecord(f"{self.id}: argument {argument} outside the text")
        if self.event is not None and self.event not in intent.events:
            raise InvalidRecord(f"{self.id}: event {self.event!r} for intent {self.intent!r}")
