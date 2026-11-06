"""The dataset's record formats, validated on load.

Records hold facts only: what was said, which spans mention times or
arguments, and who accepted or withdrew what. Rule outcomes such as effort and
agreements are never stored; `rules` computes them from `thread_facts`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

from . import rules
from .labels import INTENTS

SCHEMA = 2
PLAYERS = ("white", "black")
ROLES = ("offered", "accepted", "other")


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


@dataclass
class Message:
    id: str
    speaker: str
    sent: str
    text: str


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
    id: str
    league: str
    pairing_published: str
    deadline: str
    messages: list[Message]
    mentions: list[Mention]
    acceptances: list[Acceptance] = field(default_factory=list)
    withdrawals: list[Withdrawal] = field(default_factory=list)
    retractions: list[Retraction] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)
    schema: int = SCHEMA

    @classmethod
    def from_dict(cls, data: dict) -> Thread:
        data = upgrade(data)
        thread = cls(
            id=data["id"],
            league=data["league"],
            pairing_published=data["pairing_published"],
            deadline=data["deadline"],
            messages=[Message(**message) for message in data["messages"]],
            mentions=[
                Mention(**{**mention, "times": [Utc(**time) for time in mention["times"]]})
                for mention in data["mentions"]
            ],
            acceptances=[Acceptance(**acceptance) for acceptance in data.get("acceptances", [])],
            withdrawals=[Withdrawal(**withdrawal) for withdrawal in data.get("withdrawals", [])],
            retractions=[Retraction(**retraction) for retraction in data.get("retractions", [])],
            provenance=data.get("provenance", {}),
            schema=data.get("schema", SCHEMA),
        )
        thread.validate()
        return thread

    def to_dict(self) -> dict:
        return asdict(self)

    def message(self, message_id: str) -> Message:
        return next(message for message in self.messages if message.id == message_id)

    def text_of(self, mention: Mention) -> str:
        return self.message(mention.message).text[mention.start : mention.end]

    def validate(self) -> None:
        if self.schema != SCHEMA:
            raise InvalidRecord(f"{self.id}: schema {self.schema}, expected {SCHEMA}")
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
        thread_facts(self)


def upgrade(data: dict) -> dict:
    """A record in the current schema. Schema 1 gave each mention a single `utc`."""
    if data.get("schema") != 1:
        return data
    mentions = [
        {**{k: v for k, v in mention.items() if k != "utc"}, "times": [mention["utc"]]}
        for mention in data.get("mentions", [])
    ]
    return {**data, "mentions": mentions, "schema": SCHEMA}


def offer_id(mention: Mention, index: int) -> str:
    """The id of the offer one of a mention's times makes: the mention's own, or with the day's number."""
    return mention.id if len(mention.times) == 1 else f"{mention.id}/{index}"


def thread_facts(thread: Thread) -> tuple[list[rules.Offer], list[rules.Accepted]]:
    """The thread's offers and acceptances, as the rules module sees them: a time offered
    on several days is one offer per day, and an acceptance takes the day it picks."""
    messages = {message.id: message for message in thread.messages}
    withdrawn = {withdrawal.offer: parse_time(messages[withdrawal.message].sent) for withdrawal in thread.withdrawals}
    by_mention = {
        mention.id: [
            rules.Offer(
                id=offer_id(mention, index),
                player=messages[mention.message].speaker,
                sent=parse_time(messages[mention.message].sent),
                start=parse_time(time.start),
                end=parse_time(time.end) if time.end else None,
                open=time.open,
                withdrawn=withdrawn.get(mention.id),
            )
            for index, time in enumerate(mention.times)
        ]
        for mention in thread.mentions
        if mention.role == "offered"
    }
    retracted = {
        (retraction.offer, messages[retraction.message].speaker): parse_time(messages[retraction.message].sent)
        for retraction in thread.retractions
    }
    acceptances = []
    for acceptance in thread.acceptances:
        time = parse_time(acceptance.time) if acceptance.time else None
        offer = rules.chosen(by_mention[acceptance.offer], time)
        if offer is None:
            raise InvalidRecord(f"{thread.id}: {acceptance.message} picks none of the days {acceptance.offer} offers")
        player = messages[acceptance.message].speaker
        acceptances.append(
            rules.Accepted(
                player=player,
                sent=parse_time(messages[acceptance.message].sent),
                offer=offer.id,
                time=time,
                retracted=retracted.get((acceptance.offer, player)),
            )
        )
    return [offer for offers in by_mention.values() for offer in offers], acceptances


@dataclass
class Argument:
    label: str
    start: int
    end: int


@dataclass
class Command:
    id: str
    text: str
    intent: str
    arguments: list[Argument] = field(default_factory=list)
    event: str | None = None
    provenance: dict = field(default_factory=dict)
    schema: int = SCHEMA

    @classmethod
    def from_dict(cls, data: dict) -> Command:
        data = upgrade(data)
        command = cls(
            id=data["id"],
            text=data["text"],
            intent=data["intent"],
            arguments=[Argument(**argument) for argument in data.get("arguments", [])],
            event=data.get("event"),
            provenance=data.get("provenance", {}),
            schema=data.get("schema", SCHEMA),
        )
        command.validate()
        return command

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        if self.schema != SCHEMA:
            raise InvalidRecord(f"{self.id}: schema {self.schema}, expected {SCHEMA}")
        intent = INTENTS.get(self.intent)
        if intent is None:
            raise InvalidRecord(f"{self.id}: unknown intent {self.intent!r}")
        for argument in self.arguments:
            if not 0 <= argument.start < argument.end <= len(self.text):
                raise InvalidRecord(f"{self.id}: argument {argument} outside the text")
        if bool(intent.events) != (self.event is not None) or (self.event and self.event not in intent.events):
            raise InvalidRecord(f"{self.id}: event {self.event!r} for intent {self.intent!r}")
