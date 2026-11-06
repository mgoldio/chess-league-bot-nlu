"""The names the model is given, for commands and for scheduling threads."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

INTENTS_FILE = Path(__file__).with_name("intents.toml")


@dataclass(frozen=True)
class Intent:
    key: str
    label: str
    description: str
    events: dict[str, str] = field(default_factory=dict)


def load_intents(path: Path = INTENTS_FILE) -> dict[str, Intent]:
    with path.open("rb") as file:
        table = tomllib.load(file)
    return {
        key: Intent(key, values["label"], values["description"], dict(values.get("events", {})))
        for key, values in table.items()
    }


INTENTS = load_intents()
INTENT_BY_LABEL = {intent.label: intent.key for intent in INTENTS.values()}
EVENT_BY_LABEL = {label: key for intent in INTENTS.values() for key, label in intent.events.items()}
ARGUMENTS = ("player", "league", "round", "board", "team", "subscription id", "phrase", "recipient")

OFFERED = "offered time"
ACCEPTED = "accepted time"
WITHDRAWN = "withdrawn time"
SCHEDULING = (OFFERED, ACCEPTED, WITHDRAWN)


def command_schema(schema):
    """Ask for a message's command, ranked, with its event and argument spans."""
    return (
        schema.classification(
            "command",
            list(INTENT_BY_LABEL),
            multi_label=True,
            class_act="softmax",
            cls_threshold=0.0,
            top_k=len(INTENT_BY_LABEL),
        )
        .classification("event", list(EVENT_BY_LABEL))
        .entities(list(ARGUMENTS))
    )


def thread_schema(schema):
    """Ask for the offered, accepted and withdrawn times in a thread."""
    return schema.entities(list(SCHEDULING))
