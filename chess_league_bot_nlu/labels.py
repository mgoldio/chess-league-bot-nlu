"""The commands the model tells apart, their events, and the arguments it finds in them."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

INTENTS_FILE = Path(__file__).with_name("intents.toml")


@dataclass(frozen=True)
class Intent:
    key: str
    description: str
    events: tuple[str, ...] = ()
    unlike: str = ""


def load_intents(path: Path = INTENTS_FILE) -> dict[str, Intent]:
    with path.open("rb") as file:
        table = tomllib.load(file)
    return {
        key: Intent(key, values["description"], tuple(values.get("events", ())), values.get("unlike", ""))
        for key, values in table.items()
    }


INTENTS = load_intents()
ARGUMENTS = ("player", "league", "round", "team", "subscription id")
