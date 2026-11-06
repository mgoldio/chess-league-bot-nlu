"""Plan messages to the bot in code: an intent with its arguments, event and style. The model
only writes the words; the labels come from here."""

from __future__ import annotations

import random
from dataclasses import dataclass

from ..labels import INTENTS, Intent
from . import catalog
from .names import username


@dataclass
class CommandPlan:
    intent: Intent
    arguments: dict[str, str]
    event: str | None
    style: catalog.Style
    words: int
    none_kind: str | None = None
    angle: str = ""


def value(rng: random.Random, vocabulary: catalog.Vocabulary, values: str) -> str:
    if values == "players":
        return username(rng, vocabulary)
    return rng.choice(getattr(vocabulary, values)).format(n=rng.randint(1, 12))


def command(rng: random.Random) -> CommandPlan:
    settings, vocabulary = catalog.commands(), catalog.vocabulary()
    if rng.random() < settings.none_share:
        intent = INTENTS["none"]
    else:
        intent = rng.choice([intent for key, intent in INTENTS.items() if key != "none"])
    arguments = {
        argument.label: value(rng, vocabulary, argument.values)
        for argument in settings.arguments.get(intent.key, ())
        if rng.random() < argument.chance
    }
    moderator = intent.key in settings.moderator_intents
    styles = [style for style in settings.styles if not (moderator and style.name in settings.not_for_moderators)]
    style = rng.choices(styles, [style.weight for style in styles])[0]
    event = rng.choice(list(intent.events)) if intent.events else None
    none_kind = rng.choice(settings.none_kinds) if intent.key == "none" else None
    angles = settings.angles.get(intent.key, ())
    angle = rng.choice(angles) if angles and rng.random() < settings.angle_chance else ""
    return CommandPlan(intent, arguments, event, style, rng.choice(style.words), none_kind, angle)
