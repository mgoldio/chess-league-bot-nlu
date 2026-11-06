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
    none_kind: catalog.NoneKind | None = None
    like: str = ""
    angle: str = ""
    contrast: bool = False


NUMBER_WORDS = ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve")


def value(rng: random.Random, vocabulary: catalog.Vocabulary, values: str) -> str:
    if values == "players":
        return username(rng, vocabulary)
    if values == "teams":
        first, second = rng.sample(vocabulary.team_words, 2)
        return rng.choice(vocabulary.team_forms).format(a=first, b=second, n=rng.randint(1, 99))
    number = rng.randint(1, 11)
    return rng.choice(getattr(vocabulary, values)).format(
        n=number, m=number + 1, word=NUMBER_WORDS[number - 1], ordinal=ordinal(number)
    )


def ordinal(number: int) -> str:
    suffix = "th" if 10 <= number % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix}"


def batch(
    rng: random.Random, size: int, weights: dict[str, float] | None = None, others: float = 1.0
) -> list[CommandPlan]:
    """Plans for one writer call: by chance, one per command of a group easily mistaken for one
    another (see `contrast_groups`), the rest picked as usual, in a shuffled order. `weights` sets
    how often the named commands are picked, and `others` scales the catalog's weights of the rest."""
    settings = catalog.commands()
    plans: list[CommandPlan] = []
    if rng.random() < settings.contrast_chance:
        group = rng.choice(settings.contrast_groups)
        plans = [command(rng, INTENTS[key]) for key in rng.sample(group, min(len(group), size - 1))]
        near_misses = [kind for kind in settings.none_kinds if kind.near_miss]
        plans.append(command(rng, INTENTS["none"], rng.choice(near_misses)))
        for plan in plans:
            plan.contrast = True
    plans += [command(rng, weights=weights, others=others) for _ in range(size - len(plans))]
    rng.shuffle(plans)
    return plans


def command(
    rng: random.Random,
    intent: Intent | None = None,
    none_kind: catalog.NoneKind | None = None,
    weights: dict[str, float] | None = None,
    others: float = 1.0,
) -> CommandPlan:
    """One message's plan. A `none` message names a player, league or round only for kinds that
    allow it, and then never in a style terse enough to read as a command."""
    settings, vocabulary = catalog.commands(), catalog.vocabulary()
    if intent is None and rng.random() < settings.none_share:
        intent = INTENTS["none"]
    elif intent is None:
        commands = [intent for key, intent in INTENTS.items() if key != "none"]
        weights = weights or {}
        chances = [weights.get(each.key, settings.intent_weights.get(each.key, 1.0) * others) for each in commands]
        intent = rng.choices(commands, chances)[0]
    if intent.key == "none":
        none_kind = none_kind or rng.choice(settings.none_kinds)
    arguments = {
        argument.label: value(rng, vocabulary, argument.values)
        for argument in settings.arguments.get(intent.key, ())
        if (none_kind is None or none_kind.arguments) and rng.random() < argument.chance
    }
    styles = [
        style
        for style in settings.styles
        if not (none_kind and arguments and style.name in settings.command_like_styles)
    ]
    style = rng.choices(styles, [style.weight for style in styles])[0]
    event = rng.choice(list(intent.events)) if intent.events and rng.random() >= settings.no_event_chance else None
    like = rng.choice(none_kind.like) if none_kind else ""
    angles = settings.angles.get(intent.key, ())
    angle = rng.choice(angles) if angles and rng.random() < settings.angle_chance else ""
    return CommandPlan(intent, arguments, event, style, rng.choice(style.words), none_kind, like, angle)
