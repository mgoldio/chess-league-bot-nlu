"""Load the generator's content files (TOML and Markdown) into typed values.

Loading is strict: unknown keys, unknown beat kinds and references between
files that don't line up fail here, naming the file, rather than as odd data
later. `check()` loads everything at once.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields
from datetime import datetime
from functools import cache
from pathlib import Path

from ..labels import ARGUMENTS, INTENTS
from ..resolve import ZONES

CONTENT = Path(__file__).with_name("content")
BEAT_KINDS = (
    "offer",
    "accept",
    "withdraw",
    "correct",
    "retract",
    "mistaken_accept",
    "decline",
    "decoy",
    "nudge",
    "remark",
    "chat",
)
ACCEPT_STYLES = ("bare", "restate", "partial")


class InvalidContent(ValueError):
    pass


def toml(name: str) -> dict:
    with (CONTENT / name).open("rb") as file:
        return tomllib.load(file)


def build(cls, data: dict, where: str):
    """A dataclass from a TOML table, failing on unknown or missing keys."""
    known = {field.name for field in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise InvalidContent(f"{where}: unknown keys {sorted(unknown)}")
    try:
        return cls(**data)
    except (TypeError, InvalidContent) as error:
        raise InvalidContent(f"{where}: {error}") from None


@dataclass(frozen=True)
class Persona:
    short: str
    description: str
    words: tuple[int, int]


@dataclass(frozen=True)
class Beat:
    by: str
    do: str
    times: str | int = ""
    range_chance: float | None = None
    multi_day_chance: float | None = None
    struck_chance: float | None = None
    local_chance: float | None = None
    near: bool = False
    same_day: bool = False
    rejects: bool = False
    restates: bool = False
    layout: str = ""
    which: str = "last"
    style: str = ""
    gap: str = "normal"
    chance: float = 1.0
    note: str | tuple[str, ...] = ()
    before_agreed: bool = False

    def __post_init__(self):
        object.__setattr__(self, "note", (self.note,) if isinstance(self.note, str) else tuple(self.note))
        allowed = {
            "by": ("first", "second", "either", "other"),
            "do": BEAT_KINDS,
            "which": ("last", "any"),
            "gap": ("fast", "normal", "slow"),
            "layout": ("", "list", "inline"),
            "style": ("", *ACCEPT_STYLES),
        }
        for key, values in allowed.items():
            if getattr(self, key) not in values:
                raise InvalidContent(f"beat `{key}` must be one of {values}, not {getattr(self, key)!r}")
        if self.do == "offer" and not str(self.times):
            raise InvalidContent("an offer beat needs `times`")

    def count_range(self) -> tuple[int, int]:
        low, _, high = str(self.times).partition("-")
        return int(low), int(high or low)


@dataclass(frozen=True)
class Plot:
    name: str
    weight: float
    beats: tuple[Beat, ...]


@dataclass(frozen=True)
class Plots:
    plots: tuple[Plot, ...]
    opener_chance: float
    closer_chance: float
    closer_reply_chance: float
    range_chance: float
    open_chance: float
    zone_chance: float
    multi_day_chance: float
    struck_chance: float
    local_chance: float
    offer_note_chance: float
    accept_note_chance: float
    openers: tuple[str, ...]
    closers: tuple[str, ...]
    offer_notes: tuple[str, ...]
    accept_notes: tuple[str, ...]
    accept_styles: dict[str, float]


@dataclass(frozen=True)
class Layout:
    list_chance: float
    bullets: dict[str, float]
    blank_line_after_list: float
    paragraph_chance: float
    paragraph_min_words: int
    mention_chance: float


@dataclass(frozen=True)
class League:
    name: str
    weight: float
    description: str
    first_round: int
    last_round: int
    first_published: datetime
    round_hours: float
    publish_delay_minutes: tuple[int, ...]


@dataclass(frozen=True)
class Schedule:
    leagues: tuple[League, ...]
    gaps: dict[str, tuple[float, float]]
    hour_weights: tuple[float, ...]
    minutes: dict[int, float]
    earliest_after_pairing_hours: float
    lead_hours: float
    before_deadline_hours: float
    spacing_hours: float
    range_hours: tuple[float, ...]
    open_pick_hours: float
    accept_before_hours: float
    zones: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class Form:
    form: str
    weight: float = 1.0
    when: str | None = None
    lower: float = 0.0


@dataclass(frozen=True)
class TimeFormats:
    days: tuple[Form, ...]
    day_only: tuple[Form, ...]
    several_days: tuple[Form, ...]
    clocks: tuple[Form, ...]
    points: tuple[Form, ...]
    ranges: tuple[Form, ...]
    open: tuple[Form, ...]
    utc_suffixes: tuple[str, ...]
    utc_suffix_chance: float
    weekday_typo_chance: float
    clock_only_chance: float


@dataclass(frozen=True)
class Argument:
    label: str
    chance: float
    values: str


@dataclass(frozen=True)
class Style:
    name: str
    weight: float
    words: tuple[int, ...]
    description: str


@dataclass(frozen=True)
class Commands:
    none_share: float
    none_kinds: tuple[str, ...]
    moderator_intents: frozenset[str]
    not_for_moderators: frozenset[str]
    event_descriptions: dict[str, str]
    line: str
    none_what: str
    event: str
    include: str
    no_placeholders: str
    angle: str
    angle_chance: float
    styles: tuple[Style, ...]
    arguments: dict[str, tuple[Argument, ...]]
    angles: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class Vocabulary:
    username_words: tuple[str, ...]
    username_forms: tuple[str, ...]
    mention_chance: float
    teams: tuple[str, ...]
    leagues: tuple[str, ...]
    rounds: tuple[str, ...]
    boards: tuple[str, ...]
    subscription_ids: tuple[str, ...]
    phrases: tuple[str, ...]


@dataclass(frozen=True)
class Noise:
    edits: tuple[int, int]
    lowercase_chance: float
    min_word_length: int
    kinds: tuple[str, ...]


@dataclass(frozen=True)
class Framing:
    name: str
    system: str
    noise_chance: float
    temperature: tuple[float, float]


@dataclass(frozen=True)
class Prompts:
    framings: dict[str, Framing]
    task: str
    verify: str
    verify_system: str
    qa: str | None


@cache
def personas() -> tuple[Persona, ...]:
    return tuple(
        build(Persona, {**entry, "words": tuple(entry["words"])}, "personas.toml")
        for entry in toml("personas.toml")["persona"]
    )


@cache
def plots() -> Plots:
    data = dict(toml("plots.toml"))
    entries = data.pop("plots")
    styles = data.pop("accept_styles")
    built = tuple(
        Plot(name, entry["weight"], tuple(build(Beat, beat, f"plots.toml [{name}]") for beat in entry["beats"]))
        for name, entry in entries.items()
    )
    lists = ("openers", "closers", "offer_notes", "accept_notes")
    plots = build(
        Plots,
        {**data, "plots": built, "accept_styles": dict(styles), **{name: tuple(data[name]) for name in lists}},
        "plots.toml",
    )
    if set(styles) - set(ACCEPT_STYLES):
        raise InvalidContent(f"plots.toml: accept styles must be one of {ACCEPT_STYLES}, not {sorted(styles)}")
    return plots


@cache
def layout() -> Layout:
    data = toml("layout.toml")
    return build(Layout, {**data, "bullets": dict(data["bullets"])}, "layout.toml")


@cache
def schedule() -> Schedule:
    data = dict(toml("schedule.toml"))
    leagues = tuple(
        build(
            League,
            {
                **entry,
                "name": name,
                "first_published": datetime.fromisoformat(entry["first_published"].replace("Z", "+00:00")),
                "publish_delay_minutes": tuple(entry["publish_delay_minutes"]),
            },
            f"schedule.toml [leagues.{name}]",
        )
        for name, entry in data.pop("leagues").items()
    )
    gaps = {name: (pace["mu"], pace["sigma"]) for name, pace in data.pop("gaps").items()}
    times = dict(data.pop("game_times"))
    unknown_zones = [zone for zone in data["zones"] if zone.lower() not in ZONES]
    if unknown_zones:
        raise InvalidContent(f"schedule.toml: the resolver doesn't know zones {unknown_zones}")
    zones = tuple((zone, ZONES[zone.lower()]) for zone in data.pop("zones"))
    if data:
        raise InvalidContent(f"schedule.toml: unknown keys {sorted(data)}")
    return build(
        Schedule,
        {
            **times,
            "leagues": leagues,
            "gaps": gaps,
            "hour_weights": tuple(times["hour_weights"]),
            "minutes": {int(minute): weight for minute, weight in times["minutes"].items()},
            "range_hours": tuple(times["range_hours"]),
            "zones": zones,
        },
        "schedule.toml",
    )


@cache
def time_formats() -> TimeFormats:
    data = dict(toml("time_formats.toml"))
    lists = ("days", "day_only", "several_days", "clocks", "points", "ranges", "open")
    built = {name: tuple(build(Form, form, f"time_formats.toml {name}") for form in data.pop(name)) for name in lists}
    return build(TimeFormats, {**data, **built, "utc_suffixes": tuple(data["utc_suffixes"])}, "time_formats.toml")


@cache
def beats() -> dict:
    data = toml("beats.toml")
    needed = {
        "line",
        "offer",
        "accept_restate",
        "accept_partial",
        "accept_pick",
        "accept_bare",
        "accept_bare_reference",
        "opener",
        "closer",
        "reference_ordinal",
        "reference_last",
        "reference_day",
        "ordinals",
        "offer_list",
        "offer_same_day",
        "rejecting",
        "retract_restate",
        "local",
        "mention",
        "paragraphs",
    }
    needed |= set(BEAT_KINDS) - {"offer", "accept"}
    missing = needed - set(data)
    if missing:
        raise InvalidContent(f"beats.toml: missing {sorted(missing)}")
    return data


@cache
def vocabulary() -> Vocabulary:
    data = toml("vocabulary.toml")
    return build(
        Vocabulary,
        {key: tuple(value) if isinstance(value, list) else value for key, value in data.items()},
        "vocabulary.toml",
    )


@cache
def commands() -> Commands:
    data = dict(toml("commands.toml"))
    styles = tuple(
        build(Style, {**entry, "name": name, "words": tuple(entry["words"])}, f"commands.toml [styles.{name}]")
        for name, entry in data.pop("styles").items()
    )
    arguments = {
        intent: tuple(build(Argument, entry, f"commands.toml [arguments.{intent}]") for entry in entries)
        for intent, entries in data.pop("arguments").items()
    }
    angles = {intent: tuple(entries) for intent, entries in data.pop("angles").items()}
    built = build(
        Commands,
        {
            **data,
            "none_kinds": tuple(data["none_kinds"]),
            "moderator_intents": frozenset(data["moderator_intents"]),
            "not_for_moderators": frozenset(data["not_for_moderators"]),
            "styles": styles,
            "arguments": arguments,
            "angles": angles,
        },
        "commands.toml",
    )
    unknown_intents = (set(arguments) | set(angles) | built.moderator_intents) - set(INTENTS)
    if unknown_intents:
        raise InvalidContent(f"commands.toml: unknown intents {sorted(unknown_intents)}")
    if built.not_for_moderators - {style.name for style in styles}:
        raise InvalidContent("commands.toml: not_for_moderators names an unknown style")
    lists = {"players"} | {field.name for field in fields(Vocabulary)}
    for intent, entries in arguments.items():
        for argument in entries:
            if argument.values not in lists:
                raise InvalidContent(f"commands.toml [arguments.{intent}]: unknown values {argument.values!r}")
            if argument.label not in ARGUMENTS:
                raise InvalidContent(
                    f"commands.toml [arguments.{intent}]: {argument.label!r} is not one of the model's {ARGUMENTS}"
                )
    for intent in INTENTS.values():
        if intent.events and set(intent.events) != set(built.event_descriptions):
            raise InvalidContent(f"commands.toml: event_descriptions must cover the events of {intent.key}")
    return built


@cache
def noise() -> Noise:
    data = toml("noise.toml")
    return build(Noise, {**data, "edits": tuple(data["edits"]), "kinds": tuple(data["kinds"])}, "noise.toml")


@cache
def prompts(kind: str) -> Prompts:
    directory = CONTENT / "prompts" / kind
    settings = toml(f"prompts/{kind}/framings.toml")
    files = {path.stem: path.read_text().strip() for path in sorted((directory / "framings").glob("*.md"))}
    if set(files) != set(settings):
        raise InvalidContent(
            f"prompts/{kind}: framings/*.md {sorted(files)} and framings.toml {sorted(settings)} differ"
        )
    framings = {
        name: build(
            Framing,
            {"name": name, "system": files[name], **values, "temperature": tuple(values["temperature"])},
            f"prompts/{kind}/framings.toml",
        )
        for name, values in settings.items()
    }
    return Prompts(
        framings,
        (directory / "task.md").read_text(),
        (directory / "verify.md").read_text(),
        (directory / "verify-system.md").read_text().strip(),
        (directory / "qa.md").read_text() if (directory / "qa.md").exists() else None,
    )


def check() -> None:
    """Load every content file, so any problem shows up at once."""
    for load in (personas, plots, layout, schedule, time_formats, beats, vocabulary, commands, noise):
        load()
    for kind in ("threads", "commands"):
        prompts(kind)
