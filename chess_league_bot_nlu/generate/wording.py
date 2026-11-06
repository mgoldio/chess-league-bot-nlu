"""Write planned times the way players do, from the forms in content/time_formats.toml.

Every written time is read back by the resolver and must mean exactly the
planned time, including after any weekday typo is added; forms that could be
read another way, or that a person couldn't easily read, are never used.
"""

from __future__ import annotations

import random
import re
from datetime import datetime, timedelta

from ..dates import concrete, placeholder
from ..resolve import WEEKDAY_WORDS, clocks, resolve
from . import catalog
from .errors import Unwritable
from .plan import PlannedTime

# The `when` conditions a form in time_formats.toml may have. Day and clock forms are
# checked against the time and when it was sent; range patterns against the range's start and end.
CONDITIONS = {
    "tomorrow": lambda local, other: (local.date() - other.date()).days == 1,
    "tonight": lambda local, other: local.date() == other.date() and local.hour >= 17,
    "today": lambda local, other: local.date() == other.date(),
    "this_week": lambda local, other: 2 <= (local.date() - other.date()).days <= 6,
    "day_over_12": lambda local, other: local.day > 12,
    "whole_hour": lambda local, other: local.minute == 0,
    "whole_pm_hour": lambda local, other: local.minute == 0 and local.hour >= 13,
    "overnight": lambda local, other: other.date() != local.date(),
}


def fits(when: str | None, local: datetime, other: datetime) -> bool:
    return when is None or CONDITIONS[when](local, other)


def pick(forms: tuple[catalog.Form, ...], rng: random.Random, local: datetime, other: datetime) -> catalog.Form:
    usable = [form for form in forms if fits(form.when, local, other)]
    return rng.choices(usable, [form.weight for form in usable])[0]


# Other short forms of a weekday people write ("Tues", "Thurs"); a day without one keeps "Mon".
LONGER_SHORT_DAYS = {"Tue": "Tues", "Wed": "Weds", "Thu": "Thurs"}


def fill_day(form: catalog.Form, local: datetime, lower: bool) -> str:
    """The day as `form` writes it, with each date in it a placeholder for the day (`dates`)."""
    day = local.date()
    names = {
        "wd": f"{local:%a}",
        "wd_long": LONGER_SHORT_DAYS.get(f"{local:%a}", f"{local:%a}"),
        "weekday": f"{local:%A}",
    }
    return (form.form.lower() if lower else form.form).format(
        **{key: name.lower() if lower else name for key, name in names.items()},
        day=placeholder(day, "d"),
        month=placeholder(day, "m"),
        mon=placeholder(day, "bl" if lower else "b"),
        month_name=placeholder(day, "Bl" if lower else "B"),
        ordinal=placeholder(day, "o"),
    )


def fill_clock(form: catalog.Form, local: datetime) -> str:
    return form.form.format(
        H=local.hour,
        HH=f"{local.hour:02d}",
        MM=f"{local.minute:02d}",
        h=local.hour % 12 or 12,
        ampm="pm" if local.hour >= 12 else "am",
        AMPM="PM" if local.hour >= 12 else "AM",
        dotted="p.m." if local.hour >= 12 else "a.m.",
    )


def weekday_typo(text: str, rng: random.Random) -> str:
    for pattern in WEEKDAY_WORDS.values():
        match = pattern.search(text)
        if match and len(match.group(0)) >= 5:
            letters = list(match.group(0))
            position = rng.randrange(1, len(letters) - 1)
            letters[position], letters[position + 1] = letters[position + 1], letters[position]
            return text[: match.start()] + "".join(letters) + text[match.end() :]
    return text


def readable(text: str, day: str) -> bool:
    """Whether a person could read it: the day comes first, and a number in the day
    doesn't run into a bare hour ("Wed 16 18")."""
    if not any(character.isdigit() for character in day):
        return True
    rest = text.replace(day, " ", 1)
    clocks = re.findall(r"\d{1,2}(?::\d{2}|h\d{0,2}|\s?[ap]m)?", rest)
    return text.startswith(day.split()[0]) and all(re.search(r"[:h]|[ap]m", clock) for clock in clocks)


def means(text: str, time: PlannedTime, sent: datetime, window: tuple[datetime, datetime], context: str = "") -> bool:
    """Whether the text reads as the planned time; "my time" in the zone its player says they are in."""
    hint = time.shift() if time.own_zone else None
    found = resolve(concrete(text), sent, window, hint, context=concrete(context))
    return [(each.start, each.end) for each in found] == time.slots()


def days_text(time: PlannedTime, sent: datetime, rng: random.Random) -> str:
    """The day part: one day in any form, or several weekdays joined ("fri or sat")."""
    formats = catalog.time_formats()
    local, sent_local = time.local_start(), sent + time.shift()
    if not time.more_days:
        form = pick(formats.days, rng, local, sent_local)
        return fill_day(form, local, rng.random() < form.lower)
    form = pick(formats.day_only, rng, local, local)
    lower = rng.random() < form.lower
    names = [fill_day(form, local + timedelta(days=days), lower) for days in (0, *time.more_days)]
    joins = [join for join in formats.several_days if join.form.count("{") == len(names)]
    return rng.choices(joins, [join.weight for join in joins])[0].form.format(*names)


def zone_text(time: PlannedTime, rng: random.Random) -> str:
    formats = catalog.time_formats()
    if time.zone and time.own_zone:
        return rng.choice((" my time", " (my time)", " my time", " in my time"))
    if time.zone:
        return f" {time.zone[0]}"
    return rng.choice(formats.utc_suffixes) if rng.random() < formats.utc_suffix_chance else ""


def candidate(time: PlannedTime, sent: datetime, rng: random.Random) -> tuple[str, str]:
    """One way of writing the time, and the day part of it."""
    formats = catalog.time_formats()
    local = time.local_start()
    day = days_text(time, sent, rng)
    zone = zone_text(time, rng)
    if time.end:
        end = time.end + time.shift()
        usable = tuple(form for form in formats.clocks if fits(form.when, end, end))
        clock = pick(usable, rng, local, local)
        patterns = tuple(form for form in formats.ranges if not (time.more_days and "{end_day}" in form.form))
        day_form = pick(formats.day_only, rng, end, end)
        pattern = pick(patterns, rng, local, end).form
        text = pattern.format(
            day=day,
            start=fill_clock(clock, local),
            end=fill_clock(clock, end),
            end_day=fill_day(day_form, end, day.islower()),
            zone=zone,
        )
    elif time.open:
        start = fill_clock(pick(formats.clocks, rng, local, local), local)
        pattern = pick(formats.open, rng, local, local).form
        text = pattern.format(day=day, start=start, zone=zone)
    else:
        clock = fill_clock(pick(formats.clocks, rng, local, local), local)
        pattern = pick(formats.points, rng, local, local).form
        text = pattern.format(day=day, clock=clock, zone=zone)
    return text if "{zone}" in pattern else text + zone, day


def written(time: PlannedTime, sent: datetime, window: tuple[datetime, datetime], rng: random.Random) -> str:
    """A way of writing `time`, as sent at `sent`, that reads back as exactly the planned time."""
    formats = catalog.time_formats()
    for _ in range(60):
        text, day = candidate(time, sent, rng)
        if not (readable(concrete(text), concrete(day)) and means(text, time, sent, window)):
            continue
        if rng.random() < formats.weekday_typo_chance:
            typo = weekday_typo(text, rng)
            if means(typo, time, sent, window):
                return typo
        return text
    raise Unwritable(f"no readable form for {time}")


def written_clock(
    time: PlannedTime, sent: datetime, window: tuple[datetime, datetime], rng: random.Random, context: str
) -> str | None:
    """Only the clock, for a time on the day just named before it ("fri 01:30? 01:45?"), if it
    reads back as the planned time with that day."""
    formats = catalog.time_formats()
    local = time.local_start()
    for _ in range(10):
        text = fill_clock(pick(formats.clocks, rng, local, local), local) + zone_text(time, rng)
        if clocks(text)[0].explicit and means(text, time, sent, window, context):
            return text
    return None


def written_local(time: PlannedTime, rng: random.Random) -> str:
    """The clock in the player's own zone, unnamed, as when converting a time for oneself ("3:30pm")."""
    local = time.local_start()
    forms = [form for form in catalog.time_formats().clocks if fits(form.when, local, local)]
    explicit = [form for form in forms if clocks(fill_clock(form, local))[0].explicit]
    return fill_clock(rng.choices(explicit, [form.weight for form in explicit])[0], local)


def written_day(time: PlannedTime, rng: random.Random) -> str:
    """Only the day of a time ("thu", "Thursday"), for an acceptance that names the day alone."""
    local = time.local_start()
    form = pick(catalog.time_formats().day_only, rng, local, local)
    return fill_day(form, local, rng.random() < form.lower)
