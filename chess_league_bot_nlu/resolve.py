"""Resolve a time expression from a scheduling message to UTC.

The model only finds expressions like "thu 21:00 CET" or "tmrw 8pm"; this turns
them into instants, anchored on when the message was sent and limited to the
round. A reading that is ambiguous or falls outside the round resolves to
nothing: a date that could be either 11 June or 6 November inside the round,
or a time with no day at all. A clock time without am/pm is read as 24-hour:
"thu 7" is 07:00. One time offered on several days ("21:00 on fri or sat")
resolves to one instant per day, in the order the days are written.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta, timezone
from itertools import pairwise

ZONES = {
    "utc": 0,
    "gmt": 0,
    "cet": 1,
    "cest": 2,
    "bst": 1,
    "eet": 2,
    "est": -5,
    "edt": -4,
    "cst": -6,
    "cdt": -5,
    "mst": -7,
    "mdt": -6,
    "pst": -8,
    "pdt": -7,
    "ist": 5.5,
    "aedt": 11,
    "aest": 10,
    "brt": -3,
    "jst": 9,
    "sgt": 8,
    "msk": 3,
}
ZONE = re.compile(
    r"(?:\b|(?<=\d))(?:(utc|gmt)([+-](?:1[0-4]|0?\d)(?::?[0-5]\d)?)(?![\d:])|("
    + "|".join(sorted(ZONES, key=len, reverse=True))
    + r"))\b",
    re.IGNORECASE,
)

WEEKDAYS = {
    0: r"mon(?:day)?",
    1: r"tues?(?:day)?",
    2: r"wed(?:s|nesday|nsday)?",
    3: r"thu(?:r|rs|rsday)?|thrusday|thursdy",
    4: r"fri(?:day)?",
    5: r"sat(?:urday|urdy)?",
    6: r"sun(?:day)?",
}
WEEKDAY_WORDS = {
    weekday: re.compile(rf"\b(?:{pattern})\b(?!['’]s\b)\.?", re.IGNORECASE) for weekday, pattern in WEEKDAYS.items()
}
FULL_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
NOT_WEEKDAYS = {
    "today",
    "yesterday",
    "someday",
    "birthday",
    "holiday",
    "everyday",
    "workday",
    "weekday",
    "midday",
    "payday",
    "sundays",
}
# "I sat out last round", "the sun is up": not days.
NOT_A_DAY_AFTER = {"sat": {"i", "we", "he", "she", "they", "you", "just", "had", "have", "never"}, "sun": {"the", "a"}}

MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
RELATIVE = {
    "today": 0,
    "tonight": 0,
    "tonite": 0,
    "tmrw": 1,
    "tmr": 1,
    "tmw": 1,
    "tomorrow": 1,
    "tomorow": 1,
    "tommorow": 1,
    "tommorrow": 1,
    "tomoz": 1,
    "2moro": 1,
}
RELATIVE_WORD = re.compile(r"\b(" + "|".join(RELATIVE) + r")\b", re.IGNORECASE)
NUMERIC_DATE = re.compile(r"\b(\d{1,2})/(\d{1,2})\b")
# "17.04." or "01.05.2016" or "27-04", day first; a date only beside a clock time, since alone it may be one ("17.30", "19-21").
DOTTED_DATE = re.compile(r"(?<![\d/.:;])(\d{1,2})([.-])(\d{1,2})(?:\2(?:\d{4}|\d{2})|\.)?(?![\d:;/]|[.-]\d)")
MONTH_DAY = re.compile(
    r"\b(" + "|".join(MONTHS) + r")[a-z]*\.?\s+(\d{1,2})(?:st|nd|rd|th)?\b(?![:.;h]\d)", re.IGNORECASE
)
DAY_MONTH = re.compile(r"(?<![:.;])\b(\d{1,2})(?:st|nd|rd|th)?\s+(" + "|".join(MONTHS) + r")[a-z]*\b", re.IGNORECASE)
# "the 12th" is a date; "2nd round" or "3rd time" is not.
ORDINAL = re.compile(
    r"\b(?:the\s+)?(\d{1,2})(st|nd|rd|th)\b(?!\s+(?:round|rd|option|one|time|game|board|place|try|attempt|choice))",
    re.IGNORECASE,
)
CLOCK = re.compile(
    r"(?<![\d/.:;A-Za-z])(?:(\d{1,2})(?:([:h.;])(\d{2}))?|([01]\d|2[0-3])([0-5]\d))"
    r"(?:\s*(am|pm|a\.m\.|p\.m\.|h)\b)?(?![\d/:;])(?!(?!utc|gmt)[a-z])",
    re.IGNORECASE,
)
# "9:00 UTC-11:00" is a range written with its zone in the middle, not UTC-11.
ZONE_INSIDE_RANGE = re.compile(r"(\d(?:[:h.]\d{2})?\s*(?:[ap]m)?\s*)(utc|gmt)-(?=\d{1,2}[:h.]\d{2})", re.IGNORECASE)
ANY_WEEKDAY = re.compile("|".join(rf"\b(?:{pattern})\b\.?" for pattern in WEEKDAYS.values()), re.IGNORECASE)
RANGE_BETWEEN = re.compile(r"^\s*(?:-|–|—|to|until|till|til|and)\s*$", re.IGNORECASE)
BETWEEN = re.compile(r"\bbetween\b", re.IGNORECASE)
OPEN_START = re.compile(r"\b(?:after|from)\b", re.IGNORECASE)
DAY_JOIN = re.compile(r"^\s*(?:,|/|&|,?\s*\b(?:or|and)\b)?\s*(?:on\s+)?$", re.IGNORECASE)


@dataclass(frozen=True)
class Resolved:
    start: datetime
    end: datetime | None = None


@dataclass(frozen=True)
class Clock:
    hour: int
    minute: int
    suffix: str
    separator: str
    start: int
    end: int

    @property
    def explicit(self) -> bool:
        """Whether it looks like a clock time ("19:00", "7pm", "19h") rather than a bare number."""
        return bool(self.separator or self.suffix)


def zone_offset(text: str) -> timedelta | None:
    match = ZONE.search(text)
    if not match:
        return None
    if match.group(2):
        sign = -1 if "-" in match.group(2) else 1
        digits = re.sub(r"\D", "", match.group(2))
        hours, minutes = (int(digits[:-2]), int(digits[-2:])) if len(digits) > 2 else (int(digits), 0)
        return sign * timedelta(hours=hours, minutes=minutes)
    return timedelta(hours=ZONES[match.group(3).lower()])


def weekdays_in(text: str) -> list[tuple[int, int, int]]:
    """Every weekday named, misspelled or not, as (start, end, weekday)."""
    found = []
    for weekday, pattern in WEEKDAY_WORDS.items():
        for match in pattern.finditer(text):
            before = text[: match.start()].split()
            exceptions = NOT_A_DAY_AFTER.get(match.group(0).lower().rstrip("."), set())
            if before and before[-1].lower() in exceptions:
                continue
            found.append((match.start(), match.end(), weekday))
    for word in re.finditer(r"[A-Za-z]{5,}", text):
        lowered = word.group(0).lower()
        if (
            lowered.endswith(("day", "dy"))
            and lowered not in NOT_WEEKDAYS
            and not any(start <= word.start() < end for start, end, _ in found)
        ):
            close = difflib.get_close_matches(lowered, FULL_WEEKDAYS, n=1, cutoff=0.75)
            if close:
                found.append((word.start(), word.end(), FULL_WEEKDAYS.index(close[0])))
    return sorted(found)


def weekday_of(text: str) -> int | None:
    found = weekdays_in(text)
    return found[0][2] if found else None


def day_alternatives(text: str) -> list[tuple[int, int, int]]:
    """Weekdays offered as alternatives for one time ("fri or sat 21:00"): two or more, joined
    only by "or", "and", "/" or commas, as (start, end, weekday). A day for each end of a
    range ("fri 22:00 - sat 02:00") is not a list of alternatives."""
    days = weekdays_in(text)
    if len(days) < 2 or not all(DAY_JOIN.match(text[first[1] : second[0]]) for first, second in pairwise(days)):
        return []
    return days


def last_date_phrase(text: str) -> str | None:
    """The last thing in the text that names a day, for a later bare clock time to borrow."""
    spots = [(match.start(), match.group(0)) for match in RELATIVE_WORD.finditer(text)]
    for pattern in (MONTH_DAY, DAY_MONTH, NUMERIC_DATE):
        spots += [(match.start(), match.group(0)) for match in pattern.finditer(text)]
    spots += [(match.start(), match.group(0)) for match in ORDINAL.finditer(text)]
    spots += [(start, text[start:end]) for start, end, _ in weekdays_in(text)]
    return max(spots)[1] if spots else None


def mask(text: str, pattern: re.Pattern) -> str:
    return pattern.sub(lambda match: " " * len(match.group(0)), text)


def dotted_dates(text: str) -> list[re.Match]:
    """Day-first numeric dates like "17.04." or "27-04", which count only beside a clock time."""
    found = [
        match
        for match in DOTTED_DATE.finditer(text)
        if 1 <= int(match.group(1)) <= 31 and 1 <= int(match.group(3)) <= 12
    ]
    masked = text
    for match in found:
        masked = masked[: match.start()] + " " * len(match.group(0)) + masked[match.end() :]
    return found if found and read_clocks(masked) else []


def clocks(text: str) -> list[Clock]:
    """Clock times in the text, with dates and zones masked out so positions stay put."""
    masked = text
    for pattern in (ZONE, NUMERIC_DATE, MONTH_DAY, DAY_MONTH, ORDINAL):
        masked = mask(masked, pattern)
    for match in dotted_dates(masked):
        masked = masked[: match.start()] + " " * len(match.group(0)) + masked[match.end() :]
    return read_clocks(masked)


def read_clocks(masked: str) -> list[Clock]:
    found = []
    for match in CLOCK.finditer(masked):
        if match.group(4):
            hour, minute, separator = int(match.group(4)), int(match.group(5)), ":"
        else:
            hour, minute, separator = int(match.group(1)), int(match.group(3) or 0), match.group(2) or ""
        if hour > 23 or minute > 59:
            continue
        suffix = (match.group(6) or "").lower().replace(".", "")
        separator = ":" if separator == ";" else separator or ("h" if suffix == "h" else "")
        found.append(Clock(hour, minute, suffix, separator, match.start(), match.end()))
    return found


def to_time(clock: Clock, suffix: str | None = None) -> time:
    """The clock as a time of day; without am/pm it is a 24-hour time."""
    suffix = clock.suffix if suffix is None else suffix
    hour = clock.hour
    if suffix == "pm" and hour < 12:
        hour += 12
    elif suffix == "am" and hour == 12:
        hour = 0
    return time(hour, clock.minute)


def range_start_suffix(start: Clock, end: Clock) -> str | None:
    """The am/pm a range's start borrows from its end ("7-9pm"), keeping the start before the end ("11-1pm")."""
    if start.suffix or end.suffix not in ("am", "pm"):
        return None
    if end.suffix == "pm" and start.hour < 12 and start.hour + 12 > (end.hour % 12) + 12:
        return "am"
    return end.suffix


def day_number(text: str, found: list[Clock]) -> tuple[int, list[Clock]] | None:
    """A bare number right after a weekday, followed by a real clock time, is the day of the month: "Sun 6 20:00"."""
    if len(found) < 2 or found[0].explicit:
        return None
    first, second = found[0], found[1]
    after_weekday = any(
        end <= first.start and not text[end : first.start].strip(" .,") for _, end, _ in weekdays_in(text)
    )
    if after_weekday and 1 <= first.hour <= 31 and not RANGE_BETWEEN.match(text[first.end : second.start]):
        return first.hour, found[1:]
    return None


def around(sent: date, month: int, day: int) -> list[date]:
    """The date with this month and day nearest after `sent` (this year, or next year in early January)."""
    for year in (sent.year, sent.year + 1):
        try:
            candidate = date(year, month, day)
        except ValueError:
            return []
        if candidate >= sent - timedelta(days=31):
            return [candidate]
    return []


def candidate_dates(text: str, sent: date, day: int | None = None) -> list[date] | None:
    """Dates the expression could mean, or None when it names no date."""
    relative = RELATIVE_WORD.search(text)
    if relative:
        return [sent + timedelta(days=RELATIVE[relative.group(1).lower()])]
    weekday = weekday_of(text)
    found = []
    for pattern, month_group, day_group in ((MONTH_DAY, 1, 2), (DAY_MONTH, 2, 1)):
        for match in pattern.finditer(text):
            month = MONTHS.index(match.group(month_group).lower()[:3]) + 1
            found += around(sent, month, int(match.group(day_group)))
    for match in dotted_dates(mask(text, ZONE)):
        found += around(sent, int(match.group(3)), int(match.group(1)))
    numeric = NUMERIC_DATE.search(text)
    if numeric and not found:
        first, second = int(numeric.group(1)), int(numeric.group(2))
        found = around(sent, first, second) + around(sent, second, first)
    if found:
        matching = [candidate for candidate in found if weekday is None or candidate.weekday() == weekday]
        if matching:
            return matching
        # A date that contradicts its weekday ("sat 4/11" when the 4th is a Wednesday): the weekday wins.
    ordinal = ORDINAL.search(text)
    if day is None and ordinal:
        day = int(ordinal.group(1))
    if day is not None:
        by_day = around(sent, sent.month, day) + around(sent, sent.month % 12 + 1, day)
        if weekday is None:
            return by_day
        matching = [candidate for candidate in by_day if candidate.weekday() == weekday]
        if matching:
            return matching
    if weekday is not None:
        ahead = (weekday - sent.weekday()) % 7
        return [sent + timedelta(days=ahead), sent + timedelta(days=ahead + 7)]
    return None


def resolve(
    expression: str,
    sent: datetime,
    window: tuple[datetime, datetime],
    zone_hint: timedelta | None = None,
    context: str = "",
) -> list[Resolved]:
    """Resolve an expression: one instant, one per day it offers the time on, or none if any day
    fails. `context` names the day when the expression only gives a clock time."""
    days = day_alternatives(expression)
    if not days:
        found = resolve_one(expression, sent, window, zone_hint, context)
        return [found] if found else []
    found = []
    for kept in days:
        only = expression
        for start, end, _ in days:
            if (start, end) != kept[:2]:
                only = only[:start] + " " * (end - start) + only[end:]
        found.append(resolve_one(only, sent, window, zone_hint))
    if None in found or len({instant.start for instant in found}) < len(found):
        return []
    return found


def resolve_one(
    expression: str,
    sent: datetime,
    window: tuple[datetime, datetime],
    zone_hint: timedelta | None = None,
    context: str = "",
) -> Resolved | None:
    expression = ZONE_INSIDE_RANGE.sub(lambda match: f"{match.group(1)}{match.group(2)} - ", expression)
    offset = zone_offset(expression)
    if offset is None:
        offset = zone_hint if zone_hint is not None else timedelta(0)
    zone = timezone(offset)
    sent_local = sent.astimezone(zone).date()

    found = clocks(expression)
    numbered = day_number(expression, found)
    day, found = numbered if numbered else (None, found)
    if not found:
        return None
    ranged = False
    if len(found) > 1:
        between = mask(mask(expression, ZONE)[found[0].end : found[1].start], ANY_WEEKDAY)
        ranged = bool(RANGE_BETWEEN.match(between)) and (
            between.strip().lower() != "and" or bool(BETWEEN.search(expression))
        )
    end_time = to_time(found[1]) if ranged else None
    start_time = to_time(found[0], range_start_suffix(found[0], found[1]) if ranged else None)

    dates = candidate_dates(expression, sent_local, day)
    if dates is None and context:
        dates = candidate_dates(context, sent_local)
    if dates is None:
        return None
    instants = [datetime.combine(candidate, start_time, zone).astimezone(UTC) for candidate in dates]
    starts = sorted(instant for instant in instants if sent <= instant <= window[1])
    if not starts:
        return None
    named = expression + " " + context
    next_occurrence = weekday_of(named) is not None or RELATIVE_WORD.search(named)
    if len({instant.date() for instant in starts}) > 1 and not next_occurrence:
        return None
    start = starts[0]
    end = None
    if ranged:
        end = datetime.combine(start.astimezone(zone).date(), end_time, zone).astimezone(UTC)
        if end <= start:
            end += timedelta(days=1)
    return Resolved(start, end)
