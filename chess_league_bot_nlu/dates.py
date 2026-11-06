"""Dates in message text as placeholders: a day, and how that day is written.

A stored thread names each day by its distance from the pairing ("{+3:b}" is the short month of
the day three days after the pairing), so it can be put on any week of the year. The generator
writes the day itself ("{2026-11-24:b}") and the record turns it into a distance from the pairing.
Weekday names stay as written: a thread only moves by whole weeks, so they never change.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta

MONTH_NAMES = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


def ordinal(day: int) -> str:
    return f"{day}{'th' if 11 <= day <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')}"


# How a placeholder writes its day: "d" 4, "dd" 04, "m" 11, "mm" 11, "o" 4th, "b" Nov, "B" November,
# and "bl"/"Bl" the month names in lower case.
FIELDS = {
    "d": lambda day: str(day.day),
    "dd": lambda day: f"{day.day:02d}",
    "m": lambda day: str(day.month),
    "mm": lambda day: f"{day.month:02d}",
    "o": lambda day: ordinal(day.day),
    "b": lambda day: MONTH_NAMES[day.month - 1][:3],
    "bl": lambda day: MONTH_NAMES[day.month - 1][:3].lower(),
    "B": lambda day: MONTH_NAMES[day.month - 1],
    "Bl": lambda day: MONTH_NAMES[day.month - 1].lower(),
}
PLACEHOLDER = re.compile(r"\{([+-]\d+|\d{4}-\d{2}-\d{2}):(dd|d|mm|m|o|bl|b|Bl|B)\}")


def placeholder(day: date, field: str) -> str:
    return f"{{{day.isoformat()}:{field}}}"


def day_of(written: str, pairing: date | None) -> date:
    if written[0] in "+-":
        if pairing is None:
            raise ValueError(f"a day after the pairing ({written}) with no pairing")
        return pairing + timedelta(days=int(written))
    return date.fromisoformat(written)


@dataclass(frozen=True)
class Rendered:
    """A template with its days written, and where each placeholder went:
    (template start, template end, text start, text end)."""

    text: str
    pieces: tuple[tuple[int, int, int, int], ...]

    def in_text(self, position: int) -> int:
        return self.moved(position, 0, 2)

    def in_template(self, position: int) -> int:
        return self.moved(position, 2, 0)

    def moved(self, position: int, source: int, target: int) -> int:
        for piece in self.pieces:
            if piece[source] < position < piece[source + 1]:
                raise ValueError(f"position {position} is inside a date")
        return position + sum(
            (piece[target + 1] - piece[target]) - (piece[source + 1] - piece[source])
            for piece in self.pieces
            if piece[source + 1] <= position
        )


def rendered(template: str, pairing: date | None = None) -> Rendered:
    text, pieces, cursor = "", [], 0
    for match in PLACEHOLDER.finditer(template):
        text += template[cursor : match.start()]
        word = FIELDS[match.group(2)](day_of(match.group(1), pairing))
        pieces.append((match.start(), match.end(), len(text), len(text) + len(word)))
        text += word
        cursor = match.end()
    return Rendered(text + template[cursor:], tuple(pieces))


def concrete(template: str) -> str:
    """The text, for a template whose placeholders name their days."""
    return rendered(template).text


def relative(template: str, pairing: date) -> str:
    """The template with each day named by its distance from the pairing."""

    def replaced(match: re.Match) -> str:
        days = (day_of(match.group(1), pairing) - pairing).days
        return f"{{{days:+d}:{match.group(2)}}}"

    return PLACEHOLDER.sub(replaced, template)


MONTH = r"(january|february|march|april|may|june|july|august|september|october|november|december|sept|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)"
WEEKDAY = r"(?:mon|tue|wed|thu|fri|sat|sun)[a-z]*"
# Where people write a day of the month, as (pattern, the groups holding the day, month and suffix).
WRITTEN = (
    (re.compile(rf"\b(\d{{1,2}})(st|nd|rd|th)?(?:\s+of)?\s+{MONTH}\b", re.IGNORECASE), 1, 3, 2),
    (re.compile(rf"\b{MONTH}\.?\s+(\d{{1,2}})(st|nd|rd|th)?\b(?![:.;h]\d)", re.IGNORECASE), 2, 1, 3),
    (re.compile(r"(?<![\d/.:])(\d{1,2})/(\d{1,2})(?![\d/])"), 1, 2, None),
    (re.compile(r"(?<![\d/.:])(\d{1,2})\.(\d{1,2})\.(?!\d)"), 1, 2, None),
    (re.compile(r"\b(\d{1,2})(st|nd|rd|th)\b", re.IGNORECASE), 1, None, 2),
    (re.compile(rf"\b{WEEKDAY}\.?\s+(\d{{1,2}})(?=\s+(?:at\s+|@\s*)?\d)", re.IGNORECASE), 1, None, None),
)


def month_of(written: str) -> int:
    if written.isdigit():
        return int(written)
    return next(
        index for index, name in enumerate(MONTH_NAMES, start=1) if name.lower().startswith(written.lower()[:3])
    )


def written_dates(text: str, days: list[date]) -> list[tuple[int, int, date, str]]:
    """Each written part of a date in `text` that names exactly one of `days`, as (start, end, day,
    field), where the field writes that day exactly as it is written. A date that names none or
    several of them, or that no field writes exactly ("Sept", "3RD"), is left out."""
    found, taken = [], []
    for pattern, day_group, month_group, suffix_group in WRITTEN:
        for match in pattern.finditer(text):
            if any(match.start() < end and start < match.end() for start, end in taken):
                continue
            month = month_of(match.group(month_group)) if month_group else None
            matching = [day for day in days if day.day == int(match.group(day_group)) and month in (None, day.month)]
            if len(matching) != 1:
                continue
            day = matching[0]
            if suffix_group and match.group(suffix_group):
                parts = [(match.start(day_group), match.end(suffix_group), "o")]
            else:
                number = match.group(day_group)
                parts = [(*match.span(day_group), "dd" if number.startswith("0") else "d")]
            if month_group:
                parts.append((*match.span(month_group), month_field(match.group(month_group))))
            if all(FIELDS[field](day) == text[start:end] for start, end, field in parts):
                taken.append(match.span())
                found += [(start, end, day, field) for start, end, field in parts]
    return sorted(found)


def month_field(written: str) -> str:
    if written.isdigit():
        return "mm" if written.startswith("0") else "m"
    name = "B" if written.capitalize() in MONTH_NAMES else "b"
    return name + "l" if written.islower() else name


def round_days(published: date, deadline: date, sent: date) -> list[date]:
    """The days a message may name by their date: the round's, from the day the message was sent."""
    first, last = published - timedelta(days=1), deadline + timedelta(days=1)
    days = [first + timedelta(days=offset) for offset in range((last - first).days + 1)]
    return sorted(days, key=lambda day: (day < sent, day))


def unplaced(template: str, days: list[date]) -> list[str]:
    """Dates written outside the placeholders that could name one of `days`."""
    found = []
    for piece in PLACEHOLDER.split(template):
        for pattern, day_group, month_group, _ in WRITTEN:
            for match in pattern.finditer(piece):
                month = month_of(match.group(month_group)) if month_group else None
                if any(day.day == int(match.group(day_group)) and month in (None, day.month) for day in days):
                    found.append(match.group(0))
    return found


def templated(text: str, days: list[date]) -> str:
    """The text with each date in it that names one of `days` as a placeholder for that day. Text
    already holding placeholders keeps them."""
    parts, index = [], 0
    for match in PLACEHOLDER.finditer(text):
        parts += [text[index : match.start()], match.group(0)]
        index = match.end()
    parts.append(text[index:])
    for position in range(0, len(parts), 2):
        piece = parts[position]
        for start, end, day, field in reversed(written_dates(piece, days)):
            piece = piece[:start] + placeholder(day, field) + piece[end:]
        parts[position] = piece
    return "".join(parts)
