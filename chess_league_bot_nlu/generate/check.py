"""Mechanical checks on what the model wrote, before placeholders are filled in.

In a scheduling thread, anything that looks like a time outside the
placeholders would be an unlabelled time in the data, so it is rejected.
Messages to the bot have no time labels, so `times=False` skips those checks.
Once a thread is filled in, `misread` reads every labelled time again in its
place in the thread, as evaluation will, since a time may take its day from
the text before it, and `ambiguous` finds acceptances that name only a day
the opponent offered more than one time on.
"""

from __future__ import annotations

import re

from ..records import Mention, Thread, parse_time
from ..resolve import RELATIVE_WORD, WEEKDAY_WORDS, clocks, weekday_of, weekdays_in
from ..transcript import resolve_span
from .placeholders import PLACEHOLDER

MENTION = re.compile(r"@\*\*|@[A-Za-z]")
WEEKDAY_WITH_NUMBER = re.compile(
    r"\b(?:" + "|".join(pattern.pattern for pattern in WEEKDAY_WORDS.values()) + r")\s+\d", re.IGNORECASE
)
AT_NUMBER = re.compile(r"(?:\bat|@)\s*\d{1,2}\b", re.IGNORECASE)


def problems(
    template: str,
    placeholders: list[str],
    words: int,
    *,
    optional: list[str] = (),
    times: bool = True,
    allow_weekday: bool = False,
) -> list[str]:
    """What is wrong with a message template, or nothing. `optional` placeholders may be used once or not at all."""
    found = []
    used = PLACEHOLDER.findall(template)
    for name in [*placeholders, *optional]:
        count = used.count(f"{{{name}}}")
        if count > 1 or (count == 0 and name in placeholders):
            found.append(f"placeholder {{{name}}} used {count} times")
    unknown = sorted(set(used) - {f"{{{name}}}" for name in [*placeholders, *optional]})
    if unknown:
        found.append(f"unplanned placeholders {unknown}")
    outside = PLACEHOLDER.sub(" ", template)
    if "{" in outside or "}" in outside:
        found.append("stray braces")
    if times:
        found += time_problems(outside, allow_weekday)
    if MENTION.search(outside):
        found.append("an @-mention written out")
    count = len(re.findall(r"\w+", outside))
    if count > words * 1.5 + 4:
        found.append(f"{count} words, limit {words}")
    if not outside.strip() and not set(used) - {f"{{{name}}}" for name in optional}:
        found.append("empty message")
    return found


def time_problems(outside: str, allow_weekday: bool) -> list[str]:
    found = []
    if any(clock.explicit for clock in clocks(outside)) or AT_NUMBER.search(outside):
        found.append("a clock time outside the placeholders")
    if WEEKDAY_WITH_NUMBER.search(outside):
        found.append("a day and number outside the placeholders")
    if any(match.group(0).lower() != "today" for match in RELATIVE_WORD.finditer(outside)):
        found.append("a relative day (tomorrow, tonight) outside the placeholders")
    if not allow_weekday and any(pattern.search(outside) for pattern in WEEKDAY_WORDS.values()):
        found.append("a weekday outside the placeholders")
    return found


def misread(thread: Thread) -> list[str]:
    """Labelled times that, read in their place in the thread, don't mean the planned time.
    A day alone ("thu works") names its offer's time and is not read on its own."""
    found = []
    for mention in thread.mentions:
        text = thread.text_of(mention)
        if mention.role == "other" or not clocks(text):
            continue
        message = thread.message(mention.message)
        read = [(time.start, time.end) for time in resolve_span(thread, message, mention.start, mention.end)]
        planned = [(parse_time(time.start), parse_time(time.end) if time.end else None) for time in mention.times]
        if read != planned:
            found.append(f"{text!r} in {message.id} reads as {[f'{start:%a %H:%M}' for start, _ in read]}")
    return found


def ambiguous(thread: Thread) -> list[str]:
    """Acceptances that name only a day ("tue works") when more than one of the opponent's
    standing offers falls on that day, so the text can't say which one was accepted."""
    order = {message.id: index for index, message in enumerate(thread.messages)}
    withdrawn = {withdrawal.offer: order[withdrawal.message] for withdrawal in thread.withdrawals}
    found = []
    for mention in thread.mentions:
        text = thread.text_of(mention)
        if mention.role != "accepted" or clocks(text) or (weekday := weekday_of(text)) is None:
            continue
        message = thread.message(mention.message)
        when = order[message.id]
        matching = [
            offer.id
            for offer in thread.mentions
            if offer.role == "offered"
            and order[offer.message] < when
            and thread.message(offer.message).speaker != message.speaker
            and withdrawn.get(offer.id, len(order)) > when
            and weekday in days_of(thread, offer)
        ]
        if len(matching) > 1:
            found.append(f"{text!r} in {message.id} could accept any of {matching}")
    return found


def days_of(thread: Thread, offer: Mention) -> set[int]:
    """The weekdays an offer is on as its writer named them, or in UTC when it names none."""
    named = {day for _, _, day in weekdays_in(thread.text_of(offer))}
    return named or {parse_time(time.start).weekday() for time in offer.times}
