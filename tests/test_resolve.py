from datetime import UTC, datetime, timedelta

import pytest

from chess_league_bot_nlu.resolve import resolve, weekday_of

SENT = datetime(2026, 11, 2, 19, 12, tzinfo=UTC)  # a Monday
WINDOW = (datetime(2026, 11, 2, 18, tzinfo=UTC), datetime(2026, 11, 9, 12, tzinfo=UTC))


def utc(day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 11, day, hour, minute, tzinfo=UTC)


@pytest.mark.parametrize(
    "text, start, end",
    [
        ("tue 20:00", utc(3, 20), None),
        ("thu 21:00 CET", utc(5, 20), None),
        ("tmrw 8pm", utc(3, 20), None),
        ("11/6 19:00", utc(6, 19), None),
        ("sunday anytime after 14", utc(8, 14), None),
        ("wed 18:00 until 20:00", utc(4, 18), utc(4, 20)),
        ("wed 7-9pm", utc(4, 19), utc(4, 21)),
        ("wed between 7 and 9pm", utc(4, 19), utc(4, 21)),
        ("sat 22:30-00:00", utc(7, 22, 30), utc(8, 0)),
        ("the 5th at 9pm EST", utc(6, 2), None),
        ("Nov 6 09:00", utc(6, 9), None),
        ("Sun 8 20:00", utc(8, 20), None),
        ("2nd option: thu 20:00", utc(5, 20), None),
        ("thu 12th 20:00", None, None),
        ("fri 19h", utc(6, 19), None),
        ("thu 9:00", utc(5, 9), None),
        ("thu 7", utc(5, 7), None),
        ("thu 7:30", utc(5, 7, 30), None),
        ("wed 11-1pm", utc(4, 11), utc(4, 13)),
        ("Thu 18:00 UTC-20:00", utc(5, 18), utc(5, 20)),
        ("Thu 9:00 UTC-11:00 UTC", utc(5, 9), utc(5, 11)),
        ("sat 10:00 utc-12:00", utc(7, 10), utc(7, 12)),
        ("tue 20:00 utc-5", utc(4, 1), None),
        ("sat 18", utc(7, 18), None),
        ("fri 0300", utc(6, 3), None),
        ("2100 - 2300 UTC on Friday", utc(6, 21), utc(6, 23)),
        ("11/6 @ 1030 GMT", utc(6, 10, 30), None),
        ("Sat 01;30", utc(7, 1, 30), None),
        ("6/11@18:00GMT", utc(6, 18), None),
        ("friday 6.11. @ 16:00 GMT", utc(6, 16), None),
        ("06.11.2026 @ 16:00", utc(6, 16), None),
        ("17.00 GMT 06-11", utc(6, 17), None),
        ("sat 17.30", utc(7, 17, 30), None),
        ("wed 19-21 utc", utc(4, 19), utc(4, 21)),
        ("fri 22:00 - sat 02:00", utc(6, 22), utc(7, 2)),
        ("Friday 23:00 - 1:30 Sat", utc(6, 23), utc(7, 1, 30)),
        ("wed 4 nov 19:00", utc(4, 19), None),
        ("sat 7/11 20:00", utc(7, 20), None),
        ("sat 4/11 20:00", utc(7, 20), None),  # the 4th is a Wednesday; the weekday wins
        ("sat nov 4 19:00", utc(7, 19), None),
    ],
)
def test_expressions_resolve_to_utc(text, start, end):
    found = resolve(text, SENT, WINDOW)
    assert [(time.start, time.end) for time in found] == ([(start, end)] if start else [])


@pytest.mark.parametrize(
    "text",
    [
        "thu 24:00",  # not a time of day
        "mon 19:00",  # this Monday has passed and next Monday is after the deadline
        "20:00",  # no day
        "@**rook4h** sat",  # a name, not 04:00
    ],
)
def test_ambiguous_or_impossible_expressions_resolve_to_nothing(text):
    assert resolve(text, SENT, WINDOW) == []


def test_two_alternatives_are_not_a_range():
    [found] = resolve("tomorrow 7pm or 9pm", SENT, WINDOW)
    assert found.start == utc(3, 19) and found.end is None


def test_a_bare_clock_time_borrows_the_day_from_context():
    assert resolve("could we do 19:00 instead", SENT, WINDOW, context="wed")[0].start == utc(4, 19)


def test_a_zone_hint_applies_when_the_expression_names_none():
    assert resolve("tue 20:00", SENT, WINDOW, zone_hint=timedelta(hours=1))[0].start == utc(3, 19)


def test_sat_and_sun_as_ordinary_words_are_not_days():
    assert weekday_of("I sat out last round") is None
    assert weekday_of("the sun is out") is None
    assert weekday_of("sun's out") is None
    assert weekday_of("sat 18:00") == 5
    assert weekday_of("tursday") == 3


def test_an_ordinal_that_is_not_a_date_does_not_set_the_day():
    assert resolve("2nd round already! 20:00 on thursday", SENT, WINDOW)[0].start == utc(5, 20)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("2100 - 2300 UTC on Friday or Saturday", [(utc(6, 21), utc(6, 23)), (utc(7, 21), utc(7, 23))]),
        ("fri or sat 21:00", [(utc(6, 21), None), (utc(7, 21), None)]),
        ("Thu/Fri 19:30 CET", [(utc(5, 18, 30), None), (utc(6, 18, 30), None)]),
        ("21:00 on wed, thu or sat", [(utc(4, 21), None), (utc(5, 21), None), (utc(7, 21), None)]),
        ("sat and sun 13-20 utc", [(utc(7, 13), utc(7, 20)), (utc(8, 13), utc(8, 20))]),
        ("Friday 23:00 - 1:30 Sat", [(utc(6, 23), utc(7, 1, 30))]),
        ("sat or mon 19:00", []),  # Monday 19:00 has passed or is after the deadline, so none of it reads
    ],
)
def test_one_time_on_several_days_resolves_to_each_day(text, expected):
    assert [(time.start, time.end) for time in resolve(text, SENT, WINDOW)] == expected
