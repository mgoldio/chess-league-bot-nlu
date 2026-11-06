import pytest

from chess_league_bot_nlu.generate import check
from chess_league_bot_nlu.records import Acceptance, Mention, Message, Thread, Utc


@pytest.mark.parametrize(
    "template, problem",
    [
        ("how about {T1}", None),
        ("how about it", "placeholder {T1} used 0 times"),
        ("{T1} or {T1}", "placeholder {T1} used 2 times"),
        ("how about {T1} or 20:00", "a clock time outside the placeholders"),
        ("{T1} or maybe thu 9", "a day and number outside the placeholders"),
        ("{T1}, or tomorrow at 8", "a clock time outside the placeholders"),
        ("{T1} or tomorrow", "a relative day (tomorrow, tonight) outside the placeholders"),
        ("{T1}, @**Bob** says hi", "an @-mention written out"),
    ],
)
def test_checks_catch_what_the_writer_got_wrong(template, problem):
    found = check.problems(template, ["T1"], 20)
    assert (problem in found) if problem else not found


def test_command_messages_may_mention_days():
    assert not check.problems("cant play {round}, away tomorrow", ["round"], 20, times=False)


@pytest.mark.parametrize("seed", range(300))
def test_every_planned_time_reads_back_in_its_place_in_the_thread(seed, perfect):
    _, thread = perfect(seed)
    assert check.misread(thread) == []


def test_today_in_passing_is_not_a_time_but_tomorrow_is():
    assert not check.problems("{T1}? busy day today", ["T1"], 20)
    assert check.problems("{T1} or tomorrow?", ["T1"], 20)


def ambiguity_thread(second_offer: str) -> Thread:
    messages = [
        Message("m1", "white", "2026-12-01T10:00:00Z", f"thu 09:00 or {second_offer}"),
        Message("m2", "black", "2026-12-01T11:00:00Z", "thursday works"),
    ]
    first = len("thu 09:00")
    mentions = [
        Mention("T1", "m1", 0, first, "offered", [Utc("2026-12-03T09:00:00Z")]),
        Mention("T2", "m1", first + 4, first + 4 + len(second_offer), "offered", [Utc("2026-12-03T20:00:00Z")]),
        Mention("T3", "m2", 0, 8, "accepted", [Utc("2026-12-03T09:00:00Z")], accepts="T1"),
    ]
    return Thread(
        "t", "team4545", "2026-11-30T00:00:00Z", "2026-12-07T00:00:00Z", messages, mentions, [Acceptance("m2", "T1")]
    )


def test_a_day_alone_is_ambiguous_when_two_offers_fall_on_it():
    assert check.ambiguous(ambiguity_thread("thu 20:00"))


def test_an_offer_is_on_the_day_its_writer_named():
    assert not check.ambiguous(ambiguity_thread("fri 07:00 AEDT"))
