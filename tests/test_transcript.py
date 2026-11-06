import pytest

from chess_league_bot_nlu import records, rules
from chess_league_bot_nlu.transcript import Replay, day_context, gold_spans, resolve_span, transcript, within_window

RULES = rules.load()


def replay(thread: records.Thread, upto: int, spans=None) -> Replay:
    state = Replay(thread)
    for step in range(1, upto + 1):
        state.step(step, spans(step) if spans else gold_spans(thread, transcript(thread, step)))
    return state


@pytest.mark.parametrize("seed", range(300))
def test_replaying_gold_marks_reproduces_the_planned_outcomes(seed, perfect):
    _, thread = perfect(seed)
    offers, acceptances = records.thread_facts(thread)
    rule = RULES[thread.league]
    published = records.parse_time(thread.pairing_published)
    expected = {(a.sent, a.time) for a in rules.agreements(offers, acceptances)}
    found = {(a.sent, a.time) for a in rules.agreements(*replay(thread, len(thread.messages)).facts())}
    assert found == expected
    window = replay(thread, within_window(thread, rule))
    for player in records.PLAYERS:
        assert rules.meets_rule(rule, published, player, *window.facts()) == rules.meets_rule(
            rule, published, player, offers, acceptances
        )


THREAD = {
    "id": "t1",
    "league": "team4545",
    "pairing_published": "2026-11-02T18:00:00Z",
    "deadline": "2026-11-09T12:00:00Z",
    "messages": [
        {
            "id": "m1",
            "speaker": "white",
            "sent": "2026-11-02T19:00:00Z",
            "text": "I'm in Berlin, CET. tue 20:00 my time?",
        },
        {"id": "m2", "speaker": "black", "sent": "2026-11-02T20:00:00Z", "text": "sure"},
    ],
    "mentions": [
        {
            "id": "T1",
            "message": "m1",
            "start": 20,
            "end": 37,
            "role": "offered",
            "times": [{"start": "2026-11-03T19:00:00Z"}],
        }
    ],
    "acceptances": [{"message": "m2", "offer": "T1"}],
}


def test_my_time_uses_the_zone_the_player_named():
    thread = records.Thread.from_dict(THREAD)
    [found] = resolve_span(thread, thread.messages[0], 20, 37)
    assert found.start == records.parse_time("2026-11-03T19:00:00Z")


def test_an_acceptance_marked_in_the_offerers_own_message_is_ignored():
    thread = records.Thread.from_dict(THREAD)
    offset = transcript(thread, 1).starts["m1"]
    span = {"text": "tue 20:00 my time", "start": offset + 20, "end": offset + 37}
    state = replay(thread, 1, lambda step: {"offered time": [span], "accepted time": [span]})
    assert state.facts()[1] == []


@pytest.mark.parametrize(
    "text, start, end, day",
    [
        ("20:00 works, friday is bad for me", 0, 5, ""),
        ("20:00 thursday?", 0, 5, "thursday"),
        ("how about 20:00 on thursday", 10, 15, "thursday"),
    ],
)
def test_a_bare_clock_only_borrows_a_day_attached_to_it(text, start, end, day):
    thread = records.Thread.from_dict(
        {**THREAD, "messages": [{**THREAD["messages"][0], "text": text}], "mentions": [], "acceptances": []}
    )
    assert day_context(thread, thread.messages[0], start, end) == day


def several_days(acceptance: str, speaker: str = "black"):
    return records.Thread.from_dict(
        {
            **THREAD,
            "messages": [
                {"id": "m1", "speaker": "white", "sent": "2026-11-02T19:00:00Z", "text": "21:00 on fri or sat?"},
                {"id": "m2", "speaker": speaker, "sent": "2026-11-02T20:00:00Z", "text": acceptance},
            ],
            "mentions": [],
            "acceptances": [],
        }
    )


@pytest.mark.parametrize("acceptance, day", [("sat works", 7), ("Fri 21:00 then", 6), ("deal ok", None)])
def test_accepting_a_time_on_several_days_takes_the_day_named(acceptance, day):
    thread = several_days(acceptance)
    offer = {"start": transcript(thread, 1).starts["m1"], "end": transcript(thread, 1).starts["m1"] + 19}
    script = transcript(thread, 2)
    restated = {"start": script.starts["m2"], "end": script.starts["m2"] + len(acceptance.rsplit(" ", 1)[0])}
    marks = {1: {"offered time": [offer]}, 2: {"offered time": [offer], "accepted time": [offer, restated]}}
    offers, acceptances = replay(thread, 2, marks.get).facts()
    assert len(offers) == 2
    assert [agreement.time.day for agreement in rules.agreements(offers, acceptances)] == ([day] if day else [])


def test_taking_back_is_a_withdrawal_from_the_offerer_and_a_retraction_from_the_accepter():
    thread = records.Thread.from_dict(
        {
            **THREAD,
            "messages": [
                *THREAD["messages"],
                {"id": "m3", "speaker": "black", "sent": "2026-11-02T21:00:00Z", "text": "sorry, can't after all"},
                {"id": "m4", "speaker": "white", "sent": "2026-11-02T22:00:00Z", "text": "no worries"},
            ],
        }
    )
    offer = {"start": transcript(thread, 1).starts["m1"] + 20, "end": transcript(thread, 1).starts["m1"] + 37}
    state = replay(
        thread,
        4,
        lambda step: {
            "offered time": [offer],
            "accepted time": [offer] if step >= 2 else [],
            "withdrawn time": [offer] if step >= 3 else [],
        },
    )
    offers, acceptances = state.facts()
    assert offers[0].withdrawn is None
    assert acceptances[0].retracted == records.parse_time("2026-11-02T21:00:00Z")
