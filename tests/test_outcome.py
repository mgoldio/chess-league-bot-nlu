from chess_league_bot_nlu.outcome import Event, Time, gold, outcome
from chess_league_bot_nlu.records import Message

MESSAGES = [
    Message("m1", "white", "2026-11-09T12:00:00Z", "thu 19:00 or sat 14:00"),
    Message("m2", "black", "2026-11-09T12:10:00Z", "sat 14:00 works"),
    Message("m3", "white", "2026-11-09T12:20:00Z", "sorry, thu is out"),
    Message("m4", "black", "2026-11-09T12:30:00Z", "actually i can't do sat"),
]
THU, SAT = Time("m1", 0, 9, "offered"), Time("m1", 13, 22, "offered")
RESTATED = Time("m2", 0, 9, "accepted")


def test_an_offer_stands_until_its_player_takes_it_back():
    events = [Event("m3", THU, "take back")]
    assert outcome(MESSAGES, [THU, SAT], events, 2).standing == {THU, SAT}
    after = outcome(MESSAGES, [THU, SAT], events, 3)
    assert (after.standing, after.withdrawn) == ({SAT}, {THU})


def test_an_acceptance_keeps_the_time_it_restated():
    found = outcome(MESSAGES, [THU, SAT, RESTATED], [Event("m2", SAT, "accept")], 2)
    assert found.accepted == {SAT: RESTATED}


def test_only_the_other_player_accepts_and_only_the_offerer_withdraws():
    events = [Event("m3", SAT, "accept"), Event("m2", THU, "take back")]
    found = outcome(MESSAGES, [THU, SAT], events, 4)
    assert (found.standing, found.accepted) == ({THU, SAT}, {})


def test_the_accepter_taking_it_back_leaves_the_offer_standing():
    events = [Event("m2", SAT, "accept"), Event("m4", SAT, "take back")]
    found = outcome(MESSAGES, [THU, SAT], events, 4)
    assert (found.standing, found.accepted) == ({THU, SAT}, {})


def test_withdrawing_an_accepted_offer_ends_the_acceptance_too():
    events = [Event("m2", SAT, "accept"), Event("m3", SAT, "take back")]
    found = outcome(MESSAGES, [THU, SAT], events, 3)
    assert (found.standing, found.withdrawn, found.accepted) == ({THU}, {SAT}, {})


def test_a_generated_thread_comes_to_what_its_plot_planned(perfect):
    for seed in range(100):
        thread = perfect(seed)[1]
        found = outcome(thread.messages, *gold(thread), len(thread.messages))
        offered = {mention.id for mention in thread.mentions if mention.role == "offered"}
        withdrawn = {each.offer for each in thread.withdrawals}
        places = {(m.message, m.start, m.end): m.id for m in thread.mentions}
        assert {places[(t.message, t.start, t.end)] for t in found.standing} == offered - withdrawn
