from datetime import UTC, datetime, timedelta

from chess_league_bot_nlu import rules

RULES = rules.load()
TEAM = RULES["team4545"]
LONEWOLF = RULES["lonewolf"]
PUBLISHED = datetime(2026, 11, 2, 18, tzinfo=UTC)


def at(day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 11, day, hour, minute, tzinfo=UTC)


def offer(id, start, *, player="white", sent=None, end=None, open=False, withdrawn=None):
    return rules.Offer(id, player, sent or PUBLISHED + timedelta(hours=1), start, end, open, withdrawn)


def meets(rule, offers, acceptances=(), player="white"):
    return rules.meets_rule(rule, PUBLISHED, player, list(offers), list(acceptances))


def test_three_times_over_two_days_meets_the_team_rule():
    assert meets(TEAM, [offer("a", at(4, 18)), offer("b", at(4, 21)), offer("c", at(5, 19))])


def test_three_times_on_one_day_do_not():
    assert not meets(TEAM, [offer("a", at(4, 12)), offer("b", at(4, 15)), offer("c", at(4, 19))])


def test_times_closer_than_two_hours_count_once():
    assert not meets(TEAM, [offer("a", at(4, 18)), offer("b", at(4, 19, 30)), offer("c", at(5, 19))])


def test_any_qualifying_set_counts_not_just_a_greedy_one():
    times = [at(4, 18), at(4, 20), at(4, 22, 30), at(5, 0)]
    assert meets(TEAM, [offer(str(index), time) for index, time in enumerate(times)])


def test_a_range_of_two_hours_counts_as_two_times():
    assert meets(TEAM, [offer("a", at(4, 18), end=at(4, 20)), offer("b", at(5, 19))])


def test_a_shorter_range_or_an_open_one_counts_once():
    assert not meets(TEAM, [offer("a", at(4, 18), end=at(4, 19, 30)), offer("b", at(5, 19))])
    assert not meets(TEAM, [offer("a", at(4, 18), open=True), offer("b", at(5, 19))])


def test_offers_after_the_window_do_not_count():
    late = PUBLISHED + timedelta(hours=25)
    assert not meets(TEAM, [offer("a", at(4, 18)), offer("b", at(4, 21)), offer("c", at(5, 19), sent=late)])


def test_offers_withdrawn_within_the_window_do_not_count():
    withdrawn = PUBLISHED + timedelta(hours=5)
    assert not meets(TEAM, [offer("a", at(4, 18)), offer("b", at(4, 21)), offer("c", at(5, 19), withdrawn=withdrawn)])


def test_offers_withdrawn_after_the_window_still_counted_at_the_cutoff():
    withdrawn = PUBLISHED + timedelta(hours=30)
    assert meets(TEAM, [offer("a", at(4, 18)), offer("b", at(4, 21)), offer("c", at(5, 19), withdrawn=withdrawn)])


def test_accepting_an_opponent_offer_within_the_window_meets_the_rule():
    offers = [offer("a", at(4, 18), player="black")]
    assert meets(TEAM, offers, [rules.Accepted("white", PUBLISHED + timedelta(hours=3), "a")])


def test_late_acceptances_own_offers_and_withdrawn_offers_do_not_count():
    late = rules.Accepted("white", PUBLISHED + timedelta(hours=25), "a")
    assert not meets(TEAM, [offer("a", at(4, 18), player="black")], [late])
    own = rules.Accepted("white", PUBLISHED + timedelta(hours=3), "a")
    assert not meets(TEAM, [offer("a", at(4, 18), player="white")], [own])
    gone = offer("a", at(4, 18), player="black", withdrawn=PUBLISHED + timedelta(hours=2))
    assert not meets(TEAM, [gone], [rules.Accepted("white", PUBLISHED + timedelta(hours=3), "a")])


def test_lonewolf_needs_offers_on_two_days():
    assert meets(LONEWOLF, [offer("a", at(4, 18)), offer("b", at(5, 18, 30))])
    assert not meets(LONEWOLF, [offer("a", at(4, 18)), offer("b", at(4, 21))])


def test_agreements_take_the_offer_time_or_the_pick_inside_a_range():
    offers = [offer("a", at(4, 18), player="black"), offer("b", at(5, 18), end=at(5, 22), player="black")]
    accepted = rules.agreements(
        offers,
        [
            rules.Accepted("white", at(3, 9), "a"),
            rules.Accepted("white", at(3, 10), "b", at(5, 20)),
        ],
    )
    assert [agreement.time for agreement in accepted] == [at(4, 18), at(5, 20)]


def test_a_range_accepted_without_a_pick_or_with_one_outside_it_agrees_nothing():
    offers = [offer("b", at(5, 18), end=at(5, 22), player="black")]
    assert rules.agreements(offers, [rules.Accepted("white", at(3, 10), "b")]) == []
    assert rules.agreements(offers, [rules.Accepted("white", at(3, 10), "b", at(5, 23))]) == []


def test_an_acceptance_taken_back_by_the_cutoff_does_not_meet_the_rule():
    offers = [offer("a", at(4, 18), player="black")]
    early = PUBLISHED + timedelta(hours=3)
    assert not meets(TEAM, offers, [rules.Accepted("white", early, "a", retracted=PUBLISHED + timedelta(hours=5))])
    assert meets(TEAM, offers, [rules.Accepted("white", early, "a", retracted=PUBLISHED + timedelta(hours=30))])


def test_an_acceptance_taken_back_agrees_nothing():
    offers = [offer("a", at(4, 18), player="black")]
    assert rules.agreements(offers, [rules.Accepted("white", at(3, 9), "a", retracted=at(3, 12))]) == []


def test_an_acceptance_of_one_of_several_days_takes_the_day_its_time_falls_on():
    days = [offer("a/0", at(5, 21), player="black"), offer("a/1", at(6, 21), end=at(6, 23), player="black")]
    assert rules.chosen(days, at(6, 22)).id == "a/1"
    assert rules.chosen(days, at(5, 22)) is None
    assert rules.chosen(days[:1], None).id == "a/0"
