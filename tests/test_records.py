import copy

import pytest

from chess_league_bot_nlu import records, rules

THREAD = {
    "id": "t1",
    "league": "team4545",
    "pairing_published": "2026-11-02T18:00:00Z",
    "deadline": "2026-11-09T12:00:00Z",
    "messages": [
        {"id": "m1", "speaker": "white", "sent": "2026-11-02T19:00:00Z", "text": "wed 18:00-20:00 or thu 19?"},
        {"id": "m2", "speaker": "white", "sent": "2026-11-02T19:05:00Z", "text": "actually thu 19 is out"},
        {"id": "m3", "speaker": "black", "sent": "2026-11-02T20:00:00Z", "text": "wed 19 works"},
    ],
    "mentions": [
        {
            "id": "t1",
            "message": "m1",
            "start": 0,
            "end": 15,
            "role": "offered",
            "times": [{"start": "2026-11-04T18:00:00Z", "end": "2026-11-04T20:00:00Z"}],
        },
        {
            "id": "t2",
            "message": "m1",
            "start": 19,
            "end": 25,
            "role": "offered",
            "times": [{"start": "2026-11-05T19:00:00Z"}],
        },
        {
            "id": "t3",
            "message": "m2",
            "start": 9,
            "end": 15,
            "role": "other",
            "times": [{"start": "2026-11-05T19:00:00Z"}],
        },
        {
            "id": "t4",
            "message": "m3",
            "start": 0,
            "end": 6,
            "role": "accepted",
            "accepts": "t1",
            "times": [{"start": "2026-11-04T19:00:00Z"}],
        },
    ],
    "acceptances": [{"message": "m3", "offer": "t1", "time": "2026-11-04T19:00:00Z"}],
    "withdrawals": [{"message": "m2", "offer": "t2"}],
}


def test_a_valid_thread_round_trips():
    thread = records.Thread.from_dict(THREAD)
    assert records.Thread.from_dict(thread.to_dict()) == thread
    assert [thread.text_of(mention) for mention in thread.mentions] == ["wed 18:00-20:00", "thu 19", "thu 19", "wed 19"]


@pytest.mark.parametrize(
    "change",
    [
        lambda data: data["mentions"][0].update(end=99),
        lambda data: data["mentions"][3].update(accepts=None),
        lambda data: data["mentions"][1].update(accepts="t1"),
        lambda data: data["acceptances"][0].update(offer="t3"),
        lambda data: data["messages"][0].update(speaker="captain"),
        lambda data: data["messages"].reverse(),
    ],
)
def test_inconsistent_threads_are_rejected(change):
    data = copy.deepcopy(THREAD)
    change(data)
    with pytest.raises(records.InvalidRecord):
        records.Thread.from_dict(data)


def test_facts_carry_withdrawals_and_picks_into_the_rules():
    thread = records.Thread.from_dict(THREAD)
    offers, acceptances = records.thread_facts(thread)
    rule = rules.load()["team4545"]
    published = records.parse_time(thread.pairing_published)
    assert [offer.withdrawn is not None for offer in offers] == [False, True]
    assert not rules.meets_rule(rule, published, "white", offers, acceptances)
    assert rules.meets_rule(rule, published, "black", offers, acceptances)
    assert [agreement.time.hour for agreement in rules.agreements(offers, acceptances)] == [19]


@pytest.mark.parametrize(
    "command",
    [
        {"id": "c2", "text": "rating knight42", "intent": "rating", "event": "starts"},
        {"id": "c3", "text": "ping me", "intent": "tell"},
        {"id": "c4", "text": "ping me", "intent": "tell", "event": "whenever"},
        {"id": "c5", "text": "hi", "intent": "chat"},
    ],
)
def test_commands_need_a_known_intent_and_an_event_exactly_when_it_has_events(command):
    records.Command.from_dict({"id": "c1", "text": "ping me when it starts", "intent": "tell", "event": "starts"})
    with pytest.raises(records.InvalidRecord):
        records.Command.from_dict(command)


def test_a_schema_1_record_reads_as_schema_2():
    old = copy.deepcopy(THREAD)
    for mention in old["mentions"]:
        mention["utc"] = mention.pop("times")[0]
    thread = records.Thread.from_dict({**old, "schema": 1})
    assert thread == records.Thread.from_dict(THREAD)


SEVERAL_DAYS = {
    **THREAD,
    "messages": [
        {"id": "m1", "speaker": "white", "sent": "2026-11-02T19:00:00Z", "text": "21:00 on fri or sat"},
        {"id": "m2", "speaker": "black", "sent": "2026-11-02T20:00:00Z", "text": "sat works"},
    ],
    "mentions": [
        {
            "id": "t1",
            "message": "m1",
            "start": 0,
            "end": 19,
            "role": "offered",
            "times": [{"start": "2026-11-06T21:00:00Z"}, {"start": "2026-11-07T21:00:00Z"}],
        },
        {
            "id": "t2",
            "message": "m2",
            "start": 0,
            "end": 3,
            "role": "accepted",
            "accepts": "t1",
            "times": [{"start": "2026-11-07T21:00:00Z"}],
        },
    ],
    "acceptances": [{"message": "m2", "offer": "t1", "time": "2026-11-07T21:00:00Z"}],
    "withdrawals": [],
}


def test_a_time_on_several_days_is_an_offer_per_day_and_an_acceptance_picks_one():
    offers, acceptances = records.thread_facts(records.Thread.from_dict(SEVERAL_DAYS))
    assert [offer.id for offer in offers] == ["t1/0", "t1/1"]
    assert acceptances[0].offer == "t1/1"
    assert [agreement.time.day for agreement in rules.agreements(offers, acceptances)] == [7]


@pytest.mark.parametrize("time", [None, "2026-11-08T21:00:00Z"])
def test_accepting_a_time_on_several_days_must_pick_one_of_them(time):
    data = copy.deepcopy(SEVERAL_DAYS)
    data["acceptances"][0]["time"] = time
    with pytest.raises(records.InvalidRecord):
        records.Thread.from_dict(data)


def test_a_retracted_acceptance_agrees_nothing_and_leaves_the_offer_standing():
    data = copy.deepcopy(SEVERAL_DAYS)
    data["messages"].append({"id": "m3", "speaker": "black", "sent": "2026-11-02T22:00:00Z", "text": "sorry, can't"})
    data["retractions"] = [{"message": "m3", "offer": "t1"}]
    offers, acceptances = records.thread_facts(records.Thread.from_dict(data))
    assert acceptances[0].retracted is not None
    assert rules.agreements(offers, acceptances) == []
    assert all(offer.withdrawn is None for offer in offers)


def test_only_the_player_who_accepted_can_take_it_back():
    data = copy.deepcopy(SEVERAL_DAYS)
    data["messages"].append({"id": "m3", "speaker": "white", "sent": "2026-11-02T22:00:00Z", "text": "hmm"})
    data["retractions"] = [{"message": "m3", "offer": "t1"}]
    with pytest.raises(records.InvalidRecord):
        records.Thread.from_dict(data)
