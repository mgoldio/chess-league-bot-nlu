import copy
from datetime import date

import pytest

from chess_league_bot_nlu import records

THREAD = {
    "id": "t1",
    "messages": [
        {"id": "m1", "speaker": "white", "sent": "+0T19:00:00", "text": "wed {+2:d}/{+2:m} 18:00-20:00 or thu 19?"},
        {"id": "m2", "speaker": "white", "sent": "+0T19:05:00", "text": "actually thu 19 is out"},
        {"id": "m3", "speaker": "black", "sent": "+0T20:00:00", "text": "wed 19 works"},
    ],
    "mentions": [
        {
            "id": "t1",
            "message": "m1",
            "start": 0,
            "end": 29,
            "role": "offered",
            "times": [{"start": "+2T18:00:00", "end": "+2T20:00:00"}],
        },
        {"id": "t2", "message": "m1", "start": 33, "end": 39, "role": "offered", "times": [{"start": "+3T19:00:00"}]},
        {"id": "t3", "message": "m2", "start": 9, "end": 15, "role": "other", "times": [{"start": "+3T19:00:00"}]},
        {
            "id": "t4",
            "message": "m3",
            "start": 0,
            "end": 6,
            "role": "accepted",
            "accepts": "t1",
            "times": [{"start": "+2T19:00:00"}],
        },
    ],
    "acceptances": [{"message": "m3", "offer": "t1", "time": "+2T19:00:00"}],
    "withdrawals": [{"message": "m2", "offer": "t2"}],
    "plot": "pick_from_range",
}


def test_a_thread_is_put_on_the_week_asked_for():
    thread = records.Thread.from_dict(THREAD, date(2026, 11, 2))
    assert thread.messages[0].text == "wed 4/11 18:00-20:00 or thu 19?"
    assert [thread.text_of(mention) for mention in thread.mentions] == [
        "wed 4/11 18:00-20:00",
        "thu 19",
        "thu 19",
        "wed 19",
    ]
    assert thread.mentions[0].times[0].start == "2026-11-04T18:00:00Z"
    assert thread.acceptances[0].time == "2026-11-04T19:00:00Z"


def test_a_stored_thread_comes_back_the_same_on_any_week():
    thread = records.Thread.from_dict(THREAD, date(2026, 11, 2))
    assert thread.to_dict()["mentions"][0] == {
        **THREAD["mentions"][0],
        "accepts": None,
        "times": [{"start": "+2T18:00:00", "end": "+2T20:00:00", "open": False}],
    }
    later = thread.moved(10)
    assert later.messages[0].text == "wed 13/1 18:00-20:00 or thu 19?"
    assert later.on(date(2026, 11, 2)) == thread


def test_a_thread_loads_on_a_week_of_its_own_every_time():
    assert records.Thread.from_dict(THREAD).week == records.Thread.from_dict(THREAD).week
    assert records.Thread.from_dict(THREAD).week.weekday() == 0


def test_a_week_starts_on_a_monday():
    with pytest.raises(ValueError):
        records.Thread.from_dict(THREAD, date(2026, 11, 3))


@pytest.mark.parametrize(
    "change",
    [
        lambda data: data["mentions"][0].update(end=99),
        lambda data: data["mentions"][0].update(end=8),
        lambda data: data["mentions"][3].update(accepts=None),
        lambda data: data["mentions"][1].update(accepts="t1"),
        lambda data: data["acceptances"][0].update(offer="t3"),
        lambda data: data["acceptances"][0].update(time="+2T21:00:00"),
        lambda data: data["messages"][0].update(speaker="captain"),
        lambda data: data["messages"].reverse(),
    ],
)
def test_inconsistent_threads_are_rejected(change):
    data = copy.deepcopy(THREAD)
    change(data)
    with pytest.raises(records.InvalidRecord):
        records.Thread.from_dict(data)


@pytest.mark.parametrize(
    "command",
    [
        {"id": "c2", "text": "rating knight42", "intent": "rating", "event": "starts"},
        {"id": "c4", "text": "ping me", "intent": "tell", "event": "whenever"},
        {"id": "c5", "text": "hi", "intent": "chat"},
    ],
)
def test_commands_need_a_known_intent_and_an_event_only_one_it_has(command):
    records.Command.from_dict({"id": "c1", "text": "ping me when it starts", "intent": "tell", "event": "starts"})
    records.Command.from_dict({"id": "c3", "text": "follow my team", "intent": "tell"})
    with pytest.raises(records.InvalidRecord):
        records.Command.from_dict(command)


def test_a_stored_command_leaves_out_what_it_lacks():
    command = records.Command.from_dict({"id": "c1", "text": "faq", "intent": "faq", "style": "terse"})
    assert command.to_dict() == {"id": "c1", "text": "faq", "intent": "faq", "arguments": [], "style": "terse"}


SEVERAL_DAYS = {
    "id": "t2",
    "messages": [
        {"id": "m1", "speaker": "white", "sent": "+0T19:00:00", "text": "21:00 on fri or sat"},
        {"id": "m2", "speaker": "black", "sent": "+0T20:00:00", "text": "sat works"},
    ],
    "mentions": [
        {
            "id": "t1",
            "message": "m1",
            "start": 0,
            "end": 19,
            "role": "offered",
            "times": [{"start": "+4T21:00:00"}, {"start": "+5T21:00:00"}],
        },
        {
            "id": "t2",
            "message": "m2",
            "start": 0,
            "end": 3,
            "role": "accepted",
            "accepts": "t1",
            "times": [{"start": "+5T21:00:00"}],
        },
    ],
    "acceptances": [{"message": "m2", "offer": "t1", "time": "+5T21:00:00"}],
}


@pytest.mark.parametrize("time", [None, "+6T21:00:00"])
def test_accepting_a_time_on_several_days_must_pick_one_of_them(time):
    records.Thread.from_dict(SEVERAL_DAYS)
    data = copy.deepcopy(SEVERAL_DAYS)
    data["acceptances"][0]["time"] = time
    with pytest.raises(records.InvalidRecord):
        records.Thread.from_dict(data)


def test_only_the_player_who_accepted_can_take_it_back():
    data = copy.deepcopy(SEVERAL_DAYS)
    data["messages"].append({"id": "m3", "speaker": "white", "sent": "+0T22:00:00", "text": "hmm"})
    data["retractions"] = [{"message": "m3", "offer": "t1"}]
    with pytest.raises(records.InvalidRecord):
        records.Thread.from_dict(data)
