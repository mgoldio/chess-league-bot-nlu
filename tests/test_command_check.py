import json

from chess_league_bot_nlu import records
from chess_league_bot_nlu.generate import verify


def command(text: str, intent: str, arguments=()) -> records.Command:
    return records.Command.from_dict({"id": "c1-0", "text": text, "intent": intent, "arguments": list(arguments)})


def reader_says(monkeypatch, answer: dict) -> None:
    monkeypatch.setattr(verify.llm, "chat", lambda *_, **__: json.dumps(answer))


def test_another_reading_from_the_same_look_alike_group_rejects_the_message(monkeypatch):
    reader_says(monkeypatch, {"intent": "mods", "event": None, "also": ["summon_mods"]})
    assert verify.command_reading(command("mods", "mods"))


def test_another_reading_outside_the_group_keeps_the_message(monkeypatch):
    reader_says(monkeypatch, {"intent": "rating", "event": None, "also": ["none"]})
    assert not verify.command_reading(command("my rating?", "rating"))


def test_a_reader_that_swaps_two_arguments_rejects_the_message(monkeypatch):
    text = "pairing tallover849 round 4"
    planned = command(
        text,
        "pairing",
        [{"label": "player", "start": 8, "end": 19}, {"label": "round", "start": 20, "end": 27}],
    )
    reader_says(
        monkeypatch,
        {"intent": "pairing", "also": [], "arguments": {"player": "round 4", "round": "tallover849"}},
    )
    assert verify.command_reading(planned)
    reader_says(
        monkeypatch,
        {"intent": "pairing", "also": [], "arguments": {"player": "tallover849", "round": "round 4"}},
    )
    assert not verify.command_reading(planned)
