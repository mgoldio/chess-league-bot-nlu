from collections import Counter

import pytest

from chess_league_bot_nlu import records, rules
from chess_league_bot_nlu.evaluate import RULES, Counts, calibration_error, commands, scheduling
from chess_league_bot_nlu.labels import INTENTS
from chess_league_bot_nlu.transcript import gold_spans, transcript


class Schema:
    def classification(self, *_, **__):
        return self

    def entities(self, *_):
        return self


class Answers:
    """A model that gives fixed answers per text, so evaluation's scoring can be checked."""

    def __init__(self, answers: dict[str, dict]):
        self.answers = answers

    def schema(self):
        return Schema()

    def extract(self, text: str, schema, **options) -> dict:
        return self.answers.get(text, {})


def command(text: str, intent: str, event: str | None = None, arguments=()) -> records.Command:
    return records.Command.from_dict(
        {"id": text, "text": text, "intent": intent, "event": event, "arguments": list(arguments)}
    )


def said(intent: str, confidence: float = 0.9, **extra) -> dict:
    return {"command": [{"label": INTENTS[intent].label, "confidence": confidence}], **extra}


def test_counts_score_precision_and_recall():
    counts = Counts()
    counts.add({1, 2, 3}, {2, 3, 4, 5})
    assert counts.scores() == {"precision": 0.667, "recall": 0.5, "f1": 0.571}


def test_a_label_nobody_used_has_no_score_rather_than_zero():
    assert Counts().scores() == {"precision": None, "recall": None, "f1": None}


def test_calibration_error_is_zero_when_confidence_matches_accuracy():
    assert calibration_error([(1.0, True), (1.0, True)]) == 0.0
    assert calibration_error([(1.0, False)]) == 1.0


def test_commands_score_intents_events_and_arguments():
    test = [
        command("ping me when knight42 starts", "tell", "starts", [{"label": "player", "start": 13, "end": 21}]),
        command("show the standings", "standings_link"),
        command("nice game", "none"),
    ]
    model = Answers(
        {
            test[0].text: said(
                "tell",
                event={"label": INTENTS["tell"].events["starts"]},
                entities={"player": [{"text": "Knight42 "}]},
            ),
            test[1].text: said("pairings_link"),
            test[2].text: said("none"),
        }
    )
    scored = commands(model, test)
    assert scored["accuracy"] == 0.667
    assert scored["event_accuracy"] == 1.0
    assert scored["arguments"]["player"]["f1"] == 1.0
    assert scored["arguments"]["recipient"]["f1"] is None
    assert scored["intents"]["standings_link"]["recall"] == 0.0
    assert scored["intents"]["pairings_link"]["precision"] == 0.0
    assert scored["top_confusions"] == [["standings_link", "pairings_link", 1]]


def test_a_command_the_model_has_no_answer_for_counts_as_none():
    scored = commands(Answers({}), [command("hello", "none"), command("help", "help")])
    assert scored["accuracy"] == 0.5


@pytest.fixture
def threads(perfect):
    return [perfect(seed)[1] for seed in range(40)]


def gold_answers(threads: list[records.Thread]) -> dict[str, dict]:
    answers = {}
    for thread in threads:
        for upto in range(1, len(thread.messages) + 1):
            script = transcript(thread, upto)
            answers[script.text] = {"entities": gold_spans(thread, script)}
    return answers


def test_a_model_that_marks_what_the_labels_say_scores_perfectly(threads):
    scored = scheduling(Answers(gold_answers(threads)), threads)
    assert scored["agreements"]["expected"] > 0
    assert scored["agreements"]["f1"] == 1.0
    assert scored["effort_accuracy"] == 1.0
    assert all(scores["f1"] in (1.0, None) for scores in scored["spans"].values())


def test_a_model_that_marks_nothing_nudges_every_player_who_made_the_effort(threads):
    scored = scheduling(Answers({}), threads)
    made_effort = Counter()
    for thread in threads:
        offers, accepted = records.thread_facts(thread)
        published = records.parse_time(thread.pairing_published)
        for player in records.PLAYERS:
            made_effort[rules.meets_rule(RULES[thread.league], published, player, offers, accepted)] += 1
    assert scored["agreements"]["announced"] == 0
    assert scored["nudges"]["wrongly_nudged"] == made_effort[True]
    assert scored["nudges"]["correct"] == made_effort[False]
