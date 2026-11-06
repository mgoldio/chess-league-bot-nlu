from chess_league_bot_nlu import records
from chess_league_bot_nlu.evaluate import commands, threads
from chess_league_bot_nlu.outcome import gold


class Answers:
    """A model with fixed answers: the labels themselves for threads, and given ones for commands."""

    def __init__(self, answers: dict[str, dict] | None = None):
        self.answers = answers or {}

    def read_command(self, text: str) -> dict:
        return self.answers.get(text, {})

    def read_thread(self, thread: records.Thread):
        return gold(thread)


def test_commands_score_intents_events_and_arguments():
    test = [
        records.Command("c1", "ping me when knight42 starts", "tell", [records.Argument("player", 13, 21)], "starts"),
        records.Command("c2", "show the standings", "standings_link"),
    ]
    said = {
        test[0].text: {"commands": [("tell", 0.9)], "event": "starts", "arguments": [("player", 13, 21)]},
        test[1].text: {"commands": [("pairings_link", 0.9)], "event": None, "arguments": []},
    }
    scored = commands(Answers(said), test)
    assert (scored["accuracy"], scored["event_accuracy"], scored["arguments"]["player"]["f1"]) == (0.5, 1.0, 1.0)


def test_a_model_that_answers_what_the_labels_say_scores_perfectly(perfect):
    scored = threads(Answers(), [perfect(seed)[1] for seed in range(40)])
    assert (scored["exact_threads"], scored["exact_messages"]) == (1.0, 1.0)
