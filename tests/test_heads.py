from datetime import date

import pytest
import torch
from transformers import AutoModel, AutoTokenizer, LlamaConfig

from chess_league_bot_nlu import records
from chess_league_bot_nlu.heads import inputs
from chess_league_bot_nlu.heads.inputs import ANSWERS, TIME_TAGS, command_example, gold, spans, thread_example
from chess_league_bot_nlu.heads.model import Reader
from chess_league_bot_nlu.heads.read import HeadsModel
from chess_league_bot_nlu.labels import ARGUMENTS
from chess_league_bot_nlu.outcome import Event, Time

TRUNK = "HuggingFaceTB/SmolLM2-135M"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(TRUNK)


@pytest.fixture
def threads(perfect):
    return [perfect(seed)[1] for seed in range(40)]


def two_offers(reply: str = "sat works") -> records.Thread:
    return records.Thread.from_dict(
        {
            "id": "t1",
            "messages": [
                {"id": "m1", "speaker": "white", "sent": "+0T12:30:00", "text": "thu 19:00 or sat 14:00"},
                {"id": "m2", "speaker": "black", "sent": "+0T13:00:00", "text": reply},
            ],
            "mentions": [
                {
                    "id": "T1",
                    "message": "m1",
                    "start": 0,
                    "end": 9,
                    "role": "offered",
                    "times": [{"start": "+3T19:00:00"}],
                },
                {
                    "id": "T2",
                    "message": "m1",
                    "start": 13,
                    "end": 22,
                    "role": "offered",
                    "times": [{"start": "+5T14:00:00"}],
                },
            ],
            "acceptances": [{"message": "m2", "offer": "T2"}],
        },
        date(2026, 11, 9),
    )


def test_time_tags_decode_back_to_the_labelled_times(threads, tokenizer):
    for thread in threads:
        example = thread_example(tokenizer, thread)
        assert example is not None
        decoded = set()
        for index, (line, tags) in enumerate(zip(example.lines, example.tags, strict=True)):
            for first, last, _ in spans(tags, {1}, {2: 1}):
                decoded.add((thread.messages[index].id, *line.characters(first, last)))
        assert decoded == {(time.message, time.start, time.end) for time in gold(thread)[0]}


def test_each_message_answers_for_every_time_written_so_far(tokenizer):
    example = thread_example(tokenizer, two_offers())
    thursday, saturday = 0, 1
    assert example.answers == [
        (0, thursday, 1, 1, ANSWERS.index("offered")),
        (0, saturday, 1, 1, ANSWERS.index("offered")),
        (1, thursday, 0, 0, ANSWERS.index("none")),
        (1, saturday, 0, 0, ANSWERS.index("accepted")),
    ]


def test_a_command_argument_decodes_back_to_its_characters(tokenizer):
    command = records.Command.from_dict(
        {
            "id": "c1",
            "text": "ping me when knight42 starts",
            "intent": "tell",
            "event": "starts",
            "arguments": [{"label": "player", "start": 13, "end": 21}],
        }
    )
    example = command_example(tokenizer, command)
    begins = {inputs.ARGUMENT_TAGS.index("begins player")}
    inside = {inputs.ARGUMENT_TAGS.index("inside player"): inputs.ARGUMENT_TAGS.index("begins player")}
    [(first, last, _)] = spans(example.tags, begins, inside)
    assert example.line.characters(first, last) == (13, 21)


def test_a_run_begun_inside_a_word_reads_the_whole_word(tokenizer):
    tokens = inputs.line(tokenizer, "black: ", "thursday works!")
    pieces = [index for index, (start, _) in enumerate(tokens.offsets) if start >= 0]
    inside_word = next(index for index in pieces if tokens.offsets[index][0] > 0 and tokens.offsets[index][0] < 8)
    assert tokens.words(inside_word, inside_word) == (0, 8)


def test_a_run_begun_inside_a_username_reads_the_whole_username(tokenizer):
    tokens = inputs.line(tokenizer, "@bot ", "notify me when bob_endgame starts")
    start = "notify me when bob_".__len__()
    inside = next(index for index, (first, _) in enumerate(tokens.offsets) if first >= start and first < start + 7)
    assert tokens.words(inside, inside) == (15, 26)


def test_spans_start_a_run_at_a_stray_inside_tag():
    begins, inside = TIME_TAGS.index("begins"), TIME_TAGS.index("inside")
    assert spans([0, inside, inside, 0, begins, begins], {begins}, {inside: begins}) == [
        (1, 2, begins),
        (4, 4, begins),
        (5, 5, begins),
    ]


@pytest.fixture(scope="module")
def untrained(tokenizer):
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    return HeadsModel(Reader(AutoModel.from_config(config)).eval(), tokenizer, "cpu")


def test_a_message_is_read_the_same_whatever_comes_after_it(untrained):
    def read(thread: records.Thread):
        lines = inputs.thread_lines(untrained.tokenizer, thread)
        batch = inputs.thread_batch([inputs.ThreadExample(thread, lines, [[0] * len(e.ids) for e in lines])], 0)
        reader = untrained.reader
        with torch.inference_mode():
            states = reader.states(batch.ids, batch.mask)
            return reader.tag_scores(reader.time_tags, states, batch.ahead)[0], reader.at(states, batch.ends)

    short, long = two_offers("sat works"), two_offers("sat works")
    long.messages.append(records.Message("m3", "white", "2026-11-09T13:30:00Z", "great, actually make it sun 15:00"))
    (short_tags, short_ends), (long_tags, long_ends) = read(short), read(long)
    assert torch.allclose(short_tags, long_tags[: len(short_tags)], atol=1e-5)
    assert torch.allclose(short_ends, long_ends[:2], atol=1e-5)


def test_an_untrained_reader_answers_in_the_shape_evaluation_takes(untrained):
    times, events = untrained.read_thread(two_offers())
    assert all(isinstance(time, Time) for time in times) and all(isinstance(event, Event) for event in events)
    answer = untrained.read_command("ping me when knight42 starts")
    assert len(answer["commands"]) == len(inputs.INTENT_KEYS)
    assert all(label in ARGUMENTS for label, _, _ in answer["arguments"])


def test_swapping_players_keeps_every_answer(threads, tokenizer):
    from chess_league_bot_nlu.heads.train import swapped

    for thread in threads[:10]:
        original, swap = thread_example(tokenizer, thread), thread_example(tokenizer, swapped(thread))
        assert original.answers == swap.answers and original.times == swap.times
        assert {m.speaker for m in swapped(thread).messages} <= {"white", "black"}
        assert [m.speaker for m in swapped(swapped(thread)).messages] == [m.speaker for m in thread.messages]


def test_a_message_is_read_in_ascii_and_its_spans_stay_in_its_own_text(tokenizer):
    text = "👍 sun 14:00–18:00 can’t wait"
    read = inputs.line(tokenizer, "white: ", text)
    assert read.text == ":thumbsup: sun 14:00-18:00 can't wait"
    start, end = text.index("sun"), text.index(" can")
    tokens = read.tokens_of(start, end)
    assert tokens is not None
    assert read.words(tokens[0], tokens[-1]) == (start, end)
