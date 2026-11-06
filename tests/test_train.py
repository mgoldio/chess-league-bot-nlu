from chess_league_bot_nlu import records
from chess_league_bot_nlu.train import command_row, misaligned, rows, thread_rows
from chess_league_bot_nlu.transcript import transcript


def test_a_command_row_has_its_command_event_and_argument_offsets():
    command = records.Command.from_dict(
        {
            "id": "c1",
            "text": "ping me when knight42 starts",
            "intent": "tell",
            "event": "starts",
            "arguments": [{"label": "player", "start": 13, "end": 21}],
        }
    )
    row = command_row(command)
    assert row["output"]["entities"]["player"] == [{"text": "knight42", "start": 13, "end": 21}]
    assert [c["true_label"] for c in row["output"]["classifications"]] == [["notify about games"], ["game starts"]]


def test_a_row_with_a_span_off_word_boundaries_is_misaligned():
    row = {
        "input": "white: thu 19:00-fri 20:00\n",
        "output": {"entities": {"offered time": [{"text": "thu 19:00", "start": 7, "end": 16}]}},
    }
    assert misaligned(row)


def test_a_thread_trains_on_every_prefix_with_only_the_marks_made_so_far(perfect):
    _, thread = perfect(5)
    built = thread_rows(thread)
    assert len(built) == len(thread.messages)
    for upto, row in enumerate(built, start=1):
        assert row["input"] == transcript(thread, upto).text
        for spans in row["output"]["entities"].values():
            assert all(row["input"][span["start"] : span["end"]] == span["text"] for span in spans)


def test_rows_leave_out_whole_rows_with_a_misaligned_span():
    good = records.Command.from_dict({"id": "c1", "text": "rating knight42", "intent": "rating", "arguments": []})
    bad = records.Command.from_dict(
        {
            "id": "c2",
            "text": "rating knight42",
            "intent": "rating",
            "arguments": [{"label": "player", "start": 7, "end": 12}],
        }
    )
    assert [row["input"] for row in rows([good, bad], [])] == ["rating knight42"]
