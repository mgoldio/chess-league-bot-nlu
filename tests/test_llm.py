import pytest

from chess_league_bot_nlu.generate import llm


@pytest.mark.parametrize(
    "reply, value",
    [
        ('["a", "b"]', ["a", "b"]),
        ('Sure! ```json\n["a"]\n``` hope that helps', ["a"]),
        ('{"accepted": []} and some chatter', {"accepted": []}),
    ],
)
def test_json_is_found_inside_chatter(reply, value):
    assert llm.json_from(reply) == value


def test_a_reply_without_json_is_a_bad_reply():
    with pytest.raises(llm.BadReply):
        llm.json_from("I can't help with that")
