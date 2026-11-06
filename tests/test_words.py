from chess_league_bot_nlu import words


def test_spans_off_the_models_word_boundaries_are_found():
    text = "white: thu 19:00-fri 20:00 either works\n"
    assert words.misaligned(text, [(7, 16)]) == [(7, 16)]
    assert not words.misaligned("white: thu 19:00 or fri\n", [(7, 16)])
