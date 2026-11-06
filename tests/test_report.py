import pytest

from chess_league_bot_nlu.generate.report import marks, reason_kind


@pytest.mark.parametrize(
    "reason, kind",
    [
        ("reader found accepted in messages [], plan has [4]", "reader found accepted in messages […], plan has […]"),
        (
            "reader says message 3 accepted '4pm', plan says 'thu 4pm'; reader found withdrawn in messages [2]",
            "reader says message N accepted …, plan says …",
        ),
        (
            "message 4: unplanned placeholders ['{T4}']: \"can't do {T4} either\"",
            "message N: unplanned placeholders […]",
        ),
        ("BadReply: no JSON in the reply", "BadReply: no JSON in the reply"),
    ],
)
def test_like_reject_reasons_count_together(reason, kind):
    assert reason_kind(reason) == kind


def test_the_review_shows_acceptances_that_have_no_words_of_their_own(perfect):
    for seed in range(200):
        _, thread = perfect(seed)
        if thread.acceptances:
            break
    acceptance = thread.acceptances[0]
    offer = next(mention for mention in thread.mentions if mention.id == acceptance.offer)
    assert f"accepts {thread.text_of(offer)!r}" in "\n".join(marks(thread))
