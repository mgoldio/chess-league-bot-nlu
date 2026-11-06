import pytest

from chess_league_bot_nlu.generate.verify import Meaning, claims, disputed, same_time


@pytest.mark.parametrize(
    "quoted, written, expected",
    [
        ("thu 19:00", ["thu 19:00"], True),
        ("Thu 19:00 UTC", ["thu 19:00"], True),
        ("the thu 19:00 one", ["thu 19:00"], True),
        ("19:00", ["Thu 19:00"], False),
        ("1", ["Fri 1pm"], False),
        ("wed 20:00", ["thu 19:00", "thu"], False),
        ("thu 13:00", ["thu 1"], False),
        ("Thu 19h30", ["Thu 19h"], False),
        ("thu 19:00 and fri 20:00", ["thu 19:00"], False),
        ("Thu 21:00", ["thu"], False),
        ("Thu 21:00", ["Thu 21:00", "thu"], True),
    ],
)
def test_the_reader_must_quote_the_whole_time(quoted, written, expected):
    assert same_time(quoted, written) == expected


def test_a_quote_that_reads_as_the_planned_time_counts():
    from datetime import UTC, datetime

    from chess_league_bot_nlu.generate.verify import Meaning

    sent = datetime(2026, 11, 2, 19, tzinfo=UTC)
    window = (datetime(2026, 11, 2, 18, tzinfo=UTC), datetime(2026, 11, 9, 12, tzinfo=UTC))
    meaning = Meaning(datetime(2026, 11, 3, 21, tzinfo=UTC), sent, window)
    assert same_time("21:00", ["21:00 tmrw"], meaning)
    assert same_time("21h00", ["9:00pm on tmrw"], meaning)
    assert not same_time("13:00", ["21:00 tmrw"], meaning)


def test_qa_is_asked_about_every_offer(perfect):
    for seed in range(40):
        _, thread = perfect(seed)
        offers = [mention for mention in thread.mentions if mention.role == "offered"]
        assert all(f'"{thread.text_of(offer)}"' in claim for offer, claim in zip(offers, claims(thread), strict=True))


def test_qa_disputes_false_and_unanswered_claims():
    listed = ['message 1: white suggests "thu 19:00"', 'message 2: black suggests "fri"', "message 3: x"]
    reply = [{"claim": 1, "why": "ok", "true": True}, {"claim": 2, "why": "turned down", "true": False}]
    assert disputed(listed, reply) == [
        'QA says false: message 2: black suggests "fri" (turned down)',
        "QA didn't answer: message 3: x",
    ]


def test_a_reply_that_is_not_a_list_disputes_everything():
    assert len(disputed(["a", "b"], {"claim": 1, "true": True})) == 2


def test_a_quote_naming_only_the_planned_day_matches_an_acceptance_by_day():
    from datetime import UTC, datetime

    meaning = Meaning(
        datetime(2026, 11, 18, 21, tzinfo=UTC),
        datetime(2026, 11, 16, 9, tzinfo=UTC),
        (datetime(2026, 11, 16, 0, tzinfo=UTC), datetime(2026, 11, 23, 12, tzinfo=UTC)),
    )
    assert same_time("Wednesday evening", ["18 Nov at 21h00 Utc"], meaning, only_on_its_day=True)
    assert not same_time("Wednesday evening", ["18 Nov at 21h00 Utc"], meaning)
    assert not same_time("Thursday evening", ["18 Nov at 21h00 Utc"], meaning, only_on_its_day=True)
