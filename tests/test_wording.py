import random

import pytest

from chess_league_bot_nlu.generate.wording import readable, written
from chess_league_bot_nlu.resolve import resolve


@pytest.mark.parametrize("seed", range(300))
def test_written_times_read_back_as_the_plan(seed, perfect):
    thread_plan, _ = perfect(seed)
    rng = random.Random(seed)
    window = (thread_plan.published, thread_plan.deadline)
    for step in thread_plan.steps:
        for time in step.times:
            if time.form:
                continue
            text = written(time, step.sent, window, rng)
            found = resolve(text, step.sent, window)
            assert [(each.start, each.end) for each in found] == time.slots(), text


@pytest.mark.parametrize(
    "text, day, expected",
    [
        ("Wed 16 18", "Wed 16", False),
        ("15 Fri 20/11 EST", "Fri 20/11", False),
        ("Wed 16 18:00", "Wed 16", True),
        ("thu 19", "thu", True),
    ],
)
def test_day_numbers_never_run_into_bare_hours(text, day, expected):
    assert readable(text, day) == expected
