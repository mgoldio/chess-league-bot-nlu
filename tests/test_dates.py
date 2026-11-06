from datetime import date, timedelta

from chess_league_bot_nlu import dates, records


def test_a_placeholder_writes_its_day_as_far_from_the_week_s_monday():
    template = "{+3:b} {+3:o}, {+10:d}/{+10:m} or {-1:Bl}"
    assert dates.rendered(template, date(2026, 12, 29)).text == "Jan 1st, 8/1 or december"


def test_positions_move_with_the_text_a_placeholder_writes():
    found = dates.rendered("fri {+4:d}/{+4:m} 20:00", date(2026, 11, 23))
    assert found.text == "fri 27/11 20:00"
    assert (found.in_text(23), found.in_template(15)) == (15, 23)


def test_a_written_date_that_names_one_day_of_the_round_becomes_its_placeholder():
    days = [date(2026, 11, 23) + timedelta(days=offset) for offset in range(8)]
    assert dates.templated("Nov 26th or the 3rd", days) == "{2026-11-26:b} {2026-11-26:o} or the 3rd"


def test_a_generated_thread_means_the_same_times_on_any_week(perfect):
    for seed in range(100):
        thread = perfect(seed)[1]
        later = thread.moved(23)
        assert not any("{" in message.text for message in later.messages)
        for before, after in zip(thread.mentions, later.mentions, strict=True):
            moved = records.parse_time(after.times[0].start) - records.parse_time(before.times[0].start)
            assert moved == timedelta(weeks=23)
