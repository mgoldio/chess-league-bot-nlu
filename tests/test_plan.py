import random
from datetime import datetime

from chess_league_bot_nlu.generate import plan


def test_plans_keep_their_invariants():
    for seed in range(2000):
        plan.check_plan(plan.thread(random.Random(seed)))


def test_openers_and_every_plot_occur():
    plans = [plan.thread(random.Random(seed)) for seed in range(600)]
    assert any(step.kind == "opener" for p in plans for step in p.steps)
    assert {p.plot for p in plans} == {plot.name for plot in plan.catalog.plots().plots}


def test_ordinal_references_count_the_times_in_the_order_written():
    ordinals = plan.catalog.beats()["ordinals"]
    for seed in range(3000):
        thread_plan = plan.thread(random.Random(seed))
        for step in thread_plan.steps:
            if step.reference_kind == "ordinal":
                offer = thread_plan.offer(step.accepts or step.withdraws)
                offer_step = next(other for other in thread_plan.steps if offer in other.times)
                assert step.reference == plan.catalog.beats()["reference_ordinal"].format(
                    ordinal=ordinals[[time for time in offer_step.times if time.role == "offered"].index(offer)]
                )


def test_struck_times_come_right_before_the_time_that_replaced_them():
    for seed in range(3000):
        for step in plan.thread(random.Random(seed)).steps:
            for index, time in enumerate(step.times):
                if time.struck_for:
                    assert step.times[index + 1].id == time.struck_for and time.role == "other"


def test_questions_near_an_offer_never_repeat_one_of_its_times():
    for seed in range(3000):
        thread_plan = plan.thread(random.Random(seed))
        if thread_plan.plot != "question_counter":
            continue
        first, asked = (step for step in thread_plan.steps if step.kind == "offer")
        offered = {start for time in first.times for start, _ in time.slots()}
        assert all(time.start not in offered for time in asked.times if time.role == "offered")


def test_a_correction_withdraws_the_slip_and_offers_another_time():
    for seed in range(3000):
        for step in plan.thread(random.Random(seed)).steps:
            if step.kind == "correct":
                [moved] = step.times
                assert step.withdraws and moved.role == "offered" and moved.id != step.withdraws


def test_a_retraction_follows_the_same_players_acceptance():
    found = 0
    for seed in range(3000):
        steps = plan.thread(random.Random(seed)).steps
        for index, step in enumerate(steps):
            if step.retracts:
                found += 1
                assert any(
                    earlier.accepts == step.retracts and earlier.speaker == step.speaker for earlier in steps[:index]
                )
    assert found


def test_a_mentioned_deadline_is_the_round_end_and_never_an_offer():
    seen = 0
    for seed in range(3000):
        thread_plan = plan.thread(random.Random(seed))
        for step in thread_plan.steps:
            for time in step.times:
                if time.form == "context":
                    seen += 1
                    assert time.role == "other" and time.start == thread_plan.deadline
    assert seen


def test_a_loose_reference_names_the_part_of_the_day_the_offer_is_in():
    offer = plan.PlannedTime("T1", "offered", datetime(2026, 11, 14, 19, 0))
    assert plan.part_of_day(offer) == "evening"
    assert plan.part_of_day(plan.PlannedTime("T1", "offered", datetime(2026, 11, 14, 2, 0))) == ""
    ranged = plan.PlannedTime("T1", "offered", datetime(2026, 11, 14, 19, 0), datetime(2026, 11, 14, 21, 0))
    assert plan.part_of_day(ranged) == ""
