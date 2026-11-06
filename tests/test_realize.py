from chess_league_bot_nlu.generate import realize


def test_fill_records_where_each_value_went():
    text, spans = realize.fill("how about {T1} or {T2}?", {"T1": "thu 19:00", "T2": "fri 20h"})
    assert text == "how about thu 19:00 or fri 20h?"
    assert [text[start:end] for start, end in spans.values()] == ["thu 19:00", "fri 20h"]


def test_a_list_is_laid_out_one_time_per_line_where_the_writer_put_it():
    draft = realize.Draft("", {"T1": "sat 11:00", "T2": "sun 13:00"}, 20, listed=["T1", "T2"], bullet="- ")
    template = draft.laid_out("hi! how about: {LIST}. let me know")
    assert template == "hi! how about:\n- {T1}\n- {T2}\nlet me know"
    assert draft.placeholders == ["T1", "T2"]


def test_a_struck_time_goes_right_before_the_time_that_replaced_it():
    draft = realize.Draft("", {"T2": "sat 19:00", "T1": "sat 20:00"}, 20, struck={"T1": "T2"})
    text, _ = realize.fill(draft.edited("{T1} ok?"), draft.values)
    assert text == "~~sat 19:00~~ sat 20:00 ok?"
    assert draft.placeholders == ["T1"]


def test_a_placeholder_written_without_braces_is_mended_once():
    draft = realize.Draft("", {"T4": "sat 19:00", "T1": "sun 20:00"}, 20)
    assert draft.laid_out("T4 is too late but {T1} works") == "{T4} is too late but {T1} works"
    assert draft.laid_out("T4 or T4, {T1}") == "T4 or T4, {T1}"


def test_times_the_writer_may_leave_out_are_left_out_of_the_record():
    import random

    from chess_league_bot_nlu.generate import gold, plan

    for seed in range(300):
        rng = random.Random(seed)
        thread_plan = plan.thread(rng)
        _, _, drafts = realize.thread_prompt(thread_plan, "chat", rng)
        if not any(set(draft.optional) - {realize.OPPONENT} for draft in drafts):
            continue
        texts = []
        for draft in drafts:
            names = [name for name in draft.placeholders if name not in draft.listed]
            template = " or ".join(f"{{{name}}}" for name in names) + (" {LIST}" if draft.listed else "") + " ok"
            text, spans = realize.fill(draft.edited(draft.laid_out(template)), draft.values)
            texts.append((text, draft.mentions(spans)))
        thread = gold.thread_record("t", thread_plan, texts)
        planned = {time.id for step in thread_plan.steps for time in step.times}
        assert {mention.id for mention in thread.mentions} < planned
        return
    raise AssertionError("no plan with an optional time")
