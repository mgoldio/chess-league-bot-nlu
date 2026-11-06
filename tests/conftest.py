import random

import pytest

from chess_league_bot_nlu import records
from chess_league_bot_nlu.generate import gold, plan, realize


def perfect_thread(seed: int) -> tuple[plan.ThreadPlan, records.Thread]:
    """A plan written by a writer that does exactly what it is told."""
    rng = random.Random(seed)
    thread_plan = plan.thread(rng)
    _, _, drafts = realize.thread_prompt(thread_plan, "chat", rng)
    texts = []
    for step, draft in zip(thread_plan.steps, drafts, strict=True):
        written = {*draft.placeholders, *draft.optional} - {realize.OPPONENT, *draft.listed}
        names = [name for name in draft.values if name in written]
        template = " or ".join(f"{{{name}}}" for name in names) + (" {LIST}" if draft.listed else "") + " ok"
        cities = [time.zone[0].removesuffix(" time") for time in step.times if time.own_zone and time.role == "offered"]
        if cities:
            template = f"im in {cities[0]}, " + template
        text, spans = realize.fill(draft.edited(draft.laid_out(template)), draft.values)
        texts.append((text, draft.mentions(spans)))
    return thread_plan, gold.thread_record(f"t{seed}", thread_plan, texts)


@pytest.fixture
def perfect():
    return perfect_thread
