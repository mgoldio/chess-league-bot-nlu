import dataclasses
import random

from chess_league_bot_nlu.generate import command_plan


def test_commands_plan_an_event_only_when_the_intent_has_events_and_sometimes_none():
    rng = random.Random(3)
    planned = [command_plan.command(rng) for _ in range(3000)]
    assert all(command.event is None or command.event in command.intent.events for command in planned)
    notifying = [command for command in planned if command.intent.events]
    assert any(command.event is None for command in notifying) and any(command.event for command in notifying)


def test_a_contrast_batch_writes_a_whole_group_of_look_alikes_side_by_side():
    groups = [set(group) for group in command_plan.catalog.commands().contrast_groups]
    rng = random.Random(5)
    batches = [{plan.intent.key for plan in command_plan.batch(rng, 6)} for _ in range(200)]
    assert any(group <= intents for intents in batches for group in groups)
    assert all(len(plans) == 6 for plans in (command_plan.batch(rng, 6) for _ in range(50)))


def test_a_none_message_with_a_name_in_it_is_never_terse_enough_to_read_as_a_command():
    settings = command_plan.catalog.commands()
    rng = random.Random(7)
    for _ in range(3000):
        planned = command_plan.command(rng)
        if planned.none_kind:
            assert not planned.arguments or planned.none_kind.arguments
            assert not planned.arguments or planned.style.name not in settings.command_like_styles


def test_a_contrast_batch_puts_a_near_miss_beside_the_look_alikes(monkeypatch):
    settings = dataclasses.replace(command_plan.catalog.commands(), contrast_chance=1.0)
    monkeypatch.setattr(command_plan.catalog, "commands", lambda: settings)
    groups = [set(group) for group in settings.contrast_groups]
    rng = random.Random(9)
    for _ in range(100):
        plans = command_plan.batch(rng, 6)
        intents = {plan.intent.key for plan in plans}
        assert any(group <= intents for group in groups)
        assert any(plan.none_kind and plan.none_kind.near_miss for plan in plans)
