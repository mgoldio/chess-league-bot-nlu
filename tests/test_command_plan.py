import random

from chess_league_bot_nlu.generate import command_plan


def test_commands_plan_their_event_only_when_the_intent_has_events():
    rng = random.Random(3)
    for _ in range(500):
        command = command_plan.command(rng)
        assert (command.event is not None) == bool(command.intent.events)
