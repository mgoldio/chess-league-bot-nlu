import pytest

from chess_league_bot_nlu.generate import catalog
from chess_league_bot_nlu.generate.wording import CONDITIONS


def test_every_content_file_loads_and_agrees_with_the_others():
    catalog.check()


def test_every_plot_beat_has_wording():
    beats = catalog.beats()
    for plot in catalog.plots().plots:
        for beat in plot.beats:
            assert beat.do in beats or beat.do in ("offer", "accept")


def test_a_misspelled_key_fails_with_the_file_named():
    with pytest.raises(catalog.InvalidContent, match=r"plots\.toml"):
        catalog.build(catalog.Beat, {"by": "first", "do": "ofer"}, "plots.toml [quick]")
    with pytest.raises(catalog.InvalidContent, match="unknown keys"):
        catalog.build(catalog.Form, {"form": "{wd}", "lowr": 0.5}, "time_formats.toml days")


def test_every_when_condition_is_one_the_wording_knows():
    formats = catalog.time_formats()
    for forms in (formats.days, formats.day_only, formats.clocks, formats.points, formats.ranges, formats.open):
        for form in forms:
            assert form.when is None or form.when in CONDITIONS, form


def test_every_command_has_ways_to_put_it():
    angles = catalog.commands().angles
    assert {key for key in catalog.INTENTS if key != "none"} <= set(angles)
