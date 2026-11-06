import random

from chess_league_bot_nlu.generate import noise
from chess_league_bot_nlu.generate.placeholders import PLACEHOLDER


def test_noise_never_touches_placeholders():
    rng = random.Random(1)
    for _ in range(500):
        noisy = noise.add_noise("hey could we maybe do {T1} or {T2} instead, thanks so much", rng)
        assert PLACEHOLDER.findall(noisy) == ["{T1}", "{T2}"]


def test_noise_on_short_words_never_fails():
    rng = random.Random(2)
    for _ in range(2000):
        noise.add_noise("okay {T1} then", rng)
