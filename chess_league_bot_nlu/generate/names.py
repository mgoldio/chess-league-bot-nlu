"""Player names, drawn from the vocabulary the way they appear in chat."""

from __future__ import annotations

import random

from . import catalog


def name(rng: random.Random, vocabulary: catalog.Vocabulary) -> str:
    word, other = rng.choice(vocabulary.username_words), rng.choice(vocabulary.username_words)
    form = rng.choice(vocabulary.username_forms)
    return form.format(w=word, W=word.capitalize(), w2=other, W2=other.capitalize(), n=rng.randint(1, 1999))


def username(rng: random.Random, vocabulary: catalog.Vocabulary) -> str:
    written = name(rng, vocabulary)
    return f"@**{written}**" if rng.random() < vocabulary.mention_chance else written
