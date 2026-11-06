"""Typing noise, applied in code to the words around placeholders, never inside them."""

from __future__ import annotations

import random
import re

from . import catalog
from .placeholders import PLACEHOLDER


def add_noise(template: str, rng: random.Random) -> str:
    settings = catalog.noise()
    word = re.compile(rf"[A-Za-z']{{{settings.min_word_length},}}")
    pieces = PLACEHOLDER.split(template)
    editable = [index for index, piece in enumerate(pieces) if not PLACEHOLDER.fullmatch(piece) and word.search(piece)]
    for _ in range(rng.randint(*settings.edits)):
        if editable:
            index = rng.choice(editable)
            pieces[index] = typo(pieces[index], word, rng.choice(settings.kinds), rng)
    if rng.random() < settings.lowercase_chance:
        pieces = [piece if PLACEHOLDER.fullmatch(piece) else piece.lower() for piece in pieces]
    return "".join(pieces)


def typo(piece: str, word: re.Pattern, kind: str, rng: random.Random) -> str:
    """One typo in a piece of text between placeholders (none if an earlier typo left no word long enough)."""
    matches = list(word.finditer(piece))
    if not matches:
        return piece
    match = rng.choice(matches)
    letters = list(match.group(0))
    position = rng.randrange(len(letters) - 1)
    if kind == "swap":
        letters[position], letters[position + 1] = letters[position + 1], letters[position]
    elif kind == "drop":
        del letters[position]
    elif kind == "double":
        letters.insert(position, letters[position])
    elif kind == "apostrophe" and "'" in letters:
        letters.remove("'")
    elif kind == "join":
        rest = piece[match.end() :]
        return piece[: match.end()] + rest[1:] if re.match(r" [A-Za-z]", rest) else piece
    return piece[: match.start()] + "".join(letters) + piece[match.end() :]
