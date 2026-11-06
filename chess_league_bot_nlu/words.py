"""Whether labelled spans sit on the model's word boundaries, so it can learn them.

GLiNER2 splits text into words its own way ("00-fri" is one word); a span that
starts or ends inside one of its words can't be learned. The splitter is
imported on first use, because importing it loads torch.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import cache


@cache
def splitter():
    from gliner2.processing.word_splitter import WhitespaceTokenSplitter

    return WhitespaceTokenSplitter()


def misaligned(text: str, spans: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """The spans that don't start and end where the model's words do."""
    starts, ends = set(), set()
    for _, start, end in splitter()(text):
        starts.add(start)
        ends.add(end)
    return [(start, end) for start, end in spans if start not in starts or end not in ends]
