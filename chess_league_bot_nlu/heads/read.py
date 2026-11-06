"""A trained reader's answers: for a thread, the times it writes and what each message does with them
(`outcome`); for a command, the command ranked, its event, and its arguments.

A thread is read in one pass: the trunk reads left to right, tags look ahead
only within their message, and each message answers at its own last token, so
the answers for the first n messages are the same as reading those n alone.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from .. import records
from ..labels import INTENTS
from ..outcome import Event, Time
from .inputs import (
    ANSWERS,
    ARGUMENT_TAGS,
    COMMAND_EVENTS,
    COMMAND_PREFIX,
    INTENT_KEYS,
    TIME_TAGS,
    CommandExample,
    ThreadExample,
    asked,
    command_batch,
    line,
    spans,
    thread_batch,
    thread_lines,
)
from .model import load

ARGUMENT_BEGINS = {index for index, tag in enumerate(ARGUMENT_TAGS) if tag.startswith("begins ")}
ARGUMENT_INSIDE = {
    index: ARGUMENT_TAGS.index("begins " + tag.removeprefix("inside "))
    for index, tag in enumerate(ARGUMENT_TAGS)
    if tag.startswith("inside ")
}


class HeadsModel:
    """A trained reader."""

    def __init__(self, reader, tokenizer, device: str):
        self.reader, self.tokenizer, self.device = reader, tokenizer, device
        self.pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        self.seconds: list[float] = []

    @classmethod
    def saved(cls, where: str, device: str) -> HeadsModel:
        return cls(*load(Path(where), device), device)

    def read_thread(self, thread: records.Thread) -> tuple[list[Time], list[Event]]:
        """The times each message writes, and what each message does with earlier ones."""
        started = time.perf_counter()
        found = self.whole_thread(thread)
        self.seconds.append(time.perf_counter() - started)
        return found

    @torch.inference_mode()
    def whole_thread(self, thread: records.Thread) -> tuple[list[Time], list[Event]]:
        reader = self.reader
        lines = thread_lines(self.tokenizer, thread)
        batch = thread_batch([ThreadExample(thread, lines, [[0] * len(each.ids) for each in lines])], self.pad)
        batch = batch.to(self.device)
        states = reader.states(batch.ids, batch.mask)
        tags = reader.tag_scores(reader.time_tags, states, batch.ahead)[0].argmax(-1).tolist()
        begins, inside = {TIME_TAGS.index("begins")}, {TIME_TAGS.index("inside"): TIME_TAGS.index("begins")}
        places, written, time_lines, position = [], [], [], 0
        for index, (each, message) in enumerate(zip(lines, thread.messages, strict=True)):
            for first, last, _ in spans(tags[position : position + len(each.ids)], begins, inside):
                start, end = each.words(first, last)
                if 0 <= start < end:
                    places.append((0, position + first, position + last))
                    written.append((message.id, start, end))
                    time_lines.append(index)
            position += len(each.ids)
        if not places:
            return [], []
        questions = [
            (at, written_time, same, own)
            for at in range(len(lines))
            for written_time, same, own in asked(thread, time_lines, at)
        ]
        spans_read = reader.span_states(states, torch.tensor(places, device=self.device))
        ends = reader.at(states, batch.ends)
        found = reader.answer_scores(ends, spans_read, torch.tensor(questions, device=self.device))
        roles, done = {}, []
        for (at, written_time, same, _), answer in zip(questions, found.argmax(-1).tolist(), strict=True):
            if same:
                roles[written_time] = ANSWERS[answer] if ANSWERS[answer] in ("offered", "accepted") else "other"
            elif ANSWERS[answer] in ("accepted", "taken back"):
                done.append((at, written_time, "accept" if ANSWERS[answer] == "accepted" else "take back"))
        times = [Time(*place, roles[index]) for index, place in enumerate(written)]
        events = [Event(thread.messages[at].id, times[index], kind) for at, index, kind in done]
        return times, events

    @torch.inference_mode()
    def read_command(self, text: str) -> dict:
        """The commands ranked by probability, the event if the top one has events, and the
        arguments as (label, start, end)."""
        started = time.perf_counter()
        reader = self.reader
        tokens = line(self.tokenizer, COMMAND_PREFIX, text)
        batch = command_batch([CommandExample(tokens, [0] * len(tokens.ids))], self.pad).to(self.device)
        states = reader.states(batch.ids, batch.mask)
        ends = reader.at(states, batch.ends)
        chances = torch.softmax(reader.intents(ends)[0], -1).tolist()
        ranked = sorted(zip(INTENT_KEYS, chances, strict=True), key=lambda pair: -pair[1])
        result: dict = {"commands": ranked, "event": None, "arguments": []}
        if events := INTENTS[ranked[0][0]].events:
            event = COMMAND_EVENTS[int(reader.command_events(ends)[0].argmax())]
            result["event"] = event if event in events else None
        tags = reader.tag_scores(reader.arguments, states, batch.ahead)[0, : len(tokens.ids)].argmax(-1).tolist()
        for first, last, tag in spans(tags, ARGUMENT_BEGINS, ARGUMENT_INSIDE):
            start, end = tokens.words(first, last)
            if 0 <= start < end:
                result["arguments"].append((ARGUMENT_TAGS[tag].removeprefix("begins "), start, end))
        self.seconds.append(time.perf_counter() - started)
        return result
