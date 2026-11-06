"""Records as the token sequences the heads read, with what each head should answer.

A thread is its messages as "white: ..." lines, each tokenized on its own and
ended by a line break. A command is "@bot ..." the same way. A message is read
in ASCII: each character transliterated by `anyascii` ("–" to "-", "can’t" to
"can't", "👍" to ":thumbsup:"), which arboters does the same way before asking
the model. Labels are kept as character spans in a message's own text (`Time`,
`Event`) and turned into token labels here, so the model's answers can be
turned back the same way.

Token tags look a few tokens ahead, never past the end of their message. At a
message's last token, the message answers for every time written so far in
the thread, its own included: what it does with it (`ANSWERS`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from anyascii import anyascii

from .. import records
from ..labels import ARGUMENTS, INTENTS
from ..outcome import gold

TIME_TAGS = ("outside", "begins", "inside")
ANSWERS = ("none", "offered", "accepted", "taken back", "other")
INTENT_KEYS = tuple(INTENTS)
COMMAND_EVENTS = ("none", *sorted({key for intent in INTENTS.values() for key in intent.events}))
ARGUMENT_TAGS = ("outside", *[f"{kind} {label}" for label in ARGUMENTS for kind in ("begins", "inside")])
LOOKAHEAD = 4
COMMAND_PREFIX = "@bot "
IGNORE = -100
TRAILING = ",.!?;"


@dataclass
class Line:
    """One message as tokens. `text` is the message in ASCII, and `places` where each of the
    message's own characters starts in it (and, last, its length). `offsets` are each token's
    characters in `text`, leading spaces left out; tokens of the speaker prefix fall below zero,
    and the closing line break covers nothing. Spans in and out are in the message's own text."""

    text: str
    ids: list[int]
    offsets: list[tuple[int, int]]
    places: list[int] = field(default_factory=list)

    def tokens_of(self, start: int, end: int) -> list[int] | None:
        """The tokens from the first to the last covering a span of the message, spaces between included,
        or None when they don't give the span back (see `characters`)."""
        start, end = self.places[start], self.places[end]
        covering = [i for i, (first, last) in enumerate(self.offsets) if first < last and first < end and start < last]
        if not covering or self.characters(covering[0], covering[-1]) != (start, end):
            return None
        return list(range(covering[0], covering[-1] + 1))

    def words(self, first: int, last: int) -> tuple[int, int]:
        """The span of the message a run of tokens covers, widened to whole words, for a run the
        model began or ended inside one ("ursday" of "thursday")."""
        start, end = self.characters(first, last)
        if not 0 <= start < end:
            return start, end
        while start > 0 and word_character(self.text[start - 1]):
            start -= 1
        while end < len(self.text) and word_character(self.text[end]):
            end += 1
        return self.own(start, end)

    def own(self, start: int, end: int) -> tuple[int, int]:
        """A span of `text` in the message's own characters: from the one it starts in to the one it ends in."""
        first = max(index for index, place in enumerate(self.places) if place <= start)
        last = min(index for index, place in enumerate(self.places) if place >= end)
        return first, last

    def characters(self, first: int, last: int) -> tuple[int, int]:
        """The text of a run of tokens, less punctuation the tokenizer joined to its end (")?" in "(utc)?")."""
        start, end = self.offsets[first][0], min(self.offsets[last][1], len(self.text))
        while end > start and self.text[end - 1] in TRAILING and end - 1 > self.offsets[last][0]:
            end -= 1
        return start, end


def word_character(character: str) -> bool:
    """A character of the same word: usernames run on through underscores ("bob_endgame")."""
    return character.isalnum() or character == "_"


def ascii_text(text: str) -> tuple[str, list[int]]:
    """The text in ASCII, and where each of its characters starts there, then the length."""
    pieces = [anyascii(character) for character in text]
    places, length = [], 0
    for piece in pieces:
        places.append(length)
        length += len(piece)
    return "".join(pieces), [*places, length]


def line(tokenizer, prefix: str, text: str) -> Line:
    plain, places = ascii_text(text)
    written = f"{prefix}{plain}\n"
    encoded = tokenizer(written, add_special_tokens=False, return_offsets_mapping=True)
    offsets = []
    for start, end in encoded["offset_mapping"]:
        piece = written[start:end]
        start += len(piece) - len(piece.lstrip())
        offsets.append((start - len(prefix), end - len(prefix)))
    return Line(plain, encoded["input_ids"], offsets, places)


def spans(tags: list[int], begins: set[int], inside: dict[int, int]) -> list[tuple[int, int, int]]:
    """Tagged runs of tokens as (first, last, begin tag). `inside` maps each inside tag to its begin tag;
    an inside tag with no run open, or after another kind, starts a run."""
    found, open_run = [], None
    for index, tag in enumerate(tags):
        if open_run and inside.get(tag) == open_run[2]:
            open_run[1] = index
            continue
        if open_run:
            found.append(tuple(open_run))
            open_run = None
        if tag in begins:
            open_run = [index, index, tag]
        elif tag in inside:
            open_run = [index, index, inside[tag]]
    if open_run:
        found.append(tuple(open_run))
    return found


@dataclass
class ThreadExample:
    """A thread's lines; for training also its tags, its times as (line, first token, last token), and
    each message's answers as (line, time, same message, same sender, answer)."""

    thread: records.Thread
    lines: list[Line]
    tags: list[list[int]] = field(default_factory=list)
    times: list[tuple[int, int, int]] = field(default_factory=list)
    answers: list[tuple[int, int, int, int, int]] = field(default_factory=list)


def thread_lines(tokenizer, thread: records.Thread) -> list[Line]:
    return [line(tokenizer, f"{message.speaker}: ", message.text) for message in thread.messages]


def asked(thread: records.Thread, time_lines: list[int], at: int) -> list[tuple[int, int, int]]:
    """The times a message answers for: every time written up to it, as (time, same message, same sender)."""
    speaker = thread.messages[at].speaker
    return [
        (index, int(written == at), int(thread.messages[written].speaker == speaker))
        for index, written in enumerate(time_lines)
        if written <= at
    ]


def thread_example(tokenizer, thread: records.Thread) -> ThreadExample | None:
    """A thread with its labels, or None when a labelled time doesn't fall on token edges."""
    lines = thread_lines(tokenizer, thread)
    where = {message.id: index for index, message in enumerate(thread.messages)}
    times, events = gold(thread)
    times.sort(key=lambda time: (where[time.message], time.start))
    example = ThreadExample(thread, lines, [[TIME_TAGS.index("outside")] * len(each.ids) for each in lines])
    for time in times:
        line_index = where[time.message]
        tokens = lines[line_index].tokens_of(time.start, time.end)
        if tokens is None:
            return None
        example.tags[line_index][tokens[0]] = TIME_TAGS.index("begins")
        for token in tokens[1:]:
            example.tags[line_index][token] = TIME_TAGS.index("inside")
        example.times.append((line_index, tokens[0], tokens[-1]))
    done = {(event.message, event.offer): "accepted" if event.kind == "accept" else "taken back" for event in events}
    time_lines = [line_index for line_index, _, _ in example.times]
    for at, message in enumerate(thread.messages):
        for index, same, own in asked(thread, time_lines, at):
            answer = times[index].role if same else done.get((message.id, times[index]), "none")
            example.answers.append((at, index, same, own, ANSWERS.index(answer)))
    return example


@dataclass
class CommandExample:
    line: Line
    tags: list[int]
    intent: int = IGNORE
    event: int = IGNORE
    id: str = ""


def command_example(tokenizer, command: records.Command) -> CommandExample | None:
    tokens = line(tokenizer, COMMAND_PREFIX, command.text)
    tags = [ARGUMENT_TAGS.index("outside")] * len(tokens.ids)
    for argument in command.arguments:
        covering = tokens.tokens_of(argument.start, argument.end)
        if covering is None:
            return None
        tags[covering[0]] = ARGUMENT_TAGS.index(f"begins {argument.label}")
        for token in covering[1:]:
            tags[token] = ARGUMENT_TAGS.index(f"inside {argument.label}")
    event = COMMAND_EVENTS.index(command.event or "none") if INTENTS[command.intent].events else IGNORE
    return CommandExample(tokens, tags, INTENT_KEYS.index(command.intent), event, command.id)


@dataclass
class Batch:
    """Sequences padded to one length. `ahead` is the token each token's tag also looks at; `ends`
    and `times` are positions in the sequences: (sequence, token) and (sequence, first, last)."""

    ids: torch.Tensor
    mask: torch.Tensor
    ahead: torch.Tensor
    tags: torch.Tensor
    ends: torch.Tensor
    times: torch.Tensor = None
    asked: torch.Tensor = None
    answers: torch.Tensor = None
    intents: torch.Tensor = None
    command_events: torch.Tensor = None

    def to(self, where: str) -> Batch:
        return Batch(**{name: value.to(where) if value is not None else None for name, value in vars(self).items()})


def sequences(rows: list[list[Line]], tags: list[list[list[int]]], pad: int):
    """Each row's lines one after another, padded: ids, mask, look-ahead, tags, and where each line starts."""
    width = max(sum(len(each.ids) for each in lines) for lines in rows)
    ids = torch.full((len(rows), width), pad, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    ahead = torch.arange(width).repeat(len(rows), 1)
    tag_table = torch.full((len(rows), width), IGNORE, dtype=torch.long)
    starts = []
    for row, (lines, line_tags) in enumerate(zip(rows, tags, strict=True)):
        position = 0
        for each, each_tags in zip(lines, line_tags, strict=True):
            end = position + len(each.ids)
            ids[row, position:end] = torch.tensor(each.ids)
            mask[row, position:end] = 1
            ahead[row, position:end] = torch.clamp(torch.arange(position, end) + LOOKAHEAD, max=end - 1)
            tag_table[row, position:end] = torch.tensor(each_tags)
            starts.append((row, position, len(each.ids)))
            position = end
    return ids, mask, ahead, tag_table, starts


def thread_batch(examples: list[ThreadExample], pad: int) -> Batch:
    ids, mask, ahead, tags, starts = sequences([e.lines for e in examples], [e.tags for e in examples], pad)
    times, questions, answers = [], [], []
    first_line = 0
    for example in examples:
        first_time = len(times)
        for line_index, first, last in example.times:
            row, start, _ = starts[first_line + line_index]
            times.append((row, start + first, start + last))
        for at, time, same, own, answer in example.answers:
            questions.append((first_line + at, first_time + time, same, own))
            answers.append(answer)
        first_line += len(example.lines)
    return Batch(
        ids,
        mask,
        ahead,
        tags,
        torch.tensor([(row, start + length - 1) for row, start, length in starts], dtype=torch.long),
        torch.tensor(times, dtype=torch.long).reshape(-1, 3),
        torch.tensor(questions, dtype=torch.long).reshape(-1, 4),
        torch.tensor(answers, dtype=torch.long),
    )


def command_batch(examples: list[CommandExample], pad: int) -> Batch:
    ids, mask, ahead, tags, starts = sequences([[e.line] for e in examples], [[e.tags] for e in examples], pad)
    return Batch(
        ids,
        mask,
        ahead,
        tags,
        torch.tensor([(row, start + length - 1) for row, start, length in starts], dtype=torch.long),
        intents=torch.tensor([example.intent for example in examples], dtype=torch.long),
        command_events=torch.tensor([example.event for example in examples], dtype=torch.long),
    )
