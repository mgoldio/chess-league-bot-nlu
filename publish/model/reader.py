"""A self-contained reader for Lichess4545/chess-league-bot-nlu: the model's architecture as code.

Needs torch, transformers (5 or later), safetensors and anyascii (0.3.3). From a download of the
model repository:

    from reader import Reader

    reader = Reader.load(".")
    reader.read_command("show me the standings")
    reader.read_thread([("white", "can we do thu 19:00 or sat 14:00?"), ("black", "sat works")])
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from anyascii import anyascii
from safetensors.torch import load_file
from torch import nn
from transformers import AutoTokenizer, LlamaConfig, LlamaModel


@dataclass(frozen=True)
class Time:
    """A time written in a message: its characters in the message's own text, and its role there
    ("offered", "accepted" when restated while accepting, or "other")."""

    message: int
    start: int
    end: int
    role: str


@dataclass(frozen=True)
class Event:
    """A message accepting ("accept") or taking back ("take back") a time written in an earlier one."""

    message: int
    time: Time
    kind: str


class Heads(nn.Module):
    """The heads, as `model.safetensors` holds them (every tensor not under "trunk.")."""

    def __init__(self, hidden: int, layers: int, labels: dict, layer_norm_eps: float):
        super().__init__()
        self.mix = nn.Parameter(torch.zeros(layers + 1))
        self.norm = nn.LayerNorm(hidden, eps=layer_norm_eps)
        self.time_tags = nn.Linear(2 * hidden, len(labels["time_tags"]))
        self.arguments = nn.Linear(2 * hidden, len(labels["argument_tags"]))
        self.span = nn.Linear(2 * hidden, hidden)
        self.same = nn.Embedding(2, hidden)
        self.own = nn.Embedding(2, hidden)
        self.pointer = nn.Sequential(
            nn.Linear(3 * hidden, hidden), nn.GELU(), nn.Linear(hidden, len(labels["answers"]))
        )
        self.intents = nn.Linear(hidden, len(labels["intents"]))
        self.command_events = nn.Linear(hidden, len(labels["command_events"]))

    def states(self, hidden_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """The token states the heads read: a learned mix of the embeddings and every layer, normalized."""
        weights = torch.softmax(self.mix, 0)
        return self.norm(torch.einsum("l,lth->th", weights, torch.stack([each[0] for each in hidden_states])))

    def tags(self, layer: nn.Linear, states: torch.Tensor, ahead: torch.Tensor) -> torch.Tensor:
        """Tag scores for each token, from it and the token `ahead` of it."""
        return layer(torch.cat([states, states[ahead]], -1))

    def answers(self, query: torch.Tensor, key: torch.Tensor, same: torch.Tensor, own: torch.Tensor) -> torch.Tensor:
        """What a message (its last token's state) does with a time (its key), written in the same
        message or not, and by the same player or not."""
        key = key + self.same(same) + self.own(own)
        return self.pointer(torch.cat([query, key, query * key], -1))


class Line:
    """One line as tokens: the text in ASCII, where each of the original characters starts in it,
    and each token's characters in the ASCII text (the prefix's below zero)."""

    def __init__(self, tokenizer, prefix: str, text: str, trailing: str):
        pieces = [anyascii(character) for character in text]
        self.places, length = [], 0
        for piece in pieces:
            self.places.append(length)
            length += len(piece)
        self.places.append(length)
        self.text, self.trailing = "".join(pieces), trailing
        written = f"{prefix}{self.text}\n"
        encoded = tokenizer(written, add_special_tokens=False, return_offsets_mapping=True)
        self.ids = encoded["input_ids"]
        self.offsets = []
        for start, end in encoded["offset_mapping"]:
            piece = written[start:end]
            start += len(piece) - len(piece.lstrip())
            self.offsets.append((start - len(prefix), end - len(prefix)))

    def span(self, first: int, last: int) -> tuple[int, int] | None:
        """The original characters a run of tokens covers: less punctuation joined to its end,
        widened to whole words, mapped back from ASCII; None when it covers no text."""
        start, end = self.offsets[first][0], min(self.offsets[last][1], len(self.text))
        while end > start and self.text[end - 1] in self.trailing and end - 1 > self.offsets[last][0]:
            end -= 1
        if not 0 <= start < end:
            return None
        while start > 0 and (self.text[start - 1].isalnum() or self.text[start - 1] == "_"):
            start -= 1
        while end < len(self.text) and (self.text[end].isalnum() or self.text[end] == "_"):
            end += 1
        first_character = max(index for index, place in enumerate(self.places) if place <= start)
        last_character = min(index for index, place in enumerate(self.places) if place >= end)
        return first_character, last_character


def runs(tags: list[str], kind: str = "") -> list[tuple[int, int, str]]:
    """Tagged runs as (first token, last token, kind): a "begins" and the "inside" tags of the same
    kind after it; a stray "inside" starts a run too."""
    found, current = [], None
    for index, tag in enumerate(tags):
        position, _, label = tag.partition(" ")
        label = label or kind
        if current and position == "inside" and label == current[2]:
            current[1] = index
            continue
        if current:
            found.append(tuple(current))
            current = None
        if position in ("begins", "inside"):
            current = [index, index, label]
    if current:
        found.append(tuple(current))
    return found


class Reader:
    def __init__(self, trunk, heads: Heads, tokenizer, labels: dict):
        self.trunk, self.heads, self.tokenizer, self.labels = trunk, heads, tokenizer, labels

    @classmethod
    def load(cls, where: str | Path) -> Reader:
        where = Path(where)
        labels = json.loads((where / "labels.json").read_text())
        config = json.loads((where / "config.json").read_text())
        trunk = LlamaModel(LlamaConfig(**config["trunk"]))
        heads = Heads(trunk.config.hidden_size, trunk.config.num_hidden_layers, labels, config["layer_norm_eps"])
        tensors = load_file(where / "model.safetensors")
        trunk.load_state_dict({name[6:]: value for name, value in tensors.items() if name.startswith("trunk.")})
        heads.load_state_dict({name: value for name, value in tensors.items() if not name.startswith("trunk.")})
        return cls(trunk.eval(), heads.eval(), AutoTokenizer.from_pretrained(where), labels)

    def encode(self, lines: list[Line]) -> torch.Tensor:
        ids = torch.tensor([[token for line in lines for token in line.ids]])
        with torch.inference_mode():
            hidden = self.trunk(input_ids=ids, output_hidden_states=True).hidden_states
            return self.heads.states(hidden)

    def ahead(self, lines: list[Line]) -> torch.Tensor:
        """Each token's look-ahead: `lookahead` tokens on, never past the end of its line."""
        found, position = [], 0
        for line in lines:
            end = position + len(line.ids)
            found += [min(token + self.labels["lookahead"], end - 1) for token in range(position, end)]
            position = end
        return torch.tensor(found)

    @torch.inference_mode()
    def read_command(self, text: str) -> dict:
        """The commands ranked by probability, the event if the top one has events, and the arguments."""
        line = Line(self.tokenizer, self.labels["command_prefix"], text, self.labels["trailing"])
        states = self.encode([line])
        end = states[-1]
        chances = torch.softmax(self.heads.intents(end), -1).tolist()
        ranked = sorted(zip(self.labels["intents"], chances, strict=True), key=lambda pair: -pair[1])
        result = {"commands": ranked, "event": None, "arguments": []}
        if self.labels["commands"][ranked[0][0]]["events"]:
            event = self.labels["command_events"][int(self.heads.command_events(end).argmax())]
            result["event"] = event if event in self.labels["commands"][ranked[0][0]]["events"] else None
        tags = self.heads.tags(self.heads.arguments, states, self.ahead([line])).argmax(-1).tolist()
        for first, last, label in runs([self.labels["argument_tags"][tag] for tag in tags]):
            if span := line.span(first, last):
                result["arguments"].append((label, text[span[0] : span[1]], span))
        return result

    @torch.inference_mode()
    def read_thread(self, messages: list[tuple[str, str]]) -> tuple[list[Time], list[Event]]:
        """The times each message writes, and what each later message does with them. `messages`
        are (speaker, text), speaker "white" or "black", in order."""
        if not messages:
            return [], []
        if unknown := {speaker for speaker, _ in messages} - {"white", "black"}:
            raise ValueError(f"speakers are white and black, not {sorted(unknown)}")
        lines = [Line(self.tokenizer, f"{speaker}: ", text, self.labels["trailing"]) for speaker, text in messages]
        states = self.encode(lines)
        tags = self.heads.tags(self.heads.time_tags, states, self.ahead(lines)).argmax(-1).tolist()
        tag_names = [self.labels["time_tags"][tag] for tag in tags]
        written, keys, ends, position = [], [], [], 0
        for index, line in enumerate(lines):
            for first, last, _ in runs(tag_names[position : position + len(line.ids)], "time"):
                if span := line.span(first, last):
                    written.append((index, *span))
                    keys.append(self.heads.span(torch.cat([states[position + first], states[position + last]], -1)))
            position += len(line.ids)
            ends.append(states[position - 1])
        roles, events = {}, []
        for at, (speaker, _) in enumerate(messages):
            for number, (message, _, _) in enumerate(written):
                if message > at:
                    continue
                same, own = int(message == at), int(messages[message][0] == speaker)
                scores = self.heads.answers(ends[at], keys[number], torch.tensor(same), torch.tensor(own))
                answer = self.labels["answers"][int(scores.argmax())]
                if same:
                    roles[number] = answer if answer in ("offered", "accepted") else "other"
                elif answer in ("accepted", "taken back"):
                    events.append((at, number, "accept" if answer == "accepted" else "take back"))
        times = [Time(message, start, end, roles[number]) for number, (message, start, end) in enumerate(written)]
        return times, [Event(at, times[number], kind) for at, number, kind in events]
