"""The trunk and its heads.

Token tags read a token and the token a few ahead of it in the same message.
The pointer reads, at a message's last token, every time written so far: the
time's first and last tokens, whether it is in this message, and whether this message's
sender wrote it; and answers what this message does with it. Commands are read
at their last token, and their arguments tagged like times.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from .inputs import ANSWERS, ARGUMENT_TAGS, COMMAND_EVENTS, INTENT_KEYS, TIME_TAGS


class Reader(nn.Module):
    def __init__(self, trunk):
        super().__init__()
        config = trunk.config
        hidden = config.hidden_size
        self.trunk = trunk
        self.mix = nn.Parameter(torch.zeros(config.num_hidden_layers + 1))
        self.norm = nn.LayerNorm(hidden)
        self.time_tags = nn.Linear(2 * hidden, len(TIME_TAGS))
        self.arguments = nn.Linear(2 * hidden, len(ARGUMENT_TAGS))
        self.span = nn.Linear(2 * hidden, hidden)
        self.same = nn.Embedding(2, hidden)
        self.own = nn.Embedding(2, hidden)
        self.pointer = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.GELU(), nn.Linear(hidden, len(ANSWERS)))
        self.intents = nn.Linear(hidden, len(INTENT_KEYS))
        self.command_events = nn.Linear(hidden, len(COMMAND_EVENTS))

    def states(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """A learned mix of the trunk's layers, for every token."""
        found = self.trunk(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states
        weights = torch.softmax(self.mix, 0).to(found[0].dtype)
        return self.norm(torch.einsum("l,lbth->bth", weights, torch.stack(found)))

    def tag_scores(self, tags: nn.Linear, states: torch.Tensor, ahead: torch.Tensor) -> torch.Tensor:
        looked = states.gather(1, ahead[..., None].expand(-1, -1, states.shape[-1]))
        return tags(torch.cat([states, looked], -1))

    def at(self, states: torch.Tensor, places: torch.Tensor) -> torch.Tensor:
        """The states at (sequence, token) places."""
        return states[places[:, 0], places[:, 1]]

    def span_states(self, states: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """Each time from its first and last tokens, (sequence, first token, last token): its first
        token names the day, which a mean over its many digit tokens would drown."""
        first, last = states[times[:, 0], times[:, 1]], states[times[:, 0], times[:, 2]]
        return self.span(torch.cat([first, last], -1))

    def answer_scores(self, ends: torch.Tensor, spans: torch.Tensor, asked: torch.Tensor) -> torch.Tensor:
        """For each (message, time, same message, same sender): what the message does with the time."""
        query = ends[asked[:, 0]]
        key = spans[asked[:, 1]] + self.same(asked[:, 2]) + self.own(asked[:, 3])
        return self.pointer(torch.cat([query, key, query * key], -1))


def save(reader: Reader, tokenizer, where: Path, settings: dict) -> None:
    """The model as one `model.safetensors` (the trunk's tensors under "trunk.", the heads' by
    their own names) and `config.json` (the trunk's Llama config, with `rope_theta` also written
    flat for older readers), the tokenizer, and how it was trained (`training.json`)."""
    from safetensors.torch import save_file

    where.mkdir(parents=True, exist_ok=True)
    tensors = {name: value.detach().contiguous() for name, value in reader.state_dict().items()}
    save_file(tensors, where / "model.safetensors")
    trunk = reader.trunk.config.to_dict()
    rope = trunk.get("rope_parameters") or {}
    if "rope_theta" in rope:
        trunk["rope_theta"] = rope["rope_theta"]
    config = {"trunk": trunk, "layer_norm_eps": reader.norm.eps}
    (where / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    tokenizer.save_pretrained(where)
    (where / "training.json").write_text(json.dumps(settings, indent=2) + "\n")


def load(where: Path, device: str) -> tuple[Reader, object]:
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, LlamaConfig, LlamaModel

    config = json.loads((where / "config.json").read_text())
    reader = Reader(LlamaModel(LlamaConfig(**config["trunk"])))
    reader.load_state_dict(load_file(where / "model.safetensors"))
    return reader.to(device).eval(), AutoTokenizer.from_pretrained(where)
