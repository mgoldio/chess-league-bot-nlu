"""Fine-tune the whole trunk with the heads, the trunk at a lower learning rate, on a generated dataset.

    uv run --group train python -m chess_league_bot_nlu.heads.train data/v3+data/gaps-1 runs/smol

A thread trains every message's answers in one pass, shown each epoch on one of
several weeks of the year, with white and black swapped half the time. The
learning rate warms up and holds; after every epoch a short decay to zero is
branched off and saved (`epoch-N`), then the run goes on from before it, so each
epoch's checkpoint is a finished run of that length. `best` is the one with the
lowest loss on the test split; report final numbers on data the run never saw.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import random
import time
from pathlib import Path

from .. import dataset, device
from ..records import opponent

TRUNK = "HuggingFaceTB/SmolLM2-135M"


def swapped(thread):
    """The thread with white and black trading places; nothing the model answers depends on which is which."""

    messages = [dataclasses.replace(message, speaker=opponent(message.speaker)) for message in thread.messages]
    return dataclasses.replace(thread, messages=messages)


def examples(
    tokenizer, threads, commands, inputs, swap: bool = False, weeks: int = 0, seed: int = 0
) -> tuple[list, list]:
    """Threads and commands as examples; a record with a span off token edges is left out whole.
    Each thread is a tuple of the ways it may be shown: on its own week; with `weeks`, also on that
    many other weeks of the year; and with `swap`, each of those with white and black swapped."""
    thread_examples = []
    for thread in threads:
        moves = [random.Random(f"{seed}:{thread.id}:{copy}").randint(1, 51) for copy in range(weeks)]
        shown = [thread, *(thread.moved(count) for count in moves)]
        if swap:
            shown += [swapped(each) for each in shown]
        ways = [example for each in shown if (example := inputs.thread_example(tokenizer, each))]
        if ways:
            thread_examples.append(tuple(ways))
    command_examples = [example for command in commands if (example := inputs.command_example(tokenizer, command))]
    left_out = len(threads) - len(thread_examples), len(commands) - len(command_examples)
    if any(left_out):
        print(f"left out {left_out[0]} threads and {left_out[1]} commands with spans off token edges")
    return thread_examples, command_examples


def batches(
    thread_examples, command_examples, thread_size, command_size, rng, shares: dict[str, float] | None = None
) -> list[tuple[str, list]]:
    """Shuffled batches of one kind each; each thread shown one of its ways, at random, and a
    record with a share (`dataset.shares`) shown only that share of the time."""
    shares = shares or {}
    shown = [rng.choice(ways) for ways in thread_examples if rng.random() < shares.get(ways[0].thread.id, 1)]
    commands = [example for example in command_examples if rng.random() < shares.get(example.id, 1)]
    shuffled_threads, shuffled_commands = rng.sample(shown, len(shown)), rng.sample(commands, len(commands))
    planned = [("thread", shuffled_threads[i : i + thread_size]) for i in range(0, len(shuffled_threads), thread_size)]
    planned += [
        ("command", shuffled_commands[i : i + command_size]) for i in range(0, len(shuffled_commands), command_size)
    ]
    rng.shuffle(planned)
    return planned


def losses(reader, kind: str, batch) -> dict:
    from torch.nn.functional import cross_entropy

    from .inputs import IGNORE

    states = reader.states(batch.ids, batch.mask)
    ends = reader.at(states, batch.ends)
    if kind == "command":
        tags = reader.tag_scores(reader.arguments, states, batch.ahead)
        found = {
            "arguments": cross_entropy(tags.flatten(0, 1), batch.tags.flatten(), ignore_index=IGNORE),
            "command": cross_entropy(reader.intents(ends), batch.intents),
        }
        if (batch.command_events != IGNORE).any():
            found["command event"] = cross_entropy(
                reader.command_events(ends), batch.command_events, ignore_index=IGNORE
            )
        return found
    tags = reader.tag_scores(reader.time_tags, states, batch.ahead)
    found = {"time tags": cross_entropy(tags.flatten(0, 1), batch.tags.flatten(), ignore_index=IGNORE)}
    if len(batch.asked):
        spans = reader.span_states(states, batch.times)
        found["answers"] = cross_entropy(reader.answer_scores(ends, spans, batch.asked), batch.answers)
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="a dataset directory, or several joined with +")
    parser.add_argument("output", type=Path)
    parser.add_argument("--trunk", default=TRUNK)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--thread-batch", type=int, default=8)
    parser.add_argument("--command-batch", type=int, default=32)
    parser.add_argument("--heads-lr", type=float, default=1e-3)
    parser.add_argument("--trunk-lr", type=float, default=3e-5)
    parser.add_argument("--weeks", type=int, default=3, help="also show each training thread on this many other weeks")
    parser.add_argument("--decay-share", type=float, default=0.2, help="a decay's length, as a share of an epoch")
    parser.add_argument("--limit", type=int, help="use only this many threads and commands of each split, to try a run")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    where = device.select()
    import torch
    from transformers import AutoModel, AutoTokenizer

    from . import inputs
    from .model import Reader, save

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.trunk)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    command_train, command_test = dataset.split(dataset.commands(args.data))
    thread_train, thread_test = dataset.split(dataset.threads(args.data))
    if args.limit:
        command_train, command_test = command_train[: args.limit], command_test[: args.limit]
        thread_train, thread_test = thread_train[: args.limit], thread_test[: args.limit]
    train = examples(tokenizer, thread_train, command_train, inputs, swap=True, weeks=args.weeks, seed=args.seed)
    test = examples(tokenizer, thread_test, command_test, inputs)
    print(
        f"train {len(train[0])} threads and {len(train[1])} commands, test {len(test[0])} and {len(test[1])}, on {where}"
    )

    reader = Reader(AutoModel.from_pretrained(args.trunk, dtype=torch.float32)).to(where)
    head_parameters = [p for name, p in reader.named_parameters() if not name.startswith("trunk.")]
    groups = [
        {"params": head_parameters, "lr": args.heads_lr},
        {"params": list(reader.trunk.parameters()), "lr": args.trunk_lr},
    ]
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    shares = dataset.shares(args.data)
    per_epoch = len(batches(*train, args.thread_batch, args.command_batch, random.Random(0), shares))
    warmup = int(0.05 * args.epochs * per_epoch)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / max(1, warmup)))
    precision = (
        torch.autocast("cuda", dtype=torch.bfloat16) if where == "cuda" else torch.autocast("cpu", enabled=False)
    )
    collate = {"thread": inputs.thread_batch, "command": inputs.command_batch}

    def run(planned, learn: bool, decaying: bool = False) -> dict:
        """Losses over the batches, learning from them when `learn`. A decay lowers the learning
        rate from where the schedule holds it to zero over these batches, instead of following the schedule."""
        totals, counts = {}, {}
        reader.train(learn)
        held = [group["lr"] for group in optimizer.param_groups]
        for step, (kind, group) in enumerate(planned):
            if decaying:
                for rate, param_group in zip(held, optimizer.param_groups, strict=True):
                    param_group["lr"] = rate * (1 - step / len(planned))
            batch = collate[kind](group, pad).to(where)
            with torch.set_grad_enabled(learn), precision:
                found = losses(reader, kind, batch)
            if learn:
                loss = sum(found.values())
                if not torch.isfinite(loss):
                    raise RuntimeError(f"the loss is no longer finite: {found}")
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for group in groups for p in group["params"]], 1.0)
                optimizer.step()
                if not decaying:
                    schedule.step()
            for name, value in found.items():
                totals[name] = totals.get(name, 0.0) + value.item()
                counts[name] = counts.get(name, 0) + 1
        if decaying:
            for rate, param_group in zip(held, optimizer.param_groups, strict=True):
                param_group["lr"] = rate
        return {name: round(totals[name] / counts[name], 4) for name in totals}

    def decayed(epoch: int) -> tuple[dict, dict]:
        """A short decay branched off the held run: its training losses, and the state to go back to."""
        kept = {
            "reader": {name: value.detach().clone() for name, value in reader.state_dict().items()},
            "optimizer": copy.deepcopy(optimizer.state_dict()),
        }
        planned = batches(*train, args.thread_batch, args.command_batch, random.Random(f"decay {epoch}"), shares)
        return run(planned[: max(1, round(args.decay_share * per_epoch))], learn=True, decaying=True), kept

    args.output.mkdir(parents=True, exist_ok=True)
    settings = {key: str(value) for key, value in vars(args).items()}
    best = None
    with (args.output / "log.jsonl").open("a") as log:
        for epoch in range(1, args.epochs + 1):
            started = time.perf_counter()
            planned = batches(*train, args.thread_batch, args.command_batch, rng, shares)
            entry = {"epoch": epoch, "train": run(planned, True)}
            entry["decay"], kept = decayed(epoch)
            tested = run(batches(*test, args.thread_batch, args.command_batch, random.Random(0)), learn=False)
            entry |= {"test": tested, "test total": round(sum(tested.values()), 4)}
            entry["seconds"] = round(time.perf_counter() - started)
            print(json.dumps(entry), flush=True)
            log.write(json.dumps(entry) + "\n")
            save(reader, tokenizer, args.output / f"epoch-{epoch}", {**settings, "epoch": epoch})
            if best is None or entry["test total"] < best:
                best = entry["test total"]
                save(reader, tokenizer, args.output / "best", {**settings, "epoch": epoch})
            if epoch == args.epochs:
                save(reader, tokenizer, args.output / "final", {**settings, "epoch": epoch})
            reader.load_state_dict(kept["reader"])
            optimizer.load_state_dict(kept["optimizer"])


if __name__ == "__main__":
    main()
