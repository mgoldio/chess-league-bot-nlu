"""What a run made: `stats.json`, with counts and the commonest reasons records were rejected,
and `review.md`, a random sample to read."""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path

from .. import dataset
from ..records import Command, Thread


def report(directory: Path, sample: int) -> None:
    threads, commands = dataset.threads(directory), dataset.commands(directory)
    rejects = dataset.read_jsonl(directory / "rejects.jsonl")
    stats = {
        "threads": len(threads),
        "commands": len(commands),
        "rejects": len(rejects),
        "plots": Counter(thread.provenance.get("plot") for thread in threads),
        "thread_framings": Counter(thread.provenance.get("framing") for thread in threads),
        "intents": Counter(command.intent for command in commands),
        "command_framings": Counter(command.provenance.get("framing") for command in commands),
        "thread_rejects": Counter(
            reason_kind(reject["reason"]) for reject in rejects if reject["id"].startswith("t")
        ).most_common(15),
        "command_rejects": Counter(reject["reason"] for reject in rejects if reject["id"].startswith("c")).most_common(
            15
        ),
    }
    (directory / "stats.json").write_text(json.dumps(stats, indent=2))
    (directory / "review.md").write_text(review(threads, commands, sample))
    print(json.dumps(stats, indent=1, default=str))


def reason_kind(reason: str) -> str:
    """A thread's reject reason without its particulars, so like reasons count together."""
    first = re.sub(r"^Rejected: ", "", reason.split("; ")[0])
    first = re.sub(r": ['\"].*$", "", first)
    first = re.sub(r"(['\"]).*?\1", "…", first)
    first = re.sub(r"\[[^\]]*\]", "[…]", first)
    return re.sub(r"\d+", "N", first)


def review(threads: list[Thread], commands: list[Command], sample: int) -> str:
    rng = random.Random(0)
    lines = ["# Review sample", ""]
    for thread in rng.sample(threads, min(sample, len(threads))):
        lines += [f"## {thread.id} · {thread.provenance.get('plot')} · {thread.provenance.get('framing')}", ""]
        lines += [f"- **{message.speaker}**: {message.text}" for message in thread.messages]
        lines += [
            f"- times: {', '.join(f'{thread.text_of(mention)!r} ({mention.role})' for mention in thread.mentions)}",
            *marks(thread),
            "",
        ]
    lines += ["# Commands", ""]
    lines += [
        f"- `{command.intent}`: {command.text}" for command in rng.sample(commands, min(sample * 3, len(commands)))
    ]
    return "\n".join(lines) + "\n"


def marks(thread: Thread) -> list[str]:
    """Acceptances and withdrawals, which are marked on the offer and may have no words of their own."""
    number = {message.id: index for index, message in enumerate(thread.messages, start=1)}
    offers = {mention.id: thread.text_of(mention) for mention in thread.mentions}
    return [
        f"- message {number[item.message]} {verb} {offers[item.offer]!r}"
        for verb, items in (
            ("accepts", thread.acceptances),
            ("withdraws", thread.withdrawals),
            ("takes back accepting", thread.retractions),
        )
        for item in items
    ]
