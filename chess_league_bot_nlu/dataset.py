"""Load generated records and split them into train and test sets.

A dataset directory holds `threads.jsonl` and `commands.jsonl`; several
directories joined with "+" ("data/v3+data/gaps-1") read as one dataset. A
directory may carry the share of its records training shows each epoch
("data/gaps-1@0.5"), so a set aimed at a few situations doesn't crowd out the rest. The split is
deterministic, so a record never moves between sets when more data is
generated: a thread by its id, a command by its text with case, punctuation
and spacing ignored. Commands repeat ("faq", "tell me a joke"), so each text
is kept once and always falls on the same side.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
from pathlib import Path

from .records import Command, Thread

TEST_SHARE = 0.3


def is_test(key: str) -> bool:
    return hashlib.sha256(key.encode()).digest()[0] / 256 < TEST_SHARE


def plain(text: str) -> str:
    return " ".join(re.sub(r"[^\w@]+", " ", text.lower()).split())


def split_key(record: Command | Thread) -> str:
    return plain(record.text) if isinstance(record, Command) else record.id


def read_jsonl(path: Path) -> list[dict]:
    """The rows of a JSON-lines file; a last line cut off by an interrupted write is skipped."""
    if not path.exists():
        return []
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    rows = [json.loads(line) for line in lines[:-1]]
    if lines:
        with contextlib.suppress(json.JSONDecodeError):
            rows.append(json.loads(lines[-1]))
    return rows


def unique(rows: list[dict]) -> list[dict]:
    """Rows with each id once: a run interrupted mid-write can repeat a record when it resumes."""
    kept, seen = [], set()
    for row in rows:
        if row["id"] not in seen:
            seen.add(row["id"])
            kept.append(row)
    return kept


def directories(data: Path | str) -> list[tuple[Path, float]]:
    """Each directory, and the share of its records to show each epoch."""
    found = []
    for each in str(data).split("+"):
        path, _, share = each.partition("@")
        found.append((Path(path), float(share or 1)))
    return found


def rows(data: Path | str, name: str) -> list[dict]:
    return unique([row for directory, _ in directories(data) for row in read_jsonl(directory / name)])


def shares(data: Path | str) -> dict[str, float]:
    """The share of each record shown each epoch, by id, for records from a directory with one below 1."""
    return {
        row["id"]: share
        for directory, share in directories(data)
        if share < 1
        for name in ("threads.jsonl", "commands.jsonl")
        for row in read_jsonl(directory / name)
    }


def threads(data: Path | str) -> list[Thread]:
    return [Thread.from_dict(row) for row in rows(data, "threads.jsonl")]


def commands(data: Path | str) -> list[Command]:
    return [Command.from_dict(row) for row in rows(data, "commands.jsonl")]


def split(records: list) -> tuple[list, list]:
    """Train and test records, each command text kept once. A text written for different commands
    or events is left out: it can't teach either."""
    labels: dict[str, set] = {}
    for record in records:
        if isinstance(record, Command):
            labels.setdefault(split_key(record), set()).add((record.intent, record.event))
    train, test, seen = [], [], set()
    for record in records:
        key = split_key(record)
        if isinstance(record, Command) and (key in seen or len(labels[key]) > 1):
            continue
        seen.add(key)
        (test if is_test(key) else train).append(record)
    return train, test
