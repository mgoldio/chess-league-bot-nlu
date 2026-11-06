"""Load generated records and split them into train and test sets.

A dataset directory holds `threads.jsonl` and `commands.jsonl`. The split is
deterministic by record id, so a record never moves between sets when more
data is generated.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from pathlib import Path

from .records import Command, Thread

TEST_SHARE = 0.3


def is_test(record_id: str) -> bool:
    return hashlib.sha256(record_id.encode()).digest()[0] / 256 < TEST_SHARE


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


def threads(directory: Path) -> list[Thread]:
    return [Thread.from_dict(row) for row in unique(read_jsonl(directory / "threads.jsonl"))]


def commands(directory: Path) -> list[Command]:
    return [Command.from_dict(row) for row in unique(read_jsonl(directory / "commands.jsonl"))]


def split(records: list) -> tuple[list, list]:
    return [record for record in records if not is_test(record.id)], [
        record for record in records if is_test(record.id)
    ]
