"""Stage the dataset and a trained reader for Hugging Face, with their cards from `publish/`.

    uv run --group train python -m chess_league_bot_nlu.publish runs/smol/best runs/smol/eval/eval.json

The dataset is the one the model was trained on (its `training.json` names it), split as training
split it; the model is its weights, config, tokenizer, `labels.json`, `reader.py`, a `training.json`
that describes the training without local paths, and its evaluation on the test split. Both are
written to git-ignored folders (`data/publish/dataset`, `runs/publish/model`) to upload from.
"""

from __future__ import annotations

import json
import shutil
import sys
from collections import Counter
from pathlib import Path

from . import dataset
from .heads import inputs
from .labels import ARGUMENTS, INTENTS

CARDS = Path(__file__).parent.parent / "publish"
DATASET = Path("data/publish/dataset")
MODEL = Path("runs/publish/model")
HUB_DATASET = "Lichess4545/chess-league-bot-nlu"
HUB_MODEL = "Lichess4545/chess-league-bot-nlu"


def prefix(record_id: str) -> str:
    """The generation run a record came from, by its id: "g1-" and so on, or "" for the general set."""
    head, dash, _ = record_id.partition("-")
    return f"{head}-" if dash and head.startswith("g") else ""


def stage_dataset(data: str) -> None:
    shutil.rmtree(DATASET, ignore_errors=True)
    threads, commands = dataset.threads(data), dataset.commands(data)
    rows = {row["id"]: row for row in dataset.rows(data, "threads.jsonl")}
    published = {}
    for kind, records in (("threads", threads), ("commands", commands)):
        (DATASET / kind).mkdir(parents=True)
        for split, chosen in zip(("train", "test"), dataset.split(records), strict=True):
            lines = [rows[record.id] if kind == "threads" else record.to_dict() for record in chosen]
            (DATASET / kind / f"{split}.jsonl").write_text(
                "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines)
            )
            published[f"{kind}/{split}"] = chosen
    kept_threads = published["threads/train"] + published["threads/test"]
    kept_commands = published["commands/train"] + published["commands/test"]
    stats = {
        "splits": {name: len(records) for name, records in published.items()},
        "runs": {
            run or "general": {
                "threads": sum(prefix(each.id) == run for each in kept_threads),
                "commands": sum(prefix(each.id) == run for each in kept_commands),
            }
            for run in sorted({prefix(each.id) for each in kept_threads + kept_commands})
        },
        "plots": dict(Counter(each.plot for each in kept_threads).most_common()),
        "intents": dict(Counter(each.intent for each in kept_commands).most_common()),
        "styles": dict(Counter(each.style for each in kept_commands).most_common()),
    }
    (DATASET / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    shutil.copy(CARDS / "dataset" / "README.md", DATASET / "README.md")
    print(json.dumps({key: stats[key] for key in ("splits", "runs")}, indent=1))


def training(settings: dict) -> dict:
    """How the model was trained, from its local settings, naming the dataset rather than paths."""
    shares = {}
    for directory, share in dataset.directories(settings["data"]):
        records = dataset.rows(directory, "threads.jsonl") + dataset.rows(directory, "commands.jsonl")
        for run in sorted({prefix(row["id"]) for row in records}):
            shares[run or "general"] = share
    return {
        "base_model": settings["trunk"],
        "dataset": HUB_DATASET,
        "split": "the dataset's train split; checkpoints compared by loss on its test split",
        "share_per_epoch": shares,
        "trunk": "fine-tuned whole",
        "optimizer": "AdamW, weight decay 0.01, gradients clipped to norm 1.0, bfloat16 autocast",
        "learning_rate": {"heads": float(settings["heads_lr"]), "trunk": float(settings["trunk_lr"])},
        "schedule": "linear warm-up over 5% of the run, then held; after each epoch a decay to zero over "
        f"{float(settings['decay_share']):.0%} of an epoch is branched off and saved, and the run goes on from before it",
        "epochs": int(settings["epochs"]),
        "checkpoint": f"after epoch {settings['epoch']}, with its decay",
        "batch": {"threads": int(settings["thread_batch"]), "commands": int(settings["command_batch"])},
        "thread_variants": f"each thread on its own week and {settings['weeks']} others, with white and black "
        "swapped or not; one variant shown per epoch",
        "seed": int(settings["seed"]),
    }


def stage_model(where: Path, evaluation: Path) -> None:
    shutil.rmtree(MODEL, ignore_errors=True)
    MODEL.mkdir(parents=True)
    for name in ("model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json"):
        shutil.copy(where / name, MODEL / name)
    shutil.copy(CARDS / "model" / "reader.py", MODEL / "reader.py")
    shutil.copy(CARDS / "model" / "README.md", MODEL / "README.md")
    settings = json.loads((where / "training.json").read_text())
    (MODEL / "training.json").write_text(json.dumps(training(settings), indent=2) + "\n")
    labels = {
        "commands": {
            key: {"description": intent.description, "events": list(intent.events)} for key, intent in INTENTS.items()
        },
        "intents": list(inputs.INTENT_KEYS),
        "command_events": list(inputs.COMMAND_EVENTS),
        "arguments": list(ARGUMENTS),
        "argument_tags": list(inputs.ARGUMENT_TAGS),
        "time_tags": list(inputs.TIME_TAGS),
        "answers": list(inputs.ANSWERS),
        "lookahead": inputs.LOOKAHEAD,
        "command_prefix": inputs.COMMAND_PREFIX,
        "trailing": inputs.TRAILING,
        "anyascii": "0.3.3",
    }
    (MODEL / "labels.json").write_text(json.dumps(labels, indent=2, ensure_ascii=False) + "\n")
    report = json.loads(evaluation.read_text())
    for task in ("commands", "threads"):
        report[task].pop("rows", None)
    report.pop("latency", None)
    report |= {"model": HUB_MODEL, "data": f"{HUB_DATASET} (dataset), test split"}
    (MODEL / "evaluation").mkdir()
    (MODEL / "evaluation" / "test.json").write_text(json.dumps(report, indent=2) + "\n")
    print("model files:", sorted(str(path.relative_to(MODEL)) for path in MODEL.rglob("*") if path.is_file()))


def main():
    where, evaluation = Path(sys.argv[1]), Path(sys.argv[2])
    settings = json.loads((where / "training.json").read_text())
    stage_dataset("+".join(str(directory) for directory, _ in dataset.directories(settings["data"])))
    stage_model(where, evaluation)


if __name__ == "__main__":
    main()
