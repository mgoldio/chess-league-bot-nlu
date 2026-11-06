"""Fine-tune a GLiNER2 checkpoint on a generated dataset.

Commands train as a choice of command (and, for notification requests, of
event) plus argument spans. Scheduling threads train on every prefix of each
thread, since the bot rereads a thread at each new message.

    uv run --group train python -m chess_league_bot_nlu.train data/generated fastino/gliner2.5-small-v1 runs/small

The best checkpoint is chosen by eval loss on the test split, which leaks a
little into model choice; report final numbers on data the run never saw.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import dataset, device, words
from .labels import ARGUMENTS, INTENT_BY_LABEL, INTENTS
from .records import Command, Thread
from .transcript import gold_spans, transcript


def command_row(command: Command) -> dict:
    spans: dict[str, list[dict]] = {label: [] for label in ARGUMENTS}
    for argument in command.arguments:
        spans[argument.label].append(
            {"text": command.text[argument.start : argument.end], "start": argument.start, "end": argument.end}
        )
    intent = INTENTS[command.intent]
    classifications = [{"task": "command", "labels": list(INTENT_BY_LABEL), "true_label": [intent.label]}]
    if command.event:
        events = INTENTS[command.intent].events
        classifications.append(
            {"task": "event", "labels": list(events.values()), "true_label": [events[command.event]]}
        )
    return {"input": command.text, "output": {"entities": spans, "classifications": classifications}}


def thread_rows(thread: Thread) -> list[dict]:
    rows = []
    for upto in range(1, len(thread.messages) + 1):
        script = transcript(thread, upto)
        rows.append({"input": script.text, "output": {"entities": gold_spans(thread, script)}})
    return rows


def rows(commands: list[Command], threads: list[Thread]) -> list[dict]:
    """Training rows. A row with a span off the model's word boundaries is left out whole,
    so its times are never trained as non-times."""
    built = [command_row(command) for command in commands] + [row for thread in threads for row in thread_rows(thread)]
    kept = [row for row in built if not misaligned(row)]
    if len(kept) < len(built):
        print(f"left out {len(built) - len(kept)} rows with spans off word boundaries")
    return kept


def misaligned(row: dict) -> bool:
    spans = [(value["start"], value["end"]) for values in row["output"]["entities"].values() for value in values]
    return bool(words.misaligned(row["input"], spans))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("model")
    parser.add_argument("output")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    where = device.select()
    from gliner2 import AutoExtractor
    from gliner2.training.trainer import ExtractorTrainer, TrainingConfig

    command_train, command_test = dataset.split(dataset.commands(args.data))
    thread_train, thread_test = dataset.split(dataset.threads(args.data))
    train, test = rows(command_train, thread_train), rows(command_test, thread_test)
    print(f"train {len(train)} rows, test {len(test)}, on {where}")

    model = AutoExtractor.from_pretrained(args.model, map_location=where)
    config = TrainingConfig(
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        encoder_lr=1e-5,
        task_lr=5e-4,
        eval_strategy="epoch",
        logging_steps=10,
        save_total_limit=1,
        fp16=False,
        bf16=False,
        num_workers=1,
    )
    ExtractorTrainer(model, config).train(train_data=train, eval_data=test)


if __name__ == "__main__":
    main()
