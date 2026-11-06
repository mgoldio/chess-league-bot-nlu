"""Evaluate a GLiNER2 checkpoint, zero-shot or fine-tuned, on a dataset's test split.

Commands: which command (or none), the event for notification requests, and
argument spans. Scheduling, end to end as the bot would run it:

- agreements: replay each thread one message at a time; an agreement is
  announced at the message after which it first exists, and must be the right
  time at the right message;
- effort: on the part of each thread inside the rule's window, does each player
  meet the rule;
- spans: offered, accepted and withdrawn marks on each full thread.

    uv run --group train python -m chess_league_bot_nlu.evaluate data/generated fastino/gliner2.5-small-v1
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from . import dataset, device, records, rules
from .labels import ARGUMENTS, EVENT_BY_LABEL, INTENT_BY_LABEL, SCHEDULING, command_schema, thread_schema
from .records import PLAYERS
from .transcript import Replay, gold_spans, transcript, within_window

RULES = rules.load()


@dataclass
class Counts:
    """Hits, predictions and expected items, for precision and recall."""

    hit: int = 0
    predicted: int = 0
    expected: int = 0

    def add(self, predicted: set | Counter, expected: set | Counter) -> None:
        if isinstance(predicted, Counter):
            self.hit += sum((predicted & expected).values())
            self.predicted += sum(predicted.values())
            self.expected += sum(expected.values())
        else:
            self.hit += len(predicted & expected)
            self.predicted += len(predicted)
            self.expected += len(expected)

    def scores(self) -> dict:
        """None where there is nothing to score: a label that neither the data nor the model has."""
        if not self.predicted and not self.expected:
            return {"precision": None, "recall": None, "f1": None}
        precision = self.hit / self.predicted if self.predicted else 0.0
        recall = self.hit / self.expected if self.expected else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}


def latency(seconds: list[float]) -> dict:
    ordered = sorted(seconds) or [0.0]
    return {
        "median_ms": round(1000 * statistics.median(ordered), 1),
        "p95_ms": round(1000 * ordered[int(0.95 * (len(ordered) - 1))], 1),
    }


def calibration_error(confidences: list[tuple[float, bool]], bins: int = 10) -> float:
    """Expected calibration error of the top label: how far confidence is from accuracy."""
    if not confidences:
        return 0.0
    total = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        bucket = [
            (confidence, right)
            for confidence, right in confidences
            if low < confidence <= high or (index == 0 and confidence == 0)
        ]
        if bucket:
            total += len(bucket) * abs(statistics.mean(c for c, _ in bucket) - statistics.mean(r for _, r in bucket))
    return round(total / len(confidences), 3)


class Model:
    def __init__(self, name: str, where: str):
        from gliner2 import AutoExtractor

        self.extractor = AutoExtractor.from_pretrained(name, map_location=where)
        self.seconds: list[float] = []

    def schema(self):
        return self.extractor.create_schema()

    def extract(self, text: str, schema, **options) -> dict:
        if not text:
            return {}
        started = time.perf_counter()
        result = self.extractor.extract(text, schema, **options)
        self.seconds.append(time.perf_counter() - started)
        return result


def texts(found: list[dict]) -> Counter:
    return Counter(value["text"].strip().lower() for value in found)


def commands(model: Model, test: list[records.Command]) -> dict:
    schema = command_schema(model.schema())
    per_intent: dict[str, Counts] = defaultdict(Counts)
    per_argument: dict[str, Counts] = defaultdict(Counts)
    confusions, confidences, rows = Counter(), [], []
    events = Counts()
    for command in test:
        result = model.extract(command.text, schema, include_confidence=True, include_spans=True)
        ranked = result.get("command") or [{"label": "none", "confidence": 0.0}]
        guess = INTENT_BY_LABEL.get(ranked[0]["label"], "none")
        confidences.append((ranked[0]["confidence"], guess == command.intent))
        for intent in {guess, command.intent}:
            per_intent[intent].add(
                {command.id} if guess == intent else set(), {command.id} if command.intent == intent else set()
            )
        if guess != command.intent:
            confusions[(command.intent, guess)] += 1
        if command.event:
            event = result.get("event") or {}
            label = event.get("label") if isinstance(event, dict) else event
            events.add({EVENT_BY_LABEL.get(label)}, {command.event})
        found = result.get("entities", {})
        for label in ARGUMENTS:
            gold = Counter(command.text[a.start : a.end].strip().lower() for a in command.arguments if a.label == label)
            per_argument[label].add(texts(found.get(label, [])), gold)
        rows.append({"id": command.id, "text": command.text, "expected": command.intent, "predicted": ranked[:3]})
    correct = sum(counts.hit for counts in per_intent.values())
    return {
        "examples": len(test),
        "accuracy": round(correct / len(test), 3) if test else None,
        "none": per_intent["none"].scores(),
        "intents": {intent: counts.scores() for intent, counts in sorted(per_intent.items())},
        "event_accuracy": round(events.hit / events.expected, 3) if events.expected else None,
        "top_confusions": [[expected, guess, count] for (expected, guess), count in confusions.most_common(8)],
        "arguments": {label: counts.scores() for label, counts in per_argument.items()},
        "calibration_error": calibration_error(confidences),
        "rows": rows,
    }


def replay(model: Model, schema, thread: records.Thread, upto: int) -> Replay:
    """The model's marks on the first `upto` messages, fed one message at a time."""
    state = Replay(thread)
    for step in range(1, upto + 1):
        found = model.extract(transcript(thread, step).text, schema, include_spans=True)
        state.step(step, found.get("entities", {}))
    return state


def scheduling(model: Model, test: list[records.Thread]) -> dict:
    schema = thread_schema(model.schema())
    agreements = Counts()
    verdicts = Counter()
    spans: dict[str, Counts] = defaultdict(Counts)
    rows = []
    for thread in test:
        rule = RULES[thread.league]
        published = records.parse_time(thread.pairing_published)
        gold_offers, gold_acceptances = records.thread_facts(thread)
        expected = {(agreement.sent, agreement.time) for agreement in rules.agreements(gold_offers, gold_acceptances)}

        full = replay(model, schema, thread, len(thread.messages))
        announced = {(agreement.sent, agreement.time) for agreement in rules.agreements(*full.facts())}
        agreements.add(announced, expected)

        window = replay(model, schema, thread, within_window(thread, rule))
        efforts = {}
        for player in PLAYERS:
            gold = rules.meets_rule(rule, published, player, gold_offers, gold_acceptances)
            predicted = rules.meets_rule(rule, published, player, *window.facts())
            verdicts[(gold, predicted)] += 1
            efforts[player] = {"gold": gold, "predicted": predicted}

        script = transcript(thread)
        gold = gold_spans(thread, script)
        found = model.extract(script.text, schema, include_spans=True).get("entities", {})
        for label in SCHEDULING:
            spans[label].add(
                {(value["start"], value["end"]) for value in found.get(label, [])},
                {(value["start"], value["end"]) for value in gold[label]},
            )
        rows.append(
            {
                "thread": thread.id,
                "announced": sorted(f"{sent:%a %H:%M} -> {when:%a %d %b %H:%M}" for sent, when in announced),
                "expected": sorted(f"{sent:%a %H:%M} -> {when:%a %d %b %H:%M}" for sent, when in expected),
                "effort": efforts,
            }
        )
    judged = sum(verdicts.values())
    return {
        "threads": len(test),
        "agreements": {
            **agreements.scores(),
            "correct": agreements.hit,
            "announced": agreements.predicted,
            "expected": agreements.expected,
        },
        "effort_accuracy": round((verdicts[(True, True)] + verdicts[(False, False)]) / judged, 3) if judged else None,
        "nudges": {
            "correct": verdicts[(False, False)],
            "wrongly_nudged": verdicts[(True, False)],
            "missed": verdicts[(False, True)],
        },
        "spans": {label: counts.scores() for label, counts in spans.items()},
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("model")
    parser.add_argument("--out", type=Path, default=Path("runs"))
    parser.add_argument("--all", action="store_true", help="score every record, for data the model never trained on")
    args = parser.parse_args()

    where = device.select()
    model = Model(args.model, where)
    command_test, thread_test = dataset.commands(args.data), dataset.threads(args.data)
    if not args.all:
        _, command_test = dataset.split(command_test)
        _, thread_test = dataset.split(thread_test)
    report = {
        "model": args.model,
        "data": str(args.data),
        "device": where,
        "commands": commands(model, command_test),
        "scheduling": scheduling(model, thread_test),
        "latency": latency(model.seconds),
    }
    args.out.mkdir(exist_ok=True)
    path = args.out / f"eval_{args.model.strip('/').replace('/', '_')}.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    for task in ("commands", "scheduling"):
        print(task, json.dumps({key: value for key, value in report[task].items() if key not in ("rows", "intents")}))
    print("latency", report["latency"])
    print("wrote", path)


if __name__ == "__main__":
    main()
