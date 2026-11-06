"""Evaluate a trained reader (`heads`) on a dataset's test split.

Commands: which command (or none), the event for notification requests, and
argument spans. Threads: the times each message writes, what each message does
with earlier ones, and what that comes to (`outcome`) after every message: the
offers standing, those withdrawn, and the accepted ones with the time each
acceptance restated.

    uv run --group train python -m chess_league_bot_nlu.evaluate data/v3 runs/<model>/best
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from . import dataset, device, records
from .labels import ARGUMENTS
from .outcome import Outcome, gold, outcome


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


def commands(model, test: list[records.Command]) -> dict:
    per_intent: dict[str, Counts] = defaultdict(Counts)
    per_argument: dict[str, Counts] = defaultdict(Counts)
    confusions, confidences, rows = Counter(), [], []
    events = Counts()
    for command in test:
        result = model.read_command(command.text)
        ranked = result["commands"]
        guess, confidence = ranked[0]
        confidences.append((confidence, guess == command.intent))
        for intent in {guess, command.intent}:
            per_intent[intent].add(
                {command.id} if guess == intent else set(), {command.id} if command.intent == intent else set()
            )
        if guess != command.intent:
            confusions[(command.intent, guess)] += 1
        if command.event:
            events.add({result["event"]}, {command.event})
        for label in ARGUMENTS:
            found = Counter(
                command.text[start:end].strip().lower() for each, start, end in result["arguments"] if each == label
            )
            gold = Counter(command.text[a.start : a.end].strip().lower() for a in command.arguments if a.label == label)
            per_argument[label].add(found, gold)
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


def threads(model, test: list[records.Thread]) -> dict:
    """Scores per label over every thread, and the share of threads (and of messages) whose
    outcome comes out exactly right."""
    written, answers = Counts(), Counts()
    scores: dict[str, Counts] = defaultdict(Counts)
    right_threads = right_messages = messages = 0
    plots: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    rows = []
    for thread in test:
        expected_times, expected_events = gold(thread)
        found_times, found_events = model.read_thread(thread)
        written.add({place(time) for time in found_times}, {place(time) for time in expected_times})
        answers.add(set(found_times) | set(found_events), set(expected_times) | set(expected_events))
        right = True
        for upto in range(1, len(thread.messages) + 1):
            want = outcome(thread.messages, expected_times, expected_events, upto)
            got = outcome(thread.messages, found_times, found_events, upto)
            same = parts(got) == parts(want)
            right_messages += same
            messages += 1
            right &= same
        for label, got_part, want_part in zip(LABELS, parts(got), parts(want), strict=True):
            scores[label].add(got_part, want_part)
        right_threads += right
        plots[thread.plot or "unknown"][0] += right
        plots[thread.plot or "unknown"][1] += 1
        if not right:
            rows.append(
                {"thread": thread.id, "plot": thread.plot, "want": shown(thread, want), "got": shown(thread, got)}
            )
    return {
        "threads": len(test),
        "exact_threads": round(right_threads / len(test), 3) if test else None,
        "exact_messages": round(right_messages / messages, 3) if messages else None,
        "times_written": written.scores(),
        "answers": answers.scores(),
        "final": {label: counts.scores() for label, counts in scores.items()},
        "plots": {plot: f"{right}/{total}" for plot, (right, total) in sorted(plots.items())},
        "rows": rows,
    }


LABELS = ("standing", "withdrawn", "accepted", "restated")


def place(time) -> tuple[str, int, int]:
    return time.message, time.start, time.end


def parts(result: Outcome) -> tuple[set, set, set, set]:
    """An outcome as sets of places in the text, so a role read wrongly on the right span still counts."""
    return (
        {place(time) for time in result.standing},
        {place(time) for time in result.withdrawn},
        {place(time) for time in result.accepted},
        {(place(offer), place(restated)) for offer, restated in result.accepted.items() if restated},
    )


def shown(thread: records.Thread, result: Outcome) -> dict:
    def text(time) -> str:
        return thread.message(time.message).text[time.start : time.end]

    return {
        "standing": sorted(text(time) for time in result.standing),
        "withdrawn": sorted(text(time) for time in result.withdrawn),
        "accepted": sorted(
            text(offer) + (f" ({text(restated)})" if restated else "") for offer, restated in result.accepted.items()
        ),
    }


def main():
    from .heads.read import HeadsModel

    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="a dataset directory, or several joined with +")
    parser.add_argument("model")
    parser.add_argument("--out", type=Path, default=Path("runs"))
    parser.add_argument("--all", action="store_true", help="score every record, for data the model never trained on")
    args = parser.parse_args()

    where = device.select()
    model = HeadsModel.saved(args.model, where)
    command_test, thread_test = dataset.commands(args.data), dataset.threads(args.data)
    if not args.all:
        _, command_test = dataset.split(command_test)
        _, thread_test = dataset.split(thread_test)
    report = {"model": args.model, "data": str(args.data), "device": where, "latency": {}}
    for task, score, test in (("commands", commands, command_test), ("threads", threads, thread_test)):
        model.seconds.clear()
        report[task] = score(model, test)
        report["latency"][task] = latency(model.seconds)
    args.out.mkdir(exist_ok=True)
    path = args.out / f"eval_{args.model.strip('/').replace('/', '_')}.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    for task in ("commands", "threads"):
        print(task, json.dumps({key: value for key, value in report[task].items() if key not in ("rows", "intents")}))
    print("latency", report["latency"])
    print("wrote", path)


if __name__ == "__main__":
    main()
