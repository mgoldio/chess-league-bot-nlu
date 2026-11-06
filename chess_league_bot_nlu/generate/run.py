"""Generate a dataset: plan, write, check, verify, and keep what passes.

    uv run python -m chess_league_bot_nlu.generate.run configs/default.toml              # full run, resumable
    uv run python -m chess_league_bot_nlu.generate.run configs/default.toml --preview 2  # 2 samples per prompt framing; writes nothing

The output directory gets `threads.jsonl` and `commands.jsonl` (kept records),
`rejects.jsonl` (records that didn't pass, and why), `errors.jsonl` (jobs that
broke, for example because the model server didn't answer), `attempted.txt`
(jobs that finished), `stats.json` and `review.md` (a random sample to read).

A rerun continues where the last one stopped: finished jobs are skipped, and
jobs that broke are tried again. Delete the directory to start over. If several
jobs in a row break, the run stops, since the model server is probably down.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import threading
import tomllib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from .. import words
from ..records import Thread
from . import catalog, check, command_plan, gold, llm, noise, plan, realize, report, verify
from .errors import Rejected

CONFIG_KEYS = {
    "output": {"directory"},
    "threads": {"count", "seed", "attempts"},
    "commands": {"count", "seed", "batch"},
    "review": {"sample"},
}


BROKEN_IN_A_ROW = 8


@dataclass
class Outcome:
    """What a job produced. A job that `broke` didn't finish, and runs again next time."""

    kept: list = field(default_factory=list)
    rejects: list[dict] = field(default_factory=list)
    broke: str = ""


def provenance(framing: str) -> dict:
    return {"framing": framing, "model": llm.model()}


def generate_thread(record_id: str, rng: random.Random, framing_name: str | None = None) -> Thread:
    framings = catalog.prompts("threads").framings
    framing = framings[framing_name or rng.choice(list(framings))]
    thread_plan = plan.thread(rng)
    system, user, drafts = realize.thread_prompt(thread_plan, framing.name, rng)
    templates = realize.write(system, user, len(drafts), round(rng.uniform(*framing.temperature), 2))
    texts = []
    for index, (template, draft) in enumerate(zip(templates, drafts, strict=True), start=1):
        template = draft.laid_out(template)
        found = check.problems(
            template, draft.placeholders, draft.words, optional=draft.optional, allow_weekday=draft.allow_weekday
        )
        if found:
            raise Rejected(f"message {index}: {'; '.join(found)}: {template!r}")
        if rng.random() < framing.noise_chance:
            template = noise.add_noise(template, rng)
        text, spans = realize.fill(draft.edited(template), draft.values)
        if words.misaligned(text, spans.values()):
            raise Rejected(f"message {index}: a time is joined to the text around it: {text!r}")
        texts.append((text, draft.mentions(spans)))
    thread = gold.thread_record(record_id, thread_plan, texts, provenance(framing.name))
    misread = check.misread(thread)
    if misread:
        raise Rejected(f"misread in context: {'; '.join(misread)}")
    if unclear := check.ambiguous(thread):
        raise Rejected(f"ambiguous acceptance: {'; '.join(unclear)}")
    disagreements = verify.thread_problems(thread) or verify.label_problems(thread)
    if disagreements:
        raise Rejected("; ".join(disagreements))
    return thread


def generate_commands(batch_id: str, size: int, rng: random.Random, framing_name: str | None = None) -> Outcome:
    framings = catalog.prompts("commands").framings
    framing = framings[framing_name or rng.choice(list(framings))]
    plans = [command_plan.command(rng) for _ in range(size)]
    system, user, drafts = realize.command_prompt(plans, framing.name)
    templates = realize.write(system, user, len(drafts), round(rng.uniform(*framing.temperature), 2))
    outcome, candidates = Outcome(), []
    for index, (planned, draft, template) in enumerate(zip(plans, drafts, templates, strict=True)):
        record_id = f"{batch_id}-{index}"
        found = check.problems(template, list(draft.values), draft.words, times=False)
        if found:
            outcome.rejects.append({"id": record_id, "reason": "; ".join(found), "text": template})
            continue
        if rng.random() < framing.noise_chance:
            template = noise.add_noise(template, rng)
        text, spans = realize.fill(template, draft.values)
        if words.misaligned(text, spans.values()):
            outcome.rejects.append(
                {"id": record_id, "reason": "an argument is joined to the text around it", "text": text}
            )
            continue
        candidates.append(gold.command_record(record_id, planned, text, spans, provenance(framing.name)))
    problems = verify.command_problems(candidates) if candidates else []
    for command, found in zip(candidates, problems, strict=True):
        if found:
            outcome.rejects.append({"id": command.id, "reason": "; ".join(found), "text": command.text})
        else:
            outcome.kept.append(command)
    return outcome


def thread_job(record_id: str, seed: int, index: int, attempts: int) -> Outcome:
    outcome = Outcome()
    for attempt in range(attempts):
        try:
            outcome.kept.append(generate_thread(record_id, random.Random(f"{seed}:{index}:{attempt}")))
            return outcome
        except Rejected as error:
            outcome.rejects.append({"id": record_id, "attempt": attempt, "reason": reason(error)})
    return outcome


def commands_job(batch_id: str, seed: int, batch: int, size: int) -> Outcome:
    try:
        return generate_commands(batch_id, size, random.Random(f"{seed}:{batch}"))
    except Rejected as error:
        return Outcome(rejects=[{"id": batch_id, "reason": reason(error)}])


def reason(error: Rejected) -> str:
    return str(error) if type(error) is Rejected else f"{type(error).__name__}: {error}"


class Output:
    """The run's output directory, written to by many jobs at once."""

    def __init__(self, directory: Path):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        for path in [*directory.glob("*.jsonl"), directory / "attempted.txt"]:
            if path.exists():
                trim_partial_line(path)
        attempted = directory / "attempted.txt"
        self.attempted = set(attempted.read_text().split()) if attempted.exists() else set()

    def finish(self, job_id: str, outcome: Outcome) -> None:
        """Write a job's results; mark it done only if it didn't break."""
        with self.lock:
            for record in outcome.kept:
                self.append("threads.jsonl" if isinstance(record, Thread) else "commands.jsonl", record.to_dict())
            for reject in outcome.rejects:
                self.append("rejects.jsonl", reject)
            if outcome.broke:
                self.append("errors.jsonl", {"id": job_id, "error": outcome.broke})
                return
            with (self.directory / "attempted.txt").open("a") as file:
                file.write(job_id + "\n")

    def append(self, name: str, row: dict) -> None:
        with (self.directory / name).open("a") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def trim_partial_line(path: Path) -> None:
    """Drop a last line an interrupted run left half-written, so new rows start on their own line."""
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        path.write_bytes(data[: data.rfind(b"\n") + 1])


def safely(work, *arguments) -> Outcome:
    """Run a job. A rejected record is a normal outcome; anything else (the model server not
    answering, or a bug) marks the job broken, so it runs again next time, rather than ending
    the run for every other job."""
    try:
        return work(*arguments)
    except Exception as error:
        return Outcome(broke=f"{type(error).__name__}: {error}")


def run(config: dict) -> None:
    output = Output(Path(config["output"]["directory"]))
    threads, commands = config["threads"], config["commands"]
    jobs = {}
    for index in range(threads["count"]):
        jobs[f"t{index:06d}"] = (thread_job, f"t{index:06d}", threads["seed"], index, threads["attempts"])
    for batch in range(math.ceil(commands["count"] / commands["batch"])):
        jobs[f"c{batch:05d}"] = (commands_job, f"c{batch:05d}", commands["seed"], batch, commands["batch"])
    pending = {job_id: job for job_id, job in jobs.items() if job_id not in output.attempted}
    print(f"{len(pending)} of {len(jobs)} jobs to run", flush=True)
    broken_in_a_row = 0
    with ThreadPoolExecutor(llm.PARALLEL) as pool:
        futures = {pool.submit(safely, *job): job_id for job_id, job in pending.items()}
        try:
            for done, future in enumerate(as_completed(futures), start=1):
                outcome = future.result()
                output.finish(futures[future], outcome)
                broken_in_a_row = broken_in_a_row + 1 if outcome.broke else 0
                if broken_in_a_row >= BROKEN_IN_A_ROW:
                    raise SystemExit(
                        f"{broken_in_a_row} jobs in a row broke, last with {outcome.broke}; is the model server up?"
                    )
                if done % 25 == 0:
                    print(f"{done}/{len(futures)} jobs done", flush=True)
        except (KeyboardInterrupt, SystemExit):
            pool.shutdown(cancel_futures=True)
            raise
    report.report(output.directory, config["review"]["sample"])


def preview(per_framing: int) -> None:
    for framing in catalog.prompts("threads").framings:
        for index in range(per_framing):
            try:
                thread = generate_thread(
                    f"preview-{framing}-{index}", random.Random(f"preview:{framing}:{index}"), framing
                )
            except Rejected as error:
                print(f"\n== {framing} #{index}: REJECTED {type(error).__name__}: {str(error)[:300]}")
                continue
            print(f"\n== {framing} #{index} · {thread.provenance['plot']}")
            for message in thread.messages:
                print(f"  {message.speaker:5}: " + message.text.replace("\n", "\n         "))
            print("  times:", ", ".join(f"{thread.text_of(mention)!r}:{mention.role}" for mention in thread.mentions))
            print("\n".join(f"  {line[2:]}" for line in report.marks(thread)))
    for framing in catalog.prompts("commands").framings:
        try:
            outcome = generate_commands(
                f"preview-{framing}", 5 * per_framing, random.Random(f"preview:{framing}"), framing
            )
        except Rejected as error:
            print(f"\n== commands {framing}: FAILED {error}")
            continue
        print(f"\n== commands {framing}: kept {len(outcome.kept)}")
        for command in outcome.kept:
            print(f"  {command.intent:20} {command.text!r}")
        for reject in outcome.rejects:
            print(f"  REJECTED {reject['reason'][:160]}")


def load_config(path: Path) -> dict:
    with path.open("rb") as file:
        config = tomllib.load(file)
    for section, keys in CONFIG_KEYS.items():
        if set(config.get(section, {})) != keys:
            raise ValueError(f"{path} [{section}] needs exactly {sorted(keys)}, has {sorted(config.get(section, {}))}")
    if set(config) - set(CONFIG_KEYS):
        raise ValueError(f"{path}: unknown sections {sorted(set(config) - set(CONFIG_KEYS))}")
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--preview", type=int, metavar="N", help="try N samples per prompt framing; write nothing")
    args = parser.parse_args()
    config = load_config(args.config)
    catalog.check()
    if args.preview:
        preview(args.preview)
    else:
        run(config)


if __name__ == "__main__":
    main()
