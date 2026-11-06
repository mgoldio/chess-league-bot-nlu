"""Generate a dataset: plan, write, check, verify, and keep what passes.

    uv run python -m chess_league_bot_nlu.generate.run configs/v3.toml              # full run, resumable
    uv run python -m chess_league_bot_nlu.generate.run configs/v3.toml --preview 2  # 2 samples per prompt framing; writes nothing

The output directory gets `threads.jsonl` and `commands.jsonl` (kept records),
`generation.jsonl` (how each kept record was made: its framing, the model, and for a thread its
players' personas and its round's dates), `rejects.jsonl` (records that didn't pass, and why), `errors.jsonl` (jobs that
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

from ..labels import INTENTS
from ..records import Thread, format_time
from . import catalog, check, command_plan, gold, llm, noise, plan, realize, report, verify
from .errors import Rejected

CONFIG_KEYS = {
    "output": {"directory"},
    "threads": {"count", "seed", "attempts"},
    "commands": {"count", "seed", "batch"},
    "review": {"sample"},
}
# Keys a config may leave out: an id prefix, so a run's records can join another's, and the weights
# that steer a run towards some plots or commands (`Weights`).
OPTIONAL_KEYS = {"output": {"prefix"}, "threads": {"weights", "others"}, "commands": {"weights", "others"}}


@dataclass(frozen=True)
class Weights:
    """How often to pick each named plot (or command), in place of the catalog's weight, and how
    much to scale the catalog's weights of the rest ("others = 0" picks only the named ones)."""

    named: dict[str, float] = field(default_factory=dict)
    others: float = 1.0

    @classmethod
    def of(cls, section: dict) -> Weights:
        return cls(dict(section.get("weights", {})), section.get("others", 1.0))

    def __iter__(self):
        return iter((self.named, self.others))


BROKEN_IN_A_ROW = 8


@dataclass
class Outcome:
    """What a job produced. A job that `broke` didn't finish, and runs again next time."""

    kept: list = field(default_factory=list)
    generation: list[dict] = field(default_factory=list)
    rejects: list[dict] = field(default_factory=list)
    broke: str = ""


def generation(record_id: str, framing: str, **details) -> dict:
    return {"id": record_id, "framing": framing, "model": llm.model(), **details}


def generate_thread(
    record_id: str, rng: random.Random, framing_name: str | None = None, weights: Weights | None = None
) -> tuple[Thread, dict]:
    framings = catalog.prompts("threads").framings
    framing = framings[framing_name or rng.choice(list(framings))]
    thread_plan = plan.thread(rng, *(weights or Weights()))
    try:
        thread = written_thread(record_id, thread_plan, framing, rng)
    except Rejected as error:
        error.plot = thread_plan.plot
        raise
    personas = {player: persona.description for player, persona in thread_plan.personas.items()}
    made = generation(
        record_id,
        framing.name,
        personas=personas,
        league=thread_plan.league.name,
        pairing_published=format_time(thread_plan.published),
        deadline=format_time(thread_plan.deadline),
    )
    return thread, made


def written_thread(record_id: str, thread_plan: plan.ThreadPlan, framing, rng: random.Random) -> Thread:
    system, user, drafts = realize.thread_prompt(thread_plan, framing.name, rng)
    templates = realize.write(system, user, len(drafts), round(rng.uniform(*framing.temperature), 2))
    texts = []
    for index, (template, draft) in enumerate(zip(templates, drafts, strict=True), start=1):
        template = draft.laid_out(template)
        found = check.problems(
            template,
            draft.placeholders,
            draft.words,
            optional=draft.optional,
            allow_weekday=draft.allow_weekday,
            allow_relative=draft.allow_relative,
        )
        if found:
            raise Rejected(f"message {index}: {'; '.join(found)}: {template!r}")
        if rng.random() < framing.noise_chance:
            template = noise.add_noise(template, rng)
        text, spans = realize.fill(draft.edited(template), draft.values)
        texts.append((text, draft.mentions(spans)))
    thread = gold.thread_record(record_id, thread_plan, texts)
    window = (thread_plan.published, thread_plan.deadline)
    misread = check.misread(thread, window)
    if misread:
        raise Rejected(f"misread in context: {'; '.join(misread)}")
    if unclear := check.ambiguous(thread):
        raise Rejected(f"ambiguous acceptance: {'; '.join(unclear)}")
    disagreements = verify.thread_problems(thread, window) or verify.label_problems(thread)
    if disagreements:
        error = Rejected("; ".join(disagreements))
        error.messages = [f"{message.speaker}: {message.text}" for message in thread.messages]
        raise error
    return thread


def generate_commands(
    batch_id: str, size: int, rng: random.Random, framing_name: str | None = None, weights: Weights | None = None
) -> Outcome:
    framings = catalog.prompts("commands").framings
    framing = framings[framing_name or rng.choice(list(framings))]
    plans = command_plan.batch(rng, size, *(weights or Weights()))
    system, user, drafts = realize.command_prompt(plans, framing.name)
    templates = realize.write(system, user, len(drafts), round(rng.uniform(*framing.temperature), 2))
    outcome, candidates = Outcome(), []
    for index, (planned, draft, template) in enumerate(zip(plans, drafts, templates, strict=True)):
        record_id = f"{batch_id}-{index}"
        planned_as = {"intent": planned.intent.key, "style": planned.style.name, "contrast": planned.contrast}
        if planned.none_kind:
            planned_as["none_kind"] = planned.none_kind.kind
        found = check.problems(template, list(draft.values), draft.words, times=False)
        if found:
            outcome.rejects.append({"id": record_id, **planned_as, "reason": "; ".join(found), "text": template})
            continue
        if rng.random() < framing.noise_chance:
            template = noise.add_noise(template, rng)
        text, spans = realize.fill(template, draft.values)
        made = generation(record_id, framing.name, contrast=planned.contrast, angle=planned.angle)
        candidates.append((gold.command_record(record_id, planned, text, spans), planned_as, made))
    problems = verify.command_problems([command for command, _, _ in candidates]) if candidates else []
    for (command, planned_as, made), found in zip(candidates, problems, strict=True):
        if found:
            outcome.rejects.append({"id": command.id, **planned_as, "reason": "; ".join(found), "text": command.text})
        else:
            outcome.kept.append(command)
            outcome.generation.append(made)
    return outcome


def thread_job(record_id: str, seed: int, index: int, attempts: int, weights: Weights) -> Outcome:
    outcome = Outcome()
    for attempt in range(attempts):
        try:
            thread, made = generate_thread(record_id, random.Random(f"{seed}:{index}:{attempt}"), weights=weights)
            outcome.kept.append(thread)
            outcome.generation.append(made)
            return outcome
        except Rejected as error:
            reject = {
                "id": record_id,
                "attempt": attempt,
                "plot": getattr(error, "plot", None),
                "reason": reason(error),
            }
            if hasattr(error, "messages"):
                reject["messages"] = error.messages
            outcome.rejects.append(reject)
    return outcome


def commands_job(batch_id: str, seed: int, batch: int, size: int, weights: Weights) -> Outcome:
    try:
        return generate_commands(batch_id, size, random.Random(f"{seed}:{batch}"), weights=weights)
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
            for made in outcome.generation:
                self.append("generation.jsonl", made)
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
    prefix = config["output"].get("prefix", "")
    plots, intents = Weights.of(threads), Weights.of(commands)
    jobs = {}
    for index in range(threads["count"]):
        job_id = f"{prefix}t{index:06d}"
        jobs[job_id] = (thread_job, job_id, threads["seed"], index, threads["attempts"], plots)
    for batch in range(math.ceil(commands["count"] / commands["batch"])):
        job_id = f"{prefix}c{batch:05d}"
        jobs[job_id] = (commands_job, job_id, commands["seed"], batch, commands["batch"], intents)
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
                thread, _ = generate_thread(
                    f"preview-{framing}-{index}", random.Random(f"preview:{framing}:{index}"), framing
                )
            except Rejected as error:
                print(f"\n== {framing} #{index}: REJECTED {type(error).__name__}: {str(error)[:300]}")
                continue
            print(f"\n== {framing} #{index} · {thread.plot}")
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
        given = set(config.get(section, {}))
        if not keys <= given <= keys | OPTIONAL_KEYS.get(section, set()):
            raise ValueError(
                f"{path} [{section}] needs {sorted(keys)} and may have {sorted(OPTIONAL_KEYS.get(section, set()))}, has {sorted(given)}"
            )
    known_plots = {each.name for each in catalog.plots().plots}
    known_intents = set(INTENTS) - {"none"}
    for section, known in (("threads", known_plots), ("commands", known_intents)):
        if unknown := set(config[section].get("weights", {})) - known:
            raise ValueError(f"{path} [{section}.weights]: unknown {sorted(unknown)}")
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
