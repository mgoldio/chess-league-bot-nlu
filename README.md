# chess-league-bot-nlu

Language understanding for a chess league chat bot (built for
[arboters](https://github.com/Lichess4545/arboters)): which command a message
asks for, and which game times players offer, accept and withdraw in a pairing
thread. This repository makes the synthetic training data and trains and
evaluates the model: [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)
with small fixed heads in place of its word head. What the bot does with the
model's answers (resolving times, announcing agreements, league rules) lives in
arboters.

The data and the model are published on Hugging Face;
nothing generated is committed here:

- dataset: [Lichess4545/chess-league-bot-nlu](https://huggingface.co/datasets/Lichess4545/chess-league-bot-nlu)
- model: [Lichess4545/chess-league-bot-nlu](https://huggingface.co/Lichess4545/chess-league-bot-nlu)

## Two tasks handled by the model

- **Commands.** A message addressed to the bot (with the bot's name stripped)
  is classified as one of the bot's commands or `none` (`chess_league_bot_nlu/intents.toml`),
  with its arguments as spans (`player`, `league`, `round`, `team`,
  `subscription id`) and, for notification requests,
  which event to notify about.
- **Scheduling.** The model reads a pairing thread as `white: …` / `black: …`
  lines, tags each time written in it, and answers at each message, for every
  time written so far, what the message does with it: offers it, restates it
  while accepting, accepts it, takes it back, mentions it as something other
  than an offer, or nothing. Taking back from the player who offered a time
  withdraws it; from the player who accepted it, it retracts the acceptance.
  `outcome.py` turns the answers into what the thread comes to after each
  message: the offers standing, those withdrawn, and the accepted ones.

## What the model reads

The model reads each message in ASCII: every character is transliterated with
[anyascii](https://github.com/anyascii/anyascii) ("–" to "-", "can’t" to
"can't", "👍" to ":thumbsup:"), and its answers are mapped back to spans of
the original text. Whatever runs the model must do the same, with the same
tables: arboters uses the Rust crate `any_ascii` at the version matching the
pinned Python `anyascii`. It must also strip Zulip quote blocks, which the
training data never has.

## Records

A thread is stored on no particular week. Its times are days after the Monday
of its pairing's week and a UTC time of day (`"+3T19:00:00"`), and every date
its messages write is a placeholder for a day and how it is written
(`{+3:b} {+3:o}` is "Nov 26th" in the week of Monday 23 November). Weekday
names are written out, so a thread moves only by whole weeks. Loaded, a thread
is put on a week of the year; training shows each thread on several.

```json
{"id": "t000004",
 "messages": [{"id": "m2", "speaker": "white", "sent": "+1T06:09:00",
               "text": "would {+6:b} {+6:o} 22:00-04:00 work for you?"}, ...],
 "mentions": [{"id": "T1", "message": "m2", "start": 6, "end": 31, "role": "offered",
               "times": [{"start": "+6T22:00:00", "end": "+7T04:00:00", "open": false}], "accepts": null}, ...],
 "acceptances": [{"message": "m4", "offer": "T2", "time": null}],
 "withdrawals": [], "retractions": [],
 "plot": "question_counter"}
```

Mention offsets count characters of the stored text, placeholders included.
Placeholder fields: `d`/`dd` day of the month, `m`/`mm` month number, `o`
ordinal ("4th"), `b`/`B` month name ("Nov", "November"), `bl`/`Bl` the same in
lower case. A command is its text, intent, event, argument spans, and the
`style` (and, for non-commands, `none_kind`) it was written in.

## Layout

```
chess_league_bot_nlu/
  records.py               the dataset's record formats
  dates.py                 dates in message text as placeholders for days
  outcome.py               what a thread's times and events come to after each message
  labels.py, intents.toml  the commands, their events and their arguments
  dataset.py               loading and splitting generated records
  evaluate.py              scores of a trained reader on a test split
  device.py                where torch runs (CHESS_LEAGUE_BOT_NLU_DEVICE)
  heads/                   the reader: inputs, model, training, reading
  publish.py               stages the dataset and a model for Hugging Face
  resolve.py               reads time expressions, for checking what the generator writes
  generate/
    content/               everything the generator says or samples (see its README)
    catalog.py             loads and checks the content files
    plan.py                plans threads in code; labels come from here
    command_plan.py        plans messages to the bot in code
    names.py               player names as they appear in chat
    wording.py             writes planned times the way players do, checked by the resolver
    realize.py             asks the model to write around placeholders, then fills them in
    placeholders.py        the {T1} placeholders the writer leaves for code to fill
    noise.py, check.py     typing noise outside placeholders; mechanical checks
    verify.py              an independent reading that must agree with the plan, and QA of the labels
    gold.py                records from plans
    llm.py                 the model server client
    errors.py              why a record is dropped
    run.py                 the resumable driver
    report.py              stats.json and review.md for a run
configs/                   generation runs
scripts/serve-gemma.sh     a local model server for generation
tests/
```

## Setup

With [uv](https://docs.astral.sh/uv/):

```sh
uv sync                   # generation
uv sync --group train     # training, evaluation and the tests
uv run --group train pytest
```

If the default torch build doesn't match your CUDA driver, install a matching
one from [pytorch.org](https://pytorch.org/get-started/locally/).

Training and evaluation use a GPU when one is available. Set
`CHESS_LEAGUE_BOT_NLU_DEVICE=cpu` to stay on the CPU and `CHESS_LEAGUE_BOT_NLU_THREADS` to limit CPU
threads.

## Generating data

Data is written by a local model, [Gemma 4](https://ai.google.dev/gemma)
(Apache 2.0). The generator talks to any OpenAI-compatible server at `LLM_URL`
(default `http://127.0.0.1:8080/v1`), sending up to `LLM_PARALLEL` requests at
once (default 4); `generation.jsonl` notes the model's name for each record, read
from the server or set with `LLM_MODEL`. `scripts/serve-gemma.sh` starts a server with llama.cpp;
it needs `llama-server` on your `PATH` and downloads the model on first use.

```sh
scripts/serve-gemma.sh                                                  # Gemma 4 12B on the GPU (see the script for options)
uv run python -m chess_league_bot_nlu.generate.run configs/v3.toml --preview 2  # try every prompt framing; writes nothing
uv run python -m chess_league_bot_nlu.generate.run configs/v3.toml              # the full run, resumable
```

A run writes to the config's output directory: the kept `threads.jsonl` and
`commands.jsonl`, `generation.jsonl` (each kept record's framing, model, and for
a thread its personas, league and round dates), `rejects.jsonl` (records that didn't pass, and why),
`errors.jsonl` (jobs that broke, which a rerun tries again), `attempted.txt`
(jobs that finished, which a rerun skips), `stats.json` and `review.md`.
Delete the directory to start over.

A run can be aimed at particular situations and joined to an earlier set: in
its config, `[threads.weights]` and `[commands.weights]` set how often named
plots and commands are picked, `others` scales the weights of the rest, and
`[output] prefix` starts every id with a prefix of its own
(`configs/gaps-1.toml`). Training and evaluation read several directories
joined with `+` as one dataset (`data/v3+data/gaps-1`).

Labels never come from the model. Code plans each thread (who offers which
times, who accepts or withdraws what, when) and each command message (intent,
arguments, phrasing style). The model writes the words around placeholders;
code fills in the planned times and arguments, checks the result, and keeps it
only if an independent reading agrees with the plan, and QA, shown the labels,
agrees that each offer is what the text says.

## Training and evaluation

```sh
uv run --group train python -m chess_league_bot_nlu.heads.train data/v3+data/gaps-1 runs/smol
uv run --group train python -m chess_league_bot_nlu.evaluate data/v3+data/gaps-1 runs/smol/best
uv run --group train python -m chess_league_bot_nlu.evaluate data/other runs/smol/best --all  # every record, for a separate test set
```

A checkpoint is `model.safetensors` (the trunk's weights under `trunk.`, the
heads' by their own names), `config.json`, the tokenizer, and `training.json`.

## Publishing

`publish/` holds the Hugging Face cards and `reader.py`, the self-contained
reader shipped with the model. To stage a model and the dataset it was trained
on into git-ignored folders, then upload them:

```sh
uv run --group train python -m chess_league_bot_nlu.evaluate <its data> runs/smol/best --out runs/smol/eval
uv run --group train python -m chess_league_bot_nlu.publish runs/smol/best runs/smol/eval/<report>.json
hf upload Lichess4545/chess-league-bot-nlu runs/publish/model . --delete "*"
hf upload Lichess4545/chess-league-bot-nlu data/publish/dataset . --repo-type dataset --delete "*"
```

## Adding a command

A new command is new training data and a fresh fine-tune; nothing already
generated is regenerated.

1. **Describe it** in `chess_league_bot_nlu/intents.toml`: a `description`
   of what the message asks for, `unlike` to set it apart from
   the commands it is easily mistaken for (and add the same note to theirs),
   and `events` if it has any. A new kind of argument also goes in `ARGUMENTS`
   in `labels.py`.
2. **Tell the generator how to plan it** in
   `generate/content/commands.toml`: its `[arguments]` (and their values in
   `vocabulary.toml`), a few `[angles]` (ways people ask for it), and
   `[reads_like]` examples for the checker. If it is easily mistaken for other
   commands, add it to a `contrast_groups` entry with them, so they are
   written side by side.
3. **Make a config aimed at it**, like `configs/gaps-1.toml`: a new output
   directory and `prefix`, `[commands.weights]` with the new command high and
   `others` low (some of the others, especially its contrast group, keep the
   model telling them apart), and `[threads] count = 0`. A few thousand
   messages is a good start. Preview, then run it:

   ```sh
   uv run python -m chess_league_bot_nlu.generate.run configs/new-command.toml --preview 2
   uv run python -m chess_league_bot_nlu.generate.run configs/new-command.toml
   ```

4. **Retrain from the base model on every set**, joined with `+`. The
   heads' sizes follow the labels, so an earlier checkpoint can't be
   continued; a run takes minutes.

   ```sh
   uv run --group train python -m chess_league_bot_nlu.heads.train data/v3+data/gaps-1+data/new-command runs/smol
   ```

Older sets never asked for the new command, so a message in them that now
would (most likely one of the `none` near misses) is labelled as something
else. If the command is close to one of `none_kinds`, check for those before
training, and drop or relabel them.

## Work on it

If you want to change something, open a pull request against `main`. If you
use AI tools, follow [AI_POLICY.md](AI_POLICY.md).

Most of this repository's code was written with Claude Code, with a human
reviewing, directing and testing it. The data generator runs a local LLM
(Gemma 4 12B) with no AI assistant in the loop.

## License

Code: AGPL-3.0-or-later. Data and model licenses are stated on their
Hugging Face cards.
