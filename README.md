# chess-league-bot-nlu

Language understanding for a chess league chat bot (built for
[arboters](https://github.com/Lichess4545/arboters)): which command a message
asks for, and which game times players offer, accept and withdraw in a pairing
thread. This repository makes the synthetic training data, fine-tunes a small
[GLiNER2](https://github.com/fastino-ai/GLiNER2) model on it, and evaluates the
whole pipeline the way the bot would run it.

The data and the model are published on Hugging Face;
nothing generated is committed here:

- dataset: [Lichess4545/chess-league-bot-nlu](https://huggingface.co/datasets/Lichess4545/chess-league-bot-nlu)
- model: [Lichess4545/chess-league-bot-nlu](https://huggingface.co/Lichess4545/chess-league-bot-nlu)

## Two tasks handled by the model

- **Commands.** A message addressed to the bot (with the bot's name stripped)
  is classified as one of the bot's commands or `none` (`chess_league_bot_nlu/intents.toml`),
  with its arguments as spans (`player`, `league`, `round`, `board`, `team`,
  `subscription id`, `phrase`, `recipient`) and, for notification requests,
  which event to notify about.
- **Scheduling.** The model reads a pairing thread as plain `white: …` /
  `black: …` lines and marks `offered time`, `accepted time` and
  `withdrawn time` spans. A message keeps its own line breaks, so a list of
  times continues on lines with no speaker; only a message's first line starts
  with one. Acceptances and withdrawals are marked on the offer they refer
  to, even when the reply names no time ("deal 👍"). `withdrawn time` means
  the sender takes the time back: from the player who offered it, the offer is
  withdrawn; from the player who accepted it, the acceptance is retracted and
  the offer stands again. One time on several days ("21:00 fri or sat") is one
  mark and one offer per day. Code turns the marks into UTC times
  (`resolve.py`), replays the thread message by message as the bot would
  (`transcript.py`), and applies the league rules (`rules.py`, `rules.toml`):
  which agreements to announce, and whether each player made the required
  scheduling effort. Records are schema 2; schema-1 records still load.

## Layout

```
chess_league_bot_nlu/
  rules.py, rules.toml     the league rules: the only implementation, shared by generation and evaluation
  records.py               the dataset's record formats (facts only; outcomes are computed)
  resolve.py               time expressions to UTC
  transcript.py            what the model reads, gold spans, and marks back into rule facts
  labels.py, intents.toml  the model's labels: commands, events, argument and scheduling spans
  words.py                 whether spans sit on the model's word boundaries
  dataset.py               loading and splitting generated records
  train.py, evaluate.py    fine-tuning, and end-to-end evaluation as the bot would run
  device.py                where torch runs (CHESS_LEAGUE_BOT_NLU_DEVICE)
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
    qa.py                  QA for a run made before QA was part of generation
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
uv sync                   # generation and tests
uv sync --group train     # training and evaluation too (GLiNER2's training code)
uv run pytest
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
once (default 4); each record notes the model's name, read from the server or
set with `LLM_MODEL`. `scripts/serve-gemma.sh` starts a server with llama.cpp;
it needs `llama-server` on your `PATH` and downloads the model on first use.

```sh
scripts/serve-gemma.sh                                                  # Gemma 4 12B on the GPU (see the script for options)
uv run python -m chess_league_bot_nlu.generate.run configs/default.toml --preview 2  # try every prompt framing; writes nothing
uv run python -m chess_league_bot_nlu.generate.run configs/default.toml              # the full run, resumable
```

A run writes to the config's output directory: the kept `threads.jsonl` and
`commands.jsonl`, `rejects.jsonl` (records that didn't pass, and why),
`errors.jsonl` (jobs that broke, which a rerun tries again), `attempted.txt`
(jobs that finished, which a rerun skips), `stats.json` and `review.md`.
Delete the directory to start over.

Labels never come from the model. Code plans each thread (who offers which
times, who accepts or withdraws what, when) and each command message (intent,
arguments, phrasing style). The model writes the words around placeholders;
code fills in the planned times and arguments, checks the result, and keeps it
only if an independent reading agrees with the plan, and QA, shown the labels,
agrees that each offer is what the text says.

## Training and evaluation

```sh
uv run --group train python -m chess_league_bot_nlu.evaluate data/generated fastino/gliner2.5-small-v1
uv run --group train python -m chess_league_bot_nlu.train data/generated fastino/gliner2.5-small-v1 runs/small
uv run --group train python -m chess_league_bot_nlu.evaluate data/generated runs/small/best
uv run --group train python -m chess_league_bot_nlu.evaluate data/other runs/small/best --all  # every record, for a separate test set
```

## Work on it

If you want to change something, open a pull request against `main`. If you
use AI tools, follow [AI_POLICY.md](AI_POLICY.md).

Most of this repository's code was written with Claude Code, with a human
reviewing, directing and testing it. The data generator runs a local LLM
(Gemma 4 12B) with no AI assistant in the loop.

## License

Code: AGPL-3.0-or-later. Data and model licenses are stated on their
Hugging Face cards.
