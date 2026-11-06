---
license: cc0-1.0
annotations_creators:
  - machine-generated
language_creators:
  - machine-generated
language:
  - en
pretty_name: Chess league bot NLU
size_categories:
  - 10K<n<100K
task_categories:
  - token-classification
  - text-classification
tags:
  - synthetic
  - chess
  - scheduling
  - intent-classification
configs:
  - config_name: threads
    data_files:
      - split: train
        path: threads/train.jsonl
      - split: test
        path: threads/test.jsonl
  - config_name: commands
    data_files:
      - split: train
        path: commands/train.jsonl
      - split: test
        path: commands/test.jsonl
---

# Chess league bot NLU

Synthetic training data for the language understanding of
[arboters](https://github.com/Lichess4545/arboters), the Lichess4545 league bot
for Zulip. It has two parts:

- **threads**: 13,471 scheduling conversations between two paired players
  (`white` and `black`), with every time written in them marked, and which
  times were offered, accepted and withdrawn.
- **commands**: 19,769 distinct messages to the bot, each with the command it
  asks for (or `none`), the event for notification requests, and argument
  spans.

The model trained on it is
[Lichess4545/chess-league-bot-nlu](https://huggingface.co/Lichess4545/chess-league-bot-nlu).
The code that generated it, trains and evaluates is at
[mgoldio/chess-league-bot-nlu](https://github.com/mgoldio/chess-league-bot-nlu).

```python
from datasets import load_dataset

threads = load_dataset("Lichess4545/chess-league-bot-nlu", "threads")
commands = load_dataset("Lichess4545/chess-league-bot-nlu", "commands")
```

| | train | test |
|---|---|---|
| threads | 9,420 | 4,051 |
| commands | 13,771 | 5,998 |

The split is about 70/30 and fixed: a thread by a hash of its `id`, a command
by a hash of its text with case, punctuation and spacing ignored. Each command
text appears once, and a text written for two different commands is left out.

## How it was made

Everything is synthetic; no real user messages are included. Each record is
planned in code first: the plot of a thread (who offers which times, who
accepts or withdraws what, and when), or a command with its arguments. A local
LLM, [Gemma 4 12B](https://huggingface.co/google/gemma-4-12B-it) (the QAT
Q4_0 build, run with llama.cpp), then writes
the words around placeholders, and code fills the planned times and arguments
in. So the labels come from the plan, not from the LLM, and every span is
exact.

A record is kept only if it passes mechanical checks (every placeholder used
once, no unplanned times written outside them), a re-read of every labelled
time in its place in the thread, an independent reading by the LLM that must
agree with the plan on what was accepted and withdrawn, an LLM check that every
labelled offer really is an offer, and a check that no acceptance by day name
alone could mean more than one offer.

The data comes from six generation runs. Ids tell them apart:

| Ids | Threads | Commands | Aimed at |
|---|---|---|---|
| `t…`, `c…` | 8,415 | 18,563 | everything: 71 plots, 25 commands |
| `g1-…` | 1,013 | 881 | times picked inside a range, acceptances with no time, declines naming the day, busy times; availability commands |
| `g2-…` | 1,536 | 325 | a picked time written as its clock alone ("16:00 then"), "see you then", withdrawals |
| `g3-…` | 1,002 | 0 | numbered lists, numbers that aren't times (time controls, boards), counters that go through the other player's times again |
| `g4-…` | 976 | 0 | times in a city's zone or "my time", long threads |
| `g5-…` | 529 | 0 | a short decline naming the day with a counter, picks, withdrawals, plain offers and short yeses |

The targeted runs over-represent their situations. The model was trained with
each of `g1` to `g4` shown at half share per epoch. `stats.json` has counts
per run, plot, command and style.

## Threads

```json
{
  "id": "t000004",
  "messages": [
    {"id": "m1", "speaker": "white", "sent": "+0T18:18:00",
     "text": "so sorry for the delay! just saw this. looking forward to the game @**the_najdorf**"},
    {"id": "m2", "speaker": "white", "sent": "+1T06:09:00",
     "text": "my bad for the late reply. would {+6:b} {+6:o} 22:00-04:00 work for you?"},
    {"id": "m3", "speaker": "black", "sent": "+1T06:26:00",
     "text": "sorry just seeing this now. does 0300 on Monday UTC still work for you?"},
    {"id": "m4", "speaker": "white", "sent": "+1T06:53:00", "text": "that works for me, sorry for the wait!"},
    {"id": "m5", "speaker": "black", "sent": "+1T07:17:00", "text": "i'll schedule it. see you thhen!"}
  ],
  "mentions": [
    {"id": "T1", "message": "m2", "start": 33, "end": 58, "role": "offered",
     "times": [{"start": "+6T22:00:00", "end": "+7T04:00:00", "open": false}], "accepts": null},
    {"id": "T2", "message": "m3", "start": 33, "end": 51, "role": "offered",
     "times": [{"start": "+7T03:00:00", "end": null, "open": false}], "accepts": null}
  ],
  "acceptances": [{"message": "m4", "offer": "T2", "time": null}],
  "withdrawals": [],
  "retractions": [],
  "plot": "question_counter"
}
```

A thread is on no particular week. Its times (when each message was `sent`,
and when each mentioned time is) are days after the Monday of the pairing's
week and a UTC time of day (`"+6T22:00:00"`), and every date its
messages write is a placeholder for a day and how it is written: `{+6:b}
{+6:o}` is "Nov 29th" in the week of Monday 23 November. Fields: `d` day of
the month, `m` month number, `o` ordinal ("29th"), `b` short month name
("Nov"), `B` full month name ("November"). Weekday names are
written out, so a thread can be put on any week by moving it a whole number of
weeks: pick a Monday, add each offset to it, and write each placeholder's
field for its day. Mention offsets count characters of the stored text,
placeholders included, so map them through the substitution.

- `mentions` are the times written in the messages. `role` is `offered`,
  `accepted` (a time restated while accepting; `accepts` names the offer) or
  `other` (a time that is neither: one quoted back to turn it down, one a
  player can't make, the round's deadline, one struck through). `times` holds
  the meaning: a start, an end for a range, `open` for "from 18:00 on", and
  one entry per day for a time offered on several days ("21:00 fri or sat").
- `acceptances` name the offer accepted and the message that accepted it, even
  when the reply has no time of its own ("deal 👍"). `time` is the pick when
  the offer was a range or several days.
- `withdrawals` are offers taken back by the player who made them;
  `retractions` are acceptances taken back by the player who made them.
- `plot` is the situation the thread was written for.

A bare clock time with no am/pm is 24-hour. Relative and vague times
("tomorrow", "saturday evening") are never offers on their own, though an
acceptance may point at an offer by them ("saturday evening works").

## Commands

```json
{"id": "c00000-0", "text": "btw what is the lichess rating for @**the_fischer**", "intent": "rating",
 "arguments": [{"label": "player", "start": 35, "end": 51}], "style": "aside"}
```

`intent` is one of the bot's 25 commands or `none`. `event` (when present) is
`starts`, `ends` or `scheduled`, for notification requests that say which.
Argument labels are `player`, `league`, `round`, `team` and `subscription id`.
`style` is how the message was asked to be written, and `none_kind`, for a
message that is no command, what kind of message it is. The text is the
message with the bot mention already removed. Moderators' commands are not
included: they type them exactly, and the bot parses them itself.

## Known issues

- 85% of threads end in an agreement.
- The LLM's excuses don't always match the times ("too late for me" about an
  early time). The labels are unaffected.
- Players in a pairing topic are only ever the two paired players, and all
  messages are in English, as in arboters.

## License

CC0 1.0. The text was written by Gemma 4 12B, which is licensed under
Apache 2.0.
