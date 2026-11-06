# Generator content

Everything the generator says or samples lives here; the code in `generate/`
only reads it. Editing these files changes the data without touching code.
`uv run pytest tests/test_catalog.py` checks every file loads and that the
files agree with each other (every plot beat has wording, every argument list
exists, every framing has its settings), so run it after an edit.

| File | What it holds |
| --- | --- |
| `plots.toml` | Storylines for scheduling threads, as lists of beats |
| `schedule.toml` | League rounds, message pacing, when offered games start, zones |
| `beats.toml` | What the writer is asked to say for each kind of beat |
| `personas.toml` | Who the players are, and how long their messages run |
| `time_formats.toml` | How players write days, clock times, ranges |
| `commands.toml` | How messages to the bot are planned: arguments, styles, non-commands |
| `vocabulary.toml` | Usernames, teams, leagues, rounds and other argument values |
| `layout.toml` | How messages are laid out in code: lists of times, paragraphs, @-mentions |
| `noise.toml` | Typing noise added in code |
| `prompts/<kind>/` | Prompts for threads and for commands (below) |

The bot's intents themselves (their descriptions and events) are in
`chess_league_bot_nlu/intents.toml`, because the model uses them too.

## Prompts

Each of `prompts/threads/` and `prompts/commands/` has:

- `framings/*.md`: the system prompts the writer is given, one per framing. A
  new file here is a new framing; give it settings in `framings.toml`.
- `framings.toml`: per framing, how often messages get typing noise and the
  writer's temperature range.
- `task.md`: the request itself, with slots in `{braces}`.
- `verify.md` and `verify-system.md`: the independent reading that must agree
  with the plan.

The `.md` prompts are filled with Python's `str.format`, so a literal brace is
written doubled: `{{T1}}` shows the writer `{T1}`. Slots:

- `threads/task.md`: `{league}`, `{round}`, `{published}`, `{deadline}`, `{plan}`, `{count}`
- `threads/verify.md`: `{transcript}`
- `commands/task.md`: `{count}`, `{lines}`
- `commands/verify.md`: `{intents}`, `{notify}`, `{events}`, `{messages}`

## TOML

In TOML every key after a `[table]` header belongs to that table, so each file
puts its top-level settings before its first table.
