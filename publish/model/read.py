# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "transformers>=5", "safetensors", "anyascii==0.3.3", "huggingface_hub"]
# ///
"""Read a message to the bot, or a scheduling thread, with Lichess4545/chess-league-bot-nlu.

    uv run read.py command "ping me when knight42 starts"
    uv run read.py thread "white: thu 19:00 or sat 14:00?" "black: sat works"

MODEL=<folder> reads with a local copy of the model instead of downloading it.
"""

import os
import sys

from huggingface_hub import snapshot_download

USAGE = 'usage: read.py command "<message>" | read.py thread "white: <message>" "black: <message>" ...'


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ("command", "thread"):
        sys.exit(USAGE)
    mode, given = sys.argv[1], sys.argv[2:]
    messages = [tuple(each.split(": ", 1)) for each in given]
    if mode == "thread" and any(len(each) != 2 or each[0] not in ("white", "black") for each in messages):
        sys.exit('each message of a thread starts with "white: " or "black: "')
    where = os.environ.get("MODEL") or snapshot_download("Lichess4545/chess-league-bot-nlu")
    sys.path.insert(0, where)
    from reader import Reader

    reader = Reader.load(where)
    if mode == "command":
        found = reader.read_command(" ".join(given))
        command, chance = found["commands"][0]
        print(f"{command} ({chance:.0%})" + (f", event {found['event']}" if found["event"] else ""))
        for label, text, _ in found["arguments"]:
            print(f"  {label}: {text}")
        return
    times, events = reader.read_thread(messages)
    for time in times:
        speaker, text = messages[time.message]
        print(f"{speaker} writes {text[time.start : time.end]!r}: {time.role}")
    for event in events:
        offer = messages[event.time.message][1][event.time.start : event.time.end]
        verb = "accepts" if event.kind == "accept" else "takes back"
        print(f"{messages[event.message][0]} {verb} {offer!r}")


main()
