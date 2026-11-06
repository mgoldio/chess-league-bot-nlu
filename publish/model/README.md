---
license: agpl-3.0
language:
  - en
base_model: HuggingFaceTB/SmolLM2-135M
datasets:
  - Lichess4545/chess-league-bot-nlu
library_name: pytorch
pipeline_tag: token-classification
tags:
  - chess
  - scheduling
  - intent-classification
  - token-classification
  - safetensors
---

# Chess league bot NLU

[SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M), fine-tuned
whole, with small trained heads in place of its word head, for
[arboters](https://github.com/Lichess4545/arboters), the Lichess4545 league bot
for Zulip. It does two things:

- **commands**: which command a message to the bot asks for (or `none`), the
  event for notification requests, and argument spans;
- **scheduling**: in a conversation between two paired players, which times
  were written, and what each message does with them: offers, accepts, or
  takes back.

It was trained on
[Lichess4545/chess-league-bot-nlu](https://huggingface.co/datasets/Lichess4545/chess-league-bot-nlu),
which is fully synthetic. Code:
[mgoldio/chess-league-bot-nlu](https://github.com/mgoldio/chess-league-bot-nlu).

## Files

- `reader.py`: the whole model as code, about 200 lines of PyTorch: load it,
  read a thread or a command. It is the reference for everything below.
- `model.safetensors`: every weight: the trunk's under `trunk.` (a
  `LlamaModel`, 30 layers, hidden size 576), the heads' by their own names.
- `config.json`: the trunk's Llama config under `trunk`, and the heads'
  LayerNorm epsilon.
- `tokenizer.json`, `tokenizer_config.json`: SmolLM2's tokenizer.
- `labels.json`: the order of every output, and the constants below.
- `training.json`: how it was trained.
- `evaluation/test.json`: scores on the dataset's test split.

## Use

```python
import sys

from huggingface_hub import snapshot_download

where = snapshot_download("Lichess4545/chess-league-bot-nlu")
sys.path.insert(0, where)
from reader import Reader

reader = Reader.load(where)
reader.read_command("ping me when knight42 starts")
# {'commands': [('tell', 0.99...), ...], 'event': 'starts', 'arguments': [('player', 'knight42', (13, 21))]}
times, events = reader.read_thread([("white", "can we do thu 19:00 or sat 14:00?"), ("black", "sat works")])
```

It needs `torch`, `transformers` 5 or later (earlier versions read the
trunk's RoPE settings wrongly), `safetensors` and `anyascii==0.3.3`.

## Input

**Normalize first.** Every message is read in ASCII: each character is
transliterated on its own with [anyascii](https://github.com/anyascii/anyascii)
0.3.3 ("–" to "-", "can’t" to "can't", "👍" to ":thumbsup:"). Keep, for each
original character, where its transliteration starts, so spans can be mapped
back to the original text. In Rust, use the `any_ascii` crate, version 0.3.3.
Strip Zulip quote blocks before reading; the model never saw them.

**Tokenize each message on its own** and concatenate the token ids, with no
special tokens:

- a thread: one entry per message, `"white: " + text + "\n"` or
  `"black: " + text + "\n"`, in order, with no names or timestamps. A message
  keeps its own line breaks; only the `"\n"` added at its end closes it;
- a command: `"@bot " + text + "\n"`, the bot mention removed from `text`.

A token's characters are its offsets in that string, less the prefix, and
less any leading spaces: the tokenizer joins a space to the start of a word
(`" can"`), and a span must not start on it.

## Heads

Run the trunk with every layer's hidden states, as transformers returns them:
the embeddings, the outputs of layers 1 to 29, and layer 30's output after the
final RMSNorm, 31 in all. `S`, the states the heads read, is
`LayerNorm(Σ softmax(mix)[l] · hidden[l])`.

- **Token tags.** For each token `t`, `ahead(t)` is `min(t + 4, last token of
  its message)`. Tags are `argmax(W · [S[t], S[ahead(t)]])`: `time_tags`
  (outside, begins, inside) for threads, `arguments` (outside, then
  begins/inside for each argument) for commands. A run is a `begins` and the
  `inside` tokens of the same kind after it; a stray `inside` starts one too.
  A run's characters are its first token's start to its last token's end,
  less any of `,.!?;` the tokenizer joined to its end, widened to whole words
  (letters, digits and `_`), then mapped back to the original text.
- **Commands.** At the last token (the `"\n"`), `intents` gives one score per
  entry of `intents` in `labels.json` (softmax for confidences), and, when the
  chosen command has events, `command_events` gives one of `command_events`
  ("none", or an event key).
- **Threads.** Every time tagged gets a key `K = span([S[first token], S[last
  token]])`, a linear layer. At each message's last token, `Q = S[last
  token]`, and the message answers for every time written in it or before it:
  for a time with `same` (written in this message: 0/1) and `own` (written by
  this message's sender: 0/1), `k = K + same[same] + own[own]` (two
  embeddings), and the answer is `argmax(pointer([Q, k, Q·k]))`, an MLP
  (linear, GELU, linear) over the `answers`: none, offered, accepted, taken
  back, other.
  - In its own message, the answer is the time's role: `offered`, `accepted`
    (restated while accepting) or `other` (anything else).
  - In a later message, `accepted` accepts that time, `taken back` takes it
    back, and anything else does nothing.

A thread is read in one pass: the trunk is causal, tags look ahead only within
their message, and each message answers at its own last token, so a message's
answers don't depend on the messages after it. Reread the whole thread when a
message arrives.

## What the answers come to

A time a player offers stands until a later message of theirs takes it back. A
later message of the other player's accepting a standing offer accepts it, and
the time it restated, if any, is the pick ("16:00 then" inside a range).
Taking it back again from the player who accepted retracts the acceptance;
from the player who offered, it withdraws the offer and the acceptance with
it. Anything else changes nothing. (`chess_league_bot_nlu/outcome.py`.)

The model doesn't resolve times; arboters reads each span itself. Relative and
vague times ("tomorrow", "saturday evening") are never offers, though an
acceptance may point at an offer with one.

## Evaluation

On the dataset's test split (4,051 threads, 5,998 commands), which
training never saw; it was used to pick the checkpoint (the epoch with the
lowest test loss), so these numbers are slightly optimistic. Full report:
`evaluation/test.json`.

| metric | score |
|---|---|
| Threads whose outcome is exactly right after every message | 86.4% |
| Messages after which the outcome is exactly right | 92.7% |
| Standing offers F1, at each thread's end | 0.985 |
| Accepted offers F1, at each thread's end | 0.921 |
| Withdrawn offers F1, at each thread's end | 0.941 |
| Times tagged F1 (exact spans) | 0.983 |
| Command accuracy | 98.8% (calibration error 0.007) |
| Event accuracy | 99.3% |
| Argument F1 (text, case ignored) | player 0.97, league 0.98, subscription id 0.99, round 0.85, team 0.83 |

An outcome is the offers standing, the offers withdrawn, and the accepted
offers with their picks, as exact spans.

## Limitations

- **Synthetic data only.** Real players write differently and performance may
  not match.
- **Worst performing subsets**
  - **Taking back one of several offers** (e.g. "forget thursday")
  - **Insufficiently specific offers sometimes read as valid** (e.g. 
    "anything before 15:00 thu" is not specific, but will sometimes be counted)
  - **A picked time written before the day** (e.g. "17:00 sat it is")
  - **Commands can sometimes be read wrong** (e.g. "put me down for next round"
    incorrectly read as unavailable)

## Training

From SmolLM2-135M, the whole trunk at learning rate 3e-5 and the heads at
1e-3 (AdamW), warmed up and held, with a decay to zero branched off after each
epoch; this is the checkpoint after epoch 4 of 5. Each epoch shows each thread
on one of 8 variants (4 weeks of the year, white and black swapped or not).
The dataset's targeted runs `g1-` to `g4-` are shown at half share per epoch;
the general set and `g5-`, at full. See `training.json`.

## License

AGPL-3.0-or-later. The base model,
[SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M), is by
Hugging Face and licensed under Apache 2.0.
