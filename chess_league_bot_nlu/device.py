"""Choose where torch runs. Call `select()` before anything imports torch.

CHESS_LEAGUE_BOT_NLU_DEVICE is "auto" (the default: a GPU when one is available), "cpu"
or "cuda". CHESS_LEAGUE_BOT_NLU_THREADS limits CPU threads.
"""

from __future__ import annotations

import os


def select() -> str:
    choice = os.environ.get("CHESS_LEAGUE_BOT_NLU_DEVICE", "auto")
    if choice not in ("auto", "cpu", "cuda"):
        raise RuntimeError(f"CHESS_LEAGUE_BOT_NLU_DEVICE must be auto, cpu or cuda, not {choice!r}")
    if choice == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import torch

    if threads := os.environ.get("CHESS_LEAGUE_BOT_NLU_THREADS"):
        torch.set_num_threads(int(threads))
    available = torch.cuda.is_available()
    if choice == "cuda" and not available:
        raise RuntimeError("CHESS_LEAGUE_BOT_NLU_DEVICE=cuda but no GPU is available")
    if choice == "cpu" and available:
        raise RuntimeError(
            "the GPU should be hidden when CHESS_LEAGUE_BOT_NLU_DEVICE=cpu; was torch imported before select()?"
        )
    if choice == "auto":
        return "cuda" if available else "cpu"
    return choice
