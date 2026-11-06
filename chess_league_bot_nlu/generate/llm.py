"""A small client for an OpenAI-compatible chat server, such as llama.cpp's.

LLM_URL is the server's base URL (default http://127.0.0.1:8080/v1), and
LLM_PARALLEL how many requests to send at once (default 4). The model's name,
recorded in each record's provenance, is read from the server; LLM_MODEL
overrides it.
"""

from __future__ import annotations

import json
import os
import re
from functools import cache
from pathlib import PurePath

import requests

from .errors import Rejected

BASE = os.environ.get("LLM_URL", "http://127.0.0.1:8080/v1").rstrip("/")
PARALLEL = int(os.environ.get("LLM_PARALLEL", "4"))


class BadReply(Rejected):
    """The model's reply isn't what was asked for."""


@cache
def model() -> str:
    if name := os.environ.get("LLM_MODEL"):
        return name
    served = requests.get(f"{BASE}/models", timeout=30).json().get("data", [])
    # llama.cpp names a model by its file path; keep only the file's name, not where it lives.
    return PurePath(served[0]["id"]).stem if served else "unknown"


def chat(system: str, user: str, temperature: float, max_tokens: int = 1200, timeout: int = 600) -> str:
    response = requests.post(
        f"{BASE}/chat/completions",
        json={
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature,
            "top_p": 0.95,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def json_from(reply: str) -> list | dict:
    """The first JSON value in a reply, tolerating code fences and chatter around it."""
    fenced = re.search(r"```(?:json)?\s*(.*?)```", reply, re.S)
    text = fenced.group(1) if fenced else reply
    starts = [index for index in (text.find("["), text.find("{")) if index >= 0]
    if not starts:
        raise BadReply("no JSON in the reply")
    try:
        value, _ = json.JSONDecoder().raw_decode(text[min(starts) :])
    except json.JSONDecodeError as error:
        raise BadReply(f"unreadable JSON: {error}") from None
    return value
