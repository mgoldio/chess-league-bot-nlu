"""Placeholders in a message template, such as {T1} or {player}, where code fills in planned text."""

import re

PLACEHOLDER = re.compile(r"(\{[A-Za-z0-9 ]+\})")
