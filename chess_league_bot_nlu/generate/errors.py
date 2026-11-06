"""Why a generated record is dropped.

A `Rejected` record is a normal outcome, logged with its reason. Any other
exception means a job broke (the model server didn't answer, or a bug): the
driver logs it to errors.jsonl and runs the job again on the next run.
"""


class Rejected(Exception):
    """A record that can't be made or doesn't pass a check."""


class Impossible(Rejected):
    """The sampled plan can't meet its constraints; sample another."""


class Unwritable(Rejected):
    """No form of a planned time reads back as that time."""
