"""League scheduling rules over plain facts: who offered which times, and who accepted what.

This is the only implementation of the rules. Data generation computes gold
results with it from planned facts; evaluation computes predicted results with
it from what the model marked. Nothing here knows about records, transcripts or
models.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import combinations, pairwise
from pathlib import Path

RULES_FILE = Path(__file__).with_name("rules.toml")


@dataclass(frozen=True)
class Rule:
    window: timedelta
    min_times: int
    min_days: int
    min_gap: timedelta
    range_counts_twice: timedelta


@dataclass(frozen=True)
class Offer:
    id: str
    player: str
    sent: datetime
    start: datetime
    end: datetime | None = None
    open: bool = False
    withdrawn: datetime | None = None

    def standing(self, at: datetime) -> bool:
        return self.sent <= at and (self.withdrawn is None or self.withdrawn > at)


@dataclass(frozen=True)
class Accepted:
    """A player accepted an offer, at `sent`; `time` is the pick when the offer is a range.
    An acceptance the player took back (`retracted`) agrees nothing, and counts for effort
    only if it still stood at the cutoff, as a withdrawn offer does."""

    player: str
    sent: datetime
    offer: str
    time: datetime | None = None
    retracted: datetime | None = None

    def standing(self, at: datetime) -> bool:
        return self.sent <= at and (self.retracted is None or self.retracted > at)


@dataclass(frozen=True)
class Agreement:
    player: str
    sent: datetime
    time: datetime


def load(path: Path = RULES_FILE) -> dict[str, Rule]:
    with path.open("rb") as file:
        leagues = tomllib.load(file)
    return {
        league: Rule(
            window=timedelta(hours=values["window_hours"]),
            min_times=values["min_times"],
            min_days=values["min_days"],
            min_gap=timedelta(hours=values["min_gap_hours"]),
            range_counts_twice=timedelta(hours=values["range_counts_twice_hours"]),
        )
        for league, values in leagues.items()
    }


def counted_times(rule: Rule, offer: Offer) -> list[datetime]:
    if offer.end is not None and offer.end - offer.start >= rule.range_counts_twice:
        return [offer.start, offer.start + rule.range_counts_twice]
    return [offer.start]


def covers(offer: Offer, time: datetime) -> bool:
    """Whether a time is one the offer allows: its start, a time in its range, or any time after an open start."""
    if offer.end is not None:
        return offer.start <= time <= offer.end
    return offer.start <= time if offer.open else time == offer.start


def chosen(offers: list[Offer], time: datetime | None) -> Offer | None:
    """Which of the offers one mention made an acceptance takes: the only one, or the latest that allows its time."""
    if len(offers) == 1:
        return offers[0]
    allowing = [offer for offer in offers if time is not None and covers(offer, time)]
    return max(allowing, key=lambda offer: offer.start) if allowing else None


def valid(acceptance: Accepted, offers: dict[str, Offer]) -> bool:
    """An acceptance counts only for the opponent's offer, still standing when accepted."""
    offer = offers.get(acceptance.offer)
    return offer is not None and offer.player != acceptance.player and offer.standing(acceptance.sent)


def agreed_time(acceptance: Accepted, offer: Offer) -> datetime | None:
    """The exact time agreed: the offer's, or the one picked inside a range."""
    if offer.end is None and not offer.open:
        return offer.start
    return acceptance.time if acceptance.time is not None and covers(offer, acceptance.time) else None


def agreements(offers: list[Offer], acceptances: list[Accepted]) -> list[Agreement]:
    by_id = {offer.id: offer for offer in offers}
    found = []
    for acceptance in sorted(acceptances, key=lambda acceptance: acceptance.sent):
        if acceptance.retracted is not None or not valid(acceptance, by_id):
            continue
        time = agreed_time(acceptance, by_id[acceptance.offer])
        if time is not None:
            found.append(Agreement(acceptance.player, acceptance.sent, time))
    return found


def qualifying_set(rule: Rule, times: list[datetime]) -> bool:
    """Whether some set of the times meets the count, spacing and day requirements."""
    times = sorted(set(times))
    smallest = max(rule.min_times, rule.min_days, 1)
    for size in range(smallest, len(times) + 1):
        for chosen in combinations(times, size):
            spaced = all(later - earlier >= rule.min_gap for earlier, later in pairwise(chosen))
            if spaced and len({time.date() for time in chosen}) >= rule.min_days:
                return True
    return False


def meets_rule(rule: Rule, published: datetime, player: str, offers: list[Offer], acceptances: list[Accepted]) -> bool:
    cutoff = published + rule.window
    by_id = {offer.id: offer for offer in offers}
    accepted = any(
        acceptance.player == player and acceptance.standing(cutoff) and valid(acceptance, by_id)
        for acceptance in acceptances
    )
    if accepted:
        return True
    times = [
        time
        for offer in offers
        if offer.player == player and offer.standing(cutoff)
        for time in counted_times(rule, offer)
    ]
    return qualifying_set(rule, times)
