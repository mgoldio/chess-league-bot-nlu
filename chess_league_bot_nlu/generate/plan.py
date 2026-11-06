"""Plan scheduling threads in code, so their labels are true by construction.

A thread plan is a list of steps: who speaks, when, and what each message must
do (offer these times, accept that offer, withdraw one, chat). The model only
writes words for a plan; the labels come from here. Messages to the bot are
planned in `command_plan`.
"""

from __future__ import annotations

import dataclasses
import random
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from ..records import PLAYERS, opponent
from . import catalog
from .errors import Impossible
from .names import name


@dataclass
class PlannedTime:
    """A time in a message. `form` is how it is written: in full (""), as its day alone ("day"),
    as a clock after a day given just before it ("clock"), or converted to the player's own
    zone as a bare clock ("local"). `more_days` offers the same time that many days later too;
    `struck_for` marks a time struck through and replaced by that one in an edited message.
    A time the player can't make ("busy") is written in full and is not an offer."""

    id: str
    role: str
    start: datetime
    end: datetime | None = None
    open: bool = False
    zone: tuple[str, float] | None = None
    accepts: str | None = None
    repeats: str | None = None
    more_days: tuple[int, ...] = ()
    struck_for: str | None = None
    form: str = ""
    own_zone: bool = False

    @property
    def placeholder(self) -> str:
        """What the writer calls this time: an acceptance, or a time gone through again to turn it
        down, uses the offer's name."""
        return self.accepts or (self.repeats if self.form == "quoted" else None) or self.id

    def shift(self) -> timedelta:
        return timedelta(hours=self.zone[1]) if self.zone else timedelta(0)

    def local_start(self) -> datetime:
        """The start as the player writes it: shifted to their zone, if they use one."""
        return self.start + self.shift()

    def slots(self) -> list[tuple[datetime, datetime | None]]:
        """Each day's start and end."""
        return [
            (self.start + timedelta(days=days), self.end + timedelta(days=days) if self.end else None)
            for days in (0, *self.more_days)
        ]

    def local_days(self) -> set[date]:
        return {(start + self.shift()).date() for start, _ in self.slots()}


@dataclass
class Step:
    speaker: str
    sent: datetime
    kind: str
    words: int
    note: str = ""
    times: list[PlannedTime] = field(default_factory=list)
    accepts: str | None = None
    withdraws: str | None = None
    retracts: str | None = None
    style: str = ""
    reference: str = ""
    reference_kind: str = ""
    withdraw_reference: str = ""
    withdraw_reference_kind: str = ""
    pick: datetime | None = None
    layout: str = ""
    names_day: bool = False


@dataclass
class ThreadPlan:
    league: catalog.League
    round: int
    published: datetime
    deadline: datetime
    personas: dict[str, catalog.Persona]
    names: dict[str, str]
    plot: str
    steps: list[Step]

    def offer(self, offer_id: str) -> PlannedTime:
        return next(
            time for step in self.steps for time in step.times if time.id == offer_id and time.role == "offered"
        )


@dataclass
class Offered:
    """An offer still on the table, and the times offered together with it in one message."""

    player: str
    time: PlannedTime
    together: list[PlannedTime]


@dataclass
class Agreed:
    player: str
    offer: str
    time: datetime


def thread(rng: random.Random, weights: dict[str, float] | None = None, others: float = 1.0) -> ThreadPlan:
    """A planned thread. `weights` sets how often the named plots are picked, and `others`
    scales the catalog's weights of the rest."""
    for _ in range(50):
        try:
            plan = Planner(rng, weights or {}, others).plan()
        except Impossible:
            continue
        check_plan(plan)
        return plan
    raise Impossible("no plan after 50 attempts")


class Planner:
    """One attempt at planning a thread. Sampling dead ends raise `Impossible`."""

    def __init__(self, rng: random.Random, weights: dict[str, float] | None = None, others: float = 1.0):
        self.rng = rng
        self.weights, self.others = weights or {}, others
        self.plots = catalog.plots()
        self.schedule = catalog.schedule()
        self.ids = 0
        self.standing: dict[str, Offered] = {}
        self.last_offers: dict[str, list[str]] = {}
        self.agreed: list[Agreed] = []
        self.moving: datetime | None = None
        self.withdrawn: PlannedTime | None = None
        self.reopened: list[PlannedTime] = []
        self.busy_times: list[datetime] = []
        league = rng.choices(self.schedule.leagues, [league.weight for league in self.schedule.leagues])[0]
        number = rng.randint(league.first_round, league.last_round)
        published = league.first_published + timedelta(weeks=number - league.first_round)
        vocabulary = catalog.vocabulary()
        names = {"white": name(rng, vocabulary)}
        while (other := name(rng, vocabulary)) == names["white"]:
            pass
        self.result = ThreadPlan(
            league=league,
            round=number,
            published=published + timedelta(minutes=rng.choice(league.publish_delay_minutes)),
            deadline=published + timedelta(hours=league.round_hours),
            personas={player: rng.choice(catalog.personas()) for player in PLAYERS},
            names={**names, "black": other},
            plot="",
            steps=[],
        )
        self.at = self.result.published

    def plan(self) -> ThreadPlan:
        rng, plots = self.rng, self.plots
        weights = [self.weights.get(plot.name, plot.weight * self.others) for plot in plots.plots]
        plot = rng.choices(plots.plots, weights)[0]
        self.result.plot = plot.name
        first = rng.choice(PLAYERS)
        if rng.random() < plots.opener_chance:
            self.wait("normal")
            self.say(first, "opener", note=rng.choice(plots.openers))
        previous = first
        for beat in plot.beats:
            if rng.random() >= beat.chance:
                continue
            speaker = {
                "first": first,
                "second": opponent(first),
                "either": rng.choice(PLAYERS),
                "other": opponent(previous),
            }[beat.by]
            self.wait(beat.gap)
            self.beat_handler(beat.do)(speaker, beat)
            previous = speaker
        if rng.random() < plots.closer_chance:
            self.wait("fast")
            closer = rng.choice(PLAYERS)
            self.say(closer, "closer", note=rng.choice(plots.closers))
            if rng.random() < plots.closer_reply_chance:
                self.wait("fast")
                self.say(opponent(closer), "closer", note=rng.choice(plots.closers))
        return self.result

    def new_id(self) -> str:
        self.ids += 1
        return f"T{self.ids}"

    def note(self, beat: catalog.Beat, extra: tuple[str, ...] = (), chance: float = 0.0) -> str:
        """The beat's note, one of its alternatives or of its shared `notes` list, or now and then a general one."""
        if beat.notes:
            return self.rng.choice(self.plots.notes[beat.notes])
        if beat.note:
            return self.rng.choice(beat.note)
        return self.rng.choice(extra) if extra and self.rng.random() < chance else ""

    def wait(self, pace: str) -> None:
        mu, sigma = self.schedule.gaps[pace]
        minutes = round(max(1, min(self.rng.lognormvariate(mu, sigma), 60 * 50)))
        self.at += timedelta(minutes=minutes)

    def beat_handler(self, kind: str):
        """What a beat of this kind does; kinds without times only need a message."""
        handlers = {
            "offer": self.offer,
            "accept": self.accept,
            "withdraw": self.withdraw,
            "correct": self.correct,
            "retract": self.retract,
            "mistaken_accept": self.mistaken_accept,
            "busy": self.busy,
            "cancel": self.cancel,
        }
        return handlers.get(kind, self.plain)

    def say(self, speaker: str, kind: str, **values) -> Step:
        steps = self.result.steps
        if self.at >= self.result.deadline or (steps and self.at <= steps[-1].sent):
            raise Impossible("no room for the message")
        low, high = self.result.personas[speaker].words
        step = Step(speaker, self.at, kind, self.rng.randint(low, high), **values)
        steps.append(step)
        return step

    def plain(self, speaker: str, beat: catalog.Beat) -> None:
        self.say(speaker, beat.do, note=self.note(beat), names_day=beat.names_day)

    def bounds(self) -> tuple[datetime, datetime]:
        """The earliest and latest an offered game may start."""
        schedule = self.schedule
        earliest = max(
            self.at + timedelta(hours=schedule.lead_hours),
            self.result.published + timedelta(hours=schedule.earliest_after_pairing_hours),
        )
        return earliest, self.result.deadline - timedelta(hours=schedule.before_deadline_hours)

    def taken(self) -> list[datetime]:
        return [
            *(start for offered in self.standing.values() for start, _ in offered.time.slots()),
            *(agreed.time for agreed in self.agreed),
            *self.busy_times,
        ]

    def spaced(self, start: datetime, taken: list[datetime]) -> bool:
        return all(abs(start - other) >= timedelta(hours=self.schedule.spacing_hours) for other in taken)

    def game_time(self, earliest: datetime, latest: datetime) -> datetime:
        """A plausible game start, mostly in the evening UTC, on the hour or half hour."""
        weights = self.schedule.hour_weights
        minutes = self.schedule.minutes
        for _ in range(200):
            day = earliest + timedelta(days=self.rng.randint(0, 6))
            hour = self.rng.choices(range(24), weights)[0]
            minute = self.rng.choices(list(minutes), list(minutes.values()))[0]
            start = day.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if earliest <= start <= latest:
                return start
        raise Impossible("no game time fits")

    def offer(self, speaker: str, beat: catalog.Beat) -> None:
        schedule = self.schedule
        if beat.before_agreed:
            if not self.agreed:
                raise Impossible("nothing agreed to move")
            self.moving = self.agreed[-1].time
            self.at = min(self.at, self.moving - timedelta(hours=schedule.lead_hours + schedule.accept_before_hours))
        earliest, latest = self.bounds()
        if beat.same_day:
            times = self.same_day(speaker, earliest, latest)
        elif beat.near:
            times = self.near(speaker, beat, earliest, latest)
        else:
            times = self.fresh(beat, earliest, latest)
        offered = [time for time in times if time.role == "offered"]
        for time in offered:
            self.standing[time.id] = Offered(speaker, time, offered)
        self.last_offers[speaker] = [time.id for time in offered]
        if not beat.near and not beat.same_day:
            times = self.strike(times, beat, earliest, latest)
            busy = beat.busy_chance if beat.busy_chance is not None else self.plots.busy_chance
            if self.rng.random() < busy:
                times = [self.busy_time(), *times]
            if self.rng.random() < self.plots.context_chance:
                deadline = PlannedTime(self.new_id(), "other", self.result.deadline, form="context")
                times = [deadline, *times]
        if len(offered) == 1:
            times += self.converted(offered[0], beat)
        if beat.rejects:
            theirs = [
                self.standing[offer_id].time
                for offer_id in self.last_offers.get(opponent(speaker), [])
                if offer_id in self.standing
            ]
            if not theirs:
                raise Impossible("no offer of theirs to go through")
            times = [dataclasses.replace(self.repeat(time), form="quoted") for time in theirs] + times
        note = self.note(beat, self.plots.offer_notes, self.plots.offer_note_chance)
        step = self.say(speaker, "offer", note=note, times=times, layout=beat.layout, names_day=beat.names_day)
        step.words += self.plots.quoting_words * sum(time.form == "quoted" for time in times)

    def fresh(self, beat: catalog.Beat, earliest: datetime, latest: datetime) -> list[PlannedTime]:
        taken = self.taken()
        times = []
        for _ in range(self.rng.randint(*beat.count_range())):
            for _ in range(50):
                start = self.game_time(earliest, latest)
                if self.spaced(start, taken):
                    break
            else:
                raise Impossible("offers too crowded")
            time = self.shape(PlannedTime(self.new_id(), "offered", start), beat, latest)
            self.spread(time, beat, earliest, latest, taken)
            times.append(time)
            taken += [start for start, _ in time.slots()]
        return sorted(times, key=lambda time: time.start)

    def shape(self, time: PlannedTime, beat: catalog.Beat, latest: datetime) -> PlannedTime:
        """Make an offered time a range, open-ended, or written in another zone, by chance."""
        rng, plots = self.rng, self.plots
        ranges = beat.range_chance if beat.range_chance is not None else plots.range_chance
        roll = rng.random()
        if roll < ranges:
            time.end = min(time.start + timedelta(hours=rng.choice(self.schedule.range_hours)), latest)
            if time.end <= time.start:
                time.end = None
        elif roll < ranges + plots.open_chance:
            time.open = True
        elif rng.random() < (beat.zone_chance if beat.zone_chance is not None else plots.zone_chance):
            time.zone = rng.choice(self.schedule.zones)
            time.own_zone = time.zone[0].endswith(" time") and rng.random() < plots.my_time_chance
        return time

    def spread(
        self, time: PlannedTime, beat: catalog.Beat, earliest: datetime, latest: datetime, taken: list[datetime]
    ) -> None:
        """By chance, offer the same time on one or two more days ("21:00 fri or sat")."""
        chance = beat.multi_day_chance if beat.multi_day_chance is not None else self.plots.multi_day_chance
        if self.rng.random() >= chance:
            return
        days = tuple(sorted(self.rng.sample(range(1, 4), self.rng.choice((1, 1, 2)))))
        candidate = PlannedTime(time.id, time.role, time.start, time.end, time.open, time.zone, more_days=days)
        slots = candidate.slots()[1:]
        if all(earliest <= start and (end or start) <= latest and self.spaced(start, taken) for start, end in slots):
            time.more_days = days

    def near(self, speaker: str, beat: catalog.Beat, earliest: datetime, latest: datetime) -> list[PlannedTime]:
        """Times asked about close to the opponent's last offer: inside its range, or just around it,
        a quarter or half hour apart ("01:30? 01:45? 02:00 at the latest?")."""
        rng = self.rng
        theirs = self.their_last(speaker)
        if not theirs:
            raise Impossible("nothing to ask about")
        target = rng.choice(theirs)
        start, end = rng.choice(target.slots())
        step = timedelta(minutes=rng.choice((15, 30)))
        if end or target.open:
            last = end or start + timedelta(hours=self.schedule.open_pick_hours)
            candidates = [start + step * index for index in range(1, int((last - start) / step))]
        else:
            candidates = [start + step * index for index in range(-6, 7) if index]
        offered = {when for other in self.standing.values() for when, _ in other.time.slots()}
        candidates = [when for when in candidates if earliest <= when <= latest and when not in offered]
        count = rng.randint(*beat.count_range())
        if len(candidates) < count:
            raise Impossible("no room near the offer")
        first = rng.randrange(len(candidates) - count + 1)
        return [
            PlannedTime(self.new_id(), "offered", when, zone=target.zone) for when in candidates[first : first + count]
        ]

    def their_last(self, speaker: str) -> list[PlannedTime]:
        """The opponent's offers from their last offering message that still stand."""
        last = self.last_offers.get(opponent(speaker), [])
        return [self.standing[offer_id].time for offer_id in last if offer_id in self.standing]

    def same_day(self, speaker: str, earliest: datetime, latest: datetime) -> list[PlannedTime]:
        """A counter that keeps the day of the opponent's offer and asks for another clock time:
        "{day} is fine but can we make it {clock}?". The day is a repeat; the clock is the offer."""
        theirs = [time for time in self.their_last(speaker) if not time.more_days and not time.end and not time.open]
        if not theirs:
            raise Impossible("no single time to counter")
        target = self.rng.choice(theirs)
        taken = self.taken()
        for _ in range(30):
            start = target.start + timedelta(minutes=30 * self.rng.choice([*range(-8, -1), *range(2, 9)]))
            if (
                earliest <= start <= latest
                and (start + target.shift()).date() == target.local_start().date()
                and self.spaced(start, taken)
            ):
                break
        else:
            raise Impossible("no other time that day")
        day = PlannedTime(self.new_id(), "other", target.start, zone=target.zone, repeats=target.id, form="day")
        return [day, PlannedTime(self.new_id(), "offered", start, zone=target.zone, form="clock")]

    def strike(
        self, times: list[PlannedTime], beat: catalog.Beat, earliest: datetime, latest: datetime
    ) -> list[PlannedTime]:
        """By chance, the message was edited: one single-day time replaced an earlier one, now struck through."""
        chance = beat.struck_chance if beat.struck_chance is not None else self.plots.struck_chance
        plain = [time for time in times if not time.more_days]
        if not plain or self.rng.random() >= chance:
            return times
        kept = self.rng.choice(plain)
        shifts = [timedelta(hours=hours) for hours in (-3, -2, -1, 1, 2, 3, 24, -24)]
        self.rng.shuffle(shifts)
        shift = next((shift for shift in shifts if earliest <= kept.start + shift <= latest), None)
        if shift is None:
            return times
        struck = PlannedTime(
            self.new_id(),
            "other",
            kept.start + shift,
            kept.end + shift if kept.end else None,
            kept.open,
            kept.zone,
            struck_for=kept.id,
        )
        index = times.index(kept)
        return [*times[:index], struck, *times[index:]]

    def converted(self, time: PlannedTime, beat: catalog.Beat) -> list[PlannedTime]:
        """By chance, the time again in the player's own zone, as a bare clock ("that's 3:30pm for me"). Not an offer."""
        chance = beat.local_chance if beat.local_chance is not None else self.plots.local_chance
        if time.zone or self.rng.random() >= chance:
            return []
        zone = self.rng.choice([zone for zone in self.schedule.zones if zone[1] % 24])
        return [PlannedTime(self.new_id(), "other", time.start, zone=zone, form="local")]

    def accept(self, speaker: str, beat: catalog.Beat) -> None:
        candidates = [
            offer_id
            for offer_id, offered in self.standing.items()
            if offered.player == opponent(speaker)
            and (beat.which == "any" or offer_id in self.last_offers.get(offered.player, []))
        ]
        if not candidates:
            raise Impossible("nothing to accept")
        chosen = self.rng.choice(candidates)
        offered = self.standing[chosen]
        latest = offered.time.start - timedelta(hours=self.schedule.accept_before_hours)
        if self.moving is not None:
            latest = min(latest, self.moving - timedelta(hours=self.schedule.accept_before_hours))
        self.at = min(self.at, latest)
        rejected = [offer_id for offer_id in candidates if offer_id != chosen]
        rejecting = beat.rejects and rejected
        style = "restate" if rejecting and not offered.time.end and not offered.time.open else beat.style
        style = self.accept_style(offered, candidates, style)
        note = self.note(beat, self.plots.accept_notes, self.plots.accept_note_chance)
        # A clock alone takes its day from the message before, so only right after the offer, alone in it.
        steps = self.result.steps
        right_after = bool(steps) and [time.id for time in steps[-1].times] == [chosen]
        step = self.say(speaker, "accept", accepts=chosen, style=style, note=note)
        if rejecting:
            step.times.append(self.repeat(self.standing[self.rng.choice(rejected)].time))
        if style == "pick":
            pick_clock_chance = (
                self.plots.pick_clock_chance if beat.pick_clock_chance is None else beat.pick_clock_chance
            )
            step.pick = self.pick(offered.time)
            same_day = (step.pick + offered.time.shift()).date() == offered.time.local_start().date()
            alone = (
                right_after
                and same_day
                and len(offered.time.local_days()) == 1
                and self.rng.random() < pick_clock_chance
            )
            form = "clock" if alone else ""
            step.times.append(PlannedTime(self.new_id(), "accepted", step.pick, accepts=chosen, form=form))
        elif style in ("restate", "partial"):
            start = self.rng.choice(offered.time.slots())[0]
            if offered.time.more_days:
                step.pick = start
            form = "day" if style == "partial" else ""
            step.times.append(
                PlannedTime(self.new_id(), "accepted", start, zone=offered.time.zone, accepts=chosen, form=form)
            )
        elif len(candidates) > 1:
            references = self.references(offered)
            if beat.loosely:
                references = [reference for reference in references if reference[0] == "day_part"]
                if not references:
                    raise Impossible("no day and part of the day to point at the offer by")
            step.reference_kind, step.reference = self.rng.choice(references)
        if step.times and step.times[-1].role == "accepted":
            step.times += self.converted(step.times[-1], beat)
        self.agreed.append(Agreed(speaker, chosen, step.pick or offered.time.start))
        self.moving = None
        if beat.also_withdraws:
            self.withdraw_too(speaker, step)
        for offer_id in candidates:
            del self.standing[offer_id]

    def withdraw_too(self, speaker: str, step: Step) -> None:
        """In the same message that accepts, take back one of one's own offers ("sun works, and forget thursday").
        Planned before the accepted offers leave the table, so a reference by day counts them too."""
        own = [offer_id for offer_id, offered in self.standing.items() if offered.player == speaker]
        if not own:
            raise Impossible("no own offer to take back while accepting")
        chosen = self.rng.choice(own)
        references = self.references(self.standing[chosen])
        offer = self.standing.pop(chosen).time
        if offer.start <= self.at:
            raise Impossible("the offer has already passed")
        self.withdrawn = offer
        step.withdraws = chosen
        if references and self.rng.random() < self.plots.withdraw_reference_chance:
            step.withdraw_reference_kind, step.withdraw_reference = self.rng.choice(references)
        else:
            step.times.append(self.repeat(offer))

    def accept_style(self, offered: Offered, candidates: list[str], wanted: str) -> str:
        """How the acceptance is written. A range needs the time picked in it, and a time on several
        days the day; a bare acceptance of one of several offers needs a clear way to point at it,
        or it restates the time instead."""
        if offered.time.end or offered.time.open:
            return "pick"
        styles = dict(self.plots.accept_styles)
        others = [self.standing[offer_id].time for offer_id in candidates if offer_id != offered.time.id]
        their_days = {day for other in self.others_standing(offered) for day in other.local_days()}
        if offered.time.zone or offered.time.local_days() & their_days:
            styles.pop("partial", None)
        if offered.time.more_days or (others and not self.references(offered)):
            styles.pop("bare", None)
        if wanted in styles:
            return wanted
        return self.rng.choices(list(styles), list(styles.values()))[0]

    def others_standing(self, offered: Offered) -> list[PlannedTime]:
        return [
            other.time
            for other in self.standing.values()
            if other.player == offered.player and other.time is not offered.time
        ]

    def references(self, offered: Offered) -> list[tuple[str, str]]:
        """Unambiguous ways to point at an offer without writing it, as (kind, wording): by its place
        among the times offered with it, if no other offer of theirs is still standing, or by its day,
        if no other offer standing, theirs, the speaker's or one whose acceptance was taken back, is on that day."""
        beats = catalog.beats()
        options = []
        others = self.others_standing(offered)
        if all(other in offered.together for other in others):
            order = offered.together.index(offered.time)
            if order < len(beats["ordinals"]):
                options.append(("ordinal", beats["reference_ordinal"].format(ordinal=beats["ordinals"][order])))
            if order == len(offered.together) - 1:
                options.append(("last", beats["reference_last"]))
        days = offered.time.local_days()
        everyone = [other.time for other in self.standing.values() if other.time is not offered.time] + self.reopened
        taken_days = {day for other in everyone for day in other.local_days()}
        taken_days |= {agreed.time.date() for agreed in self.agreed} | {start.date() for start in self.busy_times}
        taken_days.add(self.result.deadline.date())
        if not offered.time.zone and len(days) == 1 and not days & taken_days:
            for wording in ("reference_day", "reference_day_bare"):
                options.append(("day", beats[wording].format(weekday=f"{offered.time.start:%A}")))
            if part := part_of_day(offered.time):
                weekday = f"{offered.time.start:%A}"
                options.append(("day_part", beats["reference_day_part"].format(weekday=weekday, part=part)))
        return options

    def pick(self, offer: PlannedTime) -> datetime:
        """The exact time chosen inside a range or after an open-ended start, on one of its days, before the deadline."""
        deadline = self.result.deadline - timedelta(hours=self.schedule.before_deadline_hours)
        slots = [(start, end) for start, end in offer.slots() if start <= deadline]
        if not slots:
            raise Impossible("no time to pick before the deadline")
        start, end = self.rng.choice(slots)
        latest = min(end or start + timedelta(hours=self.schedule.open_pick_hours), deadline)
        steps = int((latest - start).total_seconds() // 1800)
        return start + timedelta(minutes=30 * self.rng.randint(0, steps))

    def withdraw(self, speaker: str, beat: catalog.Beat) -> None:
        """Take back one of one's own offers: writing it again, or pointing at it ("forget thursday")."""
        own = [offer_id for offer_id, offered in self.standing.items() if offered.player == speaker]
        if len(own) < 2:
            raise Impossible("nothing to withdraw while keeping an offer standing")
        chosen = self.rng.choice(own)
        references = self.references(self.standing[chosen])
        offer = self.standing.pop(chosen).time
        if offer.start <= self.at:
            raise Impossible("the offer has already passed")
        self.withdrawn = offer
        note = self.note(beat)
        if references and self.rng.random() < self.plots.withdraw_reference_chance:
            kind, reference = self.rng.choice(references)
            self.say(speaker, "withdraw", withdraws=chosen, reference=reference, reference_kind=kind, note=note)
        else:
            self.say(speaker, "withdraw", withdraws=chosen, times=[self.repeat(offer)], note=note)

    def cancel(self, speaker: str, beat: catalog.Beat) -> None:
        """The player who offered an agreed time takes it back: the game is off. `restates` writes the time."""
        offers = {time.id: step.speaker for step in self.result.steps for time in step.times if time.role == "offered"}
        mine = [agreed for agreed in self.agreed if offers.get(agreed.offer) == speaker and agreed.time > self.at]
        if not mine:
            raise Impossible("no agreed offer of theirs to cancel")
        agreed = mine[-1]
        self.agreed.remove(agreed)
        times = []
        if beat.restates:
            offer = self.result.offer(agreed.offer)
            times.append(PlannedTime(self.new_id(), "other", agreed.time, zone=offer.zone, repeats=agreed.offer))
        self.say(speaker, "cancel", withdraws=agreed.offer, times=times, note=self.note(beat))

    def busy(self, speaker: str, beat: catalog.Beat) -> None:
        """A message saying the speaker can't make a time of their own: written in full, not an offer.
        `near`: a window around the opponent's last offer ("at work 18:00-21:00 thursday")."""
        time = self.busy_around(speaker) if beat.near else self.busy_time()
        self.say(speaker, "busy", times=[time], note=self.note(beat))

    def busy_around(self, speaker: str) -> PlannedTime:
        """A busy window that covers one of the opponent's last offers: it starts before and ends after it."""
        theirs = [time for time in self.their_last(speaker) if not time.more_days and not time.open]
        if not theirs:
            raise Impossible("no offer to be busy around")
        target = self.rng.choice(theirs)
        start = target.start - timedelta(hours=self.rng.choice((1, 2, 3)))
        end = (target.end or target.start) + timedelta(hours=self.rng.choice((1, 2, 3)))
        if end >= self.result.deadline:
            raise Impossible("the busy window runs past the deadline")
        self.busy_times.append(start)
        return PlannedTime(self.new_id(), "other", start, end, zone=target.zone, form="busy")

    def busy_time(self) -> PlannedTime:
        """A new time the speaker can't make, clear of every time on the table, sometimes a range."""
        earliest, latest = self.bounds()
        taken = self.taken()
        for _ in range(50):
            start = self.game_time(earliest, latest)
            if self.spaced(start, taken):
                break
        else:
            raise Impossible("no room for a busy time")
        time = PlannedTime(self.new_id(), "other", start, form="busy")
        if self.rng.random() < self.plots.busy_range_chance:
            end = min(start + timedelta(hours=self.rng.choice(self.schedule.range_hours)), latest)
            time.end = end if end > start else None
        self.busy_times.append(start)
        return time

    def correct(self, speaker: str, beat: catalog.Beat) -> None:
        """Take back a time just offered as a slip, and offer the one meant ("I mean wed 12:00")."""
        own = [offer_id for offer_id in self.last_offers.get(speaker, []) if offer_id in self.standing]
        if len(own) != 1:
            raise Impossible("a correction needs exactly one time just offered")
        chosen = self.rng.choice(own)
        old = self.standing.pop(chosen).time
        earliest, latest = self.bounds()
        taken = self.taken()
        for _ in range(30):
            shift = timedelta(hours=self.rng.choice((-48, -24, 24, 48, -3, -2, -1, 1, 2, 3)))
            moved = PlannedTime(
                self.new_id(),
                "offered",
                old.start + shift,
                old.end + shift if old.end else None,
                old.open,
                old.zone,
                more_days=old.more_days,
            )
            fits = all(earliest <= start and (end or start) <= latest for start, end in moved.slots())
            if fits and all(self.spaced(start, taken) for start, _ in moved.slots()):
                break
        else:
            raise Impossible("nothing to correct to")
        self.standing[moved.id] = Offered(speaker, moved, [moved])
        self.last_offers[speaker] = [offer_id for offer_id in own if offer_id != chosen] + [moved.id]
        self.say(speaker, "correct", withdraws=chosen, times=[moved], note=self.note(beat))

    def retract(self, speaker: str, beat: catalog.Beat) -> None:
        """Take back one's own acceptance; the opponent's offer stands again (but isn't accepted again here)."""
        mine = [agreed for agreed in self.agreed if agreed.player == speaker and agreed.time > self.at]
        if not mine:
            raise Impossible("no acceptance to take back")
        agreed = mine[-1]
        self.agreed.remove(agreed)
        self.reopened.append(self.result.offer(agreed.offer))
        times = []
        if beat.restates:
            offer = self.result.offer(agreed.offer)
            times.append(PlannedTime(self.new_id(), "other", agreed.time, zone=offer.zone, repeats=agreed.offer))
        self.say(speaker, "retract", retracts=agreed.offer, times=times, note=self.note(beat))

    def mistaken_accept(self, speaker: str, beat: catalog.Beat) -> None:
        if self.withdrawn is None or self.withdrawn.start <= self.at:
            raise Impossible("nothing withdrawn that is still to come")
        self.say(speaker, "mistaken_accept", times=[self.repeat(self.withdrawn)])

    def repeat(self, offer: PlannedTime) -> PlannedTime:
        return PlannedTime(
            self.new_id(),
            "other",
            offer.start,
            offer.end,
            offer.open,
            offer.zone,
            repeats=offer.id,
            more_days=offer.more_days,
        )


def part_of_day(time: PlannedTime) -> str:
    """The part of the day a single UTC time falls in, for pointing at it loosely ("saturday evening")."""
    if time.end or time.open or time.more_days or time.zone:
        return ""
    hour = time.start.hour
    return "morning" if 6 <= hour < 12 else "afternoon" if 12 <= hour < 17 else "evening" if 17 <= hour < 22 else ""


def check_plan(plan: ThreadPlan) -> None:
    """Invariants every plan keeps. Breaking one is a planner bug, not bad luck."""
    sent = [step.sent for step in plan.steps]
    assert sent == sorted(sent) and len(set(sent)) == len(sent), "messages out of order"
    assert not sent or (plan.published < sent[0] and sent[-1] < plan.deadline), "a message outside the round"
    offers = {time.id: (step, time) for step in plan.steps for time in step.times if time.role == "offered"}
    accepted = {}
    for step in plan.steps:
        for time in step.times:
            for start, _ in time.slots():
                assert start > step.sent or time.role == "other", f"{time.id} is not in the future"
                assert start < plan.deadline or (time.form == "context" and start == plan.deadline), (
                    f"{time.id} is after the deadline"
                )
        if step.accepts:
            offer_step, offer = offers[step.accepts]
            assert offer_step.speaker != step.speaker, "accepting an own offer"
            assert offer.start > step.sent, "accepting a past offer"
            assert not offer.more_days or step.pick, "accepting a time on several days without picking one"
            accepted[step.accepts] = step.speaker
        if step.withdraws:
            assert offers[step.withdraws][0].speaker == step.speaker, "withdrawing someone else's offer"
        if step.retracts:
            assert accepted.get(step.retracts) == step.speaker, "taking back an acceptance never made"
