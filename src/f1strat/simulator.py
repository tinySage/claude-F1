"""Race simulator (Phase 4), built in layers.

Part A (this module so far):

1. :class:`RaceState`: the race at the end of a given lap, per car.
2. Single car against the clock, lap by lap (:func:`simulate_laps`), using the
   Phase 3 degradation model's posterior (``docs/model.md``).
3. Pit stops drawn from the circuit's observed 2022 green-flag stops.
4. Strategy search (:func:`search`): every legal remaining plan, scored on
   expected race time, with a full Monte Carlo on the shortlist.
5. Common random numbers (:class:`Draws`): simulation ``i`` uses the same
   posterior draw, stint draws, pit losses and lap noise for every plan.

Rivals and safety cars are Part B. Laps are simulated as green-flag laps in
clean air (no ``dirty_air_penalty``); rain is off.

**How a plan's race time is built.** For simulation ``i`` and a car on lap
``L`` with compound ``c`` at tyre age ``a``::

    lap = race_base + driver_offset + opening·[L in 2,3] + trend·(L − L0)
        + noise                                        (the same for every plan)
        + compound_offset[c] + team_compound[c]
        + (deg[c] + team_deg[c]) · a/10 + post_stop[c]·[1st/2nd lap after a stop]
        + stint effect                                 (one draw per stint)

plus a pit loss on each in lap. The first line is identical for every plan,
so plans differ only by their stints, and a stint's total has a closed form
(:func:`plan_totals`). :func:`simulate_laps` is the lap-by-lap reference; the
tests check the two agree.

**Stint effects (one draw per stint, never per lap).**

- The stint in progress: its fitted effect if the fit contains it and
  ``use_fitted_current_stint`` (e.g. a Phase 6 cutoff fit), else a draw.
- Future non-final stints: ``sigma_stint · z_j`` (``z_j`` shared by the
  ``j``-th future stint of every plan).
- The final stint: ``mu_final + final_race[r] + w · sigma_stint_final · z_final``,
  where ``w`` is the interval widening factor (default 1.188, from Phase 3). If
  the fit has no final-stint data for the race, ``final_race[r]`` is a prior
  draw and is widened by ``w`` too.
"""

from __future__ import annotations

import itertools
from collections import Counter
from collections.abc import Collection, Iterable, Iterator, Sequence
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from f1strat.race_events import SimParams  # noqa: E402  (type for Field)
from f1strat.deg_model import (
    COMPOUNDS,
    POST_STOP_LAPS,
    TYRE_SCALE,
    BayesFit,
)

DRY_COMPOUNDS: tuple[str, ...] = ("SOFT", "MEDIUM", "HARD")
OPENING_LAPS: tuple[int, ...] = (2, 3)
#: Phase 3 result: scale for the final-stint spread so last_stint intervals
#: reach 80% coverage (docs/model.md, "Results: final fit").
DEFAULT_WIDENING = 1.188
#: Out-lap TyreLife of a used set (e.g. a set run for 3 laps in qualifying).
DEFAULT_USED_LIFE = 4


# --- State and plans -----------------------------------------------------------


@dataclass(frozen=True)
class CarState:
    """One car at the end of the state's lap.

    ``tyre_life`` is the TyreLife of the lap just completed (FastF1 convention:
    the out lap of a new set is 1). ``laps_since_stop`` is 0 on the out lap.
    ``stint`` counts stints so far, including the current one; a stint other
    than the first started with a pit stop.
    """

    driver: str
    team: str
    position: int
    race_time: float
    compound: str
    tyre_life: int
    laps_since_stop: int
    stint: int
    compounds_used: frozenset[str]

    @property
    def stint_started_with_stop(self) -> bool:
        return self.stint > 1


@dataclass(frozen=True)
class RaceState:
    """The race at the end of lap ``lap`` (``lap >= 1``; lap 1 is not modelled).

    ``track_status`` is ``"green"``, ``"sc"`` or ``"vsc"`` for the lap just
    completed. Part A simulates green laps only; Part B adds SC/VSC.
    """

    round_number: int
    lap: int
    total_laps: int
    cars: dict[str, CarState]
    track_status: str = "green"
    #: Dry sets still available per car, as (compound, out-lap TyreLife); the
    #: set on the car is not included. ``None``: unknown (unlimited in search).
    sets: dict[str, tuple[tuple[str, int], ...]] | None = None
    #: Laps already run under the current SC/VSC (0 if green).
    neutral_laps: int = 0

    @property
    def laps_remaining(self) -> int:
        return self.total_laps - self.lap

    @classmethod
    def from_laps(cls, raw_laps: pd.DataFrame, round_number: int, lap: int,
                  total_laps: int) -> "RaceState":
        """Build the actual state of a 2022 race at the end of ``lap``.

        Args:
            raw_laps: ``load_season()`` output (uncleaned laps).
            round_number: Race round.
            lap: Last completed lap (>= 1).
            total_laps: Race distance.
        """
        if lap < 1:
            raise ValueError("lap must be >= 1 (lap 1 is not modelled)")
        race = raw_laps[raw_laps["RoundNumber"] == round_number]
        upto = race[race["LapNumber"] <= lap]
        now = upto[upto["LapNumber"] == lap]
        cars = {}
        for _, row in now.iterrows():
            mine = upto[upto["Driver"] == row["Driver"]]
            stint_laps = mine[mine["Stint"] == row["Stint"]]
            cars[row["Driver"]] = CarState(
                driver=row["Driver"],
                team=row["Team"],
                position=int(row["Position"]) if pd.notna(row["Position"]) else 99,
                race_time=float(mine["LapTime"].dt.total_seconds().sum()),
                compound=str(row["Compound"]),
                tyre_life=int(row["TyreLife"]),
                laps_since_stop=int(lap - stint_laps["LapNumber"].min()),
                stint=int(row["Stint"]),
                compounds_used=frozenset(
                    str(c) for c in mine["Compound"].dropna() if str(c) in DRY_COMPOUNDS
                ),
            )
        def kind(code: str) -> str:
            return "sc" if "4" in code else "vsc" if ("6" in code or "7" in code) else "green"

        leader = race[race["Position"] == 1].set_index("LapNumber")["TrackStatus"].astype(str)
        track = kind(str(now["TrackStatus"].iloc[0]) if len(now) else "1")
        neutral = 0
        if track != "green":
            for lp in range(lap, 0, -1):
                if kind(str(leader.get(lp, "1"))) != track:
                    break
                neutral += 1
        return cls(round_number=round_number, lap=lap, total_laps=total_laps,
                   cars=cars, track_status=track, neutral_laps=neutral)

    def with_sets(self, sets: dict[str, Sequence[tuple[str, int]]]) -> "RaceState":
        """Copy of the state with each car's available sets."""
        return replace(self, sets={d: tuple(v) for d, v in sets.items()})


@dataclass(frozen=True)
class Stop:
    """Pit at the end of ``lap`` and fit ``compound``; the out lap has
    TyreLife ``tyre_life`` (1 for a new set, more for a used one)."""

    lap: int
    compound: str
    tyre_life: int = 1

    @property
    def new(self) -> bool:
        return self.tyre_life == 1


@dataclass(frozen=True)
class Plan:
    """The remaining stops for one car, in lap order."""

    stops: tuple[Stop, ...] = ()

    def label(self, current: str) -> str:
        """E.g. ``"S → 14 M(new) → 29 S(used)"``."""
        parts = [current[0]]
        for s in self.stops:
            parts.append(f"{s.lap} {s.compound[0]}({'new' if s.new else 'used'})")
        return " → ".join(parts)


def is_legal(plan: Plan, car: CarState, state: RaceState, min_stint: int = 1,
             require_two_compounds: bool = True) -> bool:
    """Legal remaining plan for ``car`` from ``state``.

    Stops are on laps after the state lap and before the last lap, in order,
    at least ``min_stint`` laps apart (and after the current stint has run
    ``min_stint`` laps in total); all compounds are dry; and, if required, the
    whole race uses at least two different dry compounds.
    """
    prev = state.lap - car.laps_since_stop
    for s in plan.stops:
        if s.compound not in DRY_COMPOUNDS or s.tyre_life < 1:
            return False
        if not (state.lap < s.lap < state.total_laps) or s.lap - prev < min_stint:
            return False
        prev = s.lap
    if state.total_laps - prev < min_stint:
        return False
    if require_two_compounds:
        used = set(car.compounds_used) | {car.compound} | {s.compound for s in plan.stops}
        if len(used & set(DRY_COMPOUNDS)) < 2:
            return False
    return True


def enumerate_plans(
    state: RaceState,
    driver: str,
    max_stops: int = 3,
    compounds: Sequence[str] = DRY_COMPOUNDS,
    used_life: int | None = DEFAULT_USED_LIFE,
    min_stint: int = 1,
    pit_laps: Iterable[int] | None = None,
) -> Iterator[Plan]:
    """Every legal remaining plan with 0..``max_stops`` more stops.

    Args:
        state: Race state.
        driver: The car.
        max_stops: Most additional stops.
        compounds: Dry compounds allowed at each stop.
        used_life: Out-lap TyreLife of a used set; ``None`` allows new sets only.
        min_stint: Shortest stint in laps.
        pit_laps: Laps to consider for stops (default: every legal lap).
    """
    car = state.cars[driver]
    laps = sorted(pit_laps) if pit_laps is not None else range(state.lap + 1, state.total_laps)
    lives = (1,) if used_life is None else (1, used_life)
    sets = [(c, life) for c in compounds for life in lives]
    for k in range(max_stops + 1):
        for lap_combo in itertools.combinations(laps, k):
            for combo in itertools.product(sets, repeat=k):
                plan = Plan(tuple(Stop(lap, c, life) for lap, (c, life) in zip(lap_combo, combo)))
                if is_legal(plan, car, state, min_stint):
                    yield plan


def actual_plan(raw_laps: pd.DataFrame, round_number: int, driver: str, after_lap: int) -> Plan:
    """The stops a driver actually made after ``after_lap``, as a :class:`Plan`.

    A stop's lap is the last lap of the previous stint; its out-lap TyreLife is
    the set's recorded TyreLife on the out lap (1 for a new set).
    """
    laps = raw_laps[(raw_laps["RoundNumber"] == round_number) & (raw_laps["Driver"] == driver)]
    stints = laps.groupby("Stint").agg(first=("LapNumber", "min"),
                                       compound=("Compound", "first"),
                                       life=("TyreLife", "first")).sort_values("first")
    stops = [Stop(int(row["first"]) - 1, str(row["compound"]), int(row["life"]))
             for _, row in stints.iloc[1:].iterrows() if row["first"] - 1 > after_lap]
    return Plan(tuple(stops))


# --- Model inputs and common random numbers ------------------------------------


def _flat_draws(post, name: str) -> np.ndarray:
    arr = post[name].to_numpy()
    return arr.reshape(-1, *arr.shape[2:]).astype(float)


@dataclass
class CarModel:
    """Posterior draws needed to simulate one car at one race, per simulation.

    Arrays have a leading simulation axis of length ``n_sims``; compound axes
    follow :data:`f1strat.deg_model.COMPOUNDS`. Degradation is in s per lap.
    """

    base: np.ndarray          # race_base + driver_offset
    trend: np.ndarray         # s per race lap
    lap_centre: float
    opening: np.ndarray
    compound: np.ndarray      # (n, 3): compound_offset + team_compound
    deg: np.ndarray           # (n, 3): (deg + team_deg) per lap of tyre age
    post_stop: np.ndarray     # (n, 3)
    sigma: np.ndarray
    nu: np.ndarray
    sigma_stint: np.ndarray
    final_mean: np.ndarray    # mu_final + final_race (prior draw widened if no data)
    sigma_stint_final: np.ndarray  # already multiplied by the widening factor
    current_stint: np.ndarray | None  # fitted effect of the stint in progress
    #: Hindsight replays (Phase 6): fitted effects of stints that actually
    #: happened, keyed by (first lap, compound, TyreLife on that lap). A plan
    #: stint with the same key uses it for all its laps instead of a draw.
    known_stints: dict[tuple[int, str, int], np.ndarray] = field(default_factory=dict)


def car_model(
    fitted: BayesFit,
    round_number: int,
    driver: str,
    team: str,
    draw_index: np.ndarray,
    rng: np.random.Generator,
    current_stint: int | None = None,
    widening: float = DEFAULT_WIDENING,
    final_race_z: np.ndarray | None = None,
) -> CarModel:
    """Collect one car's posterior draws for the given posterior indices.

    Args:
        fitted: Degradation model fit (e.g. the thinned full fit).
        round_number: Race.
        driver, team: The car.
        draw_index: Posterior draw used by each simulation.
        rng: For pooled draws (a driver with no laps at this race).
        current_stint: Stint number in progress; its fitted effect is used if
            the fit contains it. ``None`` draws it instead.
        widening: Factor on the drawn final-stint spreads (1.0 = off).
        final_race_z: If given, ``final_race[r]`` is a prior draw
            ``widening · sigma_final_race · z`` with these standard normals
            (one per simulation, shared by every car), instead of the
            posterior. Used for races whose actual final stints followed a late
            SC restart, so the SC layer does not reuse that race's SC luck.
    """
    post = fitted.posterior

    def g(name: str) -> np.ndarray:
        return _flat_draws(post, name)[draw_index]

    r = fitted.races.index(round_number)
    t = fitted.teams.index(team)
    n = len(draw_index)
    if (round_number, driver) in fitted.driver_race:
        offset = g("driver_offset")[:, fitted.driver_race.index((round_number, driver))]
    else:
        season = (g("driver_season")[:, fitted.drivers.index(driver)]
                  if driver in fitted.drivers else g("sigma_driver") * rng.standard_normal(n))
        offset = (season + g("sigma_driver_race") * rng.standard_normal(n)
                  - g("driver_race_mean")[:, r])

    has_final_data = (fitted.final_stint_races is None
                      or round_number in fitted.final_stint_races)
    final_race = g("final_race")[:, r]
    if final_race_z is not None:
        final_race = widening * g("sigma_final_race") * final_race_z
    elif not has_final_data:
        final_race = widening * final_race
    stint_now = None
    if current_stint is not None and (round_number, driver, current_stint) in fitted.stints:
        stint_now = g("stint_effect")[:, fitted.stints.index((round_number, driver, current_stint))]

    return CarModel(
        base=g("race_base")[:, r] + offset,
        trend=g("trend")[:, r],
        lap_centre=fitted.lap_centre,
        opening=g("opening"),
        compound=g("compound_offset")[:, :, r] + g("team_compound")[:, :, r, t],
        deg=(g("deg")[:, :, r] + g("team_deg")[:, :, t]) / TYRE_SCALE,
        post_stop=g("post_stop"),
        sigma=g("sigma")[:, r],
        nu=g("nu"),
        sigma_stint=g("sigma_stint"),
        final_mean=g("mu_final") + final_race,
        sigma_stint_final=widening * g("sigma_stint_final"),
        current_stint=stint_now,
    )


@dataclass
class Draws:
    """Common random numbers: everything random in simulation ``i`` that does
    not depend on the plan. Reused for every plan in a comparison."""

    noise: np.ndarray       # (n_sims, laps_remaining): Student-t lap noise in seconds
    z_current: np.ndarray   # (n_sims,): stint in progress, if not fitted
    z_stint: np.ndarray     # (n_sims, max_stops): j-th future non-final stint
    z_final: np.ndarray     # (n_sims,): the final stint, whichever it is
    pit_loss: np.ndarray    # (n_sims, max_stops): loss of the j-th stop, seconds

    @property
    def n_sims(self) -> int:
        return len(self.z_final)


def make_draws(
    model: CarModel,
    laps_remaining: int,
    pit_losses: np.ndarray,
    max_stops: int,
    rng: np.random.Generator,
) -> Draws:
    """Draw the common random numbers for ``model.base.size`` simulations.

    Pit losses are resampled (with replacement) from ``pit_losses``, the
    circuit's observed stops, so the slow tail is kept.
    """
    n = model.base.size
    t = rng.standard_t(model.nu[:, None], size=(n, laps_remaining))
    return Draws(
        noise=model.sigma[:, None] * t,
        z_current=rng.standard_normal(n),
        z_stint=rng.standard_normal((n, max(max_stops, 1))),
        z_final=rng.standard_normal(n),
        pit_loss=rng.choice(np.asarray(pit_losses, float), size=(n, max(max_stops, 1))),
    )


def observed_pit_losses(cleaned: pd.DataFrame, round_number: int) -> np.ndarray:
    """Pit losses (s) of the circuit's 2022 green-flag stops, for resampling.

    Same definition as Phase 2: (in lap + out lap) − 2 × the driver's median
    clean lap; stops with an SC/VSC/red code on either lap or on lap 1 are
    left out.
    """
    from f1strat.clean import pit_stops

    stops = pit_stops(cleaned)
    keep = (stops["RoundNumber"] == round_number) & stops["green"] & stops["loss_s"].notna()
    losses = stops.loc[keep, "loss_s"].to_numpy()
    if len(losses) == 0:
        raise ValueError(f"no green-flag stops observed at round {round_number}")
    return losses


# --- Simulation ----------------------------------------------------------------


@dataclass
class _Stint:
    first_lap: int      # first lap of the stint still to be run
    n_laps: int
    compound: int       # index into COMPOUNDS
    life0: int          # TyreLife on first_lap
    since0: int         # laps since stop on first_lap
    from_stop: bool
    kind: str           # "current", "future" or "final"
    future_index: int   # j for z_stint[:, j] (future non-final stints)


def _stints(state: RaceState, car: CarState, plan: Plan) -> list[_Stint]:
    """Split the remaining laps into stints."""
    out = []
    first, life0, since0 = state.lap + 1, car.tyre_life + 1, car.laps_since_stop + 1
    comp, from_stop = car.compound, car.stint_started_with_stop
    for j, stop in enumerate([*plan.stops, None]):
        last = stop.lap if stop is not None else state.total_laps
        kind = "final" if stop is None else ("current" if j == 0 else "future")
        out.append(_Stint(first, last - first + 1, COMPOUNDS.index(comp), life0, since0,
                          from_stop, kind, j - 1))
        if stop is not None:
            first, life0, since0 = stop.lap + 1, stop.tyre_life, 0
            comp, from_stop = stop.compound, True
    # The stint in progress keeps its own effect even when it is also the final one.
    out[0].kind = "current_final" if not plan.stops else "current"
    return out


def _stint_effect(model: CarModel, draws: Draws, stint: _Stint) -> np.ndarray:
    """One draw per stint (shared by its laps), per simulation."""
    if stint.kind.startswith("current") and model.current_stint is not None:
        return model.current_stint
    if model.known_stints and not stint.kind.startswith("current"):
        known = model.known_stints.get((stint.first_lap, COMPOUNDS[stint.compound], stint.life0))
        if known is not None:
            return known
    if stint.kind in ("final", "current_final"):
        return model.final_mean + model.sigma_stint_final * draws.z_final
    if stint.kind == "current":
        return model.sigma_stint * draws.z_current
    return model.sigma_stint * draws.z_stint[:, stint.future_index]


def _common_laps(model: CarModel, state: RaceState, draws: Draws) -> np.ndarray:
    """(n_sims, laps_remaining): the plan-independent part of every lap."""
    laps = np.arange(state.lap + 1, state.total_laps + 1)
    opening = np.isin(laps, OPENING_LAPS).astype(float)
    return (model.base[:, None] + model.opening[:, None] * opening[None, :]
            + model.trend[:, None] * (laps - model.lap_centre)[None, :] + draws.noise)


def simulate_laps(model: CarModel, state: RaceState, driver: str, plan: Plan,
                  draws: Draws) -> np.ndarray:
    """Lap times (s), shape (n_sims, laps_remaining), pit losses on in laps.

    The lap-by-lap reference implementation.
    """
    car = state.cars[driver]
    times = _common_laps(model, state, draws)
    for st in _stints(state, car, plan):
        cols = np.arange(st.first_lap, st.first_lap + st.n_laps) - (state.lap + 1)
        ages = st.life0 + np.arange(st.n_laps)
        since = st.since0 + np.arange(st.n_laps)
        warm = (np.isin(since, POST_STOP_LAPS) & st.from_stop).astype(float)
        c = st.compound
        times[:, cols] += (model.compound[:, c, None] + model.deg[:, c, None] * ages[None, :]
                           + model.post_stop[:, c, None] * warm[None, :]
                           + _stint_effect(model, draws, st)[:, None])
    for j, stop in enumerate(plan.stops):
        times[:, stop.lap - (state.lap + 1)] += draws.pit_loss[:, j]
    return times


def plan_totals(model: CarModel, state: RaceState, driver: str, plans: Sequence[Plan],
                draws: Draws) -> np.ndarray:
    """Race time at the flag (s), shape (n_plans, n_sims), in closed form.

    Equal to ``state race_time + simulate_laps(...).sum(axis=1)`` for each
    plan, without building the lap matrix.
    """
    car = state.cars[driver]
    common = car.race_time + _common_laps(model, state, draws).sum(axis=1)
    out = np.empty((len(plans), draws.n_sims))
    for p, plan in enumerate(plans):
        total = common.copy()
        for st in _stints(state, car, plan):
            n, c = st.n_laps, st.compound
            age_sum = n * st.life0 + n * (n - 1) / 2
            warm = sum(1 for k in range(n) if st.since0 + k in POST_STOP_LAPS) if st.from_stop else 0
            total += (n * model.compound[:, c] + model.deg[:, c] * age_sum
                      + warm * model.post_stop[:, c] + n * _stint_effect(model, draws, st))
        for j in range(len(plan.stops)):
            total += draws.pit_loss[:, j]
        out[p] = total
    return out


# --- Strategy search -----------------------------------------------------------


@dataclass
class SearchResult:
    """Shortlisted plans with their Monte Carlo race-time distributions."""

    plans: list[Plan]
    totals: np.ndarray          # (n_plans, n_sims), seconds
    n_legal: int                # legal plans scored on expected time
    labels: list[str] = field(default_factory=list)
    #: Plans removed by the tyre-age cap (otherwise legal), and the best few
    #: of them as (mean race time, plan), best first.
    n_capped: int = 0
    capped: list[tuple[float, Plan]] = field(default_factory=list)

    def table(self, reference: Plan | None = None) -> pd.DataFrame:
        """Ranked summary: mean, spread, quantiles, P(fastest), gap to best mean.

        With ``reference`` (one of the plans), adds the paired comparison under
        common random numbers: mean time saved vs the reference, its 10th–90th
        percentile, and the share of simulations in which the plan is faster.
        """
        best = self.totals.argmin(axis=0)
        mean = self.totals.mean(axis=1)
        df = pd.DataFrame({
            "plan": self.labels,
            "stops": [len(p.stops) for p in self.plans],
            "mean_s": mean,
            "sd_s": self.totals.std(axis=1),
            "p10_s": np.quantile(self.totals, 0.1, axis=1),
            "p90_s": np.quantile(self.totals, 0.9, axis=1),
            "p_fastest": [(best == i).mean() for i in range(len(self.plans))],
        })
        df["gap_to_best_s"] = df["mean_s"] - df["mean_s"].min()
        if reference is not None:
            ref = self.totals[self.plans.index(reference)]
            saved = ref[None, :] - self.totals
            df["saved_vs_ref_s"] = saved.mean(axis=1)
            df["saved_p10_s"] = np.quantile(saved, 0.1, axis=1)
            df["saved_p90_s"] = np.quantile(saved, 0.9, axis=1)
            df["p_beats_ref"] = (saved > 0).mean(axis=1)
        return df.sort_values("mean_s").reset_index(drop=True)


def _set_sequences(counts: Counter, k: int) -> Iterator[tuple]:
    """Ordered sequences of ``k`` sets drawn without replacement from ``counts``
    (identical sets are interchangeable, so each sequence appears once)."""
    if k == 0:
        yield ()
        return
    for item in sorted(counts):
        if counts[item] > 0:
            counts[item] -= 1
            for rest in _set_sequences(counts, k - 1):
                yield (item, *rest)
            counts[item] += 1


def _effect_means(model: CarModel, draws: Draws, max_stops: int) -> dict[str, float]:
    """Sample means (over simulations) of each kind of stint effect."""
    final = model.final_mean + model.sigma_stint_final * draws.z_final
    current = (model.current_stint if model.current_stint is not None
               else model.sigma_stint * draws.z_current)
    out = {
        "current": float(current.mean()),
        "current_final": float((model.current_stint if model.current_stint is not None
                                else final).mean()),
        "final": float(final.mean()),
    }
    for j in range(max_stops):
        out[f"future{j}"] = float((model.sigma_stint * draws.z_stint[:, j]).mean())
    return out


def _stint_mean_cost(n, first_since, from_stop, life0, a_c, b_c, p_c, e):
    """Mean stint cost over simulations, vectorised over stints.

    ``n * E[compound] + E[deg] * sum(ages) + E[post_stop] * warm laps
    + n * E[stint effect]``; each term is a known count times one
    per-simulation quantity, so its mean is exact.
    """
    age_sum = n * life0 + n * (n - 1) / 2
    last_since = first_since + n - 1
    warm = sum(((first_since <= k) & (k <= last_since)).astype(float) for k in POST_STOP_LAPS)
    warm = warm * from_stop
    return n * a_c + b_c * age_sum + p_c * warm + n * e


def search(
    model: CarModel,
    state: RaceState,
    driver: str,
    draws: Draws,
    max_stops: int = 3,
    compounds: Sequence[str] = DRY_COMPOUNDS,
    used_life: int | None = DEFAULT_USED_LIFE,
    min_stint: int = 1,
    per_stratum: int = 3,
    include: Sequence[Plan] = (),
    allowed_sets: Sequence[tuple[str, int]] | None = None,
    max_age: dict[str, int] | None = None,
    first_laps: tuple[int, int] | None = None,
    first_sets: Collection[tuple[str, int]] | None = None,
    n_capped_kept: int = 5,
) -> SearchResult:
    """Score every legal plan on expected race time, then Monte Carlo a
    stratified shortlist.

    - **Legal plans:** 0..``max_stops`` more stops, any pit lap, two-compound
      rule. Sets come from ``state.sets[driver]`` when known: each set can be
      fitted once. Otherwise they come from ``allowed_sets`` (unlimited
      (compound, out-lap TyreLife) pairs), else every compound new or used
      without limit.
    - **Scoring:** each plan's exact mean race time over the same simulations,
      vectorised by stint.
    - **Stratified shortlist:** the best ``per_stratum`` plans for every stop
      count and compound sequence, so that once traffic is modelled the
      no-traffic ranking cannot crowd out whole strategy families.
    - The shortlist, plus any ``include`` plans (e.g. what the team actually
      did), gets full distributions under common random numbers.
    - **Tyre-age cap** (``max_age``, Phase 6): no stint, including the one on
      the car, may reach a TyreLife above the cap for its compound. Removed
      plans are counted (``n_capped``) and the best few kept (``capped``).
    - **Family filters** (Phase 6): ``first_laps`` (inclusive lap range of the
      first stop) and ``first_sets`` ((compound, out-lap TyreLife) allowed at
      the first stop) restrict the search to plans with at least one stop.
    """
    if max_stops > draws.z_stint.shape[1]:
        raise ValueError("draws were made for fewer stops than max_stops")
    car = state.cars[driver]
    e = _effect_means(model, draws, max_stops)
    a = model.compound.mean(axis=0)
    b = model.deg.mean(axis=0)
    ps = model.post_stop.mean(axis=0)
    pit_mean = draws.pit_loss.mean(axis=0)
    lives = (1,) if used_life is None else (1, used_life)
    inventory = state.sets.get(driver) if state.sets is not None else None
    sets = ([(COMPOUNDS.index(c), life) for c, life in allowed_sets] if allowed_sets is not None
            else [(COMPOUNDS.index(c), life) for c in compounds for life in lives])
    stint_start = state.lap - car.laps_since_stop
    laps = np.arange(state.lap + 1, state.total_laps)
    cur_c = COMPOUNDS.index(car.compound)
    used_before = set(car.compounds_used) | {car.compound}

    strata: dict[tuple, list[tuple[float, tuple]]] = {}
    n_legal = 0
    n_capped = 0
    capped: list[tuple[float, tuple]] = []
    for k in range(max_stops + 1):
        combos = (np.array(list(itertools.combinations(laps, k)), dtype=int)
                  if k else np.zeros((1, 0), dtype=int))
        bounds = np.column_stack([np.full(len(combos), stint_start), combos,
                                  np.full(len(combos), state.total_laps)])
        ok = (np.diff(bounds, axis=1) >= min_stint).all(axis=1)
        combos = combos[ok]
        if len(combos) == 0:
            continue
        ends = np.column_stack([combos, np.full(len(combos), state.total_laps)])
        firsts = np.column_stack([np.full(len(combos), state.lap + 1), combos + 1])
        n_laps = ends - firsts + 1
        if first_laps is not None:
            if k == 0:
                continue
            keep = (combos[:, 0] >= first_laps[0]) & (combos[:, 0] <= first_laps[1])
            combos, n_laps = combos[keep], n_laps[keep]
            if len(combos) == 0:
                continue
        cur_ok = np.ones(len(combos), dtype=bool)
        if max_age is not None:
            cur_ok = car.tyre_life + n_laps[:, 0] <= max_age.get(car.compound, 10**6)
        cur_kind = "current_final" if k == 0 else "current"
        cur = _stint_mean_cost(n_laps[:, 0], car.laps_since_stop + 1,
                               float(car.stint_started_with_stop), car.tyre_life + 1,
                               a[cur_c], b[cur_c], ps[cur_c], e[cur_kind])
        base = cur + pit_mean[:k].sum()
        sequences = (_set_sequences(Counter((COMPOUNDS.index(c), life) for c, life in inventory), k)
                     if inventory is not None else itertools.product(sets, repeat=k))
        for seq in sequences:
            names = {COMPOUNDS[c] for c, _ in seq}
            if len((used_before | names) & set(DRY_COMPOUNDS)) < 2:
                continue
            if first_sets is not None and k > 0 and (COMPOUNDS[seq[0][0]], seq[0][1]) \
                    not in set(first_sets):
                continue
            total = base.copy()
            age_ok = cur_ok.copy()
            for j, (c, life) in enumerate(seq):
                kind = "final" if j == k - 1 else f"future{j}"
                total += _stint_mean_cost(n_laps[:, j + 1], 0, 1.0, life,
                                          a[c], b[c], ps[c], e[kind])
                if max_age is not None:
                    age_ok &= life + n_laps[:, j + 1] - 1 <= max_age.get(COMPOUNDS[c], 10**6)
            if not age_ok.all():
                n_capped += int((~age_ok).sum())
                for i in np.argsort(np.where(age_ok, np.inf, total))[:n_capped_kept]:
                    if not age_ok[i]:
                        capped.append((float(total[i]), (tuple(combos[i]), seq)))
                total = np.where(age_ok, total, np.inf)
                if not age_ok.any():
                    continue
            n_legal += int(age_ok.sum())
            best = [i for i in np.argsort(total)[:per_stratum] if np.isfinite(total[i])]
            key = (k, tuple(COMPOUNDS[c] for c, _ in seq))
            strata.setdefault(key, []).extend(
                (float(total[i]), (tuple(combos[i]), seq)) for i in best)

    candidates = []
    for items in strata.values():
        candidates += sorted(items, key=lambda x: x[0])[:per_stratum]
    candidates.sort(key=lambda x: x[0])
    chosen = [Plan(tuple(Stop(int(lap), COMPOUNDS[c], life) for lap, (c, life) in zip(lps, seq)))
              for _, (lps, seq) in candidates]
    chosen += [p for p in include if p not in chosen]
    totals = plan_totals(model, state, driver, chosen, draws)
    capped.sort(key=lambda x: x[0])
    capped_plans = [(t, Plan(tuple(Stop(int(lap), COMPOUNDS[c], life)
                                   for lap, (c, life) in zip(lps, seq))))
                    for t, (lps, seq) in capped[:n_capped_kept]]
    return SearchResult(chosen, totals, n_legal, [p.label(car.compound) for p in chosen],
                        n_capped, capped_plans)


# --- Part B: the whole field ----------------------------------------------------
#
# Every car is simulated together, lap by lap and vectorised over simulations:
#
# - Free-air lap times come from each car's lap model (no dirty_air_penalty:
#   following is modelled here, so it must not be counted twice).
# - Green lap: cars are processed in order of race time at the start of the
#   lap. A car that would finish the lap closer than ``follow_gap_s`` to the
#   car directly ahead on track is held at ``follow_gap_s`` behind, unless it
#   passes. It can only try to pass if it started the lap within 1 s of that
#   car (as in the 2022 events the pass model was fitted on), with probability
#   ``sigmoid(a[circuit] + b · pace gap)``, where the pace gap is the two cars'
#   free-air lap times without this lap's noise (the fitted pace gap was the
#   model mean too). A pass gains at most one place this lap. Cars on an in
#   lap or out lap are left out of the battles, as in the fitted events (the
#   whole pit loss sits on the in lap, so an out lap would otherwise be run at
#   full fresh-tyre pace through the field).
# - SC lap: no passing; the queue leader laps at ``sc_lap_ratio`` × the
#   field's free-air lap. Behind it, gaps are kept on the first SC lap and then
#   shrink by ``sc_close_ratio`` per lap, down to ``sc_gap_s`` (Phase 5; with
#   ``sc_close_ratio=None`` each car laps at ``sc_catchup_ratio`` × its own
#   until it is ``sc_gap_s`` behind, so the field bunches in one lap).
#   Pit loss is scaled by ``sc_pit_ratio``.
# - VSC lap: no passing; every car laps at ``vsc_lap_ratio`` × its free-air
#   lap (gaps are kept, not closed); pit loss is scaled by ``vsc_pit_ratio``.
# - SC/VSC starts and lengths are drawn per simulation from the 2022 rates;
#   a state that is already under SC/VSC continues it for a drawn remaining
#   length.
# - Common random numbers: lap noise, stint and pit-loss draws (per car), the
#   SC/VSC schedule, the pass coin-flips and the choice among a rival's plan
#   options are fixed per simulation, so every focal plan meets identical luck.

GREEN, SC, VSC = 0, 1, 2


@dataclass
class Field:
    """Everything for simulating the whole field from one ``RaceState``."""

    state: RaceState
    drivers: list[str]
    models: dict[str, CarModel]
    draws: dict[str, Draws]
    status: np.ndarray          # (n_sims, laps_remaining): GREEN / SC / VSC
    pass_u: np.ndarray          # (n_sims, laps_remaining, n_cars) uniforms
    choice_u: np.ndarray        # (n_sims, n_cars) uniforms for rival plan options
    params: "SimParams"
    pit_losses: np.ndarray
    #: Posterior draw used by each simulation (Phase 6 sensitivity checks).
    draw_index: np.ndarray | None = None

    @property
    def n_sims(self) -> int:
        return self.status.shape[0]


def neutral_schedule(state: RaceState, neutral, n_sims: int,
                     rng: np.random.Generator) -> np.ndarray:
    """SC/VSC status per simulation and remaining lap (plan-independent).

    A state already under SC/VSC continues it: the remaining length is drawn
    from the observed lengths longer than the laps already run. On green laps,
    an SC starts with probability ``p_sc_per_lap`` and a VSC with
    ``p_vsc_per_lap``, each with a length drawn from the observed lengths.
    After any SC/VSC the next lap is green (the restart), so periods never run
    back to back.
    """
    laps = state.laps_remaining
    out = np.zeros((n_sims, laps), dtype=np.int8)
    sc_len = np.asarray(neutral.sc_lengths)
    vsc_len = np.asarray(neutral.vsc_lengths)
    for i in range(n_sims):
        j = 0
        if state.track_status in ("sc", "vsc"):
            lengths = sc_len if state.track_status == "sc" else vsc_len
            longer = lengths[lengths > state.neutral_laps]
            remaining = (rng.choice(longer) - state.neutral_laps) if len(longer) else 1
            code = SC if state.track_status == "sc" else VSC
            out[i, :remaining] = code
            j = remaining + 1
        while j < laps:
            u = rng.random()
            if u < neutral.p_sc_per_lap:
                n = int(rng.choice(sc_len))
                out[i, j:j + n] = SC
                j += n + 1
            elif u < neutral.p_sc_per_lap + neutral.p_vsc_per_lap:
                n = int(rng.choice(vsc_len))
                out[i, j:j + n] = VSC
                j += n + 1
            else:
                j += 1
    return out


def late_restart_rounds(race: pd.DataFrame, window: int = 20) -> list[int]:
    """Races with a late safety-car restart: an SC that ended, with racing
    resuming, within the last ``window`` laps.

    Final stints after such a restart can be sprint laps on fresh tyres, which
    the fitted ``final_race`` of that race may encode (ROADMAP Phase 4 note).
    The field simulator draws its own SCs, so for these races it can draw
    ``final_race`` from the prior instead (``build_field(final_race_prior=True)``).
    """
    out = []
    for rnd, r in race.groupby("RoundNumber"):
        r = r.sort_values("LapNumber")
        last = int(r["LapNumber"].max())
        sc_laps = r.loc[r["SafetyCar"], "LapNumber"]
        if sc_laps.empty:
            continue
        ends = [int(lap) for lap, nxt in zip(r["LapNumber"], r["SafetyCar"].shift(-1, fill_value=False))
                if lap in set(sc_laps) and not nxt]
        if any(last - window <= e < last for e in ends):
            out.append(int(rnd))
    return out


def build_field(
    fitted: BayesFit,
    state: RaceState,
    pit_losses: np.ndarray,
    params: "SimParams",
    n_sims: int = 1000,
    seed: int = 0,
    max_stops: int = 3,
    widening: float = DEFAULT_WIDENING,
    use_fitted_current_stint: bool = True,
    final_race_prior: bool = False,
    status: np.ndarray | None = None,
) -> Field:
    """Draw everything plan-independent for every car (common random numbers).

    All cars in simulation ``i`` share one posterior draw. ``final_race_prior``
    draws the race's ``final_race`` from its prior (see :func:`car_model`).
    ``status`` (GREEN / SC / VSC per remaining lap) forces the SC/VSC schedule
    in every simulation instead of drawing it (Phase 5 replay backtests).
    """
    rng = np.random.default_rng(seed)
    total = fitted.posterior.sizes["chain"] * fitted.posterior.sizes["draw"]
    idx = rng.choice(total, size=n_sims, replace=n_sims > total)
    fr_z = rng.standard_normal(n_sims) if final_race_prior else None
    drivers = sorted(state.cars, key=lambda d: state.cars[d].position)
    models, draws = {}, {}
    for d in drivers:
        car = state.cars[d]
        stint = car.stint if (use_fitted_current_stint and state.lap > 1) else None
        models[d] = car_model(fitted, state.round_number, d, car.team, idx, rng,
                              current_stint=stint, widening=widening, final_race_z=fr_z)
        draws[d] = make_draws(models[d], state.laps_remaining, pit_losses, max_stops, rng)
    drawn = neutral_schedule(state, params.neutral, n_sims, rng)
    if status is not None:
        drawn = np.broadcast_to(np.asarray(status, np.int8), drawn.shape).copy()
    status = drawn
    pass_u = rng.random((n_sims, state.laps_remaining, len(drivers)))
    choice_u = rng.random((n_sims, len(drivers)))
    return Field(state, drivers, models, draws, status, pass_u, choice_u, params,
                 np.asarray(pit_losses, float), idx)


def _car_laps(field: Field, driver: str, plan: Plan) -> tuple[np.ndarray, np.ndarray]:
    """(free-air lap times without pit loss, green pit loss on in laps)."""
    model, draws = field.models[driver], field.draws[driver]
    laps = simulate_laps(model, field.state, driver, plan, draws)
    pit = np.zeros_like(laps)
    for j, stop in enumerate(plan.stops):
        col = stop.lap - (field.state.lap + 1)
        pit[:, col] = draws.pit_loss[:, j]
    return laps - pit, pit


PlanOptions = Plan | Sequence[tuple[Plan, float]]


def _select(field: Field, c: int, options: PlanOptions) -> tuple[np.ndarray, np.ndarray]:
    """Lap and pit matrices for car ``c``, choosing one option per simulation."""
    driver = field.drivers[c]
    if isinstance(options, Plan):
        return _car_laps(field, driver, options)
    plans = [p for p, _ in options]
    w = np.array([w for _, w in options], float)
    cum = np.cumsum(w / w.sum())
    pick = np.searchsorted(cum, field.choice_u[:, c], side="right").clip(0, len(plans) - 1)
    laps = np.empty((field.n_sims, field.state.laps_remaining))
    pit = np.empty_like(laps)
    for k, plan in enumerate(plans):
        rows = pick == k
        if rows.any():
            lk, pk = _car_laps(field, driver, plan)
            laps[rows], pit[rows] = lk[rows], pk[rows]
    return laps, pit


@dataclass
class FieldResult:
    """Race times (s) and finishing positions (1 = winner), per simulation and car."""

    drivers: list[str]
    times: np.ndarray          # (n_sims, n_cars)
    positions: np.ndarray      # (n_sims, n_cars)
    passes: int = 0
    holds: int = 0
    attempts: int = 0
    attempt_pace: np.ndarray | None = None   # pace gaps of all pass attempts
    #: With ``record=True``: race time at the end of every simulated lap,
    #: (n_sims, laps_remaining, n_cars), +inf once a car has retired.
    history: np.ndarray | None = None
    #: With ``record=True``: car on an in or out lap, same shape.
    pit_laps: np.ndarray | None = None


def simulate_field(field: Field, plans: dict[str, PlanOptions],
                   retire: dict[str, int] | None = None,
                   record: bool = False) -> FieldResult:
    """Simulate the rest of the race for every car.

    Args:
        field: From :func:`build_field`.
        plans: Per driver, a fixed :class:`Plan` or weighted options
            ``[(plan, weight), ...]`` (one picked per simulation, the same
            pick for every call on this field). Every car needs an entry.
        retire: Per driver, the last lap it completes (Phase 5 backtests:
            actual retirements). A retired car leaves the track (race time
            +inf) and is classified behind every running car.
        record: Keep each lap's race times and in/out-lap flags
            (``FieldResult.history``, ``FieldResult.pit_laps``).
    """
    st, prm = field.state, field.params
    n, L, C = field.n_sims, st.laps_remaining, len(field.drivers)
    free = np.empty((n, L, C))
    pit = np.empty((n, L, C))
    noise = np.empty((n, L, C))
    for c, d in enumerate(field.drivers):
        free[:, :, c], pit[:, :, c] = _select(field, c, plans[d])
        noise[:, :, c] = field.draws[d].noise
    a_circ = prm.passing.circuit.get(st.round_number, prm.passing.intercept) + prm.passing.shift
    b, gap = prm.passing.slope, prm.passing.follow_gap_s
    dirty = prm.passing.dirty_air_s
    held_q = np.asarray(prm.passing.held_gaps, float)  # quantiles of real held gaps
    neu = prm.neutral
    T = np.array([st.cars[d].race_time for d in field.drivers], float)[None, :].repeat(n, 0)
    rows = np.arange(n)
    passes = holds = attempts = 0
    attempt_pace = []
    last = np.array([(retire or {}).get(d, st.total_laps) for d in field.drivers])
    # SC lap number within its period (1 = first SC lap), per simulation and lap.
    sc_lap = np.zeros((n, L), dtype=int)
    run = np.full(n, st.neutral_laps if st.track_status == "sc" else 0)
    for j in range(L):
        run = np.where(field.status[:, j] == SC, run + 1, 0)
        sc_lap[:, j] = run
    history = np.empty((n, L, C)) if record else None
    pit_laps = np.zeros((n, L, C), dtype=bool) if record else None
    for j in range(L):
        F, P, s = free[:, j, :], pit[:, j, :], field.status[:, j]
        pace_mean = F - noise[:, j, :]
        order = np.argsort(T, axis=1)
        A = np.empty_like(T)
        green, sc, vsc = s == GREEN, s == SC, s == VSC
        # Free arrival (green); SC/VSC laps are overwritten below.
        A[:] = T + F + P
        pitting = P > 0
        out_lap = (pit[:, j - 1, :] > 0) if j > 0 else np.zeros_like(pitting)
        if j == 0:
            out_lap = np.array([st.cars[d].laps_since_stop == 0 and st.cars[d].stint > 1
                                for d in field.drivers])[None, :].repeat(n, 0)
        gone = np.isinf(T)
        # --- green: battles in order of position ---------------------------------
        tail = np.full(n, -np.inf)
        tail_car = np.zeros(n, dtype=int)
        second = np.full(n, -np.inf)
        for r in range(C):
            car = order[:, r]
            a_r = A[rows, car]
            active = green & ~pitting[rows, car] & ~out_lap[rows, car] & ~gone[rows, car]
            if r > 0:
                # Dirty air: started the lap within 1 s of the car directly ahead.
                with np.errstate(invalid="ignore"):  # retired cars: inf - inf
                    behind = (T[rows, car] - T[rows, order[:, r - 1]]) < 1.0
                a_r = a_r + dirty * (active & behind)
                cond = active & (a_r < tail + gap) & np.isfinite(tail)
                close = (T[rows, car] - T[rows, tail_car]) < 1.0
                pace = pace_mean[rows, tail_car] - pace_mean[rows, car]
                p = 1.0 / (1.0 + np.exp(-(a_circ + b * pace)))
                ok = cond & close & (field.pass_u[rows, j, car] < p)
                held = cond & ~ok
                attempts += int((cond & close).sum())
                attempt_pace.append(pace[cond & close])
                if held_q.size:
                    # Held at a gap drawn from the real gaps of cars that did
                    # not pass; the coin-flip's leftover is uniform given held.
                    u = field.pass_u[rows, j, car]
                    u = np.where(close, (u - p) / np.maximum(1 - p, 1e-9), u).clip(0, 1)
                    g_held = np.interp(u, np.linspace(0, 1, held_q.size), held_q)
                    held = cond & ~ok & (a_r < tail + g_held)
                    a_r = np.where(held, tail + g_held, a_r)
                else:
                    a_r = np.where(held, tail + gap, a_r)
                a_r = np.where(ok, np.maximum(a_r, second + gap), a_r)
                passes += int(ok.sum())
                holds += int(held.sum())
            A[rows, car] = np.where(green, a_r, A[rows, car])
            upd = active
            ahead = upd & (a_r >= tail)
            second = np.where(ahead, tail, np.where(upd, np.maximum(second, a_r), second))
            tail_car = np.where(ahead, car, tail_car)
            tail = np.where(ahead, a_r, tail)
        # --- SC: queue behind the leader, gaps close to sc_gap -------------------
        if sc.any():
            ref = np.median(F, axis=1)
            prev = np.full(n, -np.inf)
            # Gradual closing (Phase 5): gaps kept on the first SC lap, then
            # multiplied by sc_close_ratio per lap, never below sc_gap.
            ratio = (None if neu.sc_close_ratio is None
                     else np.where(sc_lap[:, j] <= 1, 1.0, neu.sc_close_ratio))
            for r in range(C):
                car = order[:, r]
                if r == 0:
                    q = T[rows, car] + neu.sc_lap_ratio * ref
                elif ratio is None:
                    own = T[rows, car] + neu.sc_catchup_ratio * F[rows, car]
                    q = np.maximum(own, prev + prm.sc_gap_s)
                else:
                    with np.errstate(invalid="ignore"):  # retired cars sort last
                        gap_now = T[rows, car] - T[rows, order[:, r - 1]]
                    q = prev + np.maximum(prm.sc_gap_s, ratio * gap_now)
                prev = q
                A[rows, car] = np.where(sc, q + neu.sc_pit_ratio * P[rows, car], A[rows, car])
        # --- VSC: everyone slows by the same ratio, no passing -------------------
        if vsc.any():
            prev = np.full(n, -np.inf)
            for r in range(C):
                car = order[:, r]
                own = T[rows, car] + neu.vsc_lap_ratio * F[rows, car]
                q = np.maximum(own, prev + 0.05)
                prev = np.where(pitting[rows, car], prev, q)
                A[rows, car] = np.where(vsc, q + neu.vsc_pit_ratio * P[rows, car], A[rows, car])
        A[:, st.lap + 1 + j > last] = np.inf
        T = A
        if record:
            history[:, j, :] = T
            pit_laps[:, j, :] = pitting | out_lap
    positions = np.argsort(np.argsort(T, axis=1), axis=1) + 1
    return FieldResult(field.drivers, T, positions, passes, holds, attempts,
                       np.concatenate(attempt_pace) if attempt_pace else None,
                       history, pit_laps)


@dataclass
class FieldSearchResult:
    """A focal car's candidate plans, each simulated with the whole field."""

    focal: str
    plans: list[Plan]
    labels: list[str]
    positions: np.ndarray      # (n_plans, n_sims, n_cars)
    times: np.ndarray          # (n_plans, n_sims, n_cars)
    drivers: list[str]

    def table(self, reference: Plan | None = None) -> pd.DataFrame:
        """Focal car per plan: finishing position and race-time summaries.

        With ``reference``, adds paired comparisons under common random numbers:
        P(finish ahead of where the reference plan finishes), and mean places
        gained.
        """
        c = self.drivers.index(self.focal)
        pos = self.positions[:, :, c]
        t = self.times[:, :, c]
        df = pd.DataFrame({
            "plan": self.labels,
            "stops": [len(p.stops) for p in self.plans],
            "mean_pos": pos.mean(axis=1),
            "p_win": (pos == 1).mean(axis=1),
            "p_podium": (pos <= 3).mean(axis=1),
            "p10_pos": np.quantile(pos, 0.1, axis=1),
            "p90_pos": np.quantile(pos, 0.9, axis=1),
            "mean_time_s": t.mean(axis=1),
        })
        if reference is not None:
            ref = pos[self.plans.index(reference)]
            df["places_gained"] = (ref[None, :] - pos).mean(axis=1)
            df["p_ahead_of_ref"] = (pos < ref[None, :]).mean(axis=1)
            df["p_behind_ref"] = (pos > ref[None, :]).mean(axis=1)
        return df.sort_values(["mean_pos", "mean_time_s"]).reset_index(drop=True)

    def field_positions(self, plan: Plan) -> pd.DataFrame:
        """Finishing-position distribution for every car under one focal plan."""
        k = self.plans.index(plan)
        pos = self.positions[k]
        return pd.DataFrame({
            "driver": self.drivers,
            "mean_pos": pos.mean(axis=0),
            "p_win": (pos == 1).mean(axis=0),
            "p_podium": (pos <= 3).mean(axis=0),
            "p10_pos": np.quantile(pos, 0.1, axis=0),
            "p90_pos": np.quantile(pos, 0.9, axis=0),
        }).sort_values("mean_pos").reset_index(drop=True)


def field_search(field: Field, focal: str, candidates: Sequence[Plan],
                 rival_plans: dict[str, PlanOptions]) -> FieldSearchResult:
    """Simulate each candidate plan for ``focal`` against fixed rival plans.

    The same field draws are reused for every candidate (common random
    numbers), so differences between plans are not luck.
    """
    positions, times = [], []
    for plan in candidates:
        res = simulate_field(field, {**rival_plans, focal: plan})
        positions.append(res.positions)
        times.append(res.times)
    car = field.state.cars[focal]
    return FieldSearchResult(focal, list(candidates), [p.label(car.compound) for p in candidates],
                             np.stack(positions), np.stack(times), field.drivers)


def rival_plan_options(field: Field, driver: str, top: int = 3, temperature_s: float = 3.0,
                       max_stops: int = 3, min_stint: int = 5,
                       max_age: dict[str, int] | None = None) -> list[tuple[Plan, float]]:
    """A rival's plan options from its own single-car search (no hindsight).

    The ``top`` plans by expected time (one per compound sequence, best
    first), weighted by ``exp(−(mean − best) / temperature_s)``.
    """
    res = search(field.models[driver], field.state, driver, field.draws[driver],
                 max_stops=max_stops, min_stint=min_stint, per_stratum=1, max_age=max_age)
    means = res.totals.mean(axis=1)
    order = np.argsort(means)[:top]
    w = np.exp(-(means[order] - means[order[0]]) / temperature_s)
    return [(res.plans[i], float(wi)) for i, wi in zip(order, w)]
