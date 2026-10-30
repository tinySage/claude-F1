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
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

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


def car_model(
    fitted: BayesFit,
    round_number: int,
    driver: str,
    team: str,
    draw_index: np.ndarray,
    rng: np.random.Generator,
    current_stint: int | None = None,
    widening: float = DEFAULT_WIDENING,
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
    if not has_final_data:
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
            total = base.copy()
            for j, (c, life) in enumerate(seq):
                kind = "final" if j == k - 1 else f"future{j}"
                total += _stint_mean_cost(n_laps[:, j + 1], 0, 1.0, life,
                                          a[c], b[c], ps[c], e[kind])
            n_legal += len(total)
            best = np.argsort(total)[:per_stratum]
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
    return SearchResult(chosen, totals, n_legal, [p.label(car.compound) for p in chosen])
