"""Phase 5 backtests: rerun 2022 races with what actually happened as inputs.

Each backtest starts from the actual state at the end of lap 1 (lap 1 is not
modelled; ``RaceState`` starts at lap >= 1, so the grid and the first lap are
taken as they happened). Every car then follows its actual pit laps and
compounds (:func:`f1strat.simulator.actual_plan`), and a car that retired
leaves at its actual last lap. Two modes:

- ``replay``: the actual SC/VSC laps are forced in every simulation, so the
  backtest tests pace, traffic and positions;
- ``random``: SC/VSC periods are drawn from the model, so it also tests the
  SC/VSC rates.

The lap model is the committed thinned **full** posterior, which includes each
race's own laps: lap-time accuracy was tested out of sample in Phase 3, and
these backtests test the simulator's layers on top of it (traffic, passing,
SC/VSC, pit losses, interval width). Every stint effect is drawn, as it would
be in a case study, not taken from the fit.

The same counters are applied to the actual race and to every simulation:

- **on-track passes** (:func:`count_passes`): one car ahead of another at the
  end of lap ``L − 1`` and behind it at the end of lap ``L``, on a green lap
  after lap 1, with neither car on an in or out lap and both still running.
  Order is by race time at the end of each lap number, so lapping (a leader
  passing a backmarker a lap down) is never a pass;
- **close following** (:func:`close_share`): share of car-laps (green, after
  lap 1, not an in/out lap, not the leader) ending within 1 s of the car
  directly ahead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from f1strat.simulator import (
    DEFAULT_WIDENING,
    GREEN,
    SC,
    VSC,
    Plan,
    RaceState,
    actual_plan,
    build_field,
    late_restart_rounds,
    observed_pit_losses,
    simulate_field,
)

#: Dry 2022 races outside the case studies (Monaco 7, Britain 10, Hungary 13)
#: and Austria (11, missing compounds; ROADMAP Decisions log).
BACKTEST_ROUNDS: tuple[int, ...] = (1, 2, 3, 5, 6, 8, 9, 12, 14, 15, 16, 19, 20, 21, 22)
CLOSE_GAP_S = 1.0


# --- What actually happened ----------------------------------------------------


@dataclass
class Actual:
    """One 2022 race as it happened, on the simulator's lap grid."""

    round_number: int
    total_laps: int
    drivers: list[str]           # cars running at the end of lap 1
    times: np.ndarray            # (total_laps + 1, n_cars): session time at the end of
                                 # lap L (row L), +inf after retiring; row 0 unused
    pit_laps: np.ndarray         # (total_laps + 1, n_cars): in or out lap
    status: np.ndarray           # (total_laps + 1,): GREEN / SC / VSC per race lap
    last_lap: dict[str, int]     # laps completed
    finished: dict[str, bool]    # running at the flag (incl. lapped cars)
    order: list[str]             # finishers in finishing order
    gap_to_winner: dict[str, float] = field(default_factory=dict)  # lead-lap finishers


def race_status(race: pd.DataFrame, round_number: int) -> np.ndarray:
    """GREEN / SC / VSC per race lap (index = lap number; red flags count as SC)."""
    r = race[race["RoundNumber"] == round_number].set_index("LapNumber").sort_index()
    out = np.full(int(r.index.max()) + 1, GREEN, dtype=np.int8)
    for lap, row in r.iterrows():
        if row["SafetyCar"] or row["RedFlag"]:
            out[lap] = SC
        elif row["VSC"]:
            out[lap] = VSC
    return out


def actual_race(raw: pd.DataFrame, race: pd.DataFrame, round_number: int) -> Actual:
    """Times, pit laps, retirements and the finishing order of a 2022 race.

    A car finished if its last lap ended at or after the winner's (lapped cars
    finish after the winner); otherwise it retired after its last lap. Finishers
    are ordered by laps completed, then by the time they crossed the line. Time
    penalties are not applied (the simulator has none either).
    """
    laps = raw[raw["RoundNumber"] == round_number]
    status = race_status(race, round_number)
    total = len(status) - 1
    drivers = sorted(laps.loc[laps["LapNumber"] == 1, "Driver"].unique())
    times = np.full((total + 1, len(drivers)), np.inf)
    pit = np.zeros((total + 1, len(drivers)), dtype=bool)
    last, end = {}, {}
    for c, d in enumerate(drivers):
        mine = laps[laps["Driver"] == d].sort_values("LapNumber")
        # A car that stopped on track gets a session time for its unfinished
        # last lap but no lap time or position (e.g. Pérez, Bahrain): drop
        # trailing laps like that.
        done = mine["LapTime"].notna() | mine["Position"].notna() | (mine["LapNumber"] == 1)
        last_done = mine.loc[done, "LapNumber"].max()
        mine = mine[(mine["LapNumber"] <= min(total, last_done))]
        for _, row in mine.iterrows():
            L = int(row["LapNumber"])
            if pd.notna(row["Time"]):
                times[L, c] = row["Time"].total_seconds()
            pit[L, c] = pd.notna(row["PitInTime"]) or (L > 1 and pd.notna(row["PitOutTime"]))
        last[d] = int(mine["LapNumber"].max())
        end[d] = times[last[d], c]
    winner_end = min(end[d] for d in drivers if last[d] == total)
    finished = {d: bool(end[d] >= winner_end - 1e-6) for d in drivers}
    order = sorted((d for d in drivers if finished[d]), key=lambda d: (-last[d], end[d]))
    gaps = {d: end[d] - winner_end for d in order if last[d] == total}
    return Actual(round_number, total, drivers, times, pit, status, last, finished,
                  order, gaps)


# --- Counters shared by the actual race and the simulations --------------------


def count_passes(times: np.ndarray, pit_laps: np.ndarray, green: np.ndarray) -> np.ndarray:
    """On-track passes per simulation (see module docstring).

    Args:
        times: (n, laps + 1, cars) race time at the end of each lap; slice 0 is
            the starting lap (passes are counted from slice 1 on).
        pit_laps: (n, laps + 1, cars) in or out lap.
        green: (laps + 1,) or (n, laps + 1): the lap is green (and counts).
    """
    n, L1, C = times.shape
    green = np.broadcast_to(green, (n, L1))
    out = np.zeros(n, dtype=int)
    for j in range(1, L1):
        if not green[:, j].any():
            continue
        prev, now = times[:, j - 1, :], times[:, j, :]
        ok = (np.isfinite(prev) & np.isfinite(now) & ~pit_laps[:, j, :]
              & green[:, j, None])
        before = prev[:, :, None] < prev[:, None, :]
        after = now[:, :, None] < now[:, None, :]
        pair = ok[:, :, None] & ok[:, None, :]
        out += (before & ~after & pair).sum(axis=(1, 2))
    return out


def close_share(times: np.ndarray, pit_laps: np.ndarray, green: np.ndarray,
                gap_s: float = CLOSE_GAP_S) -> tuple[np.ndarray, np.ndarray]:
    """(close car-laps, counted car-laps) per simulation (see module docstring)."""
    n, L1, C = times.shape
    green = np.broadcast_to(green, (n, L1))
    close = np.zeros(n, dtype=int)
    total = np.zeros(n, dtype=int)
    for j in range(1, L1):
        if not green[:, j].any():
            continue
        t = times[:, j, :]
        order = np.argsort(t, axis=1)
        ts = np.take_along_axis(t, order, axis=1)
        ps = np.take_along_axis(pit_laps[:, j, :], order, axis=1)
        gap = np.diff(ts, axis=1)
        ok = np.isfinite(ts[:, 1:]) & ~ps[:, 1:] & green[:, j, None]
        close += (ok & (gap < gap_s)).sum(axis=1)
        total += ok.sum(axis=1)
    return close, total


def actual_counts(act: Actual, from_lap: int = 1) -> dict[str, float]:
    """Passes and close following in the actual race, from ``from_lap`` on."""
    t = act.times[None, from_lap:, :]
    p = act.pit_laps[None, from_lap:, :]
    g = act.status[from_lap:] == GREEN
    close, total = close_share(t, p, g)
    return {"passes": int(count_passes(t, p, g)[0]),
            "close_share": float(close[0] / max(total[0], 1))}


def neutral_counts(status: np.ndarray, before: int = GREEN) -> tuple[np.ndarray, np.ndarray]:
    """Number of SC and VSC periods started per simulation, from a (n, laps)
    schedule; ``before`` is the status of the lap before it (a period already
    running is not a new one).

    A period is a run of non-green laps; it is an SC if any lap is SC (a VSC
    that turns into an SC is one SC period, as in
    :func:`f1strat.race_events.neutral_runs`).
    """
    s = np.atleast_2d(status)
    n, L = s.shape
    sc_n = np.zeros(n, dtype=int)
    vsc_n = np.zeros(n, dtype=int)
    for i in range(n):
        j = 0
        running = before != GREEN
        while j < L:
            if s[i, j] == GREEN:
                running = False
                j += 1
                continue
            k = j
            while k < L and s[i, k] != GREEN:
                k += 1
            if not running:
                if (s[i, j:k] == SC).any():
                    sc_n[i] += 1
                else:
                    vsc_n[i] += 1
            running = False
            j = k
    return sc_n, vsc_n


# --- One backtest ---------------------------------------------------------------


@dataclass
class Backtest:
    """Simulated race against the actual one."""

    actual: Actual
    mode: str
    drivers: list[str]            # simulated cars (running at the end of lap 1)
    positions: np.ndarray         # (n_sims, n_cars): rank among finishers, nan if retired
    gaps: np.ndarray              # (n_sims, n_cars): s behind the simulated winner
    passes: np.ndarray            # (n_sims,)
    close: np.ndarray             # (n_sims,) share of counted car-laps within 1 s
    sc_counts: np.ndarray         # (n_sims,) SC periods started after the state lap
    vsc_counts: np.ndarray
    actual_counts: dict[str, float]
    lap1_order: list[str] = field(default_factory=list)   # order after lap 1


def run_backtest(
    fitted,
    raw: pd.DataFrame,
    race: pd.DataFrame,
    cleaned: pd.DataFrame,
    params,
    round_number: int,
    mode: str = "replay",
    n_sims: int = 1000,
    seed: int = 0,
    widening: float = DEFAULT_WIDENING,
) -> Backtest:
    """Simulate one 2022 race from the end of lap 1 with actual plans and retirements.

    Args:
        fitted: Degradation fit (the thinned full fit).
        raw, race, cleaned: ``load_season()`` laps, race table, and ``clean_laps``.
        params: :class:`f1strat.race_events.SimParams`.
        round_number: Race.
        mode: ``"replay"`` (actual SC/VSC laps forced) or ``"random"`` (drawn).
        n_sims: Simulations.
        seed: Random seed.
        widening: Final-stint widening factor (Phase 3: 1.188; 1.0 = off).
    """
    if mode not in ("replay", "random"):
        raise ValueError(mode)
    act = actual_race(raw, race, round_number)
    state = RaceState.from_laps(raw, round_number, lap=1, total_laps=act.total_laps)
    # Race time at the end of lap 1: session time, the same clock for every car.
    state = RaceState(state.round_number, state.lap, state.total_laps,
                      {d: _with_time(c, act, d) for d, c in state.cars.items()},
                      state.track_status, state.sets, state.neutral_laps)
    forced = act.status[2:] if mode == "replay" else None
    late = round_number in late_restart_rounds(race)
    plans: dict[str, Plan] = {d: actual_plan(raw, round_number, d, 1) for d in state.cars}
    fld = build_field(fitted, state, observed_pit_losses(cleaned, round_number), params,
                      n_sims=n_sims, seed=seed, use_fitted_current_stint=False,
                      widening=widening,
                      final_race_prior=(mode == "random" and late), status=forced,
                      max_stops=max(3, *(len(p.stops) for p in plans.values())))
    retire = {d: act.last_lap[d] for d in fld.drivers if not act.finished[d]}
    res = simulate_field(fld, plans, retire=retire, record=True)

    start = np.array([state.cars[d].race_time for d in fld.drivers])
    hist = np.concatenate([np.broadcast_to(start, (n_sims, 1, len(start))), res.history], axis=1)
    out_lap1 = np.array([state.cars[d].laps_since_stop == 0 and state.cars[d].stint > 1
                         for d in fld.drivers])
    pits = np.concatenate([np.broadcast_to(out_lap1, (n_sims, 1, len(start))), res.pit_laps],
                          axis=1)
    green = np.concatenate([np.zeros((n_sims, 1), bool), fld.status == GREEN], axis=1)
    passes = count_passes(hist, pits, green)
    c, t = close_share(hist, pits, green)
    close = c / np.maximum(t, 1)

    final = res.times
    running = np.isfinite(final)
    ranks = np.argsort(np.argsort(final, axis=1), axis=1) + 1.0
    positions = np.where(running, ranks, np.nan)
    gaps = np.where(running, final - final.min(axis=1, keepdims=True), np.nan)
    sc_n, vsc_n = neutral_counts(fld.status, before=act.status[1])
    lap1 = sorted(fld.drivers, key=lambda d: state.cars[d].race_time)
    return Backtest(act, mode, fld.drivers, positions, gaps, passes, close, sc_n, vsc_n,
                    actual_counts(act), lap1)


def _with_time(car, act: Actual, driver: str):
    from dataclasses import replace

    return replace(car, race_time=float(act.times[1, act.drivers.index(driver)]))


# --- Metrics --------------------------------------------------------------------


def _interval(x: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
    lo = (1 - level) / 2
    return np.nanquantile(x, lo, axis=0), np.nanquantile(x, 1 - lo, axis=0)


def position_table(bt: Backtest, seed: int = 0) -> pd.DataFrame:
    """Per finisher: actual position, simulated mean, 50%/80% intervals, PIT.

    Intervals are central quantiles of the simulated positions (inclusive, so
    on integer positions they cover a little more than nominal). The PIT is
    randomised for the discrete positions: P(sim < actual) + U · P(sim = actual).
    """
    rng = np.random.default_rng(seed)
    rows = []
    lo50, hi50 = _interval(bt.positions, 0.5)
    lo80, hi80 = _interval(bt.positions, 0.8)
    for d in bt.actual.order:
        c = bt.drivers.index(d)
        pos = bt.positions[:, c]
        a = bt.actual.order.index(d) + 1
        pit = (pos < a).mean() + rng.random() * (pos == a).mean()
        naive = [x for x in bt.lap1_order if x in bt.actual.order].index(d) + 1
        rows.append({"round": bt.actual.round_number, "driver": d, "actual": a,
                     "naive": naive, "sim_mean": pos.mean(), "lo50": lo50[c], "hi50": hi50[c],
                     "lo80": lo80[c], "hi80": hi80[c],
                     "in50": lo50[c] <= a <= hi50[c], "in80": lo80[c] <= a <= hi80[c],
                     "pit": pit})
    return pd.DataFrame(rows)


def gap_table(bt: Backtest) -> pd.DataFrame:
    """Per lead-lap finisher other than the winner: gap to the winner, PIT, coverage."""
    rows = []
    lo50, hi50 = _interval(bt.gaps, 0.5)
    lo80, hi80 = _interval(bt.gaps, 0.8)
    for d, g in bt.actual.gap_to_winner.items():
        if g <= 0:
            continue
        c = bt.drivers.index(d)
        x = bt.gaps[:, c]
        rows.append({"round": bt.actual.round_number, "driver": d, "actual_gap_s": g,
                     "sim_median_s": np.median(x), "pit": (x < g).mean(),
                     "in50": lo50[c] <= g <= hi50[c], "in80": lo80[c] <= g <= hi80[c]})
    return pd.DataFrame(rows)


def race_summary(bt: Backtest) -> dict[str, float]:
    """One row per race: coverage, rank correlation, passes, close following, SC/VSC."""
    from scipy.stats import spearmanr

    pos = position_table(bt)
    gap = gap_table(bt)
    act = bt.actual
    n_sc, n_vsc = (int(x[0]) for x in neutral_counts(act.status[2:], before=act.status[1]))
    return {
        "round": act.round_number, "mode": bt.mode, "finishers": len(pos),
        "pos_in50": pos["in50"].mean(), "pos_in80": pos["in80"].mean(),
        "spearman": spearmanr(pos["actual"], pos["sim_mean"]).statistic,
        "spearman_naive": spearmanr(pos["actual"], pos["naive"]).statistic,
        "mae": (pos["actual"] - pos["sim_mean"]).abs().mean(),
        "mae_naive": (pos["actual"] - pos["naive"]).abs().mean(),
        "gap_in50": gap["in50"].mean() if len(gap) else np.nan,
        "gap_in80": gap["in80"].mean() if len(gap) else np.nan,
        "passes_actual": bt.actual_counts["passes"],
        "passes_sim": bt.passes.mean(),
        "passes_sim_p10": np.quantile(bt.passes, 0.1),
        "passes_sim_p90": np.quantile(bt.passes, 0.9),
        "close_actual": bt.actual_counts["close_share"],
        "close_sim": bt.close.mean(),
        "sc_actual": n_sc, "vsc_actual": n_vsc,
        "sc_sim": bt.sc_counts.mean(), "vsc_sim": bt.vsc_counts.mean(),
    }


# --- Passing calibration -----------------------------------------------------------

#: Fit and check rounds for the passing calibration (even / odd backtest rounds).
FIT_ROUNDS: tuple[int, ...] = tuple(r for r in BACKTEST_ROUNDS if r % 2 == 0)
CHECK_ROUNDS: tuple[int, ...] = tuple(r for r in BACKTEST_ROUNDS if r % 2 == 1)


def with_passing(params, shift: float, dirty_air_s: float = 0.0, slope_scale: float = 1.0):
    """Copy of ``params`` with the passing calibration set."""
    from dataclasses import replace

    return replace(params, passing=replace(params.passing, shift=shift,
                                           dirty_air_s=dirty_air_s,
                                           slope=params.passing.slope * slope_scale))


def pass_totals(fitted, raw, race, cleaned, params, rounds, n_sims: int = 200,
                seed: int = 0) -> tuple[float, float, float, float]:
    """(simulated passes, actual passes, simulated close share, actual close share)
    summed / averaged over ``rounds`` in replay mode."""
    sim = act = 0.0
    close_s, close_a = [], []
    for rnd in rounds:
        bt = run_backtest(fitted, raw, race, cleaned, params, rnd, "replay", n_sims, seed)
        sim += bt.passes.mean()
        act += bt.actual_counts["passes"]
        close_s.append(bt.close.mean())
        close_a.append(bt.actual_counts["close_share"])
    return sim, act, float(np.mean(close_s)), float(np.mean(close_a))


def calibrate_shift(fitted, raw, race, cleaned, params, dirty_air_s: float = 0.0,
                    slope_scale: float = 1.0, rounds=FIT_ROUNDS, lo: float = -1.5, hi: float = 2.5,
                    iters: int = 10, n_sims: int = 200) -> float:
    """Intercept shift that makes simulated passes match actual over ``rounds``
    (replay mode, with ``dirty_air_s`` fixed). Bisection: under common random
    numbers, simulated passes increase with the shift."""
    for _ in range(iters):
        mid = (lo + hi) / 2
        sim, act, _, _ = pass_totals(fitted, raw, race, cleaned,
                                     with_passing(params, mid, dirty_air_s, slope_scale),
                                     rounds, n_sims)
        lo, hi = (mid, hi) if sim < act else (lo, mid)
    return (lo + hi) / 2


# --- Safety-car closing ---------------------------------------------------------------


def sc_gap_observations(raw: pd.DataFrame, race: pd.DataFrame, rounds) -> pd.DataFrame:
    """Gaps between consecutive cars during 2022 SC periods.

    For every SC period starting after lap 1 (``k`` = SC lap within the period,
    1 = first), each car and the car directly ahead of it at the end of the lap
    before the SC: their gap then (``gap0``) and at the end of SC lap ``k``,
    while both are running and neither has had an in or out lap since the SC
    began. (Cars rejoining from the pits may slot in between; requiring the
    pair to stay adjacent would drop most pairs, since most cars pit under SC.)
    """
    rows = []
    for rnd in rounds:
        act = actual_race(raw, race, rnd)
        st = act.status
        for L in range(2, act.total_laps + 1):
            if st[L] != SC or st[L - 1] == SC:
                continue
            t0 = act.times[L - 1]
            run = [c for c in np.argsort(t0) if np.isfinite(t0[c])]
            for pos in range(1, len(run)):
                i, a = run[pos], run[pos - 1]
                k, lap = 1, L
                while lap <= act.total_laps and st[lap] == SC:
                    t = act.times[lap]
                    if (act.pit_laps[L:lap + 1, [i, a]].any()
                            or not (np.isfinite(t[i]) and np.isfinite(t[a]))):
                        break
                    rows.append({"round": rnd, "sc_lap": L, "k": k,
                                 "gap0": t0[i] - t0[a], "gap": t[i] - t[a]})
                    k, lap = k + 1, lap + 1
    return pd.DataFrame(rows)


def predicted_sc_gap(obs: pd.DataFrame, ratio: float, sc_gap_s: float) -> np.ndarray:
    """Gap the simulator gives: kept on the first SC lap, then × ``ratio`` per lap."""
    return np.maximum(sc_gap_s, obs["gap0"].to_numpy() * ratio ** (obs["k"].to_numpy() - 1))


def fit_sc_close_ratio(obs: pd.DataFrame, sc_gap_s: float, loss: str = "seconds") -> float:
    """Ratio minimising the squared error of the gaps on SC laps 2 onwards.

    ``loss="seconds"`` (used): error in seconds, so the large gaps that decide
    pit-under-SC losses and final margins count most. ``"log"``: relative
    error, which the many sub-2 s pairs dominate (fits 0.64, too slow for big
    gaps; ``docs/validation.md``).
    """
    o = obs[(obs["k"] >= 2) & (obs["gap0"] > 0) & (obs["gap"] > 0)]
    grid = np.arange(0.20, 1.001, 0.005)
    if loss == "log":
        err = [np.mean((np.log(o["gap"]) - np.log(predicted_sc_gap(o, r, sc_gap_s))) ** 2)
               for r in grid]
    else:
        err = [np.mean((o["gap"] - predicted_sc_gap(o, r, sc_gap_s)) ** 2) for r in grid]
    return float(grid[int(np.argmin(err))])


def sc_gap_table(obs: pd.DataFrame, ratio: float | None, sc_gap_s: float) -> pd.DataFrame:
    """Median observed vs predicted gap by SC lap ``k`` (``ratio=None``: Phase 4,
    every gap at ``sc_gap_s`` from the first SC lap)."""
    pred = (np.full(len(obs), sc_gap_s) if ratio is None
            else predicted_sc_gap(obs, ratio, sc_gap_s))
    o = obs.assign(pred=pred)
    return o.groupby("k").agg(pairs=("gap", "size"), gap_before=("gap0", "median"),
                              observed=("gap", "median"), predicted=("pred", "median"))


# --- Running everything -------------------------------------------------------------

FIG_DIR = Path(__file__).resolve().parents[2] / "reports" / "figures"


def run_all(fitted, raw, race, cleaned, params, rounds=BACKTEST_ROUNDS,
            n_sims: int = 1000, seed: int = 0, modes=("replay", "random"),
            widening: float = DEFAULT_WIDENING) -> dict[str, pd.DataFrame]:
    """Both modes for every round: per-race summary, positions and gaps."""
    out: dict[str, list] = {"summary": [], "positions": [], "gaps": [], "passes": []}
    for mode in modes:
        for rnd in rounds:
            bt = run_backtest(fitted, raw, race, cleaned, params, rnd, mode, n_sims, seed,
                              widening)
            out["summary"].append(race_summary(bt))
            out["positions"].append(position_table(bt).assign(mode=mode))
            out["gaps"].append(gap_table(bt).assign(mode=mode))
            out["passes"].append(pd.DataFrame({"round": rnd, "mode": mode,
                                               "sim_passes": bt.passes,
                                               "sim_sc": bt.sc_counts,
                                               "sim_vsc": bt.vsc_counts}))
    return {k: pd.concat([v if isinstance(v, pd.DataFrame) else pd.DataFrame([v])
                          for v in vs], ignore_index=True) for k, vs in out.items()}


def overall(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Pooled metrics per mode."""
    from scipy.stats import spearmanr

    rows = []
    for mode, pos in tables["positions"].groupby("mode"):
        gap = tables["gaps"][tables["gaps"]["mode"] == mode]
        s = tables["summary"][tables["summary"]["mode"] == mode]
        rows.append({
            "mode": mode, "finishers": len(pos),
            "pos_in50": pos["in50"].mean(), "pos_in80": pos["in80"].mean(),
            "pos_pit50": pos["pit"].between(0.25, 0.75).mean(),
            "pos_pit80": pos["pit"].between(0.1, 0.9).mean(),
            "spearman_mean": s["spearman"].mean(),
            "spearman_pooled": spearmanr(pos["actual"], pos["sim_mean"]).statistic,
            "gaps": len(gap), "gap_in50": gap["in50"].mean(), "gap_in80": gap["in80"].mean(),
            "pos_mean_abs_err": (pos["actual"] - pos["sim_mean"]).abs().mean(),
            "naive_spearman_mean": s["spearman_naive"].mean(),
            "naive_mean_abs_err": (pos["actual"] - pos["naive"]).abs().mean(),
            "passes_actual": s["passes_actual"].mean(), "passes_sim": s["passes_sim"].mean(),
            "close_actual": s["close_actual"].mean(), "close_sim": s["close_sim"].mean(),
            "sc_actual": s["sc_actual"].mean(), "sc_sim": s["sc_sim"].mean(),
            "vsc_actual": s["vsc_actual"].mean(), "vsc_sim": s["vsc_sim"].mean(),
        })
    return pd.DataFrame(rows)


def plot_all(tables: dict[str, pd.DataFrame], fig_dir: Path = FIG_DIR) -> list[Path]:
    """Calibration plots: PIT histograms, coverage per race, passes, SC/VSC counts."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    names = {"replay": "replay (actual SC/VSC)", "random": "random (SC/VSC drawn)"}

    # PIT histograms: positions and gaps, both modes.
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey="row")
    for col, mode in enumerate(("replay", "random")):
        for row, (key, label) in enumerate((("positions", "finishing position"),
                                            ("gaps", "gap to winner"))):
            t = tables[key][tables[key]["mode"] == mode]
            ax = axes[row, col]
            ax.hist(t["pit"], bins=10, range=(0, 1), color="#4C72B0", edgecolor="white")
            ax.axhline(len(t) / 10, color="black", lw=1, ls="--")
            ax.set_title(f"{label}: {names[mode]}", fontsize=10)
            ax.set_xlabel("PIT (share of simulations below the actual)")
            if col == 0:
                ax.set_ylabel("cars")
    fig.suptitle("Probability integral transform: flat = calibrated")
    fig.tight_layout()
    paths.append(fig_dir / "backtest_pit.png")
    fig.savefig(paths[-1], dpi=120)
    plt.close(fig)

    # Coverage per race.
    s = tables["summary"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, mode in zip(axes, ("replay", "random")):
        t = s[s["mode"] == mode]
        x = np.arange(len(t))
        for k, (col, lvl, c) in enumerate((("pos_in50", 0.5, "#4C72B0"),
                                           ("pos_in80", 0.8, "#DD8452"),
                                           ("gap_in80", 0.8, "#55A868"))):
            ax.scatter(x + (k - 1) * 0.2, t[col], color=c, s=18,
                       label={"pos_in50": "position 50%", "pos_in80": "position 80%",
                              "gap_in80": "gap 80%"}[col])
        ax.axhline(0.5, color="#4C72B0", ls="--", lw=1)
        ax.axhline(0.8, color="#DD8452", ls="--", lw=1)
        ax.set_xticks(x, t["round"].astype(int))
        ax.set_xlabel("round")
        ax.set_title(names[mode], fontsize=10)
    axes[0].set_ylabel("share of cars inside the interval")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    paths.append(fig_dir / "backtest_coverage.png")
    fig.savefig(paths[-1], dpi=120)
    plt.close(fig)

    # Passes per race: actual vs simulated (10th–90th percentile).
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, mode in zip(axes, ("replay", "random")):
        t = s[s["mode"] == mode]
        x = np.arange(len(t))
        ax.errorbar(x, t["passes_sim"], yerr=[t["passes_sim"] - t["passes_sim_p10"],
                                              t["passes_sim_p90"] - t["passes_sim"]],
                    fmt="o", color="#4C72B0", label="simulated (mean, 10–90%)")
        ax.scatter(x, t["passes_actual"], color="#C44E52", marker="x", s=40, label="2022",
                   zorder=3)
        ax.set_xticks(x, t["round"].astype(int))
        ax.set_xlabel("round (even: calibration fit; odd: check)")
        ax.set_title(names[mode], fontsize=10)
    axes[0].set_ylabel("on-track passes per race")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    paths.append(fig_dir / "backtest_passes.png")
    fig.savefig(paths[-1], dpi=120)
    plt.close(fig)

    # Close following per race.
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for mode, c in (("replay", "#4C72B0"), ("random", "#DD8452")):
        t = s[s["mode"] == mode]
        ax.scatter(t["close_actual"], t["close_sim"], color=c, label=names[mode], s=22)
    ax.plot([0, 0.5], [0, 0.5], color="black", lw=1, ls="--")
    ax.set_xlim(0, 0.45)
    ax.set_ylim(0, 0.45)
    ax.set_xlabel("2022: share of car-laps within 1 s of the car ahead")
    ax.set_ylabel("simulated (mean)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    paths.append(fig_dir / "backtest_close.png")
    fig.savefig(paths[-1], dpi=120)
    plt.close(fig)

    # SC / VSC counts per race in random mode vs 2022.
    p = tables["passes"][tables["passes"]["mode"] == "random"]
    t = s[s["mode"] == "random"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, (sim, act, label) in zip(axes, (("sim_sc", "sc_actual", "SC periods"),
                                            ("sim_vsc", "vsc_actual", "VSC periods"))):
        k = np.arange(0, 6)
        sim_share = [(p[sim] == v).mean() for v in k]
        act_share = [(t[act] == v).mean() for v in k]
        ax.bar(k - 0.2, sim_share, width=0.4, color="#4C72B0", label="simulated")
        ax.bar(k + 0.2, act_share, width=0.4, color="#C44E52", label="2022 (15 races)")
        ax.set_xlabel(f"{label} per race (after lap 1)")
        ax.set_ylabel("share of races")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    paths.append(fig_dir / "backtest_sc_vsc.png")
    fig.savefig(paths[-1], dpi=120)
    plt.close(fig)
    return paths


def main(argv: list[str] | None = None) -> None:
    """``calibrate``: fit the SC closing ratio and the pass-intercept shift on
    even rounds and save them in ``sim_params.json``. ``run``: both backtest
    modes, tables and plots. ``widening``: replay with widening 1.0."""
    import sys

    from f1strat.clean import clean_laps
    from f1strat.data import load_season
    from f1strat.deg_model import THIN_FIT_PATH, load_fit
    from f1strat.race_events import SimParams

    cmd = (argv if argv is not None else sys.argv[1:]) or ["run"]
    fitted = load_fit(THIN_FIT_PATH)
    raw = load_season(2022)
    race = load_season(2022, table="race")
    cleaned = clean_laps(raw)
    params = SimParams.load()
    if cmd[0] == "calibrate":
        from dataclasses import replace

        fit_obs = sc_gap_observations(raw, race, FIT_ROUNDS)
        check_obs = sc_gap_observations(raw, race, CHECK_ROUNDS)
        ratio = fit_sc_close_ratio(fit_obs, params.sc_gap_s)
        params = replace(params, neutral=replace(params.neutral, sc_close_ratio=ratio))
        print(f"SC closing ratio {ratio:.3f} (fit on {fit_obs['sc_lap'].groupby(fit_obs['round']).nunique().sum()} SC periods)")
        print(f"(log-error fit: {fit_sc_close_ratio(fit_obs, params.sc_gap_s, 'log'):.3f})")
        tabs = []
        for name, o in (("fit (even)", fit_obs), ("check (odd)", check_obs)):
            o = o.assign(size=pd.cut(o["gap0"], [0, 5, 200], labels=["gap0<=5s", "gap0>5s"]))
            for size, part in o.groupby("size", observed=True):
                tab = sc_gap_table(part, ratio, params.sc_gap_s)
                print(name, size)
                print(tab.round(2).to_string())
                tabs.append(tab.reset_index().assign(sample=name, size=size))
        pd.concat(tabs).round(3).to_csv(FIG_DIR / "sc_closing.csv", index=False)
        base = with_passing(params, 0.0)
        shift = calibrate_shift(fitted, raw, race, cleaned, base)
        params = with_passing(params, round(shift, 3))
        print(f"shift {shift:.3f}; wrote {params.save()}")
        for name, rounds in (("fit (even)", FIT_ROUNDS), ("check (odd)", CHECK_ROUNDS)):
            sim, act, cs, ca = pass_totals(fitted, raw, race, cleaned, params, rounds)
            print(f"{name}: passes sim {sim:.0f} vs actual {act:.0f}; "
                  f"close share sim {cs:.3f} vs actual {ca:.3f}")
        return
    if cmd[0] == "widening":  # sensitivity: replay without the final-stint widening
        tables = run_all(fitted, raw, race, cleaned, params, modes=("replay",), widening=1.0)
        ov = overall(tables).assign(widening=1.0)
        ov.to_csv(FIG_DIR / "backtest_overall_widening1.csv", index=False)
        print(ov.round(3).T.to_string())
        return
    tables = run_all(fitted, raw, race, cleaned, params)
    for name, df in tables.items():
        if name != "passes":  # per-simulation draws: plotted, not saved
            df.to_csv(FIG_DIR / f"backtest_{name}.csv", index=False)
    ov = overall(tables)
    ov.to_csv(FIG_DIR / "backtest_overall.csv", index=False)
    print(ov.round(3).to_string())
    print(tables["summary"].round(2).to_string())
    for p in plot_all(tables):
        print("wrote", p)


if __name__ == "__main__":
    main()
