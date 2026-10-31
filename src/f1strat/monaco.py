"""Phase 6, Monaco 2022: a track-position accounting study (``docs/case_studies.md``).

Monaco 2022 was a wet-to-dry race, outside the dry lap model and the
simulator. Almost nobody passes at Monaco (2021: 0 passes in 115 close
battles), so the order after the stop cycle largely sets the result. This
module reconstructs the crossover from the timing data (Part A). The
counterfactual accounting is added only after its method is pre-registered.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ROUND = 7
FERRARI_RB = ("LEC", "SAI", "PER", "VER")
TOP8 = ("PER", "SAI", "VER", "LEC", "RUS", "NOR", "ALO", "HAM")


def race_laps(raw: pd.DataFrame) -> pd.DataFrame:
    """Monaco's laps with ``lap_s`` (s) and ``time_s`` (session time at the line)."""
    s = raw[raw["RoundNumber"] == ROUND].copy()
    s["lap_s"] = s["LapTime"].dt.total_seconds()
    s["time_s"] = s["Time"].dt.total_seconds()
    return s


def stops(raw: pd.DataFrame, drivers=FERRARI_RB) -> pd.DataFrame:
    """Each stop: in lap, compounds off and on, pit entry, pit-lane time (s)."""
    from f1strat.race_events import pit_lane_stops

    pl = pit_lane_stops(raw)
    pl = pl[(pl["RoundNumber"] == ROUND) & pl["Driver"].isin(drivers)]
    s = race_laps(raw).set_index(["Driver", "LapNumber"])
    rows = []
    for _, r in pl.sort_values(["LapNumber", "PitInTime"]).iterrows():
        d, L = r["Driver"], int(r["LapNumber"])
        rows.append({"driver": d, "in_lap": L, "off": s.at[(d, L), "Compound"],
                     "on": s.at[(d, L + 1), "Compound"],
                     "on_tyre_life": s.at[(d, L + 1), "TyreLife"],
                     "pit_entry": str(r["PitInTime"])[7:18], "lane_s": round(r["lane_s"], 3),
                     "team_stops_on_lap": int(r["team_stops_on_lap"])})
    return pd.DataFrame(rows)


def order_and_gaps(raw: pd.DataFrame, laps=range(12, 33), drivers=TOP8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(positions, gap to the leader in s at the line) per lap for ``drivers``."""
    s = race_laps(raw)
    pos = s.pivot(index="LapNumber", columns="Driver", values="Position")
    t = s.pivot(index="LapNumber", columns="Driver", values="time_s")
    gap = t.sub(t.min(axis=1), axis=0)
    idx = list(laps)
    return pos.loc[idx, list(drivers)], gap.loc[idx, list(drivers)].round(2)


def queue_time(raw: pd.DataFrame, driver: str = "LEC", lap: int = 21) -> dict:
    """The double-stack queue from pit-lane times: the second car's pit-lane
    time minus the race's solo green median (and minus its own other stop),
    with the arrival gap at pit entry and the season ``D`` model's prediction."""
    from f1strat.race_events import double_stack_loss, double_stack_pairs, pit_lane_stops

    pl = pit_lane_stops(raw)
    pl = pl[pl["RoundNumber"] == ROUND]
    solo = pl[(pl["team_stops_on_lap"] == 1) & (pl["kind"] == "green")]
    me = pl[(pl["Driver"] == driver) & (pl["LapNumber"] == lap)].iloc[0]
    mate = pl[(pl["Team"] == me["Team"]) & (pl["LapNumber"] == lap)
              & (pl["Driver"] != driver)].iloc[0]
    own_other = pl[(pl["Driver"] == driver) & (pl["LapNumber"] != lap) & (pl["kind"] == "green")]
    gap = (me["PitInTime"] - mate["PitInTime"]).total_seconds()
    D = double_stack_loss(double_stack_pairs(raw))["green"]["D_s"]
    q = me["lane_s"] - solo["lane_s"].median()
    return {"driver": driver, "lap": lap, "behind": mate["Driver"], "lane_s": round(me["lane_s"], 3),
            "teammate_lane_s": round(mate["lane_s"], 3),
            "solo_green_median_s": round(float(solo["lane_s"].median()), 3),
            "n_solo_green": int(len(solo)),
            "queue_vs_solo_s": round(float(q), 3),
            "queue_vs_own_other_stop_s": (round(float(me["lane_s"] - own_other["lane_s"].median()), 3)
                                          if len(own_other) else None),
            "arrival_gap_s": round(gap, 3), "implied_D_s": round(float(q + gap), 3),
            "season_D_s": D, "season_model_queue_s": round(max(0.0, D - gap), 3)}


def passes(raw: pd.DataFrame, race: pd.DataFrame) -> dict:
    """On-track passes on green laps with neither car on an in or out lap
    (the Phase 5 counter), by lap, with who passed whom."""
    from f1strat.backtest import actual_race
    from f1strat.simulator import GREEN

    act = actual_race(raw, race, ROUND)
    out = []
    for L in range(2, act.total_laps + 1):
        if act.status[L] != GREEN:
            continue
        a, b = act.times[L - 1], act.times[L]
        for i, di in enumerate(act.drivers):
            for j, dj in enumerate(act.drivers):
                if (np.isfinite(a[i]) and np.isfinite(a[j]) and np.isfinite(b[i])
                        and np.isfinite(b[j]) and a[i] < a[j] and b[i] > b[j]
                        and not act.pit_laps[L, i] and not act.pit_laps[L, j]):
                    out.append({"lap": L, "passer": dj, "passed": di})
    df = pd.DataFrame(out)
    top8 = set(act.order[:8])
    return {"n": len(df), "list": df.to_dict("records"),
            "involving_top8": int((df["passer"].isin(top8) | df["passed"].isin(top8)).sum())
            if len(df) else 0,
            "after_lap_23": int((df["lap"] > 23).sum()) if len(df) else 0,
            "finish_order": act.order}


def main() -> None:
    """``python -m f1strat.monaco``: write the Part A reconstruction to
    ``reports/figures/case_monaco_reconstruction.json``;
    ``python -m f1strat.monaco verdict``: Part B (:func:`run_monaco`)."""
    import json
    import logging
    import sys

    if sys.argv[1:] == ["verdict"]:
        logging.disable(logging.WARNING)
        run_monaco()
        return

    from f1strat.backtest import FIG_DIR
    from f1strat.data import load_season

    logging.disable(logging.WARNING)
    raw, race = load_season(2022), load_season(2022, table="race")
    pos, gap = order_and_gaps(raw)
    out = {"stops": stops(raw).to_dict("records"),
           "positions": {int(k): v for k, v in pos.astype(float).to_dict("index").items()},
           "gaps": {int(k): v for k, v in gap.to_dict("index").items()},
           "queue": queue_time(raw), "queue_red_bull": queue_time(raw, "VER", 22),
           "passes": passes(raw, race)}
    (FIG_DIR / "case_monaco_reconstruction.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: out[k] for k in ("stops", "queue", "queue_red_bull")}, indent=1,
                     default=str))
    print({k: v for k, v in out["passes"].items() if k != "list"})


# --- Part B: track-position accounting (method pre-registered in docs/case_studies.md) ---

from dataclasses import dataclass, field  # noqa: E402

COMP = {"WET": "W", "INTERMEDIATE": "I", "HARD": "H", "MEDIUM": "M", "SOFT": "S"}
OUTCOME_LAP = 23
HELD_GAP_S = 0.68          # 2022 held-gap median (Phase 5)
D_SEASON = 7.15            # season green double-stack D (Silverstone)
D_HINDSIGHT = 5.5          # implied D of Leclerc's actual stack (Part A)
POINTS = np.array([25, 18, 15, 12, 10, 8, 6, 4, 2, 1])


def prepared(raw: pd.DataFrame) -> pd.DataFrame:
    """Monaco laps with compound letter ``c``, in/out flags and ``eligible``
    (codes 1/2 only, not an in or out lap, lap >= 3, timed)."""
    s = race_laps(raw)
    s["c"] = s["Compound"].map(COMP)
    s["in_lap"] = s["PitInTime"].notna()
    s["out_lap"] = s["PitOutTime"].notna()
    code = s["TrackStatus"].fillna("").astype(str)
    s["green_code"] = code.str.fullmatch("[12]+")
    s["eligible"] = (s["green_code"] & ~s["in_lap"] & ~s["out_lap"] & (s["LapNumber"] >= 3)
                     & s["lap_s"].notna() & s["c"].isin(list("WIHM")))
    return s


@dataclass
class LapModel:
    """``lap = a[car] + b[compound, lap]``; cells with fewer than ``min_cars``
    cars are unidentified."""

    a: dict[str, float]
    b: dict[tuple[str, int], float]
    sigma: float
    resid: dict[str, np.ndarray]
    cells: dict[tuple[str, int], int]

    def b_hat(self, c: str, lap: int, method: str = "trend") -> float | None:
        """``b`` for a cell: fitted if identified, else forecast from the
        compound's last 3 identified laps before ``lap`` (linear trend, or the
        last value); ``None`` if fewer than 2 identified laps (1 allowed for
        persistence only)."""
        if (c, lap) in self.b:
            return self.b[(c, lap)]
        laps = sorted(L for (cc, L) in self.b if cc == c and L < lap)[-3:]
        if not laps:
            return None
        if method == "persistence" or len(laps) < 2:
            return self.b[(c, laps[-1])] if method == "persistence" else None
        x = np.array(laps, float)
        y = np.array([self.b[(c, L)] for L in laps])
        slope, icpt = np.polyfit(x, y, 1)
        return float(icpt + slope * lap)

    def identified_laps(self, c: str) -> list[int]:
        return sorted(L for (cc, L) in self.b if cc == c)


def fit_lap_model(s: pd.DataFrame, max_lap: int, min_cars: int = 2,
                  rng: np.random.Generator | None = None) -> LapModel:
    """Least squares on eligible laps 3..``max_lap``; with ``rng``, each
    compound-lap cell's laps are resampled with replacement (bootstrap)."""
    e = s[s["eligible"] & (s["LapNumber"] <= max_lap)][["Driver", "c", "LapNumber", "lap_s"]]
    n = e.groupby(["c", "LapNumber"])["Driver"].nunique()
    keep = {k for k, v in n.items() if v >= min_cars}
    e = e[[(c, int(L)) in keep for c, L in zip(e["c"], e["LapNumber"])]]
    if rng is not None:
        parts = [g.iloc[rng.integers(0, len(g), len(g))] for _, g in e.groupby(["c", "LapNumber"])]
        e = pd.concat(parts)
    cars = sorted(e["Driver"].unique())
    cells = sorted({(c, int(L)) for c, L in zip(e["c"], e["LapNumber"])})
    ci = {d: i for i, d in enumerate(cars)}
    ki = {k: len(cars) + i for i, k in enumerate(cells)}
    X = np.zeros((len(e), len(cars) + len(cells)))
    rows = np.arange(len(e))
    X[rows, [ci[d] for d in e["Driver"]]] = 1.0
    X[rows, [ki[(c, int(L))] for c, L in zip(e["c"], e["LapNumber"])]] = 1.0
    y = e["lap_s"].to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    resid = {d: r[(e["Driver"] == d).to_numpy()] for d in cars}
    return LapModel(a={d: float(beta[ci[d]]) for d in cars},
                    b={k: float(beta[ki[k]]) for k in cells},
                    sigma=float(r.std(ddof=len(cars) + len(cells) - 1) if len(r) > len(beta)
                                else r.std()),
                    resid=resid, cells={k: int(n[k]) for k in cells})


@dataclass
class Plan:
    """Leclerc's stops from the cutoff on, as (in lap, compound letter)."""

    name: str
    stops: tuple[tuple[int, str], ...]
    no_queue: bool = False          # diagnostic: the stack's queue removed


def actual_plans(s: pd.DataFrame, cars, cutoff: int) -> dict[str, dict[int, str]]:
    """Each car's compound letter on every lap cutoff+1..OUTCOME_LAP (actual)."""
    out = {}
    for d in cars:
        g = s[s["Driver"] == d].set_index("LapNumber")["c"]
        out[d] = {L: g.get(L) for L in range(cutoff + 1, OUTCOME_LAP + 1)}
    return out


def actual_stops(s: pd.DataFrame, cars, cutoff: int) -> dict[str, set[int]]:
    g = s[s["in_lap"] & s["LapNumber"].between(cutoff + 1, OUTCOME_LAP)]
    return {d: set(g.loc[g["Driver"] == d, "LapNumber"].astype(int)) for d in cars}


def plan_compounds(start: str, plan: Plan, cutoff: int) -> dict[int, str]:
    """Compound on each lap cutoff+1..OUTCOME_LAP; a stop at the end of lap L
    puts the new compound on from lap L+1."""
    out, c = {}, start
    stops = dict(plan.stops)
    for L in range(cutoff + 1, OUTCOME_LAP + 1):
        out[L] = c
        if L in stops:
            c = stops[L]
    return out


def stop_costs(s: pd.DataFrame, model: LapModel, max_out_lap: int,
               method: str = "trend") -> np.ndarray:
    """In-lap excess + out-lap excess of the race's eligible stops whose out
    lap is <= ``max_out_lap`` (green-coded laps 3+, not the second car of a
    same-lap team stop)."""
    from f1strat.race_events import pit_lane_stops

    lap = s.set_index(["Driver", "LapNumber"])
    teams = s.groupby("Driver")["Team"].first()
    ins = s[s["in_lap"] & (s["LapNumber"] >= 3)]
    out = []
    for _, r in ins.iterrows():
        d, L = r["Driver"], int(r["LapNumber"])
        if L + 1 > max_out_lap or (d, L + 1) not in lap.index:
            continue
        o = lap.loc[(d, L + 1)]
        if not (r["green_code"] and o["green_code"]) or pd.isna(r["lap_s"]) or pd.isna(o["lap_s"]):
            continue
        same = ins[(ins["LapNumber"] == L) & (ins["Driver"] != d)
                   & (ins["Driver"].map(teams) == teams[d])]
        if len(same) and (same["PitInTime"] < r["PitInTime"]).any():
            continue                      # second car of a double stack
        if d not in model.a:
            continue
        b_in, b_out = model.b_hat(r["c"], L, method), model.b_hat(o["c"], L + 1, method)
        if b_in is None or b_out is None:
            continue
        out.append((r["lap_s"] - model.a[d] - b_in) + (o["lap_s"] - model.a[d] - b_out))
    return np.array(out)


def run_accounting(s: pd.DataFrame, cutoff: int, plans: list[Plan], version: str,
                   n_draws: int, seed: int, method: str = "trend",
                   focal: str = "LEC") -> dict:
    """Leclerc's position at the end of lap 23 under each plan, per draw.

    ``version``: ``"hindsight"`` (model on laps 3–26, actual laps where
    unchanged, D 5.5 s) or ``"cutoff"`` (model on laps <= cutoff, every lap
    after the cutoff forecast, season D). Returns per plan the positions,
    the gaps to the cars ahead and behind, and per draw the points.
    """
    rng = np.random.default_rng(seed)
    hind = version == "hindsight"
    max_lap = 26 if hind else cutoff
    D = D_HINDSIGHT if hind else D_SEASON
    at_cut = s[s["LapNumber"] == cutoff].set_index("Driver")
    lead_t = at_cut["time_s"].min()
    last_lap = s[(s["LapNumber"] == cutoff)]["lap_s"].min()
    cars = [d for d in at_cut.index if pd.notna(at_cut.at[d, "time_s"])
            and at_cut.at[d, "time_s"] - lead_t < last_lap
            and s[(s["Driver"] == d) & (s["LapNumber"] == OUTCOME_LAP)]["time_s"].notna().any()]
    comp = actual_plans(s, cars, cutoff)
    stops_act = actual_stops(s, cars, cutoff)
    team = s.groupby("Driver")["Team"].first()
    lap_act = s.set_index(["Driver", "LapNumber"])["lap_s"]
    T0 = {d: float(at_cut.at[d, "time_s"]) for d in cars}
    base = fit_lap_model(s, max_lap)
    pooled = np.concatenate(list(base.resid.values()))
    focal_start = s[(s["Driver"] == focal) & (s["LapNumber"] == cutoff)]["c"].iloc[0]
    cf = {p.name: plan_compounds(focal_start, p, cutoff) for p in plans}
    cf_stops = {p.name: {L for L, _ in p.stops} for p in plans}
    laps = range(cutoff + 1, OUTCOME_LAP + 1)
    res = {p.name: {"pos": [], "gap_ahead": [], "gap_behind": [], "assessable": True}
           for p in plans}
    flagged = set()
    for _ in range(n_draws):
        model = fit_lap_model(s, max_lap, rng=rng)
        costs = stop_costs(s, model, max_lap if not hind else OUTCOME_LAP + 3, method)
        if len(costs) == 0:
            costs = np.array([20.0])
        # Common random numbers: residuals and stop costs per (car, lap).
        eps = {(d, L): float(rng.choice(model.resid[d] if d in model.resid
                                        and len(model.resid[d]) >= 3 else pooled))
               for d in cars for L in laps}
        cost = {(d, L): float(rng.choice(costs)) for d in cars for L in laps}
        for p in plans:
            comps = {d: (cf[p.name] if d == focal else comp[d]) for d in cars}
            stp = {d: (cf_stops[p.name] if d == focal else stops_act[d]) for d in cars}
            T = dict(T0)
            ok = True
            for L in laps:
                arr, pitting, free, unch = {}, set(), {}, {}
                for d in cars:
                    c = comps[d][L]
                    is_in, is_out = L in stp[d], (L - 1) in stp[d]
                    unchanged = (hind and (d != focal or (
                        comps[d][L] == comp[d][L] and is_in == (L in stops_act[d])
                        and is_out == ((L - 1) in stops_act[d])
                        and (not is_out or comps[d][L] == comp[d][L])))
                        and pd.notna(lap_act.get((d, L))))
                    unch[d] = bool(unchanged)
                    if unchanged:
                        t = float(lap_act[(d, L)])
                        if d == focal and p.no_queue and is_in and L == 21:
                            t -= QUEUE_ACTUAL_S
                        free[d] = t - (cost[(d, L)] if is_in else 0.0)
                    else:
                        b = model.b_hat(c, L, method)
                        if b is None or d not in model.a:
                            ok = False
                            break
                        if (c, L) not in model.b:
                            flagged.add((c, L))
                        t = model.a[d] + b + eps[(d, L)]
                        free[d] = t
                        if is_in:
                            t += cost[(d, L)]
                    arr[d] = T[d] + t
                    if is_in or is_out:
                        pitting.add(d)
                if not ok:
                    break
                # Double stack: the car arriving first is served first.
                for tm in set(team[d] for d in cars):
                    pair = [d for d in cars if team[d] == tm and L in stp[d]]
                    if len(pair) != 2:
                        continue
                    if all(unch[d] for d in pair):
                        continue           # actual laps already include any queue
                    if p.no_queue and focal in pair:
                        continue
                    d1, d2 = sorted(pair, key=lambda d: T[d] + free[d])
                    if unch[d2] and L in stops_act[d2] and d2 != focal and \
                            d2 == "LEC":
                        continue
                    wait = max(0.0, D - ((T[d2] + free[d2]) - (T[d1] + free[d1])))
                    if unch[d2] and d2 == focal:
                        continue           # focal's actual lap already queued
                    arr[d2] += wait
                # No passing on track: in T order, a car not on an in/out lap
                # cannot arrive ahead of (or within the held gap of) the car
                # ahead that is not on an in/out lap.
                tail = None
                for d in sorted(cars, key=lambda d: T[d]):
                    if d in pitting:
                        continue
                    if tail is not None and arr[d] < arr[tail] + HELD_GAP_S:
                        arr[d] = arr[tail] + HELD_GAP_S
                    tail = d
                T = arr
            r = res[p.name]
            if not ok:
                r["assessable"] = False
                continue
            order = sorted(cars, key=lambda d: T[d])
            k = order.index(focal)
            r["pos"].append(k + 1)
            r["gap_ahead"].append(T[focal] - T[order[k - 1]] if k > 0 else np.nan)
            r["gap_behind"].append(T[order[k + 1]] - T[focal] if k + 1 < len(order) else np.nan)
    out = {"cars": cars, "sigma": base.sigma, "flagged_cells": sorted(flagged), "plans": {}}
    for p in plans:
        r = res[p.name]
        if not r["assessable"] or not r["pos"]:
            out["plans"][p.name] = {"assessable": False}
            continue
        pos = np.array(r["pos"])
        out["plans"][p.name] = {"assessable": True, "pos": pos,
                                "points": np.where(pos <= 10, POINTS[np.clip(pos, 1, 10) - 1], 0),
                                "gap_ahead": np.array(r["gap_ahead"]),
                                "gap_behind": np.array(r["gap_behind"])}
    return out


QUEUE_ACTUAL_S = 2.45


#: The decisions (``docs/case_studies.md``, amended): name -> (cutoff, plans);
#: the first plan is the call.
def decisions() -> dict[str, tuple[int, list[Plan]]]:
    P = Plan
    call_d1 = P("call: inters at 18, slicks at 21", ((18, "I"), (21, "H")))
    return {
        "D1 inters (cutoff 17)": (17, [
            call_d1,
            P("inters at 19, slicks at 21", ((19, "I"), (21, "H"))),
            P("skip inters: slicks at 21", ((21, "H"),)),
            P("skip inters: slicks at 22", ((22, "H"),))]),
        "D1 earlier stop (cutoff 16)": (16, [
            call_d1, P("inters at 17, slicks at 21", ((17, "I"), (21, "H")))]),
        "D2 slicks (cutoff 20)": (20, [
            P("call: slicks at 21 (double stack)", ((21, "H"),)),
            P("stay out: slicks at 22", ((22, "H"),))]),
        "D2 earlier stop (cutoff 19)": (19, [
            P("call: slicks at 21 (double stack)", ((21, "H"),)),
            P("slicks at 20", ((20, "H"),))]),
    }


DIAGNOSTIC = ("D2 slicks (cutoff 20)", Plan("no queue (diagnostic)", ((21, "H"),), no_queue=True))


def summarise(r: dict, name: str) -> dict:
    p = r["plans"][name]
    if not p["assessable"]:
        return {"assessable": False}
    pos, ga, gb = p["pos"], p["gap_ahead"], p["gap_behind"]

    def med(g):
        # No car ahead (P1) or behind: an unlimited gap, not a missing one.
        m = float(np.median(np.where(np.isnan(g), np.inf, g)))
        return None if np.isinf(m) else m

    return {"assessable": True, "mean_points": float(p["points"].mean()),
            "mean_pos": float(pos.mean()),
            "pos_dist": {int(v): float((pos == v).mean()) for v in np.unique(pos)},
            "median_gap_ahead_s": med(ga), "median_gap_behind_s": med(gb)}


def too_close(summary: dict, sigma: float, rebuilt: bool) -> bool:
    """Pre-registered rule: a counterfactual whose position rests on a median
    gap (to the car ahead or behind at the end of lap 23) below the lap-time
    noise. A plan with no rebuilt laps (the call in hindsight is the actual
    race) has no modelled noise."""
    if not rebuilt or not summary.get("assessable"):
        return False
    gaps = [g for g in (summary["median_gap_ahead_s"], summary["median_gap_behind_s"])
            if g is not None]
    return bool(gaps) and min(abs(g) for g in gaps) < sigma


def verdict(s: pd.DataFrame, cutoff: int, plans: list[Plan], version: str,
            method: str = "trend", n_select: int = 1000, n_measure: int = 2000,
            seeds: tuple[int, int] = (101, 202)) -> dict:
    """Committed rule: best plan chosen on the selection draws, loss measured
    on fresh draws; bands; too-close-to-call (amendment)."""
    sel = run_accounting(s, cutoff, plans, version, n_select, seeds[0], method)
    if not all(sel["plans"][p.name]["assessable"] for p in plans):
        return {"version": version, "method": method, "assessable": False,
                "why": "a compound needed has fewer than 2 identified laps by the cutoff",
                "plans": {p.name: sel["plans"][p.name]["assessable"] for p in plans}}
    call = plans[0].name
    means = {p.name: sel["plans"][p.name]["points"].mean() for p in plans}
    best = max(means, key=means.get)
    mea = run_accounting(s, cutoff, plans, version, n_measure, seeds[1], method)
    pb, pc = mea["plans"][best], mea["plans"][call]
    loss = pb["points"] - pc["points"]
    level = float((pc["pos"] <= pb["pos"]).mean())
    band = ("within range" if loss.mean() <= 1 else
            "outside range" if loss.mean() > 3 and level < 0.25 else "marginal")
    summ = {p.name: summarise(mea, p.name) for p in plans}
    rebuilt = {p.name: not (version == "hindsight" and p.name == call) for p in plans}
    tcc = {n: too_close(summ[n], mea["sigma"], rebuilt[n]) for n in summ}
    if best != call and (tcc[best] or tcc[call]):
        band_out = "too close to call"
    else:
        band_out = band
    return {"version": version, "method": method, "assessable": True, "cutoff": cutoff,
            "best": best, "call": call, "band": band_out, "band_by_points": band,
            "mean_loss": float(loss.mean()), "loss_p10": float(np.quantile(loss, 0.1)),
            "loss_p90": float(np.quantile(loss, 0.9)), "p_level_or_ahead": level,
            "sigma_s": mea["sigma"], "too_close": tcc, "plans": summ,
            "flagged_cells": [list(x) for x in mea["flagged_cells"]],
            "selection_means": {k: float(v) for k, v in means.items()}}


def run_monaco() -> dict:
    """``python -m f1strat.monaco verdict``: every decision in hindsight, at
    its cutoff (committed), and the labelled exploratory cutoff reading (last
    identified value held), plus the no-queue diagnostic. Writes
    ``reports/figures/case_monaco_results.json`` and the figures."""
    import json

    from f1strat.backtest import FIG_DIR
    from f1strat.data import load_season

    raw, race = load_season(2022), load_season(2022, table="race")
    s = prepared(raw)
    out = {"decisions": {}}
    for name, (cutoff, plans) in decisions().items():
        d = {"hindsight": verdict(s, cutoff, plans, "hindsight"),
             "hindsight_persistence": verdict(s, cutoff, plans, "hindsight", "persistence"),
             "cutoff": verdict(s, cutoff, plans, "cutoff"),
             "cutoff_exploratory_persistence": verdict(s, cutoff, plans, "cutoff",
                                                       "persistence")}
        out["decisions"][name] = d
        for k, v in d.items():
            print(name, k, v.get("band", "not assessable"), v.get("best"),
                  round(v.get("mean_loss", float("nan")), 2), flush=True)
    dname, diag = DIAGNOSTIC
    cutoff, plans = decisions()[dname]
    for version in ("hindsight", "cutoff_exploratory_persistence"):
        v, meth = ("hindsight", "trend") if version == "hindsight" else ("cutoff", "persistence")
        r = run_accounting(s, cutoff, [plans[0], diag], v, 2000, 202, meth)
        if r["plans"][diag.name]["assessable"]:
            out.setdefault("no_queue_diagnostic", {})[version] = {
                n: summarise(r, n) for n in (plans[0].name, diag.name)}
            out["no_queue_diagnostic"][version]["points_cost_of_queue"] = float(
                (r["plans"][diag.name]["points"] - r["plans"][plans[0].name]["points"]).mean())
    pos, gap = order_and_gaps(raw, laps=range(12, 33), drivers=FERRARI_RB)
    out["gap_timeline"] = {int(k): v for k, v in gap.to_dict("index").items()}
    (FIG_DIR / "case_monaco_results.json").write_text(json.dumps(out, indent=1, default=str))
    plot_gap_timeline(raw, race, FIG_DIR / "case_monaco_gaps.png")
    plot_positions(out, FIG_DIR / "case_monaco_positions.png")
    return out


# --- Figures --------------------------------------------------------------------------

BLUE, ORANGE, RED, GREY = "#2a78d6", "#eb6834", "#c8102e", "#8a8986"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
CAR_COLOURS = {"LEC": RED, "SAI": "#f28e98", "PER": "#1e41ff", "VER": "#7c93ff"}


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)


def plot_gap_timeline(raw: pd.DataFrame, race: pd.DataFrame, path) -> object:
    """Ferrari and Red Bull through the crossover: gap behind the leader at
    the line, with compounds, stops, the SC and the red flag."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = prepared(raw)
    laps = range(12, 33)
    t = s.pivot(index="LapNumber", columns="Driver", values="time_s").loc[list(laps)]
    gap = t.sub(t.min(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor=SURFACE)
    _style(ax)
    ax.axvspan(26.5, 29.5, color=GRID, alpha=0.7, lw=0)
    ax.axvspan(29.5, 30.5, color="#f5c2c2", alpha=0.7, lw=0)
    ax.text(28, 26, "SC", ha="center", fontsize=8, color=INK2)
    ax.text(30, 26, "red\nflag", ha="center", fontsize=8, color=INK2)
    marks = {"W": "o", "I": "s", "H": "^", "M": "D"}
    for d, col in CAR_COLOURS.items():
        g = s[s["Driver"] == d].set_index("LapNumber").reindex(list(laps))
        ax.plot(list(laps), gap[d], color=col, lw=2, label=d)
        for L in laps:
            c = g.at[L, "c"]
            if isinstance(c, str):
                ax.plot(L, gap.at[L, d], marks.get(c, "o"), color=col, ms=5, mec=SURFACE)
            if bool(g.at[L, "in_lap"]):
                ax.annotate(f"{d} in", (L, gap.at[L, d]), textcoords="offset points",
                            xytext=(4, 6), fontsize=7, color=col)
    ax.set_ylim(28, -2)
    ax.set_xlabel("lap", color=INK2)
    ax.set_ylabel("gap behind the leader at the line (s; in laps include the stop)",
                  color=INK2, fontsize=9)
    handles = [plt.Line2D([], [], color=c, lw=2, label=d) for d, c in CAR_COLOURS.items()]
    handles += [plt.Line2D([], [], color=INK2, marker=m, lw=0, label=n)
                for n, m in (("wet", "o"), ("inter", "s"), ("hard", "^"), ("medium", "D"))]
    ax.legend(handles=handles, fontsize=8, frameon=False, ncol=4, loc="lower left")
    ax.set_title("Monaco 2022: Ferrari and Red Bull through the crossover", fontsize=10,
                 color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_positions(out: dict, path) -> object:
    """Leclerc's position at the end of lap 23 under each plan (hindsight,
    measurement draws), per decision."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for dname, d in out["decisions"].items():
        v = d["hindsight"]
        if not v.get("assessable"):
            continue
        for pname, ps in v["plans"].items():
            rows.append((dname, pname, ps, v["too_close"][pname], pname == v["best"]))
    colours = {1: "#0d366b", 2: "#184f95", 3: "#2a78d6", 4: "#5598e7", 5: "#86b6ef"}
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(rows) + 1.6), facecolor=SURFACE)
    _style(ax)
    labels = []
    for y, (dname, pname, ps, tcc, best) in enumerate(rows[::-1]):
        left = 0.0
        for k in sorted(ps["pos_dist"], key=int):
            w = ps["pos_dist"][k]
            kk = min(int(k), 5)
            ax.barh(y, w - 0.003, left=left, color=colours[kk], height=0.6)
            if w > 0.08:
                ax.text(left + w / 2, y, f"P{k}", ha="center", va="center", fontsize=7,
                        color="white")
            left += w
        tag = f"{ps['mean_points']:.1f} pts" + (" · too close to call" if tcc else "") + \
              (" · best" if best else "")
        ax.text(1.01, y, tag, va="center", fontsize=8, color=INK)
        labels.append(f"{dname.split(' (')[0]}: {pname}")
    ax.set_yticks(range(len(rows)), labels, fontsize=8, color=INK)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of draws (hindsight, 2,000); Leclerc's position at the end of lap 23",
                  color=INK2, fontsize=9)
    ax.set_title("Monaco 2022: Leclerc's position under each counterfactual (hindsight)",
                 fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


if __name__ == "__main__":
    main()
