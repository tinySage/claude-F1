"""Phase 6 case studies: decision-lap verdicts under the rule in
``docs/case_studies.md`` (committed before any case-study simulation).

For a call (e.g. Hungary 2022: Leclerc onto hards at lap 39):

1. **Decision state** (:func:`decision_state`): the race at the cutoff lap,
   stops on the cutoff lap applied as known, each car's sets (inventory minus
   race fittings, plus race-used sets), and the tyre-age cap.
2. **Candidates** (:func:`candidates`): the stratified search shortlist, the
   best plans in every named family, the as-run and intended plans.
3. **Selection** on 1,000 simulations, **measurement** on a fresh 2,000
   (:func:`verdict`): best plan overall vs best plan in the call family,
   paired by common random numbers; band from the loss.
4. **Robustness** (:data:`CHECKS`) and the **hindsight replay**
   (:func:`hindsight`), kept separate from the verdict.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from f1strat.deg_model import COMPOUNDS, BayesFit
from f1strat.simulator import (
    Field,
    Plan,
    RaceState,
    Stop,
    build_field,
    field_search,
    rival_plan_options,
    search,
)

FIG_DIR = Path(__file__).resolve().parents[2] / "reports" / "figures"
#: 2022 points for P1-P10 (no fastest-lap point).
POINTS = np.array([25, 18, 15, 12, 10, 8, 6, 4, 2, 1])
HARD = COMPOUNDS.index("HARD")


def label(plan: Plan, current: str) -> str:
    """Like ``Plan.label`` but with a used set's out-lap TyreLife, e.g.
    ``"M → 39 H(new) → 54 S(used 4)"``."""
    parts = [current[0]]
    for st in plan.stops:
        parts.append(f"{st.lap} {st.compound[0]}({'new' if st.new else f'used {st.tyre_life}'})")
    return " → ".join(parts)


def points(positions: np.ndarray) -> np.ndarray:
    """Championship points for finishing positions (1 = winner)."""
    pos = np.asarray(positions, int)
    return np.where(pos <= 10, POINTS[np.clip(pos, 1, 10) - 1], 0)


# --- The call ---------------------------------------------------------------------


@dataclass
class Family:
    """Plans whose first stop is in ``laps`` (inclusive) onto one of ``sets``."""

    laps: tuple[int, int]
    sets: tuple[tuple[str, int], ...]

    def contains(self, plan: Plan) -> bool:
        if not plan.stops:
            return False
        s = plan.stops[0]
        return self.laps[0] <= s.lap <= self.laps[1] and (s.compound, s.tyre_life) in self.sets


@dataclass
class Call:
    """A case study's decision (see ``docs/case_studies.md``)."""

    name: str
    round_number: int
    cutoff: int
    focal: str
    call_family: str                      # key in ``families``
    as_run: Plan
    intended: Plan
    #: Stops on the cutoff lap, known at the decision: driver -> (compound, out-lap TyreLife).
    known_stops: dict[str, tuple[str, int]] = field(default_factory=dict)
    #: Filled by :func:`decision_state` from the focal car's sets.
    families: dict[str, Family] = field(default_factory=dict)


def hungary_call() -> Call:
    """Hungary 2022: Leclerc onto new hards at lap 39 (cutoff lap 38)."""
    return Call(
        name="hungary", round_number=13, cutoff=38, focal="LEC", call_family="call: hards at 39",
        as_run=Plan((Stop(39, "HARD", 1), Stop(54, "SOFT", 4))),
        intended=Plan((Stop(39, "HARD", 1),)),
        known_stops={"VER": ("MEDIUM", 1)},
    )


def hungary_families(sets: list[tuple[str, int]], cutoff: int, total: int) -> dict[str, Family]:
    """The named alternatives in ``docs/case_studies.md`` (Hungary)."""
    by = {c: tuple(sorted({s for s in sets if s[0] == c})) for c in ("SOFT", "MEDIUM", "HARD")}
    new_h = tuple(s for s in by["HARD"] if s[1] == 1)
    used_m = tuple(s for s in by["MEDIUM"] if s[1] > 1)
    now, later = (cutoff + 1, cutoff + 1), (cutoff + 2, total - 1)
    fams = {
        "call: hards at 39": Family(now, new_h),
        "used softs at 39": Family(now, by["SOFT"]),
        "used medium at 39": Family(now, used_m),
        "stay out, then hards": Family(later, new_h),
        "stay out, then used softs": Family(later, by["SOFT"]),
        "stay out, then used medium": Family(later, used_m),
    }
    return {k: v for k, v in fams.items() if v.sets}


# --- Decision state ---------------------------------------------------------------


def tyre_age_cap(model_laps: pd.DataFrame) -> dict[str, int]:
    """Oldest clean TyreLife per compound in the model's data (the cap)."""
    return {str(c): int(v) for c, v in model_laps.groupby("Compound")["TyreLife"].max().items()}


def outlap_loss(cleaned: pd.DataFrame, round_number: int) -> float:
    """Median out-lap part of the pit loss (out lap − median clean lap) of the
    race's green-flag stops in ``cleaned``."""
    from f1strat.clean import pit_stops

    st = pit_stops(cleaned)
    st = st[(st["RoundNumber"] == round_number) & st["green"]]
    return float((st["out_s"] - st["median_clean_s"]).median())


def apply_known_stop(state: RaceState, driver: str, compound: str, tyre_life: int,
                     outlap_s: float) -> RaceState:
    """The car pitted at the end of the state's lap: the next lap is its out
    lap on (compound, out-lap ``tyre_life``).

    Its lap time already contains the in-lap part of the pit loss; the out-lap
    part (``outlap_s``) is added to its race time. The fitted set leaves its
    pool and the set taken off joins it as a used set.
    """
    car = state.cars[driver]
    new = replace(car, compound=compound, tyre_life=tyre_life - 1, laps_since_stop=-1,
                  stint=car.stint + 1, race_time=car.race_time + outlap_s,
                  compounds_used=car.compounds_used | {compound})
    cars = {**state.cars, driver: new}
    sets = None
    if state.sets is not None:
        sets = dict(state.sets)
        pool = list(sets.get(driver, ()))
        if (compound, tyre_life) in pool:
            pool.remove((compound, tyre_life))
        pool.append((car.compound, car.tyre_life + 1))
        sets[driver] = tuple(sorted(pool))
    return replace(state, cars=cars, sets=sets)


def decision_state(raw: pd.DataFrame, race: pd.DataFrame, call: Call,
                   cleaned: pd.DataFrame) -> tuple[RaceState, dict]:
    """The race at the cutoff as the pit wall knew it, and notes for the report."""
    from f1strat.tyres import race_used_sets, reconstruct_inventories, sets_left

    rnd, lap = call.round_number, call.cutoff
    total = int(race.loc[race["RoundNumber"] == rnd, "TotalLaps"].iloc[0])
    state = RaceState.from_laps(raw, rnd, lap, total)
    invs = reconstruct_inventories(2022, rnd)
    sets = {d: sets_left(invs[d], raw, rnd, lap) + race_used_sets(raw, rnd, d, lap)
            for d in state.cars if d in invs}
    state = state.with_sets(sets)
    out_s = outlap_loss(cleaned, rnd)
    for d, (comp, life) in call.known_stops.items():
        state = apply_known_stop(state, d, comp, life, out_s)
    call.families = hungary_families(list(state.sets[call.focal]), lap, total)
    notes = {"total_laps": total, "outlap_loss_s": round(out_s, 2),
             "focal_sets": list(state.sets[call.focal]),
             "known_stops": call.known_stops}
    return state, notes


# --- Field with sensitivity modifications -------------------------------------------


@dataclass
class Check:
    """One robustness setting (``docs/case_studies.md``)."""

    name: str
    widening: float = 1.0
    sc_factor: float = 1.0
    hard_wear_q: float | None = None      # Ferrari deg[hard]+team_deg[hard] quantile
    hard_pace_q: float | None = None      # Ferrari team_compound[hard, r] quantile


CHECKS: tuple[Check, ...] = (
    Check("main"),
    Check("widening x1.188", widening=1.188),
    Check("Ferrari hard wear p10", hard_wear_q=0.1),
    Check("Ferrari hard wear p90", hard_wear_q=0.9),
    Check("Ferrari hard pace p10", hard_pace_q=0.1),
    Check("Ferrari hard pace p90", hard_pace_q=0.9),
    Check("SC/VSC rate x0.5", sc_factor=0.5),
    Check("SC/VSC rate x2", sc_factor=2.0),
)


def _flat(fitted: BayesFit, name: str) -> np.ndarray:
    a = fitted.posterior[name].to_numpy()
    return a.reshape(-1, *a.shape[2:]).astype(float)


def ferrari_hard_quantiles(fitted: BayesFit, round_number: int, team: str = "Ferrari",
                           qs=(0.1, 0.9)) -> dict[str, dict[float, float]]:
    """Posterior quantiles of Ferrari's hard wear (s/lap) and hard pace term (s)."""
    from f1strat.deg_model import TYRE_SCALE

    r, t = fitted.races.index(round_number), fitted.teams.index(team)
    wear = (_flat(fitted, "deg")[:, HARD, r] + _flat(fitted, "team_deg")[:, HARD, t]) / TYRE_SCALE
    pace = _flat(fitted, "team_compound")[:, HARD, r, t]
    return {"wear": {q: float(np.quantile(wear, q)) for q in qs},
            "pace": {q: float(np.quantile(pace, q)) for q in qs},
            "wear_mean": float(wear.mean()), "pace_mean": float(pace.mean())}


def make_field(fitted: BayesFit, state: RaceState, losses: np.ndarray, params, check: Check,
               n_sims: int, seed: int, team: str = "Ferrari") -> Field:
    """A field for one robustness check (common random numbers by ``seed``)."""
    if check.sc_factor != 1.0:
        params = replace(params, neutral=replace(
            params.neutral, p_sc_per_lap=params.neutral.p_sc_per_lap * check.sc_factor,
            p_vsc_per_lap=params.neutral.p_vsc_per_lap * check.sc_factor))
    fld = build_field(fitted, state, losses, params, n_sims=n_sims, seed=seed,
                      widening=check.widening, use_fitted_current_stint=True)
    if check.hard_wear_q is None and check.hard_pace_q is None:
        return fld
    q = ferrari_hard_quantiles(fitted, state.round_number, team,
                               qs=tuple(x for x in (check.hard_wear_q, check.hard_pace_q)
                                        if x is not None))
    r, t = fitted.races.index(state.round_number), fitted.teams.index(team)
    tc = _flat(fitted, "team_compound")[fld.draw_index, HARD, r, t]
    for d in fld.drivers:
        if state.cars[d].team != team:
            continue
        m = fld.models[d]
        if check.hard_wear_q is not None:
            m.deg[:, HARD] = q["wear"][check.hard_wear_q]
        if check.hard_pace_q is not None:
            m.compound[:, HARD] = m.compound[:, HARD] - tc + q["pace"][check.hard_pace_q]
    return fld


# --- Candidates, selection, measurement ----------------------------------------------


def candidates(fld: Field, call: Call, cap: dict[str, int]) -> tuple[list[Plan], dict]:
    """Leclerc's candidate plans and what the tyre-age cap removed."""
    st, d = fld.state, call.focal
    m, dr = fld.models[d], fld.draws[d]
    base = search(m, st, d, dr, max_stops=3, min_stint=5, max_age=cap,
                  include=[call.as_run, call.intended])
    plans = list(base.plans)
    for fam in call.families.values():
        res = search(m, st, d, dr, max_stops=3, min_stint=5, max_age=cap,
                     first_laps=fam.laps, first_sets=fam.sets)
        plans += [p for p in res.plans if p not in plans]
    car = st.cars[d]
    capped = {"n_capped": base.n_capped,
              # plan-dependent part of the expected race time (single car), s
              "best_removed": [{"plan": label(p, car.compound), "plan_cost_s": round(t, 1)}
                               for t, p in base.capped]}
    return plans, capped


def focal_points(fld: Field, call: Call, plans: list[Plan], rivals) -> tuple[np.ndarray, np.ndarray]:
    """(points, positions), each (n_plans, n_sims), for the focal car."""
    res = field_search(fld, call.focal, plans, rivals)
    c = res.drivers.index(call.focal)
    pos = res.positions[:, :, c]
    return points(pos), pos


def choose(plans: list[Plan], pts: np.ndarray, call: Call) -> dict[str, Plan]:
    """Best plan overall and best plan per family by mean points (selection set)."""
    mean = pts.mean(axis=1)
    out = {"best overall": plans[int(np.argmax(mean))]}
    for name, fam in call.families.items():
        idx = [i for i, p in enumerate(plans) if fam.contains(p)]
        if idx:
            out[name] = plans[max(idx, key=lambda i: mean[i])]
    out["as run"] = call.as_run
    out["hards to the flag"] = call.intended
    return out


def band(loss: np.ndarray, level_or_ahead: float) -> str:
    """Verdict band (``docs/case_studies.md``)."""
    if loss.mean() <= 1.0:
        return "within range"
    if loss.mean() > 3.0 and level_or_ahead < 0.25:
        return "outside range"
    return "marginal"


def loss_summary(best_pts, other_pts, best_pos, other_pos) -> dict[str, float]:
    loss = best_pts - other_pts
    return {"mean_loss": float(loss.mean()), "p10": float(np.quantile(loss, 0.1)),
            "p50": float(np.quantile(loss, 0.5)), "p90": float(np.quantile(loss, 0.9)),
            "p_loss_pos": float((loss > 0).mean()), "p_loss_neg": float((loss < 0).mean()),
            "p_level_or_ahead": float((other_pos <= best_pos).mean())}


def verdict(fitted: BayesFit, state: RaceState, losses: np.ndarray, params, call: Call,
            cap: dict[str, int], check: Check = CHECKS[0], n_select: int = 1000,
            n_measure: int = 2000, seeds: tuple[int, int] = (101, 202)) -> dict:
    """Selection on ``n_select`` simulations, measurement on a fresh ``n_measure``."""
    sel = make_field(fitted, state, losses, params, check, n_select, seeds[0])
    rivals = {d: rival_plan_options(sel, d, max_age=cap) for d in sel.drivers if d != call.focal}
    plans, capped = candidates(sel, call, cap)
    pts_sel, pos_sel = focal_points(sel, call, plans, rivals)
    chosen = choose(plans, pts_sel, call)

    mea = make_field(fitted, state, losses, params, check, n_measure, seeds[1])
    rivals_m = {d: rival_plan_options(mea, d, max_age=cap) for d in mea.drivers if d != call.focal}
    names = list(chosen)
    uniq = list(dict.fromkeys(chosen.values()))
    pts_m, pos_m = focal_points(mea, call, uniq, rivals_m)
    k = {n: uniq.index(p) for n, p in chosen.items()}
    best, fam = k["best overall"], k[call.call_family]
    primary = loss_summary(pts_m[best], pts_m[fam], pos_m[best], pos_m[fam])
    car = state.cars[call.focal]
    out = {
        "check": check.name,
        "band": band(pts_m[best] - pts_m[fam], primary["p_level_or_ahead"]),
        "primary": primary,
        "as_run": loss_summary(pts_m[best], pts_m[k["as run"]], pos_m[best], pos_m[k["as run"]]),
        "hards_to_flag": loss_summary(pts_m[best], pts_m[k["hards to the flag"]],
                                      pos_m[best], pos_m[k["hards to the flag"]]),
        "chosen": {n: label(chosen[n], car.compound) for n in names},
        "measure": {n: {"mean_points": float(pts_m[k[n]].mean()),
                        "mean_pos": float(pos_m[k[n]].mean()),
                        "p_win": float((pos_m[k[n]] == 1).mean()),
                        "p_podium": float((pos_m[k[n]] <= 3).mean()),
                        "points_dist": {int(v): float((pts_m[k[n]] == v).mean())
                                        for v in np.unique(pts_m[k[n]])}}
                    for n in names},
        "selection_table": pd.DataFrame({
            "plan": [label(p, car.compound) for p in plans],
            "mean_points": pts_sel.mean(axis=1), "mean_pos": pos_sel.mean(axis=1),
            "family": [next((n for n, f in call.families.items() if f.contains(p)), "")
                       for p in plans]}).sort_values("mean_points", ascending=False),
        "n_candidates": len(plans),
        "chosen_plans": chosen, "all_plans": plans,
        "capped": capped,
    }
    return out


# --- Hindsight replay ------------------------------------------------------------------


def hindsight(full_fit: BayesFit, raw: pd.DataFrame, race: pd.DataFrame,
              cleaned_full: pd.DataFrame, call: Call, state: RaceState,
              plans: dict[str, Plan], all_plans: list[Plan] = (), n_sims: int = 1000,
              seed: int = 303) -> dict:
    """Leclerc's plans under what actually happened after the cutoff (not the
    verdict). ``all_plans`` (e.g. every decision-lap candidate) are also run to
    find the best plan in hindsight."""
    from f1strat.backtest import actual_race
    from f1strat.race_events import SimParams
    from f1strat.simulator import actual_plan, observed_pit_losses, simulate_field

    act = actual_race(raw, race, call.round_number)
    forced = act.status[call.cutoff + 1:]
    losses = observed_pit_losses(cleaned_full, call.round_number)
    fld = build_field(full_fit, state, losses, SimParams.load(), n_sims=n_sims, seed=seed,
                      use_fitted_current_stint=True, status=forced)
    rivals = {d: actual_plan(raw, call.round_number, d, call.cutoff)
              for d in fld.drivers if d != call.focal}
    retire = {d: act.last_lap[d] for d in fld.drivers if not act.finished[d]}
    out = {}
    plans = dict(plans)
    if all_plans:
        mean = []
        for plan in all_plans:
            res = simulate_field(fld, {**rivals, call.focal: plan}, retire=retire)
            mean.append(points(res.positions[:, res.drivers.index(call.focal)]).mean())
        plans["best in hindsight"] = all_plans[int(np.argmax(mean))]
    for name, plan in plans.items():
        res = simulate_field(fld, {**rivals, call.focal: plan}, retire=retire)
        c = res.drivers.index(call.focal)
        pos = res.positions[:, c]
        pts = points(pos)
        out[name] = {"plan": label(plan, state.cars[call.focal].compound),
                     "mean_points": float(pts.mean()), "mean_pos": float(pos.mean()),
                     "p_win": float((pos == 1).mean()),
                     "points_dist": {int(v): float((pts == v).mean()) for v in np.unique(pts)}}
    finish = {d: i + 1 for i, d in enumerate(act.order)}
    out["actual result"] = {"position": finish.get(call.focal),
                            "points": int(points([finish[call.focal]])[0])}
    return out


# --- Figures and report ----------------------------------------------------------------

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
#: Ordinal blue ramp (light -> dark = fewer -> more points), from step 250.
POINT_BINS = [("0", [0], "#86b6ef"), ("1–8 (P6–P10)", [1, 2, 4, 6, 8], "#5598e7"),
              ("10 (P5)", [10], "#3987e5"), ("12 (P4)", [12], "#2a78d6"),
              ("15 (P3)", [15], "#256abf"), ("18 (P2)", [18], "#184f95"),
              ("25 (win)", [25], "#0d366b")]


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)


def _merged(main: dict, keep: Callable[[str], bool] = lambda n: True) -> list[tuple[str, str]]:
    """(names joined, plan label) per distinct plan, e.g. "best overall = call: ...".

    Several named plans can be the same plan; they share one row."""
    rows: dict[str, list[str]] = {}
    for n, lbl in main["chosen"].items():
        if keep(n):
            rows.setdefault(lbl, []).append(n)
    return [(" = ".join(ns), lbl) for lbl, ns in rows.items()]


def plot_points(main: dict, path: Path) -> Path:
    """Leclerc's points distribution per plan (measurement set, decision lap)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    meas = main["measure"]
    rows = sorted(_merged(main), key=lambda r: meas[r[0].split(" = ")[0]]["mean_points"])
    fig, ax = plt.subplots(figsize=(9.5, 0.6 * len(rows) + 1.8), facecolor=SURFACE)
    _style(ax)
    for y, (joined, _) in enumerate(rows):
        n = joined.split(" = ")[0]
        left = 0.0
        dist = meas[n]["points_dist"]
        for label, vals, color in POINT_BINS:
            w = sum(dist.get(v, dist.get(str(v), 0.0)) for v in vals)
            if w > 0:
                ax.barh(y, w - 0.004, left=left, color=color, height=0.62,
                        label=label if y == 0 else None)
            left += w
        ax.text(1.01, y, f"{meas[n]['mean_points']:.1f} pts", va="center", fontsize=9,
                color=INK)
    ax.set_yticks(range(len(rows)), [f"{j}\n{lbl}" for j, lbl in rows], fontsize=8.5, color=INK)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of simulations", color=INK2)
    ax.set_title("Leclerc's points per plan at the decision lap (Hungary, end of lap 38; "
                 "2,000 simulations)", fontsize=10, color=INK, loc="left")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title="points", ncol=7, fontsize=8, title_fontsize=8,
              loc="upper center", bbox_to_anchor=(0.5, -0.12 - 0.3 / len(rows)),
              frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_sensitivity(checks: list[dict], path: Path) -> Path:
    """Primary loss (best overall − best call-family plan) under every check."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 0.5 * len(checks) + 1.8), facecolor=SURFACE)
    _style(ax)
    ys = np.arange(len(checks))[::-1]
    lo = min(-0.5, min(c["primary"]["p10"] for c in checks) - 0.5)
    hi = max(4.0, max(c["primary"]["p90"] for c in checks) + 0.5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(-1.2, len(checks) - 0.4)
    for y, c in zip(ys, checks):
        p = c["primary"]
        ax.plot([p["p10"], p["p90"]], [y, y], color=BLUE, lw=2, solid_capstyle="round")
        ax.plot(p["mean_loss"], y, "o", color=BLUE, ms=8, mec=SURFACE, mew=2)
        ax.text(1.01, y, f"{c['band']} (mean {p['mean_loss']:.1f})", va="center",
                transform=ax.get_yaxis_transform(), fontsize=9, color=INK)
    for x, lbl, ha, dx in ((1, "within if mean ≤ 1", "right", -0.08),
                           (3, "outside if mean > 3\nand level/ahead < 25%", "left", 0.08)):
        ax.axvline(x, color=INK2, lw=1, ls="--")
        ax.text(x + dx, -0.85, lbl, fontsize=8, color=INK2, ha=ha, va="center")
    ax.axvline(0, color=GRID, lw=1)
    ax.set_yticks(ys, [c["check"] for c in checks], fontsize=9, color=INK)
    ax.set_xlabel("points lost by the call (best plan − best 'hards at 39' plan); "
                  "dot = mean, line = 10th–90th percentile", color=INK2, fontsize=9)
    ax.set_title("Verdict under each robustness check", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_hindsight(main: dict, hind: dict, path: Path) -> Path:
    """Mean points per plan at the decision lap vs in the hindsight replay."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _merged(main, keep=lambda n: n in hind)
    rows = sorted(rows, key=lambda r: main["measure"][r[0].split(" = ")[0]]["mean_points"])
    names = [r[0].split(" = ")[0] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(names) + 1.8), facecolor=SURFACE)
    _style(ax)
    for y, n in enumerate(names):
        a, b = main["measure"][n]["mean_points"], hind[n]["mean_points"]
        ax.plot([a, b], [y, y], color=GRID, lw=2)
        ax.plot(a, y, "o", color=BLUE, ms=8, mec=SURFACE, mew=2,
                label="decision lap (cutoff fit, drawn SC/VSC, rivals' own plans)" if y == 0 else None)
        ax.plot(b, y, "D", color=ORANGE, ms=7, mec=SURFACE, mew=2,
                label="hindsight (full fit, actual SC/VSC, rivals' actual plans)" if y == 0 else None)
    if "best in hindsight" in hind:
        y = len(names)
        names.append("best in hindsight")
        rows.append(("best in hindsight", hind["best in hindsight"]["plan"]))
        ax.plot(hind["best in hindsight"]["mean_points"], y, "D", color=ORANGE, ms=7,
                mec=SURFACE, mew=2)
    act = hind["actual result"]
    ax.axvline(act["points"], color=INK2, lw=1, ls="--")
    ax.set_ylim(-1.1, len(names) - 0.4)
    ax.text(act["points"], -0.85, f" actual race: P{act['position']}, {act['points']} pts",
            fontsize=8, color=INK2)
    ax.set_yticks(range(len(names)), [f"{j}\n{lbl}" for j, lbl in rows], fontsize=8, color=INK)
    ax.set_xlabel("Leclerc's mean points", color=INK2)
    ax.set_title("Decision lap vs hindsight (Hungary)", fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.1 - 0.4 / len(names)), ncol=1)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> None:
    """``python -m f1strat.case_study hungary``: the full Hungary case study from
    the saved divergence-free cutoff fit (``rehearsal.clean_fit``)."""
    import json
    import logging
    import sys
    import time

    from f1strat.clean import clean_laps
    from f1strat.data import load_season
    from f1strat.deg_model import THIN_FIT_PATH, Cutoff, cutoff_inputs, load_fit, thin_fit
    from f1strat.rehearsal import fit_path, leakage_free_params
    from f1strat.simulator import observed_pit_losses

    logging.disable(logging.WARNING)
    args = argv if argv is not None else sys.argv[1:]
    if not args or args[0] != "hungary":
        raise SystemExit("usage: python -m f1strat.case_study hungary")
    call = hungary_call()
    raw, race = load_season(2022), load_season(2022, table="race")
    cut = Cutoff(call.round_number, call.cutoff)
    cleaned, laps = cutoff_inputs(raw, cut)
    full = load_fit(fit_path(call.round_number, call.cutoff, "cutoff"))
    if full.diagnostics.get("divergences", 0) != 0:
        raise RuntimeError("cutoff fit has divergences")
    fitted = thin_fit(full)
    state, notes = decision_state(raw, race, call, cleaned)
    cap = tyre_age_cap(laps)
    losses = observed_pit_losses(cleaned, call.round_number)
    params, pinfo = leakage_free_params(fitted, raw, race, cleaned, call.round_number)
    results = {"notes": notes, "cap": cap, "n_pit_losses": len(losses),
               "params": pinfo, "fit_diagnostics": {k: float(v) for k, v in full.diagnostics.items()},
               "ferrari_hard": ferrari_hard_quantiles(fitted, call.round_number), "checks": []}
    print(json.dumps({k: results[k] for k in ("notes", "cap", "params", "ferrari_hard")},
                     indent=1, default=str), flush=True)
    tables = {}
    for check in CHECKS:
        t0 = time.time()
        v = verdict(fitted, state, losses, params, call, cap, check)
        tables[check.name] = v.pop("selection_table")
        objs = (v.pop("chosen_plans"), v.pop("all_plans"))
        if check.name == "main":
            chosen_plans, all_plans = objs
        results["checks"].append(v)
        print(f"{check.name}: {v['band']}, mean loss {v['primary']['mean_loss']:.2f}, "
              f"chosen {v['chosen']['best overall']} ({time.time() - t0:.0f} s)", flush=True)
    main_v = results["checks"][0]
    sel = tables["main"]
    hind = hindsight(load_fit(THIN_FIT_PATH), raw, race, clean_laps(raw), call, state,
                     chosen_plans, all_plans=all_plans)
    results["hindsight"] = hind
    stem = f"case_{call.name}"
    sel.round(3).to_csv(FIG_DIR / f"{stem}_candidates.csv", index=False)
    (FIG_DIR / f"{stem}_results.json").write_text(json.dumps(results, indent=1, default=str))
    plot_points(main_v, FIG_DIR / f"{stem}_points.png")
    plot_sensitivity(results["checks"], FIG_DIR / f"{stem}_sensitivity.png")
    plot_hindsight(main_v, hind, FIG_DIR / f"{stem}_hindsight.png")
    print(json.dumps({"checks": [{k: c[k] for k in ("check", "band", "primary", "as_run",
                                                    "hards_to_flag", "chosen")}
                                 for c in results["checks"]], "hindsight": hind},
                     indent=1, default=str))


if __name__ == "__main__":
    main()
