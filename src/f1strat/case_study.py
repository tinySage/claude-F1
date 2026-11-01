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

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from f1strat.race_events import Neutralisations, SimParams

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


def parse_label(text: str) -> Plan:
    """Inverse of :func:`label` (e.g. ``"M → 39 H(new) → 54 S(used 4)"``)."""
    names = {c[0]: c for c in COMPOUNDS}
    stops = []
    for part in text.split(" → ")[1:]:
        lap, rest = part.split(" ", 1)
        comp, tag = rest[0], rest[2:-1]
        stops.append(Stop(int(lap), names[comp], 1 if tag == "new" else int(tag.split()[1])))
    return Plan(tuple(stops))


def points(positions: np.ndarray) -> np.ndarray:
    """Championship points for finishing positions (1 = winner)."""
    pos = np.asarray(positions, int)
    return np.where(pos <= 10, POINTS[np.clip(pos, 1, 10) - 1], 0)


# --- The call ---------------------------------------------------------------------


@dataclass
class Family:
    """Plans whose first stop is in ``laps`` (inclusive) onto one of ``sets``;
    ``laps=None`` is the single plan with no further stop. ``or_none`` also
    admits the plan with no further stop (e.g. Silverstone's "stay out at 39":
    no stop on lap 39, any later stop or none)."""

    laps: tuple[int, int] | None
    sets: tuple[tuple[str, int], ...] = ()
    or_none: bool = False

    def contains(self, plan: Plan) -> bool:
        """Whether ``plan`` belongs to this family."""
        if self.laps is None:
            return not plan.stops
        if not plan.stops:
            return self.or_none
        s = plan.stops[0]
        return self.laps[0] <= s.lap <= self.laps[1] and (s.compound, s.tyre_life) in self.sets


@dataclass
class Call:
    """A case study's decision (see ``docs/case_studies.md``)."""

    name: str
    round_number: int
    cutoff: int
    focal: str
    call_family: str  # key in ``families``
    as_run: Plan
    intended: Plan
    #: Stops on the cutoff lap, known at the decision: driver -> (compound, out-lap TyreLife).
    known_stops: dict[str, tuple[str, int]] = field(default_factory=dict)
    #: Filled by :func:`decision_state` from the focal car's sets.
    families: dict[str, Family] = field(default_factory=dict)
    #: (focal sets, cutoff, total laps) -> named families.
    families_fn: Callable | None = None
    #: Short name of the call for figures, e.g. "hards at 39".
    call_label: str = ""
    #: Phase 6 Silverstone additions (defaults keep Hungary as committed).
    #: ``"session"``: gaps from the session clock (:meth:`RaceState.from_laps`).
    timing: str = "laps"
    #: The decision was made on an SC deployment lap: the state starts under
    #: the SC, with the next lap as its first SC lap.
    sc_deployed: bool = False
    #: Cars out of the race at the decision (e.g. the car that caused the SC).
    drop: tuple[str, ...] = ()
    #: Rival plans held fixed in the primary verdict (e.g. Sainz's actual stop).
    fixed: dict[str, Plan] = field(default_factory=dict)
    #: Single-car searches (focal shortlist, rivals) cost stops at their lap's
    #: expected SC/VSC pit loss.
    sc_aware: bool = False
    #: The focal car's teammate, for the joint (team) options.
    teammate: str | None = None
    #: ``leakage_free_params`` keywords (circuit method, in-race update).
    param_kwargs: dict = field(default_factory=dict)


def hungary_call() -> Call:
    """Hungary 2022: Leclerc onto new hards at lap 39 (cutoff lap 38)."""
    return Call(
        name="hungary",
        round_number=13,
        cutoff=38,
        focal="LEC",
        call_family="call: hards at 39",
        as_run=Plan((Stop(39, "HARD", 1), Stop(54, "SOFT", 4))),
        intended=Plan((Stop(39, "HARD", 1),)),
        known_stops={"VER": ("MEDIUM", 1)},
        families_fn=hungary_families,
        call_label="hards at 39",
    )


def hungary54_call() -> Call:
    """Hungary 2022, second decision: Leclerc onto used softs at lap 54 (cutoff 53).

    ``intended`` is the main alternative, hards to the flag (no further stop),
    so it is reported under that name."""
    return Call(
        name="hungary54",
        round_number=13,
        cutoff=53,
        focal="LEC",
        call_family="call: used softs at 54",
        as_run=Plan((Stop(54, "SOFT", 4),)),
        intended=Plan(),
        families_fn=hungary54_families,
        call_label="used softs at 54",
    )


def silverstone_call() -> Call:
    """Britain 2022: Leclerc left out under the lap-39 safety car (cutoff 38).

    The SC was deployed during lap 39 (``docs/case_studies.md``); Sainz pitted
    for new softs at the end of lap 39, Leclerc stayed out on 14-lap hards to
    the flag. Sainz's stop is held fixed in the primary verdict.
    """
    return Call(
        name="silverstone",
        round_number=10,
        cutoff=38,
        focal="LEC",
        call_family="call: stay out at 39",
        as_run=Plan(),
        intended=Plan(),
        # Ricciardo and Tsunoda: in lap 38 by lap number (lapped cars).
        known_stops={"RIC": ("SOFT", 1), "TSU": ("SOFT", 1)},
        families_fn=silverstone_families,
        call_label="stay out at 39",
        timing="session",
        sc_deployed=True,
        drop=("OCO",),
        fixed={"SAI": Plan((Stop(39, "SOFT", 1),))},
        sc_aware=True,
        teammate="SAI",
        param_kwargs={"circuit_method": "relative", "in_race": True, "first_racing_lap": 2},
    )


def silverstone_families(sets: list[tuple[str, int]], cutoff: int, total: int) -> dict[str, Family]:
    """Named alternatives at Silverstone (``docs/case_studies.md``): stay out
    (the call), or pit at the end of lap 39 onto each available set."""
    now = (cutoff + 1, cutoff + 1)
    fams = {
        "call: stay out at 39": Family(
            (cutoff + 2, total - 1), tuple(sorted(set(sets))), or_none=True
        )
    }
    names = {"SOFT": "soft", "MEDIUM": "medium", "HARD": "hard"}
    for comp, life in sorted(set(sets)):
        tag = "new" if life == 1 else f"used {life}"
        fams[f"pit at 39: {names[comp]} ({tag})"] = Family(now, ((comp, life),))
    return fams


def hungary54_families(sets: list[tuple[str, int]], cutoff: int, total: int) -> dict[str, Family]:
    """Named alternatives for the lap-54 decision (``docs/case_studies.md``)."""
    by = {c: tuple(sorted({s for s in sets if s[0] == c})) for c in ("SOFT", "MEDIUM", "HARD")}
    new_h = tuple(s for s in by["HARD"] if s[1] == 1)
    used_m = tuple(s for s in by["MEDIUM"] if s[1] > 1)
    now, later = (cutoff + 1, cutoff + 1), (cutoff + 2, total - 1)
    fams = {
        "call: used softs at 54": Family(now, by["SOFT"]),
        "hards to the flag": Family(None),
        "new hard at 54": Family(now, new_h),
        "used medium at 54": Family(now, used_m),
        "stay out, then used softs": Family(later, by["SOFT"]),
        "stay out, then new hard": Family(later, new_h),
        "stay out, then used medium": Family(later, used_m),
    }
    return {k: v for k, v in fams.items() if v.laps is None or v.sets}


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


def apply_known_stop(
    state: RaceState, driver: str, compound: str, tyre_life: int, outlap_s: float
) -> RaceState:
    """The car pitted at the end of the state's lap: the next lap is its out
    lap on (compound, out-lap ``tyre_life``).

    Its lap time already contains the in-lap part of the pit loss; the out-lap
    part (``outlap_s``) is added to its race time. The fitted set leaves its
    pool and the set taken off joins it as a used set.
    """
    car = state.cars[driver]
    new = replace(
        car,
        compound=compound,
        tyre_life=tyre_life - 1,
        laps_since_stop=-1,
        stint=car.stint + 1,
        race_time=car.race_time + outlap_s,
        compounds_used=car.compounds_used | {compound},
    )
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


def decision_state(
    raw: pd.DataFrame,
    race: pd.DataFrame,
    call: Call,
    cleaned: pd.DataFrame,
    outlap_scale: float = 1.0,
) -> tuple[RaceState, dict]:
    """The race at the cutoff as the pit wall knew it, and notes for the report.

    ``outlap_scale`` scales the out-lap loss added for known stops (e.g. the
    SC pit-loss ratio when those out laps ran under the SC)."""
    from f1strat.tyres import race_used_sets, reconstruct_inventories, sets_left

    rnd, lap = call.round_number, call.cutoff
    total = int(race.loc[race["RoundNumber"] == rnd, "TotalLaps"].iloc[0])
    state = RaceState.from_laps(raw, rnd, lap, total, timing=call.timing)
    if call.drop:
        state = replace(state, cars={d: c for d, c in state.cars.items() if d not in call.drop})
    if call.sc_deployed:
        state = replace(state, track_status="sc", neutral_laps=0)
    invs = reconstruct_inventories(2022, rnd)
    sets = {
        d: sets_left(invs[d], raw, rnd, lap) + race_used_sets(raw, rnd, d, lap)
        for d in state.cars
        if d in invs
    }
    state = state.with_sets(sets)
    out_s = outlap_loss(cleaned, rnd) * outlap_scale
    for d, (comp, life) in call.known_stops.items():
        state = apply_known_stop(state, d, comp, life, out_s)
    call.families = (call.families_fn or hungary_families)(list(state.sets[call.focal]), lap, total)
    notes = {
        "total_laps": total,
        "outlap_loss_s": round(out_s, 2),
        "focal_sets": list(state.sets[call.focal]),
        "known_stops": call.known_stops,
    }
    return state, notes


# --- Field with sensitivity modifications -------------------------------------------


@dataclass
class Check:
    """One robustness setting (``docs/case_studies.md``)."""

    name: str
    widening: float = 1.0
    sc_factor: float = 1.0
    hard_wear_q: float | None = None  # Ferrari deg[hard]+team_deg[hard] quantile
    hard_pace_q: float | None = None  # Ferrari team_compound[hard, r] quantile
    hard_pace_add: float = 0.0  # exploratory: s/lap added to Ferrari's hard pace
    #: Silverstone: the current SC's total length fixed at this quantile of
    #: its conditional distribution (lengths longer than the laps already run).
    sc_len_q: float | None = None
    #: Silverstone: factor on the double-stack D (green and SC).
    ds_factor: float = 1.0
    #: Silverstone: a set removed from the focal car's pool (inventory check).
    drop_set: tuple[str, int] | None = None


def current_sc_length(state: RaceState, neutral: Neutralisations, q: float) -> int:
    """Quantile ``q`` of the current SC's total length, from the observed SC
    lengths longer than the laps already run (inverted CDF: an observed length)."""
    lengths = np.asarray(neutral.sc_lengths)
    longer = lengths[lengths > state.neutral_laps]
    return int(np.quantile(longer, q, method="inverted_cdf"))


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

#: Silverstone: the committed checks plus the additions recorded before any
#: Silverstone simulation (``docs/case_studies.md``).
SILVERSTONE_CHECKS: tuple[Check, ...] = CHECKS + (
    Check("SC length p10", sc_len_q=0.1),
    Check("SC length p90", sc_len_q=0.9),
    Check("double-stack loss x0.5", ds_factor=0.5),
    Check("double-stack loss x2", ds_factor=2.0),
    Check("Leclerc's new hard removed", drop_set=("HARD", 1)),
)


def _flat(fitted: BayesFit, name: str) -> np.ndarray:
    a = fitted.posterior[name].to_numpy()
    return a.reshape(-1, *a.shape[2:]).astype(float)


def ferrari_hard_quantiles(
    fitted: BayesFit, round_number: int, team: str = "Ferrari", qs: tuple[float, ...] = (0.1, 0.9)
) -> dict[str, dict[float, float]]:
    """Posterior quantiles of Ferrari's hard wear (s/lap) and hard pace term (s)."""
    from f1strat.deg_model import TYRE_SCALE

    r, t = fitted.races.index(round_number), fitted.teams.index(team)
    wear = (_flat(fitted, "deg")[:, HARD, r] + _flat(fitted, "team_deg")[:, HARD, t]) / TYRE_SCALE
    pace = _flat(fitted, "team_compound")[:, HARD, r, t]
    return {
        "wear": {q: float(np.quantile(wear, q)) for q in qs},
        "pace": {q: float(np.quantile(pace, q)) for q in qs},
        "wear_mean": float(wear.mean()),
        "pace_mean": float(pace.mean()),
    }


def make_field(
    fitted: BayesFit,
    state: RaceState,
    losses: np.ndarray,
    params: SimParams,
    check: Check,
    n_sims: int,
    seed: int,
    team: str = "Ferrari",
) -> Field:
    """A field for one robustness check (common random numbers by ``seed``)."""
    if check.sc_factor != 1.0:
        params = replace(
            params,
            neutral=replace(
                params.neutral,
                p_sc_per_lap=params.neutral.p_sc_per_lap * check.sc_factor,
                p_vsc_per_lap=params.neutral.p_vsc_per_lap * check.sc_factor,
            ),
        )
    if check.ds_factor != 1.0 and params.double_stack:
        params = replace(
            params, double_stack={k: v * check.ds_factor for k, v in params.double_stack.items()}
        )
    sc_len = (
        current_sc_length(state, params.neutral, check.sc_len_q)
        if check.sc_len_q is not None and state.track_status == "sc"
        else None
    )
    fld = build_field(
        fitted,
        state,
        losses,
        params,
        n_sims=n_sims,
        seed=seed,
        widening=check.widening,
        use_fitted_current_stint=True,
        current_neutral_length=sc_len,
    )
    if check.hard_wear_q is None and check.hard_pace_q is None and not check.hard_pace_add:
        return fld
    q = ferrari_hard_quantiles(
        fitted,
        state.round_number,
        team,
        qs=tuple(x for x in (check.hard_wear_q, check.hard_pace_q) if x is not None) or (0.5,),
    )
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
        m.compound[:, HARD] = m.compound[:, HARD] + check.hard_pace_add
    return fld


# --- Candidates, selection, measurement ----------------------------------------------


def candidates(fld: Field, call: Call, cap: dict[str, int]) -> tuple[list[Plan], dict]:
    """Leclerc's candidate plans and what the tyre-age cap removed."""
    from f1strat.simulator import expected_pit_factor

    st, d = fld.state, call.focal
    m, dr = fld.models[d], fld.draws[d]
    pf = expected_pit_factor(fld) if call.sc_aware else None
    base = search(
        m,
        st,
        d,
        dr,
        max_stops=3,
        min_stint=5,
        max_age=cap,
        include=[call.as_run, call.intended],
        pit_factor=pf,
    )
    plans = list(base.plans)
    for fam in call.families.values():
        if fam.laps is None:
            continue
        res = search(
            m,
            st,
            d,
            dr,
            max_stops=3,
            min_stint=5,
            max_age=cap,
            first_laps=fam.laps,
            first_sets=fam.sets,
            pit_factor=pf,
        )
        plans += [p for p in res.plans if p not in plans]
    car = st.cars[d]
    capped = {
        "n_capped": base.n_capped,
        # plan-dependent part of the expected race time (single car), s
        "best_removed": [
            {"plan": label(p, car.compound), "plan_cost_s": round(t, 1)} for t, p in base.capped
        ],
    }
    return plans, capped


def focal_points(
    fld: Field, call: Call, plans: list[Plan], rivals: dict
) -> tuple[np.ndarray, np.ndarray]:
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
    out.setdefault("hards to the flag", call.intended)
    return out


def band(loss: np.ndarray, level_or_ahead: float) -> str:
    """Verdict band (``docs/case_studies.md``)."""
    if loss.mean() <= 1.0:
        return "within range"
    if loss.mean() > 3.0 and level_or_ahead < 0.25:
        return "outside range"
    return "marginal"


def loss_summary(
    best_pts: np.ndarray, other_pts: np.ndarray, best_pos: np.ndarray, other_pos: np.ndarray
) -> dict[str, float]:
    """Loss distribution (best − other points), and how often the other plan
    finishes level with or ahead of the best."""
    loss = best_pts - other_pts
    return {
        "mean_loss": float(loss.mean()),
        "p10": float(np.quantile(loss, 0.1)),
        "p50": float(np.quantile(loss, 0.5)),
        "p90": float(np.quantile(loss, 0.9)),
        "p_loss_pos": float((loss > 0).mean()),
        "p_loss_neg": float((loss < 0).mean()),
        "p_level_or_ahead": float((other_pos <= best_pos).mean()),
    }


def verdict(
    fitted: BayesFit,
    state: RaceState,
    losses: np.ndarray,
    params: SimParams,
    call: Call,
    cap: dict[str, int],
    check: Check = CHECKS[0],
    n_select: int = 1000,
    n_measure: int = 2000,
    seeds: tuple[int, int] = (101, 202),
) -> dict:
    """Selection on ``n_select`` simulations, measurement on a fresh ``n_measure``."""
    if check.drop_set is not None:
        state, call = without_set(state, call, check.drop_set)
    sel = make_field(fitted, state, losses, params, check, n_select, seeds[0])
    rivals = rivals_for(sel, call, cap)
    plans, capped = candidates(sel, call, cap)
    pts_sel, pos_sel = focal_points(sel, call, plans, rivals)
    chosen = choose(plans, pts_sel, call)

    mea = make_field(fitted, state, losses, params, check, n_measure, seeds[1])
    rivals_m = rivals_for(mea, call, cap)
    names = list(chosen)
    uniq = list(dict.fromkeys(chosen.values()))
    pts_m, pos_m = focal_points(mea, call, uniq, rivals_m)
    k = {n: uniq.index(p) for n, p in chosen.items()}
    best, fam = k["best overall"], k[call.call_family]
    primary = loss_summary(pts_m[best], pts_m[fam], pos_m[best], pos_m[fam])
    car = state.cars[call.focal]
    out = {
        "check": check.name,
        "cutoff": call.cutoff,
        "call_label": call.call_label,
        "band": band(pts_m[best] - pts_m[fam], primary["p_level_or_ahead"]),
        "primary": primary,
        "as_run": loss_summary(pts_m[best], pts_m[k["as run"]], pos_m[best], pos_m[k["as run"]]),
        "hards_to_flag": loss_summary(
            pts_m[best], pts_m[k["hards to the flag"]], pos_m[best], pos_m[k["hards to the flag"]]
        ),
        "chosen": {n: label(chosen[n], car.compound) for n in names},
        "measure": {
            n: {
                "mean_points": float(pts_m[k[n]].mean()),
                "mean_pos": float(pos_m[k[n]].mean()),
                "p_win": float((pos_m[k[n]] == 1).mean()),
                "p_podium": float((pos_m[k[n]] <= 3).mean()),
                "points_dist": {
                    int(v): float((pts_m[k[n]] == v).mean()) for v in np.unique(pts_m[k[n]])
                },
            }
            for n in names
        },
        "selection_table": pd.DataFrame(
            {
                "plan": [label(p, car.compound) for p in plans],
                "mean_points": pts_sel.mean(axis=1),
                "mean_pos": pos_sel.mean(axis=1),
                "family": [
                    next((n for n, f in call.families.items() if f.contains(p)), "") for p in plans
                ],
            }
        ).sort_values("mean_points", ascending=False),
        "n_candidates": len(plans),
        "chosen_plans": chosen,
        "all_plans": plans,
        "capped": capped,
    }
    return out


def rivals_for(fld: Field, call: Call, cap: dict[str, int], exclude: tuple[str, ...] = ()) -> dict:
    """Every rival on its own top-3 single-car plans (SC-aware if the call
    says so), except the call's fixed plans (e.g. Sainz's actual stop)."""
    out = {
        d: rival_plan_options(fld, d, max_age=cap, sc_aware=call.sc_aware)
        for d in fld.drivers
        if d != call.focal and d not in call.fixed and d not in exclude
    }
    out.update({d: p for d, p in call.fixed.items() if d not in exclude})
    return out


def without_set(state: RaceState, call: Call, item: tuple[str, int]) -> tuple[RaceState, Call]:
    """The state and call with one set removed from the focal car's pool
    (inventory robustness check); the named families are rebuilt."""
    pool = list(state.sets[call.focal])
    if item in pool:
        pool.remove(item)
    state = replace(state, sets={**state.sets, call.focal: tuple(pool)})
    call = replace(call, families=call.families_fn(pool, call.cutoff, state.total_laps))
    return state, call


# --- Joint options (Ferrari's combined points; secondary) ---------------------------------

#: Joint option name -> (focal pits on the decision lap, teammate pits on it).
JOINT: dict[str, tuple[bool, bool]] = {
    "SAI pits, LEC stays out (actual family)": (False, True),
    "both pit (double stack)": (True, True),
    "LEC pits, SAI stays out": (True, False),
    "neither pits": (False, False),
}


def _car_candidates(
    fld: Field,
    driver: str,
    pits_now: bool,
    lap: int,
    cap: dict[str, int],
    k: int,
    include: Plan | None = None,
) -> list[Plan]:
    """A car's best ``k`` plans by single-car expected time (SC-aware) that
    pit on ``lap`` (onto any of its sets) or do not (any later stop or none)."""
    from f1strat.simulator import expected_pit_factor

    st, m, dr = fld.state, fld.models[driver], fld.draws[driver]
    sets = tuple(sorted(set(st.sets[driver])))
    first = (lap, lap) if pits_now else (lap + 1, st.total_laps - 1)
    pf = expected_pit_factor(fld)
    res = search(
        m,
        st,
        driver,
        dr,
        max_stops=3,
        min_stint=5,
        max_age=cap,
        per_stratum=1,
        first_laps=first,
        first_sets=sets,
        pit_factor=pf,
    )
    plans = [res.plans[i] for i in np.argsort(res.totals.mean(axis=1))]
    if not pits_now:
        stay = search(m, st, driver, dr, max_stops=0, min_stint=5, max_age=cap, pit_factor=pf)
        plans = [p for p in stay.plans if not p.stops] + plans
        plans = sorted(plans, key=lambda p: plan_totals_mean(fld, driver, p, pf))
    out = plans[:k]
    if (
        include is not None
        and include not in out
        and (bool(include.stops) and include.stops[0].lap == lap) == pits_now
    ):
        out.append(include)
    return out


def plan_totals_mean(fld: Field, driver: str, plan: Plan, pit_factor: np.ndarray | None) -> float:
    """Mean single-car race time of ``plan`` over the field's simulations."""
    from f1strat.simulator import plan_totals

    return float(
        plan_totals(
            fld.models[driver], fld.state, driver, [plan], fld.draws[driver], pit_factor
        ).mean()
    )


def joint_options(
    fitted: BayesFit,
    state: RaceState,
    losses: np.ndarray,
    params: SimParams,
    call: Call,
    cap: dict[str, int],
    actual: dict[str, Plan],
    check: Check = CHECKS[0],
    k: int = 4,
    n_select: int = 1000,
    n_measure: int = 2000,
    seeds: tuple[int, int] = (111, 222),
) -> dict:
    """Ferrari's combined points per joint option on the decision lap
    (secondary; never changes the band).

    For each option in :data:`JOINT`, every pair of the two cars' best ``k``
    plans in that option (by single-car expected time, SC-aware) is simulated
    with the field; the pair with the most combined points on the selection
    set is measured on a fresh set, with the actual pair. Other rivals are on
    their own plans.
    """
    focal, mate, lap = call.focal, call.teammate, call.cutoff + 1
    sel = make_field(fitted, state, losses, params, check, n_select, seeds[0])
    riv = rivals_for(sel, call, cap, exclude=(mate,))
    from f1strat.simulator import simulate_field

    def combined(fld, rivals, pf, pm):
        res = simulate_field(fld, {**rivals, focal: pf, mate: pm})
        pos = res.positions
        return (points(pos[:, res.drivers.index(focal)]), points(pos[:, res.drivers.index(mate)]))

    chosen, tables = {}, []
    for name, (f_pits, m_pits) in JOINT.items():
        cf = _car_candidates(sel, focal, f_pits, lap, cap, k, actual[focal])
        cm = _car_candidates(sel, mate, m_pits, lap, cap, k, actual[mate])
        best = None
        for pf in cf:
            for pm in cm:
                a, b = combined(sel, riv, pf, pm)
                tables.append(
                    {
                        "option": name,
                        "focal": label(pf, state.cars[focal].compound),
                        "teammate": label(pm, state.cars[mate].compound),
                        "focal_points": a.mean(),
                        "teammate_points": b.mean(),
                        "combined": (a + b).mean(),
                    }
                )
                if best is None or (a + b).mean() > best[0]:
                    best = ((a + b).mean(), pf, pm)
        chosen[name] = (best[1], best[2])
    chosen["actual"] = (actual[focal], actual[mate])

    mea = make_field(fitted, state, losses, params, check, n_measure, seeds[1])
    riv_m = rivals_for(mea, call, cap, exclude=(mate,))
    pts = {n: combined(mea, riv_m, pf, pm) for n, (pf, pm) in chosen.items()}
    act = pts["actual"][0] + pts["actual"][1]
    out = {}
    for n, (pf, pm) in chosen.items():
        a, b = pts[n]
        diff = (a + b) - act
        out[n] = {
            "focal_plan": label(pf, state.cars[focal].compound),
            "teammate_plan": label(pm, state.cars[mate].compound),
            "focal_points": float(a.mean()),
            "teammate_points": float(b.mean()),
            "combined": float((a + b).mean()),
            "vs_actual_mean": float(diff.mean()),
            "vs_actual_p10": float(np.quantile(diff, 0.1)),
            "vs_actual_p90": float(np.quantile(diff, 0.9)),
            "p_better_than_actual": float((diff > 0).mean()),
        }
    return {"options": out, "selection": pd.DataFrame(tables)}


# --- Hindsight replay ------------------------------------------------------------------


def attach_actual_stints(
    fld: Field, fitted: BayesFit, raw: pd.DataFrame, round_number: int
) -> dict[str, int]:
    """Give every car the fitted effects of its stints that actually happened
    after the state lap (hindsight rule in ``docs/case_studies.md``).

    A stint is keyed by its first lap, compound and TyreLife on that lap; a plan
    that fits the same set on the same lap uses the fitted effect for the whole
    stint, however long the plan runs it. Returns the count attached per car.
    """
    post = fitted.posterior["stint_effect"].to_numpy()
    eff = post.reshape(-1, post.shape[-1])[fld.draw_index]
    laps = raw[raw["RoundNumber"] == round_number]
    out = {}
    for d in fld.drivers:
        mine = laps[laps["Driver"] == d]
        n = 0
        for stint, g in mine.groupby("Stint"):
            first = g.loc[g["LapNumber"].idxmin()]
            key = (round_number, d, int(stint))
            if (
                int(first["LapNumber"]) <= fld.state.lap
                or key not in fitted.stints
                or pd.isna(first["TyreLife"])
            ):
                continue
            fld.models[d].known_stints[
                (int(first["LapNumber"]), str(first["Compound"]), int(first["TyreLife"]))
            ] = eff[:, fitted.stints.index(key)]
            n += 1
        out[d] = n
    return out


def hindsight(
    full_fit: BayesFit,
    raw: pd.DataFrame,
    race: pd.DataFrame,
    cleaned_full: pd.DataFrame,
    call: Call,
    state: RaceState,
    plans: dict[str, Plan],
    all_plans: list[Plan] = (),
    n_sims: int = 1000,
    seed: int = 303,
    sim_params: SimParams | None = None,
) -> dict:
    """Leclerc's plans under what actually happened after the cutoff (not the
    verdict). ``all_plans`` (e.g. every decision-lap candidate) are also run to
    find the best plan in hindsight."""
    from f1strat.backtest import actual_race
    from f1strat.simulator import simulate_field

    act = actual_race(raw, race, call.round_number)
    fld, rivals, retire = hindsight_field(
        full_fit, raw, race, cleaned_full, call, state, n_sims, seed, sim_params
    )
    attached = {d: len(m.known_stints) for d, m in fld.models.items()}
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
        out[name] = {
            "plan": label(plan, state.cars[call.focal].compound),
            "mean_points": float(pts.mean()),
            "mean_pos": float(pos.mean()),
            "p_win": float((pos == 1).mean()),
            "points_dist": {int(v): float((pts == v).mean()) for v in np.unique(pts)},
        }
    finish = {d: i + 1 for i, d in enumerate(act.order)}
    out["actual result"] = {
        "position": finish.get(call.focal),
        "points": int(points([finish[call.focal]])[0]),
    }
    out["stints_attached"] = attached
    return out


def hindsight_field(
    full_fit: BayesFit,
    raw: pd.DataFrame,
    race: pd.DataFrame,
    cleaned_full: pd.DataFrame,
    call: Call,
    state: RaceState,
    n_sims: int = 1000,
    seed: int = 303,
    sim_params: SimParams | None = None,
) -> tuple[Field, dict, dict[str, int]]:
    """(field, rival actual plans, retirements) for a hindsight replay.

    ``sim_params`` defaults to the season's ``sim_params.json``."""
    from f1strat.backtest import actual_race
    from f1strat.race_events import SimParams
    from f1strat.simulator import actual_plan, observed_pit_losses

    act = actual_race(raw, race, call.round_number)
    fld = build_field(
        full_fit,
        state,
        observed_pit_losses(cleaned_full, call.round_number),
        sim_params if sim_params is not None else SimParams.load(),
        n_sims=n_sims,
        seed=seed,
        use_fitted_current_stint=True,
        status=act.status[call.cutoff + 1 :],
    )
    attach_actual_stints(fld, full_fit, raw, call.round_number)
    rivals = {
        d: actual_plan(raw, call.round_number, d, call.cutoff)
        for d in fld.drivers
        if d != call.focal
    }
    retire = {d: act.last_lap[d] for d in fld.drivers if not act.finished[d]}
    return fld, rivals, retire


def reality_check(
    dec_fit: BayesFit,
    full_fit: BayesFit,
    raw: pd.DataFrame,
    race: pd.DataFrame,
    cleaned_cut: pd.DataFrame,
    cleaned_full: pd.DataFrame,
    call: Call,
    state: RaceState,
    params: SimParams,
    cap: dict[str, int],
    plan: Plan,
    last_lap: int = 53,
    rival: str = "VER",
    window: int = 5,
    n_sims: int = 1000,
) -> dict:
    """The call's simulated laps against the real ones (``docs/case_studies.md``).

    Lap times are summarised over simulations where that lap was green (the
    real laps were), as the median and 10th/90th percentiles (``lap_mean``
    holds the median). Gaps use every simulation.

    For laps ``cutoff+1 .. last_lap``: the focal car's lap times and its gap to
    ``rival`` in (a) the decision-lap forecast (cutoff fit, rivals' own plans,
    drawn SC/VSC) and (b) the hindsight replay, against the real race. Also how
    often ``rival`` is ahead within ``window`` laps of the focal car's stop.
    """
    from f1strat.simulator import observed_pit_losses, simulate_field

    focal, rnd, cut = call.focal, call.round_number, call.cutoff
    laps_idx = np.arange(cut + 1, last_lap + 1)
    real = raw[(raw["RoundNumber"] == rnd) & raw["LapNumber"].isin(laps_idx)]
    rt = real.pivot(index="LapNumber", columns="Driver", values="Time").apply(
        lambda c: c.dt.total_seconds()
    )
    real_lap = (
        real[real["Driver"] == focal]
        .set_index("LapNumber")["LapTime"]
        .dt.total_seconds()
        .reindex(laps_idx)
    )
    real_rival_lap = (
        real[real["Driver"] == rival]
        .set_index("LapNumber")["LapTime"]
        .dt.total_seconds()
        .reindex(laps_idx)
    )
    real_gap = (rt[focal] - rt[rival]).reindex(laps_idx)
    real_pos = real.pivot(index="LapNumber", columns="Driver", values="Position")

    dec = make_field(
        dec_fit, state, observed_pit_losses(cleaned_cut, rnd), params, CHECKS[0], n_sims, seed=202
    )
    dec_rivals = {d: rival_plan_options(dec, d, max_age=cap) for d in dec.drivers if d != focal}
    hfld, h_rivals, retire = hindsight_field(full_fit, raw, race, cleaned_full, call, state, n_sims)
    stop = plan.stops[0].lap
    out = {
        "laps": laps_idx.tolist(),
        "real_lap_s": real_lap.round(3).tolist(),
        "real_rival_lap_s": real_rival_lap.round(3).tolist(),
        "real_gap_s": real_gap.round(3).tolist(),
        "real_rival_ahead_laps": [
            int(L) for L in laps_idx if real_pos.at[L, rival] < real_pos.at[L, focal]
        ],
    }
    for name, fld, rivals, ret in (
        ("forecast", dec, dec_rivals, {}),
        ("hindsight", hfld, h_rivals, retire),
    ):
        res = simulate_field(fld, {**rivals, focal: plan}, retire=ret, record=True)
        c, v = res.drivers.index(focal), res.drivers.index(rival)
        start = np.array([state.cars[d].race_time for d in res.drivers])
        hist = np.concatenate(
            [np.broadcast_to(start, (fld.n_sims, 1, len(start))), res.history], axis=1
        )
        n = len(laps_idx)
        # Lap times on green laps only (the real laps were all green); medians.
        green = np.asarray(fld.status[:, :n] == 0)
        lap_t = np.where(green, np.diff(hist[:, : n + 1, c], axis=1), np.nan)
        riv_t = np.where(green, np.diff(hist[:, : n + 1, v], axis=1), np.nan)
        gap = hist[:, 1 : n + 1, c] - hist[:, 1 : n + 1, v]
        ahead = hist[:, 1 : n + 1, v] < hist[:, 1 : n + 1, c]
        w = laps_idx <= stop + window
        out[name] = {
            "lap_mean": np.nanmedian(lap_t, axis=0).round(3).tolist(),
            "lap_p10": np.nanquantile(lap_t, 0.1, axis=0).round(3).tolist(),
            "lap_p90": np.nanquantile(lap_t, 0.9, axis=0).round(3).tolist(),
            "rival_lap_median": np.nanmedian(riv_t, axis=0).round(3).tolist(),
            "share_green": green.mean(axis=0).round(3).tolist(),
            "gap_mean": gap.mean(axis=0).round(3).tolist(),
            "gap_p10": np.quantile(gap, 0.1, axis=0).round(3).tolist(),
            "gap_p90": np.quantile(gap, 0.9, axis=0).round(3).tolist(),
            f"p_{rival}_ahead_within_{window}": float(ahead[:, w].any(axis=1).mean()),
            f"p_{rival}_ahead_at_lap_{stop + window}": float(ahead[:, w][:, -1].mean()),
            f"p_{rival}_ahead_at_lap_{last_lap}": float(ahead[:, -1].mean()),
        }
    # Post-stop laps: real minus forecast mean, first two flying laps vs later.
    hard = COMPOUNDS.index(plan.stops[0].compound)
    ps = _flat(dec_fit, "post_stop")[:, hard]
    resid = np.array(out["real_lap_s"]) - np.array(out["forecast"]["lap_mean"])
    early = (laps_idx >= stop + 2) & (laps_idx <= stop + 3)  # 1st/2nd lap after the out lap
    later = laps_idx > stop + 3
    out["post_stop"] = {
        "posterior_mean": float(ps.mean()),
        "p10": float(np.quantile(ps, 0.1)),
        "p90": float(np.quantile(ps, 0.9)),
        "resid_first_two_flying": float(np.nanmean(resid[early])),
        "resid_later": float(np.nanmean(resid[later])),
    }
    return out


def plot_reality(rc: dict, path: Path, focal: str = "LEC", rival: str = "VER") -> Path:
    """Focal lap times and the rival gap: real vs forecast vs hindsight."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.array(rc["laps"])
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, facecolor=SURFACE)
    for ax, key, real, ylab, k0 in (
        (axes[0], "lap", rc["real_lap_s"], f"{focal} lap time (s)", 0),
        (axes[1], "gap", rc["real_gap_s"], f"{focal} behind {rival} (s; negative = ahead)", 1),
    ):
        # The gap starts after both stops: the real data put the rival's out
        # lap on the first lap, the simulation applied it at the cutoff.
        _style(ax)
        xs, real = x[k0:], real[k0:]
        for name, color in (("forecast", BLUE), ("hindsight", ORANGE)):
            d = rc[name]
            ax.fill_between(
                xs, d[f"{key}_p10"][k0:], d[f"{key}_p90"][k0:], color=color, alpha=0.15, lw=0
            )
            ax.plot(
                xs,
                d[f"{key}_mean"][k0:],
                color=color,
                lw=2,
                label={
                    "forecast": "lap-38 forecast (median; gap: mean; 10–90%)",
                    "hindsight": "hindsight replay (median; gap: mean; 10–90%)",
                }[name],
            )
        ax.plot(xs, real, "o-", color=INK, lw=1.2, ms=4, label="real")
        ax.set_ylabel(ylab, color=INK2, fontsize=9)
    axes[0].set_ylim(
        min(rc["real_lap_s"][2:] + rc["forecast"]["lap_p10"][2:]) - 1,
        max(rc["real_lap_s"][2:] + rc["forecast"]["lap_p90"][2:]) + 1,
    )
    axes[0].set_title(
        "Leclerc on hards from lap 40: real vs simulated (in lap 39 and out "
        "lap 40 off the top of the scale)",
        fontsize=10,
        color=INK,
        loc="left",
    )
    axes[0].legend(fontsize=8, frameon=False)
    axes[1].axhline(0, color=GRID, lw=1)
    axes[1].set_xlabel("lap", color=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def compare_hard_posteriors(
    fits: dict[str, BayesFit],
    round_number: int,
    driver: str = "LEC",
    stint: int = 3,
    team: str = "Ferrari",
) -> dict:
    """Ferrari's hard terms in several fits (e.g. the lap-38 and lap-53 cutoff
    fits and the full fit), keyed as ``fits``: team compound term, field hard
    offset, hard wear, the driver's stint effect (if fitted) and the sum."""
    from f1strat.deg_model import TYRE_SCALE

    def summ(v):
        return {
            "mean": float(v.mean()),
            "p10": float(np.quantile(v, 0.1)),
            "p90": float(np.quantile(v, 0.9)),
        }

    out = {}
    for name, f in fits.items():
        r, t = f.races.index(round_number), f.teams.index(team)
        tc = _flat(f, "team_compound")[:, HARD, r, t]
        off = _flat(f, "compound_offset")[:, HARD, r]
        wear = (_flat(f, "deg")[:, HARD, r] + _flat(f, "team_deg")[:, HARD, t]) / TYRE_SCALE
        key = (round_number, driver, stint)
        st = _flat(f, "stint_effect")[:, f.stints.index(key)] if key in f.stints else None
        out[name] = {
            "team_compound": summ(tc),
            "field_hard_offset": summ(off),
            "wear_s_per_lap": summ(wear),
            "stint_effect": None if st is None else summ(st),
            "team_plus_stint": summ(tc + (st if st is not None else 0.0)),
        }
    return out


# --- Figures and report ----------------------------------------------------------------

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
#: Ordinal blue ramp (light -> dark = fewer -> more points), from step 250.
POINT_BINS = [
    ("0", [0], "#86b6ef"),
    ("1–8 (P6–P10)", [1, 2, 4, 6, 8], "#5598e7"),
    ("10 (P5)", [10], "#3987e5"),
    ("12 (P4)", [12], "#2a78d6"),
    ("15 (P3)", [15], "#256abf"),
    ("18 (P2)", [18], "#184f95"),
    ("25 (win)", [25], "#0d366b"),
]


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


def plot_points(main: dict, path: Path, race_name: str = "Hungary") -> Path:
    """Leclerc's points distribution per plan (measurement set, decision lap)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    meas = main["measure"]
    rows = sorted(_merged(main), key=lambda r: meas[r[0].split(" = ")[0]]["mean_points"])
    fig, ax = plt.subplots(figsize=(9.5, 0.6 * len(rows) + 1.8), facecolor=SURFACE)
    _style(ax)
    seen: set[str] = set()
    for y, (joined, _) in enumerate(rows):
        n = joined.split(" = ")[0]
        left = 0.0
        dist = meas[n]["points_dist"]
        for label, vals, color in POINT_BINS:
            w = sum(dist.get(v, dist.get(str(v), 0.0)) for v in vals)
            if w > 0:
                # Legend entry on each bin's first appearance (any row).
                ax.barh(
                    y,
                    w - 0.004,
                    left=left,
                    color=color,
                    height=0.62,
                    label=label if label not in seen else None,
                )
                seen.add(label)
            left += w
        ax.text(1.01, y, f"{meas[n]['mean_points']:.1f} pts", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(rows)), [f"{j}\n{lbl}" for j, lbl in rows], fontsize=8.5, color=INK)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of simulations", color=INK2)
    ax.set_title(
        f"Leclerc's points per plan at the decision lap ({race_name}, end of lap "
        f"{main.get('cutoff', 38)}; 2,000 simulations)",
        fontsize=10,
        color=INK,
        loc="left",
    )
    handles, labels = ax.get_legend_handles_labels()
    order = [b[0] for b in POINT_BINS]
    pairs = sorted(zip(handles, labels), key=lambda hl: order.index(hl[1]))
    handles, labels = [h for h, _ in pairs], [lbl for _, lbl in pairs]
    ax.legend(
        handles,
        labels,
        title="points",
        ncol=7,
        fontsize=8,
        title_fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12 - 0.3 / len(rows)),
        frameon=False,
    )
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
        ax.text(
            1.01,
            y,
            f"{c['band']} (mean {p['mean_loss']:.1f})",
            va="center",
            transform=ax.get_yaxis_transform(),
            fontsize=9,
            color=INK,
        )
    for x, lbl, ha, dx in (
        (1, "within if mean ≤ 1", "right", -0.08),
        (3, "outside if mean > 3\nand level/ahead < 25%", "left", 0.08),
    ):
        ax.axvline(x, color=INK2, lw=1, ls="--")
        ax.text(x + dx, -0.85, lbl, fontsize=8, color=INK2, ha=ha, va="center")
    ax.axvline(0, color=GRID, lw=1)
    ax.set_yticks(ys, [c["check"] for c in checks], fontsize=9, color=INK)
    call_label = checks[0].get("call_label", "hards at 39")
    ax.set_xlabel(
        f"points lost by the call (best plan − best '{call_label}' plan); "
        "dot = mean, line = 10th–90th percentile",
        color=INK2,
        fontsize=9,
    )
    ax.set_title("Verdict under each robustness check", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_hindsight(main: dict, hind: dict, path: Path, race_name: str = "Hungary") -> Path:
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
        ax.plot(
            a,
            y,
            "o",
            color=BLUE,
            ms=8,
            mec=SURFACE,
            mew=2,
            label="decision lap (cutoff fit, drawn SC/VSC, rivals' own plans)" if y == 0 else None,
        )
        ax.plot(
            b,
            y,
            "D",
            color=ORANGE,
            ms=7,
            mec=SURFACE,
            mew=2,
            label="hindsight (full fit, actual SC/VSC, rivals' actual plans)" if y == 0 else None,
        )
    if "best in hindsight" in hind:
        y = len(names)
        names.append("best in hindsight")
        rows.append(("best in hindsight", hind["best in hindsight"]["plan"]))
        ax.plot(
            hind["best in hindsight"]["mean_points"], y, "D", color=ORANGE, ms=7, mec=SURFACE, mew=2
        )
    act = hind["actual result"]
    ax.axvline(act["points"], color=INK2, lw=1, ls="--")
    ax.set_ylim(-1.1, len(names) - 0.4)
    ax.text(
        act["points"],
        -0.85,
        f" actual race: P{act['position']}, {act['points']} pts",
        fontsize=8,
        color=INK2,
    )
    ax.set_yticks(range(len(names)), [f"{j}\n{lbl}" for j, lbl in rows], fontsize=8, color=INK)
    ax.set_xlabel("Leclerc's mean points", color=INK2)
    ax.set_title(
        f"Decision lap (end of lap {main.get('cutoff', 38)}) vs hindsight ({race_name})",
        fontsize=10,
        color=INK,
        loc="left",
    )
    ax.legend(
        fontsize=8,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1 - 0.4 / len(names)),
        ncol=1,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


#: Exploratory, outside the committed rule: how much slower Ferrari's hards
#: would have to be (s/lap, on top of the cutoff fit) before the band changes.
BREAK_EVEN: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)


def break_even(
    fitted: BayesFit,
    state: RaceState,
    losses: np.ndarray,
    params: SimParams,
    call: Call,
    cap: dict[str, int],
    shifts: Sequence[float] = BREAK_EVEN,
) -> list[dict]:
    """The verdict with Ferrari's hard pace shifted by each of ``shifts``."""
    out = []
    for x in shifts:
        v = verdict(
            fitted,
            state,
            losses,
            params,
            call,
            cap,
            Check(f"Ferrari hard pace +{x:.2f} s/lap", hard_pace_add=x),
        )
        out.append({k: v[k] for k in ("check", "band", "primary", "chosen")})
    return out


CALLS: dict[str, Callable[[], Call]] = {
    "hungary": hungary_call,
    "hungary54": hungary54_call,
    "silverstone": silverstone_call,
}


@dataclass
class CaseInputs:
    """Everything a Hungary decision needs at its cutoff (:func:`hungary_inputs`)."""

    call: Call
    raw: pd.DataFrame
    race: pd.DataFrame
    cleaned: pd.DataFrame  # cleaning recomputed from the laps known at the cutoff
    full: BayesFit  # the divergence-free cutoff fit (all draws)
    fitted: BayesFit  # the same, thinned
    state: RaceState
    notes: dict
    cap: dict[str, int]
    losses: np.ndarray
    params: object
    pinfo: dict


def hungary_inputs(name: str = "hungary") -> CaseInputs:
    """The cutoff fit, race state, tyre-age cap, pit losses and leakage-free
    parameters for ``hungary`` (lap-38 cutoff) or ``hungary54`` (lap 53).

    Refits the cutoff fit if it is not saved (:func:`f1strat.rehearsal.cutoff_fit`).
    """
    from f1strat.data import load_season
    from f1strat.deg_model import Cutoff, cutoff_inputs, thin_fit
    from f1strat.rehearsal import cutoff_fit, leakage_free_params
    from f1strat.simulator import observed_pit_losses

    call = CALLS[name]()
    raw, race = load_season(2022), load_season(2022, table="race")
    cleaned, laps = cutoff_inputs(raw, Cutoff(call.round_number, call.cutoff))
    full = cutoff_fit(raw, call.round_number, call.cutoff)
    fitted = thin_fit(full)
    state, notes = decision_state(raw, race, call, cleaned)
    params, pinfo = leakage_free_params(fitted, raw, race, cleaned, call.round_number)
    return CaseInputs(
        call,
        raw,
        race,
        cleaned,
        full,
        fitted,
        state,
        notes,
        tyre_age_cap(laps),
        observed_pit_losses(cleaned, call.round_number),
        params,
        pinfo,
    )


def run_hungary(name: str = "hungary", hindsight_only: bool = False) -> dict:
    """One Hungary decision: the verdict under every committed check, the
    hindsight replay and (lap 38) the reality check. Writes
    ``case_<name>_results.json``, ``_candidates.csv`` and the figures.

    ``hindsight_only`` reruns the hindsight replay (and the reality check) for
    the plans saved by an earlier full run, leaving the verdict untouched.
    """
    import json
    import time

    from f1strat.clean import clean_laps
    from f1strat.deg_model import THIN_FIT_PATH, load_fit

    x = hungary_inputs(name)
    call, raw, race, fitted, state = x.call, x.raw, x.race, x.fitted, x.state
    stem = f"case_{call.name}"
    full_season = load_fit(THIN_FIT_PATH)
    cleaned_full = clean_laps(raw)

    if hindsight_only:
        results = json.loads((FIG_DIR / f"{stem}_results.json").read_text())
        main_v = results["checks"][0]
        chosen_plans = {n: parse_label(lbl) for n, lbl in main_v["chosen"].items()}
        cand = pd.read_csv(FIG_DIR / f"{stem}_candidates.csv")
        all_plans = [parse_label(lbl) for lbl in cand["plan"]]
    else:
        results = {
            "notes": x.notes,
            "cap": x.cap,
            "n_pit_losses": len(x.losses),
            "params": x.pinfo,
            "fit_diagnostics": {k: float(v) for k, v in x.full.diagnostics.items()},
            "ferrari_hard": ferrari_hard_quantiles(fitted, call.round_number),
            "checks": [],
        }
        print(
            json.dumps(
                {k: results[k] for k in ("notes", "cap", "params", "ferrari_hard")},
                indent=1,
                default=str,
            ),
            flush=True,
        )
        tables = {}
        for check in CHECKS:
            t0 = time.time()
            v = verdict(fitted, state, x.losses, x.params, call, x.cap, check)
            tables[check.name] = v.pop("selection_table")
            objs = (v.pop("chosen_plans"), v.pop("all_plans"))
            if check.name == "main":
                chosen_plans, all_plans = objs
            results["checks"].append(v)
            print(
                f"{check.name}: {v['band']}, mean loss {v['primary']['mean_loss']:.2f}, "
                f"chosen {v['chosen']['best overall']} ({time.time() - t0:.0f} s)",
                flush=True,
            )
        main_v = results["checks"][0]
        tables["main"].round(3).to_csv(FIG_DIR / f"{stem}_candidates.csv", index=False)
        plot_points(main_v, FIG_DIR / f"{stem}_points.png")
        plot_sensitivity(results["checks"], FIG_DIR / f"{stem}_sensitivity.png")

    hind = hindsight(
        full_season, raw, race, cleaned_full, call, state, chosen_plans, all_plans=all_plans
    )
    results["hindsight"] = hind
    plot_hindsight(main_v, hind, FIG_DIR / f"{stem}_hindsight.png")
    if call.name == "hungary":
        rc = reality_check(
            fitted,
            full_season,
            raw,
            race,
            x.cleaned,
            cleaned_full,
            call,
            state,
            x.params,
            x.cap,
            call.intended,
        )
        results["reality_check"] = rc
        plot_reality(rc, FIG_DIR / f"{stem}_reality.png")
    (FIG_DIR / f"{stem}_results.json").write_text(json.dumps(results, indent=1, default=str))
    print(
        json.dumps(
            {
                "checks": [
                    {
                        k: c[k]
                        for k in ("check", "band", "primary", "as_run", "hards_to_flag", "chosen")
                    }
                    for c in results["checks"]
                ],
                "hindsight": hind,
            },
            indent=1,
            default=str,
        )
    )
    return results


def hungary_break_even(shifts: Sequence[float] = BREAK_EVEN) -> list[dict]:
    """Exploratory, outside the committed rule: the lap-38 verdict with
    Ferrari's hard pace shifted by each of ``shifts``. Writes
    ``case_hungary_break_even.json``."""
    import json

    x = hungary_inputs("hungary")
    out = break_even(x.fitted, x.state, x.losses, x.params, x.call, x.cap, shifts)
    (FIG_DIR / "case_hungary_break_even.json").write_text(json.dumps(out, indent=1, default=str))
    return out


def hungary_hard_posterior() -> dict:
    """How Ferrari's Hungary hard terms moved between the lap-38 cutoff fit,
    the lap-53 cutoff fit and the full fit. Writes
    ``case_hungary_hard_posterior.json``."""
    import json

    from f1strat.data import load_season
    from f1strat.deg_model import THIN_FIT_PATH, load_fit, thin_fit
    from f1strat.rehearsal import cutoff_fit

    raw = load_season(2022)
    fits = {
        "lap38": thin_fit(cutoff_fit(raw, 13, 38)),
        "lap53": thin_fit(cutoff_fit(raw, 13, 53)),
        "full": load_fit(THIN_FIT_PATH),
    }
    out = compare_hard_posteriors(fits, 13)
    (FIG_DIR / "case_hungary_hard_posterior.json").write_text(json.dumps(out, indent=1))
    return out


#: Simulations kept per lap in ``data/figures/hungary_fan.csv`` (README banner and animation).
FAN_SIMS: int = 300


def hungary_fan(n_sims: int = 1000, keep: int = FAN_SIMS, seed: int = 202) -> Path:
    """Leclerc's simulated gap to Verstappen after lap 38 under the call
    (``M → 39 H(new)``, hards to the flag), as the lap-38 pit wall would have
    forecast it: the cutoff fit, rivals on their own plans, SC/VSC drawn.

    The same forecast as :func:`reality_check` (same field and seed), run to
    the flag. Writes ``data/figures/hungary_fan.csv``: one row per lap from 38 to 70,
    columns ``real`` (the actual gap, which includes the lap-54 stop),
    ``p10``/``p50``/``p90`` over all ``n_sims`` and the first ``keep``
    simulations as ``s0``, ``s1``, ... (s; positive = Leclerc behind).
    """
    from f1strat.simulator import simulate_field

    x = hungary_inputs("hungary")
    call, state, rnd = x.call, x.state, x.call.round_number
    fld = make_field(x.fitted, state, x.losses, x.params, CHECKS[0], n_sims, seed=seed)
    rivals = {d: rival_plan_options(fld, d, max_age=x.cap) for d in fld.drivers if d != "LEC"}
    res = simulate_field(fld, {**rivals, "LEC": call.intended}, record=True)
    c, v = res.drivers.index("LEC"), res.drivers.index("VER")
    start = state.cars["LEC"].race_time - state.cars["VER"].race_time
    gap = np.concatenate(
        [np.full((fld.n_sims, 1), start), res.history[:, :, c] - res.history[:, :, v]], axis=1
    )
    laps = np.arange(call.cutoff, state.total_laps + 1)
    real = x.raw[(x.raw["RoundNumber"] == rnd) & x.raw["LapNumber"].isin(laps)]
    rt = real.pivot(index="LapNumber", columns="Driver", values="Time").apply(
        lambda s: s.dt.total_seconds()
    )
    df = pd.DataFrame({"lap": laps, "real": (rt["LEC"] - rt["VER"]).reindex(laps).to_numpy()})
    for q in (10, 50, 90):
        df[f"p{q}"] = np.quantile(gap, q / 100, axis=0)
    sims = pd.DataFrame(gap[:keep].T, columns=[f"s{i}" for i in range(min(keep, len(gap)))])
    from f1strat.figure_data import path as table_path

    path = table_path("hungary_fan")
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([df, sims], axis=1).round(2).to_csv(path, index=False)
    return path


def main(argv: list[str] | None = None) -> None:
    """``python -m f1strat.case_study <name> [--hindsight-only] [--v2]``.

    Names: ``hungary``, ``hungary54``, ``hungary-break-even``,
    ``hungary-hard-posterior``, ``hungary-fan``, ``pass-sensitivity``,
    ``pass-sensitivity-in-race``, ``silverstone`` (``--v2``: restart layer),
    ``silverstone-break-even``. The command-line entry point
    (:mod:`f1strat.cli`, ``pitwall case-study``) groups these per race.
    """
    import logging
    import sys

    logging.disable(logging.WARNING)
    args = argv if argv is not None else sys.argv[1:]
    jobs: dict[str, Callable[[], object]] = {
        "hungary": lambda: run_hungary("hungary", "--hindsight-only" in args),
        "hungary54": lambda: run_hungary("hungary54", "--hindsight-only" in args),
        "hungary-break-even": hungary_break_even,
        "hungary-hard-posterior": hungary_hard_posterior,
        "hungary-fan": hungary_fan,
        "pass-sensitivity": pass_sensitivity,
        "pass-sensitivity-in-race": pass_sensitivity_in_race,
        "silverstone": lambda: run_silverstone(
            hindsight_only="--hindsight-only" in args, version="v2" if "--v2" in args else "v1"
        ),
        "silverstone-break-even": silverstone_break_even,
    }
    if not args or args[0] not in jobs:
        raise SystemExit(
            f"usage: python -m f1strat.case_study <{'|'.join(jobs)}> [--hindsight-only] [--v2]"
        )
    jobs[args[0]]()


# --- Post-hoc pass-difficulty sensitivity (Hungary) ---------------------------------------


def pass_sensitivity_in_race(names: Sequence[str] = ("hungary", "hungary54")) -> dict:
    """Post-hoc row for Hungary's pass-difficulty table, added with Silverstone's
    method (``docs/case_studies.md``): the relative intercept updated with
    Hungary's own on-track passes before each cutoff
    (:func:`f1strat.race_events.in_race_pass_update`), main check, same seeds.
    Appended to ``case_hungary_pass_sensitivity.json``."""
    import json
    import logging

    from f1strat.data import load_season
    from f1strat.deg_model import Cutoff, cutoff_inputs, thin_fit
    from f1strat.race_events import in_race_pass_update, relative_circuit_intercept
    from f1strat.rehearsal import cutoff_fit, leakage_free_params
    from f1strat.simulator import observed_pit_losses

    logging.disable(logging.WARNING)
    raw, race = load_season(2022), load_season(2022, table="race")
    path = FIG_DIR / "case_hungary_pass_sensitivity.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    for name in names:
        call = CALLS[name]()
        rnd = call.round_number
        cleaned, laps = cutoff_inputs(raw, Cutoff(rnd, call.cutoff))
        fitted = thin_fit(cutoff_fit(raw, rnd, call.cutoff))
        state, _ = decision_state(raw, race, call, cleaned)
        losses = observed_pit_losses(cleaned, rnd)
        params, _ = leakage_free_params(fitted, raw, race, cleaned, rnd)
        event = str(race.loc[race["RoundNumber"] == rnd, "EventName"].iloc[0])
        rel = relative_circuit_intercept(event, params.passing.intercept, 2021)
        upd = in_race_pass_update(cleaned, fitted, rnd, rel["intercept"], params.passing.slope)
        circ = {**params.passing.circuit, rnd: upd["intercept"]}
        prm = replace(params, passing=replace(params.passing, circuit=circ))
        v = verdict(fitted, state, losses, prm, call, tyre_age_cap(laps), CHECKS[0])
        out.setdefault(name, {"settings": {}})["settings"]["relative + in-race (post-hoc)"] = {
            "intercept": upd["intercept"],
            "in_race": upd,
            "band": v["band"],
            "primary": v["primary"],
            "chosen": v["chosen"],
            "measure": {
                n: {k: m[k] for k in ("mean_points", "mean_pos", "p_podium")}
                for n, m in v["measure"].items()
            },
        }
        print(
            name,
            "relative + in-race",
            round(upd["intercept"], 3),
            upd["events"],
            upd["passes"],
            v["band"],
            round(v["primary"]["mean_loss"], 2),
            round(v["primary"]["p_level_or_ahead"], 3),
            v["chosen"]["best overall"],
            flush=True,
        )
    path.write_text(json.dumps(out, indent=1, default=str))
    return out


def pass_sensitivity(names: Sequence[str] = ("hungary", "hungary54")) -> dict:
    """Post-hoc, labelled sensitivity (``docs/case_studies.md``): each decision's
    verdict (main check) with the decision circuit's pass intercept set three
    ways, everything else as committed:

    - ``committed``: from the previous running (Hungary 2021);
    - ``average``: the pooled 2022 intercept from the rounds before the race;
    - ``relative``: the pooled 2022 intercept + (circuit 2021 − 2021 average).
    """
    import json
    import logging

    from f1strat.data import load_season
    from f1strat.deg_model import Cutoff, cutoff_inputs, thin_fit
    from f1strat.race_events import relative_circuit_intercept
    from f1strat.rehearsal import cutoff_fit, leakage_free_params
    from f1strat.simulator import observed_pit_losses

    logging.disable(logging.WARNING)
    raw, race = load_season(2022), load_season(2022, table="race")
    out = {}
    for name in names:
        call = CALLS[name]()
        rnd = call.round_number
        cleaned, laps = cutoff_inputs(raw, Cutoff(rnd, call.cutoff))
        fitted = thin_fit(cutoff_fit(raw, rnd, call.cutoff))
        state, _ = decision_state(raw, race, call, cleaned)
        losses = observed_pit_losses(cleaned, rnd)
        params, info = leakage_free_params(fitted, raw, race, cleaned, rnd)
        event = str(race.loc[race["RoundNumber"] == rnd, "EventName"].iloc[0])
        rel = relative_circuit_intercept(event, params.passing.intercept, 2021)
        settings = {
            "committed": params.passing.circuit[rnd],
            "average": params.passing.intercept,
            "relative": rel["intercept"],
        }
        out[name] = {"relative_detail": rel, "settings": {}}
        for label_, value in settings.items():
            circ = {**params.passing.circuit, rnd: value}
            prm = replace(params, passing=replace(params.passing, circuit=circ))
            v = verdict(fitted, state, losses, prm, call, tyre_age_cap(laps), CHECKS[0])
            out[name]["settings"][label_] = {
                "intercept": value,
                "band": v["band"],
                "primary": v["primary"],
                "chosen": v["chosen"],
                "measure": {
                    n: {k: m[k] for k in ("mean_points", "mean_pos", "p_podium")}
                    for n, m in v["measure"].items()
                },
            }
            print(
                name,
                label_,
                round(value, 3),
                v["band"],
                round(v["primary"]["mean_loss"], 2),
                round(v["primary"]["p_level_or_ahead"], 3),
                v["chosen"]["best overall"],
                flush=True,
            )
    (FIG_DIR / "case_hungary_pass_sensitivity.json").write_text(
        json.dumps(out, indent=1, default=str)
    )
    return out


# --- Silverstone runner -----------------------------------------------------------------------


def restart_index(status: np.ndarray) -> np.ndarray:
    """Per simulation, the index of the first green lap after the SC the
    state starts under (the restart lap)."""
    neutral = status != 0
    first_green = np.argmax(~neutral, axis=1)
    return np.where(neutral.all(axis=1), status.shape[1], first_green)


def silverstone_reality(
    dec_fit: BayesFit,
    full_fit: BayesFit,
    raw: pd.DataFrame,
    race: pd.DataFrame,
    cleaned_full: pd.DataFrame,
    call: Call,
    state: RaceState,
    params: SimParams,
    cap: dict[str, int],
    plan: Plan,
    losses: np.ndarray,
    sim_params_h: SimParams,
    n_sims: int = 1000,
    first_laps: int = 3,
) -> dict:
    """Leclerc's laps after the restart against the forecast, the lap he lost
    each position, and passes in the first ``first_laps`` laps after the
    restart (whole field), simulated vs real.

    The forecast restarts on a drawn lap, so position losses and restart
    passes are counted from each simulation's own restart; lap times are
    compared lap by lap on green laps (medians), as for Hungary.
    """
    from f1strat.backtest import actual_race, count_passes
    from f1strat.simulator import simulate_field

    focal, rnd, cut = call.focal, call.round_number, call.cutoff
    act = actual_race(raw, race, rnd)
    real_restart = int(next(L for L in range(cut + 1, act.total_laps + 1) if act.status[L] == 0))
    laps_idx = np.arange(cut + 1, act.total_laps + 1)
    me = raw[(raw["RoundNumber"] == rnd) & (raw["Driver"] == focal)].set_index("LapNumber")
    real_lap = me["LapTime"].dt.total_seconds().reindex(laps_idx)
    real_pos = me["Position"].reindex(laps_idx).astype(float)
    pos_tab = raw[(raw["RoundNumber"] == rnd) & raw["LapNumber"].isin(laps_idx)].pivot(
        index="LapNumber", columns="Driver", values="Position"
    )
    lost = []
    for L in laps_idx[1:]:
        if real_pos[L] > real_pos[L - 1]:
            by = [
                d
                for d in pos_tab.columns
                if pos_tab.at[L - 1, d] > real_pos[L - 1] and pos_tab.at[L, d] < real_pos[L]
            ]
            lost.append(
                {
                    "lap": int(L),
                    "laps_after_restart": int(L - real_restart + 1),
                    "from": int(real_pos[L - 1]),
                    "to": int(real_pos[L]),
                    "by": by,
                }
            )
    win = range(real_restart - 1, real_restart + first_laps)
    real_passes = int(
        count_passes(
            act.times[None, list(win), :],
            act.pit_laps[None, list(win), :],
            act.status[list(win)] == 0,
        )[0]
    )
    out = {
        "laps": laps_idx.tolist(),
        "real_restart_lap": real_restart,
        "real_lap_s": real_lap.round(3).tolist(),
        "real_pos": real_pos.tolist(),
        "real_positions_lost": lost,
        f"real_passes_first_{first_laps}": real_passes,
    }

    dec = make_field(dec_fit, state, losses, params, CHECKS[0], n_sims, seed=202)
    dec_rivals = rivals_for(dec, call, cap)
    hfld, h_rivals, retire = hindsight_field(
        full_fit, raw, race, cleaned_full, call, state, n_sims, sim_params=sim_params_h
    )
    for name, fld, rivals, ret in (
        ("forecast", dec, dec_rivals, {}),
        ("hindsight", hfld, h_rivals, retire),
    ):
        res = simulate_field(fld, {**rivals, focal: plan}, retire=ret, record=True)
        c = res.drivers.index(focal)
        n = fld.n_sims
        start = np.array([state.cars[d].race_time for d in res.drivers])
        hist = np.concatenate([np.broadcast_to(start, (n, 1, len(start))), res.history], axis=1)
        pits = np.concatenate([np.zeros((n, 1, len(start)), bool), res.pit_laps], axis=1)
        green = np.asarray(fld.status == 0)
        lap_t = np.where(green, np.diff(hist[:, :, c], axis=1), np.nan)
        pos = np.argsort(np.argsort(hist[:, 1:, :], axis=2), axis=2)[:, :, c] + 1
        r0 = restart_index(fld.status)  # restart lap index
        # Position at the restart's start (end of the last SC lap) and after.
        rows = np.arange(n)
        pos_at_restart = pos[rows, np.maximum(r0 - 1, 0)]
        lost_lap = {}
        for k in range(1, 4):  # lost 1st, 2nd, 3rd place
            target = pos_at_restart + k
            hit = np.full(n, -1)
            for j in range(pos.shape[1]):
                newly = (hit < 0) & (j >= r0) & (pos[:, j] >= target)
                hit = np.where(newly, j - r0 + 1, hit)
            lost_lap[k] = {
                "p_lost": float((hit > 0).mean()),
                "median_laps_after_restart": (
                    float(np.median(hit[hit > 0])) if (hit > 0).any() else None
                ),
            }
        g = np.zeros((n, hist.shape[1]), bool)
        for i in range(n):
            g[i, r0[i] + 1 : r0[i] + 1 + first_laps] = True
        g &= np.concatenate([np.zeros((n, 1), bool), green], axis=1)
        passes = count_passes(hist, pits, g)
        out[name] = {
            "lap_median": np.nanmedian(lap_t, axis=0).round(3).tolist(),
            "lap_p10": np.nanquantile(lap_t, 0.1, axis=0).round(3).tolist(),
            "lap_p90": np.nanquantile(lap_t, 0.9, axis=0).round(3).tolist(),
            "share_green": green.mean(axis=0).round(3).tolist(),
            "pos_mean": pos.mean(axis=0).round(3).tolist(),
            "pos_p10": np.quantile(pos, 0.1, axis=0).tolist(),
            "pos_p90": np.quantile(pos, 0.9, axis=0).tolist(),
            "restart_lap_dist": {int(cut + 1 + k): float((r0 == k).mean()) for k in np.unique(r0)},
            "pos_at_restart_mean": float(pos_at_restart.mean()),
            "positions_lost_after_restart": lost_lap,
            f"passes_first_{first_laps}_mean": float(passes.mean()),
            f"passes_first_{first_laps}_p10_p90": [
                float(np.quantile(passes, 0.1)),
                float(np.quantile(passes, 0.9)),
            ],
            "final_pos_dist": {
                int(v): float((pos[:, -1] == v).mean()) for v in np.unique(pos[:, -1])
            },
        }
    return out


def plot_reality_silverstone(rc: dict, path: Path) -> Path:
    """Leclerc's lap times and positions after the restart: real vs simulated."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.array(rc["laps"])
    keep = x >= rc["real_restart_lap"]
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, facecolor=SURFACE)
    names = {"forecast": "lap-38 forecast", "hindsight": "hindsight replay"}
    for ax, key, real, ylab in (
        (axes[0], "lap", rc["real_lap_s"], "Leclerc lap time (s)"),
        (axes[1], "pos", rc["real_pos"], "Leclerc position"),
    ):
        _style(ax)
        for name, color in (("forecast", BLUE), ("hindsight", ORANGE)):
            d = rc[name]
            mid = d["lap_median"] if key == "lap" else d["pos_mean"]
            ax.fill_between(
                x[keep],
                np.array(d[f"{key}_p10"])[keep],
                np.array(d[f"{key}_p90"])[keep],
                color=color,
                alpha=0.15,
                lw=0,
            )
            ax.plot(
                x[keep],
                np.array(mid)[keep],
                color=color,
                lw=2,
                label=f"{names[name]} ({'median' if key == 'lap' else 'mean'}; 10–90%)",
            )
        ax.plot(x[keep], np.array(real)[keep], "o-", color=INK, lw=1.2, ms=4, label="real")
        ax.set_ylabel(ylab, color=INK2, fontsize=9)
    axes[1].invert_yaxis()
    axes[0].set_title(
        f"Leclerc on old hards after the restart (lap {rc['real_restart_lap']}): real vs simulated",
        fontsize=10,
        color=INK,
        loc="left",
    )
    axes[0].legend(fontsize=8, frameon=False)
    axes[1].set_xlabel("lap", color=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_joint(joint: dict, path: Path, hind: dict | None = None) -> Path:
    """Ferrari's combined points per joint lap-39 option (measurement set)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    opts = joint["options"]
    names = sorted(opts, key=lambda n: opts[n]["combined"])
    fig, ax = plt.subplots(figsize=(9.5, 0.7 * len(names) + 1.8), facecolor=SURFACE)
    _style(ax)
    for y, n in enumerate(names):
        o = opts[n]
        ax.barh(y, o["focal_points"], color=BLUE, height=0.6, label="Leclerc" if y == 0 else None)
        ax.barh(
            y,
            o["teammate_points"],
            left=o["focal_points"],
            color=ORANGE,
            height=0.6,
            label="Sainz" if y == 0 else None,
        )
        txt = f"{o['combined']:.1f}"
        if hind and n in hind:
            txt += f"  (hindsight {hind[n]['combined']:.1f})"
        ax.text(o["combined"] + 0.3, y, txt, va="center", fontsize=9, color=INK)
    ax.set_yticks(
        range(len(names)),
        [f"{n}\nLEC {opts[n]['focal_plan']} | SAI {opts[n]['teammate_plan']}" for n in names],
        fontsize=8,
        color=INK,
    )
    ax.set_xlabel("Ferrari's mean points (Leclerc + Sainz), 2,000 simulations", color=INK2)
    ax.set_xlim(0, max(opts[n]["combined"] for n in names) * 1.3)
    ax.set_title(
        "Ferrari's combined points per joint lap-39 option (Silverstone, end of lap 38)",
        fontsize=10,
        color=INK,
        loc="left",
    )
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


#: Exploratory (Silverstone): s/lap added to Ferrari's hard pace.
SILVERSTONE_BREAK_EVEN: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 1.5)


def silverstone_inputs() -> tuple:
    """(fitted, state, losses, params, call, cap) at Silverstone's cutoff, as in
    :func:`run_silverstone`."""
    from f1strat.data import load_season
    from f1strat.deg_model import Cutoff, cutoff_inputs, thin_fit
    from f1strat.race_events import double_stack_loss, double_stack_pairs
    from f1strat.rehearsal import cutoff_fit, leakage_free_params
    from f1strat.simulator import observed_pit_losses

    call = silverstone_call()
    rnd = call.round_number
    raw, race = load_season(2022), load_season(2022, table="race")
    cleaned, laps = cutoff_inputs(raw, Cutoff(rnd, call.cutoff))
    fitted = thin_fit(cutoff_fit(raw, rnd, call.cutoff))
    params, _ = leakage_free_params(fitted, raw, race, cleaned, rnd, **call.param_kwargs)
    params = replace(
        params,
        double_stack={k: v["D_s"] for k, v in double_stack_loss(double_stack_pairs(raw)).items()},
    )
    state, _ = decision_state(raw, race, call, cleaned, outlap_scale=params.neutral.sc_pit_ratio)
    return fitted, state, observed_pit_losses(cleaned, rnd), params, call, tyre_age_cap(laps)


def silverstone_break_even(shifts: Sequence[float] = SILVERSTONE_BREAK_EVEN) -> list[dict]:
    """Exploratory, outside the committed rule: the verdict with Ferrari's hard
    pace shifted by each of ``shifts`` (both Ferraris; only Leclerc stays on
    hards after lap 39). Writes ``case_silverstone_break_even.json``."""
    import json

    fitted, state, losses, params, call, cap = silverstone_inputs()
    out = []
    for x in shifts:
        v = verdict(
            fitted,
            state,
            losses,
            params,
            call,
            cap,
            Check(f"Ferrari hard pace +{x:.2f} s/lap", hard_pace_add=x),
        )
        r = {k: v[k] for k in ("check", "band", "primary", "chosen")}
        r["measure"] = {n: m["mean_points"] for n, m in v["measure"].items()}
        out.append(r)
        print(
            x,
            v["band"],
            round(v["primary"]["mean_loss"], 2),
            v["chosen"]["best overall"],
            v["chosen"][call.call_family],
            flush=True,
        )
    (FIG_DIR / "case_silverstone_break_even.json").write_text(
        json.dumps(out, indent=1, default=str)
    )
    return out


def run_silverstone(hindsight_only: bool = False, version: str = "v1") -> dict:
    """``python -m f1strat.case_study silverstone [--v2]``: the verdict under
    every Silverstone check, Ferrari's joint options, the hindsight replay and
    the reality check (``docs/case_studies.md``).

    ``version="v2"`` adds the restart layer (:mod:`f1strat.restart`),
    estimated from the dry rounds before Silverstone with the cutoff fit, to
    the decision simulations and the hindsight replay; outputs are
    ``case_silverstone_v2_*`` (v1 files are kept)."""
    import json
    import time

    from f1strat.clean import clean_laps
    from f1strat.data import load_season
    from f1strat.deg_model import THIN_FIT_PATH, Cutoff, cutoff_inputs, load_fit, thin_fit
    from f1strat.race_events import (
        SimParams,
        double_stack_loss,
        double_stack_pairs,
        neutral_pit_samples,
    )
    from f1strat.rehearsal import cutoff_fit, leakage_free_params
    from f1strat.simulator import actual_plan, observed_pit_losses, simulate_field

    call = silverstone_call()
    rnd = call.round_number
    raw, race = load_season(2022), load_season(2022, table="race")
    cleaned, laps = cutoff_inputs(raw, Cutoff(rnd, call.cutoff))
    full = cutoff_fit(raw, rnd, call.cutoff)
    fitted = thin_fit(full)
    params, pinfo = leakage_free_params(fitted, raw, race, cleaned, rnd, **call.param_kwargs)
    ds = double_stack_loss(double_stack_pairs(raw))
    params = replace(params, double_stack={k: v["D_s"] for k, v in ds.items()})
    pinfo["double_stack"] = ds
    pinfo["sc_lengths"] = params.neutral.sc_lengths
    pinfo["sc_pit_ratio"] = params.neutral.sc_pit_ratio
    pinfo["neutral_pit_samples"] = neutral_pit_samples(cleaned[cleaned["RoundNumber"] < rnd])
    if version == "v2":
        from f1strat.clean import MIXED_CONDITIONS_2022
        from f1strat.restart import estimate as estimate_restart

        dry_before = sorted(
            int(r)
            for r in race.loc[
                (race["RoundNumber"] < rnd) & ~race["EventName"].isin(list(MIXED_CONDITIONS_2022)),
                "RoundNumber",
            ].unique()
        )
        rs = estimate_restart(cleaned, fitted, dry_before)
        params = replace(params, restart=rs)
        pinfo["restart"] = rs.to_dict()
    state, notes = decision_state(
        raw, race, call, cleaned, outlap_scale=params.neutral.sc_pit_ratio
    )
    cap = tyre_age_cap(laps)
    losses = observed_pit_losses(cleaned, rnd)
    stem = f"case_{call.name}" + ("" if version == "v1" else f"_{version}")
    path = FIG_DIR / f"{stem}_results.json"
    full_season = load_fit(THIN_FIT_PATH)
    cleaned_full = clean_laps(raw)
    sim_h = replace(
        SimParams.load(), double_stack=params.double_stack, restart=getattr(params, "restart", None)
    )
    actual = {"LEC": call.as_run, "SAI": actual_plan(raw, rnd, "SAI", call.cutoff)}

    if hindsight_only:
        results = json.loads(path.read_text())
        main_v = results["checks"][0]
        chosen_plans = {n: parse_label(lbl) for n, lbl in main_v["chosen"].items()}
        all_plans = [
            parse_label(lbl) for lbl in pd.read_csv(FIG_DIR / f"{stem}_candidates.csv")["plan"]
        ]
    else:
        results = {
            "notes": notes,
            "cap": cap,
            "n_pit_losses": len(losses),
            "params": pinfo,
            "fit_diagnostics": {k: float(v) for k, v in full.diagnostics.items()},
            "ferrari_hard": ferrari_hard_quantiles(fitted, rnd),
            "checks": [],
        }
        print(
            json.dumps(
                {k: results[k] for k in ("notes", "params", "ferrari_hard")}, indent=1, default=str
            ),
            flush=True,
        )
        tables = {}
        for check in SILVERSTONE_CHECKS:
            t0 = time.time()
            v = verdict(fitted, state, losses, params, call, cap, check)
            tables[check.name] = v.pop("selection_table")
            objs = (v.pop("chosen_plans"), v.pop("all_plans"))
            if check.name == "main":
                chosen_plans, all_plans = objs
            fam = objs[0][call.call_family]
            v["call_family_best_first_stop"] = fam.stops[0].lap if fam.stops else None
            results["checks"].append(v)
            print(
                f"{check.name}: {v['band']}, loss {v['primary']['mean_loss']:.2f}, level/ahead "
                f"{v['primary']['p_level_or_ahead']:.2f}, best {v['chosen']['best overall']}, "
                f"call fam {v['chosen'][call.call_family]} ({time.time() - t0:.0f} s)",
                flush=True,
            )
        main_v = results["checks"][0]
        tables["main"].round(3).to_csv(FIG_DIR / f"{stem}_candidates.csv", index=False)
        plot_points(main_v, FIG_DIR / f"{stem}_points.png", "Silverstone")
        plot_sensitivity(results["checks"], FIG_DIR / f"{stem}_sensitivity.png")
        t0 = time.time()
        joint = joint_options(fitted, state, losses, params, call, cap, actual)
        joint.pop("selection").round(3).to_csv(FIG_DIR / f"{stem}_joint_pairs.csv", index=False)
        results["joint"] = joint
        print("joint:", json.dumps(joint, indent=1), f"({time.time() - t0:.0f} s)", flush=True)
        path.write_text(json.dumps(results, indent=1, default=str))

    hind = hindsight(
        full_season,
        raw,
        race,
        cleaned_full,
        call,
        state,
        chosen_plans,
        all_plans=all_plans,
        sim_params=sim_h,
    )
    results["hindsight"] = hind
    # Joint options in hindsight (rivals on their actual plans, actual SC).
    hfld, h_rivals, retire = hindsight_field(
        full_season, raw, race, cleaned_full, call, state, sim_params=sim_h
    )
    hj = {}
    for n, o in results["joint"]["options"].items():
        pf, pm = parse_label(o["focal_plan"]), parse_label(o["teammate_plan"])
        res = simulate_field(hfld, {**h_rivals, "LEC": pf, "SAI": pm}, retire=retire)
        a = points(res.positions[:, res.drivers.index("LEC")])
        b = points(res.positions[:, res.drivers.index("SAI")])
        hj[n] = {
            "focal_points": float(a.mean()),
            "teammate_points": float(b.mean()),
            "combined": float((a + b).mean()),
        }
    results["joint_hindsight"] = hj
    plot_hindsight(main_v, hind, FIG_DIR / f"{stem}_hindsight.png", "Silverstone")
    plot_joint(results["joint"], FIG_DIR / f"{stem}_joint.png", hj)
    rc = silverstone_reality(
        fitted,
        full_season,
        raw,
        race,
        cleaned_full,
        call,
        state,
        params,
        cap,
        call.as_run,
        losses,
        sim_h,
    )
    results["reality_check"] = rc
    plot_reality_silverstone(rc, FIG_DIR / f"{stem}_reality.png")
    path.write_text(json.dumps(results, indent=1, default=str))
    print(
        json.dumps(
            {
                "hindsight": hind,
                "joint_hindsight": hj,
                "reality": {k: v for k, v in rc.items() if k not in ("forecast", "hindsight")},
            },
            indent=1,
            default=str,
        )
    )
    return results


if __name__ == "__main__":
    main()
