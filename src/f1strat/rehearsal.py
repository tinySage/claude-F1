"""Phase 6 dress rehearsal (Phase 5 Part B): the whole case-study pipeline on
non-case races, with no future information.

For a race cut at a decision lap:

1. **Leakage-free data** (:func:`f1strat.deg_model.cutoff_inputs`): earlier
   rounds plus the decision race up to the cutoff; cleaning (107% outlier
   median, pit losses) recomputed from those laps only; no stint of the
   decision race labelled final.
2. **Refit** the degradation model on them (``fit(laps, cutoff=...)``).
3. **Race state** at the cutoff lap, with each car's remaining sets from its
   reconstructed inventory (practice and qualifying, so known before the race).
4. **Strategy search** for one front-runner, against rivals whose plans come
   from their **own** single-car searches on the same fit
   (:func:`f1strat.simulator.rival_plan_options`), never their actual later
   stops. SC/VSC are drawn (the future is unknown).
5. **Compare** the picks and the simulated outcomes with what happened.

Two data modes:

- **Part B** (``mode="season"``): race laps only; passing and SC/VSC
  parameters from ``sim_params.json`` (the whole 2022 season, a known leak).
- **Part C** (``mode="cutoff"``): adds the decision race's FP1–FP3 long runs
  to the refit (:mod:`f1strat.practice`), and estimates passing and SC/VSC
  parameters from the rounds before the decision race only, with the
  circuit's pass difficulty from its previous running
  (:func:`leakage_free_params`).

Fits used for results must have 0 divergences: :func:`clean_fit` reruns with
a new seed, then a higher target acceptance. See ``docs/validation.md``.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from f1strat.deg_model import MODELS_DIR, BayesFit, Cutoff, cutoff_inputs

FIG_DIR = Path(__file__).resolve().parents[2] / "reports" / "figures"

#: name -> (round, cutoff lap, focal driver). Both are dry races outside the
#: case studies, cut at mid-race.
REHEARSALS: dict[str, tuple[int, int, str]] = {
    "spain": (6, 33, "VER"),
    "canada": (9, 35, "VER"),
}


def fit_path(round_number: int, lap: int, mode: str = "season") -> Path:
    """Where a rehearsal's cutoff fit is saved (gitignored)."""
    tag = "" if mode == "season" else "_practice"
    return MODELS_DIR / f"rehearsal_{round_number:02d}_lap{lap}{tag}.nc"


def refit(raw: pd.DataFrame, round_number: int, lap: int, mode: str = "season",
          **fit_kwargs) -> BayesFit:
    """Fit the degradation model on the laps known at the cutoff and save it.

    ``mode="cutoff"`` adds the decision race's practice long runs."""
    from f1strat.deg_model import fit, save_fit
    from f1strat.practice import long_runs

    cut = Cutoff(round_number, lap)
    _, laps = cutoff_inputs(raw, cut)
    practice = long_runs(raw, round_number) if mode == "cutoff" else None
    start = time.time()
    fitted = fit(laps, cutoff=cut, practice=practice, **fit_kwargs)
    print(f"fit ({mode}) on {len(laps):,} race laps ({laps['RoundNumber'].nunique()} races)"
          f"{'' if practice is None else f' + {len(practice)} practice laps'} in "
          f"{(time.time() - start) / 60:.1f} min; {fit_kwargs}; diagnostics {fitted.diagnostics}",
          flush=True)
    save_fit(fitted, fit_path(round_number, lap, mode))
    return fitted


#: (seed, target_accept) tried in turn until a fit has no divergences.
CLEAN_FIT_ATTEMPTS: tuple[tuple[int, float], ...] = ((0, 0.95), (1, 0.95), (2, 0.99), (3, 0.99))


def clean_fit(raw: pd.DataFrame, round_number: int, lap: int, mode: str) -> BayesFit:
    """A cutoff fit with 0 divergences (rule for fits used in results)."""
    for seed, ta in CLEAN_FIT_ATTEMPTS:
        fitted = refit(raw, round_number, lap, mode, seed=seed, target_accept=ta)
        if fitted.diagnostics.get("divergences", 0) == 0:
            return fitted
    raise RuntimeError(f"no divergence-free fit for round {round_number} lap {lap}")


def leakage_free_params(fitted: BayesFit, raw: pd.DataFrame, race: pd.DataFrame,
                        cleaned: pd.DataFrame, round_number: int, n_sims: int = 200):
    """Passing and SC/VSC parameters known before the decision race.

    - Pass model (slope, pooled intercept, held gaps) from the pass events of
      the dry rounds before it, with the cutoff fit's pace.
    - The decision circuit's intercept from its previous running (e.g. Spain
      2021), with a stint-line pace proxy, shrunk toward the pooled
      intercept. If no previous running can be downloaded (Canada: 2020 and
      2021 cancelled, 2019 timing needs an F1 account), the pooled intercept.
    - SC/VSC rates, lengths and ratios from the rounds before it.
    - SC closing ratio and pass shift fitted on the earlier backtest-eligible
      rounds (replay), as in :mod:`f1strat.backtest`.

    Returns:
        ``(params, info)``: the :class:`SimParams` and a dict for the report.
    """
    from dataclasses import replace

    from f1strat.backtest import (BACKTEST_ROUNDS, calibrate_shift, fit_sc_close_ratio,
                                  sc_gap_observations, with_passing)
    from f1strat.clean import MIXED_CONDITIONS_2022
    from f1strat.deg_model import lap_features
    from f1strat.race_events import (SimParams, circuit_intercept_from, fit_neutralisations,
                                     fit_pass_model, pair_events, pass_events, previous_race,
                                     stint_line_pace)

    before = cleaned[cleaned["RoundNumber"] < round_number]
    race_before = race[race["RoundNumber"] < round_number]
    passing = fit_pass_model(pass_events(before, fitted))
    event = str(race.loc[race["RoundNumber"] == round_number, "EventName"].iloc[0])
    try:
        year, prev = previous_race(2022, event)
    except ValueError as err:
        # No downloadable previous running (e.g. Canada: 2020-21 cancelled, 2019
        # needs an F1 account): fall back to the pooled intercept.
        year, prev_events, why = None, pd.DataFrame(columns=["passed"]), str(err)
        passing.circuit[round_number] = passing.intercept
    else:
        feats = lap_features(prev[~prev["EventName"].isin(list(MIXED_CONDITIONS_2022))])
        feats = feats[feats["Compound"].isin(["SOFT", "MEDIUM", "HARD"])].copy()
        feats["pace"] = stint_line_pace(prev).loc[feats.index]
        prev_events = pair_events(feats)
        passing.circuit[round_number] = circuit_intercept_from(prev_events, passing.slope,
                                                               passing.intercept)
        why = None
    neutral = fit_neutralisations(before, race_before)
    params = SimParams(passing, neutral)
    replay_rounds = [r for r in BACKTEST_ROUNDS if r < round_number]
    ratio = fit_sc_close_ratio(sc_gap_observations(raw, race, replay_rounds), params.sc_gap_s)
    params = replace(params, neutral=replace(params.neutral, sc_close_ratio=ratio))
    shift = calibrate_shift(fitted, raw, race, cleaned, params, rounds=replay_rounds,
                            n_sims=n_sims)
    params = with_passing(params, round(shift, 3))
    dry_before = race_before[~race_before["EventName"].isin(list(MIXED_CONDITIONS_2022))]
    info = {"dry_rounds_before": sorted(int(r) for r in dry_before["RoundNumber"].unique()),
            "replay_rounds": replay_rounds, "previous_running": year,
            "previous_running_note": why,
            "previous_events": len(prev_events),
            "previous_passes": int(prev_events["passed"].sum()) if len(prev_events) else 0,
            "circuit_intercept": round(passing.circuit[round_number], 3),
            "pooled_intercept": round(passing.intercept, 3), "slope": round(passing.slope, 3),
            "pass_events_before": passing.n_events, "shift": round(shift, 3),
            "sc_close_ratio": ratio, "p_sc_per_lap": round(neutral.p_sc_per_lap, 4),
            "p_vsc_per_lap": round(neutral.p_vsc_per_lap, 4)}
    return params, info


@dataclass
class Rehearsal:
    """A cutoff rehearsal's search and how it compares with the race."""

    name: str
    round_number: int
    lap: int
    focal: str
    table: pd.DataFrame          # focal plans, ranked, vs the actual plan
    field: pd.DataFrame          # every car under the focal's actual plan vs actual result
    rivals: pd.DataFrame         # each rival's plan options vs its actual later stops
    actual_label: str
    actual_finish: int | None    # focal's actual finishing position (None if retired)
    sensitivity: pd.DataFrame    # same plans with widening x1.188
    notes: list[str]


def rehearse(
    name: str,
    fitted: BayesFit,
    raw: pd.DataFrame,
    race: pd.DataFrame,
    params=None,
    mode: str = "season",
    n_sims: int = 1000,
    seed: int = 0,
    widening: float = 1.0,
    sensitivity_widening: float = 1.188,
) -> Rehearsal:
    """Run one rehearsal from a cutoff fit (see the module docstring).

    ``params`` defaults to ``sim_params.json`` (whole season)."""
    from f1strat.backtest import actual_race
    from f1strat.race_events import SimParams
    from f1strat.simulator import (
        RaceState,
        actual_plan,
        build_field,
        field_search,
        observed_pit_losses,
        rival_plan_options,
        search,
    )
    from f1strat.tyres import reconstruct_inventories, sets_left

    rnd, lap, focal = REHEARSALS[name]
    cleaned, _ = cutoff_inputs(raw, Cutoff(rnd, lap))
    total = int(race.loc[race["RoundNumber"] == rnd, "TotalLaps"].iloc[0])
    state = RaceState.from_laps(raw, rnd, lap, total)
    invs = reconstruct_inventories(2022, rnd)
    state = state.with_sets({d: sets_left(invs[d], raw, rnd, lap) for d in state.cars if d in invs})
    losses = observed_pit_losses(cleaned, rnd)
    params = params if params is not None else SimParams.load()
    notes = [f"{len(losses)} green-flag stops known at lap {lap} (pit loss median "
             f"{np.median(losses):.1f} s)",
             f"{focal} sets left: {state.sets.get(focal)}"]

    def run(w: float, plans=None):
        fld = build_field(fitted, state, losses, params, n_sims=n_sims, seed=seed,
                          widening=w, use_fitted_current_stint=True)
        rivals = {d: rival_plan_options(fld, d, min_stint=5) for d in fld.drivers if d != focal}
        if plans is None:
            plans = search(fld.models[focal], state, focal, fld.draws[focal], max_stops=3,
                           min_stint=5, include=[actual]).plans
        return fld, rivals, field_search(fld, focal, plans, rivals)

    actual = actual_plan(raw, rnd, focal, lap)
    fld, rivals, res = run(widening)
    car = state.cars[focal]
    table = res.table(reference=actual)
    table["actual"] = table["plan"] == actual.label(car.compound)

    act = actual_race(raw, race, rnd)
    finish = {d: i + 1 for i, d in enumerate(act.order)}
    # Forecast check: simulated positions re-ranked among the cars that
    # actually finished (retirements after the cutoff are unknowable then).
    k = res.plans.index(actual)
    fin = [c for c, d in enumerate(res.drivers) if d in finish]
    t = res.times[k][:, fin]
    pos = np.argsort(np.argsort(t, axis=1), axis=1) + 1
    rows = []
    for j, c in enumerate(fin):
        d = res.drivers[c]
        lo, hi = np.quantile(pos[:, j], [0.1, 0.9])
        rows.append({"driver": d, "pos_at_cutoff": state.cars[d].position,
                     "sim_mean": pos[:, j].mean(), "p10": lo, "p90": hi,
                     "actual": finish[d], "in80": bool(lo <= finish[d] <= hi)})
    field_tab = pd.DataFrame(rows).sort_values("sim_mean").reset_index(drop=True)
    retired = [d for d in res.drivers if d not in finish]
    if retired:
        notes.append(f"retired after lap {lap} (left out of the field comparison): "
                     + ", ".join(retired))

    riv = []
    for d, opts in rivals.items():
        real = actual_plan(raw, rnd, d, lap).label(state.cars[d].compound)
        labels = [p.label(state.cars[d].compound) for p, _ in opts]
        w = np.array([wt for _, wt in opts])
        riv.append({"driver": d, "options (weight)": "; ".join(
            f"{lbl} ({wi / w.sum():.2f})" for lbl, wi in zip(labels, w)),
            "actual": real, "actual_stops_after_cutoff":
            len(actual_plan(raw, rnd, d, lap).stops),
            "option_stops": "/".join(str(len(p.stops)) for p, _ in opts)})
    riv = pd.DataFrame(riv)

    # Sensitivity: the top plans and the actual one with widening x1.188.
    top = list(dict.fromkeys([res.plans[res.labels.index(lbl)] for lbl in table["plan"].head(8)]
                             + [actual]))
    _, _, res_w = run(sensitivity_widening, plans=top)
    sens = res_w.table(reference=actual)

    return Rehearsal(name if mode == "season" else f"{name}_cutoff", rnd, lap, focal,
                     table, field_tab, riv,
                     actual.label(car.compound), finish.get(focal), sens, notes)


def report(r: Rehearsal) -> str:
    """Markdown summary, also written to ``reports/figures/rehearsal_<name>.md``."""
    cols = ["plan", "stops", "mean_pos", "p_win", "p_podium", "p10_pos", "p90_pos",
            "mean_time_s", "places_gained", "p_ahead_of_ref", "p_behind_ref"]
    t = r.table
    shown = pd.concat([t.groupby("stops").head(3), t[t["actual"]]]).drop_duplicates("plan")
    shown = shown.sort_values(["mean_pos", "mean_time_s"])
    out = [f"# Rehearsal: {r.name.title()} (round {r.round_number}), cut after lap {r.lap}, "
           f"focal {r.focal}", ""]
    out += [f"- {n}" for n in r.notes]
    out += [f"- {r.focal}'s actual plan from lap {r.lap}: {r.actual_label}; finished "
            f"{r.actual_finish if r.actual_finish else 'DNF'}", "",
            f"## {r.focal}'s plans (widening 1.0; best 3 per stop count, plus actual)", "",
            shown[cols + ["actual"]].round(3).to_markdown(index=False), "",
            "## Sensitivity: widening x1.188 (same draws, top plans and actual)", "",
            r.sensitivity[cols[:-3] + ["places_gained"]].round(3).to_markdown(index=False), "",
            f"## Field under {r.focal}'s actual plan vs the actual result", "",
            "Simulated positions re-ranked among the cars that finished; "
            f"actual inside the 10th–90th percentile for "
            f"{r.field['in80'].mean():.0%} of them.", "",
            r.field.round(2).to_markdown(index=False), "",
            "## Rival plans (own searches) vs their actual stops", "",
            r.rivals.to_markdown(index=False), ""]
    text = "\n".join(out)
    (FIG_DIR / f"rehearsal_{r.name}.md").write_text(text)
    return text


def main(argv: list[str] | None = None) -> None:
    """``python -m f1strat.rehearsal <name> [--cutoff-data] [--refit]``.

    Without ``--cutoff-data``: Part B (race laps only, season parameters).
    With it: Part C (practice long runs, leakage-free parameters, and a
    divergence-free fit). Refits if the fit is not saved yet or ``--refit``.
    """
    import json
    import logging

    from f1strat.data import load_season
    from f1strat.deg_model import load_fit, thin_fit

    logging.disable(logging.WARNING)
    args = argv if argv is not None else sys.argv[1:]
    name = args[0]
    mode = "cutoff" if "--cutoff-data" in args else "season"
    rnd, lap, _ = REHEARSALS[name]
    raw = load_season(2022)
    race = load_season(2022, table="race")
    path = fit_path(rnd, lap, mode)
    if "--refit" in args or not path.exists():
        fitted = clean_fit(raw, rnd, lap, mode) if mode == "cutoff" else refit(raw, rnd, lap)
    else:
        fitted = load_fit(path)
    print("diagnostics:", fitted.diagnostics, flush=True)
    if mode == "cutoff" and fitted.diagnostics.get("divergences", 0) > 0:
        raise RuntimeError("cutoff fit has divergences; rerun with --refit")
    thin = thin_fit(fitted)
    params = None
    if mode == "cutoff":
        cleaned, _ = cutoff_inputs(raw, Cutoff(rnd, lap))
        params, info = leakage_free_params(thin, raw, race, cleaned, rnd)
        info["fit_diagnostics"] = {k: float(v) for k, v in fitted.diagnostics.items()}
        (FIG_DIR / f"rehearsal_{name}_cutoff_params.json").write_text(json.dumps(info, indent=1))
        print(json.dumps(info, indent=1), flush=True)
    r = rehearse(name, thin, raw, race, params=params, mode=mode)
    print(report(r))


if __name__ == "__main__":
    main()
