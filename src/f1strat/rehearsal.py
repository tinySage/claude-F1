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

Still from the whole 2022 season (not cut): the passing and SC/VSC parameters
in ``sim_params.json``. See ``docs/validation.md``.
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


def fit_path(round_number: int, lap: int) -> Path:
    """Where a rehearsal's cutoff fit is saved (gitignored)."""
    return MODELS_DIR / f"rehearsal_{round_number:02d}_lap{lap}.nc"


def refit(raw: pd.DataFrame, round_number: int, lap: int, **fit_kwargs) -> BayesFit:
    """Fit the degradation model on the laps known at the cutoff and save it."""
    from f1strat.deg_model import fit, save_fit

    cut = Cutoff(round_number, lap)
    _, laps = cutoff_inputs(raw, cut)
    start = time.time()
    fitted = fit(laps, cutoff=cut, **fit_kwargs)
    print(f"fit on {len(laps):,} laps ({laps['RoundNumber'].nunique()} races) "
          f"in {(time.time() - start) / 60:.1f} min; diagnostics {fitted.diagnostics}")
    save_fit(fitted, fit_path(round_number, lap))
    return fitted


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
    n_sims: int = 1000,
    seed: int = 0,
    widening: float = 1.0,
    sensitivity_widening: float = 1.188,
) -> Rehearsal:
    """Run one rehearsal from a cutoff fit (see the module docstring)."""
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
    params = SimParams.load()
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

    return Rehearsal(name, rnd, lap, focal, table, field_tab, riv,
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
    """``python -m f1strat.rehearsal <name> [--refit]``: run one rehearsal,
    refitting the cutoff model if it is not saved yet (or with ``--refit``)."""
    import logging

    from f1strat.data import load_season
    from f1strat.deg_model import load_fit, thin_fit

    logging.disable(logging.WARNING)
    args = argv if argv is not None else sys.argv[1:]
    name = args[0]
    rnd, lap, _ = REHEARSALS[name]
    raw = load_season(2022)
    race = load_season(2022, table="race")
    path = fit_path(rnd, lap)
    fitted = refit(raw, rnd, lap) if ("--refit" in args or not path.exists()) else load_fit(path)
    print("diagnostics:", fitted.diagnostics)
    r = rehearse(name, thin_fit(fitted), raw, race)
    print(report(r))


if __name__ == "__main__":
    main()
