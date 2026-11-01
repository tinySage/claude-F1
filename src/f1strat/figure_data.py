"""The exact data behind every figure, as small tidy tables in ``data/figures/``.

Each figure in ``reports/figures/`` (docs and notebooks) and ``assets/readme/``
(README) is drawn only from its table here, so ``pitwall figures`` redraws all
of them in minutes from committed files: no fits, no simulations, no FastF1.

Two halves:

- **Extract** (:data:`EXTRACTORS`, ``pitwall figures --export``): build each
  table from the heavy sources (case-study result JSON, processed laps, the
  committed thinned fits, the backtest tables) and write it. Run after any
  step that changes results. Tables from the lap model always use the
  committed thinned fits (:data:`THIN_ONLY`), even where a full posterior is
  saved locally, so a fresh clone reproduces them exactly.
- **Load**: read a table back, and for figures whose plot function takes a
  result dict, rebuild exactly the fields it uses (:func:`points_main`,
  :func:`hindsight_dicts`, ...). Floats are written at full precision, so the
  redrawn figure matches the one drawn from the original result.

Table names follow the figure they feed (``case_hungary_points.csv`` →
``reports/figures/case_hungary_points.png``). Large lap tables are parquet.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "figures"
RESULTS_DIR = ROOT / "reports" / "figures"

#: Case-study result stems with the full figure set, and their race names.
CASE_STEMS: dict[str, str] = {
    "hungary": "Hungary",
    "hungary54": "Hungary",
    "silverstone": "Silverstone",
    "silverstone_v2": "Silverstone",
}


def path(name: str) -> Path:
    """Where table ``name`` lives (``.csv`` unless it is a known parquet)."""
    return DATA_DIR / (name if name.endswith((".csv", ".parquet")) else f"{name}.csv")


def read(name: str) -> pd.DataFrame:
    """Read a figure table."""
    p = path(name)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p, float_precision="round_trip")


def _write(df: pd.DataFrame, name: str) -> Path:
    p = path(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix == ".parquet":
        df.to_parquet(p, index=False)
    else:
        df.to_csv(p, index=False)
    return p


def _results(stem: str) -> dict:
    return json.loads((RESULTS_DIR / f"case_{stem}_results.json").read_text())


# --- Case studies: points, sensitivity, hindsight, reality, joint ---------------------


def points_table(stem: str) -> pd.DataFrame:
    """Leclerc's points distribution per distinct plan (measurement set).

    One row per (plan row, points value); ``row`` keeps the order in which the
    named plans were chosen, ``label`` joins the names that share the plan."""
    from f1strat.case_study import _merged

    main = _results(stem)["checks"][0]
    rows = []
    for i, (joined, plan) in enumerate(_merged(main)):
        m = main["measure"][joined.split(" = ")[0]]
        for pts, share in m["points_dist"].items():
            rows.append(
                {
                    "row": i,
                    "label": joined,
                    "plan": plan,
                    "mean_points": m["mean_points"],
                    "points": int(pts),
                    "share": share,
                    "race": CASE_STEMS[stem],
                    "cutoff": main.get("cutoff", 38),
                }
            )
    return pd.DataFrame(rows)


def points_main(df: pd.DataFrame) -> tuple[dict, str]:
    """The ``main`` dict :func:`case_study.plot_points` needs, and the race name."""
    chosen, measure = {}, {}
    for _, g in df.groupby("row", sort=True):
        r = g.iloc[0]
        names = r["label"].split(" = ")
        for n in names:
            chosen[n] = r["plan"]
        measure[names[0]] = {
            "mean_points": float(r["mean_points"]),
            "points_dist": {str(int(p)): float(s) for p, s in zip(g["points"], g["share"])},
        }
    return {"chosen": chosen, "measure": measure, "cutoff": int(df["cutoff"].iloc[0])}, str(
        df["race"].iloc[0]
    )


def sensitivity_table(stem: str) -> pd.DataFrame:
    """Primary loss under every robustness check."""
    checks = _results(stem)["checks"]
    return pd.DataFrame(
        [
            {
                "check": c["check"],
                "band": c["band"],
                "mean_loss": c["primary"]["mean_loss"],
                "p10": c["primary"]["p10"],
                "p90": c["primary"]["p90"],
                "call_label": checks[0].get("call_label", "hards at 39"),
            }
            for c in checks
        ]
    )


def sensitivity_checks(df: pd.DataFrame) -> list[dict]:
    """The ``checks`` list :func:`case_study.plot_sensitivity` needs."""
    return [
        {
            "check": r.check,
            "band": r.band,
            "call_label": r.call_label,
            "primary": {"mean_loss": float(r.mean_loss), "p10": float(r.p10), "p90": float(r.p90)},
        }
        for r in df.itertuples()
    ]


def hindsight_table(stem: str) -> pd.DataFrame:
    """Mean points per plan at the decision lap and in the hindsight replay."""
    from f1strat.case_study import _merged

    r = _results(stem)
    main, hind = r["checks"][0], r["hindsight"]
    act = hind["actual result"]
    base = {
        "race": CASE_STEMS[stem],
        "cutoff": main.get("cutoff", 38),
        "actual_position": act["position"],
        "actual_points": act["points"],
    }
    rows = [
        {
            "kind": "plan",
            "label": j,
            "plan": p,
            "decision_points": main["measure"][j.split(" = ")[0]]["mean_points"],
            "hindsight_points": hind[j.split(" = ")[0]]["mean_points"],
            **base,
        }
        for j, p in _merged(main, keep=lambda n: n in hind)
    ]
    if "best in hindsight" in hind:
        b = hind["best in hindsight"]
        rows.append(
            {
                "kind": "best in hindsight",
                "label": "best in hindsight",
                "plan": b["plan"],
                "decision_points": np.nan,
                "hindsight_points": b["mean_points"],
                **base,
            }
        )
    return pd.DataFrame(rows)


def hindsight_dicts(df: pd.DataFrame) -> tuple[dict, dict, str]:
    """``(main, hind, race)`` for :func:`case_study.plot_hindsight`."""
    chosen, measure, hind = {}, {}, {}
    for r in df[df["kind"] == "plan"].itertuples():
        names = r.label.split(" = ")
        for n in names:
            chosen[n] = r.plan
        measure[names[0]] = {"mean_points": float(r.decision_points)}
        for n in names:
            hind[n] = {"mean_points": float(r.hindsight_points)}
    best = df[df["kind"] == "best in hindsight"]
    if len(best):
        hind["best in hindsight"] = {
            "plan": best["plan"].iloc[0],
            "mean_points": float(best["hindsight_points"].iloc[0]),
        }
    first = df.iloc[0]
    hind["actual result"] = {
        "position": int(first["actual_position"]),
        "points": int(first["actual_points"]),
    }
    main = {"chosen": chosen, "measure": measure, "cutoff": int(first["cutoff"])}
    return main, hind, str(first["race"])


#: Per-lap series of the reality checks: (result key, column prefix).
_REALITY_SERIES = ("forecast", "hindsight")


def reality_table(stem: str) -> pd.DataFrame:
    """The reality check per lap: real laps against forecast and hindsight."""
    rc = _results(stem)["reality_check"]
    df = pd.DataFrame({"lap": rc["laps"]})
    for k, v in rc.items():
        if isinstance(v, list) and len(v) == len(df) and k != "laps":
            df[k] = v
    for name in _REALITY_SERIES:
        for k, v in rc[name].items():
            if isinstance(v, list) and len(v) == len(df):
                df[f"{name}__{k}"] = v
    if "real_restart_lap" in rc:
        df["real_restart_lap"] = rc["real_restart_lap"]
    return df


def reality_dict(df: pd.DataFrame) -> dict:
    """The reality-check dict the reality plots need."""
    rc: dict = {"laps": df["lap"].astype(int).tolist()}
    for name in _REALITY_SERIES:
        rc[name] = {}
    for col in df.columns:
        if col in ("lap", "real_restart_lap"):
            continue
        vals = [float(x) for x in df[col]]
        if "__" in col:
            name, k = col.split("__", 1)
            rc[name][k] = vals
        else:
            rc[col] = vals
    if "real_restart_lap" in df:
        rc["real_restart_lap"] = int(df["real_restart_lap"].iloc[0])
    return rc


def joint_table(stem: str) -> pd.DataFrame:
    """Ferrari's combined points per joint option (decision lap and hindsight)."""
    r = _results(stem)
    hj = r.get("joint_hindsight", {})
    return pd.DataFrame(
        [
            {
                "option": n,
                "focal_plan": o["focal_plan"],
                "teammate_plan": o["teammate_plan"],
                "focal_points": o["focal_points"],
                "teammate_points": o["teammate_points"],
                "combined": o["combined"],
                "hindsight_combined": hj.get(n, {}).get("combined", np.nan),
            }
            for n, o in r["joint"]["options"].items()
        ]
    )


def joint_dicts(df: pd.DataFrame) -> tuple[dict, dict]:
    """``(joint, hind)`` for :func:`case_study.plot_joint`."""
    opts = {
        r.option: {
            "focal_plan": r.focal_plan,
            "teammate_plan": r.teammate_plan,
            "focal_points": float(r.focal_points),
            "teammate_points": float(r.teammate_points),
            "combined": float(r.combined),
        }
        for r in df.itertuples()
    }
    hind = {
        r.option: {"combined": float(r.hindsight_combined)}
        for r in df.itertuples()
        if not pd.isna(r.hindsight_combined)
    }
    return {"options": opts}, hind


# --- Monaco ---------------------------------------------------------------------------


def monaco_gaps_table() -> pd.DataFrame:
    """Gap behind the leader at the line, compound and in laps (laps 12–32)."""
    from f1strat.data import load_season
    from f1strat.monaco import gap_timeline_frame

    return gap_timeline_frame(load_season(2022))


def monaco_positions_table() -> pd.DataFrame:
    """Leclerc's position distribution per counterfactual (hindsight)."""
    out = json.loads((RESULTS_DIR / "case_monaco_results.json").read_text())
    rows = []
    for dname, d in out["decisions"].items():
        v = d["hindsight"]
        if not v.get("assessable"):
            continue
        for pname, ps in v["plans"].items():
            for pos, share in ps["pos_dist"].items():
                rows.append(
                    {
                        "decision": dname,
                        "plan": pname,
                        "position": int(pos),
                        "share": share,
                        "mean_points": ps["mean_points"],
                        "too_close": bool(v["too_close"][pname]),
                        "best": pname == v["best"],
                        "call": pname == v["call"],
                    }
                )
    return pd.DataFrame(rows)


def monaco_out(df: pd.DataFrame) -> dict:
    """The ``out`` dict :func:`monaco.plot_positions` needs."""
    out: dict = {"decisions": {}}
    for dname, g in df.groupby("decision", sort=False):
        plans, too_close, best, call = {}, {}, None, None
        for pname, p in g.groupby("plan", sort=False):
            r = p.iloc[0]
            plans[pname] = {
                "mean_points": float(r["mean_points"]),
                "pos_dist": {str(int(k)): float(s) for k, s in zip(p["position"], p["share"])},
            }
            too_close[pname] = bool(r["too_close"])
            best = pname if r["best"] else best
            call = pname if r["call"] else call
        out["decisions"][dname] = {
            "hindsight": {
                "assessable": True,
                "plans": plans,
                "too_close": too_close,
                "best": best,
                "call": call,
            }
        }
    return out


# --- Backtests ------------------------------------------------------------------------


def backtest_tables() -> dict[str, pd.DataFrame]:
    """The calibration plots' inputs: per-race summary, PITs, SC/VSC draws."""
    summary = pd.read_csv(RESULTS_DIR / "backtest_summary.csv")
    pit = pd.concat(
        [
            pd.read_csv(RESULTS_DIR / f"backtest_{k}.csv")[["mode", "pit"]].assign(kind=k)
            for k in ("positions", "gaps")
        ],
        ignore_index=True,
    )
    draws = pd.read_csv(RESULTS_DIR / "backtest_neutral_draws.csv")
    return {"backtest_summary": summary, "backtest_pit": pit, "backtest_neutral_draws": draws}


def backtest_plot_tables() -> dict[str, pd.DataFrame]:
    """The ``tables`` dict :func:`backtest.plot_all` needs, from ``data/figures``."""
    pit = read("backtest_pit")
    return {
        "summary": read("backtest_summary"),
        "positions": pit[pit["kind"] == "positions"].drop(columns="kind"),
        "gaps": pit[pit["kind"] == "gaps"].drop(columns="kind"),
        "passes": read("backtest_neutral_draws").assign(mode="random"),
    }


# --- Notebook figures (EDA and the lap model) -----------------------------------------

#: The lap-model tables are built from the committed thinned fits only.
THIN_ONLY: bool = True


def clean_laps_table() -> pd.DataFrame:
    """Every clean 2022 lap: round, event, compound, tyre age, lap time."""
    from f1strat.clean import clean_laps
    from f1strat.data import load_season

    cleaned = clean_laps(load_season(2022))
    clean = cleaned[cleaned["exclude_reason"].isna()]
    return pd.DataFrame(
        {
            "RoundNumber": clean["RoundNumber"].astype(int),
            "EventName": clean["EventName"].astype(str),
            "Compound": clean["Compound"].astype(str),
            "TyreLife": clean["TyreLife"].astype(float),
            "lap_s": clean["LapTime"].dt.total_seconds(),
        }
    )


def exclude_reasons_table() -> pd.DataFrame:
    """Laps per exclude reason per race (first reason that applies), long form."""
    from f1strat.clean import EXCLUDE_REASONS, clean_laps
    from f1strat.data import load_season

    raw, race = load_season(2022), load_season(2022, table="race")
    cleaned = clean_laps(raw)
    counts = cleaned.assign(
        reason=cleaned["exclude_reason"].astype("object").fillna("clean")
    ).pivot_table(
        index="EventName", columns="reason", values="LapNumber", aggfunc="size", fill_value=0
    )
    order = ["clean"] + [r for r in EXCLUDE_REASONS if r in counts.columns]
    rounds = race.groupby("RoundNumber")["EventName"].first()
    counts = counts[order].loc[rounds.values]
    long = counts.reset_index().melt(id_vars="EventName", var_name="reason", value_name="laps")
    long["event_order"] = long["EventName"].map({e: i for i, e in enumerate(counts.index)})
    long["reason_order"] = long["reason"].map({r: i for i, r in enumerate(order)})
    return long.sort_values(["event_order", "reason_order"]).reset_index(drop=True)


def pit_loss_table() -> pd.DataFrame:
    """The per-circuit pit-loss table (committed parquet), as plotted."""
    return pd.read_parquet(ROOT / "data" / "processed" / "2022" / "pit_loss.parquet")


#: Races in the degradation-curve figure.
DEG_CURVE_RACES: tuple[tuple[int, str], ...] = (
    (1, "Bahrain"),
    (6, "Spain"),
    (10, "Britain"),
    (13, "Hungary"),
)


def deg_curves_table() -> pd.DataFrame:
    """Tyre effect per compound vs a new medium (field-average car, 80% band)."""
    from f1strat.clean import clean_laps
    from f1strat.data import load_season
    from f1strat.deg_model import FULL_FIT_PATH, deg_curve, load_fit, prepare, thin_path

    laps = prepare(clean_laps(load_season(2022)))
    full = load_fit(thin_path(FULL_FIT_PATH))
    out = []
    for rnd, name in DEG_CURVE_RACES:
        rl = laps[laps["RoundNumber"] == rnd]
        for comp in ("SOFT", "MEDIUM", "HARD"):
            used = rl.loc[rl["Compound"] == comp, "TyreLife"]
            if used.empty:
                continue
            c = deg_curve(full, rnd, comp, np.arange(0, used.max() + 1))
            out.append(c.assign(race=name, round=rnd, compound=comp))
    return pd.concat(out, ignore_index=True)


def long_tyre_tables() -> dict[str, pd.DataFrame]:
    """``long_tyre`` stress test: mean error by tyre age per model, and the
    longest test stint with each model's prediction."""
    from f1strat.clean import clean_laps
    from f1strat.data import load_season
    from f1strat.deg_model import (
        SPLITS,
        fit_baseline,
        fit_xgboost,
        load_fit,
        predict,
        predict_baseline,
        predict_xgboost,
        prepare,
        split_fit_path,
        thin_path,
    )

    laps = prepare(clean_laps(load_season(2022)))
    train, test = SPLITS["long_tyre"](laps)
    xgb = predict_xgboost(fit_xgboost(train), test)
    bayes = predict(load_fit(thin_path(split_fit_path("long_tyre"))), test)
    base = predict_baseline(fit_baseline(train), test)
    age_bin = pd.cut(test["TyreLife"], np.arange(15, 51, 5))
    err = (
        pd.DataFrame(
            {
                "Baseline (OLS)": base - test["lap_s"],
                "XGBoost": xgb - test["lap_s"],
                "Bayesian": bayes.mean - test["lap_s"],
            }
        )
        .groupby(age_bin, observed=True)
        .mean()
    )
    errors = err.reset_index(drop=True).assign(age_mid=[iv.mid for iv in err.index])
    errors = errors.melt(id_vars="age_mid", var_name="model", value_name="mean_error_s")
    key = ["RoundNumber", "Driver", "Stint"]
    longest = test.groupby(key)["TyreLife"].max().idxmax()
    idx = test.index[(test[key] == pd.Series(longest, index=key)).all(axis=1)]
    event = laps.groupby("RoundNumber")["EventName"].first().str.replace(" Grand Prix", "")
    stint = pd.DataFrame(
        {
            "TyreLife": test.loc[idx, "TyreLife"].to_numpy(),
            "actual": test.loc[idx, "lap_s"].to_numpy(),
            "bayes_mean": np.asarray(bayes.mean[idx]),
            "bayes_lo": np.asarray(bayes.lo[idx]),
            "bayes_hi": np.asarray(bayes.hi[idx]),
            "xgboost": np.asarray(xgb[idx]),
        }
    )
    rnd, drv, st = longest
    stint["title"] = (
        f"{event[rnd]}: {drv} stint {st} on {test.loc[idx, 'Compound'].iloc[0].title()}"
    )
    return {"xgb_vs_bayes_long_tyre_errors": errors, "xgb_vs_bayes_long_tyre_stint": stint}


def leclerc_hungary_hard_table() -> pd.DataFrame:
    """Leclerc's Hungary hard stint (laps 41–53): actual laps and the full fit."""
    from f1strat.clean import clean_laps
    from f1strat.data import load_season
    from f1strat.deg_model import FULL_FIT_PATH, load_fit, predict, prepare, thin_path

    laps = prepare(clean_laps(load_season(2022)))
    lec = laps[
        (laps["RoundNumber"] == 13) & (laps["Driver"] == "LEC") & laps["race_lap"].between(41, 53)
    ]
    pred = predict(load_fit(thin_path(FULL_FIT_PATH)), lec)
    return pd.DataFrame(
        {
            "race_lap": lec["race_lap"].to_numpy(),
            "TyreLife": lec["TyreLife"].to_numpy(),
            "actual": lec["lap_s"].to_numpy(),
            "predicted": np.asarray(pred.mean),
            "lo": np.asarray(pred.lo),
            "hi": np.asarray(pred.hi),
            "rainfall": lec["Rainfall"].to_numpy(bool),
        }
    )


# --- README ---------------------------------------------------------------------------


def hungary_trace_table() -> pd.DataFrame:
    """Leclerc's real gap to Verstappen at the line, every lap of Hungary 2022
    (s; positive = Leclerc behind)."""
    from f1strat.data import load_season

    raw = load_season(2022)
    r = raw[raw["RoundNumber"] == 13]
    t = r.pivot(index="LapNumber", columns="Driver", values="Time").apply(
        lambda c: c.dt.total_seconds()
    )
    return pd.DataFrame({"lap": t.index.astype(int), "gap_s": (t["LEC"] - t["VER"]).to_numpy()})


def readme_hungary_table() -> pd.DataFrame:
    """Mean points per named plan at both Hungary decisions, decision vs hindsight."""
    rows = []
    for stem, title, call in (
        ("hungary", "Lap 38 decision: hards at 39", "call: hards at 39"),
        ("hungary54", "Lap 53 decision: used softs at 54", "call: used softs at 54"),
    ):
        r = _results(stem)
        meas, hind = r["checks"][0]["measure"], r["hindsight"]
        for n in meas:
            if n in ("best overall", "as run") or (stem == "hungary" and n == "hards to the flag"):
                continue
            rows.append(
                {
                    "panel": title,
                    "family": n,
                    "is_call": n == call,
                    "decision_points": meas[n]["mean_points"],
                    "hindsight_points": hind[n]["mean_points"],
                }
            )
    return pd.DataFrame(rows)


def readme_silverstone_table() -> pd.DataFrame:
    """Passes in the first three laps after the restart: real, forecast, replays;
    and Leclerc's P1 share in the v2 hindsight replay."""
    v1 = _results("silverstone")["reality_check"]
    v2 = _results("silverstone_v2")["reality_check"]
    p1 = v2["hindsight"]["final_pos_dist"].get("1", 0.0)
    return pd.DataFrame(
        [
            {"source": "Real race", "passes": v1["real_passes_first_3"], "real": True},
            {
                "source": "Lap-38 forecast",
                "passes": v1["forecast"]["passes_first_3_mean"],
                "real": False,
            },
            {
                "source": "Hindsight replay (v1)",
                "passes": v1["hindsight"]["passes_first_3_mean"],
                "real": False,
            },
            {
                "source": "Hindsight replay, restart layer (v2)",
                "passes": v2["hindsight"]["passes_first_3_mean"],
                "real": False,
            },
        ]
    ).assign(v2_hindsight_p1=p1)


def readme_monaco_table() -> pd.DataFrame:
    """Hindsight points per alternative at Monaco D1 (lap 17) and D2 (lap 20)."""
    r = json.loads((RESULTS_DIR / "case_monaco_results.json").read_text())["decisions"]
    rows = []
    for panel, key in (
        ("D1 at lap 17: inters at 18?", "D1 inters (cutoff 17)"),
        ("D2 at lap 20: double stack at 21?", "D2 slicks (cutoff 20)"),
    ):
        h = r[key]["hindsight"]
        for n, p in h["plans"].items():
            rows.append(
                {
                    "panel": panel,
                    "plan": n,
                    "is_call": n == h["call"],
                    "mean_points": p["mean_points"],
                }
            )
    return pd.DataFrame(rows)


def readme_calibration_table() -> pd.DataFrame:
    """Backtest coverage against nominal (replay, widening 1.0)."""
    ov = pd.read_csv(RESULTS_DIR / "backtest_overall_widening1.csv").iloc[0]
    return pd.DataFrame(
        [
            {
                "label": "Finishing position, 50% interval",
                "nominal": 0.5,
                "achieved": ov["pos_pit50"],
            },
            {
                "label": "Finishing position, 80% interval",
                "nominal": 0.8,
                "achieved": ov["pos_pit80"],
            },
            {
                "label": "Gap to the winner, 50% interval",
                "nominal": 0.5,
                "achieved": ov["gap_in50"],
            },
            {
                "label": "Gap to the winner, 80% interval",
                "nominal": 0.8,
                "achieved": ov["gap_in80"],
            },
        ]
    )


def _fan_from_results() -> pd.DataFrame:
    """The lap-38 gap forecast saved by ``case_study.hungary_fan`` (the only
    table here that comes from a simulation, so it is never re-extracted by
    ``--export``: rerun ``pitwall case-study hungary`` to rebuild it)."""
    return read("hungary_fan")


# --- Registry -------------------------------------------------------------------------


def _case_extractors() -> dict[str, Callable[[], pd.DataFrame]]:
    out: dict[str, Callable[[], pd.DataFrame]] = {}
    for stem in CASE_STEMS:
        out[f"case_{stem}_points"] = lambda s=stem: points_table(s)
        out[f"case_{stem}_sensitivity"] = lambda s=stem: sensitivity_table(s)
        out[f"case_{stem}_hindsight"] = lambda s=stem: hindsight_table(s)
        if stem != "hungary54":
            out[f"case_{stem}_reality"] = lambda s=stem: reality_table(s)
        if stem.startswith("silverstone"):
            out[f"case_{stem}_joint"] = lambda s=stem: joint_table(s)
    return out


#: Table name → extractor, for every table except the simulated gap fan.
EXTRACTORS: dict[str, Callable[[], pd.DataFrame]] = {
    **_case_extractors(),
    "case_monaco_gaps": monaco_gaps_table,
    "case_monaco_positions": monaco_positions_table,
    "clean_laps.parquet": clean_laps_table,
    "exclude_reasons": exclude_reasons_table,
    "pit_loss": pit_loss_table,
    "deg_curves": deg_curves_table,
    "leclerc_hungary_hard": leclerc_hungary_hard_table,
    "hungary_trace": hungary_trace_table,
    "readme_hungary": readme_hungary_table,
    "readme_silverstone": readme_silverstone_table,
    "readme_monaco": readme_monaco_table,
    "readme_calibration": readme_calibration_table,
}

#: Extractors that return several tables at once.
MULTI_EXTRACTORS: tuple[Callable[[], dict[str, pd.DataFrame]], ...] = (
    backtest_tables,
    long_tyre_tables,
)


def export_all() -> list[Path]:
    """Rebuild every table in ``data/figures`` from the heavy sources."""
    paths = [_write(fn(), name) for name, fn in EXTRACTORS.items()]
    for multi in MULTI_EXTRACTORS:
        paths += [_write(df, name) for name, df in multi().items()]
    return paths


__all__ = ["DATA_DIR", "EXTRACTORS", "export_all", "path", "read"]
