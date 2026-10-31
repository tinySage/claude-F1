"""Restart layer (Phase 6, added after Silverstone v1; ``docs/case_studies.md``).

The lap model never sees the first green lap after an SC/VSC (Phase 2 labels
it ``restart_lap``), and the simulator ran restarts as ordinary green laps.
This layer adds, for the first :data:`RESTART_LAPS` green laps after an SC or
VSC ends:

- **pace:** an offset on each car's lap, by laps since the restart ``k``,
  compound and tyre age, estimated from lap-model residuals on those laps:
  ``offset_k = a_k + s_k·[soft] + h_k·[hard] + g_k · TyreLife/10``
  (medium is the baseline compound);
- **passing:** a multiplier on the pass probability on those laps, the ratio
  of observed passes to the passes the pass model expects on restart-lap
  events.

Both are estimated from chosen rounds only (Silverstone: the dry rounds before
it), with the lap model of the decision (the cutoff fit), so no future
information enters.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from f1strat.clean import MIXED_CONDITIONS_2022

RESTART_LAPS = 3
NEUTRAL_CODES = "4567"
DRY = ("SOFT", "MEDIUM", "HARD")


@dataclass
class RestartModel:
    """Fitted restart layer. ``pace[k-1] = [a, s, h, g]`` (s per lap)."""

    pace: list[list[float]]
    pace_se: list[list[float]]
    pass_mult: float
    n_laps: list[int]
    n_events: int
    n_passes: int
    expected_passes: float
    rounds: list[int]

    def offset(self, k: np.ndarray, compound: np.ndarray, age: np.ndarray) -> np.ndarray:
        """Pace offset (s) for laps ``k`` since the restart (0 = none), compound
        index into ``COMPOUNDS`` (SOFT, MEDIUM, HARD) and tyre age."""
        from f1strat.deg_model import COMPOUNDS

        p = np.asarray(self.pace, float)
        k = np.asarray(k)
        on = (k >= 1) & (k <= len(p))
        row = p[np.clip(k - 1, 0, len(p) - 1)]
        soft = (np.asarray(compound) == COMPOUNDS.index("SOFT")).astype(float)
        hard = (np.asarray(compound) == COMPOUNDS.index("HARD")).astype(float)
        out = (row[..., 0] + row[..., 1] * soft + row[..., 2] * hard
               + row[..., 3] * np.asarray(age, float) / 10.0)
        return np.where(on, out, 0.0)

    def to_dict(self) -> dict:
        return asdict(self)


def laps_since_restart(laps: pd.DataFrame) -> pd.Series:
    """Per lap, green laps since the driver's last SC/VSC/red lap (1 = the
    restart lap); 0 on neutral laps, 99 if no neutral lap came before."""
    df = laps.sort_values(["RoundNumber", "Driver", "LapNumber"])
    neutral = df["TrackStatus"].fillna("").astype(str).str.contains(f"[{NEUTRAL_CODES}]")
    out = []
    for _, q in df.assign(_n=neutral).groupby(["RoundNumber", "Driver"], sort=False):
        last = None
        for lap, n in zip(q["LapNumber"], q["_n"]):
            if n:
                last = lap
                out.append(0)
            else:
                out.append(int(lap - last) if last is not None else 99)
    return pd.Series(out, index=df.index).reindex(laps.index)


def restart_laps(cleaned: pd.DataFrame, rounds) -> pd.DataFrame:
    """The first :data:`RESTART_LAPS` green laps after an SC/VSC, dry
    compounds, in chosen rounds; in and out laps, lap 1, and laps the cleaning
    drops for other reasons (deleted, inaccurate, wet...) are left out. The
    restart lap itself (``exclude_reason == "restart_lap"``) is kept."""
    from f1strat.deg_model import lap_features

    df = cleaned[cleaned["RoundNumber"].isin(list(rounds))
                 & ~cleaned["EventName"].isin(list(MIXED_CONDITIONS_2022))]
    f = lap_features(df)
    f = f[f["Compound"].isin(DRY)].copy()
    f["k"] = laps_since_restart(f)
    keep = (f["k"].between(1, RESTART_LAPS) & f["PitInTime"].isna() & f["PitOutTime"].isna()
            & (f["LapNumber"] > 1) & f["lap_s"].notna()
            & (f["exclude_reason"].isna() | (f["exclude_reason"] == "restart_lap")))
    f = f[keep].copy()
    f["dirty_air"] = f["dirty_air"].fillna(False).astype(bool)
    return f


def residuals(laps: pd.DataFrame, fitted, n_draws: int = 500) -> pd.Series:
    """Lap time minus the lap model's posterior mean."""
    from f1strat.deg_model import predict

    return laps["lap_s"] - predict(fitted, laps, n_draws=n_draws).mean


def fit_pace(laps: pd.DataFrame) -> tuple[list[list[float]], list[list[float]], list[int]]:
    """Per ``k``: OLS of the residual on [1, soft, hard, TyreLife/10]."""
    pace, se, n = [], [], []
    for k in range(1, RESTART_LAPS + 1):
        x = laps[(laps["k"] == k) & laps["res"].notna()]
        A = np.column_stack([np.ones(len(x)), x["Compound"].eq("SOFT"),
                             x["Compound"].eq("HARD"), x["TyreLife"] / 10.0]).astype(float)
        y = x["res"].to_numpy(float)
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        r = y - A @ beta
        cov = (r @ r / max(len(y) - A.shape[1], 1)) * np.linalg.inv(A.T @ A)
        pace.append([round(float(v), 4) for v in beta])
        se.append([round(float(v), 4) for v in np.sqrt(np.diag(cov))])
        n.append(int(len(y)))
    return pace, se, n


def restart_pass_events(cleaned: pd.DataFrame, fitted, rounds) -> pd.DataFrame:
    """Pass events (:func:`f1strat.race_events.pass_events`) in chosen rounds,
    with ``k`` of the outcome lap for the attacker."""
    from f1strat.race_events import pass_events

    df = cleaned[cleaned["RoundNumber"].isin(list(rounds))]
    ev = pass_events(df, fitted)
    k = laps_since_restart(df)
    kmap = pd.Series(k.to_numpy(), index=pd.MultiIndex.from_frame(
        df[["RoundNumber", "Driver", "LapNumber"]].astype({"LapNumber": int})))
    ev["k"] = [int(kmap.get((r, d, int(l)), 99)) for r, d, l in
               zip(ev["RoundNumber"], ev["attacker"], ev["lap"])]
    return ev


def expected_pass(ev: pd.DataFrame, passing) -> np.ndarray:
    a = np.array([passing.circuit.get(int(r), passing.intercept) for r in ev["RoundNumber"]])
    return 1.0 / (1.0 + np.exp(-(a + passing.slope * ev["pace_gap_s"].to_numpy(float))))


def estimate(cleaned: pd.DataFrame, fitted, rounds) -> RestartModel:
    """Restart layer from ``rounds`` with lap model ``fitted``.

    The pass multiplier is (passes on restart-lap events) / (passes the pass
    model fitted on the same rounds' events expects), pooled over k = 1..3.
    """
    from f1strat.race_events import fit_pass_model

    laps = restart_laps(cleaned, rounds)
    laps["res"] = residuals(laps, fitted)
    pace, se, n = fit_pace(laps)
    ev = restart_pass_events(cleaned, fitted, rounds)
    pm = fit_pass_model(ev)
    r = ev[ev["k"].between(1, RESTART_LAPS)]
    exp = float(expected_pass(r, pm).sum())
    return RestartModel(pace=pace, pace_se=se, pass_mult=round(float(r["passed"].sum()) / exp, 4),
                        n_laps=n, n_events=int(len(r)), n_passes=int(r["passed"].sum()),
                        expected_passes=round(exp, 2), rounds=sorted(int(x) for x in rounds))


def laps_since_restart_sim(status: np.ndarray, start_neutral: bool) -> np.ndarray:
    """(n_sims, laps): green laps since an SC/VSC ended (1 = restart lap), 0
    otherwise. ``start_neutral``: the state lap itself was neutral."""
    n, L = status.shape
    k = np.zeros((n, L), dtype=int)
    since = np.where(start_neutral, 0, -1) * np.ones(n, dtype=int)   # -1: no restart yet
    for j in range(L):
        green = status[:, j] == 0
        since = np.where(~green, 0, np.where(since >= 0, since + 1, -1))
        k[:, j] = np.where(green & (since >= 1), since, 0)
    return k


#: Later dry rounds for validation: after Silverstone, excluding the case
#: studies (Hungary 13) and Austria (11, missing compounds); Singapore and
#: Japan are mixed-conditions races.
VALIDATION_ROUNDS: tuple[int, ...] = (12, 14, 15, 16, 19, 20, 21, 22)


def validate_pace(model: RestartModel, cleaned: pd.DataFrame, fitted,
                  rounds=VALIDATION_ROUNDS) -> pd.DataFrame:
    """Restart-lap residuals in ``rounds`` before (lap model alone) and after
    (lap model + restart offset): mean and mean absolute error per ``k``, and
    the hard-minus-soft mean residual."""
    from f1strat.deg_model import COMPOUNDS

    laps = restart_laps(cleaned, rounds)
    laps["res"] = residuals(laps, fitted)
    comp = laps["Compound"].map(COMPOUNDS.index).to_numpy()
    laps["res_after"] = laps["res"] - model.offset(laps["k"].to_numpy(), comp,
                                                   laps["TyreLife"].to_numpy())
    rows = []
    for k, g in laps.groupby("k"):
        hs = {}
        for col in ("res", "res_after"):
            by = g.groupby("Compound")[col].mean()
            hs[col] = float(by.get("HARD", np.nan) - by.get("SOFT", np.nan))
        rows.append({"k": int(k), "laps": len(g),
                     "mean_before": g["res"].mean(), "mean_after": g["res_after"].mean(),
                     "mae_before": g["res"].abs().mean(), "mae_after": g["res_after"].abs().mean(),
                     "hard_minus_soft_before": hs["res"], "hard_minus_soft_after": hs["res_after"]})
    return pd.DataFrame(rows).round(3)


def validate_passes(model: RestartModel, fitted, raw: pd.DataFrame, race: pd.DataFrame,
                    cleaned: pd.DataFrame, params, rounds=VALIDATION_ROUNDS,
                    n_sims: int = 300) -> pd.DataFrame:
    """Replay backtests (actual plans, SC/VSC and retirements) of ``rounds``
    with and without the restart layer: passes in the first laps after each
    restart, simulated (mean) vs actual, and all passes."""
    from dataclasses import replace

    from f1strat.backtest import run_backtest

    rows = []
    for rnd in rounds:
        out = {"round": int(rnd)}
        for tag, prm in (("before", replace(params, restart=None)),
                         ("after", replace(params, restart=model))):
            bt = run_backtest(fitted, raw, race, cleaned, prm, rnd, mode="replay",
                              n_sims=n_sims, widening=1.0)
            out[f"restart_passes_{tag}"] = float(bt.restart_passes.mean())
            out[f"all_passes_{tag}"] = float(bt.passes.mean())
            out["restart_passes_actual"] = bt.actual_restart_passes
            out["all_passes_actual"] = bt.actual_counts["passes"]
        rows.append(out)
    df = pd.DataFrame(rows)
    total = df.drop(columns="round").sum().to_dict()
    total["round"] = "total"
    return pd.concat([df, pd.DataFrame([total])], ignore_index=True).round(2)


def main() -> None:
    """``python -m f1strat.restart``: estimate the layer from the dry rounds
    before Silverstone (Silverstone lap-38 cutoff fit) and validate it on the
    later dry rounds (thinned full fit), writing
    ``reports/figures/restart_layer.json`` and ``restart_validation_*.csv``."""
    import json
    import logging

    from f1strat.backtest import FIG_DIR
    from f1strat.clean import clean_laps
    from f1strat.data import load_season
    from f1strat.deg_model import THIN_FIT_PATH, load_fit, thin_fit
    from f1strat.race_events import SimParams
    from f1strat.rehearsal import fit_path

    logging.disable(logging.WARNING)
    raw, race = load_season(2022), load_season(2022, table="race")
    cleaned = clean_laps(raw)
    before = [r for r in (1, 2, 3, 5, 6, 8, 9)]           # dry rounds before Silverstone
    cut = thin_fit(load_fit(fit_path(10, 38, "cutoff")))
    model = estimate(cleaned, cut, before)
    print(json.dumps(model.to_dict(), indent=1), flush=True)
    full = load_fit(THIN_FIT_PATH)
    pace = validate_pace(model, cleaned, full)
    print(pace.to_string(index=False), flush=True)
    passes = validate_passes(model, full, raw, race, cleaned, SimParams.load())
    print(passes.to_string(index=False), flush=True)
    (FIG_DIR / "restart_layer.json").write_text(json.dumps(model.to_dict(), indent=1))
    pace.to_csv(FIG_DIR / "restart_validation_pace.csv", index=False)
    passes.to_csv(FIG_DIR / "restart_validation_passes.csv", index=False)


if __name__ == "__main__":
    main()
