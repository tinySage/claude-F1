"""Tyre degradation models (Phase 3).

- Data preparation and the two fixed held-out test sets used by every model.
- Linear baseline: per-race OLS.
- Bayesian hierarchical model in PyMC (spec in ``docs/model.md``).
- XGBoost, fitted only to show it fails to extrapolate on long stints.

All fit functions take ``(laps, cutoff=None)``. ``cutoff`` keeps only what was
known at a decision lap, so Phase 6 can refit without new code.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

#: Laps on a tyre at or below this age form the long_tyre training set.
LONG_TYRE_MAX_TRAIN_AGE = 15

#: Laps since the stop that count as the first two flying laps (the out lap is 0).
WARMUP_LAPS = (1, 2)


@dataclass(frozen=True)
class Cutoff:
    """The last lap known at a decision point: (round, lap) in race order.

    Data kept: every lap of earlier rounds, plus laps of ``round_number`` up to
    and including ``lap_number``. Later rounds are dropped.
    """

    round_number: int
    lap_number: int


def apply_cutoff(laps: pd.DataFrame, cutoff: Cutoff | None) -> pd.DataFrame:
    """Keep only the laps known at ``cutoff`` (all laps if ``cutoff`` is None)."""
    if cutoff is None:
        return laps
    earlier = laps["RoundNumber"] < cutoff.round_number
    same_race = (laps["RoundNumber"] == cutoff.round_number) & (
        laps["LapNumber"] <= cutoff.lap_number
    )
    return laps[earlier | same_race]


def prepare(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Model-ready clean laps from :func:`f1strat.clean.clean_laps` output.

    Stint features are computed on all laps first (the out lap and in lap are
    excluded from modelling but still define where a stint starts and ends),
    then only clean laps (``exclude_reason`` null) with a recorded
    ``TyreLife`` are kept. Missing tyre age is not filled in (73 clean laps,
    all Austria 2022), matching the missing-compound decision.

    Returns:
        Clean laps with ``lap_s`` (lap time in seconds), ``race_lap``
        (``LapNumber``), ``laps_since_stop`` (0 on the out lap or race start),
        ``first_two_flying`` (``laps_since_stop`` is 1 or 2) and
        ``final_stint`` (the driver's last stint of the race).
    """
    df = cleaned.copy()
    key = ["RoundNumber", "Driver"]
    stint_start = df.groupby(key + ["Stint"])["LapNumber"].transform("min")
    df["laps_since_stop"] = (df["LapNumber"] - stint_start).astype("int64")
    df["first_two_flying"] = df["laps_since_stop"].isin(WARMUP_LAPS)
    df["final_stint"] = df["Stint"] == df.groupby(key)["Stint"].transform("max")
    df = df[df["exclude_reason"].isna() & df["TyreLife"].notna()].copy()
    df["lap_s"] = df["LapTime"].dt.total_seconds()
    df["race_lap"] = df["LapNumber"].astype("int64")
    df["TyreLife"] = df["TyreLife"].astype("int64")
    df["Compound"] = df["Compound"].astype(str)
    return df.reset_index(drop=True)


def split_last_stint(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out each driver's final stint in each race.

    Mimics forecasting the end of a race. A final stint is only tested if the
    same driver has clean laps from an earlier stint in that race (otherwise
    there is nothing to forecast from, e.g. a driver who retired in stint 1).
    """
    train = df[~df["final_stint"]]
    seen = train[["RoundNumber", "Driver"]].drop_duplicates()
    test = df[df["final_stint"]].merge(seen, on=["RoundNumber", "Driver"])
    return train.reset_index(drop=True), test.reset_index(drop=True)


def split_long_tyre(
    df: pd.DataFrame, max_train_age: int = LONG_TYRE_MAX_TRAIN_AGE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on ``TyreLife <= max_train_age``, test on older tyres (extrapolation)."""
    old = df["TyreLife"] > max_train_age
    return df[~old].reset_index(drop=True), df[old].reset_index(drop=True)


SPLITS: dict[str, Callable[[pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]]] = {
    "last_stint": split_last_stint,
    "long_tyre": split_long_tyre,
}


# --- Linear baseline -------------------------------------------------------


@dataclass
class RaceFit:
    """One race's OLS coefficients and the levels seen in its training data."""

    coef: pd.Series
    drivers: frozenset[str]
    compounds: frozenset[str]


@dataclass
class BaselineModel:
    """Per-race OLS fits: ``lap_s ~ driver + compound + compound:TyreLife +
    race_lap + dirty_air``, with reference coding for driver and compound."""

    races: dict[int, RaceFit] = field(default_factory=dict)


def _baseline_design(
    race_laps: pd.DataFrame, drivers: list[str], compounds: list[str]
) -> pd.DataFrame:
    """Design matrix; the first driver and compound in each list are the reference."""
    x = pd.DataFrame({"intercept": 1.0}, index=race_laps.index)
    for d in drivers[1:]:
        x[f"driver[{d}]"] = (race_laps["Driver"] == d).astype(float)
    for i, c in enumerate(compounds):
        is_c = (race_laps["Compound"] == c).astype(float)
        if i > 0:
            x[f"compound[{c}]"] = is_c
        x[f"deg[{c}]"] = is_c * race_laps["TyreLife"]
    x["race_lap"] = race_laps["race_lap"].astype(float)
    x["dirty_air"] = race_laps["dirty_air"].astype(float)
    return x


def fit_baseline(laps: pd.DataFrame, cutoff: Cutoff | None = None) -> BaselineModel:
    """Fit the per-race OLS baseline on prepared laps (see :func:`prepare`).

    The reference levels are the alphabetically first driver and compound in
    each race, so the intercept is that driver on that compound at tyre age 0
    and race lap 0. Solved by least squares.
    """
    model = BaselineModel()
    for rnd, race_laps in apply_cutoff(laps, cutoff).groupby("RoundNumber"):
        drivers = sorted(race_laps["Driver"].unique())
        compounds = sorted(race_laps["Compound"].unique())
        x = _baseline_design(race_laps, drivers, compounds)
        beta, *_ = np.linalg.lstsq(x.to_numpy(), race_laps["lap_s"].to_numpy(), rcond=None)
        model.races[int(rnd)] = RaceFit(
            coef=pd.Series(beta, index=x.columns),
            drivers=frozenset(drivers),
            compounds=frozenset(compounds),
        )
    return model


def predict_baseline(model: BaselineModel, laps: pd.DataFrame) -> pd.Series:
    """Predicted lap time in seconds.

    NaN where the race, the driver in that race, or the compound in that race
    was not in the training data (OLS has no estimate for an unseen level).
    """
    pred = pd.Series(np.nan, index=laps.index)
    for rnd, race_laps in laps.groupby("RoundNumber"):
        fit_r = model.races.get(int(rnd))
        if fit_r is None:
            continue
        drivers = sorted(fit_r.drivers)
        compounds = sorted(fit_r.compounds)
        x = _baseline_design(race_laps, drivers, compounds)[fit_r.coef.index]
        seen = race_laps["Driver"].isin(fit_r.drivers) & race_laps["Compound"].isin(
            fit_r.compounds
        )
        y = x.to_numpy() @ fit_r.coef.to_numpy()
        pred.loc[race_laps.index] = np.where(seen, y, np.nan)
    return pred


# --- Evaluation ------------------------------------------------------------


def evaluate(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """MAE and RMSE in seconds over laps with a prediction, plus counts."""
    ok = predicted.notna()
    err = (predicted[ok] - actual[ok]).to_numpy()
    return {
        "mae_s": float(np.abs(err).mean()),
        "rmse_s": float(np.sqrt((err**2).mean())),
        "n_predicted": int(ok.sum()),
        "n_unpredictable": int((~ok).sum()),
    }


def baseline_report(df: pd.DataFrame) -> pd.DataFrame:
    """Fit and score the baseline on every split in :data:`SPLITS`.

    Args:
        df: Output of :func:`prepare`.

    Returns:
        One row per split with train/test sizes, MAE and RMSE.
    """
    rows = []
    for name, split in SPLITS.items():
        train, test = split(df)
        pred = predict_baseline(fit_baseline(train), test)
        rows.append({"split": name, "n_train": len(train), "n_test": len(test),
                     **evaluate(test["lap_s"], pred)})
    return pd.DataFrame(rows)


def fit(laps: pd.DataFrame, cutoff: Cutoff | None = None) -> None:
    """Fit the Bayesian hierarchical model (Phase 3 Part B, after spec approval)."""
    raise NotImplementedError("Bayesian model awaits spec approval (docs/model.md)")
