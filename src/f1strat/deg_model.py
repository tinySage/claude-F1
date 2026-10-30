"""Tyre degradation models (Phase 3).

- Data preparation and the two fixed held-out test sets used by every model.
- Linear baseline: per-race OLS.
- Bayesian hierarchical model in PyMC (spec in ``docs/model.md``).
- XGBoost, fitted only to show it fails to extrapolate on long stints.

All fit functions take ``(laps, cutoff=None)``. ``cutoff`` keeps only what was
known at a decision lap, so Phase 6 can refit without new code.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import pymc as pm
    import xarray as xr

#: Laps on a tyre at or below this age form the long_tyre training set.
LONG_TYRE_MAX_TRAIN_AGE = 15

#: Laps since the stop that count as the first two flying laps (the out lap is 0).
WARMUP_LAPS = (1, 2)

#: Race laps that count as opening laps (bunched field, DRS off).
OPENING_LAPS = (2, 3)


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
        ``first_two_flying`` (``laps_since_stop`` is 1 or 2, only in stints
        that began with a pit stop), ``opening_laps`` (race laps 2–3) and
        ``final_stint`` (the driver's last stint of the race).
    """
    df = cleaned.copy()
    key = ["RoundNumber", "Driver"]
    stint_start = df.groupby(key + ["Stint"])["LapNumber"].transform("min")
    df["laps_since_stop"] = (df["LapNumber"] - stint_start).astype("int64")
    after_stop = stint_start > 1
    df["first_two_flying"] = df["laps_since_stop"].isin(WARMUP_LAPS) & after_stop
    df["opening_laps"] = df["LapNumber"].isin(OPENING_LAPS)
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


# --- Bayesian hierarchical model ---------------------------------------------

COMPOUNDS: tuple[str, ...] = ("SOFT", "MEDIUM", "HARD")
NONREF_COMPOUNDS: tuple[str, ...] = ("SOFT", "HARD")
#: Tyre age enters the model in units of this many laps.
TYRE_SCALE = 10.0

#: Posterior variables kept after fitting (z-scores and hyperpriors not needed
#: for prediction are dropped to keep the saved file small).
KEEP_VARS: tuple[str, ...] = (
    "race_base", "driver_season", "sigma_driver", "sigma_driver_race", "driver_offset",
    "driver_race_mean", "mu_compound", "tau_compound", "compound_offset",
    "team_compound_season", "sigma_tc_season", "sigma_tc_race", "team_compound",
    "mu_deg", "tau_deg", "deg", "team_deg", "sigma_team_deg", "cliff", "warmup",
    "opening", "mu_trend", "tau_trend", "trend", "dirty_air_penalty",
    "rain_effect", "sigma", "nu",
)


@dataclass
class BayesFit:
    """A fitted hierarchical model: thinned posterior plus index metadata.

    ``posterior`` holds draws with dims ``(chain, draw, ...)`` in model units
    (deg per 10 laps, cliff per (10 laps)^2). Use :func:`effects_table` or
    the ``*_s_per_lap`` helpers for reporting units.
    """

    posterior: "xr.Dataset"
    races: list[int]
    drivers: list[str]
    teams: list[str]
    driver_race: list[tuple[int, str]]
    lap_centre: float
    diagnostics: dict[str, float] = field(default_factory=dict)


def _index(values: pd.Series, levels: list) -> np.ndarray:
    """Position of each value in ``levels`` (-1 if absent)."""
    lookup = {v: i for i, v in enumerate(levels)}
    return values.map(lookup).fillna(-1).astype(int).to_numpy()


def build_model(laps: pd.DataFrame) -> tuple["pm.Model", dict]:
    """Build the PyMC model in ``docs/model.md`` for prepared laps.

    Returns:
        The model and its index metadata (races, drivers, teams, driver-race
        pairs, race-lap centre).
    """
    import pymc as pm
    import pytensor.tensor as pt

    races = sorted(int(r) for r in laps["RoundNumber"].unique())
    drivers = sorted(laps["Driver"].unique())
    teams = sorted(laps["Team"].unique())
    pairs = sorted(set(zip(laps["RoundNumber"].astype(int), laps["Driver"])))
    lap_centre = float(laps["race_lap"].mean())

    r = _index(laps["RoundNumber"].astype(int), races)
    t = _index(laps["Team"], teams)
    c = _index(laps["Compound"], list(COMPOUNDS))
    dr = _index(pd.Series(list(zip(laps["RoundNumber"].astype(int), laps["Driver"]))), pairs)
    pair_race = np.array([races.index(rr) for rr, _ in pairs])
    pair_driver = np.array([drivers.index(dd) for _, dd in pairs])
    average = np.zeros((len(races), len(pairs)))
    average[pair_race, np.arange(len(pairs))] = 1.0
    average /= average.sum(axis=1, keepdims=True)

    a = laps["TyreLife"].to_numpy(float) / TYRE_SCALE
    lap = laps["race_lap"].to_numpy(float) - lap_centre
    warm = laps["first_two_flying"].to_numpy(float)
    opening_x = laps["opening_laps"].to_numpy(float)
    dirty = laps["dirty_air"].to_numpy(float)
    rain = laps["Rainfall"].to_numpy(float)
    y = laps["lap_s"].to_numpy(float)
    race_median = laps.groupby("RoundNumber")["lap_s"].median().loc[races].to_numpy()

    coords = {
        "race": races, "driver": drivers, "team": teams, "compound": list(COMPOUNDS),
        "compound_nonref": list(NONREF_COMPOUNDS), "driver_race": np.arange(len(pairs)),
    }
    n_race, n_team = len(races), len(teams)
    with pm.Model(coords=coords) as model:
        race_base = pm.Normal("race_base", race_median, 3.0, dims="race")

        sigma_driver = pm.HalfNormal("sigma_driver", 1.5)
        driver = pm.Deterministic(
            "driver_season",
            sigma_driver * pm.ZeroSumNormal("z_driver_season", sigma=1.0, dims="driver"),
            dims="driver",
        )
        sigma_dr = pm.HalfNormal("sigma_driver_race", 0.5)
        z_dr = pm.Normal("z_driver_race", 0.0, 1.0, dims="driver_race")
        driver_raw = driver[pair_driver] + sigma_dr * z_dr
        race_mean = pm.Deterministic("driver_race_mean", pt.dot(average, driver_raw), dims="race")
        driver_offset = pm.Deterministic(
            "driver_offset", driver_raw - race_mean[pair_race], dims="driver_race"
        )

        mu_comp = pm.Normal("mu_compound", 0.0, 1.0, dims="compound_nonref")
        tau_comp = pm.HalfNormal("tau_compound", 0.5, dims="compound_nonref")
        z_comp = pm.Normal("z_compound", 0.0, 1.0, dims=("compound_nonref", "race"))
        comp_nonref = mu_comp[:, None] + tau_comp[:, None] * z_comp
        compound_offset = pm.Deterministic(
            "compound_offset",
            pt.stack([comp_nonref[0], pt.zeros(n_race), comp_nonref[1]]),
            dims=("compound", "race"),
        )

        sigma_tcs = pm.HalfNormal("sigma_tc_season", 0.5)
        tc_season = pm.Deterministic(
            "team_compound_season",
            sigma_tcs * pm.ZeroSumNormal("z_tc_season", sigma=1.0,
                                         dims=("compound_nonref", "team")),
            dims=("compound_nonref", "team"),
        )
        sigma_tcr = pm.HalfNormal("sigma_tc_race", 0.3)
        z_tc = pm.Normal("z_team_compound", 0.0, 1.0, dims=("compound_nonref", "race", "team"))
        tc_raw = tc_season[:, None, :] + sigma_tcr * z_tc
        tc = tc_raw - tc_raw.mean(axis=2, keepdims=True)
        team_compound = pm.Deterministic(
            "team_compound",
            pt.stack([tc[0], pt.zeros((n_race, n_team)), tc[1]]),
            dims=("compound", "race", "team"),
        )

        mu_deg = pm.Normal("mu_deg", 0.5, 1.0, dims="compound")
        tau_deg = pm.HalfNormal("tau_deg", 0.5, dims="compound")
        z_deg = pm.Normal("z_deg", 0.0, 1.0, dims=("compound", "race"))
        deg = pm.Deterministic("deg", mu_deg[:, None] + tau_deg[:, None] * z_deg,
                               dims=("compound", "race"))
        sigma_team_deg = pm.HalfNormal("sigma_team_deg", 0.3)
        team_deg = pm.Deterministic(
            "team_deg",
            sigma_team_deg * pm.ZeroSumNormal("z_team_deg", sigma=1.0, dims=("compound", "team")),
            dims=("compound", "team"),
        )
        cliff = pm.Normal("cliff", 0.0, 0.2, dims="compound")
        warmup = pm.Normal("warmup", 0.0, 1.0, dims="compound")
        opening = pm.Normal("opening", 0.0, 1.0)

        mu_trend = pm.Normal("mu_trend", -0.05, 0.05)
        tau_trend = pm.HalfNormal("tau_trend", 0.03)
        z_trend = pm.Normal("z_trend", 0.0, 1.0, dims="race")
        trend = pm.Deterministic("trend", mu_trend + tau_trend * z_trend, dims="race")

        dirty_air_penalty = pm.Normal("dirty_air_penalty", 0.0, 1.0)
        rain_effect = pm.Normal("rain_effect", 0.0, 1.0)
        sigma = pm.HalfNormal("sigma", 1.0, dims="race")
        nu = pm.Gamma("nu", alpha=2.0, beta=0.1)

        mu = (
            race_base[r]
            + driver_offset[dr]
            + compound_offset[c, r]
            + team_compound[c, r, t]
            + (deg[c, r] + team_deg[c, t]) * a
            + cliff[c] * a**2
            + warmup[c] * warm
            + opening * opening_x
            + trend[r] * lap
            + dirty_air_penalty * dirty
            + rain_effect * rain
        )
        pm.StudentT("lap_time", nu=nu, mu=mu, sigma=sigma[r], observed=y)

    meta = {"races": races, "drivers": drivers, "teams": teams,
            "driver_race": pairs, "lap_centre": lap_centre}
    return model, meta


def diagnose(idata: "xr.DataTree", var_names: list[str] | None = None) -> dict[str, float]:
    """R-hat, ESS and divergences over the sampled (non-deterministic) variables."""
    import arviz as az

    summ = az.summary(idata, var_names=var_names)
    return {
        "max_r_hat": float(summ["r_hat"].max()),
        "min_ess_bulk": float(summ["ess_bulk"].min()),
        "min_ess_tail": float(summ["ess_tail"].min()),
        "divergences": int(idata.sample_stats["diverging"].sum()),
    }


def fit(
    laps: pd.DataFrame,
    cutoff: Cutoff | None = None,
    draws: int = 1500,
    tune: int = 1000,
    chains: int = 4,
    seed: int = 0,
    adaptation: str = "low_rank",
) -> BayesFit:
    """Fit the Bayesian hierarchical model with nutpie.

    Args:
        laps: Output of :func:`prepare` (or a training split of it).
        cutoff: Keep only laps known at this decision point (Phase 6).
        draws, tune, chains, seed: Sampler settings.
        adaptation: nutpie mass-matrix adaptation. ``"low_rank"`` adapts to
            posterior correlations (e.g. between linear deg, cliff and trend).

    Returns:
        A :class:`BayesFit` with diagnostics over every sampled variable.
    """
    import nutpie

    laps = apply_cutoff(laps, cutoff)
    model, meta = build_model(laps)
    compiled = nutpie.compile_pymc_model(model)
    idata = nutpie.sample(compiled, draws=draws, tune=tune, chains=chains,
                          seed=seed, progress_bar=False, adaptation=adaptation)
    sampled = [v.name for v in model.free_RVs]
    diag = diagnose(idata, sampled)
    posterior = idata.posterior.to_dataset()[list(KEEP_VARS)].astype("float32")
    return BayesFit(posterior=posterior, diagnostics=diag, races=meta["races"],
                    drivers=meta["drivers"], teams=meta["teams"],
                    driver_race=meta["driver_race"], lap_centre=meta["lap_centre"])


def _flat(post: "xr.Dataset", name: str, idx: np.ndarray) -> np.ndarray:
    """Draws of ``name`` stacked over chains, subset to ``idx``: (n_draws, ...)."""
    arr = post[name].to_numpy()
    return arr.reshape(-1, *arr.shape[2:])[idx].astype(float)


@dataclass
class Prediction:
    """Posterior predictive summary per lap (seconds)."""

    mean: pd.Series
    lo: pd.Series
    hi: pd.Series


def predict(
    fitted: BayesFit,
    laps: pd.DataFrame,
    n_draws: int = 2000,
    interval: float = 0.8,
    seed: int = 0,
) -> Prediction:
    """Posterior predictive mean and central interval for each lap.

    Drivers with no training laps at a race get an offset drawn from the
    pooled distribution; laps at a race not in the fit are NaN.
    """
    rng = np.random.default_rng(seed)
    post = fitted.posterior
    total = post.sizes["chain"] * post.sizes["draw"]
    idx = rng.choice(total, size=min(n_draws, total), replace=False)
    g = {name: _flat(post, name, idx) for name in KEEP_VARS}
    n_s = len(idx)

    r = _index(laps["RoundNumber"].astype(int), fitted.races)
    t = _index(laps["Team"], fitted.teams)
    c = _index(laps["Compound"], list(COMPOUNDS))
    d = _index(laps["Driver"], fitted.drivers)
    dr = _index(pd.Series(list(zip(laps["RoundNumber"].astype(int), laps["Driver"])),
                          index=laps.index), fitted.driver_race)
    ok = (r >= 0) & (t >= 0) & (c >= 0)
    rr, tt, cc = np.where(ok, r, 0), np.where(ok, t, 0), np.where(ok, c, 0)

    # Driver offset: fitted where the pair was seen, else a pooled draw.
    seen = dr >= 0
    driver_part = np.where(d >= 0, g["driver_season"][:, np.where(d >= 0, d, 0)],
                           g["sigma_driver"][:, None] * rng.standard_normal((n_s, len(laps))))
    pooled = (driver_part + g["sigma_driver_race"][:, None]
              * rng.standard_normal((n_s, len(laps))) - g["driver_race_mean"][:, rr])
    offset = np.where(seen, g["driver_offset"][:, np.where(seen, dr, 0)], pooled)

    a = laps["TyreLife"].to_numpy(float) / TYRE_SCALE
    lap = laps["race_lap"].to_numpy(float) - fitted.lap_centre
    mu = (
        g["race_base"][:, rr]
        + offset
        + g["compound_offset"][:, cc, rr]
        + g["team_compound"][:, cc, rr, tt]
        + (g["deg"][:, cc, rr] + g["team_deg"][:, cc, tt]) * a
        + g["cliff"][:, cc] * a**2
        + g["warmup"][:, cc] * laps["first_two_flying"].to_numpy(float)
        + g["opening"][:, None] * laps["opening_laps"].to_numpy(float)
        + g["trend"][:, rr] * lap
        + g["dirty_air_penalty"][:, None] * laps["dirty_air"].to_numpy(float)
        + g["rain_effect"][:, None] * laps["Rainfall"].to_numpy(float)
    )
    noise = g["sigma"][:, rr] * rng.standard_t(g["nu"][:, None], size=mu.shape)
    y_rep = mu + noise
    q = (1 - interval) / 2
    lo, hi = np.quantile(y_rep, [q, 1 - q], axis=0)
    nan = np.where(ok, 1.0, np.nan)
    return Prediction(
        mean=pd.Series(mu.mean(axis=0) * nan, index=laps.index),
        lo=pd.Series(lo * nan, index=laps.index),
        hi=pd.Series(hi * nan, index=laps.index),
    )


def coverage(actual: pd.Series, pred: Prediction) -> float:
    """Share of laps whose actual time lies inside the prediction interval."""
    ok = pred.mean.notna()
    inside = (actual[ok] >= pred.lo[ok]) & (actual[ok] <= pred.hi[ok])
    return float(inside.mean())


def effects_table(fitted: BayesFit, interval: float = 0.8) -> pd.DataFrame:
    """Season-level effects in reporting units (s, s/lap, s/lap^2).

    Returns:
        One row per effect with posterior mean and central interval.
    """
    post = fitted.posterior
    q = (1 - interval) / 2
    scale = {"mu_deg": 1 / TYRE_SCALE, "cliff": 1 / TYRE_SCALE**2}
    units = {"mu_deg": "s/lap", "cliff": "s/lap^2", "mu_trend": "s/lap"}
    rows = []
    for name in ("mu_deg", "cliff", "mu_compound", "warmup", "opening", "mu_trend",
                 "dirty_air_penalty", "rain_effect"):
        da = post[name] * scale.get(name, 1.0)
        extra = [d for d in da.dims if d not in ("chain", "draw")]
        stacked = da.stack(sample=("chain", "draw"))
        labels = stacked[extra[0]].to_numpy() if extra else [None]
        for lab in labels:
            v = stacked.sel({extra[0]: lab}).to_numpy() if extra else stacked.to_numpy()
            rows.append({"effect": name, "level": lab, "unit": units.get(name, "s"),
                         "mean": float(v.mean()), "lo": float(np.quantile(v, q)),
                         "hi": float(np.quantile(v, 1 - q))})
    return pd.DataFrame(rows)


# --- XGBoost ------------------------------------------------------------------

XGB_NUMERIC = ["TyreLife", "race_lap", "dirty_air", "Rainfall", "first_two_flying",
               "opening_laps"]
XGB_CATEGORICAL = ["RoundNumber", "Driver", "Team", "Compound"]


def _xgb_features(laps: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """One-hot categoricals plus numeric features, aligned to ``columns`` if given."""
    x = pd.get_dummies(laps[XGB_CATEGORICAL].astype(str), dtype=float)
    x = pd.concat([x, laps[XGB_NUMERIC].astype(float)], axis=1)
    return x if columns is None else x.reindex(columns=columns, fill_value=0.0)


@dataclass
class XGBFit:
    """A fitted gradient-boosted tree model and its feature columns."""

    model: object
    columns: list[str]


def fit_xgboost(laps: pd.DataFrame, cutoff: Cutoff | None = None, seed: int = 0) -> XGBFit:
    """Fit XGBoost on the same features the Bayesian model uses."""
    from xgboost import XGBRegressor

    laps = apply_cutoff(laps, cutoff)
    x = _xgb_features(laps)
    model = XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, n_jobs=4, random_state=seed)
    model.fit(x, laps["lap_s"])
    return XGBFit(model=model, columns=list(x.columns))


def predict_xgboost(fitted: XGBFit, laps: pd.DataFrame) -> pd.Series:
    """Predicted lap time in seconds."""
    return pd.Series(fitted.model.predict(_xgb_features(laps, fitted.columns)), index=laps.index)


# --- Comparison and checks ----------------------------------------------------


def comparison_table(laps: pd.DataFrame, bayes_fits: dict[str, BayesFit]) -> pd.DataFrame:
    """MAE, RMSE and 80% coverage for baseline, XGBoost and Bayesian on each split.

    ``common`` rows use only laps the baseline can predict, so all three models
    are scored on the same laps; ``all`` rows use every test lap (the baseline
    is not scored there).

    Args:
        laps: Output of :func:`prepare`.
        bayes_fits: Bayesian fit on each split's training data, keyed by split.
    """
    rows = []
    for name, split in SPLITS.items():
        train, test = split(laps)
        preds = {
            "baseline": predict_baseline(fit_baseline(train), test),
            "xgboost": predict_xgboost(fit_xgboost(train), test),
        }
        bayes = predict(bayes_fits[name], test)
        preds["bayes"] = bayes.mean
        common = preds["baseline"].notna()
        for subset, mask in (("common", common), ("all", pd.Series(True, index=test.index))):
            for model, pred in preds.items():
                if model == "baseline" and subset == "all":
                    continue
                row = {"split": name, "subset": subset, "model": model,
                       **evaluate(test.loc[mask, "lap_s"], pred[mask])}
                if model == "bayes":
                    sub = Prediction(bayes.mean[mask], bayes.lo[mask], bayes.hi[mask])
                    row["coverage_80"] = coverage(test.loc[mask, "lap_s"], sub)
                rows.append(row)
    return pd.DataFrame(rows)


def _draws(fitted: BayesFit, name: str) -> "xr.DataArray":
    return fitted.posterior[name].stack(sample=("chain", "draw"))


def deg_by_race(fitted: BayesFit, interval: float = 0.8) -> pd.DataFrame:
    """Field-average linear deg per compound and race in s/lap, with flags.

    Flags (reported, never forced): ``negative`` if the posterior mean is below
    zero, and ``hard_faster_than_soft`` if hard's mean deg exceeds soft's at
    that race. ``p_*`` columns give the posterior probability of each.
    """
    deg = _draws(fitted, "deg") / TYRE_SCALE
    q = (1 - interval) / 2
    rows = []
    for rnd in fitted.races:
        d = deg.sel(race=rnd)
        soft, hard = d.sel(compound="SOFT").to_numpy(), d.sel(compound="HARD").to_numpy()
        for comp in COMPOUNDS:
            v = d.sel(compound=comp).to_numpy()
            rows.append({"RoundNumber": rnd, "compound": comp, "mean": float(v.mean()),
                         "lo": float(np.quantile(v, q)), "hi": float(np.quantile(v, 1 - q)),
                         "p_negative": float((v < 0).mean()),
                         "p_hard_gt_soft": float((hard > soft).mean())})
    out = pd.DataFrame(rows)
    out["negative"] = out["mean"] < 0
    hard_mean = out[out["compound"] == "HARD"].set_index("RoundNumber")["mean"]
    soft_mean = out[out["compound"] == "SOFT"].set_index("RoundNumber")["mean"]
    out["hard_faster_than_soft"] = out["RoundNumber"].map(hard_mean > soft_mean)
    return out


def deg_curve(
    fitted: BayesFit,
    round_number: int,
    compound: str,
    ages: np.ndarray,
    team: str | None = None,
    interval: float = 0.8,
) -> pd.DataFrame:
    """Tyre-only lap time change vs a new medium, by tyre age, in seconds.

    ``compound_offset + deg * age + cliff * age^2`` for the field-average car
    (or, with ``team``, including that team's compound and deg terms). Fuel
    and track trend are not included, so this is the tyre effect alone.
    Returns ``TyreLife``, ``mean``, ``lo`` and ``hi`` (credible band of the
    curve, not of single laps).
    """
    q = (1 - interval) / 2
    a = np.asarray(ages, dtype=float) / TYRE_SCALE
    off = _draws(fitted, "compound_offset").sel(compound=compound, race=round_number).to_numpy()
    deg = _draws(fitted, "deg").sel(compound=compound, race=round_number).to_numpy()
    cliff = _draws(fitted, "cliff").sel(compound=compound).to_numpy()
    if team is not None:
        off = off + _draws(fitted, "team_compound").sel(
            compound=compound, race=round_number, team=team).to_numpy()
        deg = deg + _draws(fitted, "team_deg").sel(compound=compound, team=team).to_numpy()
    curve = off[None, :] + deg[None, :] * a[:, None] + cliff[None, :] * a[:, None] ** 2
    return pd.DataFrame({"TyreLife": ages, "mean": curve.mean(axis=1),
                         "lo": np.quantile(curve, q, axis=1),
                         "hi": np.quantile(curve, 1 - q, axis=1)})


def season_deg_table(fitted: BayesFit, interval: float = 0.8) -> pd.DataFrame:
    """Season deg rate (``mu_deg``) and curvature (``cliff``) per compound.

    Also gives the total extra time after 10 and 20 laps of tyre age for the
    field-average car (linear + cliff), in seconds.
    """
    q = (1 - interval) / 2
    mu = _draws(fitted, "mu_deg") / TYRE_SCALE
    cliff = _draws(fitted, "cliff") / TYRE_SCALE**2
    rows = []
    for comp in COMPOUNDS:
        m, c = mu.sel(compound=comp).to_numpy(), cliff.sel(compound=comp).to_numpy()
        row = {"compound": comp}
        for label, v in (("deg_s_per_lap", m), ("cliff_s_per_lap2", c),
                         ("loss_10_laps_s", 10 * m + 100 * c), ("loss_20_laps_s", 20 * m + 400 * c)):
            row[label] = float(v.mean())
            row[f"{label}_lo"] = float(np.quantile(v, q))
            row[f"{label}_hi"] = float(np.quantile(v, 1 - q))
        rows.append(row)
    return pd.DataFrame(rows)


def team_compound_summary(
    fitted: BayesFit, team: str, round_number: int, interval: float = 0.8
) -> pd.DataFrame:
    """A team's compound terms at one race, relative to the medium, in seconds.

    Rows per compound: ``team_compound`` (the team's deviation from the field
    at that race), ``field_compound`` (the field's offset), their sum ``total``
    (the team's own hard-vs-medium or soft-vs-medium offset at tyre age 0), and
    the season-level ``team_deg`` in s/lap.
    """
    q = (1 - interval) / 2
    tc = _draws(fitted, "team_compound").sel(team=team, race=round_number)
    field = _draws(fitted, "compound_offset").sel(race=round_number)
    tdeg = _draws(fitted, "team_deg").sel(team=team) / TYRE_SCALE
    rows = []
    for comp in COMPOUNDS:
        values = {"team_compound": tc.sel(compound=comp).to_numpy(),
                  "field_compound": field.sel(compound=comp).to_numpy()}
        values["total"] = values["team_compound"] + values["field_compound"]
        values["team_deg_s_per_lap"] = tdeg.sel(compound=comp).to_numpy()
        for term, v in values.items():
            rows.append({"team": team, "RoundNumber": round_number, "compound": comp,
                         "term": term, "mean": float(v.mean()),
                         "lo": float(np.quantile(v, q)), "hi": float(np.quantile(v, 1 - q))})
    return pd.DataFrame(rows)


# --- Saving -------------------------------------------------------------------

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def save_fit(fitted: BayesFit, path: Path) -> Path:
    """Write a fit to compressed netCDF, with its index metadata as attributes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = fitted.posterior.copy()
    ds.attrs = {
        "races": json.dumps(fitted.races), "drivers": json.dumps(fitted.drivers),
        "teams": json.dumps(fitted.teams),
        "driver_race": json.dumps([[r, d] for r, d in fitted.driver_race]),
        "lap_centre": fitted.lap_centre, "diagnostics": json.dumps(fitted.diagnostics),
    }
    ds.to_netcdf(path, engine="h5netcdf",
                 encoding={v: {"zlib": True, "complevel": 5} for v in ds.data_vars})
    return path


def load_fit(path: Path) -> BayesFit:
    """Read a fit written by :func:`save_fit`."""
    import xarray as xr

    ds = xr.load_dataset(path, engine="h5netcdf")
    a = ds.attrs
    return BayesFit(
        posterior=ds, races=json.loads(a["races"]), drivers=json.loads(a["drivers"]),
        teams=json.loads(a["teams"]),
        driver_race=[(int(r), d) for r, d in json.loads(a["driver_race"])],
        lap_centre=float(a["lap_centre"]), diagnostics=json.loads(a["diagnostics"]),
    )


FULL_FIT_PATH = MODELS_DIR / "deg_full_2022.nc"


def split_fit_path(split: str) -> Path:
    """Where the fit on a split's training data is saved."""
    return MODELS_DIR / f"deg_{split}_2022.nc"


def main(argv: list[str] | None = None) -> None:
    """Fit and save models. ``full`` (default): all 2022 data; ``splits``: the
    training data of each held-out split; ``all``: both."""
    import sys
    import time

    from f1strat.clean import clean_laps
    from f1strat.data import load_season

    target = (argv if argv is not None else sys.argv[1:] or ["full"])[0]
    laps = prepare(clean_laps(load_season(2022)))
    jobs: list[tuple[str, pd.DataFrame, Path]] = []
    if target in ("full", "all"):
        jobs.append(("full", laps, FULL_FIT_PATH))
    if target in ("splits", "all"):
        for name, split in SPLITS.items():
            jobs.append((name, split(laps)[0], split_fit_path(name)))
    for name, train, path in jobs:
        start = time.time()
        fitted = fit(train)
        minutes = (time.time() - start) / 60
        print(f"{name}: {len(train)} laps, {minutes:.1f} min, {fitted.diagnostics}", flush=True)
        print(f"wrote {save_fit(fitted, path)}", flush=True)


if __name__ == "__main__":
    main()
