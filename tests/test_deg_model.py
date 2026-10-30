"""Tests for Phase 3 data preparation, held-out splits, cutoff and baseline."""

import numpy as np
import pandas as pd
import pytest

from f1strat.clean import clean_laps
from f1strat.data import load_season
from f1strat.deg_model import (
    LONG_TYRE_MAX_TRAIN_AGE,
    Cutoff,
    apply_cutoff,
    evaluate,
    fit_baseline,
    predict_baseline,
    prepare,
    split_last_stint,
    split_long_tyre,
)


@pytest.fixture(scope="module")
def model_laps() -> pd.DataFrame:
    """Prepared 2022 clean laps."""
    return prepare(clean_laps(load_season(2022)))


def test_prepare_keeps_only_clean_laps_with_tyre_age(model_laps: pd.DataFrame) -> None:
    assert model_laps["exclude_reason"].isna().all()
    assert model_laps["TyreLife"].notna().all()
    assert model_laps["lap_s"].gt(0).all()


def test_warmup_counts_from_the_stop_not_tyre_age(model_laps: pd.DataFrame) -> None:
    """first_two_flying uses laps since the stop; a used set starts above TyreLife 1."""
    warm = model_laps[model_laps["first_two_flying"]]
    assert set(warm["laps_since_stop"]) == {1, 2}
    assert (warm["TyreLife"] > 2).any()


def test_last_stint_split(model_laps: pd.DataFrame) -> None:
    """Test = final stints only, for drivers who also have training laps in that race."""
    train, test = split_last_stint(model_laps)
    assert test["final_stint"].all() and not train["final_stint"].any()
    tested = set(zip(test["RoundNumber"], test["Driver"]))
    trained = set(zip(train["RoundNumber"], train["Driver"]))
    assert tested <= trained


def test_long_tyre_split(model_laps: pd.DataFrame) -> None:
    train, test = split_long_tyre(model_laps)
    assert train["TyreLife"].max() <= LONG_TYRE_MAX_TRAIN_AGE
    assert test["TyreLife"].min() > LONG_TYRE_MAX_TRAIN_AGE
    assert len(train) + len(test) == len(model_laps)


def test_cutoff_drops_future_laps(model_laps: pd.DataFrame) -> None:
    """Earlier rounds kept in full; the cutoff race only up to the decision lap."""
    kept = apply_cutoff(model_laps, Cutoff(round_number=13, lap_number=39))
    assert kept["RoundNumber"].max() == 13
    assert kept.loc[kept["RoundNumber"] == 13, "LapNumber"].max() <= 39
    earlier = model_laps["RoundNumber"] < 13
    assert (kept["RoundNumber"] < 13).sum() == earlier.sum()
    assert apply_cutoff(model_laps, None) is model_laps


def _synthetic_race(rng: np.random.Generator) -> pd.DataFrame:
    """Two drivers, medium then hard, stopping on different laps; no noise.

    Staggered stops matter: if both stopped on the same lap, race_lap would be
    an exact linear combination of the per-compound tyre-age columns and the
    fuel/deg split would not be identified.
    """
    rows = []
    for driver, offset, stop in [("AAA", 0.0, 15), ("BBB", 0.5, 25)]:
        for lap in range(2, 41):
            stint = 1 if lap <= stop else 2
            compound = "MEDIUM" if stint == 1 else "HARD"
            age = lap - 1 if stint == 1 else lap - stop
            deg = 0.08 if compound == "MEDIUM" else 0.04
            hard = 0.6 if compound == "HARD" else 0.0
            dirty = bool(rng.random() < 0.2)
            lap_s = 90 + offset + hard + deg * age - 0.05 * lap + 0.3 * dirty
            rows.append({"RoundNumber": 1, "Driver": driver, "Compound": compound,
                         "TyreLife": age, "race_lap": lap, "dirty_air": dirty, "lap_s": lap_s})
    return pd.DataFrame(rows)


def test_baseline_recovers_known_effects() -> None:
    df = _synthetic_race(np.random.default_rng(0))
    coef = fit_baseline(df).races[1].coef
    assert coef["deg[MEDIUM]"] == pytest.approx(0.08, abs=1e-6)
    assert coef["deg[HARD]"] == pytest.approx(0.04, abs=1e-6)
    assert coef["race_lap"] == pytest.approx(-0.05, abs=1e-6)
    assert coef["dirty_air"] == pytest.approx(0.3, abs=1e-6)
    assert evaluate(df["lap_s"], predict_baseline(fit_baseline(df), df))["mae_s"] < 1e-6


def test_baseline_cannot_predict_unseen_compound() -> None:
    df = _synthetic_race(np.random.default_rng(0))
    model = fit_baseline(df[df["Compound"] == "MEDIUM"])
    pred = predict_baseline(model, df)
    assert pred[df["Compound"] == "HARD"].isna().all()
    assert pred[df["Compound"] == "MEDIUM"].notna().all()
