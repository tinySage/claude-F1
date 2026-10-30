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


def test_post_stop_counts_from_the_stop_not_tyre_age(model_laps: pd.DataFrame) -> None:
    """first_two_flying uses laps since the stop; a used set starts above TyreLife 1."""
    warm = model_laps[model_laps["first_two_flying"]]
    assert set(warm["laps_since_stop"]) == {1, 2}
    assert (warm["TyreLife"] > 2).any()


def test_last_stint_split(model_laps: pd.DataFrame) -> None:
    """Test = final stints of odd rounds; even rounds keep final stints in training."""
    train, test = split_last_stint(model_laps)
    assert test["final_stint"].all()
    assert (test["RoundNumber"].astype(int) % 2 == 1).all()
    train_finals = train[train["final_stint"]]
    assert len(train_finals) > 0
    assert (train_finals["RoundNumber"].astype(int) % 2 == 0).all()
    tested = set(zip(test["RoundNumber"], test["Driver"]))
    earlier = train[~train["final_stint"]]
    assert tested <= set(zip(earlier["RoundNumber"], earlier["Driver"]))


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


def _synthetic_season(rng: np.random.Generator) -> pd.DataFrame:
    """Five races, six drivers in three teams, three compounds, staggered stops.

    True field deg: soft 0.10, medium 0.06, hard 0.03 s/lap; trend −0.05 s/lap.
    The final (hard) stint is 0.3 s slower on average.
    Includes small team-specific hard offsets and per-stint shifts, as in real
    data (zero-variance groups make the sampler's job artificially hard).
    """
    team_hard = {"T1": 0.2, "T2": -0.1, "T3": -0.1}
    deg = {"SOFT": 0.10, "MEDIUM": 0.06, "HARD": 0.03}
    offset = {"SOFT": -0.4, "MEDIUM": 0.0, "HARD": 0.4}
    rows = []
    drivers = [("AAA", "T1", -0.3), ("BBB", "T1", -0.1), ("CCC", "T2", 0.1),
               ("DDD", "T2", 0.3), ("EEE", "T3", 0.0), ("FFF", "T3", 0.2)]
    for rnd, base in [(1, 90.0), (2, 80.0), (3, 100.0), (4, 85.0), (5, 95.0)]:
        for i, (drv, team, pace) in enumerate(drivers):
            plan = [("SOFT", 10 + 2 * i), ("MEDIUM", 30 + 2 * i), ("HARD", 51)]
            start = 2
            for stint, (comp, end) in enumerate(plan, start=1):
                shift = rng.normal(0, 0.15) + (team_hard[team] + 0.3 if comp == "HARD" else 0.0)
                for lap in range(start, end):
                    age = lap - start + 1
                    y = (base + pace + offset[comp] + deg[comp] * age - 0.05 * lap
                         + shift + rng.normal(0, 0.2))
                    rows.append({
                        "RoundNumber": rnd, "Driver": drv, "Team": team, "Compound": comp,
                        "Stint": stint, "TyreLife": age, "race_lap": lap, "LapNumber": lap,
                        "dirty_air": False, "Rainfall": False, "opening_laps": lap in (2, 3),
                        "first_two_flying": stint > 1 and age in (1, 2), "lap_s": y,
                        "final_stint": stint == len(plan),
                    })
                start = end + 1
    return pd.DataFrame(rows)


def test_bayesian_fit_recovers_deg_and_predicts() -> None:
    """A short fit on synthetic data recovers deg ordering and trend; predict works."""
    from f1strat.deg_model import fit, predict, season_deg_table

    df = _synthetic_season(np.random.default_rng(1))
    fitted = fit(df, draws=300, tune=300, chains=2, seed=1)
    # No divergence check here: the toy season has identical deg and trend in
    # every race, so several race-level scales are truly zero and the sampler
    # diverges near that boundary. Divergences are checked on the real fits.
    season = season_deg_table(fitted).set_index("compound")["deg_s_per_lap"]
    assert season["SOFT"] > season["MEDIUM"] > season["HARD"]
    trend = float(fitted.posterior["mu_trend"].mean())
    assert trend == pytest.approx(-0.05, abs=0.02)
    pred = predict(fitted, df, n_draws=200)
    assert pred.mean.notna().all()
    assert (pred.lo < pred.mean).all() and (pred.mean < pred.hi).all()


def test_unseen_stint_widens_interval() -> None:
    """Holding out a stint: its laps get a stint draw, so a wider interval."""
    from f1strat.deg_model import fit, predict

    df = _synthetic_season(np.random.default_rng(2))
    held = (df["RoundNumber"] == 1) & (df["Driver"] == "AAA") & (df["Stint"] == 3)
    fitted = fit(df[~held], draws=200, tune=200, chains=2, seed=2)
    assert (1, "AAA", 3) not in fitted.stints
    seen = predict(fitted, df[(df["RoundNumber"] == 1) & (df["Driver"] == "AAA") & (df["Stint"] == 2)])
    unseen = predict(fitted, df[held])
    assert (unseen.hi - unseen.lo).mean() > (seen.hi - seen.lo).mean()


def test_cutoff_inputs_use_no_future_laps() -> None:
    """Phase 6: cleaning for a cutoff refit must not change when laps after the
    cutoff change, and no stint of the decision race is labelled final."""
    from f1strat.clean import pit_stops
    from f1strat.data import load_season
    from f1strat.deg_model import Cutoff, cutoff_inputs

    raw = load_season(2022)
    raw = raw[raw["RoundNumber"].isin([5, 6])]
    cut = Cutoff(6, 30)
    cleaned, laps = cutoff_inputs(raw, cut)
    # Make every later Spain lap absurdly slow: nothing known at lap 30 moves.
    later = (raw["RoundNumber"] == 6) & (raw["LapNumber"] > 30)
    raw2 = raw.copy()
    raw2.loc[later, "LapTime"] = raw2.loc[later, "LapTime"] * 3
    cleaned2, laps2 = cutoff_inputs(raw2, cut)
    assert cleaned["LapNumber"][cleaned["RoundNumber"] == 6].max() == 30
    assert cleaned["exclude_reason"].astype(str).equals(cleaned2["exclude_reason"].astype(str))
    s1, s2 = pit_stops(cleaned), pit_stops(cleaned2)
    assert np.allclose(s1["loss_s"].fillna(0), s2["loss_s"].fillna(0))
    assert (s1.loc[s1["RoundNumber"] == 6, "InLap"] < 30).all()
    assert not laps.loc[laps["RoundNumber"] == 6, "final_stint"].any()
    assert laps.loc[laps["RoundNumber"] == 5, "final_stint"].any()
    assert len(laps) == len(laps2)


def test_practice_long_runs_join_the_model() -> None:
    """Practice laps add their own likelihood and share the race's tyre terms;
    without them the model is unchanged."""
    from f1strat.data import load_season
    from f1strat.deg_model import Cutoff, build_model, cutoff_inputs

    raw = load_season(2022)
    raw = raw[raw["RoundNumber"].isin([5, 6])]
    _, laps = cutoff_inputs(raw, Cutoff(6, 20))
    practice = pd.DataFrame({
        "RoundNumber": 6, "session": "FP2", "Driver": "VER", "Team": "Red Bull Racing",
        "Compound": "HARD", "TyreLife": np.arange(2, 10), "lap_s": 81.0 + 0.1 * np.arange(8),
        "run": "FP2_VER_1", "lap_in_run": np.arange(8)})
    base, _ = build_model(laps)
    with_p, meta = build_model(laps, practice)
    assert "practice_lap_time" in with_p.named_vars and "practice_lap_time" not in base.named_vars
    assert {"practice_level", "fuel_practice", "z_run"} <= set(with_p.named_vars)
    assert meta["practice_laps"] == 8
    assert np.isfinite(with_p.point_logps()["practice_lap_time"])
