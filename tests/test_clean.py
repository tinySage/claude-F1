"""Tests for Phase 2 cleaning: exclude reasons, gaps and pit loss."""

import pandas as pd
import pytest

from f1strat.clean import (
    EXCLUDE_REASONS,
    MIXED_CONDITIONS_2022,
    clean_laps,
    pit_loss,
)
from f1strat.data import LAPS_SCHEMA, PROCESSED_DIR, load_season


def _lap(driver: str, lap: int, pos: int, end_s: float, **kw: object) -> dict:
    """One synthetic lap row with sensible green-flag defaults."""
    row = {
        "RoundNumber": 1, "EventName": "Test GP", "Driver": driver, "Team": "T",
        "LapNumber": lap, "LapTime": pd.Timedelta(seconds=90), "Stint": 1,
        "Compound": "MEDIUM", "TyreLife": lap, "FreshTyre": True,
        "PitInTime": None, "PitOutTime": None, "TrackStatus": "1",
        "Position": pos, "Deleted": False, "IsAccurate": True,
        "Time": pd.Timedelta(seconds=end_s),
        "LapStartTime": pd.Timedelta(seconds=end_s - 90),
        "TrackTemp": 30.0, "Rainfall": False,
    }
    row.update(kw)
    return row


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)[list(LAPS_SCHEMA)].astype(LAPS_SCHEMA)


def test_first_matching_reason_wins() -> None:
    """A lap under SC that is also deleted is labelled 'sc', not 'deleted'."""
    rows = [_lap("AAA", n, 1, 90.0 * n) for n in range(1, 8)]
    rows[2] = _lap("AAA", 3, 1, 270.0, TrackStatus="14", Deleted=True)
    rows[4] = _lap("AAA", 5, 1, 450.0, LapTime=pd.Timedelta(seconds=100))
    got = clean_laps(_frame(rows))["exclude_reason"].astype("object").tolist()
    assert got[0] == "lap_1"
    assert got[2] == "sc"
    assert got[3] == "restart_lap"
    assert got[4] == "outlier"  # 100 s > 1.07 x 90 s median
    assert pd.isna(got[1]) and pd.isna(got[5]) and pd.isna(got[6])


def test_rainfall_alone_does_not_exclude() -> None:
    """Rainfall is a covariate, not an exclude reason."""
    rows = [_lap("AAA", n, 1, 90.0 * n, Rainfall=n >= 3) for n in range(1, 6)]
    out = clean_laps(_frame(rows))
    assert out.loc[out["Rainfall"], "exclude_reason"].isna().all()


def test_mixed_conditions_labels_whole_race() -> None:
    """Every lap of a mixed-conditions race gets a reason, after wet_compound."""
    rows = [_lap("AAA", n, 1, 90.0 * n) for n in range(1, 6)]
    rows[3] = _lap("AAA", 4, 1, 360.0, Compound="INTERMEDIATE")
    got = clean_laps(_frame(rows), mixed_conditions={"Test GP"})["exclude_reason"]
    assert got.astype("object").tolist() == [
        "lap_1", "mixed_conditions", "mixed_conditions", "wet_compound", "mixed_conditions"
    ]


def test_gap_ahead_and_dirty_air() -> None:
    """Gap is measured to the car one position ahead on the same lap."""
    rows = [
        _lap("AAA", 2, 1, 180.0),
        _lap("BBB", 2, 2, 180.6),
        _lap("CCC", 2, 3, 183.0),
    ]
    out = clean_laps(_frame(rows))
    assert pd.isna(out["gap_ahead_s"].iloc[0])
    assert out["gap_ahead_s"].iloc[1:].round(3).tolist() == [0.6, 2.4]
    assert out["dirty_air"].tolist() == [False, True, False]


@pytest.fixture(scope="module")
def cleaned_2022() -> pd.DataFrame:
    """2022 season laps with cleaning columns."""
    return clean_laps(load_season(2022))


def test_one_known_reason_per_lap(cleaned_2022: pd.DataFrame) -> None:
    """exclude_reason is a single value from the fixed list, or null."""
    reasons = cleaned_2022["exclude_reason"]
    assert len(reasons) == len(cleaned_2022)
    assert set(reasons.dropna().astype(str)) <= set(EXCLUDE_REASONS)


def test_clean_laps_are_dry_known_compound_green(cleaned_2022: pd.DataFrame) -> None:
    """No clean lap is on a wet or missing compound, under SC/VSC/red flag, or
    from a mixed-conditions race."""
    clean = cleaned_2022[cleaned_2022["exclude_reason"].isna()]
    assert len(clean) > 0
    assert clean["Compound"].isin(["SOFT", "MEDIUM", "HARD"]).all()
    assert not clean["TrackStatus"].str.contains("[4567]", regex=True).any()
    assert not clean["EventName"].isin(list(MIXED_CONDITIONS_2022)).any()


def test_rainfall_laps_kept_in_hungary(cleaned_2022: pd.DataFrame) -> None:
    """Hungary's drizzle laps on slicks are clean unless another reason applies."""
    hun = cleaned_2022[cleaned_2022["EventName"] == "Hungarian Grand Prix"]
    assert hun.loc[hun["Rainfall"], "exclude_reason"].isna().any()


def test_dirty_air_never_excludes(cleaned_2022: pd.DataFrame) -> None:
    """Dirty-air laps still appear among clean laps."""
    clean = cleaned_2022[cleaned_2022["exclude_reason"].isna()]
    assert clean["dirty_air"].any()


@pytest.fixture(scope="module")
def saved_pit_loss() -> pd.DataFrame:
    """The committed pit loss table."""
    return pd.read_parquet(PROCESSED_DIR / "2022" / "pit_loss.parquet")


def test_pit_loss_positive(saved_pit_loss: pd.DataFrame) -> None:
    """All pit loss values present are positive; one row per 2022 race."""
    assert len(saved_pit_loss) == 22
    assert (saved_pit_loss["pit_loss_s"].dropna() > 0).all()


def test_pit_loss_sources(saved_pit_loss: pd.DataFrame) -> None:
    """Mixed-conditions circuits never use their own 2022 stops; every other
    race does. A missing value is only allowed for a mixed-conditions race."""
    mixed = saved_pit_loss["EventName"].isin(list(MIXED_CONDITIONS_2022))
    assert (saved_pit_loss.loc[~mixed, "source_year"] == 2022).all()
    assert (saved_pit_loss.loc[mixed, "source_year"].fillna(0) != 2022).all()
    assert saved_pit_loss.loc[saved_pit_loss["pit_loss_s"].isna(), "EventName"].isin(
        list(MIXED_CONDITIONS_2022)
    ).all()


def test_saved_pit_loss_matches_2022_recompute(
    cleaned_2022: pd.DataFrame, saved_pit_loss: pd.DataFrame
) -> None:
    """Rows sourced from 2022 match a fresh computation (no network needed)."""
    own = pit_loss(cleaned_2022, load_season(2022, table="race"))
    saved = saved_pit_loss[saved_pit_loss["source_year"] == 2022]
    pd.testing.assert_frame_equal(
        saved.drop(columns="source_year").reset_index(drop=True),
        own.reset_index(drop=True),
        check_dtype=False,
    )
