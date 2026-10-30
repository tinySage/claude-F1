"""Tests for the Phase 1 data tables: period logic and saved-parquet schema."""

import pandas as pd
import pytest

from f1strat.data import (
    LAPS_SCHEMA,
    RACE_SCHEMA,
    load_season,
    neutralisation_periods,
)

N_RACES_2022 = 22


def _status(rows: list[tuple[int, str]]) -> pd.DataFrame:
    """Build a track status timeline from (seconds, code) pairs."""
    return pd.DataFrame(
        {
            "Time": [pd.Timedelta(seconds=t) for t, _ in rows],
            "Status": [code for _, code in rows],
        }
    )


def test_periods_from_track_status() -> None:
    """VSC deployed + ending is one VSC period; a VSC turning into a SC splits."""
    ts = _status([(0, "1"), (10, "6"), (20, "7"), (30, "1"), (40, "6"), (50, "4"), (60, "5"), (70, "1")])
    periods = neutralisation_periods(ts)
    got = {(k, p): (s.seconds, e.seconds) for k, p, s, e in periods.itertuples(index=False)}
    assert got == {
        ("VSC", 1): (10, 30),
        ("VSC", 2): (40, 50),
        ("SafetyCar", 1): (50, 60),
        ("RedFlag", 1): (60, 70),
    }


def test_periods_before_race_start_are_dropped_and_renumbered() -> None:
    """A pre-start red flag is dropped, so the in-race one becomes period 1."""
    ts = _status([(0, "1"), (5, "5"), (20, "1"), (40, "5"), (50, "1")])
    periods = neutralisation_periods(ts, after=pd.Timedelta(seconds=30))
    assert periods[["kind", "period"]].values.tolist() == [["RedFlag", 1]]
    assert periods["start"].iloc[0] == pd.Timedelta(seconds=40)


def test_period_open_at_end_of_timeline() -> None:
    """A period with no closing status runs to Timedelta.max."""
    periods = neutralisation_periods(_status([(0, "1"), (10, "4")]))
    assert periods["end"].iloc[0] == pd.Timedelta.max


@pytest.fixture(scope="module")
def laps_2022() -> pd.DataFrame:
    """Combined 2022 laps table from the committed parquet files."""
    return load_season(2022)


@pytest.fixture(scope="module")
def race_2022() -> pd.DataFrame:
    """Combined 2022 race table from the committed parquet files."""
    return load_season(2022, table="race")


@pytest.mark.parametrize(
    ("table", "schema"), [("laps_2022", LAPS_SCHEMA), ("race_2022", RACE_SCHEMA)]
)
def test_schema(table: str, schema: dict[str, str], request: pytest.FixtureRequest) -> None:
    """Saved tables have exactly the documented columns and dtypes."""
    df = request.getfixturevalue(table)
    assert list(df.columns) == list(schema)
    assert {c: str(t) for c, t in df.dtypes.items()} == schema


def test_all_races_present(laps_2022: pd.DataFrame, race_2022: pd.DataFrame) -> None:
    """All 22 rounds are in both tables."""
    expected = set(range(1, N_RACES_2022 + 1))
    assert set(laps_2022["RoundNumber"]) == expected
    assert set(race_2022["RoundNumber"]) == expected


def test_keys_are_unique(laps_2022: pd.DataFrame, race_2022: pd.DataFrame) -> None:
    """One row per (round, driver, lap) and per (round, lap)."""
    assert not laps_2022.duplicated(["RoundNumber", "Driver", "LapNumber"]).any()
    assert not race_2022.duplicated(["RoundNumber", "LapNumber"]).any()


def test_wet_compounds_kept(laps_2022: pd.DataFrame) -> None:
    """Wet-weather compounds are not filtered out."""
    assert {"INTERMEDIATE", "WET"} <= set(laps_2022["Compound"].dropna())
