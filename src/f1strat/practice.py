"""Practice long runs for cutoff fits (Phase 5 Part C; ``docs/model.md``,
"Practice long runs in cutoff fits").

A pit wall at a decision lap knows the race so far **and** the Friday and
Saturday long runs on every compound. :func:`long_runs` extracts those runs
from FP1–FP3 of a race; :func:`f1strat.deg_model.fit` takes them through its
``practice`` argument.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from f1strat.clean import OUTLIER_RATIO
from f1strat.data import race_paths

SESSIONS: tuple[str, ...] = ("FP1", "FP2", "FP3")
MIN_RUN_LAPS = 6
DRY = ("SOFT", "MEDIUM", "HARD")
NEUTRAL = "4567"

log = logging.getLogger(__name__)


def practice_path(year: int, round_number: int, event_name: str) -> Path:
    """``data/processed/<year>/<RR>_<event>_practice_long_runs.parquet``."""
    laps_path, _ = race_paths(year, round_number, event_name)
    return laps_path.with_name(
        laps_path.name.replace("_laps.parquet", "_practice_long_runs.parquet")
    )


def _session_laps(year: int, round_number: int, name: str) -> pd.DataFrame:
    import fastf1

    from f1strat.data import configure_cache

    configure_cache()
    s = fastf1.get_session(year, round_number, name)
    s.load(laps=True, telemetry=False, weather=True, messages=False)
    laps = pd.DataFrame(s.laps).copy()
    try:
        weather = s.laps.get_weather_data()
        laps["Rainfall"] = weather["Rainfall"].to_numpy(bool)
    except Exception:  # noqa: BLE001 - missing weather is not a reason to fail
        laps["Rainfall"] = False
    laps["session"] = name
    return laps


def long_runs_from_laps(
    laps: pd.DataFrame, race_drivers: set[str], min_laps: int = MIN_RUN_LAPS
) -> pd.DataFrame:
    """Clean practice laps and keep runs of ``min_laps``+ consecutive clean laps.

    Args:
        laps: FastF1 practice laps with ``session`` and ``Rainfall`` columns.
        race_drivers: Drivers who started the race (stand-ins are dropped).
        min_laps: Shortest long run.

    Returns:
        One row per lap in a long run: ``session``, ``Driver``, ``Team``,
        ``Compound``, ``TyreLife``, ``LapNumber``, ``lap_s``, ``run`` (a
        label unique within the race) and ``lap_in_run`` (0 for its first lap).
    """
    df = laps[laps["Driver"].isin(race_drivers)].copy()
    df["lap_s"] = df["LapTime"].dt.total_seconds()
    status = df["TrackStatus"].fillna("").astype(str)
    ok = (
        df["PitInTime"].isna()
        & df["PitOutTime"].isna()
        & ~df["Deleted"].fillna(False).astype(bool)
        & df["IsAccurate"].fillna(False).astype(bool)
        & df["lap_s"].notna()
        & ~status.str.contains(f"[{NEUTRAL}]")
        & df["Compound"].isin(DRY)
        & ~df["Rainfall"].fillna(False).astype(bool)
        & df["TyreLife"].notna()
    )
    df = df[ok].copy()
    median = df.groupby(["session", "Driver"])["lap_s"].transform("median")
    df = df[df["lap_s"] <= OUTLIER_RATIO * median].copy()
    df = df.sort_values(["session", "Driver", "Stint", "LapNumber"])
    # A run: consecutive lap numbers within one stint (one set).
    gap = df.groupby(["session", "Driver", "Stint"])["LapNumber"].diff().ne(1)
    df["block"] = gap.cumsum()
    size = df.groupby("block")["LapNumber"].transform("size")
    df = df[size >= min_laps].copy()
    df["run"] = (
        df["session"]
        + "_"
        + df["Driver"]
        + "_"
        + df["block"].rank(method="dense").astype(int).astype(str)
    )
    df["lap_in_run"] = df.groupby("run").cumcount()
    df["TyreLife"] = df["TyreLife"].astype(int)
    return df[
        [
            "session",
            "Driver",
            "Team",
            "Compound",
            "TyreLife",
            "LapNumber",
            "lap_s",
            "run",
            "lap_in_run",
        ]
    ].reset_index(drop=True)


def long_runs(
    raw: pd.DataFrame, round_number: int, year: int = 2022, refresh: bool = False
) -> pd.DataFrame:
    """FP1–FP3 long runs of a race, cached in ``data/processed``.

    Args:
        raw: ``load_season()`` laps (for the race's drivers and teams).
        round_number: Race.
        year: Season.
        refresh: Re-download and rebuild even if the parquet exists.
    """
    race = raw[raw["RoundNumber"] == round_number]
    path = practice_path(year, round_number, str(race["EventName"].iloc[0]))
    if path.exists() and not refresh:
        out = pd.read_parquet(path)
    else:
        drivers = set(race["Driver"].unique())
        frames = []
        for name in SESSIONS:
            try:
                frames.append(_session_laps(year, round_number, name))
            except Exception as err:  # noqa: BLE001 - sprint weekends have no FP2/FP3 long runs
                if type(err).__name__ == "RateLimitExceededError":
                    raise  # transient: do not save a partial practice table
                log.warning("no %s for round %s: %s", name, round_number, err)
        out = long_runs_from_laps(pd.concat(frames, ignore_index=True), drivers)
        # Race teams (FastF1 practice team names can differ in spelling).
        teams = race.groupby("Driver")["Team"].first()
        out["Team"] = out["Driver"].map(teams)
        out.insert(0, "RoundNumber", round_number)
        out.to_parquet(path, index=False)
    return out


def summary(runs: pd.DataFrame) -> pd.DataFrame:
    """Runs and laps per session and compound."""
    return (
        runs.groupby(["session", "Compound"])
        .agg(
            runs=("run", "nunique"),
            laps=("lap_s", "size"),
            median_run=("run", lambda r: r.value_counts().median()),
        )
        .reset_index()
    )
