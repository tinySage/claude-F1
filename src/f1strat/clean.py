"""Lap cleaning and exploratory features (Phase 2).

:func:`clean_laps` labels every lap from :func:`f1strat.data.load_season` with
the first reason (if any) it should be kept out of the degradation model, and
adds the gap to the car ahead plus a dirty-air flag. Nothing is dropped:
filter on ``exclude_reason.isna()`` to get model laps.

:func:`pit_loss` estimates the green-flag pit lane time loss per circuit.

Track status codes inside a lap's ``TrackStatus`` string: 1 green, 2 yellow,
4 SC, 5 red flag, 6 VSC deployed, 7 VSC ending.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from f1strat.data import PROCESSED_DIR

#: Exclusion reasons, in priority order. A lap gets the first that applies.
EXCLUDE_REASONS: tuple[str, ...] = (
    "lap_1",
    "after_chequered",
    "in_lap",
    "out_lap",
    "red_flag",
    "sc",
    "vsc",
    "restart_lap",
    "deleted",
    "inaccurate",
    "missing_compound",
    "wet_compound",
    "rainfall",
    "outlier",
)

WET_COMPOUNDS = frozenset({"INTERMEDIATE", "WET"})
DRY_COMPOUNDS = frozenset({"SOFT", "MEDIUM", "HARD"})
NEUTRALISED_CODES = "4567"  # SC, red flag, VSC deployed, VSC ending
OUTLIER_RATIO = 1.07
DIRTY_AIR_GAP_S = 1.0

_DRIVER_RACE = ["RoundNumber", "Driver"]


def _has_code(track_status: pd.Series, codes: str) -> pd.Series:
    """True where the lap's TrackStatus string contains any of ``codes``."""
    return track_status.fillna("").str.contains(f"[{codes}]", regex=True)


def _after_chequered(laps: pd.DataFrame) -> pd.Series:
    """True for laps started after the winner took the chequered flag.

    The final race lap is the highest lap number completed in P1. A lap is after
    the chequered flag if its number is beyond that, or if it started at or
    after the winner crossed the line on that final lap.
    """
    leader = laps[laps["Position"] == 1]
    final_lap = leader.groupby("RoundNumber")["LapNumber"].max()
    finish = (
        leader.merge(final_lap.rename("FinalLap"), on="RoundNumber")
        .query("LapNumber == FinalLap")
        .groupby("RoundNumber")["Time"]
        .min()
    )
    final_lap_per_row = laps["RoundNumber"].map(final_lap)
    finish_per_row = laps["RoundNumber"].map(finish)
    return (laps["LapNumber"] > final_lap_per_row) | (
        laps["LapStartTime"] >= finish_per_row
    )


def _restart_lap(laps: pd.DataFrame) -> pd.Series:
    """True for a driver's first non-neutralised lap after a neutralised one."""
    neutral = _has_code(laps["TrackStatus"], NEUTRALISED_CODES)
    ordered = laps.assign(_neutral=neutral).sort_values(_DRIVER_RACE + ["LapNumber"])
    prev = ordered.groupby(_DRIVER_RACE)["_neutral"].shift(1, fill_value=False)
    prev_lap = ordered.groupby(_DRIVER_RACE)["LapNumber"].shift(1)
    restart = (~ordered["_neutral"]) & prev & (ordered["LapNumber"] - prev_lap == 1)
    return restart.reindex(laps.index)


def _gap_ahead_s(laps: pd.DataFrame) -> pd.Series:
    """Seconds behind the car one position ahead at the line, same lap number."""
    key = ["RoundNumber", "LapNumber", "Position"]
    ahead = laps.loc[laps["Position"].notna(), key + ["Time"]].copy()
    ahead["Position"] = ahead["Position"] + 1  # this car is directly ahead of P+1
    merged = laps[key].merge(
        ahead.rename(columns={"Time": "TimeAhead"}), on=key, how="left"
    )
    merged.index = laps.index
    return (laps["Time"] - merged["TimeAhead"]).dt.total_seconds()


def clean_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """Add ``exclude_reason``, ``gap_ahead_s`` and ``dirty_air`` to season laps.

    Args:
        laps: Output of :func:`f1strat.data.load_season` (laps table).

    Returns:
        A copy of ``laps`` with three extra columns:

        - ``exclude_reason``: null if the lap is usable for the degradation
          model, else the first applicable reason in :data:`EXCLUDE_REASONS`.
          ``outlier`` means slower than 107% of the driver's median lap in that
          race, the median taken over laps with no earlier reason.
        - ``gap_ahead_s``: seconds behind the car one position ahead when both
          crossed the line to end this lap number (null for P1 or missing data).
        - ``dirty_air``: ``gap_ahead_s < 1.0``. A flag only; never a reason to
          exclude.
    """
    df = laps.copy()
    compound = df["Compound"]
    status = df["TrackStatus"]
    rules: dict[str, pd.Series] = {
        "lap_1": df["LapNumber"] == 1,
        "after_chequered": _after_chequered(df),
        "in_lap": df["PitInTime"].notna(),
        "out_lap": df["PitOutTime"].notna(),
        "red_flag": _has_code(status, "5"),
        "sc": _has_code(status, "4"),
        "vsc": _has_code(status, "67"),
        "restart_lap": _restart_lap(df),
        "deleted": df["Deleted"],
        "inaccurate": ~df["IsAccurate"] | df["LapTime"].isna(),
        "missing_compound": ~compound.isin(DRY_COMPOUNDS | WET_COMPOUNDS),
        "wet_compound": compound.isin(WET_COMPOUNDS),
        "rainfall": df["Rainfall"],
    }

    reason = pd.Series(pd.NA, index=df.index, dtype="object")
    for name, hit in rules.items():
        reason = reason.mask(reason.isna() & hit.fillna(False).astype(bool), name)

    lap_s = df["LapTime"].dt.total_seconds()
    median = lap_s.where(reason.isna()).groupby(
        [df["RoundNumber"], df["Driver"]]
    ).transform("median")
    reason = reason.mask(reason.isna() & (lap_s > OUTLIER_RATIO * median), "outlier")

    df["exclude_reason"] = pd.Categorical(reason, categories=EXCLUDE_REASONS)
    df["gap_ahead_s"] = _gap_ahead_s(df)
    df["dirty_air"] = (df["gap_ahead_s"] < DIRTY_AIR_GAP_S).fillna(False).astype(bool)
    return df


def pit_stops(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Pair each in lap with the same driver's next lap as its out lap.

    Args:
        cleaned: Output of :func:`clean_laps`.

    Returns:
        One row per stop with ``RoundNumber``, ``EventName``, ``Driver``,
        ``InLap``, ``in_s``, ``out_s``, ``median_clean_s``, ``loss_s`` and
        ``green`` (no SC / VSC / red flag code on either lap, and the in lap is
        not lap 1).
    """
    lap_s = cleaned["LapTime"].dt.total_seconds()
    median = (
        lap_s.where(cleaned["exclude_reason"].isna())
        .groupby([cleaned["RoundNumber"], cleaned["Driver"]])
        .transform("median")
    )
    base = cleaned.assign(
        lap_s=lap_s,
        median_clean_s=median,
        neutral=_has_code(cleaned["TrackStatus"], NEUTRALISED_CODES),
    )
    cols = _DRIVER_RACE + ["LapNumber", "lap_s", "neutral"]
    in_laps = base.loc[base["PitInTime"].notna(), cols + ["EventName", "median_clean_s"]]
    out_laps = base.loc[base["PitOutTime"].notna(), cols].assign(
        LapNumber=lambda d: d["LapNumber"] - 1
    )
    stops = in_laps.merge(
        out_laps, on=_DRIVER_RACE + ["LapNumber"], suffixes=("_in", "_out")
    )
    stops = stops.rename(
        columns={"LapNumber": "InLap", "lap_s_in": "in_s", "lap_s_out": "out_s"}
    )
    stops["loss_s"] = stops["in_s"] + stops["out_s"] - 2 * stops["median_clean_s"]
    stops["green"] = ~stops["neutral_in"] & ~stops["neutral_out"] & (stops["InLap"] > 1)
    return stops[
        ["RoundNumber", "EventName", "Driver", "InLap", "in_s", "out_s",
         "median_clean_s", "loss_s", "green"]
    ].reset_index(drop=True)


def pit_loss(cleaned: pd.DataFrame, race: pd.DataFrame) -> pd.DataFrame:
    """Median green-flag pit lane loss per circuit.

    For each green-flag stop (see :func:`pit_stops`): (in lap + out lap) minus
    twice the driver's median clean lap in that race. Stops with a missing lap
    time or no clean laps for the driver are skipped.

    Args:
        cleaned: Output of :func:`clean_laps`.
        race: Race table from ``load_season(table="race")`` (for circuit names).

    Returns:
        One row per race: ``RoundNumber``, ``EventName``, ``Circuit``,
        ``pit_loss_s`` (median), ``q25_s``, ``q75_s`` and ``n_stops``.
    """
    stops = pit_stops(cleaned)
    stops = stops[stops["green"] & stops["loss_s"].notna()]
    grouped = stops.groupby(["RoundNumber", "EventName"])["loss_s"]
    out = pd.DataFrame(
        {
            "pit_loss_s": grouped.median(),
            "q25_s": grouped.quantile(0.25),
            "q75_s": grouped.quantile(0.75),
            "n_stops": grouped.size(),
        }
    ).reset_index()
    circuits = race.groupby("RoundNumber")["Circuit"].first()
    out.insert(2, "Circuit", out["RoundNumber"].map(circuits))
    out["n_stops"] = out["n_stops"].astype("Int64")
    return out


def save_pit_loss(
    pit: pd.DataFrame, year: int = 2022, out_dir: Path = PROCESSED_DIR
) -> Path:
    """Write the pit loss table to ``<out_dir>/<year>/pit_loss.parquet``."""
    path = out_dir / str(year) / "pit_loss.parquet"
    pit.to_parquet(path, index=False)
    return path


def main() -> None:
    """Clean the 2022 season and write the pit loss table."""
    from f1strat.data import load_season

    cleaned = clean_laps(load_season(2022))
    pit = pit_loss(cleaned, load_season(2022, table="race"))
    path = save_pit_loss(pit, 2022)
    print(pit.round(2).to_string(index=False))
    print(f"\nwrote {path}")


__all__ = [
    "EXCLUDE_REASONS",
    "clean_laps",
    "pit_loss",
    "pit_stops",
    "save_pit_loss",
]


if __name__ == "__main__":
    main()
