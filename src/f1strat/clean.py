"""Lap cleaning and exploratory features (Phase 2).

:func:`clean_laps` labels every lap from :func:`f1strat.data.load_season` with
the first reason (if any) it should be kept out of the degradation model, and
adds the gap to the car ahead plus a dirty-air flag. Nothing is dropped:
filter on ``exclude_reason.isna()`` to get model laps.

:func:`pit_loss` estimates the green-flag pit lane time loss per circuit;
:func:`pit_loss_with_references` fills the mixed-conditions circuits that
need it from a dry race at the same circuit in another season.

Track status codes inside a lap's ``TrackStatus`` string: 1 green, 2 yellow,
4 SC, 5 red flag, 6 VSC deployed, 7 VSC ending.
"""

from __future__ import annotations

import logging
from collections.abc import Collection
from pathlib import Path

import fastf1
import pandas as pd

from f1strat.data import PROCESSED_DIR, configure_cache, laps_table, load_race_session

logger = logging.getLogger(__name__)

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
    "mixed_conditions",
    "outlier",
)

#: 2022 races run partly wet or on a drying track. Every lap is kept out of the
#: dry degradation fit, and their pit loss comes from a reference race.
MIXED_CONDITIONS_2022: frozenset[str] = frozenset(
    {
        "Emilia Romagna Grand Prix",
        "Monaco Grand Prix",
        "Singapore Grand Prix",
        "Japanese Grand Prix",
    }
)

#: Mixed-conditions races that need a reference pit loss. Monaco is a case
#: study (Phase 6). Imola, Singapore and Suzuka are not needed: they are out
#: of the dry fit and the backtests, and none is a case study.
PIT_LOSS_REFERENCE_2022: frozenset[str] = frozenset({"Monaco Grand Prix"})

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
    return (laps["LapNumber"] > final_lap_per_row) | (laps["LapStartTime"] >= finish_per_row)


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
    merged = laps[key].merge(ahead.rename(columns={"Time": "TimeAhead"}), on=key, how="left")
    merged.index = laps.index
    return (laps["Time"] - merged["TimeAhead"]).dt.total_seconds()


def clean_laps(
    laps: pd.DataFrame, mixed_conditions: Collection[str] = MIXED_CONDITIONS_2022
) -> pd.DataFrame:
    """Add ``exclude_reason``, ``gap_ahead_s`` and ``dirty_air`` to season laps.

    ``Rainfall`` is not an exclude reason: it stays as a column for Phase 3 to
    test as a covariate. Wet running is caught by ``wet_compound`` and
    ``mixed_conditions``.

    Args:
        laps: Output of :func:`f1strat.data.load_season` (laps table).
        mixed_conditions: Event names whose laps are all labelled
            ``mixed_conditions`` (if no earlier reason applies). Defaults to
            the 2022 list; pass an empty set for other seasons.

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
        "mixed_conditions": df["EventName"].isin(list(mixed_conditions)),
    }

    reason = pd.Series(pd.NA, index=df.index, dtype="object")
    for name, hit in rules.items():
        reason = reason.mask(reason.isna() & hit.fillna(False).astype(bool), name)

    lap_s = df["LapTime"].dt.total_seconds()
    median = (
        lap_s.where(reason.isna()).groupby([df["RoundNumber"], df["Driver"]]).transform("median")
    )
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
        ``InLap``, ``in_s``, ``out_s``, ``median_clean_s``, ``loss_s``,
        ``green`` (no SC / VSC / red flag code on either lap, and the in lap is
        not lap 1) and ``dry_to_dry`` (dry compound on both laps).
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
    cols = _DRIVER_RACE + ["LapNumber", "lap_s", "neutral", "Compound"]
    in_laps = base.loc[base["PitInTime"].notna(), cols + ["EventName", "median_clean_s"]]
    out_laps = base.loc[base["PitOutTime"].notna(), cols].assign(
        LapNumber=lambda d: d["LapNumber"] - 1
    )
    stops = in_laps.merge(out_laps, on=_DRIVER_RACE + ["LapNumber"], suffixes=("_in", "_out"))
    stops = stops.rename(columns={"LapNumber": "InLap", "lap_s_in": "in_s", "lap_s_out": "out_s"})
    stops["loss_s"] = stops["in_s"] + stops["out_s"] - 2 * stops["median_clean_s"]
    stops["green"] = ~stops["neutral_in"] & ~stops["neutral_out"] & (stops["InLap"] > 1)
    stops["dry_to_dry"] = stops["Compound_in"].isin(DRY_COMPOUNDS) & stops["Compound_out"].isin(
        DRY_COMPOUNDS
    )
    return stops[
        [
            "RoundNumber",
            "EventName",
            "Driver",
            "InLap",
            "in_s",
            "out_s",
            "median_clean_s",
            "loss_s",
            "green",
            "dry_to_dry",
        ]
    ].reset_index(drop=True)


def _summarise_stops(stops: pd.DataFrame) -> pd.DataFrame:
    """Median, quartiles and count of ``loss_s`` per race."""
    grouped = stops.groupby(["RoundNumber", "EventName"])["loss_s"]
    out = pd.DataFrame(
        {
            "pit_loss_s": grouped.median(),
            "q25_s": grouped.quantile(0.25),
            "q75_s": grouped.quantile(0.75),
            "n_stops": grouped.size(),
        }
    ).reset_index()
    out["n_stops"] = out["n_stops"].astype("Int64")
    return out


def pit_loss(
    cleaned: pd.DataFrame,
    race: pd.DataFrame,
    mixed_conditions: Collection[str] = MIXED_CONDITIONS_2022,
) -> pd.DataFrame:
    """Median green-flag pit lane loss per circuit, from the season's own races.

    For each green-flag stop (see :func:`pit_stops`): (in lap + out lap) minus
    twice the driver's median clean lap in that race. Stops with a missing lap
    time or no clean laps for the driver are skipped. Mixed-conditions races
    are left out; :func:`pit_loss_with_references` fills them in.

    Args:
        cleaned: Output of :func:`clean_laps`.
        race: Race table from ``load_season(table="race")`` (for circuit names).
        mixed_conditions: Event names to leave out.

    Returns:
        One row per race: ``RoundNumber``, ``EventName``, ``Circuit``,
        ``pit_loss_s`` (median), ``q25_s``, ``q75_s`` and ``n_stops``.
    """
    stops = pit_stops(cleaned)
    keep = (
        stops["green"] & stops["loss_s"].notna() & ~stops["EventName"].isin(list(mixed_conditions))
    )
    out = _summarise_stops(stops[keep])
    circuits = race.groupby("RoundNumber")["Circuit"].first()
    out.insert(2, "Circuit", out["RoundNumber"].map(circuits))
    return out


def reference_pit_loss(
    event_name: str,
    exclude_year: int = 2022,
    first_year: int = 2018,
    today: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Pit loss for one circuit from its most recent downloadable dry race.

    Seasons are tried newest first, skipping ``exclude_year`` and races not yet
    held. A candidate whose timing data cannot be downloaded is skipped (F1
    live timing needs a signed-in F1 account for most seasons). A race is dry
    if no lap was run on an intermediate or wet tyre; the first dry one is used
    (a wet candidate is loaded, checked and skipped). Uses green-flag,
    dry-to-dry stops only.

    Args:
        event_name: Event name, stable across seasons (e.g. "Monaco Grand Prix").
        exclude_year: Season to skip (the one being studied).
        first_year: Earliest season to try.
        today: Cut-off for races already held; defaults to now.

    Returns:
        One row: ``EventName``, ``pit_loss_s``, ``q25_s``, ``q75_s``,
        ``n_stops``, ``source_year``.

    Raises:
        LookupError: If no downloadable dry race is found.
    """
    today = today or pd.Timestamp.now()
    configure_cache()
    for year in range(today.year, first_year - 1, -1):
        if year == exclude_year:
            continue
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        match = schedule[(schedule["EventName"] == event_name) & (schedule["EventDate"] < today)]
        if match.empty:
            continue
        rnd = int(match["RoundNumber"].iloc[0])
        try:
            laps = laps_table(load_race_session(year, rnd))
        except fastf1.exceptions.DataNotLoadedError:
            logger.info(
                "%d %s: timing data unavailable; trying an earlier season", year, event_name
            )
            continue
        logger.info("loaded %d %s as pit loss reference candidate", year, event_name)
        if laps["Compound"].isin(WET_COMPOUNDS).any():
            logger.info("%d %s had wet running; trying an earlier season", year, event_name)
            continue
        stops = pit_stops(clean_laps(laps, mixed_conditions=()))
        stops = stops[stops["green"] & stops["dry_to_dry"] & stops["loss_s"].notna()]
        out = _summarise_stops(stops).drop(columns="RoundNumber")
        out["source_year"] = year
        return out
    raise LookupError(
        f"no downloadable dry {event_name} found between {first_year} and {today.year}"
    )


def pit_loss_with_references(
    cleaned: pd.DataFrame,
    race: pd.DataFrame,
    year: int = 2022,
    mixed_conditions: Collection[str] = MIXED_CONDITIONS_2022,
    reference_events: Collection[str] = PIT_LOSS_REFERENCE_2022,
) -> pd.DataFrame:
    """Season pit loss table, one row per race, with its source recorded.

    Args:
        cleaned: Output of :func:`clean_laps` for ``year``.
        race: Race table for ``year``.
        year: The season studied; its own races get ``source_year == year``.
        mixed_conditions: Event names whose own stops are not used.
        reference_events: Mixed-conditions events filled from
            :func:`reference_pit_loss`; the rest are left null.

    Returns:
        :func:`pit_loss` columns plus ``source_year`` and ``pit_loss_status``:
        ``own_race`` (stops from ``year``), ``reference`` (from
        ``source_year``) or ``not_needed`` (null value, ``n_stops == 0``).
    """
    own = pit_loss(cleaned, race, mixed_conditions).assign(
        source_year=year, pit_loss_status="own_race"
    )
    circuits = race.groupby("RoundNumber")[["EventName", "Circuit"]].first()
    extra = []
    for rnd, row in circuits[circuits["EventName"].isin(list(mixed_conditions))].iterrows():
        if row["EventName"] in reference_events:
            ref = reference_pit_loss(row["EventName"], exclude_year=year)
            ref["pit_loss_status"] = "reference"
        else:
            ref = pd.DataFrame(
                {
                    "EventName": [row["EventName"]],
                    "pit_loss_s": [float("nan")],
                    "q25_s": [float("nan")],
                    "q75_s": [float("nan")],
                    "n_stops": [0],
                    "source_year": [pd.NA],
                    "pit_loss_status": ["not_needed"],
                }
            )
        ref.insert(0, "RoundNumber", rnd)
        ref.insert(2, "Circuit", row["Circuit"])
        extra.append(ref)
    out = pd.concat([own, *extra], ignore_index=True)
    out = out.sort_values("RoundNumber").reset_index(drop=True)
    out["RoundNumber"] = out["RoundNumber"].astype("Int64")
    out["n_stops"] = out["n_stops"].astype("Int64")
    out["source_year"] = out["source_year"].astype("Int64")
    return out


def save_pit_loss(pit: pd.DataFrame, year: int = 2022, out_dir: Path = PROCESSED_DIR) -> Path:
    """Write the pit loss table to ``<out_dir>/<year>/pit_loss.parquet``."""
    path = out_dir / str(year) / "pit_loss.parquet"
    pit.to_parquet(path, index=False)
    return path


def main() -> None:
    """Clean the 2022 season and write the pit loss table."""
    from f1strat.data import load_season

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("fastf1", "requests_cache"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    cleaned = clean_laps(load_season(2022))
    pit = pit_loss_with_references(cleaned, load_season(2022, table="race"))
    path = save_pit_loss(pit, 2022)
    print(pit.round(2).to_string(index=False))
    print(f"\nwrote {path}")


__all__ = [
    "EXCLUDE_REASONS",
    "MIXED_CONDITIONS_2022",
    "PIT_LOSS_REFERENCE_2022",
    "clean_laps",
    "pit_loss",
    "pit_loss_with_references",
    "pit_stops",
    "reference_pit_loss",
    "save_pit_loss",
]


if __name__ == "__main__":
    main()
