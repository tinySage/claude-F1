"""Data acquisition (Phase 1).

Loads 2022 race sessions through FastF1 (cache in ``cache/``) and saves two
parquet tables per race in ``data/processed/<year>/``:

- ``<RR>_<slug>_laps.parquet``: one row per driver lap, the columns in
  :data:`LAPS_SCHEMA`. Track temperature and rainfall are matched to each lap
  with FastF1's ``Laps.get_weather_data`` (the first weather sample inside the
  lap, else the last one before the lap ended).
- ``<RR>_<slug>_race.parquet``: one row per race lap (leader's lap number), the
  columns in :data:`RACE_SCHEMA`: circuit, scheduled total laps, and whether a
  safety car, virtual safety car or red flag was active on that lap.

Nothing is filtered or cleaned here; that is Phase 2. See
``data/processed/README.md`` for the schema in prose.

Run the full pull with ``uv run python -m f1strat.data``.
"""

from __future__ import annotations

import logging
import re
import traceback
from datetime import UTC, datetime
from pathlib import Path

import fastf1
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "cache"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

#: Columns and dtypes of the per-lap table, in order.
LAPS_SCHEMA: dict[str, str] = {
    "RoundNumber": "Int64",
    "EventName": "object",
    "Driver": "object",
    "Team": "object",
    "LapNumber": "Int64",
    "LapTime": "timedelta64[ns]",
    "Stint": "Int64",
    "Compound": "object",
    "TyreLife": "Int64",
    "FreshTyre": "bool",
    "PitInTime": "timedelta64[ns]",
    "PitOutTime": "timedelta64[ns]",
    "TrackStatus": "object",
    "Position": "Int64",
    "Deleted": "bool",
    "IsAccurate": "bool",
    "Time": "timedelta64[ns]",
    "LapStartTime": "timedelta64[ns]",
    "TrackTemp": "float64",
    "Rainfall": "bool",
}

#: Columns and dtypes of the per-race-lap table, in order.
RACE_SCHEMA: dict[str, str] = {
    "RoundNumber": "Int64",
    "EventName": "object",
    "Circuit": "object",
    "TotalLaps": "Int64",
    "LapNumber": "Int64",
    "LapStartTime": "timedelta64[ns]",
    "SafetyCar": "bool",
    "VSC": "bool",
    "RedFlag": "bool",
    "SafetyCarPeriod": "Int64",
    "VSCPeriod": "Int64",
    "RedFlagPeriod": "Int64",
}

#: FastF1 track status codes that make up each neutralisation type.
#: 4 = SC deployed, 6 = VSC deployed, 7 = VSC ending, 5 = red flag.
PERIOD_CODES: dict[str, frozenset[str]] = {
    "SafetyCar": frozenset({"4"}),
    "VSC": frozenset({"6", "7"}),
    "RedFlag": frozenset({"5"}),
}

_INT_LAP_COLUMNS = ["LapNumber", "Stint", "TyreLife", "Position"]


def configure_cache(cache_dir: Path = CACHE_DIR) -> None:
    """Point FastF1's HTTP cache at ``cache_dir``, creating it if needed."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.configure(cache_dir=str(cache_dir))


def load_race_session(year: int, event: int | str) -> fastf1.core.Session:
    """Load a race session with laps, weather and race control messages.

    Args:
        year: Championship year.
        event: Round number or event name as FastF1 accepts it.

    Returns:
        A loaded FastF1 session (no telemetry).
    """
    configure_cache()
    session = fastf1.get_session(year, event, "R")
    session.load(laps=True, telemetry=False, weather=True, messages=True)
    return session


def laps_table(session: fastf1.core.Session) -> pd.DataFrame:
    """Build the per-lap table from a loaded session.

    Args:
        session: A race session loaded with laps and weather.

    Returns:
        DataFrame with the columns of :data:`LAPS_SCHEMA`.
    """
    laps = session.laps
    weather = laps.get_weather_data().reset_index(drop=True)
    df = pd.DataFrame(laps).reset_index(drop=True)
    df["TrackTemp"] = weather["TrackTemp"].to_numpy()
    df["Rainfall"] = weather["Rainfall"].to_numpy()
    df["RoundNumber"] = int(session.event["RoundNumber"])
    df["EventName"] = session.event["EventName"]
    for col in _INT_LAP_COLUMNS:
        df[col] = _to_int(df[col], col)
    return _apply_schema(df, LAPS_SCHEMA)


def race_table(session: fastf1.core.Session) -> pd.DataFrame:
    """Build the per-race-lap table of SC / VSC / red flag periods.

    A race lap ``n`` spans from the leader starting lap ``n`` (the earliest
    ``LapStartTime`` of any driver on lap ``n``) to the leader starting lap
    ``n + 1``. A lap is flagged if a period overlaps that span at all. Periods
    come from FastF1's track status timeline. Periods that end before the
    leader starts lap 1 (e.g. a red flag during a delayed start) are dropped;
    the rest are numbered from 1 in race order.

    Args:
        session: A race session loaded with laps and messages.

    Returns:
        DataFrame with the columns of :data:`RACE_SCHEMA`.
    """
    laps = session.laps
    lap_start = laps.groupby("LapNumber")["LapStartTime"].min().sort_index()
    race_end = laps["Time"].max()
    lap_end = lap_start.shift(-1).fillna(race_end)

    df = pd.DataFrame(
        {
            "LapNumber": _to_int(pd.Series(lap_start.index), "LapNumber"),
            "LapStartTime": lap_start.to_numpy(),
        }
    )
    race_start = lap_start.iloc[0]
    periods = neutralisation_periods(session.track_status, after=race_start)
    for kind in PERIOD_CODES:
        df[kind] = False
        df[f"{kind}Period"] = pd.array([pd.NA] * len(df), dtype="Int64")
    for kind, pid, start, end in periods.itertuples(index=False):
        hit = (lap_start.to_numpy() < end) & (lap_end.to_numpy() > start)
        df.loc[hit, kind] = True
        # Keep the first period id if two periods share a lap.
        free = hit & df[f"{kind}Period"].isna().to_numpy()
        df.loc[free, f"{kind}Period"] = pid

    df["RoundNumber"] = int(session.event["RoundNumber"])
    df["EventName"] = session.event["EventName"]
    df["Circuit"] = session.session_info["Meeting"]["Circuit"]["ShortName"]
    df["TotalLaps"] = session.total_laps
    _check_against_race_control(session, df)
    return _apply_schema(df, RACE_SCHEMA)


def neutralisation_periods(
    track_status: pd.DataFrame, after: pd.Timedelta | None = None
) -> pd.DataFrame:
    """Turn a track status timeline into SC / VSC / red flag periods.

    A period is a maximal run of consecutive status entries whose codes all
    belong to one kind in :data:`PERIOD_CODES`. It ends at the next status
    change that leaves that kind.

    Args:
        track_status: FastF1 ``Session.track_status`` with ``Time`` (session
            time) and ``Status`` (code as string) columns.
        after: If given, drop periods that end at or before this session time
            before numbering.

    Returns:
        DataFrame with columns ``kind``, ``period`` (1-based per kind),
        ``start`` and ``end`` (session times). A period still open at the end
        of the timeline gets ``end = Timedelta.max``.
    """
    rows: list[tuple[str, int, pd.Timedelta, pd.Timedelta]] = []
    times = list(track_status["Time"])
    codes = [str(c) for c in track_status["Status"]]
    for kind, kind_codes in PERIOD_CODES.items():
        count = 0
        start: pd.Timedelta | None = None
        for t, code in zip(times, codes):
            if code in kind_codes and start is None:
                start = t
            elif code not in kind_codes and start is not None:
                if after is None or t > after:
                    count += 1
                    rows.append((kind, count, start, t))
                start = None
        if start is not None:
            count += 1
            rows.append((kind, count, start, pd.Timedelta.max))
    return pd.DataFrame(rows, columns=["kind", "period", "start", "end"])


def _check_against_race_control(session: fastf1.core.Session, race: pd.DataFrame) -> None:
    """Log a warning if period counts disagree with race control messages.

    Track status gives the timing; race control messages are an independent
    record of deployments, so a mismatch is worth a look. Messages carry
    wall-clock times, so they are put on the session clock by aligning the
    "CHEQUERED FLAG" message with the session status "Finished", then only
    messages from the leader's lap 1 start up to the chequered flag are
    counted.
    """
    rcm = session.race_control_messages
    msg = rcm["Message"].fillna("").str.upper()
    status = session.session_status
    chequered = rcm.loc[msg == "CHEQUERED FLAG", "Time"]
    finished = status.loc[status["Status"] == "Finished", "Time"]
    if chequered.empty or finished.empty:
        logger.warning(
            "%s: cannot align race control messages to session time; skipping cross-check",
            session.event["EventName"],
        )
        return
    offset = chequered.iloc[0] - finished.iloc[0]
    in_race = ((rcm["Time"] - offset) >= race["LapStartTime"].iloc[0]) & (
        rcm["Time"] <= chequered.iloc[0]
    )
    msg = msg[in_race]
    from_rcm = {
        "SafetyCar": int(msg.str.fullmatch(r"SAFETY CAR DEPLOYED").sum()),
        "VSC": int(msg.str.fullmatch(r"VIRTUAL SAFETY CAR DEPLOYED").sum()),
        "RedFlag": int(msg.str.fullmatch(r"RED FLAG").sum()),
    }
    for kind, n_rcm in from_rcm.items():
        n_ts = int(race[f"{kind}Period"].nunique())
        if n_ts != n_rcm:
            logger.warning(
                "%s round %s: %d %s period(s) from track status in race laps, "
                "%d deployment(s) in race control messages",
                session.event["EventName"],
                session.event["RoundNumber"],
                n_ts,
                kind,
                n_rcm,
            )


def _to_int(series: pd.Series, name: str) -> pd.Series:
    """Cast a float column holding whole numbers (or NaN) to nullable Int64."""
    values = series.dropna()
    if not (values == values.round()).all():
        raise ValueError(f"column {name} has non-integer values")
    return series.astype("Int64")


def _apply_schema(df: pd.DataFrame, schema: dict[str, str]) -> pd.DataFrame:
    """Select the schema's columns in order and cast to the schema's dtypes."""
    return df[list(schema)].astype(schema).reset_index(drop=True)


def _slug(name: str) -> str:
    """'British Grand Prix' -> 'british_grand_prix'."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def race_paths(
    year: int, round_number: int, event_name: str, out_dir: Path = PROCESSED_DIR
) -> tuple[Path, Path]:
    """Return the (laps, race) parquet paths for one race."""
    stem = f"{round_number:02d}_{_slug(event_name)}"
    base = out_dir / str(year)
    return base / f"{stem}_laps.parquet", base / f"{stem}_race.parquet"


def build_season(year: int = 2022, out_dir: Path = PROCESSED_DIR) -> list[int]:
    """Load every race of a season and save both tables per race to parquet.

    Failures are logged with tracebacks to ``<out_dir>/<year>/load_failures.log``
    (rewritten on every run) and to the module logger; the loop carries on.

    Args:
        year: Championship year.
        out_dir: Root of the processed data directory.

    Returns:
        Round numbers that failed to load.
    """
    configure_cache()
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    year_dir = out_dir / str(year)
    year_dir.mkdir(parents=True, exist_ok=True)
    failures: list[tuple[int, str, str]] = []

    for _, event in schedule.iterrows():
        rnd, name = int(event["RoundNumber"]), str(event["EventName"])
        try:
            session = load_race_session(year, rnd)
            laps_path, race_path = race_paths(year, rnd, name, out_dir)
            laps_table(session).to_parquet(laps_path, index=False)
            race_table(session).to_parquet(race_path, index=False)
            logger.info("saved round %d %s", rnd, name)
        except Exception:  # noqa: BLE001 - record and continue
            logger.exception("failed to load round %d %s", rnd, name)
            failures.append((rnd, name, traceback.format_exc()))

    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    with open(year_dir / "load_failures.log", "w") as f:
        f.write(f"build_season({year}) run at {stamp}\n")
        f.write(f"{len(schedule)} races scheduled, {len(failures)} failed\n")
        for rnd, name, tb in failures:
            f.write(f"\n--- round {rnd} {name} ---\n{tb}")
    return [rnd for rnd, _, _ in failures]


def load_season(
    year: int = 2022, table: str = "laps", out_dir: Path = PROCESSED_DIR
) -> pd.DataFrame:
    """Read a season's saved parquet files into one combined table.

    Args:
        year: Championship year.
        table: ``"laps"`` (driver laps) or ``"race"`` (per race lap status).
        out_dir: Root of the processed data directory.

    Returns:
        All races concatenated, ordered by round, with the matching schema.
    """
    if table not in ("laps", "race"):
        raise ValueError(f"table must be 'laps' or 'race', got {table!r}")
    schema = LAPS_SCHEMA if table == "laps" else RACE_SCHEMA
    files = sorted((out_dir / str(year)).glob(f"*_{table}.parquet"))
    if not files:
        raise FileNotFoundError(f"no {table} parquet files for {year} in {out_dir}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return _apply_schema(df, schema)


def season_summary(laps: pd.DataFrame, race: pd.DataFrame) -> pd.DataFrame:
    """Summarise each race: drivers, laps, compounds and neutralisations.

    Args:
        laps: Combined laps table from :func:`load_season`.
        race: Combined race table from ``load_season(table="race")``.

    Returns:
        One row per round.
    """
    per_laps = laps.groupby(["RoundNumber", "EventName"]).agg(
        Drivers=("Driver", "nunique"),
        LapRows=("LapNumber", "size"),
        Compounds=(
            "Compound",
            lambda s: ", ".join(sorted(s.dropna().unique())),
        ),
    )
    per_race = race.groupby("RoundNumber").agg(
        RaceLaps=("LapNumber", "max"),
        TotalLaps=("TotalLaps", "first"),
        SCPeriods=("SafetyCarPeriod", "nunique"),
        VSCPeriods=("VSCPeriod", "nunique"),
        RedFlags=("RedFlagPeriod", "nunique"),
    )
    return per_laps.reset_index().merge(per_race.reset_index(), on="RoundNumber")


def main() -> None:
    """Pull the 2022 season, then print a per-race summary."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("fastf1", "requests_cache"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    failed = build_season(2022)
    summary = season_summary(load_season(2022), load_season(2022, table="race"))
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary.to_string(index=False))
    print(f"\nfailed rounds: {failed or 'none'}")


if __name__ == "__main__":
    main()
