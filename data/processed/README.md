# Processed data

Built by `uv run python -m f1strat.data` (see `src/f1strat/data.py`). Source:
FastF1 race sessions (`'R'`) loaded with laps, weather and race control
messages, no telemetry. **Nothing is filtered or cleaned here**: lap 1, in/out
laps, SC/VSC laps, deleted laps and wet-weather laps are all kept. Cleaning is
Phase 2.

Load everything with:

```python
from f1strat.data import load_season
laps = load_season(2022)                # all driver laps, one table
race = load_season(2022, table="race")  # per-race-lap status, one table
```

## Files

`2022/<RR>_<event_slug>_laps.parquet` and `2022/<RR>_<event_slug>_race.parquet`
for each of the 22 rounds, plus `2022/load_failures.log` (rewritten on every
build; lists any race that failed to load, with its traceback).

## `*_laps.parquet`: one row per driver lap

Key: (`RoundNumber`, `Driver`, `LapNumber`). Times are `timedelta` on the
FastF1 session clock unless noted.

| Column | Type | Meaning |
|---|---|---|
| RoundNumber | Int64 | Championship round (1–22) |
| EventName | str | e.g. "Hungarian Grand Prix" |
| Driver | str | Three-letter code, e.g. "LEC" |
| Team | str | Team name |
| LapNumber | Int64 | Driver's lap number |
| LapTime | timedelta | Lap time; missing for some laps (e.g. after a red flag) |
| Stint | Int64 | Stint number, increments at each tyre change |
| Compound | str | SOFT, MEDIUM, HARD, INTERMEDIATE, WET (may be missing/UNKNOWN) |
| TyreLife | Int64 | Laps on this set, including laps from earlier sessions |
| FreshTyre | bool | Set was new when fitted |
| PitInTime | timedelta | Session time the car entered the pit lane (in-lap only) |
| PitOutTime | timedelta | Session time the car left the pit lane (out-lap only) |
| TrackStatus | str | Every track status code seen during the lap, concatenated (1 green, 2 yellow, 4 SC, 5 red, 6 VSC, 7 VSC ending) |
| Position | Int64 | Position at the end of the lap |
| Deleted | bool | Lap time deleted by stewards (track limits) |
| IsAccurate | bool | FastF1's own accuracy check on the lap timing |
| Time | timedelta | Session time at the end of the lap |
| LapStartTime | timedelta | Session time at the start of the lap |
| TrackTemp | float | Track temperature (°C) |
| Rainfall | bool | Rain reported by the weather station |

Weather is sampled once a minute. Each lap gets FastF1's
`Laps.get_weather_data()` value: the first sample within the lap, else the last
sample before the lap ended.

## `*_race.parquet`: one row per race lap

Key: (`RoundNumber`, `LapNumber`). `LapNumber` is the leader's lap.

| Column | Type | Meaning |
|---|---|---|
| RoundNumber | Int64 | Championship round |
| EventName | str | Event name |
| Circuit | str | Circuit short name, e.g. "Hungaroring" |
| TotalLaps | Int64 | Scheduled race distance in laps (may exceed the laps run, e.g. a time-limited race) |
| LapNumber | Int64 | Race lap |
| LapStartTime | timedelta | When the leader started this lap (earliest `LapStartTime` on this lap) |
| SafetyCar | bool | A safety car period overlaps this lap |
| VSC | bool | A virtual safety car period overlaps this lap |
| RedFlag | bool | A red flag period overlaps this lap |
| SafetyCarPeriod | Int64 | Which SC period (1, 2, …) covers the lap, else null |
| VSCPeriod | Int64 | Which VSC period covers the lap, else null |
| RedFlagPeriod | Int64 | Which red flag period covers the lap, else null |

How periods are built:
- From FastF1's track status timeline. A period is a run of consecutive
  statuses of one kind (SC = 4; VSC = 6 or 7; red = 5), ending at the next
  status of another kind. A VSC that turns into a SC is a VSC period followed by
  a SC period.
- Race lap `n` spans from the leader starting lap `n` to the leader starting
  lap `n+1`. A lap is flagged if a period overlaps that span at all, so a
  short period can flag the lap it started on and the next.
- Periods that end before the leader starts lap 1 (e.g. Monaco's pre-start red
  flag) are dropped; the rest are numbered from 1 per kind.
- Cross-check: counts are compared with race control "SAFETY CAR DEPLOYED",
  "VIRTUAL SAFETY CAR DEPLOYED" and "RED FLAG" messages between the leader
  starting lap 1 and the chequered flag (aligned to the session clock via the
  chequered flag). Mismatches are logged as warnings, not corrected.

## Known quirks in the source data (left as-is for Phase 2)

- **Missing compound**: 334 laps have `Compound == "None"` (the string, as
  FastF1 returns it): Austria stint 3 for 12 drivers, and Monaco stint 5 for
  STR.
- **Japan phantom lap**: VER has a lap 29 with no position after the chequered
  flag (the race was 28 laps), so the Japan race table has a lap 29 row.
- **Monaco rolling start**: the restart behind the safety car shows as a short
  SC period on lap 2 in track status, with no "SAFETY CAR DEPLOYED" message,
  so the cross-check logs 2 SC periods vs 1 message.
- **São Paulo race control feed** stops at lap 64 with no chequered flag
  message, so the cross-check is skipped for that race.
- **Time-limited races**: Monaco (64 of 77 laps), Singapore (59 of 61) and
  Japan (28 of 53) ran fewer laps than `TotalLaps`.
