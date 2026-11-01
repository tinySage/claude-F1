"""Race-day tyre inventory, reconstructed from pre-race sessions (Phase 4).

For each car, start from the event's 2022 dry allocation, run every practice
and qualifying stint in order, apply the 2022 return rules, and keep what is
left for the race. All of this was known to the pit wall before the race, so
it is not hindsight.

**Rules** (FIA 2022 Formula 1 Sporting Regulations, issue 7, Art. 30.2a, 30.2d,
30.5h–i):

- Allocation: 13 sets (2 hard, 3 medium, 8 soft) without a sprint; 12 sets
  (2 hard, 4 medium, 6 soft) with one.
- One soft set (the Q3 specification) may not be used or returned before Q3.
- One set of each mandatory race specification (assumed hard and medium) may
  not be returned before the race.
- Conventional weekend: 2 sets returned after P1, 2 after P2 and 2 after P3;
  cars that reached Q3 return 1 soft set after qualifying.
- Sprint weekend: 1 set returned after P1; 1 more after qualifying, plus 1
  soft set for cars that reached Q3; the set with most laps in P2 after P2;
  the set with most laps after the sprint.

**Assumption (the rules do not say which sets go back):** the team returns its
most-used eligible sets first, and only returns unused sets when it has too
few used ones (then softs first, the most plentiful compound).

**Stand-in drivers.** A driver who runs only in practice (e.g. an FP1
stand-in) uses the tyres of the race driver he replaced (Art. 30.5d), so his
stints are charged to the race driver of the same team who did not run in that
session.

**Matching sets across stints.** FastF1 does not identify physical sets. A
stint on a new set takes a new set of that compound. A stint on a used set is
matched to the set of that compound whose laps equal the stint's first
TyreLife minus one (FastF1's TyreLife counts every lap a set has run, across
sessions); if none matches exactly, the closest set with fewer laps is used
and the mismatch is recorded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import fastf1
import pandas as pd

from f1strat.data import configure_cache

logger = logging.getLogger(__name__)

DRY = ("SOFT", "MEDIUM", "HARD")
ALLOCATION: dict[str, dict[str, int]] = {
    "conventional": {"HARD": 2, "MEDIUM": 3, "SOFT": 8},
    "sprint": {"HARD": 2, "MEDIUM": 4, "SOFT": 6},
}
MANDATORY_RACE = ("HARD", "MEDIUM")
Q3_SPEC = "SOFT"


@dataclass
class TyreSet:
    """One physical set: compound, laps run so far, and its history."""

    compound: str
    set_id: int
    laps: int = 0
    returned: bool = False
    protected: str = ""  # "q3" or "race" while it may not be returned
    history: list[str] = field(default_factory=list)

    @property
    def new(self) -> bool:
        """True if the set has run no laps."""
        return self.laps == 0


@dataclass
class Inventory:
    """A car's sets for one event; ``notes`` records assumptions and problems."""

    driver: str
    sets: list[TyreSet]
    notes: list[str] = field(default_factory=list)

    def available(self) -> list[TyreSet]:
        """Sets not returned to the supplier."""
        return [s for s in self.sets if not s.returned]

    def summary(self) -> str:
        """E.g. ``"S new×3, S(4)×2, M new×2, H new×2"``."""
        parts = []
        for comp in DRY:
            for laps, count in (
                pd.Series([s.laps for s in self.available() if s.compound == comp])
                .value_counts()
                .sort_index()
                .items()
            ):
                tag = "new" if laps == 0 else f"({laps + 1})"
                parts.append(f"{comp[0]} {tag}×{count}")
        return ", ".join(parts)


def _new_inventory(driver: str, fmt: str) -> Inventory:
    sets, k = [], 0
    for comp, n in ALLOCATION[fmt].items():
        for _ in range(n):
            sets.append(TyreSet(comp, k))
            k += 1
    inv = Inventory(driver, sets)
    # Reserve one new soft for Q3 and one of each mandatory race compound.
    next(s for s in sets if s.compound == Q3_SPEC).protected = "q3"
    for comp in MANDATORY_RACE:
        next(s for s in sets if s.compound == comp).protected = "race"
    return inv


def _use_stint(
    inv: Inventory, session: str, compound: str, fresh: bool, life_first: int, life_last: int
) -> None:
    """Apply one session stint to the inventory."""
    pool = [s for s in inv.available() if s.compound == compound]
    if fresh or life_first <= 1:
        new = [s for s in pool if s.new and s.protected != "q3"] or (
            [s for s in pool if s.new] if session == "Q" else []
        )
        if not new:
            inv.notes.append(f"{session}: new {compound} fitted but none left")
            return
        # Prefer an unprotected set; keep the race-protected ones if possible.
        chosen = sorted(new, key=lambda s: s.protected == "race")[0]
        if chosen.protected == "q3" and session == "Q":
            chosen.protected = ""
    else:
        target = life_first - 1
        used = [s for s in pool if not s.new]
        exact = [s for s in used if s.laps == target]
        if exact:
            chosen = exact[0]
        else:
            below = [s for s in used if s.laps < target]
            if not below:
                inv.notes.append(
                    f"{session}: used {compound} at TyreLife {life_first} but no matching set"
                )
                return
            chosen = max(below, key=lambda s: s.laps)
            inv.notes.append(
                f"{session}: used {compound} at TyreLife {life_first} "
                f"matched to set with {chosen.laps} laps"
            )
    chosen.laps = max(chosen.laps, life_last)
    chosen.history.append(f"{session}:{life_first}-{life_last}")


def _return(
    inv: Inventory,
    n: int,
    when: str,
    compound: str | None = None,
    most_laps_in: str | None = None,
    unused_order: tuple[str, ...] = ("SOFT", "MEDIUM", "HARD"),
) -> None:
    """Return ``n`` sets: most-used eligible set first; if none is used, an
    unused set in ``unused_order`` of compounds."""
    for _ in range(n):
        pool = [
            s
            for s in inv.available()
            if not s.protected and (compound is None or s.compound == compound)
        ]
        if most_laps_in is not None:
            pool = [s for s in pool if any(h.startswith(most_laps_in) for h in s.history)] or pool
        used = [s for s in pool if not s.new]
        if used:
            choice = max(used, key=lambda s: s.laps)
        elif pool:
            choice = min(pool, key=lambda s: (unused_order.index(s.compound), s.set_id))
            inv.notes.append(f"{when}: returned an unused {choice.compound} set (too few used)")
        else:
            inv.notes.append(f"{when}: no eligible set to return")
            return
        choice.returned = True
        choice.history.append(f"returned after {when}")


def _session_stints(session: fastf1.core.Session) -> pd.DataFrame:
    """One row per driver stint on a dry compound, in time order."""
    laps = session.laps
    laps = laps[laps["Compound"].isin(DRY)]
    return (
        laps.groupby(["Driver", "Stint"])
        .agg(
            compound=("Compound", "first"),
            fresh=("FreshTyre", "first"),
            life_first=("TyreLife", "min"),
            life_last=("TyreLife", "max"),
            start=("LapStartTime", "min"),
        )
        .reset_index()
        .sort_values(["Driver", "start"])
    )


def _load(year: int, round_number: int, name: str) -> fastf1.core.Session:
    configure_cache()
    s = fastf1.get_session(year, round_number, name)
    s.load(laps=True, telemetry=False, weather=False, messages=False)
    return s


def _stand_ins(session: fastf1.core.Session, race_drivers: set[str]) -> dict[str, str]:
    """Map practice-only drivers to the race driver of the same team who sat out."""
    laps = session.laps
    present = set(laps["Driver"].unique())
    teams = laps.groupby("Driver")["Team"].first()
    mapping = {}
    for driver in present - race_drivers:
        team = teams[driver]
        absent = [
            d
            for d in race_drivers - present
            if d in RACE_TEAMS.get(session.event["RoundNumber"], {}).get(team, set())
        ]
        if len(absent) == 1:
            mapping[driver] = absent[0]
    return mapping


#: Filled per round by reconstruct_inventories: team -> race drivers.
RACE_TEAMS: dict[int, dict[str, set[str]]] = {}


def reconstruct_inventories(
    year: int, round_number: int, unused_order: tuple[str, ...] = ("SOFT", "MEDIUM", "HARD")
) -> dict[str, Inventory]:
    """Race-day dry tyre inventory per driver, from practice and qualifying.

    Args:
        year: Season.
        round_number: Round.
        unused_order: Compound order for returning unused sets when a car has
            too few used ones (the rules do not say).

    Returns:
        Inventory per driver (three-letter code) at the start of the race.
    """
    configure_cache()
    event = fastf1.get_event(year, round_number)
    fmt = "sprint" if "sprint" in str(event["EventFormat"]).lower() else "conventional"
    order = ["FP1", "FP2", "FP3", "Q"] if fmt == "conventional" else ["FP1", "Q", "FP2", "Sprint"]
    sessions = {name: _load(year, round_number, name) for name in order}
    race = _load(year, round_number, "R")
    race_drivers = set(race.laps["Driver"].unique()) | set(race.results["Abbreviation"])
    RACE_TEAMS[round_number] = race.results.groupby("TeamName")["Abbreviation"].apply(set).to_dict()
    drivers = sorted(race_drivers)
    q3 = set(sessions["Q"].results.loc[sessions["Q"].results["Q3"].notna(), "Abbreviation"])
    invs = {d: _new_inventory(d, fmt) for d in drivers}
    for name in order:
        stints = _session_stints(sessions[name])
        stand_in = _stand_ins(sessions[name], race_drivers)
        for _, st in stints.iterrows():
            owner = stand_in.get(st["Driver"], st["Driver"])
            if owner not in invs:
                logger.warning(
                    "%s %s: %s has no race driver to charge", round_number, name, st["Driver"]
                )
                continue
            if owner != st["Driver"]:
                invs[owner].notes.append(f"{name}: run by stand-in {st['Driver']}")
            _use_stint(
                invs[owner],
                name,
                st["compound"],
                bool(st["fresh"]),
                int(st["life_first"]),
                int(st["life_last"]),
            )
        for d, inv in invs.items():
            if fmt == "conventional":
                if name in ("FP1", "FP2", "FP3"):
                    _return(inv, 2, name, unused_order=unused_order)
                elif name == "Q":
                    _release(inv, "q3")
                    if d in q3:
                        _return(inv, 1, "Q", compound=Q3_SPEC, unused_order=unused_order)
            else:
                if name == "FP1":
                    _return(inv, 1, name, unused_order=unused_order)
                elif name == "Q":
                    _release(inv, "q3")
                    _return(inv, 1, name, unused_order=unused_order)
                    if d in q3:
                        _return(inv, 1, "Q (Q3)", compound=Q3_SPEC, unused_order=unused_order)
                elif name == "FP2":
                    _return(inv, 1, name, most_laps_in="FP2", unused_order=unused_order)
                elif name == "Sprint":
                    _return(inv, 1, name, unused_order=unused_order)
    for inv in invs.values():
        _release(inv, "race")
    return invs


def _release(inv: Inventory, tag: str) -> None:
    for s in inv.sets:
        if s.protected == tag:
            s.protected = ""


def validate_against_race(
    invs: dict[str, Inventory], raw_laps: pd.DataFrame, round_number: int
) -> pd.DataFrame:
    """Check every set each car fitted in the race exists in its inventory.

    Race stints consume sets in order: a new set needs an unused set of that
    compound; a used set needs an available set of that compound whose laps
    equal the stint's first TyreLife minus one. A set used earlier in the race
    goes back into the pool with its laps, so a stint split by a red flag (same
    set before and after) matches. A stint with no TyreLife in the data is
    treated as a new set if marked fresh, else reported.

    Returns:
        One row per race stint that could not be matched (empty if none).
    """
    race = raw_laps[(raw_laps["RoundNumber"] == round_number) & raw_laps["Compound"].isin(DRY)]
    rows = []
    for driver, laps in race.groupby("Driver"):
        inv = invs.get(driver)
        stints = (
            laps.groupby("Stint")
            .agg(
                compound=("Compound", "first"),
                fresh=("FreshTyre", "first"),
                life=("TyreLife", "min"),
                life_last=("TyreLife", "max"),
                first_lap=("LapNumber", "min"),
                last_lap=("LapNumber", "max"),
            )
            .sort_values("first_lap")
        )
        avail = [] if inv is None else [[s.compound, s.laps] for s in inv.available()]
        for _, st in stints.iterrows():
            if pd.isna(st["life"]) and not bool(st["fresh"]):
                rows.append(
                    {
                        "Driver": driver,
                        "first_lap": int(st["first_lap"]),
                        "compound": st["compound"],
                        "set": "TyreLife missing in race data",
                        "laps_needed": None,
                        "nearest_laps": None,
                        "off_by": None,
                        "inventory": "" if inv is None else inv.summary(),
                    }
                )
                continue
            need_new = bool(st["fresh"]) or pd.isna(st["life"]) or st["life"] <= 1
            match = next(
                (
                    i
                    for i, (c, n) in enumerate(avail)
                    if c == st["compound"] and (n == 0 if need_new else n == st["life"] - 1)
                ),
                None,
            )
            if match is None:
                same = sorted(n for c, n in avail if c == st["compound"])
                want = 0 if need_new else int(st["life"]) - 1
                nearest = min(same, key=lambda n: abs(n - want)) if same else None
                rows.append(
                    {
                        "Driver": driver,
                        "first_lap": int(st["first_lap"]),
                        "compound": st["compound"],
                        "set": "new" if need_new else f"used, TyreLife {int(st['life'])}",
                        "laps_needed": want,
                        "nearest_laps": nearest,
                        "off_by": None if nearest is None else nearest - want,
                        "inventory": "" if inv is None else inv.summary(),
                    }
                )
            else:
                comp, laps_before = avail.pop(match)
                ran = (
                    int(st["life_last"])
                    if pd.notna(st["life_last"])
                    else laps_before + int(st["last_lap"] - st["first_lap"] + 1)
                )
                avail.append([comp, ran])
    return pd.DataFrame(
        rows,
        columns=[
            "Driver",
            "first_lap",
            "compound",
            "set",
            "laps_needed",
            "nearest_laps",
            "off_by",
            "inventory",
        ],
    )


def sets_left(
    inv: Inventory, raw_laps: pd.DataFrame, round_number: int, upto_lap: int
) -> list[tuple[str, int]]:
    """Sets still available after the race's stints that started by ``upto_lap``.

    Every stint that began on or before ``upto_lap`` (including the one on the
    car) uses up its set, matched as in :func:`validate_against_race`; a stint
    that cannot be matched takes the closest set of its compound. Returns
    (compound, out-lap TyreLife) pairs.
    """
    laps = raw_laps[
        (raw_laps["RoundNumber"] == round_number)
        & (raw_laps["Driver"] == inv.driver)
        & raw_laps["Compound"].isin(DRY)
        & (raw_laps["LapNumber"] <= upto_lap)
    ]
    avail = [[s.compound, s.laps] for s in inv.available()]
    stints = (
        laps.groupby("Stint")
        .agg(
            compound=("Compound", "first"),
            fresh=("FreshTyre", "first"),
            life=("TyreLife", "min"),
            first_lap=("LapNumber", "min"),
        )
        .sort_values("first_lap")
    )
    for _, st in stints.iterrows():
        same = [i for i, (c, _) in enumerate(avail) if c == st["compound"]]
        if not same:
            continue
        want = (
            0
            if (bool(st["fresh"]) or pd.isna(st["life"]) or st["life"] <= 1)
            else int(st["life"]) - 1
        )
        avail.pop(min(same, key=lambda i: abs(avail[i][1] - want)))
    return sorted((c, n + 1) for c, n in avail)


def race_used_sets(
    raw_laps: pd.DataFrame, round_number: int, driver: str, upto_lap: int
) -> list[tuple[str, int]]:
    """Sets the car already fitted in the race and took off by ``upto_lap``.

    Refitting a used race set is legal (Phase 6). Each completed dry stint
    gives one set, as (compound, out-lap TyreLife) = (compound, its last
    TyreLife + 1). The stint on the car at ``upto_lap`` is not included.
    """
    laps = raw_laps[
        (raw_laps["RoundNumber"] == round_number)
        & (raw_laps["Driver"] == driver)
        & (raw_laps["LapNumber"] <= upto_lap)
    ]
    if laps.empty:
        return []
    current = laps.loc[laps["LapNumber"].idxmax(), "Stint"]
    done = laps[(laps["Stint"] != current) & laps["Compound"].isin(DRY)]
    out = []
    for _, st in done.groupby("Stint"):
        life = st["TyreLife"].max()
        if pd.notna(life):
            out.append((str(st["Compound"].iloc[0]), int(life) + 1))
    return sorted(out)
