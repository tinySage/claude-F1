"""Race-event parameters for the field simulator (Phase 4 Part B), from 2022.

All estimates use the 18 dry races (no mixed-conditions race; see the ROADMAP
Decisions log).

- **Overtaking** (:func:`fit_pass_model`): a car that ended a lap within 1 s of
  the car directly ahead on the same lap number; did it end the next lap
  ahead? Excluded: lap 1, laps where either car pits (in or out lap), and laps
  with an SC / VSC / red-flag code for either car. The pace gap is the lap
  model's free-air prediction for the two cars on that lap (no dirty air).
  Logistic regression with a pooled pace slope and per-circuit intercepts,
  shrunk toward the pooled intercept by an L2 penalty.
- **Following gap** (:func:`fit_pass_model`): the median gap at the line when
  a car within 1 s did not pass, and the quantiles of those gaps (Phase 5: a
  held car's gap is drawn from them).
- **Passing calibration** (Phase 5, :mod:`f1strat.backtest`): a shift on every
  circuit intercept, fitted so that replayed even-round races have the actual
  number of on-track passes; ``main`` keeps the saved shift.
- **Safety cars** (:func:`fit_neutralisations`):
  - per-lap probability that an SC or VSC starts on a green race lap after
    lap 1;
  - each period's length in race laps (a run of neutralised laps with no
    green lap between is one period; it is an SC if any lap is under SC);
  - lap-time ratio under each (laps run entirely under SC or VSC, against the
    driver's median clean lap);
  - pit loss under each, relative to cars that stayed out on the same laps,
    as a ratio to the circuit's green-flag pit loss.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from f1strat.clean import MIXED_CONDITIONS_2022, pit_stops
from f1strat.data import PROCESSED_DIR

PARAMS_PATH = PROCESSED_DIR / "2022" / "sim_params.json"
NEUTRAL = "4567"
HELD_GAP_QUANTILES = 41


@dataclass
class PassModel:
    """P(pass on a lap) = sigmoid(intercept[circuit] + slope · pace gap)."""

    slope: float                  # per second the attacker is faster (free air)
    intercept: float              # pooled
    circuit: dict[int, float]     # round -> intercept (shrunk)
    follow_gap_s: float           # gap when held up
    n_events: int = 0
    n_passes: int = 0
    #: Phase 5 calibration (:mod:`f1strat.backtest`): shift on every circuit
    #: intercept, and the lap-time loss of a car that starts a green lap within
    #: 1 s of the car directly ahead.
    shift: float = 0.0
    dirty_air_s: float = 0.0
    #: Quantiles (0, 1/k, ..., 1) of the real next-lap gap of cars within 1 s
    #: that did not pass. Empty: every held car sits at ``follow_gap_s``.
    held_gaps: list[float] = field(default_factory=list)

    def p_pass(self, round_number: int, pace_gap: np.ndarray) -> np.ndarray:
        a = self.circuit.get(round_number, self.intercept) + self.shift
        return 1.0 / (1.0 + np.exp(-(a + self.slope * pace_gap)))


@dataclass
class Neutralisations:
    """SC / VSC frequency, length, lap-time and pit-loss ratios."""

    p_sc_per_lap: float
    p_vsc_per_lap: float
    sc_lengths: list[int]
    vsc_lengths: list[int]
    sc_lap_ratio: float           # SC lap / clean lap, median
    sc_catchup_ratio: float       # fastest-decile SC lap ratio (cars closing up)
    vsc_lap_ratio: float
    sc_pit_ratio: float           # pit loss under SC / green pit loss
    vsc_pit_ratio: float
    n_sc: int = 0
    n_vsc: int = 0
    green_laps: int = 0
    #: Phase 5: on SC laps after the first, each gap to the car ahead shrinks
    #: by this ratio per lap (down to ``sc_gap_s``); the first SC lap keeps the
    #: gaps. ``None``: Phase 4 behaviour (the field bunches in one lap at
    #: ``sc_catchup_ratio``).
    sc_close_ratio: float | None = None


@dataclass
class SimParams:
    """Everything the field simulator estimates from 2022 data."""

    passing: PassModel
    neutral: Neutralisations
    sc_gap_s: float = 0.8         # queue gap behind the car ahead under SC
    notes: list[str] = field(default_factory=list)
    #: Phase 6 (Silverstone): double-stack queue. Per track status
    #: (``"green"``, ``"sc"``, optional ``"vsc"``), ``D`` in seconds: when
    #: teammates pit on the same lap, the car behind at pit entry waits
    #: ``max(0, D − arrival gap)`` (:func:`double_stack_loss`). ``None``: no
    #: queue (Phases 4–5 and Hungary).
    double_stack: dict[str, float] | None = None

    def save(self, path: Path = PARAMS_PATH) -> Path:
        path.write_text(json.dumps(asdict(self), indent=1))
        return path

    @classmethod
    def load(cls, path: Path = PARAMS_PATH) -> "SimParams":
        d = json.loads(path.read_text())
        passing = d["passing"]
        passing["circuit"] = {int(k): v for k, v in passing["circuit"].items()}
        return cls(PassModel(**passing), Neutralisations(**d["neutral"]),
                   d["sc_gap_s"], d.get("notes", []), d.get("double_stack"))


def _dry(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df["EventName"].isin(list(MIXED_CONDITIONS_2022))]


def pass_events(cleaned: pd.DataFrame, fitted) -> pd.DataFrame:
    """Attacker/defender pairs within 1 s, with outcome and free-air pace gap."""
    from f1strat.deg_model import lap_features, predict

    feats = lap_features(_dry(cleaned))
    feats = feats[feats["Compound"].isin(["SOFT", "MEDIUM", "HARD"])].copy()
    feats["dirty_air"] = False
    feats = feats[feats["RoundNumber"].isin(fitted.races)]
    feats["pace"] = predict(fitted, feats, n_draws=300).mean
    return pair_events(feats)


def pair_events(feats: pd.DataFrame) -> pd.DataFrame:
    """The pass events of :func:`pass_events`, from laps with a ``pace`` column
    (free-air lap time estimate) and ``gap_ahead_s``."""
    key = ["RoundNumber", "Driver", "LapNumber"]
    lap = feats.set_index(key)
    neutral = lap["TrackStatus"].fillna("").str.contains(f"[{NEUTRAL}]")
    pit = lap["PitInTime"].notna() | lap["PitOutTime"].notna()

    rows = []
    base = feats[feats["Position"].notna() & (feats["LapNumber"] >= 1)]
    by_pos = base.set_index(["RoundNumber", "LapNumber", "Position"])["Driver"]
    for _, r in base[(base["gap_ahead_s"] < 1.0) & (base["Position"] > 1)].iterrows():
        rnd, L, pos = int(r["RoundNumber"]), int(r["LapNumber"]), int(r["Position"])
        defender = by_pos.get((rnd, L, pos - 1))
        if defender is None:
            continue
        a_next, d_next = (rnd, r["Driver"], L + 1), (rnd, defender, L + 1)
        if a_next not in lap.index or d_next not in lap.index:
            continue
        if neutral[a_next] or neutral[d_next] or pit[a_next] or pit[d_next] \
                or pit.get((rnd, r["Driver"], L), False) or pit.get((rnd, defender, L), False):
            continue
        pa, pd_ = lap.at[a_next, "Position"], lap.at[d_next, "Position"]
        if pd.isna(pa) or pd.isna(pd_):
            continue
        rows.append({"RoundNumber": rnd, "lap": L + 1, "attacker": r["Driver"],
                     "defender": defender, "gap_s": r["gap_ahead_s"],
                     "pace_gap_s": lap.at[d_next, "pace"] - lap.at[a_next, "pace"],
                     "passed": bool(pa < pd_),
                     "gap_next_s": lap.at[a_next, "gap_ahead_s"]})
    return pd.DataFrame(rows)


def stint_line_pace(cleaned: pd.DataFrame) -> pd.Series:
    """Free-air pace proxy without a lap model (e.g. a previous season's race).

    Per driver stint, a straight line through its clean laps' times against
    lap number (tyre age and fuel move together within a stint), evaluated on
    every lap of the stint. Stints with fewer than 3 clean laps get NaN.
    """
    out = pd.Series(np.nan, index=cleaned.index)
    lap_s = cleaned["LapTime"].dt.total_seconds()
    clean = cleaned["exclude_reason"].isna() & lap_s.notna()
    for _, g in cleaned.groupby(["RoundNumber", "Driver", "Stint"]):
        c = g[clean.loc[g.index]]
        if len(c) < 3:
            continue
        b, a = np.polyfit(c["LapNumber"].astype(float), lap_s.loc[c.index], 1)
        out.loc[g.index] = a + b * g["LapNumber"].astype(float)
    return out


def circuit_intercept_from(events: pd.DataFrame, slope: float, intercept: float,
                           penalty: float = 1.0) -> float:
    """One circuit's pass intercept from its own events, with the slope fixed
    and the deviation shrunk toward ``intercept`` (as in :func:`fit_pass_model`)."""
    ev = events.dropna(subset=["pace_gap_s"])
    if ev.empty:
        return intercept
    x = np.ones((len(ev), 1))
    offset = intercept + slope * ev["pace_gap_s"].to_numpy()
    dev = _penalised_offsets(x, ev["passed"].to_numpy(float), offset, penalty)
    return intercept + float(dev[0])


def previous_race(year: int, event_name: str, back: int = 4) -> tuple[int, pd.DataFrame]:
    """The most recent earlier running of a Grand Prix, as cleaned laps.

    Looks back up to ``back`` seasons for an event with exactly this name
    (e.g. the 2022 Canadian GP's previous running is 2019: 2020 and 2021 were
    cancelled). FastF1 fuzzy-matches event names, so the schedule is checked
    for an exact match first.
    """
    import fastf1

    from f1strat.clean import clean_laps
    from f1strat.data import configure_cache, laps_table

    configure_cache()
    for y in range(year - 1, year - 1 - back, -1):
        schedule = fastf1.get_event_schedule(y, include_testing=False)
        hit = schedule[schedule["EventName"] == event_name]
        if hit.empty:
            continue
        session = fastf1.get_session(y, int(hit["RoundNumber"].iloc[0]), "R")
        session.load(laps=True, telemetry=False, weather=True, messages=False)
        try:
            laps = laps_table(session)
        except fastf1.exceptions.DataNotLoadedError as err:
            # 2019 and earlier timing now needs an F1 account (none used here).
            raise ValueError(f"{y} {event_name} timing data cannot be downloaded") from err
        return y, clean_laps(laps, mixed_conditions=())
    raise ValueError(f"no earlier {event_name} in the last {back} seasons")


#: A race counts as dry for a season's pass model if at least this share of
#: its laps with a compound are on dry compounds.
DRY_SHARE = 0.5


def season_pass_events(year: int, refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pass events for every dry race of a season, with a stint-line pace proxy.

    The same events and pace as :func:`previous_race` + :func:`pair_events`
    (no lap model is needed for a past season), for every race, so circuits
    and the season average are estimated on one footing. Cached in
    ``data/processed/<year>/pass_events.parquet`` with a per-race status table
    (``pass_events_status.csv``: ``used``, ``wet`` or ``download failed``).

    Returns:
        ``(events, status)``; events carry ``EventName``.
    """
    import fastf1

    from f1strat.clean import clean_laps
    from f1strat.data import configure_cache, laps_table
    from f1strat.deg_model import lap_features

    out_dir = PROCESSED_DIR / str(year)
    ev_path, st_path = out_dir / "pass_events.parquet", out_dir / "pass_events_status.csv"
    if ev_path.exists() and st_path.exists() and not refresh:
        return pd.read_parquet(ev_path), pd.read_csv(st_path)
    configure_cache()
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    frames, status = [], []
    for _, ev in schedule.iterrows():
        rnd, name = int(ev["RoundNumber"]), str(ev["EventName"])
        try:
            session = fastf1.get_session(year, rnd, "R")
            session.load(laps=True, telemetry=False, weather=True, messages=False)
            laps = clean_laps(laps_table(session), mixed_conditions=())
        except Exception as err:  # noqa: BLE001 - record and move on
            status.append({"RoundNumber": rnd, "EventName": name, "status": "download failed",
                           "detail": f"{type(err).__name__}: {err}"[:200]})
            continue
        known = laps["Compound"].isin(["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"])
        dry_share = float(laps.loc[known, "Compound"].isin(["SOFT", "MEDIUM", "HARD"]).mean()) \
            if known.any() else 0.0
        if dry_share < DRY_SHARE:
            status.append({"RoundNumber": rnd, "EventName": name, "status": "wet",
                           "detail": f"dry share {dry_share:.2f}"})
            continue
        feats = lap_features(laps)
        feats = feats[feats["Compound"].isin(["SOFT", "MEDIUM", "HARD"])].copy()
        feats["pace"] = stint_line_pace(laps).loc[feats.index]
        events = pair_events(feats)
        if len(events):
            frames.append(events.assign(EventName=name))
        status.append({"RoundNumber": rnd, "EventName": name, "status": "used",
                       "detail": f"dry share {dry_share:.2f}; {len(events)} events, "
                                 f"{int(events['passed'].sum()) if len(events) else 0} passes"})
    events = pd.concat(frames, ignore_index=True)
    status = pd.DataFrame(status)
    out_dir.mkdir(parents=True, exist_ok=True)
    events.to_parquet(ev_path, index=False)
    status.to_csv(st_path, index=False)
    return events, status


def relative_circuit_intercept(event_name: str, pooled_now: float, year_ref: int) -> dict:
    """Phase 6 method for Silverstone and Monaco (``docs/case_studies.md``):
    the current pooled intercept shifted by how much harder (or easier) the
    circuit was than the average in ``year_ref``.

    ``intercept = pooled_now + (c_ref[circuit] − a_ref)``, where ``a_ref`` and
    ``c_ref`` come from one pass model fitted to every dry ``year_ref`` race
    (:func:`season_pass_events`), so the circuit and the average share a pace
    proxy, slope and shrinkage.
    """
    events, status = season_pass_events(year_ref)
    ref = fit_pass_model(events.dropna(subset=["pace_gap_s"]))
    rounds = events.groupby("RoundNumber")["EventName"].first()
    by_name = {rounds[r]: v for r, v in ref.circuit.items() if r in rounds.index}
    if event_name not in by_name:
        raise ValueError(f"no dry {year_ref} {event_name} in the pass events")
    shift = by_name[event_name] - ref.intercept
    return {"intercept": pooled_now + shift, "shift": shift,
            "ref_circuit": by_name[event_name], "ref_average": ref.intercept,
            "ref_slope": ref.slope, "ref_events": ref.n_events, "ref_year": year_ref,
            "ref_races_used": int((status["status"] == "used").sum()),
            "ref_races_not_used": status.loc[status["status"] != "used",
                                             ["EventName", "status", "detail"]
                                             ].to_dict("records")}


def in_race_pass_update(cleaned_cut: pd.DataFrame, fitted, round_number: int,
                        prior_intercept: float, slope: float, first_racing_lap: int = 1,
                        exclude_teammates: bool = False) -> dict:
    """Update a circuit's pass intercept with the race's own passes so far.

    The events are those of :func:`pass_events` for ``round_number`` in the
    cutoff data (the pace from the cutoff fit), so lap 1, in and out laps of
    either car, SC/VSC laps and lapping are already excluded (pairs are
    adjacent in the classification, within 1 s on the same lap number).
    Outcome laps up to ``first_racing_lap`` are also dropped (Silverstone
    2022: lap 1 was red-flagged and lap 2 was the restart's opening lap).
    The intercept is :func:`circuit_intercept_from` with the slope fixed and
    the deviation shrunk toward ``prior_intercept`` (the same L2 penalty as
    the season fits). ``exclude_teammates`` drops same-team pairs (e.g. a
    team-order swap).
    """
    ev = pass_events(cleaned_cut[cleaned_cut["RoundNumber"] == round_number], fitted)
    ev = ev[ev["lap"] > first_racing_lap]
    if exclude_teammates and len(ev):
        team = cleaned_cut[cleaned_cut["RoundNumber"] == round_number] \
            .groupby("Driver")["Team"].first()
        ev = ev[ev["attacker"].map(team) != ev["defender"].map(team)]
    value = circuit_intercept_from(ev, slope, prior_intercept)
    return {"prior": prior_intercept, "intercept": value, "events": int(len(ev)),
            "passes": int(ev["passed"].sum()) if len(ev) else 0,
            "pass_list": ev.loc[ev["passed"], ["lap", "attacker", "defender", "pace_gap_s"]]
            .round(3).to_dict("records") if len(ev) else []}


def fit_pass_model(events: pd.DataFrame, penalty: float = 1.0) -> PassModel:
    """Logistic regression with per-circuit intercepts shrunk to the pooled one."""
    from sklearn.linear_model import LogisticRegression

    ev = events.dropna(subset=["pace_gap_s"])
    pooled = LogisticRegression(C=1e6).fit(ev[["pace_gap_s"]], ev["passed"])
    a0, b = float(pooled.intercept_[0]), float(pooled.coef_[0, 0])
    # Circuit deviations from the pooled fit, penalised toward 0 (offset model).
    x = pd.get_dummies(ev["RoundNumber"]).astype(float)
    offset = a0 + b * ev["pace_gap_s"].to_numpy()
    dev = _penalised_offsets(x.to_numpy(), ev["passed"].to_numpy(float), offset, penalty)
    circuit = {int(r): a0 + float(d) for r, d in zip(x.columns, dev)}
    held = ev[~ev["passed"] & ev["gap_next_s"].notna()]
    quantiles = np.quantile(held["gap_next_s"], np.linspace(0, 1, HELD_GAP_QUANTILES))
    return PassModel(slope=b, intercept=a0, circuit=circuit,
                     follow_gap_s=float(held["gap_next_s"].median()),
                     n_events=len(ev), n_passes=int(ev["passed"].sum()),
                     held_gaps=[round(float(q), 3) for q in quantiles])


def _penalised_offsets(x: np.ndarray, y: np.ndarray, offset: np.ndarray,
                       penalty: float, iters: int = 200) -> np.ndarray:
    """Newton steps for a logistic model with a fixed offset and L2-penalised
    group intercepts (one column of ``x`` per circuit)."""
    w = np.zeros(x.shape[1])
    for _ in range(iters):
        eta = offset + x @ w
        p = 1 / (1 + np.exp(-eta))
        grad = x.T @ (y - p) - penalty * w
        hess = (x * (p * (1 - p))[:, None]).T @ x + penalty * np.eye(x.shape[1])
        step = np.linalg.solve(hess, grad)
        w += step
        if np.abs(step).max() < 1e-8:
            break
    return w


def neutral_runs(race_one: pd.DataFrame) -> list[tuple[str, int, int]]:
    """Neutralised periods of one race as ``(kind, first lap, laps)``.

    A period is a run of consecutive SC / VSC / red-flag laps with no green lap
    in between. Its kind is ``"SafetyCar"`` if any lap of it is under SC (or
    red flag), else ``"VSC"``. So a VSC that turns into an SC is one SC period,
    not a VSC plus an SC (Phase 5 fix: the race table numbers them as two), and
    SC periods split by a status blip with no green lap are one.
    """
    r = race_one.sort_values("LapNumber")
    out: list[tuple[str, int, int]] = []
    run: list[tuple[int, bool]] = []
    for lap, sc, vsc, red in zip(r["LapNumber"], r["SafetyCar"], r["VSC"], r["RedFlag"]):
        if sc or vsc or red:
            run.append((int(lap), bool(sc or red)))
            continue
        if run:
            out.append(("SafetyCar" if any(s for _, s in run) else "VSC", run[0][0], len(run)))
            run = []
    if run:
        out.append(("SafetyCar" if any(s for _, s in run) else "VSC", run[0][0], len(run)))
    return out


def fit_neutralisations(cleaned: pd.DataFrame, race: pd.DataFrame) -> Neutralisations:
    """SC/VSC hazard, lengths, lap-time ratios and pit-loss ratios (dry races)."""
    race = _dry(race)
    starts = {"SafetyCar": 0, "VSC": 0}
    lengths: dict[str, list[int]] = {"SafetyCar": [], "VSC": []}
    green = 0
    for _, r in race.groupby("RoundNumber"):
        r = r.sort_values("LapNumber")
        green += int(((~r["SafetyCar"]) & (~r["VSC"]) & (~r["RedFlag"])
                      & (r["LapNumber"] >= 2)).sum())
        for kind, first, n in neutral_runs(r):
            if first >= 2:
                starts[kind] += 1
                lengths[kind].append(n)

    laps = _dry(cleaned)
    lap_s = laps["LapTime"].dt.total_seconds()
    med = lap_s.where(laps["exclude_reason"].isna()).groupby(
        [laps["RoundNumber"], laps["Driver"]]).transform("median")
    ratio = lap_s / med
    status = laps["TrackStatus"].fillna("")
    no_pit = laps["PitInTime"].isna() & laps["PitOutTime"].isna()
    sc_only = status.str.fullmatch("4+") & no_pit
    vsc_only = status.str.fullmatch("[67]+") & no_pit
    sc_ratios, vsc_ratios = ratio[sc_only].dropna(), ratio[vsc_only].dropna()

    sc_pit, vsc_pit = _neutral_pit_ratios(cleaned)
    return Neutralisations(
        p_sc_per_lap=starts["SafetyCar"] / green, p_vsc_per_lap=starts["VSC"] / green,
        sc_lengths=lengths["SafetyCar"], vsc_lengths=lengths["VSC"],
        sc_lap_ratio=float(sc_ratios.median()),
        sc_catchup_ratio=float(sc_ratios.quantile(0.1)),
        vsc_lap_ratio=float(vsc_ratios.median()),
        sc_pit_ratio=sc_pit, vsc_pit_ratio=vsc_pit,
        n_sc=starts["SafetyCar"], n_vsc=starts["VSC"], green_laps=green)


def _neutral_pit_ratios(cleaned: pd.DataFrame, return_samples: bool = False):
    """Pit loss under SC / VSC as a ratio to the circuit's green-flag loss.

    For a stop whose in lap is under SC (or VSC): (in lap + out lap) minus the
    median (lap L + lap L+1) of cars that did not pit on those laps.
    """
    laps = _dry(cleaned)
    lap_s = laps["LapTime"].dt.total_seconds()
    t = laps.assign(lap_s=lap_s).set_index(["RoundNumber", "Driver", "LapNumber"])["lap_s"]
    stops = pit_stops(laps)
    green = stops[stops["green"]].groupby("RoundNumber")["loss_s"].median()
    pitting = set(zip(laps.loc[laps["PitInTime"].notna() | laps["PitOutTime"].notna(), "RoundNumber"],
                      laps.loc[laps["PitInTime"].notna() | laps["PitOutTime"].notna(), "LapNumber"],
                      laps.loc[laps["PitInTime"].notna() | laps["PitOutTime"].notna(), "Driver"]))
    status = laps.set_index(["RoundNumber", "Driver", "LapNumber"])["TrackStatus"].fillna("")
    out = {"sc": [], "vsc": []}
    for _, s in stops.iterrows():
        rnd, d, L = int(s["RoundNumber"]), s["Driver"], int(s["InLap"])
        code = status.get((rnd, d, L), "")
        kind = "sc" if "4" in code else "vsc" if ("6" in code or "7" in code) else None
        if kind is None or L < 2 or rnd not in green.index or pd.isna(s["in_s"] + s["out_s"]):
            continue
        others = laps[(laps["RoundNumber"] == rnd) & laps["LapNumber"].isin([L, L + 1])]
        others = others[[(rnd, ln, dr) not in pitting and (rnd, L, dr) not in pitting
                         and (rnd, L + 1, dr) not in pitting
                         for ln, dr in zip(others["LapNumber"], others["Driver"])]]
        pair = others.groupby("Driver")["LapTime"].apply(
            lambda x: x.dt.total_seconds().sum() if len(x) == 2 else np.nan).dropna()
        if len(pair) < 3:
            continue
        loss = s["in_s"] + s["out_s"] - float(pair.median())
        out[kind].append(loss / green[rnd])
    if return_samples:
        return out["sc"], out["vsc"]
    return float(np.median(out["sc"])), float(np.median(out["vsc"]))


def pit_lane_stops(raw: pd.DataFrame) -> pd.DataFrame:
    """Every stop with its pit-lane time (pit entry to pit exit, s) and the
    track status of its in or out lap (``green``, ``sc`` or ``vsc``).

    FastF1 has no stationary times, so the pit-lane time is the only stop
    duration available. Stops whose in lap is lap 1 and pit-lane times outside
    10–60 s (red flags, garage stops) are left out.
    """
    laps = raw.sort_values(["RoundNumber", "Driver", "LapNumber"])
    inl = laps.loc[laps["PitInTime"].notna(),
                   ["RoundNumber", "EventName", "Driver", "Team", "LapNumber", "PitInTime",
                    "TrackStatus"]]
    out = laps.loc[laps["PitOutTime"].notna(),
                   ["RoundNumber", "Driver", "LapNumber", "PitOutTime", "TrackStatus"]]
    out = out.assign(LapNumber=out["LapNumber"] - 1)
    st = inl.merge(out, on=["RoundNumber", "Driver", "LapNumber"], suffixes=("", "_out"))
    st["lane_s"] = (st["PitOutTime"] - st["PitInTime"]).dt.total_seconds()
    st = st[(st["LapNumber"] > 1) & st["lane_s"].between(10, 60)].copy()
    code = st["TrackStatus"].fillna("").astype(str) + st["TrackStatus_out"].fillna("").astype(str)
    st["kind"] = np.where(code.str.contains("4"), "sc",
                          np.where(code.str.contains("[67]"), "vsc", "green"))
    st["team_stops_on_lap"] = st.groupby(["RoundNumber", "Team", "LapNumber"])["Driver"] \
        .transform("size")
    return st.reset_index(drop=True)


def double_stack_pairs(raw: pd.DataFrame) -> pd.DataFrame:
    """Same-lap teammate stops: arrival gap at pit entry and the second car's
    extra pit-lane time over the race's solo stops (same track status if that
    race has 3+ such solo stops, else all its solo stops)."""
    st = pit_lane_stops(raw)
    solo = st[st["team_stops_on_lap"] == 1]
    by_kind = solo.groupby(["RoundNumber", "kind"])["lane_s"].agg(["median", "size"])
    by_race = solo.groupby("RoundNumber")["lane_s"].median()

    def base(rnd: int, kind: str) -> float:
        key = (rnd, kind)
        if key in by_kind.index and by_kind.loc[key, "size"] >= 3:
            return float(by_kind.loc[key, "median"])
        return float(by_race[rnd])

    rows = []
    for (rnd, team, lap), g in st[st["team_stops_on_lap"] == 2].groupby(
            ["RoundNumber", "Team", "LapNumber"]):
        if rnd not in by_race.index:
            continue
        a, b = g.sort_values("PitInTime").iloc[0], g.sort_values("PitInTime").iloc[1]
        ref = base(int(rnd), str(b["kind"]))
        rows.append({"RoundNumber": int(rnd), "EventName": a["EventName"], "Team": team,
                     "lap": int(lap), "first": a["Driver"], "second": b["Driver"],
                     "kind": b["kind"],
                     "gap_s": (b["PitInTime"] - a["PitInTime"]).total_seconds(),
                     "first_extra_s": a["lane_s"] - ref, "extra_s": b["lane_s"] - ref})
    return pd.DataFrame(rows)


def double_stack_loss(pairs: pd.DataFrame, kinds=("green", "sc")) -> dict[str, dict]:
    """``D`` per track status for the double-stack queue.

    The second car waits for the first car's stationary time plus a margin
    (``D``), less the gap by which it arrived later: ``extra = max(0, D − gap)``.
    ``D`` is fitted by least absolute deviation (slow stops and timing noise
    would dominate a squared-error fit) on a 0.05 s grid. Also reports how many
    pairs arrived closer than ``D`` (the pairs that identify it).
    """
    out = {}
    grid = np.arange(0.0, 15.0001, 0.05)
    for kind in kinds:
        k = pairs[pairs["kind"] == kind]
        if k.empty:
            out[kind] = {"D_s": None, "n_pairs": 0, "n_queued": 0}
            continue
        cost = [np.abs(k["extra_s"] - np.maximum(0.0, D - k["gap_s"])).sum() for D in grid]
        D = float(grid[int(np.argmin(cost))])
        out[kind] = {"D_s": round(D, 2), "n_pairs": int(len(k)),
                     "n_queued": int((k["gap_s"] < D).sum())}
    return out


def neutral_pit_samples(cleaned: pd.DataFrame) -> dict[str, int]:
    """How many SC and VSC stops the pit-loss ratios of
    :func:`fit_neutralisations` rest on (same selection as the ratios)."""
    sc, vsc = _neutral_pit_ratios(cleaned, return_samples=True)
    return {"sc": len(sc), "vsc": len(vsc)}


def estimate(fitted, cleaned: pd.DataFrame, race: pd.DataFrame) -> SimParams:
    """All field-simulator parameters from 2022."""
    events = pass_events(cleaned, fitted)
    return SimParams(passing=fit_pass_model(events),
                     neutral=fit_neutralisations(cleaned, race))


def main() -> None:
    """Estimate and save ``data/processed/2022/sim_params.json``."""
    from f1strat.clean import clean_laps
    from f1strat.data import load_season
    from f1strat.deg_model import THIN_FIT_PATH, load_fit

    raw = load_season(2022)
    params = estimate(load_fit(THIN_FIT_PATH), clean_laps(raw), load_season(2022, table="race"))
    if PARAMS_PATH.exists():  # keep the Phase 5 calibration
        old = SimParams.load().passing
        params.passing.shift, params.passing.dirty_air_s = old.shift, old.dirty_air_s
        params.neutral.sc_close_ratio = SimParams.load().neutral.sc_close_ratio
    print(json.dumps(asdict(params), indent=1)[:3000])
    print(f"wrote {params.save()}")


if __name__ == "__main__":
    main()
