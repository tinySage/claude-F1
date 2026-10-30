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
  - each period's length in race laps;
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


@dataclass
class SimParams:
    """Everything the field simulator estimates from 2022 data."""

    passing: PassModel
    neutral: Neutralisations
    sc_gap_s: float = 0.8         # queue gap behind the car ahead under SC
    notes: list[str] = field(default_factory=list)

    def save(self, path: Path = PARAMS_PATH) -> Path:
        path.write_text(json.dumps(asdict(self), indent=1))
        return path

    @classmethod
    def load(cls, path: Path = PARAMS_PATH) -> "SimParams":
        d = json.loads(path.read_text())
        passing = d["passing"]
        passing["circuit"] = {int(k): v for k, v in passing["circuit"].items()}
        return cls(PassModel(**passing), Neutralisations(**d["neutral"]),
                   d["sc_gap_s"], d.get("notes", []))


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
        for kind in starts:
            col = f"{kind}Period"
            periods = r[r[col].notna()].groupby(col)["LapNumber"]
            for _, laps in periods:
                if laps.min() >= 2:
                    starts[kind] += 1
                    lengths[kind].append(int(laps.max() - laps.min() + 1))

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


def _neutral_pit_ratios(cleaned: pd.DataFrame) -> tuple[float, float]:
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
    return float(np.median(out["sc"])), float(np.median(out["vsc"]))


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
    print(json.dumps(asdict(params), indent=1)[:3000])
    print(f"wrote {params.save()}")


if __name__ == "__main__":
    main()
