"""Phase 6, Monaco 2022: a track-position accounting study (``docs/case_studies.md``).

Monaco 2022 was a wet-to-dry race, outside the dry lap model and the
simulator. Almost nobody passes at Monaco (2021: 0 passes in 115 close
battles), so the order after the stop cycle largely sets the result. This
module reconstructs the crossover from the timing data (Part A). The
counterfactual accounting is added only after its method is pre-registered.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ROUND = 7
FERRARI_RB = ("LEC", "SAI", "PER", "VER")
TOP8 = ("PER", "SAI", "VER", "LEC", "RUS", "NOR", "ALO", "HAM")


def race_laps(raw: pd.DataFrame) -> pd.DataFrame:
    """Monaco's laps with ``lap_s`` (s) and ``time_s`` (session time at the line)."""
    s = raw[raw["RoundNumber"] == ROUND].copy()
    s["lap_s"] = s["LapTime"].dt.total_seconds()
    s["time_s"] = s["Time"].dt.total_seconds()
    return s


def stops(raw: pd.DataFrame, drivers=FERRARI_RB) -> pd.DataFrame:
    """Each stop: in lap, compounds off and on, pit entry, pit-lane time (s)."""
    from f1strat.race_events import pit_lane_stops

    pl = pit_lane_stops(raw)
    pl = pl[(pl["RoundNumber"] == ROUND) & pl["Driver"].isin(drivers)]
    s = race_laps(raw).set_index(["Driver", "LapNumber"])
    rows = []
    for _, r in pl.sort_values(["LapNumber", "PitInTime"]).iterrows():
        d, L = r["Driver"], int(r["LapNumber"])
        rows.append({"driver": d, "in_lap": L, "off": s.at[(d, L), "Compound"],
                     "on": s.at[(d, L + 1), "Compound"],
                     "on_tyre_life": s.at[(d, L + 1), "TyreLife"],
                     "pit_entry": str(r["PitInTime"])[7:18], "lane_s": round(r["lane_s"], 3),
                     "team_stops_on_lap": int(r["team_stops_on_lap"])})
    return pd.DataFrame(rows)


def order_and_gaps(raw: pd.DataFrame, laps=range(12, 33), drivers=TOP8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(positions, gap to the leader in s at the line) per lap for ``drivers``."""
    s = race_laps(raw)
    pos = s.pivot(index="LapNumber", columns="Driver", values="Position")
    t = s.pivot(index="LapNumber", columns="Driver", values="time_s")
    gap = t.sub(t.min(axis=1), axis=0)
    idx = list(laps)
    return pos.loc[idx, list(drivers)], gap.loc[idx, list(drivers)].round(2)


def queue_time(raw: pd.DataFrame, driver: str = "LEC", lap: int = 21) -> dict:
    """The double-stack queue from pit-lane times: the second car's pit-lane
    time minus the race's solo green median (and minus its own other stop),
    with the arrival gap at pit entry and the season ``D`` model's prediction."""
    from f1strat.race_events import double_stack_loss, double_stack_pairs, pit_lane_stops

    pl = pit_lane_stops(raw)
    pl = pl[pl["RoundNumber"] == ROUND]
    solo = pl[(pl["team_stops_on_lap"] == 1) & (pl["kind"] == "green")]
    me = pl[(pl["Driver"] == driver) & (pl["LapNumber"] == lap)].iloc[0]
    mate = pl[(pl["Team"] == me["Team"]) & (pl["LapNumber"] == lap)
              & (pl["Driver"] != driver)].iloc[0]
    own_other = pl[(pl["Driver"] == driver) & (pl["LapNumber"] != lap) & (pl["kind"] == "green")]
    gap = (me["PitInTime"] - mate["PitInTime"]).total_seconds()
    D = double_stack_loss(double_stack_pairs(raw))["green"]["D_s"]
    q = me["lane_s"] - solo["lane_s"].median()
    return {"driver": driver, "lap": lap, "behind": mate["Driver"], "lane_s": round(me["lane_s"], 3),
            "teammate_lane_s": round(mate["lane_s"], 3),
            "solo_green_median_s": round(float(solo["lane_s"].median()), 3),
            "n_solo_green": int(len(solo)),
            "queue_vs_solo_s": round(float(q), 3),
            "queue_vs_own_other_stop_s": (round(float(me["lane_s"] - own_other["lane_s"].median()), 3)
                                          if len(own_other) else None),
            "arrival_gap_s": round(gap, 3), "implied_D_s": round(float(q + gap), 3),
            "season_D_s": D, "season_model_queue_s": round(max(0.0, D - gap), 3)}


def passes(raw: pd.DataFrame, race: pd.DataFrame) -> dict:
    """On-track passes on green laps with neither car on an in or out lap
    (the Phase 5 counter), by lap, with who passed whom."""
    from f1strat.backtest import actual_race
    from f1strat.simulator import GREEN

    act = actual_race(raw, race, ROUND)
    out = []
    for L in range(2, act.total_laps + 1):
        if act.status[L] != GREEN:
            continue
        a, b = act.times[L - 1], act.times[L]
        for i, di in enumerate(act.drivers):
            for j, dj in enumerate(act.drivers):
                if (np.isfinite(a[i]) and np.isfinite(a[j]) and np.isfinite(b[i])
                        and np.isfinite(b[j]) and a[i] < a[j] and b[i] > b[j]
                        and not act.pit_laps[L, i] and not act.pit_laps[L, j]):
                    out.append({"lap": L, "passer": dj, "passed": di})
    df = pd.DataFrame(out)
    top8 = set(act.order[:8])
    return {"n": len(df), "list": df.to_dict("records"),
            "involving_top8": int((df["passer"].isin(top8) | df["passed"].isin(top8)).sum())
            if len(df) else 0,
            "after_lap_23": int((df["lap"] > 23).sum()) if len(df) else 0,
            "finish_order": act.order}


def main() -> None:
    """``python -m f1strat.monaco``: write the Part A reconstruction to
    ``reports/figures/case_monaco_reconstruction.json``."""
    import json
    import logging

    from f1strat.backtest import FIG_DIR
    from f1strat.data import load_season

    logging.disable(logging.WARNING)
    raw, race = load_season(2022), load_season(2022, table="race")
    pos, gap = order_and_gaps(raw)
    out = {"stops": stops(raw).to_dict("records"),
           "positions": {int(k): v for k, v in pos.astype(float).to_dict("index").items()},
           "gaps": {int(k): v for k, v in gap.to_dict("index").items()},
           "queue": queue_time(raw), "queue_red_bull": queue_time(raw, "VER", 22),
           "passes": passes(raw, race)}
    (FIG_DIR / "case_monaco_reconstruction.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: out[k] for k in ("stops", "queue", "queue_red_bull")}, indent=1,
                     default=str))
    print({k: v for k, v in out["passes"].items() if k != "list"})


if __name__ == "__main__":
    main()
