"""Tests for the Phase 5 backtest layer (forced SC/VSC, retirements, counters)."""

import dataclasses

import numpy as np
import pytest

from f1strat.backtest import (
    actual_race,
    close_share,
    count_passes,
    neutral_counts,
    run_backtest,
)
from f1strat.simulator import GREEN, SC, VSC, Plan, RaceState, build_field, simulate_field

SPAIN, SAUDI, BAHRAIN = 6, 2, 1


@pytest.fixture(scope="module")
def data():
    from f1strat.clean import clean_laps
    from f1strat.data import load_season
    from f1strat.deg_model import THIN_FIT_PATH, load_fit
    from f1strat.race_events import SimParams

    raw = load_season(2022)
    return {"raw": raw, "race": load_season(2022, table="race"), "cleaned": clean_laps(raw),
            "fitted": load_fit(THIN_FIT_PATH), "params": SimParams.load()}


def test_count_passes_simple() -> None:
    """Car 1 passes car 0 on lap 1; lap 2 is an SC lap; car 2 pits on lap 3."""
    t = np.array([[[0.0, 1.0, 2.0], [10.5, 10.0, 12.0], [20.0, 21.0, 22.0],
                   [30.0, 31.0, 29.0]]])
    pit = np.zeros_like(t, dtype=bool)
    pit[0, 3, 2] = True
    green = np.array([False, True, False, True])
    assert count_passes(t, pit, green)[0] == 1
    close, total = close_share(t, pit, green)
    # Lap 1: gaps 0.5 (close) and 1.5. Lap 3: car 2 is on an in lap (not
    # counted); cars 0 and 1 are each 1.0 s behind the car ahead (not close).
    assert close[0] == 1 and total[0] == 4


def test_neutral_counts_ignores_running_period() -> None:
    s = np.array([[SC, SC, GREEN, VSC, VSC, GREEN, SC]])
    sc, vsc = neutral_counts(s, before=SC)
    assert sc[0] == 1 and vsc[0] == 1


def test_vsc_turning_into_sc_is_one_sc() -> None:
    s = np.array([[GREEN, VSC, SC, SC, GREEN, VSC, GREEN]])
    sc, vsc = neutral_counts(s)
    assert sc[0] == 1 and vsc[0] == 1


def test_neutral_runs_merge_vsc_into_sc(data) -> None:
    """Saudi Arabia: the lap-16 VSC turned into an SC (one SC period), plus a
    later VSC."""
    from f1strat.race_events import neutral_runs

    r = data["race"][data["race"]["RoundNumber"] == SAUDI]
    kinds = [k for k, first, _ in neutral_runs(r) if first >= 2]
    assert kinds == ["SafetyCar", "VSC"]


def test_actual_race_retirements(data) -> None:
    """Bahrain: Verstappen and Pérez (stopped on the final lap) retired; Leclerc won."""
    act = actual_race(data["raw"], data["race"], BAHRAIN)
    assert act.order[0] == "LEC"
    assert not act.finished["VER"] and not act.finished["PER"]
    assert act.last_lap["PER"] == 56


def test_forced_status_and_retirement(data) -> None:
    raw = data["raw"]
    state = RaceState.from_laps(raw, SPAIN, lap=1, total_laps=66)
    forced = np.full(65, GREEN, np.int8)
    forced[10:14] = SC
    losses = np.array([22.0])
    field = build_field(data["fitted"], state, losses, data["params"], n_sims=20, seed=1,
                        status=forced)
    assert (field.status == forced[None, :]).all()
    res = simulate_field(field, {d: Plan() for d in field.drivers}, retire={"VER": 30},
                         record=True)
    c = field.drivers.index("VER")
    assert np.isinf(res.times[:, c]).all()
    assert np.isfinite(res.history[:, 28, c]).all() and np.isinf(res.history[:, 29, c]).all()
    assert (res.positions[:, c] == len(field.drivers)).all()


def test_replay_backtest_runs(data) -> None:
    bt = run_backtest(data["fitted"], data["raw"], data["race"], data["cleaned"],
                      data["params"], SAUDI, "replay", n_sims=30)
    assert bt.positions.shape == (30, len(bt.drivers))
    # Finishers are ranked 1..n in every simulation; retired cars are nan.
    n_fin = len(bt.actual.order)
    assert (np.nansum(bt.positions, axis=1) == n_fin * (n_fin + 1) / 2).all()
    # Replay: every simulation has the actual SC/VSC count.
    assert (bt.sc_counts == bt.sc_counts[0]).all()
