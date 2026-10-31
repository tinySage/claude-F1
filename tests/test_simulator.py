"""Tests for the Phase 4 simulator, Part A: state, single car, strategy search."""

import numpy as np
import pandas as pd
import pytest

from f1strat.clean import clean_laps
from f1strat.data import load_season
from f1strat.deg_model import THIN_FIT_PATH, load_fit
from f1strat.simulator import (
    CarState,
    Plan,
    RaceState,
    Stop,
    car_model,
    enumerate_plans,
    is_legal,
    make_draws,
    observed_pit_losses,
    plan_totals,
    search,
    simulate_laps,
)

SPAIN = 6


@pytest.fixture(scope="module")
def fitted():
    return load_fit(THIN_FIT_PATH)


@pytest.fixture(scope="module")
def raw():
    return load_season(2022)


@pytest.fixture(scope="module")
def spain_state(raw) -> RaceState:
    return RaceState.from_laps(raw, SPAIN, lap=20, total_laps=66)


@pytest.fixture(scope="module")
def setup(fitted, raw, spain_state):
    rng = np.random.default_rng(0)
    car = spain_state.cars["VER"]
    idx = rng.integers(0, 1000, size=200)
    model = car_model(fitted, SPAIN, "VER", car.team, idx, rng, current_stint=car.stint)
    losses = observed_pit_losses(clean_laps(raw), SPAIN)
    draws = make_draws(model, spain_state.laps_remaining, losses, max_stops=3, rng=rng)
    return model, draws


def _toy_state(compound: str = "SOFT", used: frozenset = frozenset({"SOFT"})) -> RaceState:
    car = CarState("AAA", "T", 1, 0.0, compound, 5, 4, 1, used)
    return RaceState(1, lap=5, total_laps=20, cars={"AAA": car})


def test_two_compound_rule() -> None:
    state = _toy_state()
    car = state.cars["AAA"]
    assert not is_legal(Plan(), car, state)  # soft only
    assert not is_legal(Plan((Stop(10, "SOFT"),)), car, state)  # soft only
    assert is_legal(Plan((Stop(10, "MEDIUM"),)), car, state)
    # Already used two compounds: no further stop needed.
    two = _toy_state(used=frozenset({"SOFT", "MEDIUM"}))
    assert is_legal(Plan(), two.cars["AAA"], two)
    plans = list(enumerate_plans(state, "AAA", max_stops=1, used_life=None))
    assert all(len({s.compound for s in p.stops} | {"SOFT"}) >= 2 for p in plans)
    assert len(plans) == 14 * 2  # laps 6..19, M or H


def test_illegal_stop_laps() -> None:
    state = _toy_state()
    car = state.cars["AAA"]
    assert not is_legal(Plan((Stop(5, "MEDIUM"),)), car, state)  # not after state lap
    assert not is_legal(Plan((Stop(20, "MEDIUM"),)), car, state)  # last lap
    assert not is_legal(Plan((Stop(12, "MEDIUM"), Stop(10, "HARD"))), car, state)


def test_state_from_actual_laps(spain_state) -> None:
    ver = spain_state.cars["VER"]
    assert spain_state.lap == 20 and len(spain_state.cars) >= 18
    assert ver.stint == 2 and ver.compound == "MEDIUM"
    assert ver.compounds_used == frozenset({"SOFT", "MEDIUM"})
    assert ver.laps_since_stop == 20 - 14


def test_state_under_safety_car(raw) -> None:
    """A state can start on an SC lap (Britain 2022, SC from lap 39)."""
    state = RaceState.from_laps(raw, 10, lap=40, total_laps=52)
    assert state.track_status == "sc"


def test_closed_form_matches_lap_by_lap(spain_state, setup) -> None:
    model, draws = setup
    plans = [
        Plan(),
        Plan((Stop(30, "SOFT", 4),)),
        Plan((Stop(28, "HARD"), Stop(50, "SOFT", 4))),
        Plan((Stop(25, "MEDIUM"), Stop(40, "MEDIUM"), Stop(55, "SOFT", 4))),
    ]
    totals = plan_totals(model, spain_state, "VER", plans, draws)
    for p, plan in enumerate(plans):
        laps = simulate_laps(model, spain_state, "VER", plan, draws)
        expected = spain_state.cars["VER"].race_time + laps.sum(axis=1)
        np.testing.assert_allclose(totals[p], expected, rtol=0, atol=1e-6)


def test_search_mean_is_exact(spain_state, setup) -> None:
    """The search's expected-time ranking uses the same means as the Monte Carlo."""
    model, draws = setup
    result = search(model, spain_state, "VER", draws, max_stops=2, min_stint=8, per_stratum=2)
    means = result.totals.mean(axis=1)
    assert np.all(np.diff(means) >= -1e-6)  # ranked by mean
    assert result.n_legal > 10
    # Stratified: every stop count / compound sequence that is legal appears.
    strata = {(len(p.stops), tuple(s.compound for s in p.stops)) for p in result.plans}
    assert {k for k, _ in strata} == {0, 1, 2}


def test_inventory_limits_sets(spain_state, setup) -> None:
    """With a known inventory, each set is fitted at most once."""
    model, draws = setup
    state = spain_state.with_sets({"VER": [("HARD", 1), ("SOFT", 4)]})
    result = search(model, state, "VER", draws, max_stops=3, min_stint=5, per_stratum=2)
    for plan in result.plans:
        fitted = [(s.compound, s.tyre_life) for s in plan.stops]
        assert sorted(fitted) == sorted(set(fitted))
        assert set(fitted) <= {("HARD", 1), ("SOFT", 4)}
    assert max(len(p.stops) for p in result.plans) <= 2


def test_common_random_numbers(spain_state, setup) -> None:
    """Two plans that share all stints and stops get identical totals; the same
    plan evaluated twice gives identical draws."""
    model, draws = setup
    plan = Plan((Stop(30, "SOFT", 4),))
    a = plan_totals(model, spain_state, "VER", [plan, plan], draws)
    np.testing.assert_array_equal(a[0], a[1])
    # Plans differing only in pit lap differ only through tyre terms, not luck.
    b = plan_totals(
        model,
        spain_state,
        "VER",
        [Plan((Stop(30, "SOFT", 4),)), Plan((Stop(31, "SOFT", 4),))],
        draws,
    )
    assert np.std(b[0] - b[1]) < 0.5 * np.std(b[0])


def test_stint_effect_is_constant_within_a_stint(spain_state, setup) -> None:
    """Removing the deterministic tyre terms leaves noise + one stint draw per stint."""
    model, draws = setup
    plan = Plan((Stop(30, "HARD"),))
    laps = simulate_laps(model, spain_state, "VER", plan, draws)
    no_noise = laps - draws.noise
    # Final stint from lap 34 (out lap 31, post-stop laps 32-33): consecutive
    # laps differ only by deg and trend, the same for every lap pair, so the
    # second differences vanish.
    final = no_noise[:, 34 - 21 :]
    np.testing.assert_allclose(np.diff(final, n=2, axis=1), 0, atol=1e-8)


def test_widening_scales_final_stint_spread(fitted, spain_state) -> None:
    rng = np.random.default_rng(1)
    idx = rng.integers(0, 1000, size=4000)
    on = car_model(fitted, SPAIN, "VER", "Red Bull Racing", idx, np.random.default_rng(2))
    off = car_model(
        fitted, SPAIN, "VER", "Red Bull Racing", idx, np.random.default_rng(2), widening=1.0
    )
    np.testing.assert_allclose(on.sigma_stint_final, 1.188 * off.sigma_stint_final)
    np.testing.assert_allclose(on.final_mean, off.final_mean)  # Spain has final data


# --- Part B: the whole field ---------------------------------------------------

import dataclasses  # noqa: E402

from f1strat.race_events import SimParams  # noqa: E402
from f1strat.simulator import (  # noqa: E402
    SC,
    build_field,
    field_search,
    neutral_schedule,
    rival_plan_options,
    simulate_field,
)


@pytest.fixture(scope="module")
def params() -> SimParams:
    return SimParams.load()


@pytest.fixture(scope="module")
def losses(raw):
    return observed_pit_losses(clean_laps(raw), SPAIN)


def _no_neutral(params: SimParams) -> SimParams:
    return dataclasses.replace(
        params, neutral=dataclasses.replace(params.neutral, p_sc_per_lap=0.0, p_vsc_per_lap=0.0)
    )


def _no_passing(params: SimParams) -> SimParams:
    return dataclasses.replace(
        params, passing=dataclasses.replace(params.passing, intercept=-50.0, circuit={}, slope=0.0)
    )


def test_no_passing_keeps_order_and_gap(fitted, spain_state, params, losses) -> None:
    """With passing impossible and no stops or SC, nobody passes through anybody:
    the order at the flag is the order at the start, and every car is at least
    the smallest held gap behind the car ahead."""
    prm = _no_passing(_no_neutral(params))
    field = build_field(fitted, spain_state, losses, prm, n_sims=50, seed=3)
    res = simulate_field(field, {d: Plan() for d in field.drivers})
    start = np.argsort([spain_state.cars[d].race_time for d in field.drivers])
    for i in range(field.n_sims):
        assert list(np.argsort(res.times[i])) == list(start)
        gaps = np.diff(np.sort(res.times[i]))
        assert (gaps >= min(prm.passing.held_gaps or [prm.passing.follow_gap_s]) - 1e-9).all()
    assert res.passes == 0


def test_passing_happens_when_allowed(fitted, spain_state, params, losses) -> None:
    field = build_field(fitted, spain_state, losses, _no_neutral(params), n_sims=50, seed=3)
    res = simulate_field(field, {d: Plan() for d in field.drivers})
    assert res.passes > 0


def test_safety_car_bunches_the_field(fitted, raw, params, losses) -> None:
    """Five SC laps from lap 20: the field closes up to the queue gap."""
    state = RaceState.from_laps(raw, SPAIN, lap=20, total_laps=66)
    prm = dataclasses.replace(
        params,
        neutral=dataclasses.replace(
            params.neutral, sc_lengths=[5], p_sc_per_lap=0.0, p_vsc_per_lap=0.0
        ),
    )
    state_sc = dataclasses.replace(state, track_status="sc", neutral_laps=0)
    field = build_field(fitted, state_sc, losses, prm, n_sims=20, seed=4)
    assert (field.status[:, :5] == SC).all() and (field.status[:, 5:] != SC).all()
    # Stop the race right after the SC so the bunching is visible.
    short = dataclasses.replace(state_sc, total_laps=state.lap + 5)
    field_short = build_field(fitted, short, losses, prm, n_sims=20, seed=4)
    res = simulate_field(field_short, {d: Plan() for d in field_short.drivers})
    gaps = np.diff(np.sort(res.times, axis=1), axis=1)
    before = np.diff(np.sort([c.race_time for c in state.cars.values()]))
    assert np.median(gaps) <= prm.sc_gap_s + 1e-6
    assert gaps.max() < before.max()


def test_start_mid_race_under_sc(fitted, raw, params) -> None:
    """Britain 2022 from lap 40 (under SC): the SC continues, then racing resumes."""
    state = RaceState.from_laps(raw, 10, lap=40, total_laps=52)
    assert state.track_status == "sc" and state.neutral_laps >= 1
    losses_gb = observed_pit_losses(clean_laps(raw), 10)
    field = build_field(fitted, state, losses_gb, params, n_sims=30, seed=5)
    assert (field.status[:, 0] == SC).all()
    res = simulate_field(field, {d: Plan() for d in field.drivers})
    assert np.isfinite(res.times).all()
    assert sorted(res.positions[0]) == list(range(1, len(field.drivers) + 1))


def test_neutral_schedule_continues_current_sc(raw, params) -> None:
    state = RaceState.from_laps(raw, 10, lap=40, total_laps=52)
    status = neutral_schedule(state, params.neutral, 200, np.random.default_rng(0))
    longer = [n for n in params.neutral.sc_lengths if n > state.neutral_laps]
    remaining = (status == SC).argmin(axis=1)
    assert remaining.min() >= 1
    assert remaining.max() <= max(longer) - state.neutral_laps


def test_field_common_random_numbers(fitted, spain_state, params, losses) -> None:
    """The same focal plan twice gives identical results; rival luck is shared."""
    field = build_field(fitted, spain_state, losses, params, n_sims=40, seed=6)
    rivals = {d: Plan() for d in field.drivers if d != "VER"}
    plans = [Plan(), Plan((Stop(30, "SOFT", 4),))]
    res = field_search(field, "VER", [plans[0], plans[0], plans[1]], rivals)
    np.testing.assert_array_equal(res.times[0], res.times[1])
    assert not np.array_equal(res.times[0], res.times[2])
    # The plan-independent draws are fixed per field.
    np.testing.assert_array_equal(field.status, field.status.copy())


def test_rival_options_sampled_by_weight(fitted, spain_state, params, losses) -> None:
    field = build_field(fitted, spain_state, losses, params, n_sims=2000, seed=7)
    opts = [(Plan(), 3.0), (Plan((Stop(40, "HARD"),)), 1.0)]
    from f1strat.simulator import _select

    c = field.drivers.index("PER")
    _, pit = _select(field, c, opts)
    share_stop = (pit.sum(axis=1) > 0).mean()
    assert share_stop == pytest.approx(0.25, abs=0.04)
    options = rival_plan_options(field, "PER", top=3)
    assert 1 <= len(options) <= 3 and options[0][1] == pytest.approx(1.0)


def test_safety_car_closes_gradually(fitted, raw, params, losses) -> None:
    """Phase 5: gaps are kept on the first SC lap, then shrink by
    sc_close_ratio per lap, never below sc_gap_s."""
    state = RaceState.from_laps(raw, SPAIN, lap=20, total_laps=66)
    prm = dataclasses.replace(
        params,
        neutral=dataclasses.replace(
            params.neutral, sc_lengths=[5], p_sc_per_lap=0.0, p_vsc_per_lap=0.0, sc_close_ratio=0.5
        ),
    )
    state_sc = dataclasses.replace(state, track_status="sc", neutral_laps=0, total_laps=22)
    field = build_field(fitted, state_sc, losses, prm, n_sims=10, seed=4)
    res = simulate_field(field, {d: Plan() for d in field.drivers}, record=True)
    start = np.sort([state.cars[d].race_time for d in field.drivers])
    gaps0 = np.diff(start)
    lap1 = np.diff(np.sort(res.history[:, 0, :], axis=1), axis=1)
    lap2 = np.diff(np.sort(res.history[:, 1, :], axis=1), axis=1)
    expect1 = np.maximum(prm.sc_gap_s, gaps0)
    assert np.allclose(lap1, expect1[None, :])
    assert np.allclose(lap2, np.maximum(prm.sc_gap_s, 0.5 * expect1)[None, :])


def test_tyre_age_cap_and_family_filters(spain_state, setup) -> None:
    """Phase 6: a capped search never lets a stint pass its compound's cap and
    reports what it removed; family filters fix the first stop."""
    model, draws = setup
    cap = {"SOFT": 15, "MEDIUM": 25, "HARD": 35}
    res = search(model, spain_state, "VER", draws, max_stops=2, min_stint=5, max_age=cap)
    car = spain_state.cars["VER"]
    assert res.n_capped > 0 and res.capped
    for plan in res.plans:
        prev_lap, comp, life0 = spain_state.lap, car.compound, car.tyre_life + 1
        for stop in [*plan.stops, None]:
            end = stop.lap if stop else spain_state.total_laps
            assert life0 + (end - prev_lap) - 1 <= cap[comp]
            if stop:
                prev_lap, comp, life0 = stop.lap, stop.compound, stop.tyre_life
    fam = search(
        model,
        spain_state,
        "VER",
        draws,
        max_stops=2,
        min_stint=5,
        first_laps=(20, 22),
        first_sets=[("HARD", 1)],
    )
    assert fam.plans and all(
        p.stops
        and 20 <= p.stops[0].lap <= 22
        and p.stops[0].compound == "HARD"
        and p.stops[0].tyre_life == 1
        for p in fam.plans
    )


def test_known_stint_effect_replaces_the_draw(spain_state, setup) -> None:
    """Phase 6 hindsight: a plan stint matching a known stint (first lap,
    compound, TyreLife) uses the known effect for all its laps."""
    model, draws = setup
    plan = Plan((Stop(30, "HARD", 1),))
    base = simulate_laps(model, spain_state, "VER", plan, draws)
    known = dataclasses.replace(model, known_stints={(31, "HARD", 1): np.full(draws.n_sims, 2.0)})
    shifted = simulate_laps(known, spain_state, "VER", plan, draws)
    final = model.final_mean + model.sigma_stint_final * draws.z_final
    cols = np.arange(31, spain_state.total_laps + 1) - (spain_state.lap + 1)
    assert np.allclose((shifted - base)[:, cols], (2.0 - final)[:, None])
    assert np.allclose((shifted - base)[:, : cols[0]], 0.0)
    # Totals agree with the lap-by-lap path.
    tot = plan_totals(known, spain_state, "VER", [plan], draws)[0]
    assert np.allclose(tot, spain_state.cars["VER"].race_time + shifted.sum(axis=1))


# --- Phase 6 (Silverstone) additions ---------------------------------------------


def test_double_stack_second_car_waits(fitted, spain_state, params, losses) -> None:
    """Teammates pitting on the same lap: the car that arrives first is served
    first and unchanged; the other waits max(0, D - arrival gap)."""
    prm = _no_neutral(params)
    field = build_field(fitted, spain_state, losses, prm, n_sims=40, seed=8)
    rivals = {d: Plan() for d in field.drivers}
    stop = Plan((Stop(25, "HARD", 1),))
    plans = {**rivals, "VER": stop, "PER": stop}
    col = 25 - (spain_state.lap + 1)
    base = simulate_field(field, plans, record=True)
    stacked_field = dataclasses.replace(
        field, params=dataclasses.replace(prm, double_stack={"green": 30.0, "sc": 30.0})
    )
    stacked = simulate_field(stacked_field, plans, record=True)
    v, p = field.drivers.index("VER"), field.drivers.index("PER")
    a_v = base.history[:, col, v] - field.draws["VER"].pit_loss[:, 0]
    a_p = base.history[:, col, p] - field.draws["PER"].pit_loss[:, 0]
    wait = np.maximum(0.0, 30.0 - np.abs(a_v - a_p))
    first_v = a_v <= a_p
    got_v = stacked.history[:, col, v] - base.history[:, col, v]
    got_p = stacked.history[:, col, p] - base.history[:, col, p]
    np.testing.assert_allclose(np.where(first_v, got_v, got_p), 0.0, atol=1e-9)
    np.testing.assert_allclose(np.where(first_v, got_p, got_v), wait, atol=1e-9)
    assert stacked.double_stacks == int((wait > 0).sum()) > 0
    # Only one Red Bull stopping: nothing changes.
    solo = {**rivals, "VER": stop}
    np.testing.assert_allclose(
        simulate_field(stacked_field, solo).times, simulate_field(field, solo).times
    )


def test_double_stack_loss_recovers_d() -> None:
    from f1strat.race_events import double_stack_loss

    gaps = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0, 12.0, 3.5, 2.5])
    pairs = pd.DataFrame({"kind": "green", "gap_s": gaps, "extra_s": np.maximum(0, 6.0 - gaps)})
    out = double_stack_loss(pairs, kinds=("green",))
    assert out["green"]["D_s"] == pytest.approx(6.0, abs=0.051)
    assert out["green"]["n_queued"] == int((gaps < 6.0).sum())


def test_search_pit_factor_scales_stop_cost(spain_state, setup) -> None:
    """SC-aware search: a stop on a lap with factor f costs f x its green loss,
    and the search's scoring stays exact."""
    model, draws = setup
    factor = np.ones(spain_state.laps_remaining)
    factor[30 - (spain_state.lap + 1)] = 0.5
    plan = Plan((Stop(30, "HARD", 1),))
    full = plan_totals(model, spain_state, "VER", [plan], draws)[0]
    cheap = plan_totals(model, spain_state, "VER", [plan], draws, pit_factor=factor)[0]
    np.testing.assert_allclose(full - cheap, 0.5 * draws.pit_loss[:, 0])
    res = search(
        model,
        spain_state,
        "VER",
        draws,
        max_stops=1,
        min_stint=5,
        per_stratum=1,
        pit_factor=factor,
        first_laps=(26, 34),
        first_sets=[("HARD", 1)],
    )
    assert res.plans[0].stops[0].lap == 30  # the discounted lap wins
    assert np.allclose(
        res.totals, plan_totals(model, spain_state, "VER", res.plans, draws, pit_factor=factor)
    )


def test_session_timing_uses_the_clock(raw) -> None:
    """Silverstone 2022 (lap 1 red-flagged, restart lap untimed): lap-time sums
    give wrong gaps; the session clock gives the real gaps at the line."""
    laps = RaceState.from_laps(raw, 10, lap=38, total_laps=52)
    clock = RaceState.from_laps(raw, 10, lap=38, total_laps=52, timing="session")
    gap = clock.cars["HAM"].race_time - clock.cars["LEC"].race_time
    assert gap == pytest.approx(6.125, abs=1e-3)
    assert laps.cars["HAM"].race_time < laps.cars["LEC"].race_time  # the bug it fixes


def test_current_neutral_length_is_fixed(fitted, raw, params) -> None:
    state = dataclasses.replace(
        RaceState.from_laps(raw, 10, lap=38, total_laps=52, timing="session"),
        track_status="sc",
        neutral_laps=0,
    )
    status = neutral_schedule(
        state, params.neutral, 100, np.random.default_rng(0), current_length=6
    )
    assert (status[:, :6] == SC).all() and (status[:, 6] != SC).all()


def test_restart_layer_applies_only_after_neutral_laps(fitted, raw, params, losses) -> None:
    """Restart layer: offsets on the first laps after the SC, by compound; none
    elsewhere; laps_since_restart_sim counts from the first green lap."""
    from f1strat.restart import RestartModel, laps_since_restart_sim

    status = np.array([[1, 1, 0, 0, 0, 0, 2, 0, 0]])
    np.testing.assert_array_equal(
        laps_since_restart_sim(status, False), [[0, 0, 1, 2, 3, 4, 0, 1, 2]]
    )
    np.testing.assert_array_equal(laps_since_restart_sim(np.array([[0, 0]]), True), [[1, 2]])
    rs = RestartModel(
        pace=[[1.0, 0.0, 0.5, 0.0], [0.3, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        pace_se=[[0] * 4] * 3,
        pass_mult=1.0,
        n_laps=[1] * 3,
        n_events=0,
        n_passes=0,
        expected_passes=0.0,
        rounds=[],
    )
    prm = dataclasses.replace(
        params,
        neutral=dataclasses.replace(
            params.neutral, sc_lengths=[2], p_sc_per_lap=0.0, p_vsc_per_lap=0.0
        ),
    )
    state = dataclasses.replace(
        RaceState.from_laps(raw, SPAIN, lap=20, total_laps=30), track_status="sc", neutral_laps=0
    )
    fld = build_field(fitted, state, losses, prm, n_sims=10, seed=9)
    from f1strat.simulator import _car_laps

    base, _ = _car_laps(fld, "VER", Plan())
    fld_rs = dataclasses.replace(fld, params=dataclasses.replace(prm, restart=rs))
    with_rs, _ = _car_laps(fld_rs, "VER", Plan())
    hard = state.cars["VER"].compound == "HARD"
    expect = np.zeros(state.laps_remaining)
    expect[2] = 1.0 + (0.5 if hard else 0.0)  # SC laps 21-22, restart lap 23
    expect[3] = 0.3
    np.testing.assert_allclose(with_rs - base, np.broadcast_to(expect, base.shape))
