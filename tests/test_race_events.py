"""Tests for race-event parameters (Phase 6 relative pass difficulty)."""

import pytest

from f1strat.race_events import relative_circuit_intercept


def test_relative_intercept_shifts_the_current_average() -> None:
    """intercept = current pooled + (circuit 2021 - 2021 average); the shift
    does not depend on the current pooled value. Monaco 2021 (no passes) is
    harder than average."""
    a = relative_circuit_intercept("Hungarian Grand Prix", -2.8, 2021)
    b = relative_circuit_intercept("Hungarian Grand Prix", -2.0, 2021)
    assert a["intercept"] == pytest.approx(-2.8 + a["shift"])
    assert a["shift"] == pytest.approx(b["shift"])
    assert a["shift"] == pytest.approx(a["ref_circuit"] - a["ref_average"])
    assert relative_circuit_intercept("Monaco Grand Prix", -2.8, 2021)["shift"] < 0
    wet = {r["EventName"] for r in a["ref_races_not_used"]}
    assert "Turkish Grand Prix" in wet
    with pytest.raises(ValueError):
        relative_circuit_intercept("Turkish Grand Prix", -2.8, 2021)
