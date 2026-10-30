"""Smoke test: FastF1 can load 2022 race data with the tyre fields we rely on."""

from pathlib import Path

import fastf1
import pytest

CACHE_DIR = Path(__file__).resolve().parents[1] / "cache"


@pytest.fixture(scope="module")
def hungary_2022_race() -> fastf1.core.Session:
    """Load the 2022 Hungarian GP race (laps only) through the local FastF1 cache."""
    CACHE_DIR.mkdir(exist_ok=True)
    fastf1.Cache.configure(cache_dir=str(CACHE_DIR))
    session = fastf1.get_session(2022, "Hungary", "R")
    session.load(laps=True, telemetry=False, weather=False, messages=False)
    return session


def test_leclerc_laps_have_tyre_data(hungary_2022_race: fastf1.core.Session) -> None:
    """Leclerc's laps exist and have Compound and TyreLife populated."""
    laps = hungary_2022_race.laps.pick_drivers("LEC")
    assert len(laps) > 0
    assert laps["Compound"].notna().all()
    assert laps["TyreLife"].notna().all()
