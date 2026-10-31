"""Every figure draws from data/figures alone, and the tables carry exactly
the values in the result files they were extracted from."""

import json
import re
from pathlib import Path

import pytest

from f1strat import figure_data as fd
from f1strat.case_study import _merged
from f1strat.figures import backtest_figures, case_study_figures, notebook_figures

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reports" / "figures"


@pytest.mark.parametrize("stem", list(fd.CASE_STEMS))
def test_tables_round_trip_the_results_exactly(stem: str) -> None:
    r = json.loads((RESULTS / f"case_{stem}_results.json").read_text())
    main = r["checks"][0]
    rebuilt, _ = fd.points_main(fd.read(f"case_{stem}_points"))
    assert _merged(rebuilt) == _merged(main)
    for joined, _plan in _merged(main):
        n = joined.split(" = ")[0]
        assert rebuilt["measure"][n]["mean_points"] == main["measure"][n]["mean_points"]
        assert rebuilt["measure"][n]["points_dist"] == {
            k: v for k, v in main["measure"][n]["points_dist"].items()
        }
    checks = fd.sensitivity_checks(fd.read(f"case_{stem}_sensitivity"))
    assert [c["primary"]["mean_loss"] for c in checks] == [
        c["primary"]["mean_loss"] for c in r["checks"]
    ]


def test_figures_draw_from_data_figures(tmp_path: Path) -> None:
    assert len(case_study_figures(tmp_path)) == 19
    assert len(backtest_figures(tmp_path)) == 5
    assert len(notebook_figures(tmp_path)) == 7


def test_every_figure_in_the_docs_has_its_table() -> None:
    text = "".join(
        (ROOT / "docs" / n).read_text()
        for n in ("case_studies.md", "validation.md", "model.md", "simulator.md")
    )
    shown = set(re.findall(r"reports/figures/(\w+)\.png", text))
    tables = {p.name.split(".")[0] for p in fd.DATA_DIR.iterdir()}
    backtest = {
        "backtest_pit",
        "backtest_coverage",
        "backtest_passes",
        "backtest_close",
        "backtest_sc_vsc",
    }
    missing = [s for s in shown - backtest if s not in tables]
    assert not missing, missing
