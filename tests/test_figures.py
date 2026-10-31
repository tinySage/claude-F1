"""Figures redraw from data/figures alone, matching the committed images."""

import re
from pathlib import Path

from f1strat import figure_data as fd
from f1strat.figures import case_study_figures

ROOT = Path(__file__).resolve().parents[1]


def test_case_study_figures_match_committed_bytes(tmp_path: Path) -> None:
    paths = case_study_figures(tmp_path)
    assert len(paths) == 19
    for p in paths:
        assert p.read_bytes() == (ROOT / "reports" / "figures" / p.name).read_bytes(), p.name


def test_every_figure_in_the_docs_has_its_table() -> None:
    text = "".join(
        (ROOT / "docs" / n).read_text()
        for n in ("case_studies.md", "validation.md", "model.md", "simulator.md")
    )
    shown = set(re.findall(r"reports/figures/([\w]+)\.png", text))
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
