"""Redraw every figure from committed files (``pitwall figures``).

No refits and no new simulations: each figure is replotted from the result
files its pipeline step saved in ``reports/figures/`` (JSON and CSV), the
committed processed data and the committed thinned posteriors.

- **Case studies:** ``case_<name>_results.json`` → points, sensitivity,
  hindsight, reality-check and joint-option figures (Hungary, Silverstone v1
  and v2); Monaco from its results JSON and the processed laps.
- **Backtests:** ``backtest_*.csv`` → the five calibration plots.
- **Notebooks:** executed top to bottom in place (EDA, degradation model,
  simulator demo), which rewrites their figures and tables.
- **README assets:** :mod:`f1strat.readme_assets` (banner, verdict cards,
  animation, restyled figures, light and dark).
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from f1strat.case_study import FIG_DIR

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = ("02_eda.ipynb", "03_deg_model.ipynb", "04_simulator.ipynb")


def case_study_figures(fig_dir: Path = FIG_DIR) -> list[Path]:
    """Hungary (laps 38 and 53), Silverstone (v1, v2) and Monaco figures."""
    from f1strat import case_study as cs
    from f1strat import monaco
    from f1strat.data import load_season

    out = []
    for stem, race_name in (
        ("hungary", "Hungary"),
        ("hungary54", "Hungary"),
        ("silverstone", "Silverstone"),
        ("silverstone_v2", "Silverstone"),
    ):
        r = json.loads((fig_dir / f"case_{stem}_results.json").read_text())
        main_v = r["checks"][0]
        p = fig_dir / f"case_{stem}"
        out += [
            cs.plot_points(main_v, Path(f"{p}_points.png"), race_name),
            cs.plot_sensitivity(r["checks"], Path(f"{p}_sensitivity.png")),
            cs.plot_hindsight(main_v, r["hindsight"], Path(f"{p}_hindsight.png"), race_name),
        ]
        if stem == "hungary":
            out.append(cs.plot_reality(r["reality_check"], Path(f"{p}_reality.png")))
        if stem.startswith("silverstone"):
            out.append(cs.plot_joint(r["joint"], Path(f"{p}_joint.png"), r["joint_hindsight"]))
            out.append(cs.plot_reality_silverstone(r["reality_check"], Path(f"{p}_reality.png")))
    r = json.loads((fig_dir / "case_monaco_results.json").read_text())
    raw, race = load_season(2022), load_season(2022, table="race")
    monaco.plot_gap_timeline(raw, race, fig_dir / "case_monaco_gaps.png")
    monaco.plot_positions(r, fig_dir / "case_monaco_positions.png")
    out += [fig_dir / "case_monaco_gaps.png", fig_dir / "case_monaco_positions.png"]
    return [Path(x) for x in out]


def backtest_figures(fig_dir: Path = FIG_DIR) -> list[Path]:
    """The Phase 5 calibration plots from the saved backtest tables."""
    from f1strat.backtest import plot_all, saved_tables

    return plot_all(saved_tables(fig_dir), fig_dir)


def run_notebooks(names: tuple[str, ...] = NOTEBOOKS) -> list[Path]:
    """Execute each notebook top to bottom in place (raises on any error)."""
    paths = []
    for name in names:
        path = ROOT / "notebooks" / name
        subprocess.run(
            [
                sys.executable,
                "-m",
                "jupyter",
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                "--inplace",
                "--ExecutePreprocessor.timeout=1800",
                str(path),
            ],
            check=True,
            cwd=path.parent,
        )
        paths.append(path)
    return paths


def main(notebooks: bool = True, readme_only: bool = False) -> None:
    """Redraw everything (``readme_only``: just the README assets)."""
    import logging

    from f1strat import readme_assets

    logging.disable(logging.WARNING)
    steps = [("README assets", readme_assets.build_all)]
    if not readme_only:
        steps = [
            ("case-study figures", case_study_figures),
            ("backtest figures", backtest_figures),
        ] + steps
        if notebooks:
            steps.append(("notebooks", run_notebooks))
    for name, step in steps:
        t0 = time.time()
        paths = step()
        print(f"{name}: {len(paths)} files in {time.time() - t0:.0f} s", flush=True)
        for p in paths:
            print("  ", Path(p).relative_to(ROOT))


__all__ = ["backtest_figures", "case_study_figures", "main", "run_notebooks"]
