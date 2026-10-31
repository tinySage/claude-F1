"""Redraw every figure from ``data/figures/`` (``pitwall figures``).

Every figure in the docs, the notebooks and the README is drawn here from its
table in ``data/figures/`` (:mod:`f1strat.figure_data`): no fits, no
simulations, no downloads, a few minutes in all.

- **Case studies:** points, sensitivity, hindsight, reality-check and
  joint-option figures (Hungary, Silverstone v1 and v2), and Monaco's two.
- **Backtests:** the five calibration plots.
- **Notebook figures:** the EDA scatters, exclude reasons and pit losses
  (notebook 02), and the degradation curves, the long-tyre stress test and
  Leclerc's Hungary hard stint (notebook 03). The notebooks call the same
  functions.
- **README assets:** :mod:`f1strat.readme_assets`.

``pitwall figures --export`` first rebuilds the tables from the heavy sources
(result files, processed laps, committed thinned fits); ``--notebooks`` also
executes the notebooks top to bottom.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from f1strat import figure_data as fd
from f1strat.case_study import FIG_DIR

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = ("02_eda.ipynb", "03_deg_model.ipynb", "04_simulator.ipynb")

# --- Notebook figure style (light, as in the docs) ------------------------------------

COMPOUND_COLORS = {"SOFT": "#d9342b", "MEDIUM": "#eda100", "HARD": "#2a78d6"}
MODEL_COLORS = {"Baseline (OLS)": "#52514e", "XGBoost": "#eb6834", "Bayesian": "#2a78d6"}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
NOTEBOOK_RC = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
}


def _plt():
    import matplotlib

    if "ipykernel" not in sys.modules:  # keep a notebook's inline backend
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _finish(fig, path: Path | None, show: bool) -> Path | None:
    plt = _plt()
    if path is not None:
        fig.savefig(path)
    if show:  # in a notebook: display inline, whatever the backend
        from IPython.display import display

        display(fig)
    plt.close(fig)
    return path


def _deg_scatter(ax, df: pd.DataFrame, title: str) -> None:
    for compound, color in COMPOUND_COLORS.items():
        d = df[df["Compound"] == compound]
        if d.empty:
            continue
        ax.scatter(
            d["TyreLife"],
            d["lap_s"],
            s=9,
            color=color,
            alpha=0.55,
            linewidths=0,
            label=compound.title(),
        )
    ax.set_title(title, loc="left")
    ax.set_xlabel("Tyre age (laps)")


def plot_deg_three_races(
    clean: pd.DataFrame, path: Path | None = None, show: bool = False
) -> Path | None:
    """Clean lap time against tyre age at Bahrain, Spain and Hungary."""
    plt = _plt()
    with plt.rc_context(NOTEBOOK_RC):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=False)
        for ax, (rnd, name) in zip(axes, [(1, "Bahrain"), (6, "Spain"), (13, "Hungary")]):
            _deg_scatter(ax, clean[clean["RoundNumber"] == rnd], name)
        axes[0].set_ylabel("Lap time (s)")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", ncol=3, frameon=False, markerscale=2)
        fig.suptitle(
            "Clean lap time vs tyre age (fuel effect not removed)",
            x=0.01,
            ha="left",
            fontweight="bold",
            color=INK,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        return _finish(fig, path, show)


def plot_deg_all_races(
    clean: pd.DataFrame, rounds: pd.Series, path: Path | None = None, show: bool = False
) -> Path | None:
    """The same view for every 2022 race, one panel each (``rounds``: round
    number → event name, including races with no clean laps)."""
    plt = _plt()
    with plt.rc_context(NOTEBOOK_RC):
        fig, axes = plt.subplots(5, 5, figsize=(16, 15))
        for ax, (rnd, name) in zip(axes.flat, rounds.items()):
            d = clean[clean["RoundNumber"] == rnd]
            _deg_scatter(ax, d, f"{rnd}. {name.replace(' Grand Prix', '')} (n={len(d)})")
            ax.set_xlabel("")
        for ax in axes.flat[len(rounds) :]:
            ax.set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower right",
            ncol=3,
            frameon=False,
            markerscale=2,
            bbox_to_anchor=(0.95, 0.08),
        )
        fig.supxlabel("Tyre age (laps)", color=INK_2)
        fig.supylabel("Lap time (s)", color=INK_2)
        fig.tight_layout()
        return _finish(fig, path, show)


def plot_exclude_reasons(
    long: pd.DataFrame, path: Path | None = None, show: bool = False
) -> Path | None:
    """Heatmap of laps per exclude reason per race."""
    plt = _plt()
    counts = long.pivot(index="EventName", columns="reason", values="laps")
    events = long.sort_values("event_order")["EventName"].unique()
    order = long.sort_values("reason_order")["reason"].unique()
    counts = counts.loc[events, order].astype(int)
    with plt.rc_context(NOTEBOOK_RC):
        fig, ax = plt.subplots(figsize=(13, 8.5))
        im = ax.imshow(
            counts.to_numpy(),
            cmap="Blues",
            aspect="auto",
            norm=plt.matplotlib.colors.PowerNorm(0.5),
        )
        ax.set_xticks(
            range(len(order)), [o.replace("_", " ") for o in order], rotation=40, ha="right"
        )
        ax.set_yticks(range(len(counts)), [e.replace(" Grand Prix", "") for e in counts.index])
        ax.grid(False)
        vmax = counts.to_numpy().max()
        for i in range(counts.shape[0]):
            for j in range(counts.shape[1]):
                v = counts.iat[i, j]
                if v:
                    ax.text(
                        j,
                        i,
                        v,
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white" if v > 0.3 * vmax else INK,
                    )
        ax.set_title("Laps per exclude reason (first reason that applies)", loc="left")
        fig.colorbar(im, ax=ax, shrink=0.6, label="Laps")
        fig.tight_layout()
        return _finish(fig, path, show)


def plot_pit_loss(pit: pd.DataFrame, path: Path | None = None, show: bool = False) -> Path | None:
    """Median green-flag pit loss per circuit with its interquartile range."""
    plt = _plt()
    p = pit.dropna(subset=["pit_loss_s"]).sort_values("pit_loss_s")
    ref = (p["source_year"] != 2022).to_numpy()
    with plt.rc_context(NOTEBOOK_RC):
        fig, ax = plt.subplots(figsize=(9, 7))
        y = np.arange(len(p))
        ax.hlines(y, p["q25_s"], p["q75_s"], color="#2a78d6", alpha=0.35, linewidth=4)
        ax.scatter(
            p["pit_loss_s"][~ref],
            y[~ref],
            color="#2a78d6",
            s=40,
            zorder=3,
            edgecolors=SURFACE,
            linewidths=2,
            label="2022 stops",
        )
        if ref.any():
            ax.scatter(
                p["pit_loss_s"][ref],
                y[ref],
                facecolors=SURFACE,
                edgecolors="#2a78d6",
                s=40,
                zorder=3,
                linewidths=2,
                label="Reference season",
            )
            ax.legend(loc="lower right", frameon=False)
        for yi, (_, r) in zip(y, p.iterrows()):
            ax.text(
                r["q75_s"] + 0.6,
                yi,
                f"{r['pit_loss_s']:.1f} s  (n={r['n_stops']})",
                va="center",
                fontsize=8,
                color=INK_2,
            )
        labels = [
            c if yr == 2022 else f"{c} ({int(yr)})" for c, yr in zip(p["Circuit"], p["source_year"])
        ]
        ax.set_yticks(y, labels)
        ax.set_xlabel("Pit lane loss (s)")
        ax.set_xlim(0, p["q75_s"].max() + 9)
        ax.grid(axis="y", visible=False)
        ax.set_title("Median green-flag pit loss per circuit (bar = IQR)", loc="left")
        fig.tight_layout()
        return _finish(fig, path, show)


def plot_deg_curves(
    curves: pd.DataFrame, path: Path | None = None, show: bool = False
) -> Path | None:
    """Tyre effect per compound against a new medium, with 80% bands."""
    plt = _plt()
    with plt.rc_context(NOTEBOOK_RC):
        fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), sharey=True)
        for ax, (rnd, name) in zip(axes, fd.DEG_CURVE_RACES):
            for comp, color in COMPOUND_COLORS.items():
                c = curves[(curves["round"] == rnd) & (curves["compound"] == comp)]
                if c.empty:
                    continue
                ax.fill_between(
                    c["TyreLife"], c["lo"], c["hi"], color=color, alpha=0.2, linewidth=0
                )
                ax.plot(c["TyreLife"], c["mean"], color=color, linewidth=2, label=comp.title())
                ax.text(
                    c["TyreLife"].iloc[-1] + 0.5,
                    c["mean"].iloc[-1],
                    comp.title(),
                    color=INK_2,
                    fontsize=8,
                    va="center",
                )
            ax.axhline(0, color=INK_2, linewidth=0.8)
            ax.set_title(name, loc="left")
            ax.set_xlabel("Tyre age (laps)")
        axes[0].set_ylabel("Lap time vs new medium (s)")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", ncol=3, frameon=False)
        fig.suptitle(
            "Tyre degradation per compound, field-average car (80% bands)",
            x=0.01,
            ha="left",
            fontweight="bold",
            color=INK,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        return _finish(fig, path, show)


def plot_long_tyre(
    errors: pd.DataFrame, stint: pd.DataFrame, path: Path | None = None, show: bool = False
) -> Path | None:
    """Mean error by tyre age per model, and the longest test stint."""
    plt = _plt()
    with plt.rc_context(NOTEBOOK_RC):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.6))
        for model, color in MODEL_COLORS.items():
            e = errors[errors["model"] == model]
            ax1.plot(
                e["age_mid"], e["mean_error_s"], marker="o", color=color, linewidth=2, label=model
            )
        ax1.axhline(0, color=INK_2, linewidth=0.8)
        ax1.set_xlabel("Tyre age (laps)")
        ax1.set_ylabel("Mean error, predicted − actual (s)")
        ax1.set_title("Error grows with tyre age beyond the training range", loc="left")
        ax1.legend(frameon=False)
        x = stint["TyreLife"]
        ax2.fill_between(
            x,
            stint["bayes_lo"],
            stint["bayes_hi"],
            color=MODEL_COLORS["Bayesian"],
            alpha=0.18,
            linewidth=0,
            label="Bayesian 80% interval",
        )
        ax2.plot(
            x, stint["bayes_mean"], color=MODEL_COLORS["Bayesian"], linewidth=2, label="Bayesian"
        )
        ax2.plot(x, stint["xgboost"], color=MODEL_COLORS["XGBoost"], linewidth=2, label="XGBoost")
        ax2.scatter(x, stint["actual"], color=INK, s=14, zorder=3, label="Actual")
        ax2.set_title(stint["title"].iloc[0], loc="left")
        ax2.set_xlabel("Tyre age (laps)")
        ax2.set_ylabel("Lap time (s)")
        ax2.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        return _finish(fig, path, show)


def plot_leclerc_hungary_hard(
    lec: pd.DataFrame, path: Path | None = None, show: bool = False
) -> Path | None:
    """Leclerc's Hungary hard stint: actual laps against the model's interval."""
    plt = _plt()
    with plt.rc_context(NOTEBOOK_RC):
        fig, ax = plt.subplots(figsize=(9, 4.4))
        ax.fill_between(
            lec["race_lap"],
            lec["lo"],
            lec["hi"],
            color=COMPOUND_COLORS["HARD"],
            alpha=0.18,
            linewidth=0,
            label="80% prediction interval",
        )
        ax.plot(
            lec["race_lap"],
            lec["predicted"],
            color=COMPOUND_COLORS["HARD"],
            linewidth=2,
            label="Predicted",
        )
        ax.scatter(lec["race_lap"], lec["actual"], color=INK, s=22, zorder=3, label="Actual")
        rain = lec[lec["rainfall"].astype(bool)]
        ax.scatter(
            rain["race_lap"],
            rain["actual"],
            facecolors="none",
            edgecolors=INK,
            s=90,
            linewidths=1,
            label="Rainfall reported",
        )
        ax.set_xlabel("Race lap")
        ax.set_ylabel("Lap time (s)")
        ax.set_title("Leclerc on hards, Hungary 2022 (full-data fit)", loc="left")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        return _finish(fig, path, show)


# --- Drawing everything from data/figures ---------------------------------------------


def case_study_figures(fig_dir: Path = FIG_DIR) -> list[Path]:
    """Hungary (laps 38 and 53), Silverstone (v1, v2) and Monaco figures."""
    from f1strat import case_study as cs
    from f1strat import monaco

    out = []
    for stem in fd.CASE_STEMS:
        p = fig_dir / f"case_{stem}"
        main, race = fd.points_main(fd.read(f"case_{stem}_points"))
        out.append(cs.plot_points(main, Path(f"{p}_points.png"), race))
        out.append(
            cs.plot_sensitivity(
                fd.sensitivity_checks(fd.read(f"case_{stem}_sensitivity")),
                Path(f"{p}_sensitivity.png"),
            )
        )
        main, hind, race = fd.hindsight_dicts(fd.read(f"case_{stem}_hindsight"))
        out.append(cs.plot_hindsight(main, hind, Path(f"{p}_hindsight.png"), race))
        if stem == "hungary":
            out.append(
                cs.plot_reality(
                    fd.reality_dict(fd.read(f"case_{stem}_reality")), Path(f"{p}_reality.png")
                )
            )
        if stem.startswith("silverstone"):
            joint, hj = fd.joint_dicts(fd.read(f"case_{stem}_joint"))
            out.append(cs.plot_joint(joint, Path(f"{p}_joint.png"), hj))
            out.append(
                cs.plot_reality_silverstone(
                    fd.reality_dict(fd.read(f"case_{stem}_reality")), Path(f"{p}_reality.png")
                )
            )
    monaco.plot_gap_timeline(fd.read("case_monaco_gaps"), fig_dir / "case_monaco_gaps.png")
    monaco.plot_positions(
        fd.monaco_out(fd.read("case_monaco_positions")), fig_dir / "case_monaco_positions.png"
    )
    out += [fig_dir / "case_monaco_gaps.png", fig_dir / "case_monaco_positions.png"]
    return [Path(x) for x in out]


def backtest_figures(fig_dir: Path = FIG_DIR) -> list[Path]:
    """The Phase 5 calibration plots."""
    from f1strat.backtest import plot_all

    return plot_all(fd.backtest_plot_tables(), fig_dir)


def rounds_from(exclude_long: pd.DataFrame) -> pd.Series:
    """Round number → event name for every 2022 race (from the exclude table)."""
    ev = exclude_long.drop_duplicates("EventName").sort_values("event_order")
    return pd.Series(ev["EventName"].to_numpy(), index=ev["event_order"].to_numpy() + 1)


def notebook_figures(fig_dir: Path = FIG_DIR) -> list[Path]:
    """The figures notebooks 02 and 03 save, from their tables."""
    clean = fd.read("clean_laps.parquet")
    return [
        plot_deg_three_races(clean, fig_dir / "deg_bahrain_spain_hungary.png"),
        plot_deg_all_races(
            clean, rounds_from(fd.read("exclude_reasons")), fig_dir / "deg_all_races.png"
        ),
        plot_exclude_reasons(fd.read("exclude_reasons"), fig_dir / "exclude_reasons.png"),
        plot_pit_loss(fd.read("pit_loss"), fig_dir / "pit_loss.png"),
        plot_deg_curves(fd.read("deg_curves"), fig_dir / "deg_curves.png"),
        plot_long_tyre(
            fd.read("xgb_vs_bayes_long_tyre_errors"),
            fd.read("xgb_vs_bayes_long_tyre_stint"),
            fig_dir / "xgb_vs_bayes_long_tyre.png",
        ),
        plot_leclerc_hungary_hard(
            fd.read("leclerc_hungary_hard"), fig_dir / "leclerc_hungary_hard.png"
        ),
    ]


def run_notebooks(names: tuple[str, ...] = NOTEBOOKS) -> list[Path]:
    """Execute each notebook top to bottom in place (raises on any error)."""
    paths = []
    for name in names:
        nb = ROOT / "notebooks" / name
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
                str(nb),
            ],
            check=True,
            cwd=nb.parent,
        )
        paths.append(nb)
    return paths


def main(export: bool = False, notebooks: bool = False, readme_only: bool = False) -> None:
    """Redraw everything from ``data/figures`` (``export``: rebuild the tables
    first; ``notebooks``: also execute the notebooks; ``readme_only``: just
    the README assets)."""
    import logging

    from f1strat import readme_assets

    logging.disable(logging.WARNING)
    steps = []
    if export:
        steps.append(("export tables to data/figures", fd.export_all))
    if not readme_only:
        steps += [
            ("case-study figures", case_study_figures),
            ("backtest figures", backtest_figures),
            ("notebook figures", notebook_figures),
        ]
    steps.append(("README assets", readme_assets.build_all))
    if notebooks and not readme_only:
        steps.append(("notebooks", run_notebooks))
    for name, step in steps:
        t0 = time.time()
        paths = step()
        print(f"{name}: {len(paths)} files in {time.time() - t0:.0f} s", flush=True)
        for p in paths:
            print("  ", Path(p).relative_to(ROOT))


__all__ = ["backtest_figures", "case_study_figures", "main", "notebook_figures", "run_notebooks"]
