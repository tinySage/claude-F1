"""README images, each in a dark and a light version (``assets/readme/``).

Style: a timing screen. Near-black (or near-white) surface, white and grey
text, monospace numbers, one racing-red accent. Red never carries meaning on
its own: every red mark has a text label next to it. SVGs use generic font
families only (GitHub renders SVG as an image, so web fonts would not load).

Built only from the tables in ``data/figures/`` (``pitwall figures --readme``):

- ``banner``: Leclerc's Hungary race trace to lap 38 (known), then the lap-38
  forecast's simulated futures for "hards at 39" (uncertain) with the real
  race threaded through (``data/figures/hungary_fan.csv``).
- ``verdicts``: one card per decision (``data/figures/readme_verdicts.csv``,
  written by hand from ``docs/case_studies.md``); the decision-lap verdict is
  the large label, any hindsight outcome a smaller line marked "Hindsight".
- ``hero``: the same forecast building lap by lap from lap 38 (GIF, about
  10 s), with a static PNG of the last frame.
- ``fig_hungary``, ``fig_silverstone``, ``fig_monaco``, ``fig_calibration``:
  the case-study and calibration figures the README shows, restyled.
"""

from __future__ import annotations

from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from f1strat import figure_data as fd

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "assets" / "readme"

SANS = "system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace"

#: Colour tokens per theme. Red vs the grey series: CVD ΔE 12.3 (dark) and
#: 10.0 (light), both above 3:1 against the surface (dataviz validator).
THEMES: dict[str, dict[str, str]] = {
    "dark": {
        "bg": "#0d0d0d",
        "panel": "#161616",
        "line": "#2a2a28",
        "ink": "#f2f1ee",
        "ink2": "#b3b2ac",
        "muted": "#85847e",
        "grid": "#262624",
        "red": "#e8352c",
        "fan": "#9a9993",
    },
    "light": {
        "bg": "#fcfcfb",
        "panel": "#f2f1ee",
        "line": "#dcdbd5",
        "ink": "#0b0b0b",
        "ink2": "#4a4946",
        "muted": "#6f6e69",
        "grid": "#e1e0d9",
        "red": "#c42820",
        "fan": "#6f6e69",
    },
}


def _text(
    x: float,
    y: float,
    s: str,
    size: float,
    fill: str,
    family: str = SANS,
    weight: str = "400",
    anchor: str = "start",
    extra: str = "",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"{extra}>'
        f"{escape(s)}</text>"
    )


def _svg(w: int, h: int, body: list[str], title: str, desc: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" role="img" aria-labelledby="t d">'
        f'<title id="t">{escape(title)}</title><desc id="d">{escape(desc)}</desc>'
        + "".join(body)
        + "</svg>\n"
    )


#: First lap the simulated gap is drawn. The state at lap 38 already carries
#: Verstappen's known out-lap loss and the simulator books Leclerc's whole pit
#: loss on his in lap (39), so the simulated and real gaps are comparable only
#: once both stops are done (as in the reality check, ``docs/case_studies.md``).
FAN_FROM_LAP: int = 40


def load_fan(fan_table: str = "hungary_fan") -> tuple[pd.DataFrame, pd.Series]:
    """The lap-38 forecast (from lap :data:`FAN_FROM_LAP`) and the real race from lap 1, both as
    Leclerc's lead over Verstappen in seconds (positive = Leclerc ahead), from
    ``data/figures/hungary_fan.csv`` and ``hungary_trace.csv``."""
    fan = -fd.read(fan_table).set_index("lap")  # stored as the gap (positive = behind)
    fan = fan.loc[FAN_FROM_LAP:]
    trace = fd.read("hungary_trace")
    real = pd.Series(-trace["gap_s"].to_numpy(float), index=trace["lap"].astype(int))
    return fan, real


def _poly(xs: np.ndarray, ys: np.ndarray, sx, sy) -> str:
    return " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, ys) if np.isfinite(y))


def banner(theme: str, fan: pd.DataFrame, real: pd.Series) -> str:
    """The banner SVG: title, subtitle and the known/uncertain race trace."""
    c = THEMES[theme]
    w, h = 1280, 500
    x0, x1, y0, y1 = 64, 1216, 240, 452
    lo, hi = -36.0, 30.0
    cut, total = 38, 70

    def sx(lap: float) -> float:
        return x0 + (lap - 1) / (total - 1) * (x1 - x0)

    def sy(v: float) -> float:
        return y1 - (np.clip(v, lo, hi) - lo) / (hi - lo) * (y1 - y0)

    b = [
        f'<rect width="{w}" height="{h}" rx="14" fill="{c["bg"]}"/>',
        f'<rect x="64" y="56" width="10" height="10" fill="{c["red"]}"/>',
        _text(
            84,
            66,
            "2022 SEASON · THREE STRATEGY CALLS, RE-EXAMINED",
            13,
            c["muted"],
            MONO,
            extra=' letter-spacing="1.5"',
        ),
        _text(62, 126, "What the Pit Wall Knew", 54, c["ink"], weight="700"),
        _text(
            64,
            166,
            "Judging Ferrari's 2022 strategy calls using only what was known at the time",
            22,
            c["ink2"],
        ),
        f'<defs><clipPath id="plot"><rect x="{x0}" y="{y0 - 4}" width="{x1 - x0}" '
        f'height="{y1 - y0 + 8}"/></clipPath></defs>',
    ]
    # Level line (lead 0) and the cutoff.
    b.append(
        f'<line x1="{x0}" x2="{x1}" y1="{sy(0):.1f}" y2="{sy(0):.1f}" '
        f'stroke="{c["grid"]}" stroke-width="1"/>'
    )
    b.append(
        f'<line x1="{sx(cut):.1f}" x2="{sx(cut):.1f}" y1="{y0 - 22}" y2="{y1 + 6}" '
        f'stroke="{c["muted"]}" stroke-width="1"/>'
    )
    b.append(
        _text(
            sx(cut) - 10,
            y0 - 10,
            "KNOWN",
            12,
            c["muted"],
            MONO,
            anchor="end",
            extra=' letter-spacing="1.5"',
        )
    )
    b.append(
        _text(
            sx(cut) + 10, y0 - 10, "UNCERTAIN", 12, c["muted"], MONO, extra=' letter-spacing="1.5"'
        )
    )
    b.append(_text(sx(cut), y1 + 24, "LAP 38", 12, c["ink2"], MONO, anchor="middle"))
    for lap in (1, 70):
        b.append(
            _text(
                sx(lap),
                y1 + 24,
                f"LAP {lap}",
                12,
                c["muted"],
                MONO,
                anchor="start" if lap == 1 else "end",
            )
        )
    # Fan of simulated futures.
    laps = fan.index.to_numpy()
    sims = [k for k in fan.columns if k.startswith("s")][:200]
    b.append(
        f'<g clip-path="url(#plot)" fill="none" stroke="{c["fan"]}" '
        f'stroke-width="1" stroke-opacity="{0.10 if theme == "dark" else 0.09}" '
        f'stroke-linejoin="round">'
    )
    for k in sims:
        b.append(f'<polyline points="{_poly(laps, fan[k].to_numpy(), sx, sy)}"/>')
    b.append("</g>")
    known = real.loc[1:cut]
    after = real.loc[cut:]
    b.append(
        f'<g clip-path="url(#plot)" fill="none" stroke-width="2.5" '
        f'stroke-linejoin="round" stroke-linecap="round">'
        f'<polyline stroke="{c["ink"]}" points="'
        f'{_poly(known.index.to_numpy(), known.to_numpy(), sx, sy)}"/>'
        f'<polyline stroke="{c["red"]}" points="'
        f'{_poly(after.index.to_numpy(), after.to_numpy(), sx, sy)}"/></g>'
    )
    # Legend, top right of the plot (identity never by colour alone).
    lx, ly = 900, y0 - 52
    b += [
        f'<line x1="{lx}" x2="{lx + 26}" y1="{ly}" y2="{ly}" stroke="{c["ink"]}" '
        f'stroke-width="2.5"/>',
        _text(lx + 34, ly + 4, "Race to lap 38", 13, c["ink2"]),
        f'<line x1="{lx}" x2="{lx + 26}" y1="{ly + 22}" y2="{ly + 22}" stroke="{c["fan"]}" '
        f'stroke-width="2" stroke-opacity="0.6"/>',
        _text(lx + 34, ly + 26, "Simulated futures, hards at 39", 13, c["ink2"]),
        f'<line x1="{lx}" x2="{lx + 26}" y1="{ly + 44}" y2="{ly + 44}" stroke="{c["red"]}" '
        f'stroke-width="2.5"/>',
        _text(lx + 34, ly + 48, "Real race", 13, c["ink2"]),
    ]
    b.append(_text(x0, y1 + 24 + 0, "", 1, c["muted"]))
    b.append(_text(x1 + 6, sy(0) + 4, "level", 11, c["muted"], MONO))
    return _svg(
        w,
        h,
        b,
        "What the Pit Wall Knew",
        "Judging Ferrari's 2022 strategy calls using only what was known at the "
        "time. Leclerc's lead over Verstappen at Hungary 2022: the real race to "
        "lap 38, then simulated futures for new hards at lap 39 with the real race "
        "drawn through them.",
    )


def _wrap(s: str, width: int) -> list[str]:
    import textwrap

    return textwrap.wrap(s, width)


def verdict_cards(theme: str) -> str:
    """One SVG, a 2×2 grid of cards (one per decision): the decision-lap
    verdict is the large label; the hindsight outcome a smaller, marked line."""
    c = THEMES[theme]
    cw, ch, gap, pad = 432, 248, 16, 22
    w, h = 2 * cw + gap, 2 * ch + gap
    b = []
    cards = fd.read("readme_verdicts").to_dict("records")
    for i, card in enumerate(cards):
        x, y0 = (i % 2) * (cw + gap), (i // 2) * (ch + gap)
        b.append(
            f'<rect x="{x + 0.5}" y="{y0 + 0.5}" width="{cw - 1}" height="{ch - 1}" '
            f'rx="10" fill="{c["panel"]}" stroke="{c["line"]}"/>'
        )
        b.append(
            _text(
                x + pad, y0 + 34, card["race"], 13, c["ink2"], MONO, extra=' letter-spacing="1.2"'
            )
        )
        b.append(
            _text(
                x + cw - pad, y0 + 34, card["lap"], 13, c["ink"], MONO, weight="700", anchor="end"
            )
        )
        b.append(_text(x + pad, y0 + 60, card["call"], 15, c["ink2"]))
        # The verdict, with a marker beside it (the label carries the meaning).
        vy = y0 + 100
        mx, my = x + pad, vy - 16
        if card["mark"] == "red":
            b.append(f'<rect x="{mx}" y="{my}" width="14" height="14" rx="2" fill="{c["red"]}"/>')
        elif card["mark"] == "hollow":
            b.append(
                f'<rect x="{mx + 1}" y="{my + 1}" width="12" height="12" rx="2" '
                f'fill="none" stroke="{c["ink"]}" stroke-width="2"/>'
            )
        else:
            b.append(
                f'<rect x="{mx + 1}" y="{my + 1}" width="12" height="12" rx="2" '
                f'fill="{c["muted"]}"/>'
            )
        b.append(_text(x + pad + 24, vy, card["verdict"], 25, c["ink"], weight="700"))
        ny = y0 + 142
        b.append(_text(x + pad, ny, card["number"], 22, c["ink"], MONO, weight="700"))
        for k, line in enumerate(_wrap(card["caption"], 56)):
            b.append(_text(x + pad, ny + 22 + 17 * k, line, 13.5, c["ink2"]))
        hy = y0 + ch - 40
        b.append(
            f'<line x1="{x + pad}" x2="{x + cw - pad}" y1="{hy - 14}" y2="{hy - 14}" '
            f'stroke="{c["line"]}"/>'
        )
        b.append(
            _text(x + pad, hy + 4, "HINDSIGHT", 11, c["muted"], MONO, extra=' letter-spacing="1.2"')
        )
        for k, line in enumerate(_wrap(card["hindsight"], 50)):
            b.append(_text(x + pad + 84, hy + 4 + 16 * k, line, 12.5, c["muted"]))
    return _svg(
        w,
        h,
        b,
        "Verdicts",
        "Hungary lap 39, hards: within range, 0.0 points lost. Hungary lap 54, used "
        "softs: outside range, 5.1 points lost. Silverstone lap 39, left out under "
        "the safety car: inconclusive. Monaco laps 18 and 21: not assessable at the "
        "time. Hindsight outcomes are listed separately on each card.",
    )


# --- Matplotlib figures ---------------------------------------------------------------


def _mpl(theme: str) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    c = THEMES[theme]
    return {
        "figure.facecolor": c["bg"],
        "axes.facecolor": c["bg"],
        "savefig.facecolor": c["bg"],
        "axes.edgecolor": c["grid"],
        "axes.labelcolor": c["ink2"],
        "xtick.color": c["muted"],
        "ytick.color": c["ink2"],
        "axes.grid": True,
        "grid.color": c["grid"],
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlecolor": c["ink"],
        "axes.titlelocation": "left",
        "text.color": c["ink"],
        "xtick.labelsize": 10,
        "ytick.labelsize": 10.5,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "axes.axisbelow": True,
    }


def _save(fig, name: str, theme: str) -> Path:
    import matplotlib.pyplot as plt

    path = OUT_DIR / f"{name}_{theme}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.25, metadata={"Software": None})
    plt.close(fig)
    return path


def fig_hungary(theme: str) -> Path:
    """Leclerc's mean points per plan at each Hungary decision: decision-lap
    forecast (filled) against the hindsight replay (hollow); the call labelled."""
    import matplotlib.pyplot as plt

    c = THEMES[theme]
    panels = []
    tab = fd.read("readme_hungary")
    for title, g in tab.groupby("panel", sort=False):
        call = g.loc[g["is_call"], "family"].iloc[0]
        meas = {r.family: {"mean_points": r.decision_points} for r in g.itertuples()}
        hind = {r.family: {"mean_points": r.hindsight_points} for r in g.itertuples()}
        rows = sorted(meas, key=lambda n: meas[n]["mean_points"])
        panels.append((title, call, rows, meas, hind))
    with plt.rc_context(_mpl(theme)):
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharex=True)
        for ax, (title, call, rows, meas, hind) in zip(axes, panels):
            y = np.arange(len(rows))
            for i, n in enumerate(rows):
                is_call = n == call
                col = c["red"] if is_call else c["fan"]
                d, hs = meas[n]["mean_points"], hind[n]["mean_points"]
                ax.plot([d, hs], [i, i], color=c["grid"], lw=2, zorder=1, solid_capstyle="round")
                ax.scatter(hs, i, s=70, facecolor=c["bg"], edgecolor=col, lw=2, zorder=3)
                ax.scatter(d, i, s=95, color=col, edgecolor=c["bg"], lw=1.5, zorder=4)
            labels = [("THE CALL  " if n == call else "") + n.replace("call: ", "") for n in rows]
            ax.set_yticks(y, labels)
            for t, n in zip(ax.get_yticklabels(), rows):
                if n == call:
                    t.set_color(c["ink"])
                    t.set_fontweight("bold")
            ax.set_title(title, pad=10)
            ax.grid(axis="y", visible=False)
            ax.set_xlim(0, 25)
            ax.set_xlabel("Leclerc's mean points (P1 = 25)")
            ax.tick_params(axis="y", length=0)
        h1 = plt.Line2D(
            [],
            [],
            ls="",
            marker="o",
            ms=8,
            color=c["fan"],
            mec=c["bg"],
            label="Decision lap (information at the cutoff)",
        )
        h2 = plt.Line2D(
            [],
            [],
            ls="",
            marker="o",
            ms=8,
            mfc=c["bg"],
            mec=c["fan"],
            mew=2,
            label="Hindsight replay (what actually happened)",
        )
        fig.legend(
            handles=[h1, h2],
            loc="lower center",
            ncol=2,
            bbox_to_anchor=(0.5, -0.07),
            labelcolor=c["ink2"],
        )
        fig.tight_layout(w_pad=3)
        return _save(fig, "fig_hungary", theme)


def fig_silverstone(theme: str) -> Path:
    """Passes in the first three laps after the restart: the real race against
    the model's forecast and hindsight replays (v1, and v2 with the restart layer)."""
    import matplotlib.pyplot as plt

    c = THEMES[theme]
    tab = fd.read("readme_silverstone")
    rows = [
        (r.source, int(r.passes) if r.real else float(r.passes), bool(r.real))
        for r in tab.itertuples()
    ]
    p1 = float(tab["v2_hindsight_p1"].iloc[0])
    with plt.rc_context(_mpl(theme)):
        fig, ax = plt.subplots(figsize=(9.5, 3.3))
        y = np.arange(len(rows))[::-1]
        for yi, (_name, v, real) in zip(y, rows):
            ax.barh(yi, v, height=0.42, color=c["red"] if real else c["fan"])
            ax.text(
                v + 0.15,
                yi,
                f"{v:.1f}" if not real else f"{v}",
                va="center",
                color=c["ink"],
                fontfamily="DejaVu Sans Mono",
                fontsize=11,
            )
        ax.set_yticks(y, [r[0] for r in rows])
        ax.get_yticklabels()[0].set_fontweight("bold")
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, 10.5)
        ax.set_xlabel("On-track passes, whole field, laps 43–45")
        ax.set_title("Silverstone: the model can't reproduce the restart", pad=10)
        ax.text(
            0,
            -0.36,
            f"Hindsight replay (v2): Leclerc P1 in {p1:.0%} of replays. He finished P4.",
            transform=ax.transAxes,
            color=c["ink2"],
            fontsize=10.5,
        )
        fig.tight_layout()
        return _save(fig, "fig_silverstone", theme)


def fig_monaco(theme: str) -> Path:
    """Monaco hindsight outcomes per alternative (not verdicts): Leclerc's points
    if the end-of-lap-23 order held, with the call labelled."""
    import matplotlib.pyplot as plt

    c = THEMES[theme]
    panels = []
    for title, g in fd.read("readme_monaco").groupby("panel", sort=False):
        panels.append(
            (
                title,
                {
                    "plans": {r.plan: {"mean_points": r.mean_points} for r in g.itertuples()},
                    "call": g.loc[g["is_call"], "plan"].iloc[0],
                },
            )
        )
    with plt.rc_context(_mpl(theme)):
        fig, axes = plt.subplots(
            1, 2, figsize=(11.5, 3.4), sharex=True, gridspec_kw={"width_ratios": [1, 1]}
        )
        for ax, (title, h) in zip(axes, panels):
            names = sorted(h["plans"], key=lambda n: h["plans"][n]["mean_points"])
            y = np.arange(len(names))
            for yi, n in zip(y, names):
                call = n == h["call"]
                v = h["plans"][n]["mean_points"]
                ax.barh(yi, v, height=0.42, color=c["red"] if call else c["fan"])
                ax.text(
                    v + 0.4,
                    yi,
                    f"{v:.1f}",
                    va="center",
                    color=c["ink"],
                    fontfamily="DejaVu Sans Mono",
                    fontsize=10.5,
                )
            labels = [
                ("THE CALL  " + n.replace("call: ", "")) if n == h["call"] else n for n in names
            ]
            ax.set_yticks(y, labels)
            for t, n in zip(ax.get_yticklabels(), names):
                if n == h["call"]:
                    t.set_fontweight("bold")
                    t.set_color(c["ink"])
            ax.set_ylim(-0.6, 3.6)
            ax.tick_params(axis="y", length=0)
            ax.grid(axis="y", visible=False)
            ax.set_xlim(0, 28)
            ax.set_title(title, pad=10)
            ax.set_xlabel("Leclerc's points if the lap-23 order held")
        fig.suptitle(
            "Monaco in hindsight (outcomes, not verdicts)",
            x=0.01,
            ha="left",
            fontsize=13,
            fontweight="bold",
            color=c["ink"],
        )
        fig.tight_layout(w_pad=3)
        return _save(fig, "fig_monaco", theme)


def fig_calibration(theme: str) -> Path:
    """Backtest interval coverage (15 races, replay, widening 1.0) against nominal."""
    import matplotlib.pyplot as plt

    c = THEMES[theme]
    rows = [(r.label, r.nominal, r.achieved) for r in fd.read("readme_calibration").itertuples()]
    with plt.rc_context(_mpl(theme)):
        fig, ax = plt.subplots(figsize=(9.5, 3.2))
        y = np.arange(len(rows))[::-1]
        for yi, (_, nominal, got) in zip(y, rows):
            ax.plot([nominal, nominal], [yi - 0.28, yi + 0.28], color=c["ink2"], lw=2)
            ax.scatter(got, yi, s=70, color=c["red"], edgecolor=c["bg"], lw=2, zorder=3)
            ax.text(
                max(got, nominal) + 0.02,
                yi,
                f"{got:.2f}",
                va="center",
                color=c["ink"],
                fontfamily="DejaVu Sans Mono",
                fontsize=11,
            )
        ax.set_yticks(y, [r[0] for r in rows])
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0.3, 1.0)
        ax.set_xlabel("Share of actual results inside the simulated interval")
        ax.set_title("Backtests: 15 dry 2022 races, replayed from lap 1", pad=10)
        h1 = plt.Line2D(
            [],
            [],
            color=c["ink2"],
            lw=2,
            marker="|",
            ms=12,
            ls="",
            label="Nominal (perfect calibration)",
        )
        h2 = plt.Line2D(
            [], [], ls="", marker="o", ms=8, color=c["red"], mec=c["bg"], label="Achieved"
        )
        ax.legend(handles=[h1, h2], loc="lower left", labelcolor=c["ink2"])
        fig.tight_layout()
        return _save(fig, "fig_calibration", theme)


# --- Animation ------------------------------------------------------------------------

#: Frames per lap and frame length (ms): 32 laps × 2 × 120 ms + a 2.3 s hold ≈ 10 s.
FRAMES_PER_LAP, FRAME_MS, HOLD_MS = 2, 120, 2300


def hero(theme: str, fan: pd.DataFrame, real: pd.Series) -> tuple[Path, Path]:
    """The animated forecast (GIF) and its last frame (PNG)."""
    import io

    import matplotlib.pyplot as plt
    from PIL import Image

    c = THEMES[theme]
    laps = fan.index.to_numpy()
    sims = [k for k in fan.columns if k.startswith("s")]
    start, cut, total = 30, 38, 70
    lo, hi = -36, 30
    frames: list[Image.Image] = []
    stops = [(L, f) for L in range(cut, total + 1) for f in range(FRAMES_PER_LAP)]
    stops = stops[FRAMES_PER_LAP - 1 :]  # start on lap 38 exactly
    with plt.rc_context(_mpl(theme)):
        for lap, f in stops:
            upto = lap if f == FRAMES_PER_LAP - 1 else lap - 0.5
            fig, ax = plt.subplots(figsize=(8, 4.5), dpi=110)
            fig.subplots_adjust(left=0.09, right=0.97, top=0.80, bottom=0.13)
            m = laps <= np.floor(upto)
            sel = fan.loc[laps[m]]
            if m.sum() > 1:
                ax.fill_between(laps[m], sel["p90"], sel["p10"], color=c["fan"], alpha=0.16, lw=0)
                for s in sims[:150]:
                    ax.plot(laps[m], sel[s], color=c["fan"], lw=0.7, alpha=0.10)
            known = real.loc[start:cut]
            ax.plot(known.index, known.to_numpy(), color=c["ink"], lw=2.2, solid_capstyle="round")
            ra = real.loc[cut : int(np.floor(upto))]
            ax.plot(ra.index, ra.to_numpy(), color=c["red"], lw=2.4, solid_capstyle="round")
            if len(ra):
                ax.scatter(
                    ra.index[-1],
                    ra.iloc[-1],
                    s=40,
                    color=c["red"],
                    edgecolor=c["bg"],
                    lw=1.5,
                    zorder=5,
                )
            ax.axvline(cut, color=c["muted"], lw=1)
            ax.axhline(0, color=c["grid"], lw=1)
            ax.text(
                cut - 0.5,
                hi - 2,
                "KNOWN",
                ha="right",
                color=c["muted"],
                fontfamily="DejaVu Sans Mono",
                fontsize=9,
            )
            ax.text(
                cut + 0.5,
                hi - 2,
                "FORECAST AT LAP 38",
                ha="left",
                color=c["muted"],
                fontfamily="DejaVu Sans Mono",
                fontsize=9,
            )
            if upto >= 40:
                ax.text(39.5, lo + 2.5, "laps 39–40:\nboth cars pit", ha="center",
                        color=c["muted"], fontsize=8.5)
            if upto >= 54:
                ax.annotate(
                    "lap 54: second stop\n(judged separately)",
                    xy=(54.6, real.loc[55] / 2),
                    xytext=(57, -12),
                    color=c["ink2"],
                    fontsize=9,
                    arrowprops={"arrowstyle": "-", "color": c["muted"], "lw": 0.8},
                )
            ax.set_xlim(start, total)
            ax.set_ylim(lo, hi)
            ax.set_xlabel("Lap")
            ax.set_ylabel("Leclerc ahead of Verstappen (s)")
            ax.grid(axis="x", visible=False)
            fig.text(
                0.09,
                0.92,
                "Hungary 2022: hards at lap 39, as forecast at lap 38",
                fontsize=13,
                fontweight="bold",
                color=c["ink"],
            )
            known_ink = "White" if theme == "dark" else "Black"
            fig.text(
                0.09,
                0.865,
                "Grey: 300 simulated futures (band: 10–90%).  "
                f"Red: the real race.  {known_ink}: known at the decision.",
                fontsize=9.5,
                color=c["ink2"],
            )
            fig.text(
                0.97,
                0.92,
                f"LAP {int(np.floor(upto)):>2}/{total}",
                ha="right",
                fontsize=13,
                fontfamily="DejaVu Sans Mono",
                fontweight="bold",
                color=c["ink"],
            )
            buf = io.BytesIO()
            fig.savefig(buf, format="png", facecolor=c["bg"])
            plt.close(fig)
            buf.seek(0)
            frames.append(Image.open(buf).convert("RGB"))
    png = OUT_DIR / f"hero_{theme}.png"
    frames[-1].save(png, optimize=True)
    pal = frames[-1].quantize(colors=96, method=Image.Quantize.MEDIANCUT)
    q = [fr.quantize(palette=pal, dither=Image.Dither.NONE) for fr in frames]
    durations = [FRAME_MS] * (len(q) - 1) + [HOLD_MS]
    gif = OUT_DIR / f"hero_{theme}.gif"
    q[0].save(
        gif,
        save_all=True,
        append_images=q[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=1,
    )
    return gif, png


def build_all() -> list[Path]:
    """Write every README asset in both themes; returns the paths."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fan, real = load_fan()
    paths = []
    for theme in THEMES:
        for name, svg in (("banner", banner(theme, fan, real)), ("verdicts", verdict_cards(theme))):
            p = OUT_DIR / f"{name}_{theme}.svg"
            p.write_text(svg)
            paths.append(p)
        paths += [
            fig_hungary(theme),
            fig_silverstone(theme),
            fig_monaco(theme),
            fig_calibration(theme),
        ]
        paths += list(hero(theme, fan, real))
    return paths


__all__ = [
    "THEMES",
    "banner",
    "build_all",
    "fig_calibration",
    "fig_hungary",
    "fig_monaco",
    "fig_silverstone",
    "hero",
    "load_fan",
    "verdict_cards",
]
