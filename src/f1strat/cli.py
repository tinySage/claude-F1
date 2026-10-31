"""Command-line entry point: ``pitwall <command>`` (or ``python -m f1strat``).

Two paths (``README.md``, "Quick start" and "Full reproduction"):

- **Quick:** ``pitwall figures`` redraws every figure from the tables in
  ``data/figures/`` (:mod:`f1strat.figure_data`). No fits, no simulations.
- **Full:** ``data`` → ``clean`` → ``fit`` (→ ``thin``) → ``backtest`` →
  ``case-study hungary|silverstone|monaco`` → ``figures --export``, from raw FastF1
  downloads. Runtimes are listed per command in ``pitwall --help``.

Each command calls the ``main`` of the module that owns the step, so the
module entry points (``python -m f1strat.<module>``) keep working.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Callable

#: Rough wall-clock times on a 4-core machine, shown in ``--help``.
RUNTIMES: dict[str, str] = {
    "data": (
        "~5 min of downloads, but FastF1 allows 500 API calls an hour: from an empty "
        "cache expect one rerun after the limit resets (cached in cache/)"
    ),
    "clean": "~1 min",
    "fit": "~40 min for the full fit; ~60 min more for the two held-out splits",
    "thin": "~10 s",
    "backtest": "~25 min (params, calibrate, run, widening); rehearsals 15-75 min each",
    "case-study": (
        "hungary ~3 h (two cutoff refits, 8 checks each); "
        "silverstone ~2.5 h (v1 and v2); monaco ~2 min"
    ),
    "figures": "~1 min (--export adds ~2 min; --notebooks ~4 min)",
}


def _run(steps: list[tuple[str, Callable[[], object]]]) -> None:
    for name, step in steps:
        t0 = time.time()
        print(f"== {name}", flush=True)
        step()
        print(f"== {name}: done in {(time.time() - t0) / 60:.1f} min", flush=True)


def cmd_data(args: argparse.Namespace) -> None:
    """2022 race laps, the four decision races' practice long runs, and the
    2021 pass events (the case studies' circuit pass difficulty)."""
    from f1strat import data

    def practice() -> None:
        from f1strat.practice import long_runs

        raw = data.load_season(2022)
        for rnd in (6, 9, 10, 13):  # Spain and Canada rehearsals, Britain, Hungary
            long_runs(raw, rnd, refresh=True)

    def pass_2021() -> None:
        from f1strat.race_events import season_pass_events

        season_pass_events(2021, refresh=True)

    def races() -> None:
        failed = data.build_season(2022)
        if failed:
            raise SystemExit(
                f"rounds {failed} failed to download (see data/processed/2022/"
                "load_failures.log). FastF1 allows 500 API calls an hour: rerun "
                "`pitwall data` later; downloads so far are cached."
            )

    _run(
        [
            ("2022 races", races),
            ("practice long runs", practice),
            ("2021 pass events", pass_2021),
        ]
    )


def cmd_clean(args: argparse.Namespace) -> None:
    """Exclude reasons, dirty-air flags and the pit-loss table."""
    from f1strat import clean

    _run([("clean", clean.main)])


def cmd_fit(args: argparse.Namespace) -> None:
    """The Bayesian degradation model (writes the thinned posterior too)."""
    from f1strat import deg_model

    extra = [] if args.draws is None else ["--draws", str(args.draws)]
    _run([(f"fit {args.target}", lambda: deg_model.main([args.target, *extra]))])


def cmd_thin(args: argparse.Namespace) -> None:
    """Thin the saved full posterior to the committed 1,000-draw file."""
    from f1strat import deg_model

    _run([("thin", lambda: deg_model.main(["thin"]))])


def cmd_backtest(args: argparse.Namespace) -> None:
    """Simulator parameters, Phase 5 calibration, backtests and rehearsals."""
    from f1strat import backtest, race_events, rehearsal

    steps = {
        "params": [("simulator parameters", race_events.main)],
        "calibrate": [("calibration", lambda: backtest.main(["calibrate"]))],
        "run": [("backtests", lambda: backtest.main(["run"]))],
        "widening": [("widening 1.0", lambda: backtest.main(["widening"]))],
    }
    steps["all"] = steps["params"] + steps["calibrate"] + steps["run"] + steps["widening"]
    if args.action == "rehearsal":
        names = args.races or ["spain", "canada"]
        _run(
            [
                (
                    f"rehearsal {n}{' (cutoff data)' if c else ''}",
                    lambda n=n, c=c: rehearsal.main([n] + (["--cutoff-data"] if c else [])),
                )
                for n in names
                for c in (False, True)
            ]
        )
        return
    _run(steps[args.action])


def cmd_case_study(args: argparse.Namespace) -> None:
    """One case study end to end: cutoff refits (if not saved), verdicts under
    every check, hindsight replays and the exploratory extras."""
    from f1strat import case_study, monaco, restart

    hs = ["--hindsight-only"] if args.hindsight_only else []
    if args.race == "hungary":
        steps = [
            ("hungary lap 38", lambda: case_study.main(["hungary", *hs])),
            ("hungary lap 53", lambda: case_study.main(["hungary54", *hs])),
        ]
        if not args.hindsight_only:
            steps += [
                ("break-even", lambda: case_study.main(["hungary-break-even"])),
                ("hard posterior", lambda: case_study.main(["hungary-hard-posterior"])),
                ("pass sensitivity", lambda: case_study.main(["pass-sensitivity"])),
                (
                    "pass sensitivity, in-race",
                    lambda: case_study.main(["pass-sensitivity-in-race"]),
                ),
            ]
        steps += [("gap fan (README)", lambda: case_study.main(["hungary-fan"]))]
    elif args.race == "silverstone":
        steps = [("silverstone v1", lambda: case_study.main(["silverstone", *hs]))]
        if not args.hindsight_only:
            steps += [
                ("break-even", lambda: case_study.main(["silverstone-break-even"])),
                ("restart layer", restart.main),
            ]
        steps += [("silverstone v2", lambda: case_study.main(["silverstone", "--v2", *hs]))]
    else:
        steps = [
            ("monaco reconstruction", lambda: monaco.main([])),
            ("monaco verdict", lambda: monaco.main(["verdict"])),
        ]
    _run(steps)


def cmd_figures(args: argparse.Namespace) -> None:
    """Redraw every figure from the tables in data/figures; no fits, no
    simulations, no downloads."""
    from f1strat import figures

    figures.main(export=args.export, notebooks=args.notebooks, readme_only=args.readme)


def build_parser() -> argparse.ArgumentParser:
    """The ``pitwall`` argument parser."""
    p = argparse.ArgumentParser(
        prog="pitwall",
        description="What the Pit Wall Knew: Ferrari's 2022 strategy calls, judged with "
        "only what was known at the time.",
        epilog="Quick path: `pitwall figures`. Full path: data, clean, fit, backtest all, "
        "case-study hungary|silverstone|monaco, figures --export.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    def add(name: str, func: Callable, helptext: str) -> argparse.ArgumentParser:
        s = sub.add_parser(
            name,
            help=f"{helptext} [{RUNTIMES[name]}]",
            description=f"{func.__doc__.strip()} Runtime: {RUNTIMES[name]}.",
        )
        s.set_defaults(func=func)
        return s

    add("data", cmd_data, "download 2022 races, practice and 2021 pass events")
    add("clean", cmd_clean, "clean laps and estimate pit losses")
    s = add("fit", cmd_fit, "fit the Bayesian lap-time model")
    s.add_argument(
        "target",
        nargs="?",
        default="full",
        choices=["full", "splits", "all", "last_stint", "long_tyre"],
    )
    s.add_argument("--draws", type=int, default=None, help="posterior draws per chain")
    add("thin", cmd_thin, "thin the full posterior to the committed file")
    s = add("backtest", cmd_backtest, "simulator parameters, calibration, backtests")
    s.add_argument(
        "action",
        nargs="?",
        default="all",
        choices=["all", "params", "calibrate", "run", "widening", "rehearsal"],
    )
    s.add_argument("races", nargs="*", help="rehearsal races (spain, canada)")
    s = add("case-study", cmd_case_study, "run one case study")
    s.add_argument("race", choices=["hungary", "silverstone", "monaco"])
    s.add_argument(
        "--hindsight-only",
        action="store_true",
        help="rerun only the hindsight replays from the saved verdicts",
    )
    s = add("figures", cmd_figures, "redraw every figure from data/figures")
    s.add_argument(
        "--export",
        action="store_true",
        help="first rebuild data/figures from the result files, processed laps and "
        "committed thinned fits (after a full-path step)",
    )
    s.add_argument("--notebooks", action="store_true", help="also execute the notebooks")
    s.add_argument("--readme", action="store_true", help="only the README assets")
    return p


def main(argv: list[str] | None = None) -> None:
    """Parse ``argv`` and run the command."""
    args = build_parser().parse_args(argv)
    for noisy in ("fastf1", "requests_cache", "pymc", "nutpie"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
