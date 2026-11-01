# CLAUDE.md

Repository: **what-the-pit-wall-knew** (Python distribution of the same name;
the import package is `f1strat`).

## Project goal
An F1 race strategy simulator that re-examines three of Ferrari's controversial
2022 strategy calls:
- **Hungary**: Leclerc onto hards
- **Silverstone**: Leclerc left out under the late safety car
- **Monaco**: the double-stack

For each, the question is: using only what the pit wall knew at the decision lap,
was the call within the range of good options?

## Current phase
**Phase 7: Ship.** Phases 0–6 are complete (case-study verdicts and their limits: `docs/case_studies.md`). Model: `docs/model.md`; simulator: `docs/simulator.md`; validation and the Phase 6 dress rehearsal: `docs/validation.md`. The simulator reads the committed thinned posterior `models/deg_full_2022_thin.nc`; the full posterior is gitignored (rebuild: `uv run python -m f1strat.deg_model full`, ~40 min, also rewrites the thinned file). Case-study fits go through `f1strat.rehearsal` (cutoff data, practice long runs, leakage-free parameters, divergence-free fits). See `ROADMAP.md` for the full plan and checkpoints.

## Repo layout
```
src/f1strat/        Package code. All logic lives here.
  data.py           FastF1 loading and parquet I/O
  clean.py          Lap filtering, dirty-air flags, pit loss estimates
  deg_model.py      Tyre degradation models (linear, PyMC hierarchical, XGBoost)
  simulator.py      Race state, single car, strategy search, whole field with traffic and SC/VSC
  tyres.py          Race-day tyre inventory from practice and qualifying
  race_events.py    Passing and SC/VSC parameters from 2022 (sim_params.json)
  backtest.py       Phase 5 backtests and calibration (passing, SC closing)
  practice.py       FP1-FP3 long runs for cutoff fits
  rehearsal.py      Cutoff pipeline: refit, leakage-free parameters, search, compare
  case_study.py     Phase 6 verdicts (Hungary, Silverstone): rule, checks, hindsight
  restart.py        Restart layer (Silverstone v2): pace offset and pass multiplier
  monaco.py         Monaco track-position accounting (wet crossover; no simulator)
  cli.py            `pitwall` command line (data, clean, fit, thin, backtest, case-study, figures)
  figure_data.py    The data behind every figure (data/figures), extract and load
  figures.py        Draw every figure from data/figures (no fits or simulations)
  readme_assets.py  README banner, verdict cards, animation, light/dark figures
models/             Posteriors; only thinned fits are committed (full fit and held-out splits)
docs/               model.md, simulator.md, validation.md, case_studies.md (verdict rule, results)
notebooks/          Exploration and plots only. Imports from f1strat.
tests/              pytest tests
data/processed/     Processed parquet per race (committed). Schema: data/processed/README.md
data/figures/       The exact data each figure plots (committed); see its README.md
reports/figures/    Saved figures and result files (committed)
assets/readme/      README images, light and dark (from `pitwall figures`)
.github/workflows/  CI: ruff and the fast tests
cache/              FastF1 HTTP cache. Gitignored.
ROADMAP.md          Phase checklist with "done when" criteria
```

## Environment
- Python 3.11+, managed with `uv`. Run things with `uv run ...` (e.g. `uv run pytest`).
- Commands: `uv run pitwall --help`. Quick path: `uv run pitwall figures` (draws every
  figure from `data/figures`). After a full-path step: `uv run pitwall figures --export`.
- Every new figure gets its table in `data/figures` (`f1strat.figure_data`) and is drawn
  only from it, so no figure ever needs a refit to redraw.
- Tests: `uv run pytest` is fast; `uv run pytest -m ""` adds the slow (PyMC) and network tests.
- Lint: `uvx ruff check src tests notebooks` and `uvx ruff format src tests`.
- Modelling stack (Phase 3): PyMC sampled with nutpie, ArviZ, XGBoost, scikit-learn.

## Conventions
- Logic lives in `src/f1strat/`. Notebooks are only for exploration and plots;
  anything reusable moves into the package.
- Use type hints and docstrings on all public functions and modules.
- **Ask the user before any decision that changes the method** (modelling
  choices, filtering rules, data sources, validation approach).
- No future leakage in case studies: fits for a decision lap use only data
  available at that lap.
- After each step, explain in two or three lines what was done and why. The user
  needs to be able to explain every part of this in interviews.
- Small, logical commits.

## Roadmap upkeep
When a checkpoint in `ROADMAP.md` is met, tick its box (`- [x]`) in the same
commit as the work that met it. When a phase's "done when" is satisfied, update
"Current phase" above.
