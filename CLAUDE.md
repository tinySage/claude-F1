# CLAUDE.md

## Project goal
An F1 race strategy simulator that re-examines three of Ferrari's controversial
2022 strategy calls:
- **Hungary**: Leclerc onto hards
- **Silverstone**: Leclerc left out under the late safety car
- **Monaco**: the double-stack

For each, the question is: using only what the pit wall knew at the decision lap,
was the call within the range of good options?

## Current phase
**Phase 6: Case studies.** Phases 0–5 are complete. Model: `docs/model.md`; simulator: `docs/simulator.md`; validation and the Phase 6 dress rehearsal: `docs/validation.md`. The simulator reads the committed thinned posterior `models/deg_full_2022_thin.nc`; the full posterior is gitignored (rebuild: `uv run python -m f1strat.deg_model full`, ~40 min, also rewrites the thinned file). Case-study fits go through `f1strat.rehearsal` (cutoff data, practice long runs, leakage-free parameters, divergence-free fits). See `ROADMAP.md` for the full plan and checkpoints.

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
models/             Posteriors; only the thinned full fit is committed
docs/               model.md, simulator.md, validation.md, case_studies.md (verdict rule, results)
notebooks/          Exploration and plots only. Imports from f1strat.
tests/              pytest tests
data/processed/     Processed parquet per race (committed). Schema: data/processed/README.md
reports/figures/    Saved figures from notebooks (committed)
cache/              FastF1 HTTP cache. Gitignored.
ROADMAP.md          Phase checklist with "done when" criteria
```

## Environment
- Python 3.11+, managed with `uv`. Run things with `uv run ...` (e.g. `uv run pytest`).
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
