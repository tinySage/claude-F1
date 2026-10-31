# Documentation

Read in this order: the model feeds the simulator, the simulator is validated,
and the validated pipeline judges the calls.

| Document | What it covers |
|---|---|
| [Tyre degradation model](model.md) | The Bayesian hierarchical lap-time model (PyMC): every term, priors, identifiability, held-out tests against an OLS baseline and XGBoost, and practice long runs in cutoff fits. |
| [Race simulator](simulator.md) | Race state, tyre inventory reconstruction, the single-car strategy search, the whole field with traffic, passing and SC/VSC, and the Phase 6 additions (session clock, double stack, SC-aware search). |
| [Validation](validation.md) | Backtests on 15 dry 2022 races, passing and SC-queue calibration, interval coverage, and the dress rehearsals (Spain, Canada) of the cutoff pipeline. |
| [Case studies](case_studies.md) | The verdict rule (committed before any case-study simulation), then Hungary (two decisions), Silverstone (v1, v2 with a restart layer) and Monaco (track-position accounting), each with its hindsight replay and limits. |

Also:
- [`../ROADMAP.md`](../ROADMAP.md): phase checklist, the decisions log, and what comes next.
- [`../data/processed/README.md`](../data/processed/README.md): the processed data schema.
- [`../data/figures/README.md`](../data/figures/README.md): the exact data behind every figure in these documents (`uv run pitwall figures` redraws them all).
- Commands: `uv run pitwall --help` (the module commands quoted in these documents, `python -m f1strat.<module>`, still work).
