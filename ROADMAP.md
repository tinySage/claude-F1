# Roadmap

Question for each case study: using only what the Ferrari pit wall knew at the
decision lap, was the call within the range of good options?

Case studies (2022):
- **Hungary**: Leclerc onto hards
- **Silverstone**: Leclerc left out under the late safety car
- **Monaco**: the double-stack

## Decisions log
- **Missing compounds (Phase 2):** laps with no recorded compound (334 laps:
  Austria stint 3 for 12 drivers, Monaco stint 5 for STR) are not filled in.
  They are excluded from modelling (`exclude_reason = missing_compound`), and
  **Austria is dropped from all backtests** (Phase 5).
- **Mixed-conditions races (Phase 2):** Imola, Monaco, Singapore and Japan
  2022 get the race-level reason `mixed_conditions` (after `wet_compound`), so
  none of their laps feed the dry degradation fit. Monaco's case study gets
  its own handling in Phase 6.
- **Pit loss for mixed-conditions circuits (Phase 2):** recorded per circuit in
  `pit_loss.parquet` via `source_year` and `pit_loss_status`.
  - **Monaco: 2021** (`reference`). Green-flag, dry-to-dry stops from the most
    recent dry Monaco GP that can be downloaded. No F1 account credentials are
    used, and 2023–2026 timing data needs sign-in (live timing returns 401; the
    FastF1 mirror has no copy), so 2021 is the most recent available. Pit lane
    length barely changes between years. Only needed for the Phase 6 case
    study.
  - **Imola, Singapore, Suzuka: not needed** (`not_needed`, value null). All
    three are mixed-conditions races, so they are out of the dry fit and the
    Phase 5 backtests, and none is a case study.
- **Rainfall (Phase 2):** not an exclude reason. `Rainfall` stays as a boolean
  column for Phase 3 to test as a covariate. Hungary's cool, damp final third
  (drizzle, everyone on slicks) is part of why the hard tyre struggled, so
  those laps are evidence. Real wet running is caught by `wet_compound` and
  `mixed_conditions`.
- **Pit loss stops (Phase 2):** stops whose in lap is lap 1 are not used.
- **Yellow flags (Phase 2):** not an exclude reason.
- **Missing tyre age (Phase 3):** 73 clean Austria laps have a compound but no
  `TyreLife`. They are dropped from model data, not filled in, matching the
  missing-compound decision.
- **Degradation model spec (Phase 3):** `docs/model.md`, approved with three
  changes:
  - car-specific tyre terms (`team_compound[t,c,r]` for soft/hard and
    `team_deg[t,c]`), so Ferrari's hards can behave differently from the
    field's;
  - warm-up only after pit stops, with a separate `opening` term for race laps
    2–3;
  - tyre age scaled per 10 laps and race lap centred inside the model, with
    every effect reported in s and s/lap.
- **Held-out test sets (Phase 3):** `last_stint` (each driver's final stint per
  race, for drivers with an earlier clean stint) and `long_tyre` (train on
  `TyreLife <= 15`, test on older tyres). Fixed and reused for every model.

---

## Phase 0: Setup
- [x] ROADMAP.md and CLAUDE.md written
- [x] Python 3.11+ environment with uv; core dependencies installed
- [x] Repo layout (`src/f1strat/`, `notebooks/`, `tests/`, `data/`, `cache/`) and `.gitignore`
- [x] Smoke test: FastF1 loads 2022 Hungarian GP race; Leclerc's laps have `Compound` and `TyreLife`

**Done when:** `uv run pytest` passes, including the FastF1 smoke test, and the setup is pushed.

## Phase 1: Data
- [x] Pull all 2022 race laps with FastF1
- [x] Save to parquet in `data/processed/`

**Done when:** every 2022 race's laps are in parquet with a documented schema and a loader in `src/f1strat/data.py`.

## Phase 2: Cleaning + EDA
- [x] Label laps unusable for the deg model with one `exclude_reason` (lap 1, after chequered, in/out laps, red flag, SC, VSC, restart lap, deleted, inaccurate, missing/wet compound, mixed conditions, >107% outlier)
- [x] Flag dirty-air laps (`gap_ahead_s < 1.0`; keep them, flag only)
- [x] Estimate green-flag pit lane loss per circuit (`data/processed/2022/pit_loss.parquet`): 18 circuits from their own 2022 stops, Monaco from 2021, and Imola, Singapore and Suzuka marked not needed (see Decisions log)
- [x] EDA notebook with plots (`notebooks/02_eda.ipynb`, figures in `reports/figures/`)

**Done when:** `src/f1strat/clean.py` produces a tested clean dataset with a dirty-air flag, and a per-circuit pit loss table exists.

## Phase 3: Tyre degradation model
- [x] Linear baseline (per-race OLS; results in `docs/model.md`)
- [ ] Bayesian hierarchical model in PyMC: car pace offsets, per-compound degradation with a cliff term, fuel effect, track evolution
- [ ] XGBoost fit, used only to show it fails to extrapolate on long stints

**Done when:** the PyMC model converges (R-hat, ESS checked), beats the linear baseline on held-out stints, and the XGBoost extrapolation failure is shown in a plot.

## Phase 4: Race simulator (built in layers)
- [ ] Single car vs the clock
- [ ] Strategy search over legal 1- and 2-stop plans
- [ ] Rivals with dirty air and overtake probability
- [ ] Safety cars
- [ ] Monte Carlo over the model's posterior

**Done when:** `src/f1strat/simulator.py` returns a distribution of finishing positions/times per strategy, with tests for each layer.

## Phase 5: Validation
- [ ] Backtest on 2022 races outside the case studies (excluding Austria; see Decisions log)
- [ ] Check interval calibration
- [ ] Loop back to Phase 3/4 if needed

**Done when:** backtest results and calibration plots are documented, and predicted intervals are calibrated well enough to support the case studies (or limitations are stated).

## Phase 6: Case studies
Refit using only data up to each decision lap (no future leakage).

Open issues for Phase 6:
- **Cleaning leaks future information into a cutoff refit.** The 107% outlier
  median and the pit loss are computed from the whole race. A refit at a
  decision lap must recompute both from laps up to that lap only.
- [ ] Hungary
- [ ] Silverstone
- [ ] Monaco (wet crossover; state limitations)

**Done when:** each case study has a verdict on whether the call was within the range of good options, backed by leakage-free simulations.

## Phase 7: Ship
- [ ] Charts
- [ ] README
- [ ] Clean modules
- [ ] Portfolio case study page

**Done when:** a reader can clone the repo, reproduce the results, and read the case study page.
