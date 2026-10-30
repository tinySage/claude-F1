# Roadmap

Question for each case study: using only what the Ferrari pit wall knew at the
decision lap, was the call within the range of good options?

Case studies (2022):
- **Hungary**: Leclerc onto hards
- **Silverstone**: Leclerc left out under the late safety car
- **Monaco**: the double-stack

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
- [ ] Remove lap 1, in/out laps, SC/VSC/yellow-flag laps and deleted laps
- [ ] Flag dirty-air laps (keep them, with a flag)
- [ ] Estimate pit lane loss per circuit
- [ ] EDA notebook with plots

**Done when:** `src/f1strat/clean.py` produces a tested clean dataset with a dirty-air flag, and a per-circuit pit loss table exists.

## Phase 3: Tyre degradation model
- [ ] Linear baseline
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
- [ ] Backtest on 2022 races outside the case studies
- [ ] Check interval calibration
- [ ] Loop back to Phase 3/4 if needed

**Done when:** backtest results and calibration plots are documented, and predicted intervals are calibrated well enough to support the case studies (or limitations are stated).

## Phase 6: Case studies
Refit using only data up to each decision lap (no future leakage).
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
