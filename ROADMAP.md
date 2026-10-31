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
- **Held-out test sets (Phase 3):** `last_stint` and `long_tyre` (train on
  `TyreLife <= 15`, test on older tyres), fixed and reused for every model.
  - `last_stint` tests final stints of **odd-numbered rounds only**, for
    drivers with an earlier clean stint. Even rounds keep their final stints in
    training so the final-stint terms can be estimated. It was originally every
    driver's final stint; changed with the final-stint terms.
- **Stint-level terms (Phase 3):** `stint[d,r,s]`, added because held-out
  intervals under-covered (68–70%) and 37–50% of held-out error was a constant
  shift per stint.
  - Non-final stints: spread `sigma_stint`, centred within each driver's race.
  - Final stints: a pooled mean shift `mu_final` and their own spread
    `sigma_stint_final`, shared across races, because final stints varied about
    twice as much as others.
  - The simulator knows which stint is final from the strategy, so these terms
    can be used at prediction time.
- **No cliff term (Phase 3):** degradation is linear in `TyreLife` per compound
  and race.
  - Why: the linear baseline was the best model beyond 30 laps; the
    season-level quadratic was correlated with deg and trend and hurt mixing;
    and it could not represent track-specific cliffs anyway.
  - Linear degradation on very long stints is a stated limitation.
- **`warmup` renamed `post_stop` (Phase 3):** first two flying laps after a
  stop, net of tyre warm-up and drivers pushing on new tyres. Unconstrained;
  its negative sign is a finding.
- **Phase 3 done-when changed:** `long_tyre` has no coverage threshold.
  - Why: it trains only on `TyreLife <= 15`, but every real use (Phase 4
    simulations, Phase 6 cutoff fits) includes long stints from earlier races,
    so it is a stress test, not the use case.
  - Its error-by-age table and coverage are reported as limitations.
  - `last_stint` keeps the 75–85% coverage target.
- **Race-wide final-stint shift (Phase 3):** `final_race[r]`, pooled across
  races (`sigma_final_race`), drawn from its spread for a race with no
  final-stint data.
  - Non-centred, because the centred form made a funnel at races with no
    final-stint data.
  - Sampler for all fits: 5,000 draws, low-rank adaptation, target acceptance
    0.95.
- **`last_stint` coverage accepted below target (Phase 3):** 74.0% against
  75–85%, accepted under the agreed rule because all fits pass diagnostics.
  - Widening factor ×1.188 on the 80% interval half-widths reaches 80%.
    Phase 4 can apply it.
  - The likely cause is `sigma_final_race` being estimated from the 9 even
    rounds only in that split (0.12 s vs 0.24 s in the full fit).
- **Thinned posterior (Phase 4):** the simulator reads
  `models/deg_full_2022_thin.nc` (1,000 draws, 250 per chain, of the final
  full fit; 6.7 MB, committed), so new sessions skip the ~40 min refit.
- **Tyre inventory (Phase 4):** reconstructed from FP1–FP3 and qualifying
  (FIA 2022 Sporting Regulations Art. 30.2, 30.5).
  - The rules don't say which sets are returned. Assumed: the most-used
    eligible set first; when a car has too few used sets, unused softs first.
    Softs first was chosen by validation: 40 race-stint mismatches over Spain,
    Hungary, Britain and Monaco, against 56 (hards first) and 63 (mediums
    first).
  - FP1 stand-ins are charged to the driver they replaced.
  - Remaining mismatches are reported in `reports/figures/`, not patched.
- **Stratified shortlist (Phase 4):** the best 3 plans per stop count and
  compound sequence are Monte Carlo'd, so traffic can promote a strategy
  family the no-traffic ranking would have dropped.
- **Late SC restarts (Phase 4):** races where an SC ended in the last 20 laps
  can draw `final_race` from the prior (`final_race_prior=True`), so the
  simulator's own SCs are the only SC luck. Only Bahrain and São Paulo show the
  restart sprint in `final_race`.
- **Backtest setup (Phase 5):** `docs/validation.md`.
  - **Races:** the 15 dry races outside the case studies and Austria.
  - **Start:** the actual state after lap 1 (lap 1 is not modelled).
  - **Plans and retirements:** actual pit laps, compounds and retirements.
  - **Pace:** the full posterior (the simulator layers are under test; lap
    times were tested out of sample in Phase 3). Stint effects are drawn.
  - **Modes:** replay (actual SC/VSC forced) and random (SC/VSC drawn).
- **Passing calibration (Phase 5):**
  - Held gaps are drawn from the 2022 gaps of cars that did not pass (no free
    parameter).
  - One intercept shift (+0.674), fitted so that replayed even-round races
    match the actual number of on-track passes; checked on odd rounds (303 vs
    318).
  - The Phase 4 "2.5× too many passes" was mostly a counting mismatch.
  - Shift refitted to +0.986 after the SC queue fix (was +0.674); odd-round
    check 311 vs 318.
- **SC queue closing (Phase 5, approved):** gaps are kept on the first SC lap,
  then multiplied by a fixed ratio per SC lap, down to 0.8 s.
  - The ratio is 0.58, fitted on even-round SC periods (squared error in
    seconds, so large gaps count most; a log-error fit gives 0.64 and closes
    large gaps too slowly) and checked on odd rounds.
  - Replaces Phase 4's one-lap bunching.
- **SC/VSC counting (Phase 5):** a VSC that turns into an SC counts as one SC
  period, not as both. 13 SCs and 13 VSCs (were 14 and 18).
- **Practice long runs in cutoff fits (Phase 5 Part C):**
  - The decision race's FP1–FP3 long runs (6+ consecutive clean laps on one
    set) join the cutoff refit (`docs/model.md`). They share that race's
    deg, compound, team and driver terms; each run has its own offset, and
    each session one fuel slope with a prior on the race trend.
  - Practice is before the race, so this adds no leakage.
- **Leakage-free simulator parameters for cutoff fits (Phase 5 Part C):**
  - Passing (slope, intercepts, held gaps, shift), SC/VSC rates and the SC
    closing ratio come from the dry rounds before the decision race.
  - The decision circuit's pass intercept comes from its previous running,
    or the pooled intercept if that cannot be downloaded (Canada 2019 needs
    an F1 account).
- **Divergence rule for result fits (Phase 5):** fits used for verdicts must
  have 0 divergences. `rehearsal.clean_fit` reruns with a new seed, then
  target acceptance 0.99.
- **Phase 6 verdict rule (committed before any case-study simulation):**
  `docs/case_studies.md`.
  - **Metric:** Leclerc's 2022 points.
  - **Best plan:** chosen on 1,000 simulations; the paired loss is measured on
    a fresh 2,000.
  - **Bands:** within ≤ 1 point; outside > 3 points with the actual plan
    level or ahead in < 25% of simulations; marginal otherwise.
  - **Robust** only if the band holds under widening ×1.188, Ferrari hard
    wear at its 10th/90th percentile, and the SC rate ×0.5 and ×2.
  - The hindsight replay is reported separately.
- **Circuit pass difficulty for Silverstone and Monaco (Phase 6, recorded
  before simulating them):** the 2022 pooled intercept from the rounds before
  the decision race, shifted by (the circuit's 2021 intercept − the 2021
  season average). Both 2021 terms come from one model over every dry 2021
  race.
  - **Why:** 2022 cars follow more closely, so absolute 2021 values
    overstate difficulty (Hungary 2021: −4.60, against −2.78 pooled for
    2022).
  - Hungary's committed verdicts keep the previous-running intercept; the
    relative method is a post-hoc sensitivity there (`docs/case_studies.md`).
- **Silverstone additions (Phase 6, committed before any Silverstone
  simulation):** `docs/case_studies.md`, "Silverstone 2022".
  - Decision at Leclerc's lap 39 (SC deployed during it); cutoff lap 38; the
    state starts under the SC; its length from the 6 pre-race 2022 SCs.
  - Race time from the session clock (lap-1 red flag, untimed restart lap).
  - Double-stack queue: the second car waits `max(0, D − arrival gap)`, `D`
    fitted on 2022 same-lap teammate stops (green 7.15 s, SC 6.45 s).
  - Rivals' and Leclerc's searches cost stops at the expected SC/VSC pit loss.
  - Pass intercept: relative method, updated with the race's own passes
    before the cutoff (also a post-hoc row for Hungary).
  - Primary verdict with Sainz's actual stop fixed; Ferrari's combined points
    per joint option reported alongside; added checks: SC length p10/p90,
    double-stack loss ×0.5/×2, Leclerc's new hard removed.
- **Final-stint widening in Phase 6: 1.0 (Phase 5).** The replay backtests
  are better calibrated without the ×1.188 widening (fair position coverage
  0.57 / 0.85 vs 0.60 / 0.87; gaps 0.54 / 0.82 vs 0.55 / 0.84). Phase 6 uses
  1.0 and reports ×1.188 as a sensitivity check.
- **Hungary confounding (Phase 3 → Phase 6):** Leclerc's stint 3 is Ferrari's
  only hard stint at Hungary. The data identify the sum of
  `team_compound[Ferrari, hard, Hungary]` and his stint effect, not the split,
  so Phase 6 reports the sum.

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
- [x] Bayesian hierarchical model in PyMC: car pace offsets, per-compound linear degradation, car-specific tyre terms, stint and final-stint terms (incl. race-wide `final_race`), fuel + track trend (`docs/model.md`, "Results: final fit")
- [x] XGBoost fit, used only to show it fails to extrapolate on long stints (`reports/figures/xgb_vs_bayes_long_tyre.png`)

Result:
- All three fits pass diagnostics.
- The Bayesian model beats the baseline on `last_stint` (MAE 0.598 vs 0.874 s)
  and matches it on `long_tyre` (0.505 vs 0.524 s).
- `last_stint` 80% coverage is 74.0%, accepted as a limitation with a widening
  factor of ×1.188 (see Decisions log).

**Done when:** every PyMC fit has R-hat < 1.01 on every parameter, zero divergences and ESS ≥ 400; the model matches or beats the baseline on both test sets; `last_stint` 80% coverage is in 75–85% (`long_tyre` coverage and error by tyre age are reported as limitations, no threshold); and the XGBoost comparison plot exists.

## Phase 4: Race simulator (built in layers)
- [x] Single car vs the clock (`RaceState`, one stint draw per stint, Student-t lap noise, pit losses resampled from observed stops, ×1.188 final-stint widening on by default; `docs/simulator.md`)
- [x] Strategy search over every legal remaining plan (0–3 more stops, any lap, any dry compound, new or used sets, two-compound rule), common random numbers; Spain 2022 demo in `notebooks/04_simulator.ipynb`
- [x] Rivals: whole field lap by lap, held up within the following gap unless passing; pass probability by pace gap and circuit, fitted on 2022 passes; rival plans are inputs (fixed or weighted options)
- [x] Safety cars: SC/VSC rates and lengths, slower laps, SC bunching, pit loss under SC/VSC (`data/processed/2022/sim_params.json`); start mid-race and under SC
- [x] Monte Carlo over the model's posterior with common random numbers; Spain 2022 rerun with traffic (`notebooks/04_simulator.ipynb`, `docs/simulator.md`)

Notes from Phase 3:
- **Check before reusing `final_race` in the SC layer.** The largest race-wide
  final-stint shifts (São Paulo −0.41 s, Bahrain −0.34 s) look like late
  safety-car restarts followed by sprint laps on fresh tyres. If so,
  `final_race` partly encodes SC timing, and the SC layer must not count it
  twice.

Result:
- From any `RaceState`, a strategy search returns finishing-position and
  race-time distributions for the whole field.
- Speed: ~0.5 s per plan per 1,000 simulations (50 plans ≈ 25 s).
- Calibration item for Phase 5: simulated passes per race are ~2.5× the 2022
  rate (see `docs/simulator.md`, Known limitations).

**Done when:** `src/f1strat/simulator.py` returns a distribution of finishing positions/times per strategy, with tests for each layer.

## Phase 5: Validation
- [x] Backtest on 2022 races outside the case studies (excluding Austria; see Decisions log): replay and random modes, `docs/validation.md`
- [x] Check interval calibration (replay, widening 1.0: fair 50%/80% position coverage 0.57/0.85, gaps 0.54/0.82; Spearman 0.96 vs 0.76 naive; random-mode gaps too wide)
- [x] Loop back to Phase 3/4 if needed: SC queue closing, VSC counting, passing calibration, widening choice
- [x] Phase 6 dress rehearsal: Spain and Canada cut at mid-race, leakage-free cleaning, practice long runs and parameters (`docs/validation.md`, Part B)

Result:
- **Backtests** (15 races, replay, widening 1.0): fair 50%/80% position
  coverage 0.57/0.85, gaps 0.54/0.82, Spearman 0.96 (naive lap-1 order
  0.76).
- **Passing** calibrated (odd-round check 2% low). SC queue closes
  gradually; VSC counting fixed.
- **Rehearsals:** with practice data and leakage-free parameters, Spain
  ranks the actual call 0.05 places from the best plan. Canada's pick moves
  to a 30-lap soft stint the race cannot check (see Phase 6 items).

Watch items from Phase 4:
- Passing: simulated passes per race are ~2.5× the 2022 rate. Backtest pass
  counts and position changes per race before relying on position verdicts.
  **Done:** mostly a counting mismatch; calibrated (see Decisions log).

Watch items from Phase 3 (if these races backtest badly, look here first):
- Negative field hard degradation: Australia, Saudi Arabia, Miami.
- Hard degrading faster than soft: São Paulo.

**Done when:** backtest results and calibration plots are documented, and predicted intervals are calibrated well enough to support the case studies (or limitations are stated).

## Phase 6: Case studies
Refit using only data up to each decision lap (no future leakage).

Open issues for Phase 6:
- **Cleaning leaks future information into a cutoff refit. Resolved in
  Phase 5:** `deg_model.cutoff_inputs` recomputes cleaning from the laps
  known at the cutoff.
  - The 107% outlier median and the pit stops (and so the pit losses) come
    from laps up to the cutoff only.
  - No stint of the decision race is labelled final (on truncated data every
    current stint would be).
  - Tested by changing every later lap and checking nothing moves.
- **Passing and SC/VSC parameters from the whole season. Resolved in
  Phase 5:** `rehearsal.leakage_free_params` estimates them from earlier
  rounds, and the circuit's pass difficulty from its previous running.
  Phase 6 must confirm the 2021 Hungary and Britain races download.
- **A compound not yet raced at the decision race takes its degradation from
  the season pool (Phase 5 rehearsal).**
  - Spain cut at lap 33 had 2 clean hard laps, so Spain's hard degradation
    came out at 0.039 s/lap, against 0.123 s/lap in the full fit. The search
    then sent Verstappen and most rivals onto hards, which almost nobody used.
  - Hungary is the same situation (Leclerc onto hards).
  - Partly resolved in Phase 5 Part C: practice long runs bring Spain's
    hard to 0.116 s/lap (full fit 0.123).
  - **But practice runs are short.** In Canada, ~12-lap soft runs made a
    30-lap soft stint look best, where no team ran the soft (linear
    degradation extrapolated).
  - For Hungary, Phase 6 must report the practice run lengths on the hard
    against the stint the verdict needs, and show sensitivity to the hard's
    degradation.
- **Pit losses at the cutoff can rest on few stops** (Canada lap 35: 3, the
  rest were under VSC). Report the count; consider a fallback (previous
  running or a prior).
- **SC rates from early rounds are noisy** (Spain: 2.7% per lap from 4
  races). Show verdict sensitivity to the SC rate.
- **Cutoff by lap number:** a lapped car's lap `L` ends a little after the
  leader's lap `L`, so it is a few seconds "in the future" at the cutoff.
  Negligible, but state it.
- **Monaco has no pass-difficulty estimate.** It currently gets the season
  average (pooled) intercept, which is wrong for Monaco, where passing is
  nearly impossible. Phase 6 sets it with the relative method (Decisions log),
  from Monaco 2021.
- **Hungary inventory mismatches.** Hungary's reconstructed inventory had 8
  mismatches, including new softs that the "softs first" return assumption
  sent back. Phase 6 must print Ferrari's reconstructed Hungary sets and
  check them against Leclerc's actual fittings.
- **SC timing verdicts (from Phase 5).** The SC queue now closes gradually
  (fitted ratio 0.58 per lap), but small gaps close too fast and the ratio
  rests on 3 SC periods. Silverstone's verdict depends on this
  (`docs/validation.md`); state it as a limitation.
- **Use widening 1.0** and report ×1.188 as a sensitivity check.
- **Hindsight replays use fitted effects for stints that happened**
  (`CarModel.known_stints`; fixed during Hungary). Silverstone and Monaco
  inherit this.
- **Several decisions per race:** judge each call at its own cutoff (Hungary
  had two). A later stop is never charged to an earlier call.
- [x] Hungary (`docs/case_studies.md`), two decisions:
  - **Hards at lap 39: within range, robust.** It is the model's best plan
    at lap 38.
  - **Used softs at lap 54: outside range, robust under the committed
    checks.** Hards to the flag was better by 5.1 points with lap-53
    information.
    - Post-hoc pass-difficulty sensitivity: the switch was the worse option
      in every setting, costing 3.5–5.1 points (outside range with the
      relative method; marginal with average passing and with the in-race
      update).
    - Disclosed limitation: 2021 pass difficulty likely overstates 2022.
  - The points were lost at lap 54. About 4 more points lost in hindsight
    at lap 38 came from what the pit wall could not know (Verstappen's stop
    phase and pace).
- [x] Silverstone (`docs/case_studies.md`): **inconclusive (the model can't
  reproduce the restart).**
  - v1 and v2 put the call within range, but neither model reproduces the
    race in hindsight (Leclerc P1 in 95% / 93% of replays; he finished P4).
  - Diagnosis: the miss sits in the three laps after the restart (+0.87 s/lap
    Leclerc − Sainz; 9 passes vs 1–3 simulated).
  - The restart layer, estimated from rounds before Silverstone and
    validated on later rounds, removes the field-wide restart bias. But its
    compound and age effects and its pass multiplier do not generalise, and
    it cannot produce this restart.
  - v1 on record:
  - It is the model's best plan at lap 38 (23.3 points against 21.4 for new
    softs at 39). Within range under all 13 checks (committed checks plus SC
    length 4/6 laps, double-stack loss ×0.5/×2, and the new hard removed).
    The best call-family plan is no further stop, not a lap-40 stop.
  - Ferrari's combined points: the actual split was the best joint option,
    level with leaving both out; the double stack was the worst (−3.0).
  - **Fragile:** staying out stops being best if Leclerc's hards were
    0.25–0.5 s/lap worse than modelled. The real extra deficit was about
    0.3–0.5 s/lap.
  - **Reality check:** 9 passes in the first three laps after the restart,
    against 3.1 forecast. Leclerc lost P1 on the restart lap and finished
    P4. The simulator has no restart effects (cold tyres, restart passing),
    so the hindsight replay does not reproduce the race (P1 in 95%).
- **Restart laps are normal green laps** (found at Silverstone): no
  cold-tyre penalty and green-flag pass rates. Silverstone had 9 passes in
  the first three restart laps against 3.1 simulated. State it for any
  verdict that turns on a restart.
- [x] Monaco (`docs/case_studies.md`; track-position accounting,
  `f1strat.monaco`): **D1 (inters at 18) and D2 (double stack at 21): not
  assessable at the cutoff.**
  - The public timing data had no usable slick pace at either decision (no
    hard cell identified by 2+ cars on 2+ laps).
  - Exploratory lap-20 reading (one lap, two cars): slicks still 3.8 s/lap
    slower than inters. No verdict drawn.
  - Hindsight outcomes, not verdicts: skipping inters, or staying out to 22,
    would likely have kept Leclerc P1 (22–25 points against 12). The queue
    cost at most one place, within the noise.
  - Ferrari's pit wall had more information than this dataset.

**Done when:** each case study has a verdict on whether the call was within the range of good options, backed by leakage-free simulations, or a documented reason none is supported.

## Phase 7: Ship
- [ ] Charts
- [ ] README
- [ ] Clean modules
- [ ] Portfolio case study page

**Done when:** a reader can clone the repo, reproduce the results, and read the case study page.
