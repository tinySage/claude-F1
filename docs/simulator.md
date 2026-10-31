# Race simulator (Phase 4)

Status: **done** (race state, tyre inventory, single car, strategy search,
whole field with traffic and safety cars). Code:
- `src/f1strat/simulator.py`
- `src/f1strat/tyres.py` (tyre inventory)
- `src/f1strat/race_events.py` (passing and safety-car parameters)

Demo and figures: `notebooks/04_simulator.ipynb`, `reports/figures/spain_*.md`,
`reports/figures/tyre_inventory_*`.

## Inputs

- **Lap model:** the Phase 3 degradation model (`docs/model.md`), from the
  committed thinned posterior `models/deg_full_2022_thin.nc` (1,000 draws of
  the final full fit; rebuild with `uv run python -m f1strat.deg_model thin`).
- **Pit losses:** the circuit's observed 2022 green-flag stops ((in lap + out
  lap) − 2 × the driver's median clean lap). Each simulated stop resamples one
  observed stop, keeping the slow tail (Spain: 51 stops, median 22.7 s,
  slowest 37.6 s).
- **Tyre inventory:** each car's race-day sets, reconstructed from practice and
  qualifying (below).
- **Race-event parameters:** `data/processed/2022/sim_params.json`, built by
  `uv run python -m f1strat.race_events` (below).

## Race state

`RaceState` is the race at the end of lap `lap` (≥ 1; lap 1 is not modelled):
- per car: position, cumulative race time, compound, `TyreLife`, laps since
  the stop, stints so far, and dry compounds used;
- the track status of the lap just completed (green, SC or VSC), and laps
  already run under the current SC/VSC;
- optionally, each car's remaining sets.

`RaceState.from_laps` builds the actual 2022 state at any lap, including an SC
lap; `with_sets` adds the inventory.

## Tyre inventory (`tyres.py`)

Reconstructed from pre-race data only, so it is what the pit wall knew before
the race, not hindsight.

**Rules** (FIA 2022 Formula 1 Sporting Regulations, issue 7, Art. 30.2, 30.5):
- Allocation: 13 dry sets (2 hard, 3 medium, 8 soft); sprint weekends 12
  (2 hard, 4 medium, 6 soft).
- One soft (the Q3 tyre) may not be used or returned before Q3.
- One set of each mandatory race compound (assumed hard and medium) may not be
  returned before the race.
- Conventional weekends: 2 sets returned after each of FP1–FP3, and 1 soft
  after qualifying for cars that reached Q3. Sprint weekends follow
  Art. 30.5h.

**Method:**
- **Order:** every FP and qualifying stint is replayed in order. FastF1 gives
  each stint's compound, whether it started on a new set, and `TyreLife`
  (laps a set has run, across sessions).
- **Matching sets:** a used set is matched to the set whose laps equal the
  stint's first `TyreLife` − 1.
- **Stand-ins:** FP1 stand-ins (e.g. Vips in Pérez's car) are charged to the
  race driver they replaced (Art. 30.5d).

**Assumptions** (the rules do not say which sets go back):
- the most-used eligible set is returned first;
- when a car has too few used sets, unused **softs** go back first.
  Validation chose this order: softs first gives 40 mismatches over the four
  races, against 56 (hards first) and 63 (mediums first).

**Validation:** every set a car fitted in the race should exist in its
inventory (race stints are consumed in order, and a set used earlier in the
race goes back into the pool). Mismatches are reported, not patched:

| Race | Dry race stints | Mismatches | of which: TyreLife missing in race data | off by one lap |
|---|---|---|---|---|
| Spain | 73 | 6 | 0 | 4 |
| Hungary | 62 | 8 | 0 | 3 |
| Britain | 67 | 1 | 0 | 1 |
| Monaco | 39 | 15 | 11 | 0 |

- **One-lap differences:** most mismatches are sets whose race `TyreLife` is
  one lap off the qualifying count. FastF1's lap count for some runs is off by
  one.
- **Inconsistent new-set flags in FastF1:** e.g. Stroll in Spain FP3, two
  "new" mediums that look like one set.
- **New softs missing in Hungary:** a few new softs fitted in the race were
  returned by the "softs first" assumption.
- **Monaco:** 11 of 15 are hard sets fitted at the wet-to-dry crossover, which
  FastF1 records with no `TyreLife` and flagged used. Monaco's case study
  handles this in Phase 6.
- Full list: `reports/figures/tyre_inventory_mismatches.csv`.

## One car's laps

Simulation `i` uses one posterior draw (shared by every car in a field
simulation). For lap `L` on compound `c` at tyre age `a`:

```
lap = race_base + driver_offset + opening·[L in 2,3] + trend·(L − L0) + noise
    + compound_offset[c] + team_compound[c]
    + (deg[c] + team_deg[c]) · a/10 + post_stop[c]·[1st/2nd lap after a stop]
    + stint effect
```

- **Noise and pit loss:** Student-t lap noise; pit loss is added to the in
  lap. A stop at the end of lap `p` puts the new set on lap `p+1` (TyreLife 1
  for a new set).
- **No dirty air:** `dirty_air_penalty` is not used, because following is
  modelled explicitly in the field layer. Rain is off.
- **Stint effects: one draw per stint, never per lap.**
  - The stint in progress uses its fitted effect when the fit contains it,
    else a draw.
  - Future non-final stints draw `sigma_stint · z_j`.
  - The final stint draws `mu_final + final_race[r] + w · sigma_stint_final · z`.
- **Widening factor `w`** (default 1.188, Phase 3; `widening=1.0` turns it
  off) scales the drawn final-stint spread. If the fit has no final-stint
  data for the race (a Phase 6 cutoff fit), `final_race[r]` is a prior draw,
  widened too.

## Strategy search

- **Legal plans:** 0 to `max_stops` (default 3) more stops at any lap (after
  the state lap, before the last), and stints of at least `min_stint` laps.
- **Sets:**
  - from the car's inventory when known, each set fitted at most once;
  - otherwise `allowed_sets`, or every compound new and used without limit.
- **Two-compound rule:** the whole race uses at least two dry compounds.
- **Scoring:** every legal plan is scored on its **exact** mean race time
  over the simulations, single car, vectorised by stint. A plan's total is the
  plan-independent laps plus closed-form stint costs, and a test checks this
  equals the lap-by-lap simulation.
- **Stratified shortlist:** the best `per_stratum` (default 3) plans for every
  stop count and compound sequence. Once traffic is modelled, the no-traffic
  ranking would otherwise crowd out whole families: in Spain the top 50 were
  all three-stoppers, but with traffic a two-stopper is best.
- **Common random numbers:** `Draws` (per car) fixes the lap noise, the
  stint-in-progress draw, `z_j` for the `j`-th future stint, `z` for the final
  stint and the `j`-th stop's pit loss, so every plan meets identical luck.

## The whole field (Part B)

`build_field` draws everything plan-independent once. `simulate_field` runs
every car lap by lap, vectorised over simulations.

- **Following and passing (green laps).** Cars are processed in race order at
  the start of each lap.
  - **Held up:** a car that would finish the lap too close to the car
    directly ahead on track is held behind it, unless it passes. The gap is
    drawn from the 2022 gaps of cars within 1 s that did not pass
    (`held_gaps`, 5th–95th percentile 0.26–1.45 s; Phase 5). Phase 4 used the
    fixed median, `follow_gap_s` (0.68 s).
  - **Pass attempt:** only if it started the lap within 1 s of that car, with
    P = `sigmoid(a[circuit] + shift + b · pace gap)`, where `shift`
    (+0.986) is the Phase 5 calibration (`docs/validation.md`).
    - The pace gap is the free-air lap-time difference without this lap's
      noise, as in the fitted events.
    - A pass gains at most one place per lap.
    - Cars on an in or out lap don't battle (as in the fitted events).
- **Safety car.**
  - No passing.
  - The queue leader laps at `sc_lap_ratio` (1.49) × the field's free-air
    lap.
  - Behind it, gaps to the car ahead are kept on the first SC lap and then
    multiplied by `sc_close_ratio` (0.58) per SC lap, down to `sc_gap_s`
    (0.8 s). Fitted on 2022 SC periods (Phase 5, `docs/validation.md`).
  - Phase 4 instead bunched the field in one lap (each car at
    `sc_catchup_ratio`, 1.09, × its own lap); `sc_close_ratio = None`
    restores that.
  - Pit loss × `sc_pit_ratio` (0.72).
- **Virtual safety car.** No passing; every car laps at `vsc_lap_ratio` (1.33)
  × its free-air lap, so gaps are kept, not closed. Pit loss × `vsc_pit_ratio`
  (0.60).
- **SC/VSC schedule** (per simulation, plan-independent):
  - On green laps after lap 1, an SC starts with P = 1.35% per lap and a VSC
    with 1.35% (13 SCs and 13 VSCs over 961 green race laps in the 18 dry
    races).
  - A run of neutralised laps with no green lap between is one period, an SC
    if any lap is under SC (`neutral_runs`). Phase 4 counted a VSC that
    turned into an SC as both, which gave 14 SCs and 18 VSCs.
  - Lengths are drawn from the observed ones (SC 3–6 laps, VSC 1–4).
  - The lap after a period is green (the restart).
  - A state already under SC/VSC continues it for a remaining length drawn
    from the observed lengths longer than the laps already run.
- **Rival plans are inputs:** a fixed `Plan`, or weighted options picked once
  per simulation (the same pick for every focal plan). `rival_plan_options`
  builds options without hindsight: a rival's own top-3 single-car plans,
  weighted by `exp(−Δmean / 3 s)`. Phase 6 must not use rivals' actual future
  stops.
- **Common random numbers:** lap noise, stint and pit draws per car, the
  SC/VSC schedule, pass coin-flips and rival option picks are fixed per
  simulation.
- **Output** (`field_search`), for every focal plan:
  - the focal car's finishing-position distribution (mean, P(win),
    P(podium), 10th–90th percentile) and race time;
  - paired comparisons with a reference plan;
  - every car's position distribution (`field_positions`).

### Phase 6 additions (Silverstone; off by default)

Recorded before any Silverstone simulation (`docs/case_studies.md`). The
defaults leave Phases 4–5 and Hungary unchanged.
- **Session-clock state** (`RaceState.from_laps(timing="session")`): race
  time is the session clock at the end of the lap, not a sum of lap times.
  Silverstone 2022 needs it: lap 1 was red-flagged and the restart lap has no
  lap time for any car, so the sums put Hamilton 1.8 s *ahead* of Leclerc at
  lap 38 (really 6.1 s behind).
- **SC deployed during the next lap:** `track_status="sc"` with
  `neutral_laps=0` makes the next lap the period's first SC lap (gaps kept),
  and its remaining length is drawn from all observed SC lengths.
  `build_field(current_neutral_length=n)` fixes that length instead
  (robustness check).
- **Double-stack queue** (`SimParams.double_stack`). When teammates pit on
  the same lap, the car that arrives first (earlier arrival without its pit
  loss) is served first. The other waits `max(0, D − arrival gap)`. `D` (the
  first car's stationary time plus a margin) is set per track status (green,
  SC; VSC uses the SC value) and fitted on 2022 same-lap teammate stops
  (`race_events.double_stack_loss`).
- **SC-aware single-car search** (`search(pit_factor=...)`,
  `rival_plan_options(sc_aware=True)`). Each stop's loss is scaled by its
  lap's expected SC/VSC pit-loss factor over the field's simulations
  (`expected_pit_factor`), so a rival's own search can choose to stop under
  the SC. Without it every stop is costed at the green loss.

### Passing model (`race_events.py`)

**Events:** 3,645 cases in the 18 dry races where a car ended a lap within 1 s
of the car directly ahead on the same lap number. The outcome is whether it
was ahead a lap later; 563 passed (15%). Excluded: lap 1, either car's in or
out lap, and SC/VSC/red-flag laps.

**Model:** logistic regression on the model's free-air pace gap, with a pooled
slope (2.86 per s) and per-circuit intercepts shrunk toward the pooled one
(−2.75).
- Equal pace: P ≈ 6%. Attacker 1 s/lap faster: P ≈ 46%.
- Hardest circuits: Montreal and Zandvoort. Easiest: Abu Dhabi, Miami and
  Jeddah.
- Circuits without a fit (mixed-conditions races, e.g. Monaco) use the pooled
  intercept: **Phase 6 must decide Monaco's.**

**Phase 5 calibration** (`docs/validation.md`): replaying the 2022 races with
actual plans and SC/VSC, the simulator was 2–8% short of the real number of
on-track passes. It also had 1.5× too much close following. Two changes:
- held gaps are drawn from 2022 (no free parameter);
- one intercept `shift` (+0.986), fitted on even rounds after the SC queue
  fix.

Odd-round check: 311 simulated passes vs 318 real; close following is still
1.15–1.25× real.

### Late SC restarts and `final_race` (ROADMAP note)

`late_restart_rounds` lists races where an SC ended with racing resuming in the
last 20 laps:
- Bahrain, Miami, Canada, Britain, Zandvoort and São Paulo (plus Singapore, a
  mixed-conditions race outside the fit).
- Only Bahrain (−0.34 s) and São Paulo (−0.42 s) show the expected fast final
  stints in `final_race`; the other four are −0.04 to +0.23 s.
- The SC layer adds no post-restart speed-up, so restarts are not counted
  twice through it. For a race on the list, `build_field(final_race_prior=True)`
  draws `final_race` from the prior, so the simulation's own SCs are the only
  SC luck.
- Phase 6 cutoff fits draw the decision race's `final_race` from the prior
  anyway.

## Spain 2022 demo (Verstappen from lap 1)

**Inventory after lap 1:** 1 new hard, 2 new mediums and 2 used softs (plus
the used soft on the car).

**Single car** (no traffic): 764,000 legal plans scored, 87 Monte Carlo'd,
0.08 s.
- Best: three stops on two used softs and a new medium.
- Best two-stopper +2.6 s; best one-stopper +29 s.
- Verstappen's actual plan is +5.4 s from the best.

**Whole field** (traffic, SC/VSC at pooled rates, rivals on their own top-3
plans): 82 plans × 1,000 simulations in 43 s (0.52 s per plan).

| Verstappen's plan | Stops | Mean position | P(win) | P(podium) | Mean race time (s) |
|---|---|---|---|---|---|
| S → 22 M(new) → 44 S(used) | 2 | **1.99** | 0.29 | 0.93 | 6,058.7 |
| S → 13 M → 28 S(used) → 44 M (actual) | 3 | 2.23 | 0.24 | 0.88 | 6,062.7 |
| S → 16 M(new) → 33 S(used) → 50 S(used) | 3 | 2.38 | 0.29 | 0.87 | 6,058.4 |
| S → 33 M(new) | 1 | 3.58 | 0.06 | 0.56 | 6,080.7 |

- **Rerun with the Phase 5 parameters** (passing calibration, SC queue
  closing; `notebooks/04_simulator.ipynb` as executed in Phase 7): the best
  two-stopper 1.96, the best three-stopper 2.02 (still the fastest on time),
  the actual plan 2.20, the best one-stopper 3.55. The table above is the
  Phase 4 run; the conclusions below hold for both.
- **Traffic reverses the single-car ranking.** The best three-stopper is the
  fastest on time, but it finishes worse on average than the best
  two-stopper: the extra stop costs track position, and passing back is hard.
- **The two-stopper:** finishes ahead of Verstappen's actual plan in 21% of
  simulations and behind in 9%.
- **Leclerc:** wins 64% of simulations. He led the real race before retiring.

## Known limitations

- **Passing (Phase 5).** The Phase 4 figure of ~2.5× too many passes was
  mostly a counting mismatch (`docs/validation.md`). After calibration, pass
  counts match 2022, but cars still spend 1.15–1.25× as many laps within 1 s
  of the car ahead as in 2022.
- **SC closing (Phase 5).** Large gaps close at the 2022 rate. Gaps under
  ~5 s close to 0.8 s from the second SC lap, while in 2022 they stayed
  around 1.6 s for three laps. The ratio is fitted on 3 SC periods.
- **Final-stint widening.** The Phase 4 default is ×1.188. The Phase 5
  backtests are better calibrated with 1.0, so Phase 6 uses 1.0 and reports
  ×1.188 as a sensitivity check.
- **Stint order does not matter on its own.** The lap model has no fuel ×
  compound interaction, so a plan's tyre cost depends only on each stint's
  compound, set and length, not on when in the race the stint is run.
  - Two plans that swap stints of equal length tie exactly in every
    single-car simulation (e.g. "S → M → S → S" and "S → S → S → M").
  - In the field simulation, order matters only through traffic and safety
    cars, not through tyre performance with fuel load.
- **Tyre inventory:** the remaining mismatches above, and the assumption about
  which sets were returned.
- **SC/VSC rates are pooled over the season** (not per circuit). Restart laps
  are simulated as normal green laps.
- **No lapped-car or blue-flag effects; no random retirements** (backtests
  force the actual ones with `simulate_field(retire=...)`); cars don't change
  plans in reaction to an SC (plans are inputs).
- **Pit loss under SC/VSC** is a pooled ratio to the circuit's green-flag
  loss.
