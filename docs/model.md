# Tyre degradation model (Phase 3)

Status: **spec approved** (with car-specific tyre terms, a separate opening-laps
term and scaled inputs). Code: `src/f1strat/deg_model.py`. Results are in the
"Results" section at the end once fitted.

## What the model is for

Predict a car's clean lap time from who is driving, which team's car, which
tyre, how old the tyre is, and where in the race it is, with honest
uncertainty. The simulator (Phase 4) draws lap times from this model, and the
case studies (Phase 6) refit it on data up to a decision lap.

## Data

- **Laps:** 2022 laps with `exclude_reason` null (see `data/processed/README.md`
  and the ROADMAP Decisions log), with a recorded `TyreLife`. That is 15,604
  laps from 18 dry races. The four mixed-conditions races are excluded, and 73
  Austria laps with no tyre age are dropped (not filled in).
- **Units:** lap time `y` in seconds; `TyreLife` in laps (it includes laps run on
  that set before the race, so a used set starts at 3+, not 1); `race_lap` is
  the race lap number.

Indices:

| Symbol | Meaning | Count |
|---|---|---|
| `r` | race (round) | 18 |
| `d` | driver | 22 (some drivers race only a few rounds) |
| `t` | team (the car) | 10, all present at every race |
| `c` | compound: soft, medium, hard | 3 |

## Model

For lap `i` by driver `d` of team `t` in race `r` on compound `c`:

```
y_i ~ StudentT(nu, mu_i, sigma[r])

mu_i = race_base[r]
     + driver_offset[d,r]
     + compound_offset[c,r] + team_compound[t,c,r]
     + (deg[c,r] + team_deg[t,c]) * TyreLife_i
     + cliff[c] * TyreLife_i^2
     + warmup[c] * first_two_flying_i
     + opening   * opening_laps_i
     + trend[r]  * race_lap_i
     + dirty_air_penalty * dirty_air_i
     + rain_effect       * rainfall_i
```

### Every term in plain words

| Term | What it means | How it is estimated |
|---|---|---|
| `race_base[r]` | Lap time of the field-average car on a new medium, mid-race, in clean air, in the dry, at race `r`. Sets each track's level. | One free value per race. |
| `driver_offset[d,r]` | How much faster or slower this car and driver are than the field average at this race, on the medium. Mostly the car. | Partial pooling toward `driver[d]`, the driver's season-average offset. |
| `compound_offset[c,r]` | How much faster a new soft or hard is than a new medium **for the field average** at this race. | Medium is the reference (0). Soft and hard pool toward a season mean per compound. |
| `team_compound[t,c,r]` | How much better or worse team `t`'s car gets on soft or hard, relative to the medium, compared with the field at this race. "Ferrari's hards are worse than everyone else's here." | Medium is the reference (0). Zero-sum over teams within each race and compound. Pooled toward `team_compound_season[t,c]`, the team's season tendency, which pools toward 0. |
| `deg[c,r]` | Seconds lost per extra lap of tyre age, for compound `c`, for the field average at race `r`. Linear degradation. | Partial pooling toward a season deg rate per compound. |
| `team_deg[t,c]` | How much faster or slower team `t` wears compound `c` than the field, every race. "Ferrari is harder on its tyres." | Season level, all three compounds, zero-sum over teams. |
| `cliff[c]` | Extra loss that grows with the square of tyre age. Positive means wear accelerates on long stints. | One season-level value per compound. Smooth curvature, not a sudden cliff. |
| `warmup[c]` | Extra time on the first two flying laps after a **pit stop**, while new tyres come up to temperature. | One value per compound. `first_two_flying` is 1 when laps since the stop is 1 or 2, **only for stints that begin with a pit stop** (not the race start). Counted from the stop, not from `TyreLife`, because a used set starts at `TyreLife` 3+. |
| `opening` | Extra time on race laps 2 and 3: the field is bunched and DRS is off. Not tyre warm-up. | One season-level value. `opening_laps` is 1 on race laps 2–3. |
| `trend[r]` | Seconds gained or lost per race lap from everything that changes with race lap: fuel burning off and the track rubbering in. Expected negative. | Partial pooling across races. Fuel and track evolution both move one-for-one with race lap, so they are not split. |
| `dirty_air_penalty` | Time lost when within 1 s of the car ahead. | One season-level value. |
| `rain_effect` | Time lost on laps with `Rainfall` reported while still on slicks. | One season-level value. In practice only Hungary informs it (see below). |
| `sigma[r]`, `nu` | Lap-to-lap noise at race `r`, and how heavy its tails are. | Student-t, so a few leftover odd laps don't drag the fit. |

### Scaled inputs inside the model

The sampler sees two transformed inputs, so it is not fighting large,
correlated values. Every effect is reported back in seconds and s/lap.

- **Tyre age:** `a = TyreLife / 10` (tyre age in units of 10 laps). Inside the
  model, deg is in s per 10 laps and cliff in s per (10 laps)². Reported as
  deg_s_per_lap = deg / 10 and cliff_s_per_lap² = cliff / 100. Tyre age is
  scaled but not centred, so a value of 0 still means a new tyre.
- **Race lap:** `l = race_lap − L0`, where `L0` is the mean race lap of the
  training data (30.1 for the full data), stored with the fit. `trend` is
  still in s/lap. `race_base[r]` is then the lap time at race lap `L0`, not at
  lap 0.

### Hierarchy (partial pooling)

All pooled terms use a non-centred parametrisation (`value = mean + scale * z`,
`z ~ Normal(0, 1)`), which samples better when a group has little data.

```
driver[d]                  ~ ZeroSumNormal(sigma_driver)               over drivers
driver_raw[d,r]             = driver[d] + sigma_driver_race * z[d,r]
driver_offset[d,r]          = driver_raw[d,r] − mean over drivers in race r

compound_offset[medium,r]   = 0
compound_offset[c,r]        = mu_compound[c] + tau_compound[c] * z[c,r]      c in {soft, hard}

team_compound[t,medium,r]   = 0
team_compound_season[t,c]   ~ ZeroSumNormal(sigma_tc_season)            over teams, c in {soft, hard}
team_compound_raw[t,c,r]     = team_compound_season[t,c] + sigma_tc_race * z[t,c,r]
team_compound[t,c,r]         = team_compound_raw[t,c,r] − mean over teams of team_compound_raw[.,c,r]

deg[c,r]                    = mu_deg[c] + tau_deg[c] * z[c,r]
team_deg[t,c]               ~ ZeroSumNormal(sigma_team_deg)             over teams, all 3 compounds
trend[r]                    = mu_trend + tau_trend * z[r]
```

### Priors

Weakly informative. Shown in reporting units (s, s/lap). Degradation is **not**
constrained to be positive, so a negative or inverted result can be seen and
flagged, not hidden.

| Parameter | Prior (reporting units) | Inside the model | Reasoning |
|---|---|---|---|
| `race_base[r]` | Normal(median training lap at r, 3) s | same | Centred on the track's typical lap. |
| `sigma_driver` | HalfNormal(1.5) s | same | Car/driver spread is up to ~2–3 s/lap. |
| `sigma_driver_race` | HalfNormal(0.5) s | same | Race-to-race variation in relative pace. |
| `mu_compound[soft, hard]` | Normal(0, 1) s | same | No sign assumed. |
| `tau_compound[c]` | HalfNormal(0.5) s | same | |
| `sigma_tc_season` | HalfNormal(0.5) s | same | Team-specific compound offsets of a few tenths. |
| `sigma_tc_race` | HalfNormal(0.3) s | same | Race-to-race variation around the team's tendency. |
| `mu_deg[c]` | Normal(0.05, 0.1) s/lap | Normal(0.5, 1.0) s/10 laps | Typical deg is a few hundredths to ~0.1 s/lap. |
| `tau_deg[c]` | HalfNormal(0.05) s/lap | HalfNormal(0.5) | Track-to-track variation. |
| `sigma_team_deg` | HalfNormal(0.03) s/lap | HalfNormal(0.3) | Team differences in wear of a few hundredths/lap. |
| `cliff[c]` | Normal(0, 0.002) s/lap² | Normal(0, 0.2) s/(10 laps)² | ±2 sd gives ±3.6 s extra at 30 laps. |
| `warmup[c]` | Normal(0, 1) s | same | |
| `opening` | Normal(0, 1) s | same | |
| `mu_trend` | Normal(−0.05, 0.05) s/lap | same | Fuel alone is roughly −0.03 to −0.06 s/lap. |
| `tau_trend` | HalfNormal(0.03) s/lap | same | |
| `dirty_air_penalty` | Normal(0, 1) s | same | |
| `rain_effect` | Normal(0, 1) s | same | |
| `sigma[r]` | HalfNormal(1) s | same | Clean-lap noise is typically 0.3–0.8 s. |
| `nu` | Gamma(2, 0.1) | same | Mean 20; lets the data choose heavy tails. |

## Identifiability: how each confusion is handled

1. **Level (who absorbs the average?).** `race_base`, the driver offsets and
   the compound offsets could all shift together. Fixed by:
   - medium is the compound reference, for the field and for every team;
   - driver offsets sum to zero within each race;
   - season driver effects `driver[d]` sum to zero across drivers.

   So `race_base[r]` is the field-average car on a medium.
2. **Field compound vs team compound.** `team_compound[.,c,r]` sums to zero over
   teams within each race and compound, so `compound_offset[c,r]` keeps its
   meaning (the field average), and `team_compound` is only each team's
   deviation from it. The same holds for `team_deg` vs `deg`: `team_deg` sums
   to zero over teams for each compound.
3. **Driver pace vs team compound, and a team that ran no medium.** A team's
   medium laps fix its drivers' offsets, and its soft/hard laps then fix its
   `team_compound`.
   - With no medium at a race, only the sum `driver_offset + team_compound`
     is seen in that race's data: the split between them is not identified
     from that race alone.
   - It is resolved by pooling. `driver_offset[d,r]` is pulled toward the
     driver's season pace, and `team_compound[t,c,r]` toward the team's season
     tendency. The posterior splits the sum according to those two pulls and
     shows the extra uncertainty.
   - This happens three times in 2022: Aston Martin in Australia (hard only),
     AlphaTauri at Silverstone and Alpine at Zandvoort (soft and hard). The
     prediction of that team's laps at that race is unaffected (it depends on
     the sum); only the split is uncertain.
4. **Team compound in a cutoff refit.** Before a team has run a compound at a
   race (e.g. Ferrari before Leclerc's first hard lap in Hungary),
   `team_compound[t,c,r]` has no direct data and falls back to
   `team_compound_season[t,c]` with its uncertainty. That is what the pit
   wall could know: the team's form on that compound at earlier races.
   - In the 2022 data, Ferrari's only Hungary hard laps are Leclerc's 13 clean
     laps.
5. **Fuel vs degradation.** These are separable **only because `TyreLife`
   resets to a low value at each stop while `race_lap` keeps counting.**
   - Within one stint the two rise together one-for-one, so a race where most
     cars ran a single stint carries almost no information to separate them.
   - Because `deg` is per compound, a stop helps only if the **same compound**
     is seen at different (race lap, tyre age) pairs. That happens with
     staggered stop laps, two stints on one compound, or used sets starting at
     different ages.
   - If every car ran medium then hard and stopped on the same lap, `race_lap`
     would be an exact linear combination of the tyre-age columns, and the
     split would be unidentified. The synthetic baseline test staggers its
     stops for exactly this reason.
   - The baseline shows the failure on real data: in the `last_stint` split at
     Saudi Arabia, 15 of 17 drivers have only one training stint (correlation
     0.90). OLS lands on deg −1.1 s/lap and trend +1.1 s/lap, which cancel in
     training and blow up on the final stint (MAE 6.9 s).
   - Partial pooling of `deg` and `trend` across races is the fix: a
     poorly-identified race borrows its split from the other 17.
6. **Linear deg vs cliff.** Both rise with tyre age. `cliff` is season-level,
   so it is pinned down by all long stints across the season, not a few laps at
   one race.
7. **Warm-up vs degradation vs race start.** Warm-up counts laps since a pit
   stop, while `deg` uses `TyreLife`, so on a used set the two differ.
   - Stints that start at the race start get no warm-up flag; race laps 2–3
     get `opening` instead.
   - This keeps the bunched, DRS-off opening laps from being read as tyre
     warm-up (or as negative degradation on the first stint).
8. **Fuel vs track evolution.** Not separable (both move with race lap).
   Accepted: `trend` is their sum, and the simulator only needs the sum.
9. **Rain.** All 285 clean laps with `Rainfall` are from Hungary, in its final
   third.
   - `rain_effect` is therefore really a Hungary late-race parameter and
     overlaps Hungary's `trend` and late-stint deg. Expect a wide interval.
   - This matters for the Hungary case study: the model can attribute late
     slowness to rain, to the hard tyre, or to Ferrari's hard specifically,
     and the posterior shows how well it can tell them apart.
10. **Dirty air.** The flag is set when a car is within 1 s of the car ahead,
    which also happens when it is simply being held up. The penalty includes
    both effects. That is what the simulator needs, but it is not a pure
    aerodynamic estimate.

## Prediction

- **Point prediction:** the posterior mean of `mu` (the Student-t mean).
- **80% prediction interval:** 10th–90th percentiles of draws of `mu` plus
  Student-t noise with that race's `sigma[r]` and `nu`.
- **Unseen driver at a race:** if a test lap's driver has no training laps at
  that race, their offset is drawn from the pooled distribution: `driver[d]`
  plus `sigma_driver_race` noise, centred like the others.
- **Unseen compound or team-compound at a race:** the race-level parameter
  exists for every combination, so it falls back to its pooled value with the
  pooled uncertainty. OLS cannot predict these laps.

## Evaluation (fixed held-out sets, reused for every model)

- **`last_stint`:** each driver's final stint in each race, trained on the
  earlier stints. Mimics forecasting the end of a race. Only drivers with clean
  laps from an earlier stint in that race are tested.
- **`long_tyre`:** laps with `TyreLife > 15`, trained only on `TyreLife <= 15`.
  Tests extrapolation to old tyres.
- **Metrics:**
  - MAE and RMSE of the point prediction;
  - coverage of the 80% prediction interval (should land at about 75–85%).
- **Common subset:** all three models are compared on the laps the baseline
  can predict; the Bayesian model and XGBoost are also scored on every test
  lap.

### Baseline (per-race OLS): results to beat

`lap_time ~ driver + compound + compound:TyreLife + race_lap + dirty_air`, fit
separately per race.

| Split | Train laps | Test laps | Predicted | MAE (s) | RMSE (s) |
|---|---|---|---|---|---|
| last_stint | 9,842 | 5,618 | 5,542 | 1.238 | 4.685 |
| long_tyre | 10,065 | 5,539 | 5,539 | 0.524 | 0.742 |

- **Unpredictable laps:** 76 `last_stint` laps are on a compound that race's
  training data never saw, so OLS has no estimate for them.
- **`last_stint` MAE by race:**
  - 0.43–0.95 s at 15 races;
  - Australia 1.6 s, Abu Dhabi 2.5 s, Saudi Arabia 6.9 s, the fuel-vs-deg
    failure above.
- **`long_tyre`:** MAE grows with tyre age, from 0.47 s at 16–20 laps to
  0.88 s beyond 40.

## Fitting

- **Interface:** `fit(laps, cutoff=None)`. `cutoff=Cutoff(round, lap)` keeps
  every lap of earlier rounds plus laps of that round up to and including
  `lap`, and drops later rounds. Phase 6 refits through the same function.
  Caveat: cleaning currently uses whole-race information (see ROADMAP Phase 6
  open issues).
- **Sampler:** nutpie (NUTS), 4 chains, 1,000 tuning and 1,000 draws.
- **Diagnostics required:** R-hat < 1.01, zero divergences, bulk and tail
  ESS ≥ 400.
- **Time budget:** ~10 minutes per fit. If it won't fit in that, I report it
  before simplifying anything.

## Not in this model

- **Track temperature:** not a covariate (it largely moves with race time and
  is absorbed by `trend` and `race_base`).
- **Team-level pooling of driver offsets:** teammates share a car, but driver
  offsets pool only to the driver's own season pace. Car-specific tyre
  behaviour is carried by `team_compound` and `team_deg`.
- **Race-specific team deg:** `team_deg` is season-level, so a team's wear
  relative to the field is assumed stable across tracks.
