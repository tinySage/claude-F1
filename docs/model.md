# Tyre degradation model (Phase 3)

Status: **third fit** (final-stint terms, no cliff, odd-round `last_stint`
split). Results are in "Results: final fit" at the end; earlier fits are kept
below it as history. Code: `src/f1strat/deg_model.py`.

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
| `s` | stint: a driver's run on one set of tyres within a race | 942 with clean laps |

## Model

For lap `i` by driver `d` of team `t` in race `r` on compound `c`:

```
y_i ~ StudentT(nu, mu_i, sigma[r])

mu_i = race_base[r]
     + driver_offset[d,r]
     + stint[d,r,s]
     + compound_offset[c,r] + team_compound[t,c,r]
     + (deg[c,r] + team_deg[t,c]) * TyreLife_i
     + post_stop[c] * first_two_flying_i
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
| `stint[d,r,s]` | How much faster or slower this particular stint ran than the driver's typical pace at this race: tyre management, fuel saving, pushing, a set that worked better or worse. | **Non-final stints:** spread `sigma_stint`, centred to sum to zero over the driver's non-final stints in that race. **Final stints:** a pooled mean shift `mu_final` and their own spread `sigma_stint_final`, shared across races, not centred. So `driver_offset` is the car's pace over its non-final stints, and a final stint is that plus its shift. An unseen stint draws from Normal(`mu_final`, `sigma_stint_final`) if it is the final stint (the simulator knows this from the strategy), else Normal(0, `sigma_stint`). |
| `compound_offset[c,r]` | How much faster a new soft or hard is than a new medium **for the field average** at this race. | Medium is the reference (0). Soft and hard pool toward a season mean per compound. |
| `team_compound[t,c,r]` | How much better or worse team `t`'s car gets on soft or hard, relative to the medium, compared with the field at this race. "Ferrari's hards are worse than everyone else's here." | Medium is the reference (0). Zero-sum over teams within each race and compound. Pooled toward `team_compound_season[t,c]`, the team's season tendency, which pools toward 0. |
| `deg[c,r]` | Seconds lost per extra lap of tyre age, for compound `c`, for the field average at race `r`. Linear degradation. | Partial pooling toward a season deg rate per compound. |
| `team_deg[t,c]` | How much faster or slower team `t` wears compound `c` than the field, every race. "Ferrari is harder on its tyres." | Season level, all three compounds, zero-sum over teams. |
| `post_stop[c]` | First two flying laps after a stop, net of tyre warm-up and drivers pushing on new tyres. Can come out either sign: warm-up makes these laps slower, pushing makes them faster. | One value per compound, unconstrained. `first_two_flying` is 1 when laps since the stop is 1 or 2, **only for stints that begin with a pit stop** (not the race start). Counted from the stop, not from `TyreLife`, because a used set starts at `TyreLife` 3+. |
| `opening` | Extra time on race laps 2 and 3: the field is bunched and DRS is off. Not tyre warm-up. | One season-level value. `opening_laps` is 1 on race laps 2–3. |
| `trend[r]` | Seconds gained or lost per race lap from everything that changes with race lap: fuel burning off and the track rubbering in. Expected negative. | Partial pooling across races. Fuel and track evolution both move one-for-one with race lap, so they are not split. |
| `dirty_air_penalty` | Time lost when within 1 s of the car ahead. | One season-level value. |
| `rain_effect` | Time lost on laps with `Rainfall` reported while still on slicks. | One season-level value. In practice only Hungary informs it (see below). |
| `sigma[r]`, `nu` | Lap-to-lap noise at race `r`, and how heavy its tails are. | Student-t, so a few leftover odd laps don't drag the fit. |

### Scaled inputs inside the model

The sampler sees two transformed inputs, so it is not fighting large,
correlated values. Every effect is reported back in seconds and s/lap.

- **Tyre age:** `a = TyreLife / 10` (tyre age in units of 10 laps). Inside the
  model, deg is in s per 10 laps. Reported as deg_s_per_lap = deg / 10. Tyre age is
  scaled but not centred, so a value of 0 still means a new tyre.
- **Race lap:** `l = race_lap − L0`, where `L0` is the mean race lap of the
  training data (30.1 for the full data), stored with the fit. `trend` is
  still in s/lap. `race_base[r]` is then the lap time at race lap `L0`, not at
  lap 0.

### Hierarchy (partial pooling)

Pooled terms use a non-centred parametrisation (`value = mean + scale * z`,
`z ~ Normal(0, 1)`), which samples better when a group has little data. The
exception is `stint_raw`, which uses the centred form: each stint has ~15 clean
laps, so the data dominate its prior. A centred `deg` was also tried; it mixed
worse overall than the non-centred form, so `deg` stays non-centred.

```
driver_season[d]           ~ ZeroSumNormal(sigma_driver)               over drivers
driver_raw[d,r]             = driver_season[d] + sigma_driver_race * z[d,r]
driver_offset[d,r]          = driver_raw[d,r] − mean over drivers in race r

stint_raw[d,r,s]            ~ Normal(0, sigma_stint)                    non-final stints (centred form)
stint[d,r,s]                = stint_raw[d,r,s] − mean over the driver's non-final stints in race r
stint[d,r,final]            ~ Normal(mu_final, sigma_stint_final)        final stint, not centred

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
| `sigma_stint` | HalfNormal(0.3) s | same | Stint-to-stint shifts of a few tenths. |
| `mu_final` | Normal(0, 0.5) s | same | No sign assumed for how final stints differ. |
| `sigma_stint_final` | HalfNormal(0.5) s | same | Final stints were seen to vary about twice as much as others. |
| `post_stop[c]` | Normal(0, 1) s | same | No sign assumed; the sign is a finding. |
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
6. **Linear degradation (limitation).** Degradation is linear in `TyreLife`,
   per compound and race. There is no cliff term.
   - The earlier season-level quadratic was correlated with `deg` and `trend`,
     hurt mixing, over-extrapolated past 30 laps, and could not represent a
     track-specific cliff anyway.
   - On `long_tyre` the linear OLS baseline was the best model beyond 30 laps.
   - **Limitation:** on very long stints a real tyre cliff, where lap time
     rises sharply, is not represented. Predictions past the longest stints
     seen at a track should be treated with care.
7. **Post-stop vs degradation vs race start.** `post_stop` counts laps since a
   pit stop, while `deg` uses `TyreLife`, so on a used set the two differ.
   - Stints that start at the race start get no post-stop flag; race laps 2–3
     get `opening` instead.
   - This keeps the bunched, DRS-off opening laps from being read as a tyre
     effect (or as negative degradation on the first stint).
8. **Stint vs driver vs team compound.** Non-final stint effects sum to zero
   within each driver's race, so they cannot absorb the driver's race pace.
   - The final stint is not centred. Its mean shift `mu_final` is identified
     from the final stints in training (all races in the full fit, even rounds
     in the `last_stint` split), compared with the same drivers' earlier
     stints.
   - A stint effect is a constant over the stint, so it cannot absorb deg
     (which grows with tyre age within the stint).
   - It **can** trade off against `compound_offset` and `team_compound` when
     a compound appears in only one stint. In particular, if a team's only
     stint on a compound at a race is one driver's stint, that stint effect
     and `team_compound[t,c,r]` are fitted from the same laps. Only their sum
     is pinned down by that race's data; pooling (stint toward 0, team
     compound toward the team's season tendency) decides the split.
   - **Hungary 2022:** Leclerc's stint 3 (laps 41–53) is Ferrari's only hard
     stint. The data identify the **sum** of `team_compound[Ferrari, hard,
     Hungary]` and Leclerc's stint 3 effect, not the split between them. The
     second fit gave a posterior correlation of −0.45 between the two. **Phase
     6 must report the sum**, not either term alone.
9. **Fuel vs track evolution.** Not separable (both move with race lap).
   Accepted: `trend` is their sum, and the simulator only needs the sum.
10. **Rain.** All 285 clean laps with `Rainfall` are from Hungary, in its final
   third.
   - `rain_effect` is therefore really a Hungary late-race parameter and
     overlaps Hungary's `trend` and late-stint deg. Expect a wide interval.
   - This matters for the Hungary case study: the model can attribute late
     slowness to rain, to the hard tyre, or to Ferrari's hard specifically,
     and the posterior shows how well it can tell them apart.
11. **Dirty air.** The flag is set when a car is within 1 s of the car ahead,
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
- **Unseen stint:** a stint not in the training data (every `last_stint` test
  stint) gets one draw shared by all its laps, so the interval includes
  stint-to-stint variation. A final stint draws from Normal(`mu_final`,
  `sigma_stint_final`); any other stint from Normal(0, `sigma_stint`).
- **Unseen compound or team-compound at a race:** the race-level parameter
  exists for every combination, so it falls back to its pooled value with the
  pooled uncertainty. OLS cannot predict these laps.

## Evaluation (fixed held-out sets, reused for every model)

- **`last_stint`:** final stints of **odd-numbered rounds**, trained on
  everything else. Even rounds keep their final stints in training so that
  `mu_final` and `sigma_stint_final` can be estimated. Mimics forecasting the
  end of a race. Only drivers with clean laps from an earlier stint in that
  race are tested. 2,508 test laps in 151 stints over 9 rounds.
- **`long_tyre`:** laps with `TyreLife > 15`, trained only on `TyreLife <= 15`.
  A **stress test**, not the use case: every real use (Phase 4 simulations,
  Phase 6 cutoff fits) includes long stints from earlier races.
- **Metrics:**
  - MAE and RMSE of the point prediction;
  - coverage of the 80% prediction interval. Target 75–85% on `last_stint`.
    `long_tyre` has no coverage threshold; its coverage and error-by-age
    table are reported as limitations.
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
- **Sampler:** nutpie (NUTS), 4 chains, 1,500 tuning and 2,500 draws, with
  low-rank mass-matrix adaptation. The stint term adds 942 parameters and
  makes the posterior slower to mix. At 1,500 draws, hyperparameters such as
  `mu_deg`, `sigma_driver` and `mu_trend` sat at R-hat 1.01–1.024. Low-rank
  adaptation handles correlated parameters (deg, trend and compound offsets)
  without changing the model.
- **Saved fits:** `models/deg_full_2022.nc` and the two split fits. With the
  stint term each is about 40 MB, over the 25 MB limit for committing, so all
  are gitignored. Rebuild with `uv run python -m f1strat.deg_model all`
  (~35 min), or `full` for the full-data fit only.
- **Diagnostics required:** R-hat < 1.01 on every parameter, zero divergences,
  bulk and tail ESS ≥ 400. A fit that misses is rerun longer.
- **Time budget:** up to ~20 minutes per fit, run in the background.

## Not in this model

- **Track temperature:** not a covariate (it largely moves with race time and
  is absorbed by `trend` and `race_base`).
- **Team-level pooling of driver offsets:** teammates share a car, but driver
  offsets pool only to the driver's own season pace. Car-specific tyre
  behaviour is carried by `team_compound` and `team_deg`.
- **Race-specific team deg:** `team_deg` is season-level, so a team's wear
  relative to the field is assumed stable across tracks.

## Results: third fit (final-stint terms, no cliff; provisional)

Not final. All three fits miss R-hat < 1.01, and `last_stint` coverage is
below target (see below).

### Diagnostics (2,500 draws, 1,500 tuning, low-rank adaptation)

| Fit | Time | Max R-hat | Parameters over 1.01 | Min ESS bulk / tail | Divergences |
|---|---|---|---|---|---|
| full | 17.6 min | 1.021 | `mu_deg[HARD]`, `sigma_tc_race`, `sigma_driver`, `mu_trend`, `tau_deg[HARD]` | 226 / 194 | 0 |
| last_stint | 16.8 min | 1.017 | hyperparameters | 240 / 356 | 0 |
| long_tyre | 13.5 min | 1.018 | `mu_deg[SOFT]`, `sigma_driver`, `tau_trend` | 247 / 499 | 0 |

Only hyperparameters miss: season means and spreads of deg, trend and team or
driver terms. Every race-, team-, driver- and stint-level term has R-hat below
1.01.

### Held-out comparison

| Split | Model | MAE (s) | RMSE (s) | 80% coverage |
|---|---|---|---|---|
| last_stint (odd rounds; common 2,477 laps) | Baseline | 0.874 | 1.129 | |
| | XGBoost | 1.010 | 1.270 | |
| | Bayesian | **0.598** | **0.841** | **72.3%** |
| long_tyre (5,539 laps) | Baseline | 0.524 | 0.742 | |
| | XGBoost | 0.510 | 0.760 | |
| | Bayesian | **0.505** | **0.726** | 66.6% (stress test; no threshold) |

### `long_tyre` error by tyre age (linear deg)

MAE in seconds, with mean error (predicted − actual) in brackets.

| Tyre age | Laps | Baseline | XGBoost | Bayesian |
|---|---|---|---|---|
| 16–20 | 2,433 | 0.468 (−0.024) | 0.429 (−0.179) | 0.443 (−0.032) |
| 21–30 | 2,291 | 0.523 (−0.036) | 0.507 (−0.226) | 0.524 (+0.025) |
| 31–40 | 708 | 0.662 (−0.332) | 0.712 (−0.427) | **0.617 (−0.226)** |
| 40+ | 107 | 0.882 (−0.652) | 1.067 (−0.555) | **0.769 (−0.248)** |

- Without the cliff, the Bayesian model has the lowest error and the smallest
  bias beyond 30 laps.
- XGBoost's error grows fastest past its training range.
- All three under-predict very old tyres (negative mean error). That is the
  linear-deg limitation.

### Why `last_stint` coverage is still 72%

The final-stint terms are small:
- `mu_final` +0.03 s;
- `sigma_stint_final` 0.31 s, against `sigma_stint` 0.27 s.

But held-out final stints shift by 0.60 s (spread of per-stint mean residuals).
That shift splits roughly in half:
- **Race-wide:** spread 0.44 s. The whole field's final stints at a race are
  off together: Bahrain −0.48, Austria −0.56, São Paulo −0.67, Miami +0.46,
  Zandvoort +0.38 s. A per-driver stint term cannot capture this, and in an
  odd test round no final stint was seen.
- **Driver-to-driver:** spread 0.45 s.

Removing the race-wide part lifts coverage to 76.2%. The cause looks like a
late-race change the linear trend misses (track or temperature evolution,
field-wide management to the flag). Laps more than 30 into a stint also run
0.49 s slower than predicted, consistent with the linear-deg limitation.

### Hungary: Ferrari hard + Leclerc stint 3 (full fit)

Report the sum:

| Term | Mean (s) | 80% interval |
|---|---|---|
| Sum | **+0.237** | [+0.056, +0.418] |
| `team_compound` | +0.122 | [−0.028, +0.277] |
| Stint effect | +0.116 | [−0.061, +0.297] |

The two parts have a posterior correlation of −0.41.

## Results: first fit (no stint term; `post_stop` was called `warmup`)

### Diagnostics

| Fit | Laps | Time | Max R-hat | Min ESS bulk / tail | Divergences |
|---|---|---|---|---|---|
| full | 15,604 | 10.8 min | 1.0097 | 689 / 1,116 | 0 |
| last_stint training | 9,842 | 7.9 min | 1.0052 | 811 / 1,308 | 0 |
| long_tyre training | 10,065 | 9.0 min | 1.0044 | 1,366 / 1,883 | 0 |

- In the full fit, the parameter with the highest R-hat (1.009) and the lowest
  bulk/tail ESS (773 / 1,249) among reported parameters is `mu_deg[HARD]`.
  Lower values above (689 / 1,116) are non-centred z-scores, which are not
  saved.
- Ferrari's Hungary terms and Hungary's deg terms all have bulk and tail ESS
  above 4,400.
- These fits were sampled with the three zero-sum terms (`driver_season`,
  `team_compound_season`, `team_deg`) written in centred form. The code now
  writes them non-centred, as the spec says. It is the same model, and the
  next refit uses the new form.

### Held-out comparison

The 80% coverage column is for the Bayesian model only.

| Split | Subset | Model | MAE (s) | RMSE (s) | 80% coverage |
|---|---|---|---|---|---|
| last_stint | common (5,542) | Baseline OLS | 1.238 | 4.685 | |
| last_stint | common | XGBoost | 0.819 | 1.064 | |
| last_stint | common | Bayesian | **0.578** | **0.832** | 67.9% |
| last_stint | all (5,618) | XGBoost | 0.822 | 1.070 | |
| last_stint | all | Bayesian | 0.585 | 0.849 | 68.1% |
| long_tyre | all (5,539) | Baseline OLS | 0.524 | 0.742 | |
| long_tyre | all | XGBoost | 0.510 | 0.760 | |
| long_tyre | all | Bayesian | 0.531 | 0.741 | 70.2% |

**Intervals under-cover (68–70% vs the 75–85% target).**
- In-sample coverage is 82%, so the per-lap noise model is fine.
- On held-out data, 37% (`last_stint`) and 50% (`long_tyre`) of the residual
  variance is a constant shift per stint: a driver's whole stint runs faster
  or slower than predicted.
- The model has no stint-level term, so its intervals omit that uncertainty.
- Removing each stint's mean residual restores coverage to 79% and 86%.

On `long_tyre` the Bayesian model is slightly too slow beyond ~40 laps (mean
error +0.3 to +0.75 s). XGBoost is flat beyond its training range and too fast
(−0.2 to −0.6 s).

### Season degradation (field-average car, 80% intervals)

| Compound | Deg (s/lap) | Cliff (s/lap²) | Loss after 10 laps (s) | Loss after 20 laps (s) |
|---|---|---|---|---|
| Soft | 0.091 [0.067, 0.115] | 0.00026 [0.00001, 0.00052] | 0.94 [0.70, 1.17] | 1.92 [1.47, 2.39] |
| Medium | 0.076 [0.062, 0.091] | −0.00009 [−0.00020, 0.00002] | 0.75 [0.61, 0.90] | 1.49 [1.20, 1.78] |
| Hard | 0.051 [0.036, 0.067] | 0.00010 [0.00003, 0.00018] | 0.52 [0.37, 0.68] | 1.07 [0.76, 1.38] |

Other season effects:
- trend −0.055 s/lap [−0.060, −0.050];
- dirty air +0.22 s [0.21, 0.23];
- opening laps +0.37 s [0.33, 0.41];
- soft −0.38 s and hard +0.39 s vs a new medium.

### Flags (reported, not forced)

- **Negative field hard deg:**
  - Australia −0.010 s/lap [−0.016, −0.004], P(<0) = 0.98;
  - Saudi Arabia −0.004 (P = 0.80);
  - Miami −0.000 (P = 0.55).
- **Hard degrading faster than soft:** São Paulo, hard 0.103 vs soft
  0.040 s/lap, P = 0.98.
- **`rain_effect` is negative:** −0.17 s [−0.23, −0.11], so rain laps come
  out faster. It is identified by Hungary alone and overlaps Hungary's late
  trend and deg (identifiability point 9).
- **`warmup` is negative for medium and hard** (−0.07 and −0.11 s): the first
  flying laps after a stop are faster than the linear deg line predicts,
  rather than slower. A peak in fresh-tyre grip that linear deg doesn't
  capture is one reading.

### Ferrari at Hungary (full data; interpretation is Phase 6)

All values are relative to medium, 80% intervals.

| Compound | `team_compound` (s) | Field offset (s) | Ferrari total (s) | Ferrari `team_deg` (s/lap, season) |
|---|---|---|---|---|
| Soft | +0.193 [0.056, 0.330] | −0.750 | −0.557 [−0.740, −0.377] | +0.007 [0.002, 0.011] |
| Medium | 0 (reference) | 0 | 0 | +0.007 [0.005, 0.010] |
| Hard | +0.307 [0.111, 0.510] | +1.111 | +1.418 [1.207, 1.631] | +0.001 [−0.002, 0.004] |

## Results: second fit (stint term, with cliff; superseded)

### Diagnostics

| Fit | Time | Max R-hat (parameter) | Min ESS bulk / tail | Divergences |
|---|---|---|---|---|
| full | 21.8 min | **1.0104** (`sigma_tc_race`) | 632 / 775 | 0 |
| last_stint | 16.7 min | 1.0095 | 536 / 997 | 0 |
| long_tyre | 13.9 min | **1.020** (`mu_deg`; also `mu_trend` 1.013, `cliff` 1.011) | 311 / 309 | 0 |

Sampler: 1,500 tuning and 2,500 draws, low-rank adaptation.

### Held-out comparison

| Split | Model | MAE (s) | RMSE (s) | 80% coverage |
|---|---|---|---|---|
| last_stint (common) | Baseline | 1.238 | 4.685 | |
| | XGBoost | 0.819 | 1.064 | |
| | Bayesian | 0.590 | 0.845 | **72.3%** (first fit 67.9%) |
| long_tyre | Baseline | 0.524 | 0.742 | |
| | XGBoost | 0.510 | 0.760 | |
| | Bayesian | 0.552 | 0.769 | **65.9%** (first fit 70.2%) |

**Why `last_stint` still under-covers: final stints are not typical stints.**
- The fitted stint spread is `sigma_stint` = 0.27 s. The held-out final
  stints' mean residuals have a spread of 0.58 s, and they run 0.18 s slower
  than predicted on average.
- Laps fall above the interval 18% of the time and below it 10%.
- Coverage is lowest at São Paulo (35%, a Phase 5 watch item) and Bahrain
  (52%); most races are at 72–84%.
- Plausible causes: cruising or managing to the flag, pushing for fastest lap,
  late restarts.

**Why `long_tyre` got worse.** Its test laps continue stints whose first 15
laps are in training, so the stint effect is fitted, not drawn. The interval no
longer carries stint uncertainty, but the extrapolation bias remains.

### `long_tyre` error by tyre age

Mean error is predicted − actual, in seconds.

| Tyre age | Laps | MAE baseline | MAE XGBoost | MAE Bayesian | Mean error baseline | Mean error XGBoost | Mean error Bayesian |
|---|---|---|---|---|---|---|---|
| 16–20 | 2,433 | 0.468 | 0.429 | 0.446 | −0.024 | −0.179 | −0.014 |
| 21–30 | 2,291 | 0.523 | 0.507 | 0.564 | −0.036 | −0.226 | +0.136 |
| 31–40 | 708 | 0.662 | 0.712 | 0.747 | −0.332 | −0.427 | +0.195 |
| 40+ | 107 | 0.882 | 1.067 | 1.429 | −0.652 | −0.555 | +0.748 |

- **XGBoost:** too fast at every age, with the bias growing past the training
  range. That is the expected tree-model failure.
- **Bayesian:** too slow and growing, worst beyond 40 laps (+0.75 s). Its
  quadratic `cliff`, fitted on tyre ages up to 15, over-extrapolates.
- **Baseline OLS:** has the lowest MAE beyond 30 laps.
- So on this evidence the extrapolation claim is "both flexible models drift,
  in opposite directions", not "the Bayesian model extrapolates better".

### Ferrari at Hungary: stint vs team-compound confounding

Leclerc's stint 3 is Ferrari's only hard stint at Hungary.

| Term | Mean (s) | 80% interval |
|---|---|---|
| `team_compound[Ferrari, hard, Hungary]` | +0.089 | [−0.077, +0.258] |
| `stint_effect[LEC, Hungary, stint 3]` | +0.142 | [−0.054, +0.340] |
| Sum | +0.231 | [+0.034, +0.429] |

- Posterior correlation between the two terms: **−0.45**. The data pin down
  the sum, not the split.
- The first fit's +0.307 s Ferrari hard term is now split roughly 0.09 team /
  0.14 stint. The sum is lower (0.23 s), and the field hard offset moved from
  1.111 to 1.091 s.
- Ferrari's total hard-vs-medium at Hungary is +1.180 s [0.937, 1.424].

Other changes: `post_stop` (formerly `warmup`) is +0.04 / −0.05 / −0.11 s for
soft / medium / hard.
