# Tyre degradation model (Phase 3)

Status: **spec for review**. Nothing below has been fitted yet. Code:
`src/f1strat/deg_model.py`.

## What the model is for

Predict a car's clean lap time from who is driving, which tyre, how old the
tyre is, and where in the race it is, with honest uncertainty. The simulator
(Phase 4) draws lap times from this model, and the case studies (Phase 6) refit
it on data up to a decision lap.

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
| `c` | compound: soft, medium, hard | 3 |

## Model

For lap `i` by driver `d` in race `r` on compound `c`:

```
y_i ~ StudentT(nu, mu_i, sigma[r])

mu_i = race_base[r]
     + driver_offset[d,r]
     + compound_offset[c,r]
     + deg[c,r]   * TyreLife_i
     + cliff[c]   * TyreLife_i^2
     + warmup[c]  * first_two_flying_i
     + trend[r]   * race_lap_i
     + dirty_air_penalty * dirty_air_i
     + rain_effect       * rainfall_i
```

### Every term in plain words

| Term | What it means | How it is estimated |
|---|---|---|
| `race_base[r]` | Lap time of an average car, on a new medium, at race lap 0, in clean air, in the dry, at race `r`. Sets each track's level. | One free value per race. |
| `driver_offset[d,r]` | How much faster or slower this car and driver are than the field average at this race. Mostly the car. | Partial pooling toward `driver[d]`, the driver's season-average offset. A driver's pace at one race borrows strength from their other races. |
| `compound_offset[c,r]` | How much faster a new soft or hard is than a new medium at this race. | Medium is the reference (fixed at 0). Soft and hard pool toward a season mean per compound, so a race where few cars ran a compound still gets a sensible value. |
| `deg[c,r]` | Seconds lost per extra lap of tyre age, for compound `c` at race `r`. Linear degradation. | Partial pooling toward a season deg rate per compound. |
| `cliff[c]` | Extra loss that grows with the square of tyre age. Positive means wear accelerates on long stints. | One season-level value per compound (not per race). This is smooth curvature, not a sudden cliff. |
| `warmup[c]` | Extra time on the first two flying laps after a stop while the tyres come up to temperature. | One value per compound. `first_two_flying` is 1 when laps since the stop is 1 or 2. It is counted from the stop, not from `TyreLife`, because a used set starts at `TyreLife` 3+. |
| `trend[r]` | Seconds gained or lost per race lap from everything that changes with race lap: fuel burning off (car gets lighter) and the track rubbering in. Expected negative. | Partial pooling across races. Fuel and track evolution both move one-for-one with race lap, so they cannot be separated and are not split. |
| `dirty_air_penalty` | Time lost when within 1 s of the car ahead (turbulent air, overheating tyres, being held up). | One season-level value. |
| `rain_effect` | Time lost on laps with `Rainfall` reported while still on slicks. | One season-level value. See the caveat below: in practice only Hungary informs it. |
| `sigma[r]`, `nu` | Lap-to-lap noise at race `r`, and how heavy its tails are. | Student-t, so a few leftover odd laps (traffic, small errors) don't drag the fit. |

### Hierarchy (partial pooling)

All pooled terms use a non-centred parametrisation (`value = mean + scale * z`,
`z ~ Normal(0, 1)`), which samples better when a group has little data.

```
driver[d]             ~ ZeroSumNormal(sigma_driver)            over all 22 drivers
driver_raw[d,r]        = driver[d] + sigma_driver_race * z_dr[d,r]
driver_offset[d,r]     = driver_raw[d,r] - mean over drivers in race r of driver_raw[., r]

compound_offset[medium, r] = 0
compound_offset[c,r]   = mu_compound[c] + tau_compound[c] * z_cr[c,r]   for c in {soft, hard}

deg[c,r]               = mu_deg[c] + tau_deg[c] * z_deg[c,r]
trend[r]               = mu_trend + tau_trend * z_trend[r]
```

### Priors

Weakly informative, in seconds. They rule out absurd values without pushing
the answer. Degradation is **not** constrained to be positive, so a negative
or inverted result can be seen and flagged (Phase 3 item 12), not hidden.

| Parameter | Prior | Reasoning |
|---|---|---|
| `race_base[r]` | Normal(median training lap at r, 3) | Centred on the track's typical lap. |
| `sigma_driver` | HalfNormal(1.5) | Car/driver spread is up to ~2–3 s per lap. |
| `sigma_driver_race` | HalfNormal(0.5) | Race-to-race variation in a car's relative pace. |
| `mu_compound[soft]`, `mu_compound[hard]` | Normal(0, 1) | No sign assumed. |
| `tau_compound[c]` | HalfNormal(0.5) | |
| `mu_deg[c]` | Normal(0.05, 0.1) s/lap | Typical deg is a few hundredths to ~0.1 s/lap. |
| `tau_deg[c]` | HalfNormal(0.05) | Track-to-track variation in deg. |
| `cliff[c]` | Normal(0, 0.002) s/lap² | ±2 sd gives ±3.6 s extra at 30 laps: allows a real cliff, rules out nonsense. |
| `warmup[c]` | Normal(0, 1) | |
| `mu_trend` | Normal(−0.05, 0.05) s/lap | Fuel alone is roughly −0.03 to −0.06 s/lap. |
| `tau_trend` | HalfNormal(0.03) | |
| `dirty_air_penalty` | Normal(0, 1) | |
| `rain_effect` | Normal(0, 1) | |
| `sigma[r]` | HalfNormal(1) | Clean-lap noise is typically 0.3–0.8 s. |
| `nu` | Gamma(2, 0.1) | Mean 20; lets the data choose heavy tails. |

## Identifiability: how each confusion is handled

1. **Level (who absorbs the average?).** `race_base`, the driver offsets and
   the compound offsets could all shift up and down together. Fixed by:
   - medium is the compound reference;
   - driver offsets are centred to sum to zero within each race;
   - the season driver effects `driver[d]` sum to zero across drivers.

   So `race_base[r]` is the field-average car on a medium.
2. **Fuel vs degradation.** These are separable **only because `TyreLife`
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
   - The baseline shows the failure: in the `last_stint` split at Saudi
     Arabia, 15 of 17 drivers have only one training stint (correlation 0.90).
     OLS lands on deg −1.1 s/lap and trend +1.1 s/lap, which cancel in
     training and blow up on the final stint (MAE 6.9 s).
   - Partial pooling of `deg` and `trend` across races is the fix: a
     poorly-identified race borrows its split from the other 17.
3. **Linear deg vs cliff.** `deg` and `cliff` are correlated (both rise with
   tyre age). `cliff` is season-level, so it is pinned down by all long stints
   across the season rather than by a few laps at one race.
4. **Warm-up vs degradation.** The warm-up flag counts laps since the stop,
   while `deg` uses `TyreLife`. On a used set these differ, so the out-lap
   penalty is not mistaken for negative degradation.
5. **Fuel vs track evolution.** Not separable (both move with race lap).
   Accepted: `trend` is their sum, and the simulator only needs the sum.
6. **Rain.** All 285 clean laps with `Rainfall` are from Hungary, in its final
   third.
   - `rain_effect` is therefore really a Hungary late-race parameter and
     overlaps Hungary's `trend` and late-stint deg. Expect a wide interval.
   - This matters for the Hungary case study: the model can attribute late
     slowness either to rain or to the hard tyre, and the posterior will show
     how much it can tell them apart.
7. **Dirty air.** The flag is set when a car is within 1 s of the car ahead,
   which also happens when it is simply being held up. The penalty includes
   both effects. That is what the simulator needs, but it is not a pure
   aerodynamic estimate.

## Evaluation (fixed held-out sets, reused for every model)

- **`last_stint`:** each driver's final stint in each race, trained on the
  earlier stints. Mimics forecasting the end of a race. Only drivers with clean
  laps from an earlier stint in that race are tested.
- **`long_tyre`:** laps with `TyreLife > 15`, trained only on `TyreLife <= 15`.
  Tests extrapolation to old tyres.
- **Metrics:**
  - MAE and RMSE of the posterior-mean prediction;
  - coverage of the 80% prediction interval (10th–90th percentile of the
    posterior predictive, Student-t noise included), which should land at
    about 75–85%.

### Baseline (per-race OLS): results to beat

`lap_time ~ driver + compound + compound:TyreLife + race_lap + dirty_air`, fit
separately per race.

| Split | Train laps | Test laps | Predicted | MAE (s) | RMSE (s) |
|---|---|---|---|---|---|
| last_stint | 9,842 | 5,618 | 5,542 | 1.238 | 4.685 |
| long_tyre | 10,065 | 5,539 | 5,539 | 0.524 | 0.742 |

- **Unpredictable laps:** 76 `last_stint` laps are on a compound that race's
  training data never saw, so OLS has no estimate for them. The hierarchical
  model can predict them from the season mean.
- **`last_stint` MAE by race:**
  - 0.43–0.95 s at 15 races;
  - Australia 1.6 s, Abu Dhabi 2.5 s, Saudi Arabia 6.9 s, the
    fuel-vs-deg failure above.
- **`long_tyre`:** MAE grows with tyre age, from 0.47 s at 16–20 laps to
  0.88 s beyond 40.

## Fitting

- **Interface:** `fit(laps, cutoff=None)`. `cutoff=Cutoff(round, lap)` keeps
  every lap of earlier rounds plus laps of that round up to and including
  `lap`, and drops later rounds. Phase 6 refits through the same function.
- **Sampler:** nutpie (NUTS), 4 chains, 1,000 tuning and 1,000 draws.
- **Diagnostics required:** R-hat < 1.01, zero divergences, bulk and tail
  ESS ≥ 400.
- **Time budget:** ~10 minutes per fit. If it won't fit in that, I report it
  before simplifying anything.

## Not in this model (open for discussion)

- **Car-specific degradation:** `deg[c,r]` is shared by all cars at a race, so
  "Ferrari is harder on its tyres" is not represented, only "this car is
  slower". This is relevant to the Hungary verdict. Adding a pooled team × deg
  term is possible but is a method change.
- **Track temperature:** not a covariate (it largely moves with race time and
  is absorbed by `trend` and `race_base`).
- **Team-level pooling of driver offsets:** teammates share a car, but
  drivers pool only to themselves across the season.
