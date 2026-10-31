# Case studies (Phase 6)

Question for each call: **using only what the Ferrari pit wall knew at the
decision lap, was the call within the range of good options?**

This file fixes the verdict rule **before any case-study simulation is run**
(committed first, so the rule cannot be tuned to the answer). Results are
added below the rule, case by case.

## Verdict rule (all case studies)

### Metric

- **Leclerc's championship points** from the finishing position, 2022 table:
  25, 18, 15, 12, 10, 8, 6, 4, 2, 1 for P1–P10, 0 below. No fastest-lap
  point.
- **Judge the call, not later decisions.** The call is the decision made at
  the cutoff (e.g. "hards at lap 39"). Later stops in the actual race were
  separate decisions made with more information.
  - **Primary comparison:** the **best plan overall** against the **best plan
    that starts with the call** (the call family), both chosen on the
    selection set.
  - **Loss** = best-overall points − best-call-family points, in the same
    simulation.
- Plans are paired by common random numbers, so every plan meets identical
  luck (lap noise, stint and pit-loss draws, SC/VSC timing, pass coin-flips,
  rival plan picks).
- **Also reported, not used for the band:** the loss of the actual plan as
  run, and of the call held to the flag where that applies.

### Choosing the best plan without flattering it

1. **Selection set:** 1,000 simulations. Every candidate plan is simulated.
   The **best plan overall** and the **best plan in each named family**
   (including the call family) are the ones with the highest mean points.
2. **Measurement set:** a **fresh** 2,000 simulations with a different seed:
   new posterior draws, new field draws, and rival plan options rebuilt from
   their own searches on that field.
   - The loss distribution is measured here, for the plans chosen in step 1.
   - Re-using the selection set would let the best plan's own good luck count
     as an advantage (the winner's curse). The fresh set removes that bias.

### Bands

On the measurement set:
- **Within range:** mean loss ≤ 1 point.
- **Outside range:** mean loss > 3 points **and** the best call-family plan
  finishes level with or ahead of the best plan overall (its position ≤ the
  best plan's) in **under 25%** of simulations.
- **Marginal:** anything else.

Also reported: the loss distribution (mean, 10th/50th/90th percentiles,
P(loss > 0), P(loss < 0)) and Leclerc's points distribution for every
candidate plan.

### Tyre-age cap (search, every car)

- **No set may run beyond the oldest tyre age the model has clean 2022 data
  for on that compound,** from the data known at the cutoff.
- **Why:** wear is linear in tyre age (Phase 3), which underpredicts very old
  tyres. A plan beyond the data could win for the wrong reason.
- **Applies to:** every stint, including the one on the car, in Leclerc's
  search and in every rival's own search.
- Every plan the cap removes is reported: how many, and the best removed plans
  by single-car expected time.

### Tyre sets

- Sets come from the reconstructed practice/qualifying inventory, minus the
  sets fitted in the race.
- **Sets already used in the race go back into the pool** with their laps:
  refitting a used race set is legal. This applies to every car; the set on
  the car is not double-counted.

### Robustness

The band is recomputed under each of these, with the same selection and
measurement procedure:

| Check | Change |
|---|---|
| Widening | final-stint widening ×1.188 (Phase 6 default: 1.0; `docs/validation.md`) |
| Ferrari hard wear, low | `deg[hard, r] + team_deg[Ferrari, hard]` fixed at its posterior 10th percentile in every draw (both Ferraris) |
| Ferrari hard wear, high | the same at its 90th percentile |
| Ferrari hard pace, low | `team_compound[Ferrari, hard, r]` fixed at its posterior 10th percentile in every draw (both Ferraris); the field's `compound_offset[hard, r]` keeps its draws |
| Ferrari hard pace, high | the same at its 90th percentile |
| SC rate halved | SC and VSC start probabilities per green lap × 0.5 |
| SC rate doubled | SC and VSC start probabilities × 2 |
| Medium availability | both ways, **only if** the inventory reconstruction is uncertain about it |

The verdict is **robust** only if the band is the same under every check.
Otherwise the verdict states which check moves it and in which direction.

### Hindsight replay (separate from the verdict)

The verdict judges the **call**; the hindsight replay shows the **outcome**.
It is reported next to the verdict, never mixed into it:
- full-season fit (the committed thinned full posterior), which includes this
  race's own later laps;
- the actual SC/VSC periods after the decision lap, forced;
- rivals on their **actual** plans; actual retirements;
- Leclerc's candidate plans, as chosen at the decision lap, plus the best plan
  in hindsight among the same candidates.

### Decision-lap inputs (shared by every case study)

Built by the Phase 5 pipeline (`f1strat.rehearsal`; `docs/validation.md`,
Part B/C):
- **Refit** on laps known at the cutoff (`cutoff_inputs`: cleaning
  recomputed from known laps only; no stint of the decision race labelled
  final), **plus the decision race's FP1–FP3 long runs**
  (`docs/model.md`).
  - **0 divergences** required (`rehearsal.clean_fit`), plus the Phase 3
    diagnostics: R-hat < 1.01, ESS ≥ 400.
- **Leakage-free simulator parameters:**
  - passing, SC/VSC rates, SC closing ratio and pass shift from the dry
    rounds before the decision race;
  - the circuit's pass intercept from its previous running
    (`rehearsal.leakage_free_params`).
- **Pit losses** resampled from the decision race's green-flag stops
  completed by the cutoff.
- **Race state** at the cutoff; **tyre sets** as above, with the tyre-age cap.
- **Rivals:** each on its own top-3 single-car plans from the same fit,
  weighted by `exp(−Δmean / 3 s)`. Never their actual later stops.
- **Simulator settings:** widening 1.0, 1,000 thinned posterior draws.

### Stated limits (all case studies)

- **The verdict concerns Leclerc's points only,** not Ferrari's team result
  (e.g. Sainz, or the constructors' points).
- **Ferrari knew more than this model.**
  - Practice long runs inform tyre **wear**, not pace level: each run has a
    free offset for its unknown fuel load and engine mode.
  - Ferrari knew its own fuel loads and modes, so its view of how its cars
    would run on each compound was sharper.
- **Only the sum is identified** of Ferrari's team-compound term on a
  compound and Leclerc's stint effect on it, when Leclerc's stint is
  Ferrari's only stint on that compound (Phase 3, `docs/model.md`).
- The simulator's limitations in `docs/simulator.md` and
  `docs/validation.md` apply:
  - close following 1.15–1.25× real;
  - linear tyre wear (no cliff);
  - SC small-gap closing;
  - no lap-1 model, time penalties, blue flags or damage.

## Hungary 2022: Leclerc onto hards

### Decision point

- **Leclerc's switch:** in lap **39** (pit entry 1:57:42 session time), new
  hards from lap 40.
- **Cutoff: lap 38**, the last lap he completed before that in-lap began.
- **State at the cutoff** (end of lap 38, 70-lap race):
  - Leclerc **P1** on mediums (TyreLife 17; fitted new on lap 22).
  - Russell P2 +5.4 s, Sainz P3 +6.7 s, Verstappen P4 +9.7 s, Hamilton P5
    +12.7 s.
- **Rival stops known at the cutoff** (in lap ≤ 38):

| Lap | Stops (driver: off → on) |
|---|---|
| 2 (VSC) | ALB: S → M |
| 6 | MAG: M → H |
| 13–19 | TSU 13 S→M; STR 14 S→M(used); NOR 14 S→M; RIC 15 S→M; VET 15 S→M(used); GAS 16 S→M; RUS 16 S→M; VER 16 S→M; SAI 17 M→M; PER 18 S→M; LAT 18 M→M; HAM 19 M→M |
| 21–27 | MSC 21 M→H; ALB 21 M→M; LEC 21 M→M; ALO 21 M→H; OCO 23 M→H; BOT 26 M→H; ZHO 27 M→H |
| 33–35 | TSU 33 M→S; MAG 35 H→M |
| **38** | **VER: M → M (new)**: pit entry 1:56:25, 8 s after Leclerc began lap 39. A stop on the cutoff lap, so it counts as known. |

- **Not known:** Russell's and Latifi's stops at the end of lap 39 (same lap
  as Leclerc's).
- **Verstappen's known stop in the state:** Verstappen is set to new mediums
  for lap 39 (his out lap: TyreLife 1, post-stop laps after it).
  - His lap-38 time already contains the in-lap part of the pit loss. The
    out-lap part is added to his race time as the median out-lap loss
    (out-lap time − median clean lap) of the green-flag Hungary stops known
    at the cutoff: **19.5 s** over 22 stops. At the Hungaroring most of the
    loss falls on the out lap; the in-lap part is 3.5 s.

### Data checks

**Ferrari's tyre sets.**

Reconstruction from FP1–Q; FastF1 timing, FIA 2022 SpR Art. 30.

| Car | Race-day sets | Fitted in the race | Left after lap 38 (besides the set on the car) |
|---|---|---|---|
| Leclerc | M new×2, H new×2, S (4 laps on out lap), S (6) | M new (lap 1), M new (lap 22), H new (lap 40), S TyreLife 4 (lap 55) | **H new×2, S (4), S (6)** |
| Sainz | M new×2, H new×2, S (4), S (6) | M new (lap 1), M new (lap 18), S TyreLife 4 (lap 48) | H new×2, S (4), S (6) |

- **Every fitting matches** a reconstructed set: 0 mismatches for both
  Ferraris. Hungary's 8 mismatches are other cars' (GAS, PER, RIC, STR,
  TSU, VER, VET, ZHO).
- **Leclerc had no new medium at lap 38.** This does not rest on the
  uncertain return assumption:
  - Ferrari was allocated 3 mediums per car.
  - FastF1 shows Leclerc running one of them for 18 laps in FP2. So at most
    2 were new on race day, and he fitted both (laps 1 and 22).
  - **Medium availability is not uncertain, so it is not run both ways.**
- **Used mediums the pit wall did have:**
  - his first-stint race set (21 laps, out-lap TyreLife 22);
  - his FP2 set (18 laps; returned after FP2 under the reconstruction's
    assumption).
  - The race-used first-stint medium is back in the pool (approved; see Tyre
    sets above): out-lap TyreLife 22, so the cap lets it run 14 laps at most.

**Practice long runs on hards** (`f1strat.practice`; 6+ consecutive clean
laps on one set):

| Session | Team | Driver | Clean run (laps) | Tyre age covered |
|---|---|---|---|---|
| FP1 | AlphaTauri | TSU | 9 | 12–20 |
| FP1 | Haas | MAG | 6 | 11–16 |
| FP1 | Haas | MSC | 7 | 10–16 |
| FP1 | Mercedes | HAM | 7 | 9–15 |
| FP1 | Williams | LAT | 7 | 11–17 |
| FP2 | Williams | ALB | 7 | 2–8 |

- **The longest raw hard stint in practice** (any laps, not only clean runs)
  was 12 laps (TSU, FP1, ages 11–22). FP3 was wet (rain on every lap).
- **Ferrari ran no hard laps in practice at all,** and none in the race
  before the cutoff. Nor did Red Bull.
- **Against the decision:** going to hards on lap 40 implies a stint of
  about **31 laps** (laps 40–70) on a tyre the practice runs observed for at
  most 6–9 clean laps, up to age 20.
- **Race laps on hards before the cutoff** carry more: 94 clean laps from
  Haas (MAG laps 7–35), Alpine and Alfa Romeo, up to TyreLife 28.
  - So Hungary's field hard degradation `deg[hard, r]` is informed by the
    race itself, unlike Spain in Phase 5.
  - **Ferrari's own hard terms are not:** `team_compound[Ferrari, hard,
    Hungary]` and `team_deg[Ferrari, hard]` rest on Ferrari's season
    tendency and the pooled race spread.
  - Hence the hard-wear robustness checks.

**Pass difficulty.**
- Hungary 2021 downloads: 985 laps, 130 pass events, 14 passes. The 2021
  race started wet; intermediate laps are dropped by the dry-compound
  filter.
- The circuit intercept comes from that race, shrunk toward the pooled
  pre-Hungary intercept.

**Pit losses.**
- **22 green-flag stops** completed by lap 38: median 23.3 s, range
  19.8–40.6 s.
- Albon's lap-2 stop under VSC is excluded. Verstappen's lap-38 stop is not
  yet complete (out lap 39).
- For reference, the whole race has 40 green-flag stops, median 23.8 s.

**Tyre-age caps** (oldest clean lap in the data known at the cutoff; the
same with or without Hungary's laps):

| Compound | Cap (TyreLife) | Where |
|---|---|---|
| Hard | 63 | Magnussen, Canada |
| Medium | 35 | Schumacher, Spain |
| Soft | 24 | Schumacher, Bahrain |

What the caps mean for Leclerc:
- his current mediums (TyreLife 17) can run to lap 56 at most;
- his used softs (out-lap TyreLife 4 or 6) cannot reach the flag from lap 39;
- his used race medium (TyreLife 22) can run 14 laps;
- hards to the flag (TyreLife 31 at lap 70) are within the cap.

**The call and Leclerc's plans** (from the cutoff):
- **The call:** hards at lap 39 (stop at the end of lap 39 for new hards).
  The **call family** is every plan whose first stop is lap 39 onto new
  hards, with any continuation.
- **As run:** 39 H(new) → 54 S(TyreLife 4); finished **6th**. Reported, not
  used for the band.
- **Hards to the flag:** 39 H(new). Reported, not used for the band.

**Named alternatives** (the best plan in each family is chosen on the
selection set):

| Family | First stop |
|---|---|
| Call: hards at 39 | lap 39, new hard (any continuation) |
| Used softs at 39 | lap 39, used soft (S 4 or S 6) |
| Used medium at 39 | lap 39, used race medium (TyreLife 22) |
| Stay out, then hards | lap 40 or later, new hard (extending the 17-lap mediums) |
| Stay out, then used softs | lap 40 or later, used soft |
| Stay out, then used medium | lap 40 or later, used race medium |
| Best overall | any legal plan within the caps |

- New mediums: **not available** (above).
- Candidates are the best plans by single-car expected time in every family,
  stop count and compound sequence (3 per stratum), plus the as-run and
  hards-to-the-flag plans. All of them are simulated with the whole field.

### Hungary-specific robustness and limits

- **Medium availability:** not uncertain (above); not run both ways.
- **Ferrari hard pace** is the least-known input. Ferrari ran no hard laps
  all weekend, so `team_compound[Ferrari, hard, Hungary]` is its season
  tendency plus pooled race spread; hence the pace checks next to the wear
  checks.
- **Retirements after the cutoff:** Bottas (lap 65). The verdict simulation
  cannot know this; the hindsight replay applies it.
- **The cutoff fit identifies only the sum** of Ferrari's Hungary hard term
  and Leclerc's own hard stint effect, and the stint lies after the cutoff.
  So at the decision lap Ferrari's hard pace is a prior plus the season
  tendency. This is exactly what the hard-wear checks probe.

### Results

Run: `uv run python -m f1strat.case_study hungary`. Outputs in
`reports/figures/`:
- `case_hungary_results.json` (every number below);
- `case_hungary_candidates.csv` (all 164 candidates on the selection set);
- `case_hungary_break_even.json`;
- figures `case_hungary_points.png`, `case_hungary_sensitivity.png`,
  `case_hungary_hindsight.png`.

**Inputs at the cutoff.**
- **Fit:** 11 races (8,709 race laps) plus 233 Hungary practice laps; 40.7
  min. Max R-hat 1.003, min ESS 2,002 (bulk) / 3,194 (tail), **0
  divergences** on the first attempt.
- **Ferrari's hard terms in the cutoff fit:**
  - wear (`deg[hard] + team_deg[Ferrari, hard]`): 0.027 s/lap, p10–p90
    0.014–0.039;
  - pace (`team_compound[Ferrari, hard, Hungary]`): +0.01 s, p10–p90 −0.15
    to +0.17.
  - The field's new hard is 1.12 s/lap slower than a new medium at Hungary.
- **Leakage-free parameters** (dry rounds 1–12 before Hungary):
  - **Hungary pass intercept −4.61**, from the 2021 race (130 events, 14
    passes), against a pooled −2.78: Hungary is much harder to pass at than
    average.
  - Slope 2.74; pass shift +0.91 (replay of rounds 1, 2, 3, 5, 6, 8, 9, 12).
  - SC closing ratio 0.66; SC and VSC start 1.54% per green lap each.
- **Pit losses:** 22 green-flag stops. Verstappen's known stop adds a 19.5 s
  out-lap loss.
- **Leclerc's sets:** new hard ×2, used soft (out-lap TyreLife 4 and 6), used
  race medium (TyreLife 22).
- **Candidates:** 164 plans simulated with the whole field.
- **The tyre-age cap removed 2,753 otherwise legal plans** from Leclerc's
  search.
  - The best removed plans are one stop to the 4-lap soft at laps 45–48:
    `M → 48 S(used 4)` and similar. That runs the soft 23–26 laps, past its
    24-lap cap.
  - The best surviving plan in that family stops at lap 50 (soft to TyreLife
    24 exactly).

**Verdict: within range, robust.**
- **The best plan overall is the call itself:** `M → 39 H(new)`, hards at
  lap 39 run to the flag.
- So the primary loss (best overall − best call-family plan) is exactly 0 on
  the measurement set: mean 0.0, 10th–90th percentile 0 to 0. The call
  family finishes level with or ahead of the best plan in 100% of
  simulations.

Measurement set (2,000 fresh simulations), Leclerc's points:

| Plan (family) | Plan | Mean points | Mean position | P(win) | P(podium) |
|---|---|---|---|---|---|
| **Best overall = call: hards at 39 = hards to the flag** | M → 39 H(new) | **19.9** | 2.03 | 0.47 | 0.86 |
| Stay out, then hards | M → 43 H(new) | 18.8 | 2.22 | 0.35 | 0.87 |
| Stay out, then used softs | M → 50 S(used 4) | 17.2 | 2.67 | 0.24 | 0.81 |
| Used softs at 39 | M → 39 S(used 6) → 53 S(used 4) | 12.1 | 4.43 | 0.05 | 0.33 |
| **As run** | M → 39 H(new) → 54 S(used 4) | 10.9 | 4.87 | 0.02 | 0.21 |
| Used medium at 39 | M → 39 M(used 22) → 44 H(new) | 10.5 | 4.88 | 0.01 | 0.15 |
| Stay out, then used medium | M → 42 M(used 22) → 47 H(new) | 9.9 | 5.18 | 0.00 | 0.12 |

- **The as-run plan** loses a mean **9.0 points** to the best plan (median
  9, 10th–90th percentile 0–17). It finishes level or ahead in only 11%.
  - In the model, the cost is entirely the later decision: a third stop at
    lap 54 (≈ 23 s of pit loss at a circuit where the 2021 data say passing
    is very hard).
  - That later decision was made with information the model does not have
    at lap 38: how Leclerc's hards actually ran in the cool conditions.
  - It is not part of the verdict on the call.
- **New mediums:** not available (no new medium left).
- **Used race medium:** the worst family (≈ 10 points). That set was 21 laps
  old and capped at 14 more laps, so it forces a further stop.

![Leclerc's points per plan at the decision lap](../reports/figures/case_hungary_points.png)

**Robustness: the band is the same under every check.**

| Check | Band | Primary mean loss | Best plan overall | As-run mean loss |
|---|---|---|---|---|
| Main | within range | 0.0 | M → 39 H(new) | 9.0 |
| Widening ×1.188 | within range | 0.0 | M → 39 H(new) | 8.5 |
| Ferrari hard wear p10 (0.014 s/lap) | within range | −0.08 | M → 40 H(new) | 9.9 |
| Ferrari hard wear p90 (0.039 s/lap) | within range | 0.0 | M → 39 H(new) | 7.8 |
| Ferrari hard pace p10 (−0.15 s) | within range | 0.0 | M → 39 H(new) | 9.8 |
| Ferrari hard pace p90 (+0.17 s) | within range | 0.0 | M → 39 H(new) | 8.2 |
| SC/VSC rate ×0.5 | within range | 0.0 | M → 39 H(new) | 8.6 |
| SC/VSC rate ×2 | within range | 0.0 | M → 39 H(new) | 9.6 |

- **Ferrari hard wear p10:** the selection set picked hards one lap later
  (lap 40). On the fresh set the call scored slightly better (mean loss
  −0.08), which is selection noise.

![Verdict under each robustness check](../reports/figures/case_hungary_sensitivity.png)

**Exploratory, not part of the committed rule: how much slower would
Ferrari's hards have to be?**
- The p10/p90 pace checks span only the cutoff fit's uncertainty (±0.16 s),
  and Ferrari ran no hards all weekend. So the full procedure was rerun with
  Ferrari's hard pace shifted further:

| Ferrari hards slower by | Band | Mean loss | Call level or ahead | Best plan overall |
|---|---|---|---|---|
| +0.25 s/lap | within range | 0.0 | 100% | M → 39 H(new) |
| +0.50 s/lap | marginal | 2.4 | 45% | M → 50 S(used 4) |
| +0.75 s/lap | marginal | 4.7 | 28% | M → 50 S(used 4) |
| +1.00 s/lap | **outside range** | 6.7 | 16% | M → 50 S(used 4) |

- **Break-even:** the call stops being the best plan between +0.25 and
  +0.5 s/lap, and falls outside the range of good options at about
  +1 s/lap. The alternative that takes over is staying out on the mediums
  to lap 50, then the 4-lap soft.
- **What the full fit says in hindsight:** with Leclerc's hard laps
  included, Ferrari's Hungary hard deficit is +0.22 s/lap (Ferrari's
  Hungary hard term plus Leclerc's hard-stint effect, 80% interval
  +0.04 to +0.40; Phase 3). That is below the +0.5 s where the band would
  move.
- So even knowing how the hards ran, the model does not say the call was
  outside the range. The model's linear wear and its season-level Ferrari
  terms would miss a larger, temperature-driven problem; nothing in the data
  up to lap 38 shows one.

**Hindsight replay (outcome, not the verdict).**
- Full-season fit, the actual SC/VSC after lap 38 (VSC laps 68–69), rivals
  on their actual plans, Bottas retiring on lap 65.
- **Leclerc actually finished P6 (8 points).**

| Plan | Hindsight mean points | Mean position | P(win) |
|---|---|---|---|
| Best in hindsight: M → 51 S(used 4) | **17.4** | 2.56 | 0.24 |
| Stay out, then used softs: M → 50 S(used 4) | 17.3 | 2.57 | 0.23 |
| Stay out, then hards: M → 43 H(new) | 15.9 | 3.01 | 0.18 |
| **The call: M → 39 H(new)** (hards to the flag) | 15.8 | 3.09 | 0.20 |
| Used softs at 39 | 11.8 | 4.33 | 0.02 |
| **As run: M → 39 H(new) → 54 S(used 4)** | 10.9 | 4.65 | 0.01 |
| Used medium at 39 / stay out, then used medium | 9.2 / 9.3 | 5.4 | 0.00 |

- **In hindsight the call is still a good plan:** 15.8 points, 1.6 behind the
  best in hindsight.
- **The best plan in hindsight** stays out to lap 51 and finishes on the
  used soft. Likely reasons (not separately tested):
  - the actual race had no SC after lap 38, only a late VSC, so nothing
    rewarded an early stop;
  - rivals ran their actual, less aggressive plans.
- **The as-run result** (8 points) is close to the as-run plan's hindsight
  distribution (10.9 points, mean P4.7). The big loss came from adding the
  lap-54 stop, not from the lap-39 call.
- **Limitation:** future stints draw a fresh stint effect, so the replay
  does not reuse the full fit's estimate of Leclerc's actual hard stint
  (+0.09 s/lap). That flatters hard plans in hindsight by about
  0.09 × 31 ≈ 3 s, roughly one place at most here.

![Decision lap vs hindsight](../reports/figures/case_hungary_hindsight.png)

### Hungary conclusion

**Using only what the pit wall knew at the end of lap 38, hards at lap 39 was
within the range of good options. In fact it was the model's best plan, and
the band holds under every robustness check.**
- The plausible alternative, staying out on the mediums to about lap 50 and
  finishing on the 4-lap soft, overtakes the call only if Ferrari's hards
  were at least ~0.5 s/lap slower than the season-level model expects. Only
  at about +1 s/lap does the call fall outside the range.
- What cost Leclerc the result in the model is the later switch to softs
  at lap 54, a separate decision taken with information the model does not
  have.

Limits specific to this verdict:
- It concerns Leclerc's points only.
- Ferrari's hard pace rests on its season tendency (no Ferrari hard laps all
  weekend).
- Wear is linear, so a temperature-driven hard problem would not be seen.
- Ferrari knew its own fuel loads and modes.
