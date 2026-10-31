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
