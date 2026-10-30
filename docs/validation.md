# Validation (Phase 5)

Status: **Part A done** (passing diagnosis and calibration, backtests).
Part B (Phase 6 dress rehearsal) follows after review. Code:
`src/f1strat/backtest.py`. Rebuild everything with:

```
uv run python -m f1strat.backtest calibrate   # pass-intercept shift -> sim_params.json
uv run python -m f1strat.backtest run         # backtests, tables and plots
```

Tables: `reports/figures/backtest_*.csv`. Plots: `reports/figures/backtest_*.png`.

## What the backtests test

**The lap model is the full posterior.** These backtests use the committed
thinned full fit (`models/deg_full_2022_thin.nc`), which includes each race's
own laps. That is deliberate:
- lap-time accuracy was already tested out of sample in Phase 3 (`last_stint`,
  `long_tyre`; `docs/model.md`);
- these backtests test the **simulator layers** on top of it: traffic,
  passing, SC/VSC, pit losses and how wide the intervals are.

Every stint effect is **drawn**, not taken from the fit, as it would be in a
case study. The race-level terms (base pace, driver offsets, tyre terms) come
from the fit, so these backtests are optimistic about pace. Phase 6 cutoff
refits remove that optimism.

**Races:** the 15 dry 2022 races outside the case studies (Monaco, Britain,
Hungary) and Austria (missing compounds; ROADMAP Decisions log): Bahrain,
Saudi Arabia, Australia, Miami, Spain, Azerbaijan, Canada, France, Belgium,
Netherlands, Italy, United States, Mexico, São Paulo, Abu Dhabi.

**Setup** (`run_backtest`), 1,000 simulations per race and mode:
- **Start:** the actual state at the end of lap 1. The simulator does not
  model lap 1 (`RaceState` starts at lap ≥ 1), so the grid and the first lap
  are taken as they happened, including the starting tyres.
- **Plans:** every car makes its actual stops on its actual laps with its
  actual compounds and set ages (`actual_plan`).
- **Retirements:** a car leaves at its actual last completed lap. Retired
  cars are left out of every metric; the metrics rank the finishers only.
  - A car finished if its last lap ended at or after the winner's (so lapped
    cars count as finishers).
  - A car that stops on track gets a session time but no lap time or position
    for its unfinished lap (e.g. Pérez in Bahrain); that lap is dropped.
- **Two modes:**
  - **replay:** the actual SC/VSC laps are forced in every simulation (red
    flag laps count as SC). Tests pace, traffic and positions.
  - **random:** SC/VSC periods are drawn at the 2022 rates. Tests the SC/VSC
    layer as well. Races with a late SC restart draw `final_race` from its
    prior (Phase 4 rule).
- **Not modelled:** time penalties (finishing order is the order on the
  road), blue flags, damage.

## Metrics

The same counters run on the actual race and on every simulation:
- **On-track pass:** one car is ahead of another at the end of lap `L − 1`
  and behind it at the end of lap `L`. Only green laps after lap 1 count, with
  neither car on an in or out lap and both still running.
  - The order is by race time at the end of each lap number, so lapping a
    backmarker is never a pass.
- **Close following:** the share of car-laps (green, after lap 1, not an
  in/out lap, not the leader) that end within 1 s of the car directly ahead.
- **Finishing position:**
  - share of actual positions inside the simulated central 50% and 80%
    intervals;
  - the PIT (probability integral transform): the share of simulations below
    the actual value, randomised on ties because positions are integers;
  - Spearman rank correlation between the actual order and the mean
    simulated position.
  - On integer positions an inclusive quantile interval covers more than its
    nominal level (a car simulated in P1 or P2 every time has a 50% interval
    of [1, 2]). So the PIT-based coverage (share of PITs in [0.25, 0.75] and
    [0.1, 0.9]) is the fair test.
- **Gap to the winner:** lead-lap finishers other than the winner; PIT
  histogram and interval coverage (continuous, so no tie issue).
- **SC/VSC counts** per race (random mode), against 2022.

## 1. Passing diagnosis

Phase 4 reported about 89 simulated passes per race against 31 in 2022 (about
2.5×). Before changing anything, two candidate causes were checked.

**(a) Lapping and blue flags: explains none of the gap.**
- The 2022 pass events (`race_events.pass_events`) pair a car with the car
  one position ahead *on the same lap number*. A leader lapping a backmarker
  is never such a pair, so lapping is not in the 2022 data.
- The simulator orders cars by race time and has no lapping, so it never
  counts lapping either.
- The backtest counter compares cars on the same lap number too, so lapping
  is excluded on both sides.

**(b) Close following: the simulator does too much of it, but this is not
what made passes look high.**
- Replaying the 15 races with actual plans and actual SC/VSC (Phase 4
  passing), simulated cars spend 1.5× as many car-laps within 1 s of the car
  ahead as in 2022 (even rounds 0.34 vs 0.22; odd 0.39 vs 0.27).
- Yet with the same counter on both sides, simulated passes were close to
  2022: 316 vs 322 over the even rounds and 294 vs 318 over the odd rounds,
  2–8% low.

**What made passes look too high: the two sides of the comparison measured
different things.**

| Effect | Spain example |
|---|---|
| The simulator's internal counter counts every successful pass coin-flip, including cars that pass and are passed back. Net position swaps are fewer. | 100 coin-flip passes vs 80 net swaps (actual plans, drawn SC/VSC) |
| The 2022 figure (31 per race) counted only the fitted events: pairs that started a lap within 1 s, with no pit on either lap. Real on-track swaps are 42.7 per race over these 15 races. 26% of real swaps started more than 1 s behind (a car with a problem), and 24% were not between adjacent cars. | 35 fitted events vs 52 swaps in the real race |
| The demo's rivals ran optimiser plans, and its SC/VSC were drawn. Drawn SCs bunch the field and add passes (see the random-mode results below). | 80 swaps with drawn SC/VSC vs 51 replayed (Spain had none) |

**Answer.** Most of the 2.5× was a counting mismatch, not too much passing.
Close following is a real excess. Held cars sit at exactly 0.68 s behind the
car ahead every lap they are held, so they stay "close". In 2022 the gap of a
car that did not pass ranged from 0.26 to 1.45 s (5th–95th percentile), and
17% fell back beyond 1 s.

## 2. Passing calibration

**Fewest parameters:** one free parameter, plus one change with no free
parameter.
1. **Held gap drawn from 2022 (no free parameter).** A held car's gap is
   drawn from the 2022 gaps of cars within 1 s that did not pass (41
   quantiles stored in `sim_params.json`, `passing.held_gaps`), instead of the
   fixed median 0.68 s.
   - The draw reuses the pass coin-flip's leftover uniform, which is uniform
     given that the car was held, so no random numbers are added and common
     random numbers still hold.
2. **One shift on every circuit's pass intercept** (`passing.shift`), fitted
   by bisection so that replayed **even-round** races have the actual number
   of on-track passes. Result: **+0.674**.
   - Equal-pace pass probability rises from 6% to 11% per lap; an attacker
     1 s/lap faster passes with 62% instead of 46%.
   - What it absorbs: real passes the simulator has no mechanism for, such as
     a car with a problem lap passed from more than 1 s back, or a two-car
     pass.

Checked on the **odd rounds** (not used in the fit):

| Passing | Even rounds (fit): passes, sim / 2022 | close share, sim / 2022 | Odd rounds (check): passes | close share |
|---|---|---|---|---|
| Phase 4 (gap fixed at 0.68 s, no shift) | 316 / 322 | 0.342 / 0.216 | 294 / 318 | 0.393 / 0.267 |
| Dirty-air loss 0.21 s/lap only (the Phase 3 `dirty_air_penalty`) | 292 / 322 | 0.328 / 0.216 | 269 / 318 | 0.376 / 0.267 |
| Held gaps drawn, no shift | 278 / 322 | 0.288 / 0.216 | 252 / 318 | 0.332 / 0.267 |
| **Held gaps drawn + shift 0.674 (final)** | **322 / 322** | **0.279 / 0.216** | **303 / 318** | **0.323 / 0.267** |

- **Passes:** on the odd rounds the calibrated simulator is 5% low
  (303 vs 318).
- **Close following:** improved from 1.5× to 1.2–1.3× of 2022, but still
  high. No pass setting fixes it. With the slope scaled ×0.5 to ×3 and the
  shift refitted each time, the close share stayed between 0.31 and 0.34. So
  the rest is in how cars follow, not in how they pass.
- **Dirty air:** applying the Phase 3 `dirty_air_penalty` (0.21 s/lap) to
  following cars barely helped, because a held car is pinned to the car ahead
  anyway. It is left out (`dirty_air_s = 0`; the option stays in the code).

![passes](../reports/figures/backtest_passes.png)

![close following](../reports/figures/backtest_close.png)

## 3–4. Backtest results

### Overall (249 finishers, 186 lead-lap gaps, 15 races)

| Metric | Target | replay | random |
|---|---|---|---|
| Positions inside 50% interval (inclusive) | ≥ 0.50 | 0.82 | 0.80 |
| Positions inside 80% interval (inclusive) | ≥ 0.80 | 0.95 | 0.98 |
| PIT in [0.25, 0.75] (fair 50% coverage) | 0.50 | **0.59** | **0.59** |
| PIT in [0.1, 0.9] (fair 80% coverage) | 0.80 | **0.86** | **0.90** |
| Spearman, mean over races | | 0.956 | 0.942 |
| Mean abs. error, actual vs mean simulated position | | 1.03 | 1.23 |
| Gap to winner inside 50% interval | 0.50 | 0.59 | 0.62 |
| Gap to winner inside 80% interval | 0.80 | 0.88 | 0.97 |
| On-track passes per race, 2022 / sim | | 42.7 / 41.7 | 42.7 / 44.2 |
| Close share, 2022 / sim | | 0.24 / 0.30 | 0.24 / 0.29 |
| SC periods per race, 2022 / sim | | | 0.80 / 0.77 |
| VSC periods per race, 2022 / sim | | | 0.73 / 0.96 |

![PIT histograms](../reports/figures/backtest_pit.png)

**Reading the PIT histograms:**
- **Positions, replay:** close to flat, with a mild hump in the middle and thin
  tails. The intervals are slightly **too wide** (fair 80% coverage 0.86).
- **Gaps, replay:** roughly flat; a thin top tail (actual gaps seldom exceed
  the simulated 80th percentile), so slightly wide at the top.
- **Gaps, random:** clearly skewed to low PITs. Real gaps are smaller than
  most simulated ones and the 80% interval covers 97%. Two reasons:
  - drawn SC/VSC widen the spread;
  - 44% of simulated races have no SC, against 27% of these 15 races, and an
    SC compresses the gaps (Italy finished under SC: real gaps of a few
    seconds, 50% coverage 0).
- **Positions, random:** flat enough, wider than replay (fair 80% coverage
  0.90).

### Per race: replay (actual SC/VSC)

| Round | Race | Finishers | Pos in 50% | Pos in 80% | Spearman | Gap in 50% | Gap in 80% | Passes 2022 | Passes sim (10–90%) | Close 2022 | Close sim |
|--:|:--|--:|--:|--:|--:|--:|--:|--:|:--|--:|--:|
| 1 | Bahrain | 17 | 0.88 | 0.88 | 0.95 | 0.31 | 0.56 | 57 | 56 (49–64) | 0.18 | 0.30 |
| 2 | Saudi Arabia | 13 | 0.54 | **0.69** | 0.89 | 0.09 | 0.64 | 34 | 42 (34–51) | 0.26 | 0.34 |
| 3 | Australia | 17 | 0.76 | 0.94 | 0.92 | 0.55 | 0.91 | 32 | 26 (19–34) | 0.38 | 0.36 |
| 5 | Miami | 16 | 0.81 | 0.88 | 0.97 | 0.80 | 0.87 | 47 | 36 (28–44) | 0.25 | 0.31 |
| 6 | Spain | 18 | 0.89 | 1.00 | 0.99 | 0.57 | 1.00 | 52 | 51 (44–59) | 0.12 | 0.21 |
| 8 | Azerbaijan | 15 | 0.80 | 0.93 | 0.96 | 0.44 | 1.00 | 33 | 27 (20–34) | 0.14 | 0.25 |
| 9 | Canada | 17 | 0.82 | 1.00 | 0.97 | 0.94 | 1.00 | 31 | 41 (34–50) | 0.32 | 0.33 |
| 12 | France | 15 | 0.93 | 1.00 | 0.98 | 0.86 | 1.00 | 29 | 36 (28–45) | 0.30 | 0.34 |
| 14 | Belgium | 18 | 0.89 | 1.00 | 0.98 | 0.79 | 1.00 | 49 | 39 (32–47) | 0.29 | 0.30 |
| 15 | Netherlands | 18 | 0.67 | 1.00 | 0.95 | 0.38 | 0.94 | 23 | 32 (24–40) | 0.23 | 0.29 |
| 16 | Italy | 16 | 0.94 | 1.00 | 0.96 | 0.55 | 0.82 | 48 | 45 (37–53) | 0.33 | 0.33 |
| 19 | United States | 17 | 0.82 | 1.00 | 0.96 | 0.50 | 0.88 | 60 | 58 (49–67) | 0.30 | 0.37 |
| 20 | Mexico | 18 | 0.83 | 0.94 | 0.96 | 1.00 | 1.00 | 22 | 32 (24–40) | 0.12 | 0.22 |
| 21 | São Paulo | 17 | 0.71 | 0.88 | 0.93 | 0.80 | 0.93 | 68 | 55 (47–63) | 0.21 | 0.32 |
| 22 | Abu Dhabi | 17 | 0.88 | 1.00 | 0.98 | 0.20 | 0.70 | 55 | 49 (40–57) | 0.16 | 0.25 |

### Per race: random (SC/VSC drawn)

| Round | Race | Pos in 50% | Pos in 80% | Spearman | Gap in 80% | Passes 2022 | Passes sim (10–90%) | SC 2022 / sim | VSC 2022 / sim |
|--:|:--|--:|--:|--:|--:|--:|:--|:--|:--|
| 1 | Bahrain | 0.88 | 0.94 | 0.96 | 1.00 | 57 | 58 (44–78) | 1 / 0.75 | 0 / 0.92 |
| 2 | Saudi Arabia | 0.62 | 1.00 | 0.88 | 1.00 | 34 | 29 (20–43) | 1 / 0.64 | 1 / 0.78 |
| 3 | Australia | 0.82 | 0.94 | 0.92 | 1.00 | 32 | 21 (12–35) | 2 / 0.74 | 1 / 0.95 |
| 5 | Miami | 0.62 | 1.00 | 0.93 | 1.00 | 47 | 38 (26–56) | 1 / 0.75 | 0 / 0.92 |
| 6 | Spain | 0.89 | 1.00 | 0.96 | 1.00 | 52 | 67 (46–98) | 0 / 0.85 | 0 / 1.15 |
| 8 | Azerbaijan | 0.80 | 0.93 | 0.91 | 1.00 | 33 | 36 (24–55) | 0 / 0.65 | 2 / 0.82 |
| 9 | Canada | 0.71 | 1.00 | 0.94 | 1.00 | 31 | 47 (33–65) | 1 / 0.92 | 2 / 1.14 |
| 12 | France | 0.93 | 1.00 | 0.98 | 1.00 | 29 | 29 (20–38) | 1 / 0.74 | 1 / 0.85 |
| 14 | Belgium | 0.94 | 1.00 | 0.98 | 1.00 | 49 | 42 (32–54) | 1 / 0.58 | 0 / 0.68 |
| 15 | Netherlands | 0.67 | 0.94 | 0.95 | 1.00 | 23 | 38 (22–58) | 1 / 0.94 | 1 / 1.21 |
| 16 | Italy | 0.94 | 1.00 | 0.95 | 0.64 | 48 | 58 (45–74) | 1 / 0.74 | 1 / 0.85 |
| 19 | United States | 0.71 | 1.00 | 0.91 | 1.00 | 60 | 45 (30–67) | 1 / 0.71 | 0 / 0.93 |
| 20 | Mexico | 0.89 | 1.00 | 0.96 | 1.00 | 22 | 41 (26–62) | 0 / 0.91 | 1 / 1.18 |
| 21 | São Paulo | 0.71 | 0.88 | 0.92 | 0.93 | 68 | 54 (40–70) | 1 / 0.85 | 1 / 1.11 |
| 22 | Abu Dhabi | 0.88 | 1.00 | 0.96 | 1.00 | 55 | 59 (42–80) | 0 / 0.74 | 0 / 0.96 |

![coverage per race](../reports/figures/backtest_coverage.png)

![SC/VSC counts](../reports/figures/backtest_sc_vsc.png)

**SC/VSC rates (random mode).**
- SCs per race match on average (0.77 simulated vs 0.80 in 2022). The
  simulated count is more spread out: 44% of simulated races have no SC
  against 27% in 2022, and more simulated races have two or more.
- VSCs run high (0.96 vs 0.73). The pooled rate comes from all 18 dry races
  (18 VSCs); these 15 races had 11.
- With 15 races, neither difference is clear evidence against the pooled
  rates.

**Passes in random mode** have wide intervals (Spain 46–98). A drawn SC
bunches the field, and passes follow at the restart. The SC bunching issue
below makes this stronger than it should be.

## 5. Australia, Saudi Arabia, Miami, São Paulo (Phase 3 watch items)

- **Australia, Miami, São Paulo backtest fine:** 80% position coverage of
  0.94, 0.88 and 0.88 in replay; Spearman 0.92–0.97. The watch items (flat or
  negative hard degradation in Australia and Miami; hard degrading faster than
  soft in São Paulo) do not show up as position misses.
- **Saudi Arabia is the weakest race** (replay 80% coverage 0.69, gap 80%
  0.64). Watch item first:
  - Saudi's field hard degradation is about zero (−0.001 s/lap, 80% interval
    −0.010 to +0.008), not clearly negative. Almost every car ran a long hard
    final stint, so a flat hard affects everyone about equally.
  - **The big misses are safety-car pit timing**, not tyre wear:
    - Pérez pitted on lap 15 and the SC (after a VSC) came on lap 16. In
      2022 Leclerc, Verstappen and Sainz pitted under it and came out ahead
      of him; Pérez finished 4th. The simulator has him finishing 1.4th on
      average.
    - Hamilton (hard start, stayed out under the SC, pitted lap 40):
      simulated 6.5th, finished 10th.
  - Both point at the SC bunching issue below.

## Found while backtesting: the SC queue closes up in one lap

In the simulator, the first SC lap compresses the whole field to 0.8 s gaps.
- **Cause:** the leader laps at 1.49× and everyone behind at 1.09× their own
  pace, which closes about 38 s per lap.
- **2022:** in the 2022 SC periods, the median gap between consecutive cars
  *grows* on the first SC lap (4.6 → 6.3 s) and then closes over 3–5 laps:
  3.5, 2.0, 1.6, 1.2 s after laps 2–5.
- **Effect:** with instant bunching, a car that pits under the SC loses
  0.72 × its green pit loss against a field that is already queued. So it
  drops behind cars it really stays ahead of: Pérez in Saudi Arabia, above.
  It also exaggerates the restart passes in random mode.
- **Why it matters:** the **Silverstone** case study (Leclerc left out under
  the late SC) turns on exactly this.

**Proposed fix (needs approval; it changes the method).** Limit how fast gaps
close under SC, estimated from these 2022 SC periods:
- gaps are kept on the first SC lap;
- on later SC laps each gap shrinks by a fixed ratio per lap (about 0.6,
  from the medians above), down to `sc_gap_s`.

That is one parameter, fitted on the even-round SCs and checked on the odd
ones, as for passing.

## Is calibration good enough for the case studies?

- **Positions:** yes, slightly conservative. In replay, fair 50% / 80%
  coverage is 0.59 / 0.86 and Spearman 0.96. Intervals are a little wide, so
  a verdict that a call was outside the range of good options is unlikely to
  come from intervals that are too narrow.
- **Gaps:** good in replay (0.59 / 0.88). Too wide in random mode (0.62 /
  0.97). Case-study probabilities that depend on SC luck are therefore
  conservative.
- **Passing:** race-level pass counts match after calibration (odd-round
  check 5% low).

**Limitations to state in Phase 6:**
- Close following is still about 1.2× too high, so time lost in traffic may
  be slightly overstated.
- The SC queue bunches in one lap (above). Until it is fixed, SC-timing
  verdicts, **Silverstone above all**, are not reliable.
- Pace terms come from the full posterior, so these backtests are optimistic
  about pace. Phase 6 cutoff refits are the honest test.
- Lap 1 is taken as it happened; no time penalties, blue flags or damage.
