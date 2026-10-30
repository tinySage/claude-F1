# Race simulator (Phase 4)

Status: **Part A done** (race state, single car, strategy search). Part B
(rivals, safety cars) is next. Code: `src/f1strat/simulator.py`. Demo:
`notebooks/04_simulator.ipynb`.

## Inputs

- **Lap model:** the Phase 3 degradation model (`docs/model.md`), read from
  the committed thinned posterior `models/deg_full_2022_thin.nc` (1,000 draws,
  250 per chain, of the final full fit). Rebuild it with
  `uv run python -m f1strat.deg_model thin`.
- **Pit losses:** the circuit's observed 2022 green-flag stops, with the same
  definition as Phase 2: (in lap + out lap) − 2 × the driver's median clean
  lap. Each simulated stop resamples one observed stop, so the slow tail is
  kept (Spain: 51 stops, median 22.7 s, slowest 37.6 s).

## Race state

`RaceState` is the race at the end of lap `lap`:
- per car: position, cumulative race time, compound, `TyreLife`, laps since the
  stop, stints so far, and dry compounds used;
- the track status of the lap just completed (green, SC or VSC).

`RaceState.from_laps` builds the actual state of a 2022 race at any lap,
including an SC lap. Lap 1 is not modelled (standing start), so states start
at lap ≥ 1. Part A simulates green laps only; SC/VSC handling is Part B.

## One simulated race (single car)

Simulation `i` uses one posterior draw. For lap `L` on compound `c` at tyre age
`a`:

```
lap = race_base + driver_offset + opening·[L in 2,3] + trend·(L − L0) + noise
    + compound_offset[c] + team_compound[c]
    + (deg[c] + team_deg[c]) · a/10 + post_stop[c]·[1st/2nd lap after a stop]
    + stint effect
```

- **Noise:** Student-t with the race's `sigma` and `nu`, one value per lap.
- **Pit loss:** added to each in lap. A stop at the end of lap `p` puts the
  new set on lap `p+1` (the out lap, TyreLife 1 for a new set).
- **Clean air:** `dirty_air_penalty` is not used, because Part B models
  following explicitly. Rain is off.
- **Stint effects: one draw per stint, never per lap.**
  - The stint in progress uses its fitted effect when the fit contains it
    (e.g. a Phase 6 cutoff fit), else a draw.
  - Future non-final stints draw `sigma_stint · z_j`.
  - The final stint draws `mu_final + final_race[r] + w · sigma_stint_final · z`.
- **Widening factor `w`** (default 1.188, from Phase 3; `widening=1.0` turns it
  off). It scales the drawn final-stint spread. If the fit has no final-stint
  data for the race (as in a Phase 6 cutoff fit), `final_race[r]` is a prior
  draw and is scaled by `w` too.

## Common random numbers

`Draws` holds everything random in simulation `i` that does not depend on the
plan:
- the lap noise;
- the draw for the stint in progress;
- `z_j` for the `j`-th future non-final stint;
- `z` for the final stint (whichever stint is final);
- the pit loss of the `j`-th stop.

Every plan is evaluated on the same `Draws`, so plans are compared on identical
luck. Paired differences (`saved_vs_ref_s`, `p_beats_ref`) are therefore much
tighter than the spread of race times themselves.

## Strategy search

- **Legal plans:** from a `RaceState`, 0 to `max_stops` (default 3) more stops,
  any pit lap (after the state lap, before the last lap), any dry compound,
  new or used sets (a used set's out lap has a given TyreLife, default 4), and
  stints of at least `min_stint` laps (default 1).
- **Two-compound rule:** the whole race must use at least two dry compounds,
  counting compounds already used.
- **Search in two steps:**
  1. Every legal plan is scored on its **exact** mean race time over the
     simulations. A plan's total is the plan-independent laps plus a sum of
     per-stint costs, and each stint cost has a closed form, so the mean is
     computed vectorised over all pit-lap combinations.
  2. The best `shortlist` plans (default 50), plus any plans to compare (e.g.
     what teams actually did), get full Monte Carlo distributions under common
     random numbers.
- **Check:** a test confirms the closed form equals the lap-by-lap simulation.

## Part A demo: Spain 2022, Verstappen from lap 1

Spain was chosen as dry, with no safety car, and not a case study. The
actual plans of the front-runners are applied to Verstappen's car. Driver pace
comes from the full fit (hindsight is fine for a demo).

- **Unlimited sets** (every compound new or used, no limit): 8.7 million legal
  plans scored. All top plans use new softs, about 20 s faster than the plans
  teams ran, because real teams had few new softs left after qualifying.
- **Realistic sets** (new mediums and hards, used softs, as the front-runners
  fitted): 1.1 million legal plans scored.
  - The best plans are three stops, mixing used softs with one new medium.
  - Actual plans: Pérez 3.6 s, Russell 3.8 s, Hamilton 4.2 s and Verstappen
    5.4 s behind the best; Sainz 8.5 s and Bottas (2 stops) 15.6 s.
  - The best plan beats Verstappen's actual plan in 94% of simulations
    (saving 0.8–10.2 s, 10th–90th percentile).
- **Runtime:** 50 plans × 1,000 simulations of Monte Carlo take ~5 ms. A full
  search (scoring every legal plan, then Monte Carlo on the shortlist) takes
  0.1–0.9 s.

## Known limitations (Part A)

- **Tyre allocation is not in the data.** Without `allowed_sets`, the search
  assumes unlimited new sets, which favours new softs.
- **Stint order does not matter on its own.** The lap model has no fuel ×
  compound interaction, so swapping two stints of the same length gives the
  same time in every simulation, and such plans tie. Order will matter once
  traffic and safety cars are modelled (Part B).
- **No traffic.** Extra stops cost only pit loss, not track position, which
  favours three-stop plans here. Part B adds rivals.
