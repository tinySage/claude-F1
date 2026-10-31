# Rehearsal: Canada_Cutoff (round 9), cut after lap 35, focal VER

- 3 green-flag stops known at lap 35 (pit loss median 19.3 s)
- VER sets left: (('HARD', 1), ('SOFT', 1), ('SOFT', 1), ('SOFT', 1))
- retired after lap 35 (left out of the field comparison): TSU
- VER's actual plan from lap 35: H → 43 H(new); finished 1

## VER's plans (widening 1.0; best 3 per stop count, plus actual)

| plan                                  |   stops |   mean_pos |   p_win |   p_podium |   p10_pos |   p90_pos |   mean_time_s |   places_gained |   p_ahead_of_ref |   p_behind_ref | actual   |
|:--------------------------------------|--------:|-----------:|--------:|-----------:|----------:|----------:|--------------:|----------------:|-----------------:|---------------:|:---------|
| H → 40 S(new)                         |       1 |      1.276 |   0.777 |      0.989 |         1 |         2 |       5627.61 |           0.573 |            0.409 |          0.037 | False    |
| H → 39 S(new)                         |       1 |      1.292 |   0.765 |      0.991 |         1 |         2 |       5627.48 |           0.557 |            0.398 |          0.046 | False    |
| H → 38 S(new)                         |       1 |      1.321 |   0.745 |      0.986 |         1 |         2 |       5627.65 |           0.528 |            0.386 |          0.056 | False    |
| H → 36 S(new) → 52 S(new)             |       2 |      1.377 |   0.783 |      0.964 |         1 |         2 |       5628.55 |           0.472 |            0.424 |          0.071 | False    |
| H → 36 S(new) → 51 S(new)             |       2 |      1.386 |   0.782 |      0.965 |         1 |         2 |       5628.48 |           0.463 |            0.416 |          0.068 | False    |
| H → 36 S(new) → 53 S(new)             |       2 |      1.422 |   0.776 |      0.957 |         1 |         2 |       5628.59 |           0.427 |            0.408 |          0.076 | False    |
| H → 43 H(new)                         |       1 |      1.849 |   0.442 |      0.942 |         1 |         3 |       5643.5  |           0     |            0     |          0     | True     |
| H → 36 H(new) → 41 S(new) → 54 S(new) |       3 |      2.126 |   0.367 |      0.908 |         1 |         3 |       5645.98 |          -0.277 |            0.171 |          0.284 | False    |
| H → 36 H(new) → 41 S(new) → 55 S(new) |       3 |      2.148 |   0.365 |      0.905 |         1 |         3 |       5645.97 |          -0.299 |            0.176 |          0.283 | False    |
| H → 36 H(new) → 42 S(new) → 55 S(new) |       3 |      2.238 |   0.362 |      0.889 |         1 |         4 |       5646.49 |          -0.389 |            0.169 |          0.289 | False    |
| H                                     |       0 |      4.542 |   0.026 |      0.489 |         2 |         9 |       5669.82 |          -2.693 |            0.012 |          0.856 | False    |

## Sensitivity: widening x1.188 (same draws, top plans and actual)

| plan                      |   stops |   mean_pos |   p_win |   p_podium |   p10_pos |   p90_pos |   mean_time_s |   places_gained |
|:--------------------------|--------:|-----------:|--------:|-----------:|----------:|----------:|--------------:|----------------:|
| H → 40 S(new)             |       1 |      1.307 |   0.758 |      0.987 |         1 |         2 |       5627.56 |           0.56  |
| H → 39 S(new)             |       1 |      1.323 |   0.747 |      0.988 |         1 |         2 |       5627.45 |           0.544 |
| H → 38 S(new)             |       1 |      1.363 |   0.72  |      0.986 |         1 |         2 |       5627.62 |           0.504 |
| H → 36 S(new) → 52 S(new) |       2 |      1.4   |   0.756 |      0.962 |         1 |         2 |       5628.45 |           0.467 |
| H → 36 S(new) → 51 S(new) |       2 |      1.404 |   0.763 |      0.966 |         1 |         2 |       5628.31 |           0.463 |
| H → 36 S(new) → 53 S(new) |       2 |      1.446 |   0.754 |      0.954 |         1 |         2 |       5628.54 |           0.421 |
| H → 36 H(new) → 47 S(new) |       2 |      1.552 |   0.583 |      0.967 |         1 |         2 |       5637.65 |           0.315 |
| H → 36 H(new) → 48 S(new) |       2 |      1.599 |   0.582 |      0.961 |         1 |         2 |       5637.78 |           0.268 |
| H → 43 H(new)             |       1 |      1.867 |   0.444 |      0.937 |         1 |         3 |       5643.37 |           0     |

## Field under VER's actual plan vs the actual result

Simulated positions re-ranked among the cars that finished; actual inside the 10th–90th percentile for 82% of them.

| driver   |   pos_at_cutoff |   sim_mean |   p10 |   p90 |   actual | in80   |
|:---------|----------------:|-----------:|------:|------:|---------:|:-------|
| SAI      |               2 |       1.73 |     1 |     3 |        2 | True   |
| VER      |               1 |       1.84 |     1 |     3 |        1 | True   |
| HAM      |               3 |       3.35 |     2 |     5 |        3 | True   |
| LEC      |               6 |       4.6  |     3 |     6 |        5 | True   |
| RUS      |               4 |       4.78 |     3 |     7 |        4 | True   |
| BOT      |               8 |       5.92 |     4 |     8 |        8 | True   |
| OCO      |               5 |       7.22 |     5 |     9 |        6 | True   |
| ALO      |               7 |       8.25 |     6 |    11 |        7 | True   |
| VET      |              13 |      10.34 |     8 |    13 |       12 | True   |
| STR      |               9 |      10.74 |     8 |    14 |       10 | True   |
| GAS      |              14 |      11.37 |     9 |    15 |       14 | True   |
| ZHO      |              10 |      11.43 |     9 |    15 |        9 | True   |
| NOR      |              17 |      12.35 |     9 |    15 |       15 | True   |
| RIC      |              12 |      13    |    10 |    15 |       11 | True   |
| MAG      |              15 |      13.89 |    11 |    16 |       17 | False  |
| ALB      |              16 |      15.27 |    14 |    16 |       13 | False  |
| LAT      |              18 |      16.93 |    17 |    17 |       16 | False  |

## Rival plans (own searches) vs their actual stops

| driver   | options (weight)                                                                                     | actual         |   actual_stops_after_cutoff | option_stops   |
|:---------|:-----------------------------------------------------------------------------------------------------|:---------------|----------------------------:|:---------------|
| SAI      | H → 43 S(new) (0.61); H → 36 S(new) → 52 S(new) (0.38); H → 36 H(new) → 49 S(new) (0.02)             | H → 49 H(new)  |                           1 | 1/2/2          |
| HAM      | H → 36 S(new) → 52 S(new) (0.44); H → 38 H(new) (0.28); H → 45 S(new) (0.28)                         | H → 44 H(new)  |                           1 | 2/1/1          |
| RUS      | H → 47 S(new) (0.43); H → 36 S(new) → 52 S(new) (0.29); H → 41 H(new) (0.28)                         | H → 45 H(new)  |                           1 | 1/2/1          |
| OCO      | H → 42 S(new) (0.42); H → 38 M(new) (0.41); H → 36 S(new) → 52 S(new) (0.18)                         | H → 49 M(new)  |                           1 | 1/1/2          |
| LEC      | H → 36 M(new) (0.61); H → 36 S(new) → 52 S(new) (0.24); H → 36 S(new) (0.15)                         | H → 41 M(new)  |                           1 | 1/2/1          |
| ALO      | H → 45 S(new) (0.53); H → 42 M(new) (0.37); H → 36 S(new) → 52 S(new) (0.09)                         | H → 49 M(new)  |                           1 | 1/1/2          |
| BOT      | H → 36 M(new) (0.97); H → 36 S(new) (0.02); H → 36 S(new) → 47 M(new) (0.01)                         | H → 49 M(new)  |                           1 | 1/1/2          |
| STR      | H → 36 S(new) (0.73); H → 36 S(new) → 52 S(new) (0.15); H → 36 M(used) (0.11)                        | H → 47 M(used) |                           1 | 1/2/1          |
| ZHO      | H → 41 S(new) (0.77); H → 36 S(new) → 52 S(new) (0.16); H → 40 H(new) (0.06)                         | H → 49 H(new)  |                           1 | 1/2/1          |
| TSU      | H → 38 S(new) (0.78); H → 36 S(new) → 52 S(new) (0.20); H → 36 H(new) → 48 S(new) (0.02)             | H              |                           0 | 1/2/2          |
| RIC      | H → 45 S(new) (0.61); H → 41 H(new) (0.23); H → 36 S(new) → 52 S(new) (0.16)                         | H → 49 H(new)  |                           1 | 1/1/2          |
| VET      | H → 41 S(new) (0.86); H → 36 S(new) → 52 S(new) (0.14); H → 36 S(new) → 47 S(new) → 57 S(new) (0.00) | H              |                           0 | 1/2/3          |
| GAS      | H → 36 S(new) (0.78); H → 36 S(new) → 52 S(new) (0.21); H → 36 H(new) → 47 S(new) (0.02)             | H → 36 H(new)  |                           1 | 1/2/2          |
| MAG      | H → 39 M(new) (0.47); H → 36 S(new) → 52 S(new) (0.28); H → 36 S(new) → 51 M(new) (0.25)             | H              |                           0 | 1/2/2          |
| ALB      | H → 40 H(new) (0.90); H → 48 S(new) (0.07); H → 36 S(new) → 48 H(new) (0.04)                         | H → 48 H(new)  |                           1 | 1/1/2          |
| NOR      | H → 36 M(new) (0.93); H → 45 S(new) (0.05); H → 36 S(new) → 48 M(new) (0.02)                         | H → 42 M(new)  |                           1 | 1/1/2          |
| LAT      | H → 38 H(new) (0.91); H → 36 S(new) → 48 H(new) (0.05); H → 47 S(new) (0.04)                         | H → 37 H(new)  |                           1 | 1/2/1          |
