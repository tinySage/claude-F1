# Rehearsal: Spain_Cutoff (round 6), cut after lap 33, focal VER

- 30 green-flag stops known at lap 33 (pit loss median 21.8 s)
- VER sets left: (('HARD', 1), ('MEDIUM', 1), ('SOFT', 4))
- VER's actual plan from lap 33: S → 44 M(new); finished 1

## VER's plans (widening 1.0; best 3 per stop count, plus actual)

| plan                                   |   stops |   mean_pos |   p_win |   p_podium |   p10_pos |   p90_pos |   mean_time_s |   places_gained |   p_ahead_of_ref |   p_behind_ref | actual   |
|:---------------------------------------|--------:|-----------:|--------:|-----------:|----------:|----------:|--------------:|----------------:|-----------------:|---------------:|:---------|
| S → 46 M(new)                          |       1 |      1.455 |   0.683 |      0.972 |         1 |       2.1 |       6033.44 |           0.047 |            0.123 |          0.077 | False    |
| S → 45 M(new)                          |       1 |      1.473 |   0.668 |      0.968 |         1 |       3   |       6033.59 |           0.029 |            0.089 |          0.063 | False    |
| S → 44 M(new)                          |       1 |      1.502 |   0.649 |      0.969 |         1 |       3   |       6033.86 |           0     |            0     |          0     | True     |
| S → 39 S(used) → 50 M(new)             |       2 |      2.322 |   0.398 |      0.851 |         1 |       5   |       6043.06 |          -0.82  |            0.078 |          0.426 | False    |
| S → 39 S(used) → 51 M(new)             |       2 |      2.537 |   0.37  |      0.821 |         1 |       5   |       6043.85 |          -1.035 |            0.077 |          0.465 | False    |
| S → 40 H(new) → 49 M(new)              |       2 |      2.613 |   0.276 |      0.826 |         1 |       5   |       6046.05 |          -1.111 |            0.052 |          0.581 | False    |
| S → 37 H(new) → 43 S(used) → 53 M(new) |       3 |      4.284 |   0.067 |      0.497 |         2 |       8   |       6057.19 |          -2.782 |            0.011 |          0.876 | False    |
| S → 38 H(new) → 44 S(used) → 53 M(new) |       3 |      4.389 |   0.073 |      0.486 |         2 |       8.1 |       6057.61 |          -2.887 |            0.011 |          0.877 | False    |
| S → 38 H(new) → 44 S(used) → 54 M(new) |       3 |      4.457 |   0.068 |      0.479 |         2 |       9   |       6057.93 |          -2.955 |            0.012 |          0.879 | False    |
| S                                      |       0 |      5.384 |   0.022 |      0.271 |         3 |       9   |       6065.79 |          -3.882 |            0.002 |          0.947 | False    |

## Sensitivity: widening x1.188 (same draws, top plans and actual)

| plan                       |   stops |   mean_pos |   p_win |   p_podium |   p10_pos |   p90_pos |   mean_time_s |   places_gained |
|:---------------------------|--------:|-----------:|--------:|-----------:|----------:|----------:|--------------:|----------------:|
| S → 46 M(new)              |       1 |      1.486 |   0.666 |      0.969 |         1 |         3 |       6033.31 |           0.036 |
| S → 45 M(new)              |       1 |      1.511 |   0.645 |      0.962 |         1 |         3 |       6033.51 |           0.011 |
| S → 44 M(new)              |       1 |      1.522 |   0.638 |      0.965 |         1 |         3 |       6033.73 |           0     |
| S → 46 S(used)             |       1 |      1.774 |   0.516 |      0.939 |         1 |         3 |       6037.17 |          -0.252 |
| S → 47 S(used)             |       1 |      1.798 |   0.522 |      0.927 |         1 |         3 |       6037.08 |          -0.276 |
| S → 45 S(used)             |       1 |      1.832 |   0.498 |      0.923 |         1 |         3 |       6037.77 |          -0.31  |
| S → 39 S(used) → 50 M(new) |       2 |      2.352 |   0.376 |      0.849 |         1 |         5 |       6043.01 |          -0.83  |
| S → 47 H(new)              |       1 |      2.394 |   0.308 |      0.821 |         1 |         4 |       6043.37 |          -0.872 |

## Field under VER's actual plan vs the actual result

Simulated positions re-ranked among the cars that finished; actual inside the 10th–90th percentile for 94% of them.

| driver   |   pos_at_cutoff |   sim_mean |   p10 |   p90 |   actual | in80   |
|:---------|----------------:|-----------:|------:|------:|---------:|:-------|
| VER      |               3 |       1.5  |     1 |     3 |        1 | True   |
| PER      |               1 |       2.46 |     1 |     4 |        2 | True   |
| RUS      |               2 |       2.76 |     1 |     4 |        3 | True   |
| BOT      |               4 |       4.33 |     3 |     6 |        6 | True   |
| SAI      |               7 |       4.82 |     3 |     6 |        4 | True   |
| HAM      |               8 |       5.83 |     5 |     7 |        5 | True   |
| OCO      |               5 |       7.41 |     6 |     9 |        7 | True   |
| ALO      |              11 |       8.68 |     7 |    11 |        9 | True   |
| NOR      |               6 |       9.44 |     7 |    12 |        8 | True   |
| TSU      |              10 |       9.98 |     8 |    12 |       10 | True   |
| MSC      |              12 |      11.91 |    10 |    15 |       14 | True   |
| VET      |               9 |      12.48 |     9 |    15 |       11 | True   |
| RIC      |              13 |      12.9  |    11 |    16 |       12 | True   |
| GAS      |              14 |      13.42 |    12 |    15 |       13 | True   |
| STR      |              15 |      14.1  |    11 |    16 |       15 | True   |
| MAG      |              18 |      14.76 |    11 |    16 |       17 | False  |
| LAT      |              16 |      16.57 |    15 |    17 |       16 | True   |
| ALB      |              17 |      17.64 |    17 |    18 |       18 | True   |

## Rival plans (own searches) vs their actual stops

| driver   | options (weight)                                                                                           | actual                      |   actual_stops_after_cutoff | option_stops   |
|:---------|:-----------------------------------------------------------------------------------------------------------|:----------------------------|----------------------------:|:---------------|
| PER      | M → 40 M(new) (0.43); M → 34 S(used) → 48 M(new) (0.32); M → 34 M(new) → 51 S(used) (0.26)                 | M → 37 M(new) → 53 S(used)  |                           2 | 1/2/2          |
| RUS      | M → 37 M(new) (0.75); M → 34 S(used) → 46 M(new) (0.15); M → 34 M(new) → 52 S(used) (0.10)                 | M → 36 M(new) → 51 S(used)  |                           2 | 1/2/2          |
| BOT      | M → 38 M(new) (0.85); M → 34 S(used) → 46 M(new) (0.09); M → 34 M(new) → 52 S(used) (0.06)                 | M → 34 M(new)               |                           1 | 1/2/2          |
| OCO      | M → 38 M(new) (0.59); M → 34 S(used) → 47 M(new) (0.24); M → 34 M(new) → 51 S(used) (0.17)                 | M → 35 S(used) → 52 S(used) |                           2 | 1/2/2          |
| NOR      | M → 44 S(used) (0.61); M → 35 S(used) → 50 S(used) (0.31); M → 43 H(new) (0.08)                            | M → 51 S(used)              |                           1 | 1/2/1          |
| SAI      | M → 50 S(used) (0.92); M → 51 H(new) (0.06); M (0.02)                                                      | M → 45 S(used)              |                           1 | 1/1/0          |
| HAM      | M → 48 S(used) (0.86); M → 49 H(new) (0.08); M → 41 S(used) → 53 S(used) (0.05)                            | M → 48 S(used)              |                           1 | 1/1/2          |
| VET      | S → 36 H(used) → 46 S(new) (0.49); S → 36 S(new) → 54 H(used) (0.27); S → 37 M(used) → 45 S(new) (0.24)    | S → 42 M(used)              |                           1 | 2/2/2          |
| TSU      | S → 48 S(used) (0.96); S → 43 S(used) → 54 S(used) (0.02); S (0.02)                                        | S → 52 S(used)              |                           1 | 1/2/0          |
| ALO      | S → 47 M(new) (0.74); S → 49 S(used) (0.25); S → 50 H(new) (0.01)                                          | S → 53 S(used)              |                           1 | 1/1/1          |
| MSC      | M → 50 S(used) (0.84); M → 50 H(new) (0.13); M → 44 S(used) → 54 S(used) (0.02)                            | M                           |                           0 | 1/1/2          |
| RIC      | M → 50 S(used) (0.86); M (0.10); M → 51 H(used) (0.04)                                                     | M → 48 S(used)              |                           1 | 1/0/1          |
| GAS      | S → 48 S(used) (0.96); S → 42 S(used) → 53 S(used) (0.03); S → 50 H(new) (0.01)                            | S → 46 S(used)              |                           1 | 1/2/1          |
| STR      | M → 45 S(new) (0.87); M → 37 S(new) → 51 S(new) (0.12); M → 41 H(used) → 48 S(new) (0.00)                  | M → 46 S(new)               |                           1 | 1/2/2          |
| LAT      | S → 42 M(new) (0.97); S → 44 H(new) (0.02); S → 39 S(used) → 48 M(new) (0.02)                              | S → 50 S(used)              |                           1 | 1/1/2          |
| ALB      | S → 41 M(new) (0.96); S → 38 S(used) → 48 M(new) (0.02); S → 37 H(new) → 48 M(new) (0.01)                  | S → 45 M(new) → 54 S(used)  |                           2 | 1/2/2          |
| MAG      | H → 46 S(used) (0.81); H → 39 S(used) → 52 S(used) (0.19); H → 35 S(used) → 45 S(used) → 55 S(used) (0.00) | H                           |                           0 | 1/2/3          |
