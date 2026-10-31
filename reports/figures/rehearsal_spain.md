# Rehearsal: Spain (round 6), cut after lap 33, focal VER

- 30 green-flag stops known at lap 33 (pit loss median 21.8 s)
- VER sets left: (('HARD', 1), ('MEDIUM', 1), ('SOFT', 4))
- VER's actual plan from lap 33: S → 44 M(new); finished 1

## VER's plans (widening 1.0; best 3 per stop count, plus actual)

| plan                                   |   stops |   mean_pos |   p_win |   p_podium |   p10_pos |   p90_pos |   mean_time_s |   places_gained |   p_ahead_of_ref |   p_behind_ref | actual   |
|:---------------------------------------|--------:|-----------:|--------:|-----------:|----------:|----------:|--------------:|----------------:|-----------------:|---------------:|:---------|
| S → 39 H(new)                          |       1 |      1.868 |   0.46  |      0.962 |         1 |         3 |       5928.57 |           1.079 |            0.457 |          0.095 | False    |
| S → 40 H(new)                          |       1 |      1.91  |   0.464 |      0.947 |         1 |         3 |       5928.37 |           1.037 |            0.454 |          0.078 | False    |
| S → 41 H(new)                          |       1 |      1.942 |   0.46  |      0.942 |         1 |         3 |       5928.4  |           1.005 |            0.44  |          0.056 | False    |
| S → 37 S(used) → 46 H(new)             |       2 |      2.735 |   0.285 |      0.759 |         1 |         4 |       5940.67 |           0.212 |            0.228 |          0.221 | False    |
| S → 38 S(used) → 47 H(new)             |       2 |      2.801 |   0.297 |      0.744 |         1 |         5 |       5940.48 |           0.146 |            0.205 |          0.207 | False    |
| S → 37 M(new) → 47 H(new)              |       2 |      2.815 |   0.29  |      0.738 |         1 |         5 |       5941.18 |           0.132 |            0.206 |          0.222 | False    |
| S → 44 M(new)                          |       1 |      2.947 |   0.384 |      0.713 |         1 |         6 |       5936.77 |           0     |            0     |          0     | True     |
| S → 36 S(used) → 43 M(new) → 51 H(new) |       3 |      4.197 |   0.135 |      0.46  |         1 |         8 |       5956.93 |          -1.25  |            0.069 |          0.649 | False    |
| S → 36 M(new) → 43 S(used) → 51 H(new) |       3 |      4.207 |   0.133 |      0.458 |         1 |         8 |       5956.88 |          -1.26  |            0.067 |          0.65  | False    |
| S → 36 S(used) → 44 M(new) → 51 H(new) |       3 |      4.23  |   0.133 |      0.459 |         1 |         8 |       5956.99 |          -1.283 |            0.063 |          0.657 | False    |
| S                                      |       0 |      5.338 |   0.147 |      0.352 |         1 |        10 |       5968.02 |          -2.391 |            0.003 |          0.791 | False    |

## Sensitivity: widening x1.188 (same draws, top plans and actual)

| plan                       |   stops |   mean_pos |   p_win |   p_podium |   p10_pos |   p90_pos |   mean_time_s |   places_gained |
|:---------------------------|--------:|-----------:|--------:|-----------:|----------:|----------:|--------------:|----------------:|
| S → 39 H(new)              |       1 |      1.881 |   0.45  |      0.957 |         1 |         3 |       5928.4  |           1.078 |
| S → 40 H(new)              |       1 |      1.94  |   0.447 |      0.937 |         1 |         3 |       5928.3  |           1.019 |
| S → 41 H(new)              |       1 |      1.978 |   0.452 |      0.929 |         1 |         3 |       5928.36 |           0.981 |
| S → 37 S(used) → 46 H(new) |       2 |      2.758 |   0.283 |      0.747 |         1 |         5 |       5940.65 |           0.201 |
| S → 38 S(used) → 47 H(new) |       2 |      2.809 |   0.295 |      0.743 |         1 |         5 |       5940.47 |           0.15  |
| S → 37 M(new) → 47 H(new)  |       2 |      2.834 |   0.282 |      0.735 |         1 |         5 |       5941.21 |           0.125 |
| S → 38 M(new) → 47 H(new)  |       2 |      2.849 |   0.284 |      0.736 |         1 |         5 |       5940.98 |           0.11  |
| S → 38 S(used) → 48 H(new) |       2 |      2.856 |   0.297 |      0.732 |         1 |         5 |       5940.57 |           0.103 |
| S → 44 M(new)              |       1 |      2.959 |   0.38  |      0.711 |         1 |         6 |       5936.77 |           0     |

## Field under VER's actual plan vs the actual result

Simulated positions re-ranked among the cars that finished; actual inside the 10th–90th percentile for 94% of them.

| driver   |   pos_at_cutoff |   sim_mean |   p10 |   p90 |   actual | in80   |
|:---------|----------------:|-----------:|------:|------:|---------:|:-------|
| PER      |               1 |       1.88 |     1 |     3 |        2 | True   |
| RUS      |               2 |       2.35 |     1 |     4 |        3 | True   |
| VER      |               3 |       2.95 |     1 |     6 |        1 | True   |
| BOT      |               4 |       4.49 |     3 |     6 |        6 | True   |
| SAI      |               7 |       4.77 |     3 |     7 |        4 | True   |
| HAM      |               8 |       5.59 |     4 |     7 |        5 | True   |
| OCO      |               5 |       7.09 |     5 |     8 |        7 | True   |
| NOR      |               6 |       8.11 |     6 |    10 |        8 | True   |
| ALO      |              11 |      10.32 |     8 |    13 |        9 | True   |
| VET      |               9 |      10.92 |     9 |    14 |       11 | True   |
| TSU      |              10 |      11.11 |     8 |    14 |       10 | True   |
| MSC      |              12 |      11.26 |     9 |    13 |       14 | False  |
| RIC      |              13 |      12.21 |    10 |    14 |       12 | True   |
| GAS      |              14 |      14.2  |    12 |    16 |       13 | True   |
| MAG      |              18 |      14.58 |    12 |    17 |       17 | True   |
| STR      |              15 |      14.99 |    13 |    17 |       15 | True   |
| LAT      |              16 |      16.51 |    15 |    17 |       16 | True   |
| ALB      |              17 |      17.68 |    17 |    18 |       18 | True   |

## Rival plans (own searches) vs their actual stops

| driver   | options (weight)                                                                            | actual                      |   actual_stops_after_cutoff | option_stops   |
|:---------|:--------------------------------------------------------------------------------------------|:----------------------------|----------------------------:|:---------------|
| PER      | M → 34 H(new) (0.97); M → 34 S(used) → 44 H(new) (0.02); M → 34 M(new) → 45 H(new) (0.01)   | M → 37 M(new) → 53 S(used)  |                           2 | 1/2/2          |
| RUS      | M → 34 H(new) (0.99); M → 34 S(used) → 42 H(new) (0.00); M → 34 M(new) → 44 H(new) (0.00)   | M → 36 M(new) → 51 S(used)  |                           2 | 1/2/2          |
| BOT      | M → 34 H(new) (0.95); M → 38 M(new) (0.04); M → 34 M(new) → 46 H(new) (0.02)                | M → 34 M(new)               |                           1 | 1/1/2          |
| OCO      | M → 34 H(new) (0.98); M → 34 M(new) → 45 H(new) (0.01); M → 34 S(used) → 43 H(new) (0.01)   | M → 35 S(used) → 52 S(used) |                           2 | 1/2/2          |
| NOR      | M → 34 H(new) (1.00); M → 34 S(used) → 42 H(new) (0.00); M → 34 H(new) → 56 S(used) (0.00)  | M → 51 S(used)              |                           1 | 1/2/2          |
| SAI      | M → 43 H(new) (0.83); M → 49 S(used) (0.16); M → 43 S(used) → 54 S(used) (0.00)             | M → 45 S(used)              |                           1 | 1/1/2          |
| HAM      | M → 37 H(new) (0.99); M → 48 S(used) (0.01); M → 35 S(used) → 43 H(new) (0.00)              | M → 48 S(used)              |                           1 | 1/1/2          |
| VET      | S → 34 H(used) (0.97); S → 34 S(new) → 46 H(used) (0.02); S → 34 H(used) → 52 S(new) (0.01) | S → 42 M(used)              |                           1 | 1/2/2          |
| TSU      | S → 44 H(new) (0.69); S → 48 S(used) (0.31); S → 43 S(used) → 54 S(used) (0.01)             | S → 52 S(used)              |                           1 | 1/1/2          |
| ALO      | S → 43 H(new) (0.71); S → 47 M(new) (0.18); S → 49 S(used) (0.11)                           | S → 53 S(used)              |                           1 | 1/1/1          |
| MSC      | M → 43 H(new) (0.96); M → 49 S(used) (0.04); M → 41 S(used) → 49 H(new) (0.00)              | M                           |                           0 | 1/1/2          |
| RIC      | M → 39 H(used) (0.97); M → 49 S(used) (0.03); M → 37 S(used) → 44 H(used) (0.00)            | M → 48 S(used)              |                           1 | 1/1/2          |
| GAS      | S → 43 H(new) (0.74); S → 48 S(used) (0.25); S → 42 S(used) → 53 S(used) (0.01)             | S → 46 S(used)              |                           1 | 1/1/2          |
| STR      | M → 36 H(used) (0.73); M → 44 S(new) (0.21); M → 37 S(new) → 51 S(new) (0.06)               | M → 46 S(new)               |                           1 | 1/1/2          |
| LAT      | S → 35 H(new) (1.00); S → 34 M(new) → 42 H(new) (0.00); S → 43 M(new) (0.00)                | S → 50 S(used)              |                           1 | 1/2/1          |
| ALB      | S → 34 H(new) (1.00); S → 34 M(new) → 42 H(new) (0.00); S → 34 S(used) → 40 H(new) (0.00)   | S → 45 M(new) → 54 S(used)  |                           2 | 1/2/2          |
| MAG      | H (0.75); H → 52 S(used) (0.25); H → 45 S(used) → 55 S(used) (0.00)                         | H                           |                           0 | 0/1/2          |
