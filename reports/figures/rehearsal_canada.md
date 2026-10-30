# Rehearsal: Canada (round 9), cut after lap 35, focal VER

- 3 green-flag stops known at lap 35 (pit loss median 19.3 s)
- VER sets left: (('HARD', 1), ('SOFT', 1), ('SOFT', 1), ('SOFT', 1))
- retired after lap 35 (left out of the field comparison): TSU
- VER's actual plan from lap 35: H → 43 H(new); finished 1

## VER's plans (widening 1.0; best 3 per stop count, plus actual)

| plan                                  |   stops |   mean_pos |   p_win |   p_podium |   p10_pos |   p90_pos |   mean_time_s |   places_gained |   p_ahead_of_ref |   p_behind_ref | actual   |
|:--------------------------------------|--------:|-----------:|--------:|-----------:|----------:|----------:|--------------:|----------------:|-----------------:|---------------:|:---------|
| H → 40 H(new)                         |       1 |      1.501 |   0.599 |      0.981 |         1 |         2 |       5614.56 |           0.121 |            0.07  |          0.05  | False    |
| H → 41 H(new)                         |       1 |      1.53  |   0.604 |      0.966 |         1 |         2 |       5614.68 |           0.092 |            0.061 |          0.032 | False    |
| H → 42 H(new)                         |       1 |      1.558 |   0.608 |      0.961 |         1 |         2 |       5614.82 |           0.064 |            0.048 |          0.022 | False    |
| H → 43 H(new)                         |       1 |      1.622 |   0.59  |      0.951 |         1 |         2 |       5615.22 |           0     |            0     |          0     | True     |
| H → 36 S(new) → 51 S(new)             |       2 |      1.869 |   0.626 |      0.886 |         1 |         4 |       5609.45 |          -0.247 |            0.289 |          0.273 | False    |
| H → 36 S(new) → 51 H(new)             |       2 |      1.895 |   0.487 |      0.917 |         1 |         3 |       5616.56 |          -0.273 |            0.143 |          0.242 | False    |
| H → 36 H(new) → 51 S(new)             |       2 |      1.909 |   0.468 |      0.916 |         1 |         3 |       5616.53 |          -0.287 |            0.163 |          0.284 | False    |
| H → 36 S(new) → 46 S(new) → 57 S(new) |       3 |      2.494 |   0.408 |      0.823 |         1 |         5 |       5619.1  |          -0.872 |            0.153 |          0.386 | False    |
| H → 36 S(new) → 47 S(new) → 57 S(new) |       3 |      2.563 |   0.403 |      0.812 |         1 |         5 |       5619.35 |          -0.941 |            0.155 |          0.393 | False    |
| H → 36 S(new) → 47 S(new) → 58 S(new) |       3 |      2.583 |   0.4   |      0.813 |         1 |         5 |       5619.3  |          -0.961 |            0.154 |          0.393 | False    |
| H                                     |       0 |      3.738 |   0.115 |      0.586 |         1 |         7 |       5641.27 |          -2.116 |            0.014 |          0.772 | False    |

## Sensitivity: widening x1.188 (same draws, top plans and actual)

| plan                      |   stops |   mean_pos |   p_win |   p_podium |   p10_pos |   p90_pos |   mean_time_s |   places_gained |
|:--------------------------|--------:|-----------:|--------:|-----------:|----------:|----------:|--------------:|----------------:|
| H → 40 H(new)             |       1 |      1.537 |   0.585 |      0.977 |         1 |       2   |       5614.52 |           0.105 |
| H → 41 H(new)             |       1 |      1.559 |   0.589 |      0.964 |         1 |       2   |       5614.64 |           0.083 |
| H → 42 H(new)             |       1 |      1.576 |   0.596 |      0.959 |         1 |       2   |       5614.79 |           0.066 |
| H → 43 H(new)             |       1 |      1.642 |   0.581 |      0.949 |         1 |       2.1 |       5615.16 |           0     |
| H → 46 S(new)             |       1 |      1.886 |   0.631 |      0.887 |         1 |       4   |       5611.37 |          -0.244 |
| H → 36 S(new) → 51 S(new) |       2 |      1.902 |   0.612 |      0.884 |         1 |       4   |       5609.39 |          -0.26  |
| H → 36 S(new) → 51 H(new) |       2 |      1.933 |   0.474 |      0.906 |         1 |       3   |       5616.56 |          -0.291 |
| H → 36 H(new) → 51 S(new) |       2 |      1.946 |   0.461 |      0.911 |         1 |       3   |       5616.5  |          -0.304 |

## Field under VER's actual plan vs the actual result

Simulated positions re-ranked among the cars that finished; actual inside the 10th–90th percentile for 94% of them.

| driver   |   pos_at_cutoff |   sim_mean |   p10 |   p90 |   actual | in80   |
|:---------|----------------:|-----------:|------:|------:|---------:|:-------|
| VER      |               1 |       1.62 |     1 |     2 |        1 | True   |
| SAI      |               2 |       2.49 |     1 |     5 |        2 | True   |
| HAM      |               3 |       3.1  |     2 |     4 |        3 | True   |
| LEC      |               6 |       4.32 |     3 |     5 |        5 | True   |
| RUS      |               4 |       4.6  |     3 |     6 |        4 | True   |
| BOT      |               8 |       6.26 |     5 |     8 |        8 | True   |
| OCO      |               5 |       6.85 |     5 |     8 |        6 | True   |
| ALO      |               7 |       8.12 |     6 |    10 |        7 | True   |
| STR      |               9 |      10.06 |     8 |    13 |       10 | True   |
| ZHO      |              10 |      11.69 |     9 |    15 |        9 | True   |
| VET      |              13 |      11.77 |     8 |    15 |       12 | True   |
| NOR      |              17 |      11.89 |     9 |    15 |       15 | True   |
| RIC      |              12 |      12.56 |    10 |    15 |       11 | True   |
| MAG      |              15 |      12.94 |    10 |    15 |       17 | False  |
| GAS      |              14 |      13.01 |     9 |    16 |       14 | True   |
| ALB      |              16 |      14.92 |    13 |    16 |       13 | True   |
| LAT      |              18 |      16.81 |    16 |    17 |       16 | True   |

## Rival plans (own searches) vs their actual stops

| driver   | options (weight)                                                                         | actual         |   actual_stops_after_cutoff | option_stops   |
|:---------|:-----------------------------------------------------------------------------------------|:---------------|----------------------------:|:---------------|
| SAI      | H → 48 S(new) (0.51); H → 37 S(new) → 53 S(new) (0.38); H → 43 H(new) (0.12)             | H → 49 H(new)  |                           1 | 1/2/1          |
| HAM      | H → 39 H(new) (0.79); H → 36 S(new) → 48 H(new) (0.11); H → 38 S(new) → 53 S(new) (0.10) | H → 44 H(new)  |                           1 | 1/2/2          |
| RUS      | H → 41 H(new) (0.74); H → 51 S(new) (0.18); H → 41 S(new) → 55 S(new) (0.08)             | H → 45 H(new)  |                           1 | 1/1/2          |
| OCO      | H → 38 M(new) (0.93); H → 36 S(new) → 48 M(new) (0.04); H → 36 M(new) → 56 S(new) (0.03) | H → 49 M(new)  |                           1 | 1/2/2          |
| LEC      | H → 36 M(new) (0.93); H → 36 S(new) → 47 M(new) (0.04); H → 36 M(new) → 56 S(new) (0.03) | H → 41 M(new)  |                           1 | 1/2/2          |
| ALO      | H → 41 M(new) (0.91); H → 50 S(new) (0.06); H → 36 S(new) → 48 M(new) (0.03)             | H → 49 M(new)  |                           1 | 1/1/2          |
| BOT      | H → 36 M(new) (0.99); H → 36 S(new) → 43 M(new) (0.01); H → 36 H(new) → 41 M(new) (0.00) | H → 49 M(new)  |                           1 | 1/2/2          |
| STR      | H → 36 M(used) (0.95); H → 45 S(new) (0.02); H → 36 S(new) → 48 M(used) (0.02)           | H → 47 M(used) |                           1 | 1/1/2          |
| ZHO      | H → 41 H(new) (0.62); H → 48 S(new) (0.30); H → 37 S(new) → 53 S(new) (0.08)             | H → 49 H(new)  |                           1 | 1/1/2          |
| TSU      | H → 36 S(new) → 52 S(new) (0.43); H → 44 S(new) (0.41); H → 41 H(new) (0.15)             | H              |                           0 | 2/1/1          |
| RIC      | H → 41 H(new) (0.78); H → 51 S(new) (0.18); H → 36 S(new) → 48 H(new) (0.04)             | H → 49 H(new)  |                           1 | 1/1/2          |
| VET      | H → 48 S(new) (0.82); H → 37 S(new) → 53 S(new) (0.16); H (0.03)                         | H              |                           0 | 1/2/0          |
| GAS      | H → 36 S(new) → 52 S(new) (0.54); H → 42 S(new) (0.31); H → 39 H(new) (0.15)             | H → 36 H(new)  |                           1 | 2/1/1          |
| MAG      | H → 38 M(new) (0.81); H → 36 S(new) → 48 M(new) (0.11); H → 36 M(new) → 56 S(new) (0.08) | H              |                           0 | 1/2/2          |
| ALB      | H → 40 H(new) (0.97); H → 36 S(new) → 46 H(new) (0.02); H → 36 H(new) → 58 S(new) (0.01) | H → 48 H(new)  |                           1 | 1/2/2          |
| NOR      | H → 36 M(new) (0.99); H → 36 S(new) → 44 M(new) (0.01); H → 36 M(new) → 59 S(new) (0.00) | H → 42 M(new)  |                           1 | 1/2/2          |
| LAT      | H → 38 H(new) (0.96); H → 36 S(new) → 46 H(new) (0.03); H → 36 H(new) → 58 S(new) (0.01) | H → 37 H(new)  |                           1 | 1/2/2          |
