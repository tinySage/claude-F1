# Figure data

The exact data behind every figure in the docs, the notebooks and the README:
one small table per figure, named after it. `uv run pitwall figures` draws every
figure from these tables alone, in about a minute, with no fits, simulations or
downloads.

| Table | Figure(s) | Built from |
|---|---|---|
| `case_<stem>_points.csv` | `reports/figures/case_<stem>_points.png` | `case_<stem>_results.json` (measurement set) |
| `case_<stem>_sensitivity.csv` | `case_<stem>_sensitivity.png` | the same, every robustness check |
| `case_<stem>_hindsight.csv` | `case_<stem>_hindsight.png` | the same, hindsight replay |
| `case_<stem>_reality.csv` | `case_hungary_reality.png`, `case_silverstone[_v2]_reality.png` | the reality checks |
| `case_silverstone[_v2]_joint.csv` | `case_silverstone[_v2]_joint.png` | Ferrari's joint options |
| `case_monaco_gaps.csv`, `case_monaco_positions.csv` | Monaco's two figures | processed laps; `case_monaco_results.json` |
| `backtest_summary.csv`, `backtest_pit.csv`, `backtest_neutral_draws.csv` | the five `backtest_*.png` | `pitwall backtest run` |
| `clean_laps.parquet`, `exclude_reasons.csv`, `pit_loss.csv` | notebook 02's four figures | processed laps |
| `deg_curves.csv`, `xgb_vs_bayes_long_tyre_*.csv`, `leclerc_hungary_hard.csv` | notebook 03's three figures | processed laps, committed thinned fits |
| `hungary_fan.csv`, `hungary_trace.csv` | README banner and animation | the lap-38 forecast (`pitwall case-study hungary`), processed laps |
| `readme_*.csv` | the other README images | result files; `readme_verdicts.csv` is written by hand from `docs/case_studies.md` |

`<stem>` is `hungary`, `hungary54`, `silverstone` or `silverstone_v2`.
After any full-path step, `uv run pitwall figures --export` rebuilds every table
from its source (except the two hand- or simulation-made ones noted above) and
redraws. Floats are stored at full precision, so a redrawn figure matches the
original; `tests/test_figures.py` checks the case-study figures byte for byte.
