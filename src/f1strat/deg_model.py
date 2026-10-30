"""Tyre degradation models (Phase 3).

- Linear baseline.
- Bayesian hierarchical model in PyMC: car pace offsets, per-compound
  degradation with a cliff term, fuel effect, and track evolution.
- XGBoost, fitted only to show it fails to extrapolate on long stints.
"""
