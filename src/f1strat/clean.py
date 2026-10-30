"""Lap cleaning and exploratory features (Phase 2).

Removes laps that do not reflect representative race pace: lap 1, in/out laps,
SC/VSC/yellow-flag laps and deleted laps. Flags (but keeps) laps driven in
dirty air, and estimates pit lane time loss per circuit.
"""
