"""
make_amoc_labels.py
-------------------
Author: Farshid Daryabor
Created: July 2025

Description
-----------
This module provides a helper function `make_labels()` for generating binary classification
labels from an Atlantic Meridional Overturning Circulation (AMOC) transport time series.

The function smooths the input AMOC time series, applies a specified transport threshold
(e.g., 15 Sv), and identifies "weakened" AMOC states based on user-defined persistence.
It returns a time-aligned pandas Series of integer labels (1 = weakened, 0 = normal).

This labeling procedure is designed for use in machine learning pipelines studying
AMOC variability, prediction, and regime transition classification.

Key Features
-------------
- **Rolling smoothing**: removes short-term noise using a centered moving mean.
- **Persistence enforcement**: ensures that transient drops below threshold are ignored
  unless sustained for `persistence` consecutive time steps.
- **Flexible input**: accepts both pandas Series and array-like data with or without
  datetime indices.
- **NaN-safe**: missing values are handled conservatively (default = 0).

Example
-------
>>> from make_amoc_labels import make_labels
>>> labels = make_labels(amoc_series, threshold=15.0, persistence=3, smoothing=3)

References
----------
- RAPID AMOC array data (https://rapid.ac.uk)
- Daryabor, F. et al. (2025). Machine Learning Framework for AMOC State Prediction.

License
-------
MIT License
"""

import pandas as pd

def make_labels(amoc_series,
                threshold=15.0,
                persistence=3,
                smoothing=3,
                min_periods=1,
                center=True):
    """
    Create binary labels from an AMOC time series (pandas Series-like).
    ...
    """
    # ensure pandas Series
    amoc_s = pd.Series(amoc_series).copy()
    try:
        amoc_s.index = amoc_series.index
    except Exception:
        pass

    if smoothing is not None and int(smoothing) > 1:
        smoothed = amoc_s.rolling(window=int(smoothing), min_periods=min_periods, center=center).mean()
    else:
        smoothed = amoc_s

    weakened_bool = smoothed < threshold

    if persistence is not None and int(persistence) > 1:
        run_sum = weakened_bool.astype(int).rolling(window=int(persistence), min_periods=1).sum()
        weakened_persistent = run_sum >= int(persistence)
    else:
        weakened_persistent = weakened_bool

    labels = weakened_persistent.astype(int)
    labels[smoothed.isna()] = 0

    if isinstance(amoc_series, pd.Series):
        labels.index = amoc_series.index

    return labels

