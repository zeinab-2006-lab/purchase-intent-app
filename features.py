"""Shared feature engineering for the purchase-intent project.

This module is imported by BOTH the notebook and the Streamlit app, so the
saved joblib pipeline can be loaded anywhere (joblib needs `add_features`
to be importable under the same name).
"""
import numpy as np
import pandas as pd

TARGET = "Revenue"

# Raw columns of the UCI "Online Shoppers Purchasing Intention" dataset
PAGE_COUNT_COLS = ["Administrative", "Informational", "ProductRelated"]
DURATION_COLS = [
    "Administrative_Duration",
    "Informational_Duration",
    "ProductRelated_Duration",
]
OTHER_NUM_COLS = ["BounceRates", "ExitRates", "PageValues", "SpecialDay"]
CAT_FEATURES = ["Month", "VisitorType", "OperatingSystems", "Browser", "Region", "TrafficType"]
BIN_FEATURES = ["Weekend"]

# Columns the user / app must provide (everything except the target)
RAW_FEATURES = PAGE_COUNT_COLS + DURATION_COLS + OTHER_NUM_COLS + CAT_FEATURES + BIN_FEATURES

# Numeric columns the model sees after feature engineering
ENGINEERED_COLS = ["TotalPages", "TotalDuration", "AvgTimePerPage", "ProductShare"]
NUM_FEATURES = PAGE_COUNT_COLS + DURATION_COLS + OTHER_NUM_COLS + ENGINEERED_COLS

_LOG_COLS = DURATION_COLS + ["TotalDuration", "AvgTimePerPage", "PageValues"]


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Row-wise feature engineering (no statistics learned from data -> no leakage)."""
    X = X.copy()

    # Durations are in seconds; a few rows can hold invalid negative values.
    for c in DURATION_COLS:
        X[c] = X[c].clip(lower=0)

    X["TotalPages"] = X[PAGE_COUNT_COLS].sum(axis=1)
    X["TotalDuration"] = X[DURATION_COLS].sum(axis=1)
    pages = X["TotalPages"].replace(0, np.nan)
    X["AvgTimePerPage"] = (X["TotalDuration"] / pages).fillna(0)
    X["ProductShare"] = (X["ProductRelated"] / pages).fillna(0)

    # Heavily right-skewed columns -> log1p
    for c in _LOG_COLS:
        X[c] = np.log1p(X[c].clip(lower=0))

    X["Weekend"] = X["Weekend"].astype(int)
    for c in CAT_FEATURES:
        X[c] = X[c].astype(str)
    return X
