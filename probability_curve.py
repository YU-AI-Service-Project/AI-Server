"""Probability calibration shared by the training and inference paths."""

import numpy as np


def calibration_input(raw_score, horizon_minutes):
    h = np.asarray(horizon_minutes, dtype=float) / 60
    return np.column_stack([raw_score, h, raw_score * h])


def probabilities(raw_score, horizon_minutes, current_available, calibrators):
    result = np.empty_like(raw_score, dtype=float)
    for current in (0, 1):
        keep = np.asarray(current_available) == current
        config = calibrators[str(current)]
        x = calibration_input(raw_score[keep], horizon_minutes[keep])
        log_odds = x @ np.asarray(config["coefficients"]) + config["intercept"]
        result[keep] = 1 / (1 + np.exp(-np.clip(log_odds, -700, 700)))
    return result
