#!/usr/bin/env python
#####################################################################################################################################################################################################
# Project:       Juniper
# Prototype:     Monitoring and Diagnostic Frontend for Cascade Correlation Neural Network
# File Name:     statistics.py
# Author:        Paul Calnon
# Version:       0.1.0
#
# Date:          2025-11-16
# Last Modified: 2025-12-03
#
# License:       MIT License
# Copyright:     Copyright (c) 2024-2025 Paul Calnon
#
# Description:
#    This module provides comprehensive weight statistics computation for neural networks.
#    Computes various statistical measures including mean, standard deviation, variance,
#    skewness, kurtosis, median absolute deviation, and weight distribution analysis.
#
#####################################################################################################################################################################################################
# Notes:
#
#####################################################################################################################################################################################################
# References:
#
#####################################################################################################################################################################################################
# TODO :
#
#####################################################################################################################################################################################################
# COMPLETED:
#
#####################################################################################################################################################################################################
# from typing import Dict, Any, Optional
from typing import Any, Dict

import numpy as np
from scipy import stats


def compute_weight_statistics(weights: np.ndarray) -> Dict[str, Any]:
    """
    Compute comprehensive statistics on network weights.

    Args:
        weights: NumPy array of weight values

    Returns:
        Dictionary containing all computed statistics

    Raises:
        ValueError: If weights array is invalid
    """
    if weights is None:
        raise ValueError("Weights array cannot be None")

    if not isinstance(weights, np.ndarray):
        weights = np.array(weights)

    # Flatten if multidimensional
    weights_flat = weights.flatten()

    # Handle edge cases
    n = len(weights_flat)
    if n == 0:
        return _empty_statistics()

    if n == 1:
        return _single_value_statistics(weights_flat[0])

    # Check if all values are the same
    if np.all(weights_flat == weights_flat[0]):
        return _constant_statistics(weights_flat[0], n)

    # Compute basic statistics
    total_weights = n
    mean = float(np.mean(weights_flat))
    std_dev = float(np.std(weights_flat, ddof=1))  # Sample standard deviation
    variance = float(np.var(weights_flat, ddof=1))  # Sample variance

    # Compute distribution counts
    positive_count = int(np.sum(weights_flat > 0))
    negative_count = int(np.sum(weights_flat < 0))
    zero_count = int(np.sum(weights_flat == 0))

    # Compute advanced statistics
    try:
        skewness = float(stats.skew(weights_flat, bias=False))
    except Exception:
        skewness = 0.0

    try:
        kurtosis = float(stats.kurtosis(weights_flat, bias=False))
    except Exception:
        kurtosis = 0.0

    # Median Absolute Deviation (MAD)
    median = float(np.median(weights_flat))
    mad = float(np.median(np.abs(weights_flat - median)))

    # Median Absolute Deviation from median
    median_ad = mad  # Same as MAD

    # Interquartile Range (IQR)
    q1, q3 = np.percentile(weights_flat, [25, 75])
    iqr = float(q3 - q1)

    # Z-score distribution (count of values in different ranges)
    if std_dev > 0:
        z_scores = (weights_flat - mean) / std_dev
        z_score_dist = {
            "within_1_sigma": int(np.sum(np.abs(z_scores) <= 1)),
            "within_2_sigma": int(np.sum(np.abs(z_scores) <= 2)),
            "within_3_sigma": int(np.sum(np.abs(z_scores) <= 3)),
            "beyond_3_sigma": int(np.sum(np.abs(z_scores) > 3)),
        }
    else:
        # All values are identical (shouldn't reach here due to earlier check)
        z_score_dist = {
            "within_1_sigma": n,
            "within_2_sigma": n,
            "within_3_sigma": n,
            "beyond_3_sigma": 0,
        }

    return {
        "total_weights": total_weights,
        "positive_weights": positive_count,
        "negative_weights": negative_count,
        "zero_weights": zero_count,
        "mean": mean,
        "std_dev": std_dev,
        "variance": variance,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "median": median,
        "mad": median_ad,
        "median_ad": median_ad,
        "iqr": iqr,
        "q1": float(q1),
        "q3": float(q3),
        "min": float(np.min(weights_flat)),
        "max": float(np.max(weights_flat)),
        "z_score_distribution": z_score_dist,
    }


def _empty_statistics() -> Dict[str, Any]:
    """Return statistics for empty weight array."""
    return {
        "total_weights": 0,
        "positive_weights": 0,
        "negative_weights": 0,
        "zero_weights": 0,
        "mean": 0.0,
        "std_dev": 0.0,
        "variance": 0.0,
        "skewness": 0.0,
        "kurtosis": 0.0,
        "median": 0.0,
        "mad": 0.0,
        "median_ad": 0.0,
        "iqr": 0.0,
        "q1": 0.0,
        "q3": 0.0,
        "min": 0.0,
        "max": 0.0,
        "z_score_distribution": {
            "within_1_sigma": 0,
            "within_2_sigma": 0,
            "within_3_sigma": 0,
            "beyond_3_sigma": 0,
        },
    }


def _single_value_statistics(value: float) -> Dict[str, Any]:
    """Return statistics for single weight value."""
    is_positive = 1 if value > 0 else 0
    is_negative = 1 if value < 0 else 0
    is_zero = 1 if value == 0 else 0

    return {
        "total_weights": 1,
        "positive_weights": is_positive,
        "negative_weights": is_negative,
        "zero_weights": is_zero,
        "mean": value,
        "std_dev": 0.0,
        "variance": 0.0,
        "skewness": 0.0,
        "kurtosis": 0.0,
        "median": value,
        "mad": 0.0,
        "median_ad": 0.0,
        "iqr": 0.0,
        "q1": value,
        "q3": value,
        "min": value,
        # "max": float(value),
        "max": value,
        "z_score_distribution": {
            "within_1_sigma": 1,
            "within_2_sigma": 1,
            "within_3_sigma": 1,
            "beyond_3_sigma": 0,
        },
    }


def _constant_statistics(value: float, count: int) -> Dict[str, Any]:
    """Return statistics for array with all same values."""
    positive_count = count if value > 0 else 0
    negative_count = count if value < 0 else 0
    zero_count = count if value == 0 else 0

    return {
        "total_weights": count,
        "positive_weights": positive_count,
        "negative_weights": negative_count,
        "zero_weights": zero_count,
        "mean": value,
        "std_dev": 0.0,
        "variance": 0.0,
        "skewness": 0.0,
        "kurtosis": 0.0,
        "median": value,
        "mad": 0.0,
        "median_ad": 0.0,
        "iqr": 0.0,
        "q1": value,
        "q3": value,
        "min": value,
        # "max": float(value),
        "max": value,
        "z_score_distribution": {
            "within_1_sigma": count,
            "within_2_sigma": count,
            "within_3_sigma": count,
            "beyond_3_sigma": 0,
        },
    }
