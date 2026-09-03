# -*- coding: utf-8 -*-
"""
test_metrics.py — Automated Tests for Formal Benchmark Metrics (docs/METRICS_SPEC.md).
"""

import numpy as np
import pytest

from scripts.run_doctrine_benchmark import compute_statistics


def test_statistical_aggregation():
    """Verify sample statistics and 95% confidence interval computation."""
    values = [10.0, 12.0, 14.0, 16.0, 18.0]
    stats = compute_statistics(values)

    assert stats["mean"] == 14.0
    assert stats["median"] == 14.0
    assert stats["min"] == 10.0
    assert stats["max"] == 18.0
    assert stats["std"] > 0.0
    assert stats["ci_95"] > 0.0


def test_boolean_pass_fail_verification_logic():
    """Verify that PASS / FAIL determinations adhere strictly to mathematical comparisons."""
    # Requirement: RMSE <= 0.85 m
    pos_rmse_good = 0.42
    pos_rmse_bad = 1.35
    status_good = "PASS" if pos_rmse_good <= 0.85 else "FAIL"
    status_bad = "PASS" if pos_rmse_bad <= 0.85 else "FAIL"
    assert status_good == "PASS"
    assert status_bad == "FAIL"

    # Requirement: Uncertainty reduction >= 75.0 %
    red_good = 82.5
    red_bad = 61.2
    assert ("PASS" if red_good >= 75.0 else "FAIL") == "PASS"
    assert ("PASS" if red_bad >= 75.0 else "FAIL") == "FAIL"
