# -*- coding: utf-8 -*-
"""
test_metrics.py — Automated Tests for Production Evaluation Metrics Engine (src/evaluation/metrics.py).
"""

import numpy as np
import pytest

from src.evaluation.metrics import (
    evaluate_enclosure,
    evaluate_tti,
    evaluate_coverage,
    evaluate_position_rmse,
    evaluate_network_retention,
    evaluate_requirement,
)


def test_evaluate_enclosure_geometry():
    """Verify angular enclosure calculation for 0, 90, and 180 degrees."""
    pt = np.array([0.0, 0.0])

    # Collinear same side: 0 degrees
    p1_0 = np.array([5.0, 0.0])
    p2_0 = np.array([8.0, 0.0])
    assert np.isclose(evaluate_enclosure(p1_0, p2_0, pt), 0.0, atol=1e-3)

    # Orthogonal: 90 degrees
    p1_90 = np.array([5.0, 0.0])
    p2_90 = np.array([0.0, 5.0])
    assert np.isclose(evaluate_enclosure(p1_90, p2_90, pt), 90.0, atol=1e-3)

    # Opposite sides: 180 degrees
    p1_180 = np.array([5.0, 0.0])
    p2_180 = np.array([-5.0, 0.0])
    assert np.isclose(evaluate_enclosure(p1_180, p2_180, pt), 180.0, atol=1e-3)


def test_evaluate_tti_continuous_holding_window():
    """Verify continuous holding window requirement for TTI."""
    dt = 0.1
    # 1. Case A: Held for only 1.2s (< 1.5s threshold) -> FAIL
    times_a = [i * dt for i in range(25)]
    target_pos = [np.array([0.0, 0.0]) for _ in times_a]

    # Drones hold pincer for 12 steps (1.2s), then break
    d0_pos_a = [np.array([4.0, 0.0]) if i < 12 else np.array([20.0, 0.0]) for i in range(25)]
    d1_pos_a = [np.array([0.0, 4.0]) if i < 12 else np.array([0.0, 20.0]) for i in range(25)]

    res_a = evaluate_tti(
        timestamps=times_a,
        drone_positions={0: d0_pos_a, 1: d1_pos_a},
        target_positions=target_pos,
        holding_window_s=1.5,
        standoff_threshold_m=6.0,
        enclosure_threshold_deg=60.0,
    )
    assert not res_a.interception_success
    assert res_a.tti_seconds is None
    assert res_a.failure_reason == "INSUFFICIENT_HOLD_DURATION"
    assert res_a.longest_hold_duration_s == 1.1

    # 2. Case B: Held for 1.8s (>= 1.5s threshold) -> PASS
    d0_pos_b = [np.array([4.0, 0.0]) if i < 18 else np.array([20.0, 0.0]) for i in range(25)]
    d1_pos_b = [np.array([0.0, 4.0]) if i < 18 else np.array([0.0, 20.0]) for i in range(25)]

    res_b = evaluate_tti(
        timestamps=times_a,
        drone_positions={0: d0_pos_b, 1: d1_pos_b},
        target_positions=target_pos,
        holding_window_s=1.5,
        standoff_threshold_m=6.0,
        enclosure_threshold_deg=60.0,
    )
    assert res_b.interception_success
    assert res_b.tti_seconds == 0.0
    assert res_b.failure_reason == "SUCCESS"


def test_evaluate_coverage_and_t90():
    """Verify uncertainty reduction percentage and T90 calculation."""
    times = [0.0, 5.0, 10.0, 15.0, 20.0]
    # Drops from 80% to 7% (below 8% = T90 reached at t=15.0)
    u_hist = [80.0, 50.0, 25.0, 7.0, 5.0]

    cov = evaluate_coverage(u_hist, times, required_reduction_pct=75.0, required_t90_s=18.0)
    assert cov.initial_pct == 80.0
    assert cov.final_pct == 5.0
    assert cov.reduction_pct == 93.75
    assert cov.time_to_90pct_coverage_s == 15.0
    assert cov.passed_reduction is True
    assert cov.passed_t90 is True


def test_evaluate_requirement_boolean_evaluation():
    """Verify production evaluate_requirement produces strict PASS / FAIL."""
    r_pass = evaluate_requirement("Position RMSE", 0.45, 0.85, "<=")
    r_fail = evaluate_requirement("Position RMSE", 1.25, 0.85, "<=")

    assert r_pass.passed is True
    assert r_pass.status_label == "PASS"

    assert r_fail.passed is False
    assert r_fail.status_label == "FAIL"
