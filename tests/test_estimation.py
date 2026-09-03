# -*- coding: utf-8 -*-
"""
test_estimation.py — Automated Tests for Kalman Filter Target Tracking,
Track Lifecycle State Machine, Joseph Stabilization, and Estimation Error Metrics.
"""

import numpy as np
import pytest

from src.ecs.target_tracker import KalmanTargetTracker, TrackState


def test_tracker_lifecycle_transitions():
    """Verify state transitions UNINITIALIZED -> CONFIRMED -> PREDICTED -> LOST."""
    tracker = KalmanTargetTracker(n_targets=1)
    track = tracker.tracks[0]
    assert track.state == TrackState.UNINITIALIZED

    # Update with observation -> CONFIRMED
    tracker.update(0, np.array([5.0, 5.0]), np.eye(2) * 0.25)
    assert track.state == TrackState.CONFIRMED

    # Predict without update for 3.0s -> PREDICTED (threshold is 2.0s)
    tracker.predict(3.0)
    assert track.state == TrackState.PREDICTED

    # Predict further past 8.0s -> LOST (threshold is 8.0s)
    tracker.predict(6.0)
    assert track.state == TrackState.LOST


def test_tracker_convergence_on_noisy_trajectory():
    """Filter must converge and maintain tracking accuracy on noisy linear trajectory."""
    tracker = KalmanTargetTracker(n_targets=1, process_noise_accel=0.5, default_meas_noise=0.5)
    rng = np.random.RandomState(42)

    true_v = np.array([1.5, 0.8])  # 1.7 m/s ground target
    dt = 0.1
    n_steps = 100

    true_pos = np.array([0.0, 0.0])
    R_meas = np.eye(2) * (0.4 ** 2)

    for step in range(n_steps):
        true_pos = true_pos + true_v * dt
        tracker.predict(dt)

        # Add zero-mean Gaussian noise to observation
        noise = rng.normal(0.0, 0.4, size=2)
        z = true_pos + noise

        tracker.update(0, z, R_meas)
        tracker.record_ground_truth_for_evaluation(0, true_pos, true_v)

    track = tracker.tracks[0]
    assert track.state == TrackState.CONFIRMED

    # Final estimated position error must be small (< 0.5m)
    final_pos_err = np.linalg.norm(track.position - true_pos)
    assert final_pos_err < 0.5

    # Overall RMSE must be low (< 0.6m)
    assert track.position_rmse < 0.6
    # Estimated velocity should match true velocity
    assert np.allclose(track.velocity, true_v, atol=0.3)


def test_covariance_symmetry_and_positive_definiteness():
    """Joseph-stabilized covariance matrix P must remain symmetric and positive-definite."""
    tracker = KalmanTargetTracker(n_targets=1)
    tracker.update(0, np.array([2.0, 3.0]), np.eye(2) * 0.5)

    for _ in range(50):
        tracker.predict(0.1)
        tracker.update(0, np.array([2.0, 3.0]), np.eye(2) * 0.5)

        P = tracker.tracks[0].P
        # Check symmetry: P == P.T
        assert np.allclose(P, P.T, atol=1e-8)
        # Check positive-definiteness: all eigenvalues > 0
        eigvals = np.linalg.eigvals(P)
        assert np.all(eigvals > 0.0)
