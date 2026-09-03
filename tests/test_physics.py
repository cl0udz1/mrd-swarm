# -*- coding: utf-8 -*-
"""
test_physics.py — Automated Tests for 6-DoF Physics, Coordinate Transforms,
Rotor Allocation, and Dryden Atmospheric Turbulence.
"""

import math
import numpy as np
import pytest

from src.config.airframes import FLEET_CONFIGS, AirframeConfig, DroneClass, get_airframe_config
from src.physics import (
    GRAVITY,
    quat_to_rotation_matrix,
    rotation_matrix_to_euler,
    euler_to_rotation_matrix,
    rotation_matrix_to_quat,
    build_allocation_matrix,
    solve_motor_thrusts,
    DrydenTurbulenceModel,
    SyntheticPeriodicWindDisturbance,
)


def test_airframe_configs_validity():
    """Verify all fleet airframes satisfy strict physical invariants."""
    assert len(FLEET_CONFIGS) == 4
    for did, cfg in FLEET_CONFIGS.items():
        assert cfg.mass_kg > 0.1
        assert cfg.arm_length_m > 0.05
        assert cfg.thrust_margin >= 2.0
        assert cfg.battery_capacity_wh >= 15.0
        assert cfg.max_speed_mps >= 5.0
        # Check inertia tensor positive-definiteness
        J = cfg.inertia_matrix
        eigvals = np.linalg.eigvals(J)
        assert np.all(eigvals > 0.0)


def test_quaternion_so3_transforms():
    """Verify quaternion <-> rotation matrix isomorphism and roundtrip precision."""
    angles = [
        (0.0, 0.0, 0.0),
        (0.2, -0.1, 0.5),
        (-0.4, 0.3, -1.2),
        (0.0, math.radians(45.0), 0.0),
    ]
    for roll, pitch, yaw in angles:
        R = euler_to_rotation_matrix(roll, pitch, yaw)
        # Check orthogonality: R @ R.T = I and det(R) = 1
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-6)

        q = rotation_matrix_to_quat(R)
        assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-6)

        R_rec = quat_to_rotation_matrix(q)
        assert np.allclose(R, R_rec, atol=1e-5)


def test_allocation_matrix_invertibility():
    """Verify rotor allocation mixer matrix B is non-singular for all airframes."""
    for did, cfg in FLEET_CONFIGS.items():
        c = cfg.k_m / (cfg.k_f + 1e-12)
        B = build_allocation_matrix(cfg.arm_length_m, c)
        assert B.shape == (4, 4)
        det = np.linalg.det(B)
        assert abs(det) > 1e-6

        # Hover equilibrium test
        hover_thrust = cfg.weight_n
        zero_torque = np.zeros(3)
        t_motors, saturated = solve_motor_thrusts(hover_thrust, zero_torque, cfg)
        assert not saturated
        # Equal motor thrust at hover
        assert np.allclose(t_motors, hover_thrust / 4.0, atol=1e-4)
        assert np.isclose(np.sum(t_motors), hover_thrust, atol=1e-4)


def test_actuator_saturation_clamping():
    """Verify motor thrust limits are strictly clamped when exceeding max thrust."""
    cfg = get_airframe_config(1)  # Fast Interceptor
    excessive_thrust = cfg.max_total_thrust_n * 2.5
    excessive_torque = np.array([2.0, 2.0, 2.0])

    t_motors, saturated = solve_motor_thrusts(excessive_thrust, excessive_torque, cfg)
    assert saturated
    assert np.all(t_motors <= cfg.max_thrust_per_motor_n + 1e-6)
    assert np.all(t_motors >= 0.0)


def test_dryden_turbulence_model():
    """Verify discrete Dryden gust model properties, stochasticity, and seed reproducibility."""
    dt = 0.01
    model_a1 = DrydenTurbulenceModel(dt=dt, altitude_m=10.0, wind_speed_20m=3.0, seed=42)
    model_a2 = DrydenTurbulenceModel(dt=dt, altitude_m=10.0, wind_speed_20m=3.0, seed=42)
    model_b = DrydenTurbulenceModel(dt=dt, altitude_m=10.0, wind_speed_20m=3.0, seed=99)

    gusts_a1 = [model_a1.step() for _ in range(200)]
    gusts_a2 = [model_a2.step() for _ in range(200)]
    gusts_b = [model_b.step() for _ in range(200)]

    # Seed reproducibility: identical seeds must yield identical trajectories
    for g1, g2 in zip(gusts_a1, gusts_a2):
        assert np.allclose(g1, g2, atol=1e-12)

    # Distinct seeds must differ
    diff = np.sum(np.abs(np.array(gusts_a1) - np.array(gusts_b)))
    assert diff > 10.0

    # Bounds: Low altitude gusts should be physically bounded (< 15 m/s)
    arr_a1 = np.array(gusts_a1)
    assert np.all(np.abs(arr_a1) < 15.0)


def test_synthetic_periodic_wind():
    """Verify deterministic periodic disturbance returns expected dimensions."""
    wind = SyntheticPeriodicWindDisturbance()
    w0 = wind.get_wind(0.0)
    assert w0.shape == (3,)
    w10 = wind.get_wind(10.0)
    assert not np.allclose(w0, w10)
