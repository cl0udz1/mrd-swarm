# -*- coding: utf-8 -*-
"""
test_controller.py — Automated Tests for Geometric SE(3) Quadrotor Flight Control.
"""

import math
import numpy as np
import pytest

from src.config.airframes import get_airframe_config
from src.controller import GeometricSE3Controller, ControllerGains, ControlOutput
from src.physics import GRAVITY, euler_to_rotation_matrix, rotation_matrix_to_quat


def test_controller_hover_equilibrium():
    """At desired hover position, controller must produce zero torque and thrust = weight."""
    cfg = get_airframe_config(0)
    controller = GeometricSE3Controller(airframe=cfg)

    pos = np.array([0.0, 0.0, 5.0], dtype=np.float64)
    vel = np.zeros(3, dtype=np.float64)
    quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    omega = np.zeros(3, dtype=np.float64)

    out = controller.compute_control(
        pos_current=pos,
        vel_current=vel,
        quat_current=quat,
        omega_current=omega,
        pos_desired=pos,
        vel_desired=vel,
        yaw_desired=0.0,
    )

    # Commanded thrust should match hover weight
    expected_thrust = cfg.weight_n
    assert math.isclose(out.total_thrust_n, expected_thrust, rel_tol=1e-3)
    # Commanded body torques should be near zero
    assert np.allclose(out.torque_cmd_nm, 0.0, atol=1e-3)
    assert not out.actuator_saturated
    assert out.position_error_m == 0.0
    assert out.attitude_error_rad < 1e-4


def test_controller_step_position_command():
    """A positive X position step must command positive X tilt/force."""
    cfg = get_airframe_config(1)
    controller = GeometricSE3Controller(airframe=cfg)

    pos = np.array([0.0, 0.0, 2.0], dtype=np.float64)
    pos_desired = np.array([5.0, 0.0, 2.0], dtype=np.float64)
    vel = np.zeros(3, dtype=np.float64)
    quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    omega = np.zeros(3, dtype=np.float64)

    out = controller.compute_control(
        pos_current=pos,
        vel_current=vel,
        quat_current=quat,
        omega_current=omega,
        pos_desired=pos_desired,
        vel_desired=vel,
        yaw_desired=0.0,
    )

    assert out.position_error_m == 5.0
    # Pitch forward to accelerate in +X: desired body x component of z-axis should be positive
    b3_d = out.r_desired[:, 2]
    assert b3_d[0] > 0.1


def test_so3_attitude_error_monotonicity():
    """Attitude error e_R on SO(3) must increase monotonically with tilt angle."""
    cfg = get_airframe_config(2)
    controller = GeometricSE3Controller(airframe=cfg)

    pos = np.array([0.0, 0.0, 3.0])
    vel = np.zeros(3)
    omega = np.zeros(3)

    errors = []
    angles = [0.0, 0.1, 0.3, 0.6]
    for roll in angles:
        R = euler_to_rotation_matrix(roll, 0.0, 0.0)
        q = rotation_matrix_to_quat(R)
        out = controller.compute_control(
            pos_current=pos,
            vel_current=vel,
            quat_current=q,
            omega_current=omega,
            pos_desired=pos,
            vel_desired=vel,
            yaw_desired=0.0,
        )
        errors.append(out.attitude_error_rad)

    # Verify monotonic increase
    for i in range(len(errors) - 1):
        assert errors[i+1] > errors[i]


def test_actuator_saturation_metric():
    """Controller must properly register actuator saturation frequency."""
    cfg = get_airframe_config(3)
    controller = GeometricSE3Controller(airframe=cfg)

    # Hover step (not saturated)
    controller.compute_control(
        pos_current=np.array([0.0, 0.0, 1.0]),
        vel_current=np.zeros(3),
        quat_current=np.array([1.0, 0.0, 0.0, 0.0]),
        omega_current=np.zeros(3),
        pos_desired=np.array([0.0, 0.0, 1.0]),
        vel_desired=np.zeros(3),
    )
    assert controller.saturation_count == 0
    assert controller.saturation_frequency_pct == 0.0
