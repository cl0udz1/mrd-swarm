# -*- coding: utf-8 -*-
"""
controller.py — Authoritative Geometric SE(3) / SO(3) Flight Controller

Implements the single authoritative flight control pipeline:
    Goal / Setpoint (p_d, v_d, yaw_d)
    → Position Error & Desired Acceleration
    → SO(3) Attitude Construction (R_d)
    → Attitude Error & Torque Command (tau)
    → 4-Rotor Thrust Allocation (T_1, T_2, T_3, T_4)
    → Actuator Saturation Clamping

Consumes authoritative vehicle specifications from src.config.airframes.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
import numpy as np
from numpy.typing import NDArray

from .config.airframes import AirframeConfig, get_airframe_config
from .physics import (
    GRAVITY,
    quat_to_rotation_matrix,
    rotation_matrix_to_euler,
    solve_motor_thrusts,
)


@dataclass
class ControllerGains:
    """Tunable feedback gains for position and attitude loops."""
    kp_pos: float = 4.5       # Position error gain [s^-2]
    kv_vel: float = 3.2       # Velocity error gain [s^-1]
    kr_att: float = 8.0       # Attitude SO(3) gain [N·m/rad]
    kw_rate: float = 2.5      # Angular rate damping gain [N·m·s/rad]


@dataclass
class ControlOutput:
    """Complete output of the geometric control loop for a single time-step."""
    total_thrust_n: float
    torque_cmd_nm: NDArray[np.float64]
    motor_thrusts_n: NDArray[np.float64]
    r_desired: NDArray[np.float64]
    attitude_error_rad: float
    position_error_m: float
    actuator_saturated: bool


class GeometricSE3Controller:
    """
    Authoritative Non-Linear Geometric Controller on SE(3) × SO(3).
    Follows Lee, Leok, McClamroch (2010) formulation adapted for heterogeneous quadrotors.
    """

    def __init__(
        self,
        airframe: AirframeConfig,
        gains: Optional[ControllerGains] = None,
    ):
        self.airframe = airframe
        self.gains = gains or ControllerGains()

        # Telemetry metrics
        self.total_control_steps = 0
        self.saturation_count = 0
        self.max_body_rate_observed = 0.0

    def compute_control(
        self,
        pos_current: NDArray[np.float64],
        vel_current: NDArray[np.float64],
        quat_current: NDArray[np.float64],
        omega_current: NDArray[np.float64],
        pos_desired: NDArray[np.float64],
        vel_desired: NDArray[np.float64],
        yaw_desired: float = 0.0,
    ) -> ControlOutput:
        """
        Executes one full tick of the geometric cascaded controller.
        """
        self.total_control_steps += 1

        # ── 1. Translational Error Dynamics ────────────────────────────────────
        e_p = pos_desired - pos_current
        e_v = vel_desired - vel_current
        pos_err_norm = float(np.linalg.norm(e_p))

        # Desired acceleration: PD + Gravity Feedforward
        a_des = self.gains.kp_pos * e_p + self.gains.kv_vel * e_v + np.array([0.0, 0.0, GRAVITY])

        # Enforce physical bank angle limit from airframe config
        max_a_horiz = GRAVITY * math.tan(self.airframe.max_tilt_rad)
        norm_a_horiz = float(np.linalg.norm(a_des[:2]))
        if norm_a_horiz > max_a_horiz:
            a_des[:2] = (a_des[:2] / norm_a_horiz) * max_a_horiz

        # Desired force vector and commanded total thrust
        f_des = self.airframe.mass_kg * a_des
        f_norm = float(np.linalg.norm(f_des))
        total_thrust = float(np.clip(
            f_norm,
            0.1 * self.airframe.weight_n,
            self.airframe.max_total_thrust_n,
        ))

        # Desired body z-axis (thrust vector direction)
        b3_d = f_des / (f_norm + 1e-9)

        # ── 2. SO(3) Attitude Reference Construction ───────────────────────────
        b1_c = np.array([math.cos(yaw_desired), math.sin(yaw_desired), 0.0], dtype=np.float64)
        b2_d_raw = np.cross(b3_d, b1_c)
        b2_norm = float(np.linalg.norm(b2_d_raw))
        if b2_norm < 1e-4:
            b2_d = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        else:
            b2_d = b2_d_raw / b2_norm
        b1_d = np.cross(b2_d, b3_d)
        R_d = np.column_stack([b1_d, b2_d, b3_d])

        # ── 3. Attitude Error & Torques on SO(3) ───────────────────────────────
        R = quat_to_rotation_matrix(quat_current)

        # Skew-symmetric attitude error: e_R = 0.5 * (R_d^T R - R^T R_d)^vee
        e_R_skew = R_d.T @ R - R.T @ R_d
        e_R = 0.5 * np.array([e_R_skew[2, 1], e_R_skew[0, 2], e_R_skew[1, 0]], dtype=np.float64)
        att_err_norm = float(np.linalg.norm(e_R))

        # Angular velocity error (tracking hover/target body rates)
        e_omega = omega_current.copy()
        current_body_rate = float(np.linalg.norm(omega_current))
        if current_body_rate > self.max_body_rate_observed:
            self.max_body_rate_observed = current_body_rate

        # Commanded torque with gyroscopic cross-coupling compensation:
        # tau = -k_R * e_R - k_omega * e_omega + omega x (J omega)
        J = np.diag(self.airframe.inertia_matrix)
        gyroscopic = np.cross(omega_current, J * omega_current)
        tau_raw = -self.gains.kr_att * e_R - self.gains.kw_rate * e_omega + gyroscopic

        # Physical torque clamp based on motor thrust capability
        max_tau_xy = self.airframe.arm_length_m * self.airframe.max_thrust_per_motor_n * 2.0
        max_tau_z = (self.airframe.k_m / (self.airframe.k_f + 1e-9)) * self.airframe.max_thrust_per_motor_n * 4.0
        tau_cmd = np.array([
            np.clip(tau_raw[0], -max_tau_xy, max_tau_xy),
            np.clip(tau_raw[1], -max_tau_xy, max_tau_xy),
            np.clip(tau_raw[2], -max_tau_z, max_tau_z),
        ], dtype=np.float64)

        # ── 4. Rotor Allocation & Actuator Saturation ─────────────────────────
        motor_thrusts, is_saturated = solve_motor_thrusts(total_thrust, tau_cmd, self.airframe)
        if is_saturated:
            self.saturation_count += 1

        return ControlOutput(
            total_thrust_n=total_thrust,
            torque_cmd_nm=tau_cmd,
            motor_thrusts_n=motor_thrusts,
            r_desired=R_d,
            attitude_error_rad=att_err_norm,
            position_error_m=pos_err_norm,
            actuator_saturated=is_saturated,
        )

    @property
    def saturation_frequency_pct(self) -> float:
        """Percentage of control steps where actuator saturation occurred."""
        if self.total_control_steps == 0:
            return 0.0
        return (self.saturation_count / self.total_control_steps) * 100.0


class CascadedQuadrotorController:
    """Legacy compatibility wrapper routing to GeometricSE3Controller."""

    def __init__(self, mass: float = 0.5, gains: Optional[ControllerGains] = None):
        from .config.airframes import FLEET_CONFIGS
        best_cfg = FLEET_CONFIGS[0]
        for cfg in FLEET_CONFIGS.values():
            if abs(cfg.mass_kg - mass) < abs(best_cfg.mass_kg - mass):
                best_cfg = cfg
        self.controller = GeometricSE3Controller(airframe=best_cfg, gains=gains)

    def compute_control(self, *args, **kwargs):
        return self.controller.compute_control(*args, **kwargs)


class TrajectoryGenerator:
    """Polynomial trajectory generator for smooth flight setpoints."""
    def __init__(self):
        pass

