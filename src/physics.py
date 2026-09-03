# -*- coding: utf-8 -*-
"""
physics.py — Physical Constants, Coordinate Transforms, Rotor Allocation,
and Dryden Atmospheric Turbulence for the MRD-Swarm Platform.

All units are SI: meters, kilograms, seconds, radians, Newtons.
Consumes authoritative vehicle specifications from src.config.airframes.
"""

from __future__ import annotations
import math
from typing import Tuple, Optional
import numpy as np
from numpy.typing import NDArray

from .config.airframes import FLEET_CONFIGS, AirframeConfig, get_airframe_config

# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL PHYSICAL CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

GRAVITY: float = 9.80665                  # Standard gravity [m/s²]
AIR_DENSITY: float = 1.225                 # Sea level standard air density [kg/m³]
DYNAMIC_VISCOSITY: float = 1.81e-5         # Air dynamic viscosity [Pa·s]


# ═══════════════════════════════════════════════════════════════════════════════
# COORDINATE TRANSFORMS ON SO(3) AND QUATERNIONS
# ═══════════════════════════════════════════════════════════════════════════════

def quat_to_rotation_matrix(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Convert unit quaternion [w, x, y, z] to 3x3 rotation matrix R in SO(3).
    R maps vectors from body frame to world frame: v_world = R @ v_body.
    """
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = q / norm

    return np.array([
        [1.0 - 2.0*(y*y + z*z),       2.0*(x*y - w*z),       2.0*(x*z + w*y)],
        [      2.0*(x*y + w*z), 1.0 - 2.0*(x*x + z*z),       2.0*(y*z - w*x)],
        [      2.0*(x*z - w*y),       2.0*(y*z + w*x), 1.0 - 2.0*(x*x + y*y)]
    ], dtype=np.float64)


def rotation_matrix_to_euler(R: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Extract Euler angles [roll, pitch, yaw] from SO(3) rotation matrix using Z-Y-X (yaw-pitch-roll) convention.
    """
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0

    return np.array([roll, pitch, yaw], dtype=np.float64)


def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> NDArray[np.float64]:
    """Construct SO(3) rotation matrix from [roll, pitch, yaw] Euler angles."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    R_x = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    R_y = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    R_z = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)

    return R_z @ R_y @ R_x


def rotation_matrix_to_quat(R: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert 3x3 rotation matrix R in SO(3) to unit quaternion [w, x, y, z]."""
    tr = np.trace(R)
    if tr > 0.0:
        S = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S

    q = np.array([w, x, y, z], dtype=np.float64)
    q /= np.linalg.norm(q)
    if q[0] < 0.0:
        q = -q
    return q


# ═══════════════════════════════════════════════════════════════════════════════
# ROTOR ALLOCATION & ACTUATOR MIXING (X-Configuration Quadrotor)
# ═══════════════════════════════════════════════════════════════════════════════

def build_allocation_matrix(arm_length: float, k_m_over_k_f: float) -> NDArray[np.float64]:
    """
    Construct 4x4 mixer allocation matrix B mapping motor thrusts [T1, T2, T3, T4]^T
    to total thrust and body torques [T_total, tau_x, tau_y, tau_z]^T.

    Layout (X-frame):
      Motor 1 (Front-Right, CCW): (+d, +d)
      Motor 2 (Rear-Left,   CCW): (-d, -d)
      Motor 3 (Front-Left,  CW):  (+d, -d)
      Motor 4 (Rear-Right,  CW):  (-d, +d)
      where d = arm_length / sqrt(2)
    """
    d = arm_length / math.sqrt(2.0)
    c = k_m_over_k_f  # torque to thrust ratio [m]

    return np.array([
        [ 1.0,  1.0,  1.0,  1.0],  # Total thrust
        [-d,    d,    d,   -d  ],  # Roll torque tau_x
        [-d,    d,   -d,    d  ],  # Pitch torque tau_y
        [-c,   -c,    c,    c  ],  # Yaw torque tau_z
    ], dtype=np.float64)


def solve_motor_thrusts(
    total_thrust_cmd: float,
    torque_cmd: NDArray[np.float64],
    airframe: AirframeConfig,
) -> Tuple[NDArray[np.float64], bool]:
    """
    Solve for individual motor thrusts T_i in [0, T_max] from commanded wrench [T, tau_x, tau_y, tau_z].
    Returns (motor_thrusts, was_saturated).
    """
    c = airframe.k_m / (airframe.k_f + 1e-12)
    B = build_allocation_matrix(airframe.arm_length_m, c)
    B_inv = np.linalg.inv(B)

    wrench = np.array([
        total_thrust_cmd,
        torque_cmd[0],
        torque_cmd[1],
        torque_cmd[2],
    ], dtype=np.float64)

    t_motors_raw = B_inv @ wrench
    t_motors = np.clip(t_motors_raw, 0.0, airframe.max_thrust_per_motor_n)
    saturated = bool(np.any(t_motors != t_motors_raw))
    return t_motors, saturated


# ═══════════════════════════════════════════════════════════════════════════════
# DRYDEN ATMOSPHERIC TURBULENCE MODEL (MIL-F-8785C)
# ═══════════════════════════════════════════════════════════════════════════════

class DrydenTurbulenceModel:
    """
    Discrete stochastic Dryden wind gust model per MIL-F-8785C low-altitude specification.
    Produces colored noise gusts [u_g, v_g, w_g] driven by continuous spectral shaping filters
    excited by discrete Gaussian white noise N(0, 1) with strict seed reproducibility.
    """

    def __init__(
        self,
        dt: float = 0.01,
        altitude_m: float = 10.0,
        wind_speed_20m: float = 3.0,
        seed: Optional[int] = 42,
    ):
        self.dt = dt
        self.altitude_m = max(1.0, altitude_m)
        self.rng = np.random.RandomState(seed)

        # MIL-F-8785C Low-altitude scale lengths (h < 300m)
        h = self.altitude_m
        denom = (0.177 + 0.000823 * h) ** 1.2
        self.L_w = h
        self.L_u = h / denom
        self.L_v = self.L_u

        # Turbulence intensities [m/s]
        self.sigma_w = 0.1 * wind_speed_20m
        self.sigma_u = self.sigma_w / ((0.177 + 0.000823 * h) ** 0.4)
        self.sigma_v = self.sigma_u

        # Airspeed nominal operating point (hover/cruise)
        self.V = 5.0  # m/s nominal relative speed

        # Filter internal states (discrete 1st/2nd order ARMA)
        self.state_u = 0.0
        self.state_v = np.zeros(2, dtype=np.float64)
        self.state_w = np.zeros(2, dtype=np.float64)

        # Precompute discretization poles
        self.alpha_u = math.exp(-self.V / self.L_u * dt)
        self.alpha_v = math.exp(-self.V / self.L_v * dt)
        self.alpha_w = math.exp(-self.V / self.L_w * dt)

    def step(self) -> NDArray[np.float64]:
        """Advance the stochastic Dryden gust generator by dt and return [u_g, v_g, w_g] in m/s."""
        w_in = self.rng.randn(3)  # Unit variance white noise

        # Longitudinal u_g (1st order filter)
        gain_u = self.sigma_u * math.sqrt(2.0 * self.L_u / (math.pi * self.V)) * math.sqrt(1.0 - self.alpha_u**2)
        self.state_u = self.alpha_u * self.state_u + gain_u * w_in[0]
        u_g = self.state_u

        # Lateral v_g (approximate 1st-order shaping filter)
        gain_v = self.sigma_v * math.sqrt(self.L_v / (math.pi * self.V)) * math.sqrt(1.0 - self.alpha_v**2)
        self.state_v[0] = self.alpha_v * self.state_v[0] + gain_v * w_in[1]
        v_g = self.state_v[0]

        # Vertical w_g
        gain_w = self.sigma_w * math.sqrt(self.L_w / (math.pi * self.V)) * math.sqrt(1.0 - self.alpha_w**2)
        self.state_w[0] = self.alpha_w * self.state_w[0] + gain_w * w_in[2]
        w_g = self.state_w[0]

        return np.array([u_g, v_g, w_g], dtype=np.float64)


class SyntheticPeriodicWindDisturbance:
    """
    Deterministic periodic wind disturbance for exact regression testing.
    Explicitly labeled as synthetic periodic disturbance (NOT stochastic Dryden).
    """

    def __init__(self, amplitude: Tuple[float, float, float] = (1.2, 1.0, 0.2)):
        self.amp = amplitude

    def get_wind(self, t: float) -> NDArray[np.float64]:
        return np.array([
            self.amp[0] * math.sin(0.4 * t) + 0.3 * math.sin(1.1 * t),
            self.amp[1] * math.cos(0.5 * t) + 0.2 * math.cos(1.0 * t),
            self.amp[2] * math.sin(0.8 * t),
        ], dtype=np.float64)
