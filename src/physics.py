"""
physics.py — Physical constants, coordinate transforms, and dynamics utilities
for the MRD-Swarm reconnaissance quadrotor platform.

All units are SI: meters, kilograms, seconds, radians, Newtons.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Tuple

# ═══════════════════════════════════════════════════════════════════════════════
# PHYSICAL CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

GRAVITY: float = 9.81                    # m/s²
AIR_DENSITY: float = 1.225               # kg/m³ at sea level
DYNAMIC_VISCOSITY: float = 1.8e-5        # Pa·s

# ── Drone Platform (Crazyflie 2.0-class) ──────────────────────────────────────
DRONE_MASS: float = 0.47                 # kg (chassis 0.35 + arms 0.06 + motors 0.048 + props 0.012)
DRONE_ARM_LENGTH: float = 0.085          # m (motor-to-center distance)
DRONE_HEIGHT: float = 0.05               # m (chassis height above ground)

# Moments of inertia (kg·m²) — estimated from CAD mass distribution
INERTIA_IXX: float = 1.1e-3
INERTIA_IYY: float = 1.1e-3
INERTIA_IZZ: float = 2.1e-3
INERTIA_MATRIX: NDArray[np.float64] = np.diag([INERTIA_IXX, INERTIA_IYY, INERTIA_IZZ])

# ── Rotor / Propeller Model ───────────────────────────────────────────────────
# Thrust coefficient: F_i = K_F * omega_i²
# Torque coefficient: τ_z,i = K_M * omega_i²  (sign alternates CW/CCW)
K_F: float = 1.5e-4                      # N / (rad/s)²
K_M: float = 2.0e-5                      # N·m / (rad/s)²  (= K_F * C_M/C_F ratio)
OMEGA_MAX: float = 2500.0                 # rad/s (max rotor speed)
THRUST_MAX_PER_MOTOR: float = K_F * OMEGA_MAX**2  # ≈ 93.75 N... too high
# Actuator gainprm = 1.5 N per unit ctrl in MJCF
# Total mass = 0.47 kg → weight = 4.61 N → hover ctrl ≈ 0.77
# Max thrust per motor = 1.5 N (ctrl=1.0)
MAX_THRUST_PER_MOTOR: float = 1.5        # N (matches MJCF gainprm)
MOMENT_ARM: float = DRONE_ARM_LENGTH     # m

# ── Aerodynamic Drag ──────────────────────────────────────────────────────────
# Translational drag: F_drag = -0.5 * rho * C_D * A * |v| * v
DRAG_COEFF_TRANSLATIONAL: float = 0.47   # sphere approximation
DRONE_CROSS_SECTION: float = 0.015       # m² (effective frontal area)
DRAG_TRANSLATIONAL: float = 0.5 * AIR_DENSITY * DRAG_COEFF_TRANSLATIONAL * DRONE_CROSS_SECTION

# Rotational drag torque: τ_drag = -C_rot * ω
DRAG_COEFF_ROTATIONAL: float = 1.0e-4    # N·m·s/rad

# Rotor translational drag (each rotor creates horizontal drag force)
ROTOR_DRAG_COEFF: float = 8.0e-6         # N/(rad/s)² — horizontal drag from rotor wash

# ── Ground Effect ─────────────────────────────────────────────────────────────
# Approximate ground effect: thrust augmentation when z < ~2*D (prop diameter)
PROP_DIAMETER: float = 0.12              # m
GROUND_EFFECT_HEIGHT: float = 2.0 * PROP_DIAMETER  # 0.24 m
GROUND_EFFECT_GAIN: float = 0.15         # max 15% thrust increase at z=0

# ── Battery / Power Model ─────────────────────────────────────────────────────
BATTERY_CAPACITY_WH: float = 4.5         # Wh (typical nano-quad LiPo)
BATTERY_VOLTAGE_NOMINAL: float = 3.7     # V
BATTERY_CAPACITY_MAH: float = BATTERY_CAPACITY_WH * 1000 / BATTERY_VOLTAGE_NOMINAL
P_AVIONICS: float = 0.5                  # W (flight controller, radio)
P_SENSOR_PAYLOAD: float = 0.3            # W (camera, IMU, rangefinders)
P_HOVER_BASE: float = 5.0               # W (hover power draw)
MOTOR_EFFICIENCY: float = 0.7            # propulsive efficiency

# ── Sensor Noise Parameters ───────────────────────────────────────────────────
ACCEL_NOISE_STD: float = 0.05            # m/s² (σ for Gaussian noise)
ACCEL_BIAS_DRIFT: float = 0.001          # m/s² per step (random walk)
GYRO_NOISE_STD: float = 0.01             # rad/s
GYRO_BIAS_DRIFT: float = 0.0005          # rad/s per step
RANGEFINDER_NOISE_STD: float = 0.02      # m
RANGEFINDER_DROPOUT_PROB: float = 0.02   # probability of NaN reading
RANGEFINDER_MAX_RANGE: float = 10.0      # m
CAMERA_LATENCY_STEPS: int = 2            # frames of latency


# ═══════════════════════════════════════════════════════════════════════════════
# COORDINATE TRANSFORM UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def quat_to_rotation_matrix(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Convert unit quaternion [w, x, y, z] to 3×3 rotation matrix R ∈ SO(3).

    Uses the Rodrigues formula for numerical stability:
        R = I + 2w(v×) + 2(v×)²
    where v = [x, y, z] and (v×) is the skew-symmetric cross-product matrix.

    Parameters
    ----------
    q : array, shape (4,)
        Unit quaternion [w, x, y, z].

    Returns
    -------
    R : array, shape (3, 3)
        Rotation matrix mapping body-frame vectors to world frame.
    """
    q = np.asarray(q, dtype=np.float64)
    w, x, y, z = q[0], q[1], q[2], q[3]
    # Normalize to guard against drift
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm

    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ], dtype=np.float64)
    return R


def rotation_matrix_to_euler(R: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Extract ZYX Euler angles [roll, pitch, yaw] from rotation matrix.

    Parameters
    ----------
    R : array, shape (3, 3)

    Returns
    -------
    euler : array, shape (3,) — [roll, pitch, yaw] in radians
    """
    pitch = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
    if np.abs(np.cos(pitch)) > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[0, 1], R[1, 1])
        yaw = 0.0
    return np.array([roll, pitch, yaw])


def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> NDArray[np.float64]:
    """ZYX Euler angles to rotation matrix."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    R = np.array([
        [cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
        [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
        [  -sp,            cp*sr,            cp*cr   ]
    ], dtype=np.float64)
    return R


def skew_symmetric(v: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Skew-symmetric (cross-product) matrix from 3-vector.
    v× such that v× @ u = v × u.
    """
    return np.array([
        [  0,   -v[2],  v[1]],
        [ v[2],   0,   -v[0]],
        [-v[1],  v[0],   0  ]
    ], dtype=np.float64)


def quaternion_multiply(q1: NDArray[np.float64], q2: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Hamilton quaternion product q1 ⊗ q2.
    Convention: q = [w, x, y, z].
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=np.float64)


def quaternion_conjugate(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Conjugate (inverse for unit quaternions)."""
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quaternion_error(q_desired: NDArray[np.float64], q_current: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Attitude error quaternion: q_err = q_desired ⊗ q_current^{-1}.
    Returns the vector part [x, y, z] which approximates the rotation error
    for small angles (used as attitude error signal in the controller).
    """
    q_err = quaternion_multiply(q_desired, quaternion_conjugate(q_current))
    # Ensure shortest path (flip if w < 0)
    if q_err[0] < 0:
        q_err = -q_err
    return q_err[1:4]  # vector part only


def thrust_to_normalized(thrust_n: float) -> float:
    """Convert thrust in Newtons to normalized control [0, 1]."""
    return np.clip(thrust_n / MAX_THRUST_PER_MOTOR, 0.0, 1.0)


def normalized_to_thrust(u: float) -> float:
    """Convert normalized control [0, 1] to thrust in Newtons."""
    return u * MAX_THRUST_PER_MOTOR


# ═══════════════════════════════════════════════════════════════════════════════
# THRUST ALLOCATION MATRIX
# ═══════════════════════════════════════════════════════════════════════════════

def build_allocation_matrix(
    arm_length: float = MOMENT_ARM,
    k_f: float = K_F,
    k_m: float = K_M,
) -> NDArray[np.float64]:
    """
    Build the 4×4 thrust-to-wrench allocation matrix A such that:

        [T, τ_x, τ_y, τ_z]^T = A @ [u_0, u_1, u_2, u_3]^T

    where u_i ∈ [0, 1] are normalized motor commands.

    X-configuration layout (top view):
        Motor 0 (FR): +x +y, CW  → +T, +τ_x, +τ_y, -τ_z
        Motor 1 (FL): +x -y, CCW → +T, +τ_x, -τ_y, +τ_z
        Motor 2 (RL): -x -y, CW  → +T, -τ_x, -τ_y, -τ_z
        Motor 3 (RR): -x +y, CCW → +T, -τ_x, +τ_y, +τ_z

    Returns
    -------
    A : array, shape (4, 4)
        Allocation matrix. Row 0 = thrust, rows 1-3 = roll/pitch/yaw moments.
    """
    d = arm_length
    # Normalized thrust coefficient (thrust per unit control input)
    c_t = MAX_THRUST_PER_MOTOR  # N per motor at u=1
    # Moment per unit control: torque_arm * thrust + yaw reaction torque
    c_m = c_t  # thrust moment arm contribution
    c_yaw = K_M / K_F * c_t    # yaw reaction torque (ratio of torque to thrust coeff)

    # Motor positions relative to CG in body frame (x-forward, y-left)
    # FR: (+d/√2, +d/√2), FL: (+d/√2, -d/√2), RL: (-d/√2, -d/√2), RR: (-d/√2, +d/√2)
    d_eff = d / np.sqrt(2.0)

    A = np.array([
        #  T (thrust along body z)
        [ c_t,    c_t,    c_t,    c_t  ],
        # τ_x (roll): y-position * thrust
        [ c_m * d_eff,  -c_m * d_eff,  -c_m * d_eff,   c_m * d_eff],
        # τ_y (pitch): x-position * thrust
        [ c_m * d_eff,   c_m * d_eff,  -c_m * d_eff,  -c_m * d_eff],
        # τ_z (yaw): CW/CCW alternation
        [-c_yaw,  c_yaw, -c_yaw,  c_yaw],
    ], dtype=np.float64)

    return A


def solve_motor_commands(
    wrench: NDArray[np.float64],
    A_inv: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """
    Solve for normalized motor commands u ∈ [0,1]⁴ from desired wrench [T, τx, τy, τz].

    Parameters
    ----------
    wrench : array, shape (4,)
        Desired [total_thrust, roll_moment, pitch_moment, yaw_moment].
    A_inv : array, shape (4, 4), optional
        Pre-computed inverse allocation matrix. Computed if not provided.

    Returns
    -------
    u : array, shape (4,)
        Clamped normalized motor commands in [0, 1].
    """
    if A_inv is None:
        A = build_allocation_matrix()
        A_inv = np.linalg.inv(A)

    u = A_inv @ wrench
    return np.clip(u, 0.0, 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# AERODYNAMIC EFFECTS
# ═══════════════════════════════════════════════════════════════════════════════

def translational_drag_force(velocity: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Quadratic translational drag: F = -0.5 * rho * C_D * A * |v| * v

    Parameters
    ----------
    velocity : array, shape (3,) — world-frame velocity

    Returns
    -------
    F_drag : array, shape (3,)
    """
    v = np.asarray(velocity, dtype=np.float64)
    speed = np.linalg.norm(v)
    if speed < 1e-6:
        return np.zeros(3)
    return -DRAG_TRANSLATIONAL * speed * v


def rotational_drag_torque(omega: NDArray[np.float64]) -> NDArray[np.float64]:
    """Linear rotational drag: τ = -C_rot * ω"""
    return -DRAG_COEFF_ROTATIONAL * np.asarray(omega, dtype=np.float64)


def ground_effect_factor(height: float) -> float:
    """
    Ground effect thrust augmentation factor.

    Model: f_ge = 1 + k * max(0, 1 - (z / h_ge)^2)
    where h_ge = 2 * prop_diameter, k = GROUND_EFFECT_GAIN.

    Returns multiplicative factor ≥ 1.0 when close to ground.
    """
    z = max(height, 0.0)
    if z >= GROUND_EFFECT_HEIGHT:
        return 1.0
    ratio = z / GROUND_EFFECT_HEIGHT
    return 1.0 + GROUND_EFFECT_GAIN * (1.0 - ratio * ratio)


def downwash_force(
    drone_pos: NDArray[np.float64],
    other_drone_pos: NDArray[np.float64],
    other_thrust: float,
) -> NDArray[np.float64]:
    """
    Approximate downwash force from another drone.

    Model: A column of air pushed downward below each drone.
    Force on the affected drone is primarily vertical (downward) when
    within the downwash column, with Gaussian lateral spread.

    Parameters
    ----------
    drone_pos : position of the affected drone
    other_drone_pos : position of the thrust-generating drone
    other_thrust : total thrust of the generating drone

    Returns
    -------
    F_downwash : array, shape (3,) — force vector (typically negative z)
    """
    delta = drone_pos - other_drone_pos
    dz = delta[2]  # positive if affected drone is above
    if dz <= 0 or dz > 3.0:
        return np.zeros(3)

    # Lateral distance
    d_lateral = np.sqrt(delta[0]**2 + delta[1]**2)
    # Downwash column radius increases with distance below
    column_radius = 0.15 + 0.3 * dz  # spreading angle ~17°
    if d_lateral > column_radius:
        return np.zeros(3)

    # Gaussian lateral falloff
    lateral_factor = np.exp(-0.5 * (d_lateral / (column_radius * 0.4))**2)
    # Vertical falloff with distance
    vertical_factor = 1.0 / (1.0 + dz * dz)

    # Downwash force magnitude (fraction of other drone's thrust)
    magnitude = 0.05 * other_thrust * lateral_factor * vertical_factor
    return np.array([0.0, 0.0, -magnitude])


def compute_power_consumption(
    thrust_ratio: float,
    p_hover: float = P_HOVER_BASE,
    p_avionics: float = P_AVIONICS,
    p_payload: float = P_SENSOR_PAYLOAD,
) -> float:
    """
    Total power consumption model:
        P_total = P_hover * (T / mg)^{3/2} + P_avionics + P_payload

    The 3/2 exponent comes from the relationship between thrust and
    induced power in momentum theory: P_induced ∝ T^{3/2}.

    Parameters
    ----------
    thrust_ratio : T / (m*g), normalized thrust (1.0 = hover)

    Returns
    -------
    P_total : Watts
    """
    return p_hover * (max(thrust_ratio, 0.0) ** 1.5) + p_avionics + p_payload
