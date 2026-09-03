"""
controller.py — Quadrotor Trajectory Tracking Controller
for the MRD-Swarm reconnaissance quadrotor.

Architecture:
    Position PD → Desired Thrust + Tilt Commands
    Gyro Damping → Attitude Stability
    Motor Mixer → Normalized Actuator Commands

This controller uses a simplified approach that prioritizes stability
over aggressive maneuvering. It uses gyro damping for attitude stability
instead of a full attitude control loop.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Optional, Tuple

from .physics import (
    GRAVITY, DRONE_MASS, INERTIA_MATRIX,
    quat_to_rotation_matrix, rotation_matrix_to_euler,
    build_allocation_matrix, solve_motor_commands,
    MAX_THRUST_PER_MOTOR,
)


@dataclass
class PIDGains:
    """PID gain triplets for each control axis."""
    kp: float
    ki: float
    kd: float
    integral_limit: float = 5.0
    output_min: float = -np.inf
    output_max: float = np.inf


@dataclass
class ControllerState:
    """Mutable state for the controller."""
    pos_integral: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))
    time: float = 0.0


class CascadedQuadrotorController:
    """
    Quadrotor trajectory tracking controller.

    Uses position PD control with gyro damping for attitude stability.
    This approach is more stable than a full cascaded attitude controller
    for the MuJoCo simulation environment.
    """

    def __init__(
        self,
        mass: float = DRONE_MASS,
        gravity: float = GRAVITY,
        # Position PID gains (no integral - gyro damping handles steady-state)
        pos_xy_gains: PIDGains = PIDGains(kp=0.3, ki=0.0, kd=0.5, integral_limit=0.0),
        pos_z_gains: PIDGains = PIDGains(kp=0.5, ki=0.0, kd=0.8, integral_limit=0.0),
        # Gyro damping gains (scaled to hover control to prevent saturation)
        # hover_ctrl ≈ 0.768, so gains should be << 0.768 to avoid saturation
        kd_gyro_roll: float = 0.3,
        kd_gyro_pitch: float = 0.3,
        kd_gyro_yaw: float = 0.2,
        # Tilt limits
        max_tilt_command: float = 0.15,  # ~8.6 degrees
        # Thrust limits
        min_thrust: float = 0.5,
        max_thrust: float = 4 * MAX_THRUST_PER_MOTOR,
    ):
        self.mass = mass
        self.gravity = gravity
        self.weight = mass * gravity

        self.pos_xy_gains = pos_xy_gains
        self.pos_z_gains = pos_z_gains

        self.kd_gyro_roll = kd_gyro_roll
        self.kd_gyro_pitch = kd_gyro_pitch
        self.kd_gyro_yaw = kd_gyro_yaw

        self.max_tilt_command = max_tilt_command
        self.min_thrust = min_thrust
        self.max_thrust = max_thrust

        # Hover control per motor
        self.hover_ctrl = self.weight / (4.0 * MAX_THRUST_PER_MOTOR)

        # Pre-compute allocation matrix inverse
        self.A = build_allocation_matrix()
        self.A_inv = np.linalg.inv(self.A)

        # Controller state
        self.state = ControllerState()

    def reset(self) -> None:
        """Reset integrators and state."""
        self.state = ControllerState()

    def _pid_step(
        self,
        error: NDArray[np.float64],
        d_error: NDArray[np.float64],
        integral: NDArray[np.float64],
        gains: PIDGains,
        dt: float,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Single PID computation with anti-windup clamping.

        Returns (output, updated_integral).
        """
        # Update integral with clamping
        new_integral = integral + error * dt
        new_integral = np.clip(new_integral, -gains.integral_limit, gains.integral_limit)

        # Anti-windup: reset integral when error changes sign
        for i in range(len(error)):
            if abs(integral[i]) > 0.01 and np.sign(error[i]) != np.sign(integral[i]):
                new_integral[i] = 0.0

        output = gains.kp * error + gains.ki * new_integral + gains.kd * d_error
        output = np.clip(output, gains.output_min, gains.output_max)

        return output, new_integral

    def compute_control(
        self,
        position: NDArray[np.float64],
        velocity: NDArray[np.float64],
        quaternion: NDArray[np.float64],
        angular_velocity: NDArray[np.float64],
        target_position: NDArray[np.float64],
        target_velocity: NDArray[np.float64] | None = None,
        target_acceleration: NDArray[np.float64] | None = None,
        target_yaw: float = 0.0,
        dt: float = 0.001,
    ) -> Tuple[NDArray[np.float64], dict]:
        """
        Full control pipeline: position → thrust + tilt → motor commands.

        Parameters
        ----------
        position : (3,) world position [x, y, z]
        velocity : (3,) world velocity
        quaternion : (4,) body quaternion [w, x, y, z]
        angular_velocity : (3,) body angular velocity [p, q, r]
        target_position : (3,) desired position
        target_velocity : (3,) desired velocity feedforward (optional)
        target_acceleration : (3,) desired acceleration feedforward (optional)
        target_yaw : desired yaw angle (rad)
        dt : timestep

        Returns
        -------
        ctrl : (4,) normalized motor commands [u0, u1, u2, u3] ∈ [0,1]⁴
        info : dict with intermediate control signals for telemetry
        """
        pos = np.asarray(position, dtype=np.float64)
        vel = np.asarray(velocity, dtype=np.float64)
        quat = np.asarray(quaternion, dtype=np.float64)
        omega = np.asarray(angular_velocity, dtype=np.float64)
        target_pos = np.asarray(target_position, dtype=np.float64)
        target_vel = np.asarray(target_velocity, dtype=np.float64) if target_velocity is not None else np.zeros(3)
        target_acc = np.asarray(target_acceleration, dtype=np.float64) if target_acceleration is not None else np.zeros(3)

        # ── Position Control ─────────────────────────────────────────────────
        pos_error = target_pos - pos
        vel_error = target_vel - vel

        # XY position control
        acc_xy, self.state.pos_integral[:2] = self._pid_step(
            pos_error[:2], vel_error[:2],
            self.state.pos_integral[:2],
            self.pos_xy_gains, dt
        )

        # Z position control
        acc_z_arr, int_z_arr = self._pid_step(
            pos_error[2:3], vel_error[2:3],
            self.state.pos_integral[2:3],
            self.pos_z_gains, dt
        )
        self.state.pos_integral[2] = float(int_z_arr[0])

        # ── Thrust Computation ───────────────────────────────────────────────
        # Base thrust: hover + altitude correction
        thrust_z = self.hover_ctrl + float(acc_z_arr[0])

        # XY tilt commands (limited for stability)
        roll_cmd = np.clip(float(acc_xy[1]), -self.max_tilt_command, self.max_tilt_command)
        pitch_cmd = np.clip(-float(acc_xy[0]), -self.max_tilt_command, self.max_tilt_command)

        # ── Gyro Damping ─────────────────────────────────────────────────────
        # Use gyro data for attitude stability (prevents oscillations)
        roll_corr = roll_cmd - self.kd_gyro_roll * omega[0]
        pitch_corr = pitch_cmd - self.kd_gyro_pitch * omega[1]
        yaw_corr = -self.kd_gyro_yaw * omega[2]

        # ── Motor Mixing ─────────────────────────────────────────────────────
        # Mix thrust and attitude corrections into motor commands
        ctrl = np.array([
            thrust_z + roll_corr + pitch_corr + yaw_corr,
            thrust_z - roll_corr + pitch_corr - yaw_corr,
            thrust_z - roll_corr - pitch_corr + yaw_corr,
            thrust_z + roll_corr - pitch_corr - yaw_corr,
        ])
        ctrl = np.clip(ctrl, 0.0, 1.0)

        # Compute actual thrust for telemetry
        total_thrust = sum(ctrl) * MAX_THRUST_PER_MOTOR

        # Telemetry
        R = quat_to_rotation_matrix(quat)
        euler = rotation_matrix_to_euler(R)
        info = {
            "total_thrust": total_thrust,
            "moments": [roll_corr, pitch_corr, yaw_corr],
            "pos_error": pos_error.tolist(),
            "vel_error": vel_error.tolist(),
            "attitude_error_deg": [0.0, 0.0, 0.0],  # not directly controlled
            "euler_deg": np.degrees(euler).tolist(),
            "euler_desired_deg": [np.degrees(roll_cmd), np.degrees(pitch_cmd), 0.0],
            "tilt_angle_deg": np.degrees(np.sqrt(roll_cmd**2 + pitch_cmd**2)),
            "motor_commands": ctrl.tolist(),
        }

        self.state.time += dt
        return ctrl, info


class TrajectoryGenerator:
    """
    Generates smooth reference trajectories for waypoint following.
    Uses minimum-jerk (5th-order polynomial) interpolation between waypoints.
    """

    def __init__(self, waypoints: list[NDArray[np.float64]], speeds: list[float] | None = None):
        """
        Parameters
        ----------
        waypoints : list of (3,) position waypoints
        speeds : list of speeds between consecutive waypoints (m/s)
        """
        self.waypoints = [np.asarray(w, dtype=np.float64) for w in waypoints]
        self.n_segments = len(waypoints) - 1
        if speeds is None:
            speeds = [1.0] * self.n_segments
        self.speeds = speeds

        # Compute segment durations based on distance and speed
        self.durations = []
        for i in range(self.n_segments):
            dist = np.linalg.norm(self.waypoints[i+1] - self.waypoints[i])
            self.durations.append(max(dist / speeds[i], 0.1))

        self.segment_start_times = [0.0]
        for d in self.durations:
            self.segment_start_times.append(self.segment_start_times[-1] + d)
        self.total_time = self.segment_start_times[-1]

    def get_reference(self, t: float) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Get position, velocity, and acceleration reference at time t.

        Uses 5th-order minimum-jerk polynomial:
            s(τ) = 10τ³ - 15τ⁴ + 6τ⁵     (τ ∈ [0,1])
            ṡ(τ) = 30τ² - 60τ³ + 30τ⁴
            s̈(τ) = 60τ - 180τ² + 120τ³

        Returns
        -------
        pos_ref, vel_ref, acc_ref : each (3,)
        """
        t = np.clip(t, 0.0, self.total_time)

        # Find active segment
        seg = 0
        for i in range(self.n_segments):
            if t >= self.segment_start_times[i]:
                seg = i

        # Normalized time within segment
        T = self.durations[seg]
        tau = np.clip((t - self.segment_start_times[seg]) / T, 0.0, 1.0)

        # Minimum-jerk basis functions
        s = 10*tau**3 - 15*tau**4 + 6*tau**5
        ds = (30*tau**2 - 60*tau**3 + 30*tau**4) / T
        dds = (60*tau - 180*tau**2 + 120*tau**3) / (T**2)

        p0 = self.waypoints[seg]
        p1 = self.waypoints[seg + 1]
        dp = p1 - p0

        pos_ref = p0 + dp * s
        vel_ref = dp * ds
        acc_ref = dp * dds

        return pos_ref, vel_ref, acc_ref
