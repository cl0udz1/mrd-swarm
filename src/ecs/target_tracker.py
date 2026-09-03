# -*- coding: utf-8 -*-
"""
target_tracker.py — Discrete Linear Kalman Filter (KF) Target Track Persistence

Maintains state estimates for ground targets:
    State:       x = [px, py, vx, vy]^T  (2D constant-velocity kinematic model)
    Measurement: z = [px_meas, py_meas]^T (Noisy synthetic observation with covariance R)

Provides:
- Joseph-stabilized covariance update: P = (I - KH)P(I - KH)^T + KRK^T
- Track lifecycle: UNINITIALIZED → CONFIRMED → PREDICTED → LOST
- Online estimation quality tracking: Position RMSE, Velocity RMSE, and NEES against ground truth
"""

from __future__ import annotations
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Dict, Optional, Set, Tuple, List
import numpy as np
from numpy.typing import NDArray


class TrackState(IntEnum):
    UNINITIALIZED = 0
    CONFIRMED = 1     # Recent noisy measurement received within 2.0s
    PREDICTED = 2     # Running on Kalman dead-reckoning prediction (2.0s to 8.0s)
    LOST = 3          # Unobserved for > 8.0s; track is stale


@dataclass
class TargetTrack:
    """Belief state for an individual target maintained by the Kalman Filter."""
    target_id: int
    state: TrackState = TrackState.UNINITIALIZED

    # State vector [px, py, vx, vy]
    x: NDArray[np.float64] = field(default_factory=lambda: np.zeros(4, dtype=np.float64))
    # State covariance matrix P (4x4)
    P: NDArray[np.float64] = field(default_factory=lambda: np.eye(4, dtype=np.float64) * 25.0)

    # Timing and update telemetry
    time_since_update: float = 0.0
    total_track_time: float = 0.0
    n_updates: int = 0

    # Verification telemetry against ground truth (used strictly for evaluation)
    estimation_errors_pos: List[float] = field(default_factory=list)
    estimation_errors_vel: List[float] = field(default_factory=list)
    nees_history: List[float] = field(default_factory=list)

    # State thresholds
    PREDICTED_THRESHOLD: float = 2.0
    LOST_THRESHOLD: float = 8.0

    @property
    def position(self) -> NDArray[np.float64]:
        return self.x[:2].copy()

    @property
    def velocity(self) -> NDArray[np.float64]:
        return self.x[2:4].copy()

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.x[2:4]))

    @property
    def position_uncertainty_m(self) -> float:
        """RMS 1-sigma position uncertainty from covariance diagonal."""
        return float(np.sqrt(max(0.0, self.P[0, 0] + self.P[1, 1])))

    @property
    def position_rmse(self) -> float:
        """Root Mean Square Error against ground truth over track lifetime."""
        if not self.estimation_errors_pos:
            return 0.0
        return float(np.sqrt(np.mean(np.array(self.estimation_errors_pos)**2)))

    @property
    def velocity_rmse(self) -> float:
        """Velocity RMSE against ground truth."""
        if not self.estimation_errors_vel:
            return 0.0
        return float(np.sqrt(np.mean(np.array(self.estimation_errors_vel)**2)))


class KalmanTargetTracker:
    """
    Multi-target Linear Kalman Filter (KF) tracker.
    Operates strictly on noisy synthetic sensor observations (never ground truth).
    """

    def __init__(
        self,
        n_targets: int = 3,
        process_noise_accel: float = 0.8,     # continuous white noise spectral density q_w [m²/s³]
        default_meas_noise: float = 0.6,      # default sensor noise std [m]
    ):
        self.n_targets = n_targets
        self.q_w = process_noise_accel
        self.default_R = np.eye(2, dtype=np.float64) * (default_meas_noise ** 2)

        # Measurement matrix H: maps [px, py, vx, vy] -> [px, py]
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=np.float64)

        self.tracks: Dict[int, TargetTrack] = {
            tid: TargetTrack(target_id=tid) for tid in range(n_targets)
        }

    def _build_F(self, dt: float) -> NDArray[np.float64]:
        """State transition matrix F for constant-velocity kinematics."""
        return np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)

    def _build_Q(self, dt: float) -> NDArray[np.float64]:
        """Discrete process noise covariance Q for continuous white noise acceleration."""
        dt2 = dt * dt
        dt3 = dt2 * dt / 2.0
        dt4 = dt2 * dt2 / 4.0
        q = self.q_w
        return np.array([
            [q * dt4, 0.0,     q * dt3, 0.0    ],
            [0.0,     q * dt4, 0.0,     q * dt3],
            [q * dt3, 0.0,     q * dt2, 0.0    ],
            [0.0,     q * dt3, 0.0,     q * dt2],
        ], dtype=np.float64)

    def predict(self, dt: float) -> None:
        """
        Advance Kalman Filter state prediction for all initialized tracks.
        """
        F = self._build_F(dt)
        Q = self._build_Q(dt)

        for tid, track in self.tracks.items():
            if track.state == TrackState.UNINITIALIZED:
                continue

            track.x = F @ track.x
            track.P = F @ track.P @ F.T + Q

            track.time_since_update += dt
            track.total_track_time += dt

            # Update lifecycle state
            if track.time_since_update > track.LOST_THRESHOLD:
                track.state = TrackState.LOST
            elif track.time_since_update > track.PREDICTED_THRESHOLD:
                track.state = TrackState.PREDICTED

    def update(
        self,
        target_id: int,
        measured_pos_2d: NDArray[np.float64],
        covariance_r: Optional[NDArray[np.float64]] = None,
    ) -> None:
        """
        Incorporate noisy synthetic sensor observation z = [x_meas, y_meas]^T.
        Uses Joseph-stabilized covariance update to guarantee positive semi-definiteness.
        """
        if target_id not in self.tracks:
            self.tracks[target_id] = TargetTrack(target_id=target_id)

        track = self.tracks[target_id]
        z = np.asarray(measured_pos_2d, dtype=np.float64)[:2]
        R = covariance_r if covariance_r is not None else self.default_R

        if track.state == TrackState.UNINITIALIZED:
            # First initialization from observation
            track.x = np.array([z[0], z[1], 0.0, 0.0], dtype=np.float64)
            track.P = np.eye(4, dtype=np.float64) * 10.0
            track.P[0, 0] = R[0, 0]
            track.P[1, 1] = R[1, 1]
            track.state = TrackState.CONFIRMED
            track.time_since_update = 0.0
            track.n_updates = 1
            return

        # Kalman Innovation
        y = z - self.H @ track.x
        S = self.H @ track.P @ self.H.T + R
        S_inv = np.linalg.inv(S)
        K = track.P @ self.H.T @ S_inv

        # State correction
        track.x = track.x + K @ y

        # Joseph form: P = (I - KH) P (I - KH)^T + K R K^T
        I4 = np.eye(4, dtype=np.float64)
        I_KH = I4 - K @ self.H
        track.P = I_KH @ track.P @ I_KH.T + K @ R @ K.T

        track.state = TrackState.CONFIRMED
        track.time_since_update = 0.0
        track.n_updates += 1

    def record_ground_truth_for_evaluation(
        self,
        target_id: int,
        true_pos_2d: NDArray[np.float64],
        true_vel_2d: NDArray[np.float64],
    ) -> Optional[Tuple[float, float, float]]:
        """
        Strictly for benchmark evaluation. Computes (pos_err, vel_err, nees).
        NEVER affects the internal state of the filter.
        """
        if target_id not in self.tracks or self.tracks[target_id].state == TrackState.UNINITIALIZED:
            return None

        track = self.tracks[target_id]
        x_true = np.array([true_pos_2d[0], true_pos_2d[1], true_vel_2d[0], true_vel_2d[1]], dtype=np.float64)
        err = x_true - track.x

        pos_err = float(np.linalg.norm(err[:2]))
        vel_err = float(np.linalg.norm(err[2:4]))

        try:
            P_inv = np.linalg.inv(track.P)
            nees = float(err.T @ P_inv @ err)
        except np.linalg.LinAlgError:
            nees = 0.0

        track.estimation_errors_pos.append(pos_err)
        track.estimation_errors_vel.append(vel_err)
        track.nees_history.append(nees)

        return pos_err, vel_err, nees

    def get_confirmed_ids(self) -> Set[int]:
        return {tid for tid, tr in self.tracks.items() if tr.state == TrackState.CONFIRMED}

    def get_tracked_ids(self) -> Set[int]:
        """All active tracks (CONFIRMED or PREDICTED)."""
        return {
            tid for tid, tr in self.tracks.items()
            if tr.state in (TrackState.CONFIRMED, TrackState.PREDICTED)
        }

    def get_lost_ids(self) -> Set[int]:
        return {tid for tid, tr in self.tracks.items() if tr.state == TrackState.LOST}

    def get_predicted_position(self, target_id: int) -> Optional[NDArray[np.float64]]:
        if target_id in self.tracks and self.tracks[target_id].state != TrackState.UNINITIALIZED:
            return self.tracks[target_id].position
        return None

    def get_predicted_velocity(self, target_id: int) -> Optional[NDArray[np.float64]]:
        if target_id in self.tracks and self.tracks[target_id].state != TrackState.UNINITIALIZED:
            return self.tracks[target_id].velocity
        return None

    def get_escape_radius(self, target_id: int) -> float:
        """Maximum possible distance target could have reached since last sighting."""
        if target_id not in self.tracks:
            return 5.0
        tr = self.tracks[target_id]
        spd = max(1.5, tr.speed)
        return float(min(16.0, max(3.0, spd * tr.time_since_update)))

    def get_telemetry(self) -> Dict[int, Dict[str, Any]]:
        """Structured telemetry dictionary for logging and visualizer HUD."""
        return {
            tid: {
                "state": tr.state.name,
                "position": tr.position.tolist(),
                "velocity": tr.velocity.tolist(),
                "speed": round(tr.speed, 2),
                "uncertainty_m": round(tr.position_uncertainty_m, 2),
                "time_since_update": round(tr.time_since_update, 2),
                "n_updates": tr.n_updates,
            }
            for tid, tr in self.tracks.items()
            if tr.state != TrackState.UNINITIALIZED
        }


# Backwards-compatible alias for existing codebase imports
EKFTargetTracker = KalmanTargetTracker
