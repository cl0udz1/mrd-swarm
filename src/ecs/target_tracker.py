# -*- coding: utf-8 -*-
"""
target_tracker.py — Extended Kalman Filter (EKF) Target Track Persistence

Maintains predicted tracks for ground targets even when Line-of-Sight (LOS)
is temporarily broken (e.g., target passes behind buildings).

State vector: x = [px, py, vx, vy]^T  (constant-velocity model)
Measurement:  z = [px, py]^T           (position from sensor)

Track states:
    CONFIRMED  — recent measurement update (< 2s ago)
    PREDICTED  — no update for 2–8s, running on prediction only
    LOST       — no update for > 8s, track is stale
"""

from __future__ import annotations
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Dict, Optional, Set, Tuple
import numpy as np


class TrackState(IntEnum):
    UNINITIALIZED = 0
    CONFIRMED = 1
    PREDICTED = 2
    LOST = 3


@dataclass
class TargetTrack:
    """Single target track maintained by the EKF."""
    target_id: int
    state: TrackState = TrackState.UNINITIALIZED

    # EKF state vector [px, py, vx, vy]
    x: np.ndarray = field(default_factory=lambda: np.zeros(4))
    # Covariance matrix P (4x4)
    P: np.ndarray = field(default_factory=lambda: np.eye(4) * 100.0)

    # Timing
    time_since_update: float = 0.0     # seconds since last measurement
    total_track_time: float = 0.0      # total time this track has existed
    n_updates: int = 0                 # number of measurement updates received

    # Thresholds
    PREDICTED_THRESHOLD: float = 2.0   # seconds without update → PREDICTED
    LOST_THRESHOLD: float = 8.0        # seconds without update → LOST

    @property
    def position(self) -> np.ndarray:
        return self.x[:2].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[2:4].copy()

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.x[2:4]))

    @property
    def position_uncertainty(self) -> float:
        """RMS position uncertainty from covariance diagonal (meters)."""
        return float(np.sqrt(self.P[0, 0] + self.P[1, 1]))


class EKFTargetTracker:
    """
    Multi-target Extended Kalman Filter tracker.

    Maintains one TargetTrack per known target ID. Handles:
    - Prediction step (constant-velocity model) each simulation tick
    - Measurement update when LOS sensor provides position data
    - Track state management (CONFIRMED → PREDICTED → LOST)
    - Predicted escape region computation for lost-target recovery
    """

    def __init__(
        self,
        n_targets: int = 3,
        process_noise_pos: float = 0.1,
        process_noise_vel: float = 1.0,
        measurement_noise: float = 0.5,
    ):
        self.tracks: Dict[int, TargetTrack] = {}
        self.n_targets = n_targets

        # Process noise covariance Q
        self.q_pos = process_noise_pos
        self.q_vel = process_noise_vel

        # Measurement noise covariance R (2x2)
        self.R = np.eye(2) * measurement_noise**2

        # Measurement matrix H (2x4): we observe [px, py]
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

        # Initialize tracks
        for tid in range(n_targets):
            self.tracks[tid] = TargetTrack(target_id=tid)

    def _build_F(self, dt: float) -> np.ndarray:
        """State transition matrix for constant-velocity model."""
        return np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=np.float64)

    def _build_Q(self, dt: float) -> np.ndarray:
        """Process noise covariance (discrete-time piecewise constant white noise)."""
        dt2 = dt * dt
        dt3 = dt2 * dt / 2.0
        dt4 = dt2 * dt2 / 4.0
        qp = self.q_pos
        qv = self.q_vel
        return np.array([
            [qp * dt4, 0,        qp * dt3, 0       ],
            [0,        qp * dt4, 0,        qp * dt3],
            [qp * dt3, 0,        qv * dt2, 0       ],
            [0,        qp * dt3, 0,        qv * dt2],
        ], dtype=np.float64)

    def predict(self, dt: float) -> None:
        """
        Run EKF prediction step for all tracks.
        Called once per simulation tick.
        """
        F = self._build_F(dt)
        Q = self._build_Q(dt)

        for tid, track in self.tracks.items():
            if track.state == TrackState.UNINITIALIZED:
                continue

            # State prediction
            track.x = F @ track.x
            # Covariance prediction
            track.P = F @ track.P @ F.T + Q

            # Update timing
            track.time_since_update += dt
            track.total_track_time += dt

            # State transitions based on time since update
            if track.time_since_update > track.LOST_THRESHOLD:
                track.state = TrackState.LOST
            elif track.time_since_update > track.PREDICTED_THRESHOLD:
                track.state = TrackState.PREDICTED

    def update(self, target_id: int, measured_position: np.ndarray) -> None:
        """
        Run EKF measurement update for a specific target.

        Parameters
        ----------
        target_id : int
        measured_position : array (2,) — [px, py] from sensor
        """
        if target_id not in self.tracks:
            return

        track = self.tracks[target_id]
        z = np.asarray(measured_position[:2], dtype=np.float64)

        if track.state == TrackState.UNINITIALIZED:
            # First measurement: initialize state directly
            track.x = np.array([z[0], z[1], 0.0, 0.0])
            track.P = np.diag([1.0, 1.0, 5.0, 5.0])
            track.state = TrackState.CONFIRMED
            track.time_since_update = 0.0
            track.n_updates = 1
            return

        # Innovation (measurement residual)
        y = z - self.H @ track.x

        # Innovation covariance
        S = self.H @ track.P @ self.H.T + self.R

        # Kalman gain
        K = track.P @ self.H.T @ np.linalg.inv(S)

        # State update
        track.x = track.x + K @ y

        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(4) - K @ self.H
        track.P = I_KH @ track.P @ I_KH.T + K @ self.R @ K.T

        # Reset timing and state
        track.time_since_update = 0.0
        track.state = TrackState.CONFIRMED
        track.n_updates += 1

    def get_predicted_position(self, target_id: int) -> Optional[np.ndarray]:
        """Get the current predicted 2D position for a target, or None if uninitialized."""
        track = self.tracks.get(target_id)
        if track is None or track.state == TrackState.UNINITIALIZED:
            return None
        return track.position

    def get_predicted_velocity(self, target_id: int) -> Optional[np.ndarray]:
        """Get the current predicted 2D velocity for a target."""
        track = self.tracks.get(target_id)
        if track is None or track.state == TrackState.UNINITIALIZED:
            return None
        return track.velocity

    def get_escape_radius(self, target_id: int) -> float:
        """
        Compute predicted escape region radius for lost-target recovery.
        r = v_target * t_age * 1.5 + position_uncertainty
        """
        track = self.tracks.get(target_id)
        if track is None:
            return 10.0
        speed = track.speed
        t_age = track.time_since_update
        return speed * t_age * 1.5 + track.position_uncertainty

    def get_confirmed_ids(self) -> Set[int]:
        """Return set of target IDs with CONFIRMED tracks."""
        return {tid for tid, t in self.tracks.items() if t.state == TrackState.CONFIRMED}

    def get_tracked_ids(self) -> Set[int]:
        """Return set of target IDs with CONFIRMED or PREDICTED tracks."""
        return {
            tid for tid, t in self.tracks.items()
            if t.state in (TrackState.CONFIRMED, TrackState.PREDICTED)
        }

    def get_lost_ids(self) -> Set[int]:
        """Return set of target IDs with LOST tracks."""
        return {tid for tid, t in self.tracks.items() if t.state == TrackState.LOST}

    def get_telemetry(self) -> Dict:
        """Return track states for telemetry broadcast."""
        return {
            tid: {
                "state": track.state.name,
                "pos": track.position.tolist() if track.state != TrackState.UNINITIALIZED else None,
                "vel": track.velocity.tolist() if track.state != TrackState.UNINITIALIZED else None,
                "age_s": round(track.time_since_update, 2),
                "uncertainty_m": round(track.position_uncertainty, 2),
                "n_updates": track.n_updates,
            }
            for tid, track in self.tracks.items()
        }
