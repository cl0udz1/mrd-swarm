"""
sensors.py — Sensor models, noise injection, and reconnaissance pipelines
for the MRD-Swarm quadrotor.

Includes:
    - IMU (accelerometer + gyroscope) with bias drift and Gaussian noise
    - Rangefinder array with dropout and noise
    - Camera frustum / target-in-view detection
    - Battery discharge model
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

from .physics import (
    ACCEL_NOISE_STD, ACCEL_BIAS_DRIFT,
    GYRO_NOISE_STD, GYRO_BIAS_DRIFT,
    RANGEFINDER_NOISE_STD, RANGEFINDER_DROPOUT_PROB, RANGEFINDER_MAX_RANGE,
    CAMERA_LATENCY_STEPS,
    BATTERY_CAPACITY_WH, P_HOVER_BASE, P_AVIONICS, P_SENSOR_PAYLOAD,
    DRONE_MASS, GRAVITY,
    quat_to_rotation_matrix, compute_power_consumption,
)


@dataclass
class IMUReading:
    """Processed IMU measurement with noise."""
    acceleration: NDArray[np.float64]   # m/s², body frame, with noise + bias
    angular_velocity: NDArray[np.float64]  # rad/s, body frame, with noise + bias
    timestamp: float


@dataclass
class RangefinderReading:
    """Single rangefinder measurement."""
    distance: float          # meters, NaN if dropout
    is_valid: bool
    sensor_name: str


@dataclass
class ProximityArray:
    """Multi-directional proximity sensor array."""
    forward: RangefinderReading
    left: RangefinderReading
    right: RangefinderReading
    rear: RangefinderReading
    down: RangefinderReading   # altimeter


@dataclass
class TargetObservation:
    """Detected target in camera frame."""
    target_id: int
    pixel_x: int              # image coordinate u
    pixel_y: int              # image coordinate v
    pixel_width: int          # bounding box width
    pixel_height: int         # bounding box height
    distance: float           # estimated range (m)
    bearing: float            # bearing angle (rad) from camera forward
    confidence: float         # detection confidence [0, 1]
    in_fov: bool              # within camera field of view


@dataclass
class BatteryState:
    """Battery telemetry."""
    capacity_wh: float        # remaining capacity (Wh)
    percentage: float         # [0, 100]
    voltage: float            # estimated voltage (V)
    is_critical: bool         # < 10%
    total_energy_consumed_wh: float


class IMUSensor:
    """
    6-axis IMU model with realistic noise characteristics.

    Noise model:
        a_measured = a_true + bias_a + N(0, σ_a²)
        ω_measured = ω_true + bias_ω + N(0, σ_ω²)

    Bias evolves as random walk:
        bias(k+1) = bias(k) + N(0, σ_drift²)
    """

    def __init__(
        self,
        accel_noise_std: float = ACCEL_NOISE_STD,
        gyro_noise_std: float = GYRO_NOISE_STD,
        accel_bias_drift: float = ACCEL_BIAS_DRIFT,
        gyro_bias_drift: float = GYRO_BIAS_DRIFT,
        seed: int | None = None,
    ):
        self.rng = np.random.default_rng(seed)
        self.accel_noise_std = accel_noise_std
        self.gyro_noise_std = gyro_noise_std
        self.accel_bias_drift = accel_bias_drift
        self.gyro_bias_drift = gyro_bias_drift

        # Internal bias state
        self.accel_bias = np.zeros(3)
        self.gyro_bias = np.zeros(3)

    def reset(self) -> None:
        self.accel_bias = np.zeros(3)
        self.gyro_bias = np.zeros(3)

    def read(
        self,
        true_acceleration: NDArray[np.float64],
        true_angular_velocity: NDArray[np.float64],
        timestamp: float,
    ) -> IMUReading:
        """
        Generate noisy IMU reading from ground-truth values.

        Parameters
        ----------
        true_acceleration : (3,) body-frame acceleration (including gravity)
        true_angular_velocity : (3,) body-frame angular velocity
        timestamp : current time

        Returns
        -------
        IMUReading with noise and bias injected
        """
        # Evolve bias (random walk)
        self.accel_bias += self.rng.normal(0, self.accel_bias_drift, 3)
        self.gyro_bias += self.rng.normal(0, self.gyro_bias_drift, 3)

        # Clip bias to prevent runaway
        self.accel_bias = np.clip(self.accel_bias, -0.5, 0.5)
        self.gyro_bias = np.clip(self.gyro_bias, -0.1, 0.1)

        # Add noise + bias
        noisy_accel = (
            true_acceleration
            + self.accel_bias
            + self.rng.normal(0, self.accel_noise_std, 3)
        )
        noisy_gyro = (
            true_angular_velocity
            + self.gyro_bias
            + self.rng.normal(0, self.gyro_noise_std, 3)
        )

        return IMUReading(
            acceleration=noisy_accel,
            angular_velocity=noisy_gyro,
            timestamp=timestamp,
        )


class RangefinderSensor:
    """
    Single-point rangefinder with noise and dropout model.

    Dropout simulates measurement failures (surface reflectivity, angle of incidence).
    """

    def __init__(
        self,
        noise_std: float = RANGEFINDER_NOISE_STD,
        dropout_prob: float = RANGEFINDER_DROPOUT_PROB,
        max_range: float = RANGEFINDER_MAX_RANGE,
        sensor_name: str = "rangefinder",
        seed: int | None = None,
    ):
        self.rng = np.random.default_rng(seed)
        self.noise_std = noise_std
        self.dropout_prob = dropout_prob
        self.max_range = max_range
        self.sensor_name = sensor_name

    def read(self, true_distance: float) -> RangefinderReading:
        """
        Generate noisy rangefinder reading.

        Parameters
        ----------
        true_distance : ground-truth distance in meters

        Returns
        -------
        RangefinderReading
        """
        # Dropout check
        if self.rng.random() < self.dropout_prob:
            return RangefinderReading(
                distance=float('nan'),
                is_valid=False,
                sensor_name=self.sensor_name,
            )

        # Clamp to max range
        if true_distance > self.max_range:
            return RangefinderReading(
                distance=self.max_range,
                is_valid=True,
                sensor_name=self.sensor_name,
            )

        # Add Gaussian noise
        noisy_dist = true_distance + self.rng.normal(0, self.noise_std)
        noisy_dist = max(noisy_dist, 0.0)

        return RangefinderReading(
            distance=noisy_dist,
            is_valid=True,
            sensor_name=self.sensor_name,
        )


class ProximityArraySensor:
    """Multi-directional proximity sensor array (5 directions)."""

    def __init__(self, seed: int | None = None):
        base_seed = seed if seed is not None else 42
        self.down = RangefinderSensor(sensor_name="altimeter", seed=base_seed)
        self.forward = RangefinderSensor(sensor_name="prox_forward", seed=base_seed + 1)
        self.left = RangefinderSensor(sensor_name="prox_left", seed=base_seed + 2)
        self.right = RangefinderSensor(sensor_name="prox_right", seed=base_seed + 3)
        self.rear = RangefinderSensor(sensor_name="prox_rear", seed=base_seed + 4)

    def read(
        self,
        dist_down: float,
        dist_forward: float,
        dist_left: float,
        dist_right: float,
        dist_rear: float,
    ) -> ProximityArray:
        return ProximityArray(
            forward=self.forward.read(dist_forward),
            left=self.left.read(dist_left),
            right=self.right.read(dist_right),
            rear=self.rear.read(dist_rear),
            down=self.down.read(dist_down),
        )


class ReconCamera:
    """
    Reconnaissance camera with frustum-based target detection.

    Projects 3D world coordinates to 2D image coordinates and checks
    field-of-view constraints for target-in-view determination.
    """

    def __init__(
        self,
        hfov: float = 70.0,       # horizontal FOV in degrees
        vfov: float = 55.0,       # vertical FOV in degrees
        image_width: int = 640,
        image_height: int = 480,
        max_detection_range: float = 50.0,  # meters
        latency_steps: int = CAMERA_LATENCY_STEPS,
    ):
        self.hfov = np.radians(hfov)
        self.vfov = np.radians(vfov)
        self.image_width = image_width
        self.image_height = image_height
        self.max_detection_range = max_detection_range
        self.latency_steps = latency_steps

        # Focal lengths in pixels
        self.fx = (image_width / 2.0) / np.tan(self.hfov / 2.0)
        self.fy = (image_height / 2.0) / np.tan(self.vfov / 2.0)
        self.cx = image_width / 2.0
        self.cy = image_height / 2.0

        # Latency buffer
        self._buffer: list[NDArray[np.float64]] = []

    def project_target(
        self,
        drone_position: NDArray[np.float64],
        drone_quaternion: NDArray[np.float64],
        target_position: NDArray[np.float64],
        target_radius: float = 0.5,
    ) -> Optional[TargetObservation]:
        """
        Check if target is within camera FOV and compute image coordinates.

        Uses pinhole camera model:
            [u]   [fx  0  cx] [X_c/Z_c]
            [v] = [ 0 fy  cy]·[Y_c/Z_c]
            [1]   [ 0  0   1] [   1    ]

        where [X_c, Y_c, Z_c] = R^T @ (p_target - p_drone) in camera frame.

        Parameters
        ----------
        drone_position : (3,) world position
        drone_quaternion : (4,) [w,x,y,z]
        target_position : (3,) world position
        target_radius : bounding sphere radius (m)

        Returns
        -------
        TargetObservation or None if not in FOV
        """
        R = quat_to_rotation_matrix(drone_quaternion)

        # Vector from drone to target in world frame
        delta_world = target_position - drone_position
        distance = np.linalg.norm(delta_world)

        if distance > self.max_detection_range or distance < 0.1:
            return None

        # Transform to body frame (camera is forward-facing along body +X)
        # Camera frame: X=right, Y=down, Z=forward
        delta_body = R.T @ delta_world

        # Camera frame convention: Z forward, X right, Y down
        # Body +X = camera +Z (forward), Body +Y = camera -X (left→right negated), Body -Z = camera +Y (down)
        x_c = delta_body[1]   # right
        y_c = -delta_body[2]  # down (body Z is up, camera Y is down)
        z_c = delta_body[0]   # forward

        # Behind camera check
        if z_c <= 0.1:
            return None

        # Project to image plane
        u = int(self.fx * (x_c / z_c) + self.cx)
        v = int(self.fy * (y_c / z_c) + self.cy)

        # FOV check (with margin for target size)
        half_w = int(self.fx * (target_radius / z_c))
        half_h = int(self.fy * (target_radius / z_c))

        # Check if any part of bounding box is in frame
        bbox_left = u - half_w
        bbox_right = u + half_w
        bbox_top = v - half_h
        bbox_bottom = v + half_h

        if (bbox_right < 0 or bbox_left >= self.image_width or
            bbox_bottom < 0 or bbox_top >= self.image_height):
            return None

        # Bearing angle from camera forward axis
        bearing = np.arctan2(np.sqrt(x_c**2 + y_c**2), z_c)

        # Confidence based on distance and angle
        range_factor = 1.0 - (distance / self.max_detection_range)
        angle_factor = 1.0 - (bearing / (self.hfov / 2.0))
        confidence = np.clip(range_factor * angle_factor, 0.0, 1.0)

        return TargetObservation(
            target_id=-1,  # filled by caller
            pixel_x=u,
            pixel_y=v,
            pixel_width=2 * half_w,
            pixel_height=2 * half_h,
            distance=distance,
            bearing=bearing,
            confidence=confidence,
            in_fov=True,
        )


class BatteryModel:
    """
    Battery discharge model with capacity tracking.

    Power model:
        P_total = P_hover * (T/mg)^{3/2} + P_avionics + P_payload

    Energy integration:
        E(k+1) = E(k) - P_total * dt
    """

    def __init__(
        self,
        initial_capacity_wh: float = BATTERY_CAPACITY_WH,
        critical_pct: float = 10.0,
    ):
        self.initial_capacity_wh = initial_capacity_wh
        self.remaining_wh = initial_capacity_wh
        self.critical_pct = critical_pct
        self.total_consumed_wh = 0.0

    def reset(self) -> None:
        self.remaining_wh = self.initial_capacity_wh
        self.total_consumed_wh = 0.0

    def update(self, thrust_ratio: float, dt: float) -> BatteryState:
        """
        Update battery state based on power consumption.

        Parameters
        ----------
        thrust_ratio : T / (m*g), normalized thrust
        dt : timestep in seconds

        Returns
        -------
        BatteryState
        """
        power = compute_power_consumption(thrust_ratio)
        energy_wh = power * dt / 3600.0  # convert W·s to Wh

        self.remaining_wh = max(0.0, self.remaining_wh - energy_wh)
        self.total_consumed_wh += energy_wh

        pct = (self.remaining_wh / self.initial_capacity_wh) * 100.0
        # Linear voltage approximation: 4.2V (full) → 3.0V (empty)
        voltage = 3.0 + 1.2 * (pct / 100.0)

        return BatteryState(
            capacity_wh=self.remaining_wh,
            percentage=pct,
            voltage=voltage,
            is_critical=pct < self.critical_pct,
            total_energy_consumed_wh=self.total_consumed_wh,
        )


class SensorSuite:
    """
    Complete sensor package for a single reconnaissance drone.
    Aggregates IMU, proximity array, camera, and battery.
    """

    def __init__(self, drone_id: int, seed: int | None = None):
        self.drone_id = drone_id
        base_seed = seed if seed is not None else (42 + drone_id * 100)

        self.imu = IMUSensor(seed=base_seed)
        self.proximity = ProximityArraySensor(seed=base_seed + 10)
        self.camera = ReconCamera()
        self.battery = BatteryModel()

    def reset(self) -> None:
        self.imu.reset()
        self.battery.reset()
