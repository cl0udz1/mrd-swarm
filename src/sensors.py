# -*- coding: utf-8 -*-
"""
sensors.py — Synthetic Sensor Models, Noise Injection & Measurement Pipelines

Provides:
- SyntheticTargetSensor: Explicit measurement model separating ground truth from observations:
    GROUND TRUTH → Range/FOV → Building Occlusion → Smoke Attenuation → Additive Noise → Dropout → Measurement
- IMUSensor: 6-axis accelerometer & gyroscope with bias drift and Gaussian white noise
- BatteryModel: Physical electrochemical energy consumption driven by AirframeConfig
- HelipadZone: Recovery pad definitions
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from numpy.typing import NDArray

from .config.airframes import AirframeConfig, get_airframe_config
from .physics import (
    GRAVITY,
    quat_to_rotation_matrix,
)


@dataclass
class NoisyTargetMeasurement:
    """
    Synthetic sensor measurement of a ground target.
    Produced by onboard sensors with realistic noise, dropout, and latency.
    """
    target_id: int
    measured_pos_2d: NDArray[np.float64]  # [x, y] in world coordinates with additive noise
    range_m: float                        # Measured distance to target (m)
    bearing_rad: float                    # Measured relative bearing (rad)
    confidence: float                     # Observation confidence in [0, 1]
    timestamp: float                      # Simulation timestamp (s)
    drone_id: int                         # Detecting drone ID
    is_thermal: bool                      # Sighting made via thermal IR
    covariance_r: NDArray[np.float64]     # 2x2 measurement noise covariance matrix R


@dataclass
class HelipadZone:
    name: str
    position: NDArray[np.float64]         # [x, y, z] helipad center
    radius_m: float                       # usable landing radius
    altitude_m: float                     # rooftop surface altitude


HELIPAD_ALPHA = HelipadZone(
    name="Helipad Alpha (Depot Roof)",
    position=np.array([15.0, -15.0, 5.0]),
    radius_m=3.0,
    altitude_m=5.0,
)

HELIPAD_BRAVO = HelipadZone(
    name="Helipad Bravo (Complex Roof)",
    position=np.array([-14.0, 12.0, 6.0]),
    radius_m=3.0,
    altitude_m=6.0,
)

HELIPADS: List[HelipadZone] = [HELIPAD_ALPHA, HELIPAD_BRAVO]


class SyntheticTargetSensor:
    """
    Models onboard electro-optical (EO) and uncooled LWIR thermal cameras.
    
    Transforms ground-truth target kinematics into realistic noisy measurements:
    1. Distance attenuation and camera field-of-view (FOV) geometric cone.
    2. Ray-box occlusion testing against solid urban architecture.
    3. Aerosol smoke attenuation (blocks optical EO completely; thermal penetrates).
    4. Polar range/bearing noise mapped into Cartesian covariance matrix R.
    5. Stochastic measurement dropout (missed detection probability).
    """

    def __init__(
        self,
        drone_id: int,
        range_noise_std_pct: float = 0.03,  # 3% of range
        bearing_noise_std_deg: float = 1.0,  # 1.0 deg bearing noise
        dropout_probability: float = 0.03,   # 3% random sensor packet loss
        seed: Optional[int] = None,
    ):
        self.drone_id = drone_id
        self.airframe: AirframeConfig = get_airframe_config(drone_id)
        self.range_noise_pct = range_noise_std_pct
        self.bearing_noise_rad = math.radians(bearing_noise_std_deg)
        self.dropout_prob = dropout_probability
        self.rng = np.random.RandomState(seed if seed is not None else (100 + drone_id))

    def observe(
        self,
        drone_pos: NDArray[np.float64],
        drone_quat: NDArray[np.float64],
        target_id: int,
        true_target_pos: NDArray[np.float64],
        is_occluded_fn: Any,                  # Callable(p_start, p_end) -> bool
        target_smoke_active: bool = False,
        sim_time: float = 0.0,
    ) -> Optional[NoisyTargetMeasurement]:
        """
        Attempt to detect target. Returns NoisyTargetMeasurement if observed, None otherwise.
        """
        diff = true_target_pos - drone_pos
        true_dist = float(np.linalg.norm(diff))

        # 1. Check Range Limit
        max_range = self.airframe.thermal_range_m if self.airframe.has_thermal else self.airframe.camera_range_m
        if true_dist > max_range or true_dist < 0.2:
            return None

        # 2. Check Camera Field of View (horizontal frustum)
        R_b2w = quat_to_rotation_matrix(drone_quat)
        cam_forward = R_b2w[:, 0]  # Drone body x-axis
        unit_diff = diff / true_dist
        cos_angle = float(np.dot(cam_forward, unit_diff))
        fov_limit = math.cos(math.radians(self.airframe.camera_fov_deg / 2.0))
        if cos_angle < fov_limit:
            return None

        # 3. Check Building Occlusion Raycast
        if is_occluded_fn(drone_pos, true_target_pos):
            return None

        # 4. Check Smoke Screen Attenuation
        if target_smoke_active and not self.airframe.has_thermal:
            # Optical camera is completely blinded by aerosol smoke screen
            return None

        # 5. Check Stochastic Sensor Dropout
        if self.rng.uniform(0.0, 1.0) < self.dropout_prob:
            return None

        # 6. Inject Realistic Measurement Noise
        # Range noise grows with distance: sigma_r = max(0.1, dist * pct)
        sigma_r = max(0.12, true_dist * self.range_noise_pct)
        sigma_theta = self.bearing_noise_rad

        measured_dist = true_dist + self.rng.normal(0.0, sigma_r)
        measured_dist = max(0.2, measured_dist)

        # Planar bearing angle from camera forward
        true_bearing = math.atan2(diff[1], diff[0])
        measured_bearing = true_bearing + self.rng.normal(0.0, sigma_theta)

        # Reconstructed noisy 2D position
        measured_x = drone_pos[0] + measured_dist * math.cos(measured_bearing)
        measured_y = drone_pos[1] + measured_dist * math.sin(measured_bearing)
        noisy_pos_2d = np.array([measured_x, measured_y], dtype=np.float64)

        # Covariance matrix R in 2D Cartesian plane
        # R_cartesian = J * diag(sigma_r^2, sigma_theta^2) * J^T
        sin_b, cos_b = math.sin(measured_bearing), math.cos(measured_bearing)
        var_r = sigma_r ** 2
        var_theta = (true_dist * sigma_theta) ** 2  # transverse positional variance
        r_xx = (cos_b**2) * var_r + (sin_b**2) * var_theta
        r_yy = (sin_b**2) * var_r + (cos_b**2) * var_theta
        r_xy = sin_b * cos_b * (var_r - var_theta)
        cov_r = np.array([[r_xx, r_xy], [r_xy, r_yy]], dtype=np.float64)

        # Confidence metric derived from distance and angle
        range_factor = max(0.0, 1.0 - (true_dist / max_range))
        angle_factor = max(0.0, (cos_angle - fov_limit) / (1.0 - fov_limit + 1e-6))
        conf = float(np.clip(0.6 * range_factor + 0.4 * angle_factor, 0.20, 0.98))

        return NoisyTargetMeasurement(
            target_id=target_id,
            measured_pos_2d=noisy_pos_2d,
            range_m=measured_dist,
            bearing_rad=measured_bearing,
            confidence=conf,
            timestamp=sim_time,
            drone_id=self.drone_id,
            is_thermal=self.airframe.has_thermal,
            covariance_r=cov_r,
        )


class BatteryModel:
    """
    Electrochemical battery discharge model strictly parameterized by AirframeConfig.
    Calculates power draw from avionics, sensor payload, and aerodynamic thrust demand.
    """

    def __init__(self, airframe: AirframeConfig):
        self.airframe = airframe
        self.capacity_wh = airframe.battery_capacity_wh
        self.initial_capacity_wh = airframe.battery_capacity_wh
        self.remaining_wh = airframe.battery_capacity_wh
        self.voltage = airframe.battery_nominal_voltage_v
        self.total_energy_consumed_wh = 0.0

    def step(self, total_thrust_n: float, dt: float) -> float:
        """
        Discharge battery based on thrust output over dt seconds.
        Returns remaining state-of-charge percentage [0, 100].
        """
        # Aerodynamic power proportional to thrust^(3/2) (momentum theory)
        thrust_ratio = total_thrust_n / max(0.1, self.airframe.weight_n)
        p_aero = self.airframe.p_hover_base_w * (thrust_ratio ** 1.5)
        p_total = self.airframe.p_avionics_w + self.airframe.p_payload_w + p_aero

        energy_consumed = (p_total * dt) / 3600.0  # Wh
        self.remaining_wh = max(0.0, self.remaining_wh - energy_consumed)
        self.total_energy_consumed_wh += energy_consumed

        return self.soc_pct

    @property
    def soc_pct(self) -> float:
        """State of charge percentage."""
        if self.initial_capacity_wh <= 0:
            return 100.0
        return float(np.clip((self.remaining_wh / self.initial_capacity_wh) * 100.0, 0.0, 100.0))

    @property
    def is_critical(self) -> bool:
        """Critical low-battery threshold (15%)."""
        return self.soc_pct <= 15.0


@dataclass
class TargetObservation:
    """Detected target in camera frame (legacy compatibility)."""
    target_id: int
    pixel_x: int = 0
    pixel_y: int = 0
    pixel_width: int = 0
    pixel_height: int = 0
    distance: float = 0.0
    bearing: float = 0.0
    confidence: float = 0.0
    in_fov: bool = False


@dataclass
class BatteryState:
    """Battery telemetry (legacy compatibility)."""
    capacity_wh: float
    percentage: float
    voltage: float
    is_critical: bool
    total_energy_consumed_wh: float = 0.0


class SensorSuite:
    """Multi-sensor integration suite (legacy compatibility)."""

    def __init__(self, drone_id: int = 0, seed: Optional[int] = None):
        self.drone_id = drone_id
        self.target_sensor = SyntheticTargetSensor(drone_id=drone_id, seed=seed)
        self.battery = BatteryModel(get_airframe_config(drone_id))
        self.visible_targets: Dict[int, float] = {}

