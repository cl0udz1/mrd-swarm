# -*- coding: utf-8 -*-
"""
airframes.py — Authoritative Heterogeneous Airframe Configuration

Single source of truth for all vehicle specifications across the MRD-Swarm platform.
No subsystem or script may define conflicting airframe constants.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Dict
import numpy as np


class DroneClass(Enum):
    HEAVY_SCOUT = "HEAVY_SCOUT"
    FAST_INTERCEPTOR = "FAST_INTERCEPTOR"
    THERMAL_SURVEYOR = "THERMAL_SURVEYOR"
    COMMS_RELAY = "COMMS_RELAY"


@dataclass(frozen=True)
class AirframeConfig:
    """Rigid physical and avionics specifications for a drone airframe."""
    drone_id: int
    drone_class: DroneClass
    name: str

    # Mass & Inertia
    mass_kg: float                    # Total takeoff mass (kg)
    arm_length_m: float               # Center-to-motor distance (m)
    inertia_ixx: float                # kg·m²
    inertia_iyy: float                # kg·m²
    inertia_izz: float                # kg·m²

    # Propulsion & Actuators
    thrust_margin: float              # Max total thrust / weight ratio
    k_f: float                        # Thrust coefficient [N / (rad/s)²]
    k_m: float                        # Torque coefficient [N·m / (rad/s)²]
    omega_max_rad_s: float            # Maximum rotor speed (rad/s)
    max_thrust_per_motor_n: float     # Maximum thrust per motor (N)
    max_speed_mps: float              # Maximum forward flight speed (m/s)
    max_tilt_rad: float               # Maximum physical tilt/bank angle (rad)
    max_body_rate_rad_s: float        # Maximum body angular rate (rad/s)

    # Battery & Power
    battery_capacity_wh: float        # Total usable pack capacity (Wh)
    battery_nominal_voltage_v: float  # Nominal voltage (V)
    p_avionics_w: float               # Base avionics + compute power draw (W)
    p_payload_w: float                # Sensor payload power draw (W)
    p_hover_base_w: float             # Aerodynamic power draw at hover (W)

    # Perception & Sensors
    camera_fov_deg: float             # Primary optical camera horizontal FOV (deg)
    camera_range_m: float             # Maximum effective detection range (m)
    has_thermal: bool                 # Uncooled LWIR thermal sensor equipped
    thermal_range_m: float            # Effective thermal detection range (m)
    rangefinder_max_m: float          # Downward / forward LiDAR range (m)

    # Mesh Communication
    rf_comm_range_m: float            # Maximum 1-hop RF line-of-sight range (m)
    rf_power_dbm: float               # Transmit power (dBm)

    @property
    def inertia_matrix(self) -> np.ndarray:
        return np.diag([self.inertia_ixx, self.inertia_iyy, self.inertia_izz])

    @property
    def weight_n(self) -> float:
        return self.mass_kg * 9.80665

    @property
    def max_total_thrust_n(self) -> float:
        return self.weight_n * self.thrust_margin


# ═══════════════════════════════════════════════════════════════════════════════
# AUTHORITATIVE FLEET SPECIFICATIONS (4 Heterogeneous Airframes)
# ═══════════════════════════════════════════════════════════════════════════════

FLEET_CONFIGS: Dict[int, AirframeConfig] = {
    0: AirframeConfig(
        drone_id=0,
        drone_class=DroneClass.HEAVY_SCOUT,
        name="Heavy Scout (Falcon-0)",
        mass_kg=0.65,
        arm_length_m=0.14,
        inertia_ixx=1.8e-3,
        inertia_iyy=1.8e-3,
        inertia_izz=3.2e-3,
        thrust_margin=2.2,
        k_f=1.8e-4,
        k_m=2.4e-5,
        omega_max_rad_s=2200.0,
        max_thrust_per_motor_n=3.6,
        max_speed_mps=12.0,
        max_tilt_rad=np.radians(45.0),
        max_body_rate_rad_s=18.0,
        battery_capacity_wh=45.0,
        battery_nominal_voltage_v=14.8,  # 4S LiPo
        p_avionics_w=3.5,
        p_payload_w=4.0,                 # High-res optical EO payload
        p_hover_base_w=42.0,
        camera_fov_deg=85.0,
        camera_range_m=28.0,
        has_thermal=False,
        thermal_range_m=0.0,
        rangefinder_max_m=15.0,
        rf_comm_range_m=18.0,
        rf_power_dbm=20.0,
    ),
    1: AirframeConfig(
        drone_id=1,
        drone_class=DroneClass.FAST_INTERCEPTOR,
        name="Fast Interceptor (Falcon-1)",
        mass_kg=0.28,
        arm_length_m=0.09,
        inertia_ixx=0.6e-3,
        inertia_iyy=0.6e-3,
        inertia_izz=1.1e-3,
        thrust_margin=3.5,
        k_f=1.2e-4,
        k_m=1.6e-5,
        omega_max_rad_s=2800.0,
        max_thrust_per_motor_n=2.5,
        max_speed_mps=18.0,
        max_tilt_rad=np.radians(55.0),
        max_body_rate_rad_s=25.0,
        battery_capacity_wh=22.0,
        battery_nominal_voltage_v=11.1,  # 3S LiPo
        p_avionics_w=2.5,
        p_payload_w=1.5,
        p_hover_base_w=28.0,
        camera_fov_deg=75.0,
        camera_range_m=22.0,
        has_thermal=False,
        thermal_range_m=0.0,
        rangefinder_max_m=10.0,
        rf_comm_range_m=18.0,
        rf_power_dbm=20.0,
    ),
    2: AirframeConfig(
        drone_id=2,
        drone_class=DroneClass.THERMAL_SURVEYOR,
        name="Thermal Surveyor (Falcon-2)",
        mass_kg=0.42,
        arm_length_m=0.11,
        inertia_ixx=1.1e-3,
        inertia_iyy=1.1e-3,
        inertia_izz=2.0e-3,
        thrust_margin=2.4,
        k_f=1.5e-4,
        k_m=2.0e-5,
        omega_max_rad_s=2400.0,
        max_thrust_per_motor_n=2.6,
        max_speed_mps=14.0,
        max_tilt_rad=np.radians(45.0),
        max_body_rate_rad_s=18.0,
        battery_capacity_wh=35.0,
        battery_nominal_voltage_v=14.8,
        p_avionics_w=3.0,
        p_payload_w=5.5,                 # LWIR thermal sensor core
        p_hover_base_w=34.0,
        camera_fov_deg=70.0,
        camera_range_m=24.0,
        has_thermal=True,
        thermal_range_m=26.0,
        rangefinder_max_m=12.0,
        rf_comm_range_m=18.0,
        rf_power_dbm=20.0,
    ),
    3: AirframeConfig(
        drone_id=3,
        drone_class=DroneClass.COMMS_RELAY,
        name="Comms Relay (Falcon-3)",
        mass_kg=0.50,
        arm_length_m=0.12,
        inertia_ixx=1.4e-3,
        inertia_iyy=1.4e-3,
        inertia_izz=2.5e-3,
        thrust_margin=2.0,
        k_f=1.6e-4,
        k_m=2.2e-5,
        omega_max_rad_s=2300.0,
        max_thrust_per_motor_n=2.6,
        max_speed_mps=8.0,
        max_tilt_rad=np.radians(35.0),
        max_body_rate_rad_s=14.0,
        battery_capacity_wh=55.0,        # Large high-endurance pack
        battery_nominal_voltage_v=14.8,
        p_avionics_w=4.0,
        p_payload_w=6.0,                 # High-power RF repeater pod
        p_hover_base_w=38.0,
        camera_fov_deg=60.0,
        camera_range_m=15.0,
        has_thermal=False,
        thermal_range_m=0.0,
        rangefinder_max_m=10.0,
        rf_comm_range_m=32.0,            # High-gain antenna
        rf_power_dbm=27.0,
    ),
}


def get_airframe_config(drone_id: int) -> AirframeConfig:
    """Retrieve authoritative airframe config by drone ID."""
    if drone_id not in FLEET_CONFIGS:
        raise KeyError(f"Unknown drone_id {drone_id}. Authorized IDs: {list(FLEET_CONFIGS.keys())}")
    return FLEET_CONFIGS[drone_id]
