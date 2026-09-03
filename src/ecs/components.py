# -*- coding: utf-8 -*-
"""
components.py — Data-Oriented Entity Component System (ECS) Components

Enhanced for high-intensity real-world tactical combat missions:
- EW Electronic Warfare Jamming
- Thermal Smoke Countermeasures
- Tactical Laser Target Designation
- Rooftop Helipads & Station Relief
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class DroneClassID(IntEnum):
    HEAVY_SCOUT = 0
    FAST_INTERCEPTOR = 1
    THERMAL_SURVEYOR = 2
    COMMS_RELAY = 3


class TacticalRoleID(IntEnum):
    EXPLORER = 0
    TRACKER = 1
    FLANKER = 2
    RELAY = 3
    LOST_TARGET_SWEEP = 4
    RTB_RECOVERY = 5


class TargetStateID(IntEnum):
    PATROL = 0
    ACTIVE_EVASION = 1
    SMOKE_SCREEN_EVASION = 2


@dataclass
class TransformComponent:
    """Rigid body spatial pose and kinematic state on SE(3)."""
    position: np.ndarray           # (3,) [x, y, z] in meters
    quaternion: np.ndarray         # (4,) [w, x, y, z] orientation
    velocity: np.ndarray           # (3,) [vx, vy, vz] in m/s
    angular_velocity: np.ndarray   # (3,) [p, q, r] in rad/s


@dataclass
class PhysicsBodyComponent:
    """Physical rigid body inertial and aerodynamic parameters."""
    mass: float                    # kg
    arm_length: float              # m
    thrust_margin: float           # Max thrust / weight ratio
    motor_thrusts: np.ndarray      # (4,) normalized motor commands [u0, u1, u2, u3]
    total_thrust_N: float = 0.0
    # Diagonal inertia tensor J = diag(Ixx, Iyy, Izz) in kg·m²
    # Default: Crazyflie 2.0-class platform estimates
    inertia_diag: np.ndarray = field(default_factory=lambda: np.array([1.1e-3, 1.1e-3, 2.1e-3]))


@dataclass
class SensorComponent:
    """Optical, thermal, and proximity perception payload."""
    camera_fov_deg: float          # degrees
    max_sensor_range: float        # meters
    visible_targets: Dict[int, float] = field(default_factory=dict)  # target_id -> confidence
    noisy_measurements: Dict[int, Any] = field(default_factory=dict)  # target_id -> (noisy_pos_2d, cov_r, conf)
    optical_heading: float = 0.0   # rad
    has_thermal_ir: bool = False   # True for Drone 2 (can penetrate smoke)


@dataclass
class BatteryComponent:
    """Electrochemical battery state and discharge dynamics."""
    capacity_wh: float             # Total capacity in Watt-hours
    remaining_wh: float            # Current charge in Watt-hours
    nominal_voltage: float = 14.8  # Volts (4S LiPo)
    soc_pct: float = 100.0         # State of Charge percentage

    @property
    def total_energy_consumed_wh(self) -> float:
        return max(0.0, self.capacity_wh - self.remaining_wh)


@dataclass
class TacticalComponent:
    """High-level cognitive mission state and dynamic assignments."""
    role: TacticalRoleID = TacticalRoleID.EXPLORER
    assigned_target_id: Optional[int] = None
    goal_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    desired_speed: float = 8.5
    active_tool: str = "recon_area_search()"
    reasoning: str = ""
    last_decision_time: float = 0.0
    # Coordinated pincer geometry
    formation_angle_deg: float = 0.0    # angular separation from partner (°)
    tti_seconds: float = float("inf")   # estimated time to intercept target
    # Target priority
    threat_score: float = 0.0           # utility/threat score for assigned target


@dataclass
class LaserDesignatorComponent:
    """Tactical laser designation beam projecting SE(3) target coordinates."""
    active: bool = False
    target_id: Optional[int] = None
    target_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    laser_color: str = "#22c55e"   # Green laser by default, Red when firing lock


@dataclass
class RFMeshComponent:
    """Decentralized peer-to-peer radio frequency communication link."""
    comm_range_m: float = 18.0
    connected_peers: List[int] = field(default_factory=list)
    jammed: bool = False
    signal_quality_pct: float = 100.0


@dataclass
class TargetEntityComponent:
    """Dynamic evasive ground vehicle state and road network navigation."""
    target_id: int
    name: str
    state: TargetStateID = TargetStateID.PATROL
    waypoints: List[np.ndarray] = field(default_factory=list)
    current_wp_idx: int = 0
    base_speed: float = 2.4
    evasion_speed: float = 4.2
    evasion_goal: Optional[np.ndarray] = None
    time_in_evasion: float = 0.0
    smoke_active: bool = False
    smoke_timer: float = 0.0
    smoke_position: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class EWJammingField:
    """Electronic Warfare directional jamming zone."""
    active: bool = False
    center: np.ndarray = field(default_factory=lambda: np.array([12.0, 12.0, 4.0]))
    radius: float = 14.0
    intensity: float = 0.85
