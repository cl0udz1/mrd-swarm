# -*- coding: utf-8 -*-
"""
targets.py — Dynamic Evasive Ground Targets Sub-Agent

Implements intelligent ground targets that navigate urban road corridors and
actively evade reconnaissance drones by seeking building shadow zones (turning corners
to break line of sight).
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


class TargetState(Enum):
    PATROL = "PATROL"
    ACTIVE_EVASION = "ACTIVE_EVASION"
    SHADOW_LOITER = "SHADOW_LOITER"


@dataclass
class TargetTelemetry:
    target_id: int
    name: str
    position: np.ndarray
    velocity: np.ndarray
    state: TargetState
    detected_by_drone: bool
    nearest_building: Optional[str] = None
    smoke_active: bool = False
    smoke_position: Optional[np.ndarray] = None


class EvasiveTarget:
    """
    An autonomous ground target entity with reactive evasion AI.
    """

    def __init__(
        self,
        target_id: int,
        name: str,
        waypoints: List[np.ndarray],
        base_speed: float = 1.4,
        evasion_speed: float = 2.2,
    ):
        self.target_id = target_id
        self.name = name
        self.waypoints = [np.array(wp, dtype=np.float64) for wp in waypoints]
        self.base_speed = base_speed
        self.evasion_speed = evasion_speed

        self.current_wp_idx = 0
        self.position = self.waypoints[0].copy()
        self.position[2] = 0.30  # Ground height
        self.velocity = np.zeros(3, dtype=np.float64)
        self.state = TargetState.PATROL

        self.time_in_evasion = 0.0
        self.evasion_goal: Optional[np.ndarray] = None
        self.last_threat_pos: Optional[np.ndarray] = None

        # Reactive smoke countermeasure
        self.smoke_active = False
        self.smoke_timer = 0.0
        self.smoke_cooldown = 0.0
        self.continuous_lock_timer = 0.0
        self.smoke_position: Optional[np.ndarray] = None

    def update(
        self,
        dt: float,
        sim_time: float,
        drone_positions: Dict[int, np.ndarray],
        obstacles: List[Dict[str, Any]],
        sensor_sightings: List[int],
    ) -> TargetTelemetry:
        """
        Step the target's evasive state machine and position.
        """
        is_spotted = self.target_id in sensor_sightings

        # State Transitions
        if is_spotted:
            self.state = TargetState.ACTIVE_EVASION
            self.time_in_evasion = 0.0
        elif self.state == TargetState.ACTIVE_EVASION:
            self.time_in_evasion += dt
            if self.time_in_evasion > 4.0:
                self.state = TargetState.PATROL
                self.evasion_goal = None

        # Movement Execution
        if self.state == TargetState.ACTIVE_EVASION:
            # 1. Find nearest threatening drone
            closest_dist = float("inf")
            threat_p = None
            for d_id, d_pos in drone_positions.items():
                d = float(np.linalg.norm(self.position[:2] - d_pos[:2]))
                if d < closest_dist:
                    closest_dist = d
                    threat_p = d_pos.copy()
            self.last_threat_pos = threat_p

            # 2. Seek nearest building shadow zone (corner that breaks LOS)
            if self.evasion_goal is None or np.linalg.norm(self.position[:2] - self.evasion_goal[:2]) < 1.0:
                best_shadow = self._find_best_shadow_corner(threat_p, obstacles)
                self.evasion_goal = best_shadow

            diff = self.evasion_goal - self.position
            dist = float(np.linalg.norm(diff[:2]))
            if dist > 0.1:
                direction = diff[:2] / dist
                speed = self.evasion_speed * (1.0 + 0.15 * math.sin(2.0 * sim_time + self.target_id))
                self.velocity[:2] = direction * speed
            else:
                self.velocity[:2] = np.zeros(2)

        else:
            # Normal Patrol / Road Navigation
            target_wp = self.waypoints[self.current_wp_idx]
            diff = target_wp - self.position
            dist = float(np.linalg.norm(diff[:2]))

            if dist < 1.2:
                self.current_wp_idx = (self.current_wp_idx + 1) % len(self.waypoints)
                target_wp = self.waypoints[self.current_wp_idx]
                diff = target_wp - self.position
                dist = float(np.linalg.norm(diff[:2]))

            if dist > 0.1:
                direction = diff[:2] / (dist + 1e-6)
                speed = self.base_speed * (1.0 + 0.20 * math.sin(0.8 * sim_time + self.target_id))
                self.velocity[:2] = direction * speed
            else:
                self.velocity[:2] = np.zeros(2)

        # Reactive Smoke Screen Countermeasure
        if self.smoke_cooldown > 0.0:
            self.smoke_cooldown -= dt

        if is_spotted:
            self.continuous_lock_timer += dt
            # If illuminated by drones continuously for > 2.5s, deploy aerosol smoke
            if self.continuous_lock_timer > 2.5 and not self.smoke_active and self.smoke_cooldown <= 0.0:
                self.smoke_active = True
                self.smoke_timer = 6.0
                self.smoke_cooldown = 18.0
                self.smoke_position = self.position.copy()
        else:
            self.continuous_lock_timer = max(0.0, self.continuous_lock_timer - dt * 0.5)

        if self.smoke_active:
            self.smoke_timer -= dt
            if self.smoke_timer <= 0.0:
                self.smoke_active = False
                self.smoke_position = None

        # Integrate ground position
        self.position[:2] += self.velocity[:2] * dt
        self.position[2] = 0.30

        # Boundary clamping (-27m to 27m)
        self.position[0] = float(np.clip(self.position[0], -27.0, 27.0))
        self.position[1] = float(np.clip(self.position[1], -27.0, 27.0))

        return TargetTelemetry(
            target_id=self.target_id,
            name=self.name,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            state=self.state,
            detected_by_drone=is_spotted,
            smoke_active=self.smoke_active,
            smoke_position=self.smoke_position.copy() if self.smoke_position is not None else None,
        )

    def _find_best_shadow_corner(
        self,
        threat_pos: Optional[np.ndarray],
        obstacles: List[Dict[str, Any]],
    ) -> np.ndarray:
        """Finds a corner behind an obstacle relative to the threat direction."""
        if threat_pos is None or len(obstacles) == 0:
            return self.waypoints[(self.current_wp_idx + 1) % len(self.waypoints)]

        threat_dir = self.position[:2] - threat_pos[:2]
        threat_norm = np.linalg.norm(threat_dir) + 1e-6
        threat_unit = threat_dir / threat_norm

        best_corner = None
        best_score = -1e9

        for obs in obstacles:
            ox, oy = obs["pos"][:2]
            hw, hl = obs["size"][:2]
            # 4 corners of obstacle with safety buffer
            corners = [
                np.array([ox - hw - 1.2, oy - hl - 1.2, 0.3]),
                np.array([ox + hw + 1.2, oy - hl - 1.2, 0.3]),
                np.array([ox + hw + 1.2, oy + hl + 1.2, 0.3]),
                np.array([ox - hw - 1.2, oy + hl + 1.2, 0.3]),
            ]

            for corner in corners:
                d_to_corner = float(np.linalg.norm(corner[:2] - self.position[:2]))
                if d_to_corner < 18.0:
                    # Shadow alignment score: want corner behind obstacle relative to threat
                    corner_rel = corner[:2] - np.array([ox, oy])
                    shadow_alignment = float(np.dot(corner_rel, threat_unit))
                    score = shadow_alignment * 2.0 - d_to_corner * 0.8
                    if score > best_score:
                        best_score = score
                        best_corner = corner

        if best_corner is not None:
            return best_corner
        return self.waypoints[(self.current_wp_idx + 1) % len(self.waypoints)]


class EvasiveTargetManager:
    """Manages the full fleet of dynamic evasive ground targets."""

    def __init__(self, targets: Optional[List[EvasiveTarget]] = None):
        if targets is None:
            self.targets = self._create_default_targets()
        else:
            self.targets = targets

    def _create_default_targets(self) -> List[EvasiveTarget]:
        return [
            EvasiveTarget(
                target_id=0,
                name="Convoy Alpha",
                waypoints=[
                    np.array([16.0, -8.0, 0.35]), np.array([4.0, -14.0, 0.35]),
                    np.array([-8.0, -8.0, 0.35]), np.array([-20.0, 8.0, 0.35]),
                    np.array([8.0, 20.0, 0.35]), np.array([22.0, 6.0, 0.35]),
                ],
                base_speed=1.4,
                evasion_speed=2.2,
            ),
            EvasiveTarget(
                target_id=1,
                name="Fast Interceptor Bravo",
                waypoints=[
                    np.array([-22.0, 10.0, 0.30]), np.array([-12.0, 22.0, 0.30]),
                    np.array([10.0, 16.0, 0.30]), np.array([-4.0, -2.0, 0.30]),
                    np.array([-18.0, -14.0, 0.30]), np.array([-6.0, -18.0, 0.30]),
                ],
                base_speed=1.8,
                evasion_speed=2.5,
            ),
            EvasiveTarget(
                target_id=2,
                name="Shadow Asset Charlie",
                waypoints=[
                    np.array([8.0, 24.0, 0.30]), np.array([22.0, 16.0, 0.30]),
                    np.array([14.0, -10.0, 0.30]), np.array([-6.0, -18.0, 0.30]),
                    np.array([-16.0, 14.0, 0.30]), np.array([4.0, 6.0, 0.30]),
                ],
                base_speed=1.3,
                evasion_speed=2.0,
            ),
        ]

    def update_all(
        self,
        dt: float,
        sim_time: float,
        drone_positions: Dict[int, np.ndarray],
        obstacles: List[Dict[str, Any]],
        sensor_sightings: List[int],
    ) -> Dict[int, TargetTelemetry]:
        results = {}
        for target in self.targets:
            res = target.update(dt, sim_time, drone_positions, obstacles, sensor_sightings)
            results[target.target_id] = res
        return results

    def get_positions(self) -> Dict[int, np.ndarray]:
        return {t.target_id: t.position.copy() for t in self.targets}

    def get_velocities(self) -> Dict[int, np.ndarray]:
        return {t.target_id: t.velocity.copy() for t in self.targets}
