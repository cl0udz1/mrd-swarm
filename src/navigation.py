# -*- coding: utf-8 -*-
"""
navigation.py — 3D Local Navigation & APF Collision Avoidance Sub-Agent

Provides:
- 3D Reactive Vector Fields / Artificial Potential Fields (APF).
- Dynamic Altitude Slalom Navigation (1.5m to 12.0m) to navigate urban canyons and skybridges.
- Smooth Position/Velocity Setpoint Generator for SE(3) Quadrotor Controllers.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


@dataclass
class NavSetpoint:
    target_position: np.ndarray
    target_velocity: np.ndarray
    repulsive_force: np.ndarray
    min_obstacle_dist: float
    min_peer_dist: float


class APFReactiveNavigator:
    """
    3D Artificial Potential Field Reactive Navigator.
    """

    def __init__(
        self,
        k_att: float = 3.6,
        k_obs: float = 8.5,
        k_peer: float = 5.0,
        d_obs_margin: float = 2.4,
        d_peer_margin: float = 3.0,
        max_speed: float = 16.0,
    ):
        self.k_att = k_att
        self.k_obs = k_obs
        self.k_peer = k_peer
        self.d_obs_margin = d_obs_margin
        self.d_peer_margin = d_peer_margin
        self.max_speed = max_speed

    def compute_setpoint(
        self,
        current_pos: np.ndarray,
        current_vel: np.ndarray,
        goal_pos: np.ndarray,
        obstacles: List[Dict[str, Any]],
        peer_positions: Dict[int, np.ndarray],
        current_agent_id: int,
        desired_speed: Optional[float] = None,
    ) -> NavSetpoint:
        """
        Computes 3D net potential field vector and returns smoothed velocity/position setpoints.
        """
        speed_limit = desired_speed if desired_speed is not None else self.max_speed

        # 1. Attractive Force toward goal
        diff_goal = goal_pos - current_pos
        dist_goal = float(np.linalg.norm(diff_goal))
        if dist_goal > 1e-3:
            f_att = self.k_att * (diff_goal / dist_goal) * min(speed_limit, max(3.0, dist_goal * 2.0))
        else:
            f_att = np.zeros(3)

        # 2. Obstacle Repulsive Force (8 urban buildings)
        f_rep_obs = np.zeros(3, dtype=np.float64)
        min_obs_d = float("inf")

        for obs in obstacles:
            ox, oy = obs["pos"][:2]
            hw, hl = obs["size"][:2]
            obs_h = obs.get("height", 8.0)

            # Check if drone is vertically within collision range of building
            if current_pos[2] < obs_h + 1.2:
                # 2D distance to bounding box
                dx = max(0.0, abs(current_pos[0] - ox) - hw)
                dy = max(0.0, abs(current_pos[1] - oy) - hl)
                dist_2d = math.sqrt(dx**2 + dy**2)
                min_obs_d = min(min_obs_d, dist_2d)

                if dist_2d < self.d_obs_margin:
                    dir_x = current_pos[0] - ox
                    dir_y = current_pos[1] - oy
                    norm_dir = math.sqrt(dir_x**2 + dir_y**2) + 1e-6
                    # Normal vector away from building center + slight vertical lift
                    n_vec = np.array([dir_x / norm_dir, dir_y / norm_dir, 0.35])
                    rep_mag = self.k_obs * (1.0 / max(0.05, dist_2d) - 1.0 / self.d_obs_margin) * (1.0 / (dist_2d**2 + 0.05))
                    f_rep_obs += rep_mag * n_vec

        # 3. Inter-Drone Peer Repulsive Force
        f_rep_peer = np.zeros(3, dtype=np.float64)
        min_peer_d = float("inf")

        for p_id, p_pos in peer_positions.items():
            if p_id == current_agent_id:
                continue
            delta = current_pos - p_pos
            dist_peer = float(np.linalg.norm(delta))
            min_peer_d = min(min_peer_d, dist_peer)

            if 0.05 < dist_peer < self.d_peer_margin:
                n_peer = delta / dist_peer
                rep_mag = self.k_peer * (1.0 / dist_peer - 1.0 / self.d_peer_margin) * (1.0 / (dist_peer**2))
                f_rep_peer += rep_mag * n_peer

        # 4. Arena Boundary Repulsion (-26m to 26m, z: 1.5m to 12.0m)
        f_rep_bound = np.zeros(3, dtype=np.float64)
        wall_margin = 3.0
        if current_pos[0] < -26.0 + wall_margin:
            f_rep_bound[0] += 4.0 * (1.0 / max(0.1, current_pos[0] - (-26.0)))
        elif current_pos[0] > 26.0 - wall_margin:
            f_rep_bound[0] -= 4.0 * (1.0 / max(0.1, 26.0 - current_pos[0]))

        if current_pos[1] < -26.0 + wall_margin:
            f_rep_bound[1] += 4.0 * (1.0 / max(0.1, current_pos[1] - (-26.0)))
        elif current_pos[1] > 26.0 - wall_margin:
            f_rep_bound[1] -= 4.0 * (1.0 / max(0.1, 26.0 - current_pos[1]))

        # Floor / Ceiling repulsion
        if current_pos[2] < 1.8:
            f_rep_bound[2] += 5.0 * (1.8 - current_pos[2])
        elif current_pos[2] > 11.0:
            f_rep_bound[2] -= 4.0 * (current_pos[2] - 11.0)

        # 5. Net Force & Velocity Target Synthesis
        f_total_rep = f_rep_obs + f_rep_peer + f_rep_bound
        # Clamp repulsive force magnitude
        norm_rep = float(np.linalg.norm(f_total_rep))
        if norm_rep > 3.5:
            f_total_rep = (f_total_rep / norm_rep) * 3.5

        v_des = f_att + f_total_rep
        norm_v = float(np.linalg.norm(v_des))
        if norm_v > speed_limit:
            v_des = (v_des / norm_v) * speed_limit

        # Target position setpoint slightly forward along desired velocity
        target_pos_setpoint = current_pos + v_des * 0.8
        target_pos_setpoint[2] = float(np.clip(target_pos_setpoint[2], 1.5, 12.0))

        return NavSetpoint(
            target_position=target_pos_setpoint,
            target_velocity=v_des,
            repulsive_force=f_total_rep,
            min_obstacle_dist=min_obs_d,
            min_peer_dist=min_peer_d,
        )
