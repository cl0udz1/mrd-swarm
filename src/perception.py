# -*- coding: utf-8 -*-
"""
perception.py — Perception & 3D Voxel Uncertainty Sub-Agent (Vectorized High-Performance)

Provides:
- 3D Voxel Uncertainty Grid: Represents spatial unknown volumes across the 45m x 45m x 15m theater.
- Vectorized Camera Frustum Raycast Coverage & Occlusion Checks.
- Fast NumPy-Accelerated Information Gain Estimator (<0.1ms per evaluation).
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from .physics import quat_to_rotation_matrix


def ray_intersects_box(
    ray_origin: np.ndarray,
    ray_dir: np.ndarray,
    box_min: np.ndarray,
    box_max: np.ndarray,
) -> Tuple[bool, float, float]:
    """Slab method for 3D ray-AABB intersection test."""
    t_min = 0.0
    t_max = 1.0

    for i in range(3):
        if abs(ray_dir[i]) < 1e-8:
            if ray_origin[i] < box_min[i] or ray_origin[i] > box_max[i]:
                return False, 0.0, 0.0
        else:
            inv_d = 1.0 / ray_dir[i]
            t1 = (box_min[i] - ray_origin[i]) * inv_d
            t2 = (box_max[i] - ray_origin[i]) * inv_d
            if t1 > t2:
                t1, t2 = t2, t1
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
            if t_min > t_max:
                return False, 0.0, 0.0

    return True, t_min, t_max


class LineOfSightSensor:
    """Computes optical/thermal visibility with 3D building occlusion."""

    def __init__(self, obstacles: List[Dict[str, Any]]):
        self.obstacles = obstacles
        self.building_boxes = []
        for obs in obstacles:
            ox, oy, oz = obs["pos"]
            hw, hl = obs["size"][:2]
            h = obs.get("height", 8.0)
            self.building_boxes.append((
                np.array([ox - hw, oy - hl, 0.0]),
                np.array([ox + hw, oy + hl, h]),
            ))

    def is_occluded(self, p_start: np.ndarray, p_end: np.ndarray) -> bool:
        ray_dir = p_end - p_start
        dist = np.linalg.norm(ray_dir)
        if dist < 1e-4:
            return False

        for b_min, b_max in self.building_boxes:
            hit, t1, t2 = ray_intersects_box(p_start, ray_dir, b_min, b_max)
            if hit and 0.02 < t1 < 0.98:
                return True
        return False

    def evaluate_target_visibility(
        self,
        drone_pos: np.ndarray,
        drone_quat: np.ndarray,
        target_pos: np.ndarray,
        fov_deg: float,
        max_range: float,
    ) -> Tuple[bool, float]:
        diff = target_pos - drone_pos
        dist = float(np.linalg.norm(diff))

        if dist > max_range:
            return False, 0.0

        R_b2w = quat_to_rotation_matrix(drone_quat)
        cam_fwd = R_b2w[:, 0]
        cos_angle = float(np.dot(cam_fwd, diff / (dist + 1e-6)))
        fov_limit = math.cos(math.radians(fov_deg / 2.0))

        if cos_angle < fov_limit:
            return False, 0.0

        if self.is_occluded(drone_pos, target_pos):
            return False, 0.0

        range_factor = 1.0 - (dist / max_range)
        angle_factor = (cos_angle - fov_limit) / (1.0 - fov_limit + 1e-6)
        conf = float(np.clip(range_factor * 0.6 + angle_factor * 0.4, 0.25, 0.99))
        return True, conf


class VoxelUncertaintyGrid:
    """
    3D Voxel Uncertainty Field with fast NumPy vectorized coverage and evaluation.
    """

    def __init__(
        self,
        x_bounds: Tuple[float, float] = (-22.5, 22.5),
        y_bounds: Tuple[float, float] = (-22.5, 22.5),
        z_bounds: Tuple[float, float] = (0.5, 14.0),
        resolution: float = 2.0,
        obstacles: Optional[List[Dict[str, Any]]] = None,
    ):
        self.x_min, self.x_max = x_bounds
        self.y_min, self.y_max = y_bounds
        self.z_min, self.z_max = z_bounds
        self.res = resolution

        self.nx = int(math.ceil((self.x_max - self.x_min) / self.res))
        self.ny = int(math.ceil((self.y_max - self.y_min) / self.res))
        self.nz = int(math.ceil((self.z_max - self.z_min) / self.res))

        self.grid = np.ones((self.nx, self.ny, self.nz), dtype=np.float32)

        # Coordinate arrays
        xs = np.linspace(self.x_min + self.res / 2, self.x_max - self.res / 2, self.nx)
        ys = np.linspace(self.y_min + self.res / 2, self.y_max - self.res / 2, self.ny)
        zs = np.linspace(self.z_min + self.res / 2, self.z_max - self.res / 2, self.nz)
        self.X, self.Y, self.Z = np.meshgrid(xs, ys, zs, indexing="ij")
        self.voxel_coords = np.stack([self.X, self.Y, self.Z], axis=-1)  # (nx, ny, nz, 3)

        self.los_sensor = LineOfSightSensor(obstacles if obstacles else [])

        # Persistent boolean mask: True for voxels that are free space (not inside buildings).
        # This mask NEVER changes, so explored voxels (value → 0) are still counted in metrics.
        self.free_space_mask = np.ones((self.nx, self.ny, self.nz), dtype=bool)

        # Zero out voxels inside solid buildings and mark them as non-free-space
        if obstacles:
            for obs in obstacles:
                ox, oy = obs["pos"][:2]
                hw, hl = obs["size"][:2]
                h = obs.get("height", 8.0)
                building_mask = (
                    (np.abs(self.X - ox) <= hw)
                    & (np.abs(self.Y - oy) <= hl)
                    & (self.Z <= h)
                )
                self.grid[building_mask] = 0.0
                self.free_space_mask[building_mask] = False

        self.n_free_voxels = int(np.sum(self.free_space_mask))

    def update_coverage(
        self,
        drone_pos: np.ndarray,
        drone_quat: np.ndarray,
        fov_deg: float,
        max_range: float,
        decay_rate: float = 0.80,
    ) -> int:
        """Vectorized camera frustum decay."""
        R_b2w = quat_to_rotation_matrix(drone_quat)
        cam_fwd = R_b2w[:, 0]
        fov_limit = math.cos(math.radians(fov_deg / 2.0))

        diff = self.voxel_coords - drone_pos
        dists_sq = np.sum(diff**2, axis=-1)
        in_range = dists_sq <= (max_range**2)

        if not np.any(in_range):
            return 0

        dists = np.sqrt(dists_sq[in_range]) + 1e-6
        cos_angles = np.sum(diff[in_range] * cam_fwd, axis=-1) / dists
        in_frustum = cos_angles >= fov_limit

        full_mask = np.zeros_like(in_range, dtype=bool)
        full_mask[in_range] = in_frustum

        # Decay voxels in frustum
        self.grid[full_mask] *= (1.0 - decay_rate)
        return int(np.sum(full_mask))

    def calculate_information_gain(
        self,
        candidate_pos: np.ndarray,
        candidate_yaw: float,
        fov_deg: float,
        max_range: float,
    ) -> float:
        """Fast vectorized information gain computation."""
        cam_fwd = np.array([math.cos(candidate_yaw), math.sin(candidate_yaw), -0.2], dtype=np.float32)
        cam_fwd /= (np.linalg.norm(cam_fwd) + 1e-6)
        fov_limit = math.cos(math.radians(fov_deg / 2.0))

        diff = self.voxel_coords - candidate_pos
        dists_sq = np.sum(diff**2, axis=-1)
        in_range = (dists_sq <= (max_range**2)) & (self.grid > 0.15)

        if not np.any(in_range):
            return 0.0

        dists = np.sqrt(dists_sq[in_range]) + 1e-6
        cos_angles = np.sum(diff[in_range] * cam_fwd, axis=-1) / dists
        in_frustum = cos_angles >= fov_limit

        return float(np.sum(self.grid[in_range][in_frustum]))

    def get_best_frontier(
        self,
        current_pos: np.ndarray,
        cruise_altitude: float = 4.0,
        fov_deg: float = 75.0,
        max_range: float = 25.0,
    ) -> Tuple[np.ndarray, float]:
        """Evaluates candidate exploration positions efficiently."""
        best_pos = current_pos.copy()
        best_pos[2] = cruise_altitude
        best_gain = -1.0

        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        step_distances = [6.0, 12.0]

        for r in step_distances:
            for theta in angles:
                cand_x = current_pos[0] + r * math.cos(theta)
                cand_y = current_pos[1] + r * math.sin(theta)

                if not (self.x_min + 3.0 <= cand_x <= self.x_max - 3.0 and self.y_min + 3.0 <= cand_y <= self.y_max - 3.0):
                    continue

                cand_pos = np.array([cand_x, cand_y, cruise_altitude])
                gain = self.calculate_information_gain(cand_pos, theta, fov_deg, max_range)
                score = gain / (1.0 + 0.05 * r)

                if score > best_gain:
                    best_gain = score
                    best_pos = cand_pos

        return best_pos, best_gain

    def get_mean_uncertainty(self) -> float:
        """Mean uncertainty over all free-space voxels (NOT inside buildings).
        
        Uses the persistent free_space_mask so that fully-explored voxels
        (uncertainty → 0.0) are still counted, driving the metric toward 0%.
        """
        if self.n_free_voxels == 0:
            return 0.0
        return float(np.mean(self.grid[self.free_space_mask]) * 100.0)

    def get_explored_pct(self) -> float:
        """Percentage of free-space voxels with uncertainty < 10%."""
        if self.n_free_voxels == 0:
            return 100.0
        explored = np.sum(self.grid[self.free_space_mask] < 0.10)
        return float(explored / self.n_free_voxels * 100.0)
