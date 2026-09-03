# -*- coding: utf-8 -*-
"""
perception.py — 3D Voxel Epistemic Uncertainty Grid & Raycast Line-of-Sight Sensing

Provides:
- LineOfSightSensor: Ray-box intersection testing for urban obstacle occlusions.
- VoxelUncertaintyGrid: 3D volumetric epistemic uncertainty field U(x, y, z) with
  vectorized camera frustum decay and rigorous building occlusion raycasting.
"""

from __future__ import annotations
import math
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from numpy.typing import NDArray

from .physics import quat_to_rotation_matrix


def ray_intersects_box(
    ray_origin: np.ndarray,
    ray_dir: np.ndarray,
    box_min: np.ndarray,
    box_max: np.ndarray,
) -> Tuple[bool, float, float]:
    """
    Kay-Kajiya slab method for ray-AABB intersection.
    Returns (hit, t_min, t_max) along ray_origin + t * ray_dir.
    """
    safe_dir = np.where(np.abs(ray_dir) < 1e-7, 1e-7, ray_dir)
    inv_dir = 1.0 / safe_dir
    t1 = (box_min - ray_origin) * inv_dir
    t2 = (box_max - ray_origin) * inv_dir

    t_near = np.minimum(t1, t2)
    t_far = np.maximum(t1, t2)

    if ray_dir.ndim == 1:
        t_enter = float(np.max(t_near))
        t_exit = float(np.min(t_far))
        hit = (t_exit >= t_enter) and (t_exit >= 0.0)
        return hit, t_enter, t_exit
    else:
        t_enter = np.max(t_near, axis=-1)
        t_exit = np.min(t_far, axis=-1)
        hit = (t_exit >= t_enter) & (t_exit >= 0.0)
        return hit, t_enter, t_exit


class LineOfSightSensor:
    """Performs building occlusion raycasts across urban architecture."""

    def __init__(self, obstacles: List[Dict[str, Any]]):
        self.obstacles = obstacles
        self.building_boxes: List[Tuple[np.ndarray, np.ndarray]] = []
        for obs in obstacles:
            ox, oy = obs["pos"][:2]
            hw, hl = obs["size"][:2]
            h = obs.get("height", obs["size"][2] * 2.0 if len(obs["size"]) > 2 else 8.0)
            self.building_boxes.append((
                np.array([ox - hw, oy - hl, 0.0], dtype=np.float64),
                np.array([ox + hw, oy + hl, h], dtype=np.float64),
            ))

    def is_occluded(self, p_start: np.ndarray, p_end: np.ndarray) -> bool:
        """Returns True if the line segment between p_start and p_end hits a building."""
        ray_dir = p_end - p_start
        dist = float(np.linalg.norm(ray_dir))
        if dist < 1e-4:
            return False

        for b_min, b_max in self.building_boxes:
            hit, t1, t2 = ray_intersects_box(p_start, ray_dir, b_min, b_max)
            # Must intersect strictly between start and end point (t in [0.02, 0.98])
            if hit and (0.02 < t1 < 0.98):
                return True
        return False

    def batch_is_occluded(self, p_start: np.ndarray, p_targets: np.ndarray) -> np.ndarray:
        """Vectorized occlusion testing from p_start to N target points. Shape (N,) bool."""
        n_pts = p_targets.shape[0]
        if n_pts == 0:
            return np.zeros(0, dtype=bool)

        ray_dirs = p_targets - p_start
        dists = np.linalg.norm(ray_dirs, axis=-1, keepdims=True)
        valid = (dists.squeeze(-1) > 1e-4)
        occluded = np.zeros(n_pts, dtype=bool)

        for b_min, b_max in self.building_boxes:
            hit, t1, t2 = ray_intersects_box(p_start, ray_dirs, b_min, b_max)
            blocked = hit & valid & (t1 > 0.02) & (t1 < 0.98)
            occluded |= blocked

        return occluded

    def evaluate_target_visibility(
        self,
        drone_pos: np.ndarray,
        drone_quat: np.ndarray,
        target_pos: np.ndarray,
        fov_deg: float,
        max_range: float,
    ) -> Tuple[bool, float]:
        """
        Check if target is inside camera frustum, within range, and not occluded by buildings.
        Returns (is_visible, confidence).
        """
        diff = target_pos - drone_pos
        dist = float(np.linalg.norm(diff))
        if dist > max_range or dist < 0.2:
            return False, 0.0

        R_b2w = quat_to_rotation_matrix(drone_quat)
        cam_fwd = R_b2w[:, 0]
        cos_ang = float(np.dot(cam_fwd, diff / dist))
        fov_limit = math.cos(math.radians(fov_deg / 2.0))
        if cos_ang < fov_limit:
            return False, 0.0

        if self.is_occluded(drone_pos, target_pos):
            return False, 0.0

        range_factor = max(0.0, 1.0 - (dist / max_range))
        angle_factor = max(0.0, (cos_ang - fov_limit) / (1.0 - fov_limit + 1e-6))
        conf = float(np.clip(0.6 * range_factor + 0.4 * angle_factor, 0.25, 0.98))
        return True, conf



class VoxelUncertaintyGrid:
    """
    3D Voxel Uncertainty Field with building occlusion raycasting.
    Only voxels with unobstructed line-of-sight to the camera are decayed.
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

        # Coordinate meshgrid
        xs = np.linspace(self.x_min + self.res / 2, self.x_max - self.res / 2, self.nx)
        ys = np.linspace(self.y_min + self.res / 2, self.y_max - self.res / 2, self.ny)
        zs = np.linspace(self.z_min + self.res / 2, self.z_max - self.res / 2, self.nz)
        self.X, self.Y, self.Z = np.meshgrid(xs, ys, zs, indexing="ij")
        self.voxel_coords = np.stack([self.X, self.Y, self.Z], axis=-1)  # (nx, ny, nz, 3)

        self.los_sensor = LineOfSightSensor(obstacles if obstacles else [])

        # Free space mask: zero out solid voxels inside buildings
        self.free_space_mask = np.ones((self.nx, self.ny, self.nz), dtype=bool)
        if obstacles:
            for obs in obstacles:
                ox, oy = obs["pos"][:2]
                hw, hl = obs["size"][:2]
                h = obs.get("height", 8.0)
                b_mask = (
                    (np.abs(self.X - ox) <= hw)
                    & (np.abs(self.Y - oy) <= hl)
                    & (self.Z <= h)
                )
                self.grid[b_mask] = 0.0
                self.free_space_mask[b_mask] = False

        self.n_free_voxels = int(np.sum(self.free_space_mask))

    def update_coverage(
        self,
        drone_pos: np.ndarray,
        drone_quat: np.ndarray,
        fov_deg: float,
        max_range: float,
        decay_rate: float = 0.80,
    ) -> int:
        """
        Decay uncertainty for voxels inside camera frustum THAT ARE NOT OCCLUDED by buildings.
        Returns the number of unobstructed voxels updated.
        """
        R_b2w = quat_to_rotation_matrix(drone_quat)
        cam_fwd = R_b2w[:, 0]
        fov_limit = math.cos(math.radians(fov_deg / 2.0))

        diff = self.voxel_coords - drone_pos
        dists_sq = np.sum(diff**2, axis=-1)
        in_range = (dists_sq <= (max_range**2)) & self.free_space_mask

        if not np.any(in_range):
            return 0

        dists = np.sqrt(dists_sq[in_range]) + 1e-6
        cos_angles = np.sum(diff[in_range] * cam_fwd, axis=-1) / dists
        in_frustum = cos_angles >= fov_limit

        if not np.any(in_frustum):
            return 0

        # Candidate indices in flat array
        cand_indices = np.nonzero(in_range)
        cand_coords = self.voxel_coords[cand_indices]  # (K, 3)
        frustum_coords = cand_coords[in_frustum]        # (M, 3)

        # Vectorized occlusion raycasting against solid buildings
        occluded = self.los_sensor.batch_is_occluded(drone_pos, frustum_coords)
        visible_mask = ~occluded

        # Map back to full grid
        frustum_indices = tuple(c[in_frustum][visible_mask] for c in cand_indices)
        self.grid[frustum_indices] *= (1.0 - decay_rate)

        return int(np.sum(visible_mask))

    def calculate_information_gain(self, candidate_pos: np.ndarray, fov_deg: float, max_range: float) -> float:
        """Estimates potential uncertainty reduction from candidate position."""
        diff = self.voxel_coords - candidate_pos
        dists_sq = np.sum(diff**2, axis=-1)
        in_range = (dists_sq <= (max_range**2)) & self.free_space_mask
        if not np.any(in_range):
            return 0.0
        return float(np.sum(self.grid[in_range]))

    def get_mean_uncertainty(self) -> float:
        """Mean uncertainty across all navigable free-space voxels (0 to 100%)."""
        if self.n_free_voxels == 0:
            return 0.0
        return float((np.sum(self.grid[self.free_space_mask]) / self.n_free_voxels) * 100.0)

    def get_coverage_pct(self, threshold: float = 0.15) -> float:
        """Percentage of free-space voxels thoroughly explored (uncertainty < threshold)."""
        if self.n_free_voxels == 0:
            return 100.0
        cleared = (self.grid[self.free_space_mask] < threshold)
        return float((np.sum(cleared) / self.n_free_voxels) * 100.0)

    def get_best_frontier(
        self,
        drone_pos: np.ndarray,
        cruise_altitude: float = 3.5,
        n_candidates: int = 16,
    ) -> Tuple[np.ndarray, float]:
        """
        Identifies highest-information frontier waypoint in free space.
        Returns (best_waypoint_3d, expected_gain).
        """
        angles = np.linspace(0, 2 * np.pi, n_candidates, endpoint=False)
        radii = [8.0, 14.0]

        best_p = drone_pos.copy()
        best_p[2] = cruise_altitude
        max_gain = -1.0

        for r in radii:
            for ang in angles:
                cx = float(np.clip(drone_pos[0] + r * math.cos(ang), self.x_min + 3.0, self.x_max - 3.0))
                cy = float(np.clip(drone_pos[1] + r * math.sin(ang), self.y_min + 3.0, self.y_max - 3.0))
                cand_pos = np.array([cx, cy, cruise_altitude], dtype=np.float64)

                if not self.los_sensor.is_occluded(drone_pos, cand_pos):
                    gain = self.calculate_information_gain(cand_pos, fov_deg=80.0, max_range=15.0)
                    dist = float(np.linalg.norm(cand_pos[:2] - drone_pos[:2]))
                    score = gain / (1.0 + 0.1 * dist)
                    if score > max_gain:
                        max_gain = score
                        best_p = cand_pos

        return best_p, max(0.0, max_gain)

