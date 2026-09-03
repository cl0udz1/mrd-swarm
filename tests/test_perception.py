# -*- coding: utf-8 -*-
"""
test_perception.py — Automated Tests for Line-of-Sight Raycasting, Building Occlusion,
Thermal vs Optical Smoke Penetration, and Voxel Uncertainty Decay.
"""

import numpy as np
import pytest

from src.perception import LineOfSightSensor, VoxelUncertaintyGrid
from src.sensors import SyntheticTargetSensor, NoisyTargetMeasurement


# Standard test obstacles (one skyscraper at origin)
TEST_OBSTACLES = [
    {"name": "Central Skyscraper", "pos": [0.0, 0.0, 5.0], "size": [4.0, 4.0, 5.0], "height": 10.0},
]


def test_line_of_sight_occlusion():
    """Verify ray-box intersection blocks visibility through buildings."""
    los = LineOfSightSensor(TEST_OBSTACLES)

    # Ray across building from -10 to +10 through (0, 0)
    p_start = np.array([-10.0, 0.0, 2.0])
    p_end = np.array([10.0, 0.0, 2.0])
    assert los.is_occluded(p_start, p_end) is True

    # Ray clear of building
    p_clear_start = np.array([-10.0, 15.0, 2.0])
    p_clear_end = np.array([10.0, 15.0, 2.0])
    assert los.is_occluded(p_clear_start, p_clear_end) is False

    # Ray over the building (altitude 15m > height 10m)
    p_high_start = np.array([-10.0, 0.0, 15.0])
    p_high_end = np.array([10.0, 0.0, 15.0])
    assert los.is_occluded(p_high_start, p_high_end) is False


def test_voxel_grid_occlusion_preservation():
    """Voxels occluded behind a solid building must NOT be decayed."""
    grid = VoxelUncertaintyGrid(
        x_bounds=(-20.0, 20.0),
        y_bounds=(-20.0, 20.0),
        z_bounds=(1.0, 10.0),
        resolution=2.0,
        obstacles=TEST_OBSTACLES,
    )

    # Drone at (-10, 0, 3) facing +X towards the central building at (0, 0)
    drone_pos = np.array([-10.0, 0.0, 3.0])
    drone_quat = np.array([1.0, 0.0, 0.0, 0.0])  # facing +X

    initial_mean = grid.get_mean_uncertainty()
    updated_count = grid.update_coverage(
        drone_pos=drone_pos,
        drone_quat=drone_quat,
        fov_deg=90.0,
        max_range=25.0,
        decay_rate=0.80,
    )
    assert updated_count > 0

    # Voxel directly in front of drone (visible)
    # (-6, 0, 3) is in front of the building
    # Find grid coordinate for (-6, 0, 3)
    ix_front = int(round((-6.0 - grid.x_min) / grid.res)) - 1
    iy_center = int(round((0.0 - grid.y_min) / grid.res)) - 1
    iz_mid = 1
    val_front = grid.grid[ix_front, iy_center, iz_mid]

    # Voxel on the OTHER side of the skyscraper (occluded)
    # (+8, 0, 3) is behind the building relative to the drone
    ix_back = int(round((8.0 - grid.x_min) / grid.res)) - 1
    val_back = grid.grid[ix_back, iy_center, iz_mid]

    # Visible voxel in front should have decayed significantly (< 0.5)
    assert val_front < 0.5
    # Occluded voxel behind building MUST REMAIN UN-DECAYED (1.0)
    assert np.isclose(val_back, 1.0, atol=1e-4)


def test_synthetic_sensor_smoke_and_noise():
    """Verify thermal sensor penetrates smoke, optical is blocked, and measurements contain noise."""
    los = LineOfSightSensor([])

    # Drone 0: Optical EO scout (no thermal)
    sensor_eo = SyntheticTargetSensor(drone_id=0, dropout_probability=0.0, seed=42)
    # Drone 2: Thermal surveyor (has thermal)
    sensor_flir = SyntheticTargetSensor(drone_id=2, dropout_probability=0.0, seed=42)

    drone_pos = np.array([0.0, 0.0, 3.0])
    drone_quat = np.array([1.0, 0.0, 0.0, 0.0])
    target_pos = np.array([5.0, 0.0, 0.3])

    # 1. Clear conditions: both detect target
    meas_eo = sensor_eo.observe(drone_pos, drone_quat, target_id=0, true_target_pos=target_pos, is_occluded_fn=los.is_occluded, target_smoke_active=False)
    meas_flir = sensor_flir.observe(drone_pos, drone_quat, target_id=0, true_target_pos=target_pos, is_occluded_fn=los.is_occluded, target_smoke_active=False)

    assert meas_eo is not None
    assert meas_flir is not None
    # Verify noise: measurement must NOT equal perfect ground truth
    assert not np.allclose(meas_eo.measured_pos_2d, target_pos[:2], atol=1e-6)

    # 2. Smoke active conditions: EO blinded, FLIR penetrates
    meas_eo_smoke = sensor_eo.observe(drone_pos, drone_quat, target_id=0, true_target_pos=target_pos, is_occluded_fn=los.is_occluded, target_smoke_active=True)
    meas_flir_smoke = sensor_flir.observe(drone_pos, drone_quat, target_id=0, true_target_pos=target_pos, is_occluded_fn=los.is_occluded, target_smoke_active=True)

    assert meas_eo_smoke is None          # Optical completely blinded by smoke aerosol
    assert meas_flir_smoke is not None     # Thermal IR penetrates smoke successfully
    assert meas_flir_smoke.is_thermal is True
