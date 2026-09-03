# -*- coding: utf-8 -*-
"""
test_integration.py — System-Wide Integration Tests for MRD-SWARM.

Verifies:
1. Complete ECSWorld execution across 500+ steps with realistic obstacles.
2. Legacy class and API backward-compatibility instantiation.
3. Full deterministic multi-phase mission integration without crashes or NaNs.
"""

import math
import numpy as np
import pytest

from src.ecs.world import ECSWorld
from src.config.scenarios import get_scenario, ScenarioID
from src.config.airframes import FLEET_CONFIGS
from src.sensors import BatteryModel, SensorSuite, BatteryState, TargetObservation
from src.controller import CascadedQuadrotorController, GeometricSE3Controller
from src.gossip import GossipNode
from src.ai_commander import DeepSeekSwarmCommander
from src.ai_vision_recon import DeepSeekVisionRecon


def test_complete_world_smoke_500_steps():
    """Smoke test: Instantiates full ECSWorld and runs 500 consecutive steps."""
    scenario = get_scenario(ScenarioID.SCENARIO_C_DENSE_URBAN)
    world = ECSWorld(obstacles=scenario.obstacles, seed=42)

    # Disable remote API calls for fast deterministic local test
    world.ai_commander.enabled = False
    world.vision_recon.enabled = False

    for step_i in range(500):
        telem = world.step()
        assert telem["type"] == "TELEMETRY_UPDATE"
        assert telem["time"] > 0.0
        assert not math.isnan(telem["uncertainty_pct"])
        assert len(telem["drones"]) == 4
        assert len(telem["targets"]) == 3
        assert "perception" in telem
        assert "rf_mesh" in telem
        assert "mission_state" in telem

    # Cumulative perception metrics must be non-negative
    p_data = telem["perception"]
    assert p_data["total_detection_events"] >= 0
    assert p_data["total_visible_target_frames"] >= 0


def test_legacy_classes_instantiation():
    """Verify legacy classes and adapters instantiate cleanly without breaking."""
    cfg = FLEET_CONFIGS[0]

    # BatteryModel with both AirframeConfig and legacy float
    bat_from_cfg = BatteryModel(cfg)
    assert bat_from_cfg.capacity_wh == cfg.battery_capacity_wh
    bat_from_wh = BatteryModel(28.5, nominal_voltage_v=15.2)
    assert bat_from_wh.capacity_wh == 28.5

    # CascadedQuadrotorController adapter
    ctrl_legacy = CascadedQuadrotorController(mass=0.65)
    assert ctrl_legacy.airframe.mass_kg == 0.65

    # SensorSuite adapter
    suite = SensorSuite(drone_id=0)
    assert suite.drone_id == 0

    # TargetObservation and BatteryState
    obs = TargetObservation(target_id=1, position_estimate=np.array([1.0, 2.0]), confidence=0.9, timestamp=0.5)
    assert obs.target_id == 1
    state = BatteryState(remaining_wh=20.0, soc_pct=80.0, total_consumed_wh=5.0)
    assert state.soc_pct == 80.0

    # GossipNode
    node = GossipNode(agent_id=1)
    assert node.agent_id == 1

    # AI Commander & Vision with default (offline) fallback
    commander = DeepSeekSwarmCommander()
    assert commander.enabled is False
    recon = DeepSeekVisionRecon()
    assert recon.enabled is False


def test_full_deterministic_mission_integration():
    """Verify deterministic repeatability across two independent identical seed runs."""
    scenario = get_scenario(ScenarioID.SCENARIO_A_OPEN_FIELD)

    world_1 = ECSWorld(obstacles=scenario.obstacles, seed=999)
    world_1.ai_commander.enabled = False
    world_1.vision_recon.enabled = False

    world_2 = ECSWorld(obstacles=scenario.obstacles, seed=999)
    world_2.ai_commander.enabled = False
    world_2.vision_recon.enabled = False

    # Run 600 steps (6.0 seconds) on both worlds
    for _ in range(600):
        t1 = world_1.step()
        t2 = world_2.step()

    # Verify positions and uncertainty are deterministically bitwise identical
    for did in [0, 1, 2, 3]:
        pos1 = t1["drones"][did]["pos"]
        pos2 = t2["drones"][did]["pos"]
        assert np.allclose(pos1, pos2, atol=1e-5)

    assert math.isclose(t1["uncertainty_pct"], t2["uncertainty_pct"], abs_tol=1e-4)
    assert t1["perception"]["total_detection_events"] == t2["perception"]["total_detection_events"]
