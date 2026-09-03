# -*- coding: utf-8 -*-
"""
test_ai_safety.py — Automated Tests for DeepSeek AI Authority Model,
Schema Validation, Speed Clamping, and Deterministic Fallback Handling.
"""

import pytest
from src.config.airframes import FLEET_CONFIGS
from src.ai_commander import sanitize_directive, DeepSeekSwarmCommander


def test_sanitize_directive_speed_clamping():
    """Verify that unphysically high LLM speeds are clamped to airframe physical max."""
    raw_json = {
        "strategic_posture": "AGGRESSIVE_PINCER",
        "target_priority": [0],
        "drone_assignments": {
            "0": {"role": "TRACKER", "target_id": 0, "desired_speed": 45.0},   # Max is 12.0 m/s
            "1": {"role": "FLANKER", "target_id": 0, "desired_speed": 100.0},  # Max is 18.0 m/s
            "2": {"role": "TRACKER", "target_id": 0, "desired_speed": -5.0},   # Below min (clamped to 1.0)
            "3": {"role": "TRACKER", "target_id": 0, "desired_speed": 30.0},   # Max is 8.0 m/s (must remain RELAY)
        }
    }

    directive = sanitize_directive(
        raw_data=raw_json,
        sim_time=5.0,
        known_target_ids={0, 1, 2},
    )

    assert directive.strategic_posture == "AGGRESSIVE_PINCER"
    assert directive.drone_assignments[0]["desired_speed"] == FLEET_CONFIGS[0].max_speed_mps
    assert directive.drone_assignments[1]["desired_speed"] == FLEET_CONFIGS[1].max_speed_mps
    assert directive.drone_assignments[2]["desired_speed"] == 1.0
    assert directive.drone_assignments[3]["role"] == "RELAY"  # Invariant: Drone 3 is always RELAY


def test_target_id_hallucination_pruning():
    """Hallucinated target IDs not in known_target_ids must be stripped."""
    raw_json = {
        "strategic_posture": "CONCENTRIC_CONTAINMENT",
        "target_priority": [99, 100, 1, 0, 777],  # 99, 100, 777 do not exist
        "drone_assignments": {},
    }

    directive = sanitize_directive(
        raw_data=raw_json,
        sim_time=10.0,
        known_target_ids={0, 1, 2},
    )

    assert directive.target_priority == [1, 0]


def test_deterministic_fallback_when_disabled():
    """When API is unavailable or disabled, commander must return deterministic baseline."""
    commander = DeepSeekSwarmCommander(api_key="")
    assert not commander.enabled

    directive = commander.get_latest_directive()
    assert directive is not None
    assert directive.is_fallback is True
    assert directive.strategic_posture == "COORDINATED_SWEEP"
    assert len(directive.drone_assignments) == 4
