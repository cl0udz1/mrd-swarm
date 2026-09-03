# -*- coding: utf-8 -*-
"""
mission_state.py — Multi-Phase Mission State Machine

Drives the swarm through structured operational phases:
    LAUNCH → AREA_SWEEP → HUNT → CONTAIN → MISSION_COMPLETE
                ↑            ↓
                ← LOST_TARGET_RECOVERY
"""

from __future__ import annotations
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Dict, Set
import numpy as np


class MissionPhase(IntEnum):
    LAUNCH = 0
    AREA_SWEEP = 1
    HUNT = 2
    CONTAIN = 3
    LOST_TARGET_RECOVERY = 4
    MISSION_COMPLETE = 5


@dataclass
class MissionState:
    """Global mission state shared across all drones."""
    phase: MissionPhase = MissionPhase.LAUNCH
    phase_start_time: float = 0.0
    phase_duration: float = 0.0

    # Phase-specific counters
    drones_at_altitude: Set[int] = field(default_factory=set)
    targets_under_track: Set[int] = field(default_factory=set)
    cumulative_track_time: Dict[int, float] = field(default_factory=dict)
    lost_target_ids: Set[int] = field(default_factory=set)

    # Mission statistics
    total_phase_transitions: int = 0
    phase_history: list = field(default_factory=list)


class MissionStateManager:
    """
    Evaluates global swarm state and triggers phase transitions.

    Transition Rules:
        LAUNCH → AREA_SWEEP:           All drones above cruise altitude
        AREA_SWEEP → HUNT:             ≥1 target detected OR uncertainty < 5%
        HUNT → CONTAIN:                All 3 targets under active EKF track
        HUNT → LOST_TARGET_RECOVERY:   Any confirmed track goes to LOST state
        LOST_TARGET_RECOVERY → HUNT:   Lost target re-acquired
        CONTAIN → MISSION_COMPLETE:    All targets tracked for ≥30s cumulative
    """

    def __init__(self, n_targets: int = 3, cruise_altitude: float = 1.0):
        self.state = MissionState()
        self.n_targets = n_targets
        self.cruise_altitude = cruise_altitude
        for tid in range(n_targets):
            self.state.cumulative_track_time[tid] = 0.0

    @property
    def phase(self) -> MissionPhase:
        return self.state.phase

    def _transition(self, new_phase: MissionPhase, sim_time: float) -> None:
        """Execute a phase transition with logging."""
        old_phase = self.state.phase
        self.state.phase_duration = sim_time - self.state.phase_start_time
        self.state.phase_history.append({
            "from": old_phase.name,
            "to": new_phase.name,
            "at_time": round(sim_time, 2),
            "duration": round(self.state.phase_duration, 2),
        })
        self.state.phase = new_phase
        self.state.phase_start_time = sim_time
        self.state.total_phase_transitions += 1

    def evaluate_transitions(
        self,
        sim_time: float,
        drone_altitudes: Dict[int, float],
        detected_target_ids: Set[int],
        tracked_target_ids: Set[int],
        lost_target_ids: Set[int],
        uncertainty_pct: float,
        dt: float,
    ) -> MissionPhase:
        """
        Evaluate phase transition conditions and return current phase.
        """
        phase = self.state.phase

        # Update cumulative track times for tracked targets
        for tid in tracked_target_ids:
            if tid in self.state.cumulative_track_time:
                self.state.cumulative_track_time[tid] += dt
        self.state.targets_under_track = tracked_target_ids
        self.state.lost_target_ids = lost_target_ids

        if phase == MissionPhase.LAUNCH:
            for did, alt in drone_altitudes.items():
                if alt >= self.cruise_altitude:
                    self.state.drones_at_altitude.add(did)
            if len(self.state.drones_at_altitude) >= len(drone_altitudes):
                self._transition(MissionPhase.AREA_SWEEP, sim_time)

        elif phase == MissionPhase.AREA_SWEEP:
            if len(detected_target_ids) > 0 or uncertainty_pct < 5.0:
                self._transition(MissionPhase.HUNT, sim_time)

        elif phase == MissionPhase.HUNT:
            if len(tracked_target_ids) >= self.n_targets:
                self._transition(MissionPhase.CONTAIN, sim_time)
            elif len(lost_target_ids) > 0:
                self._transition(MissionPhase.LOST_TARGET_RECOVERY, sim_time)

        elif phase == MissionPhase.CONTAIN:
            all_tracked_enough = all(
                self.state.cumulative_track_time.get(tid, 0.0) >= 30.0
                for tid in range(self.n_targets)
            )
            if all_tracked_enough:
                self._transition(MissionPhase.MISSION_COMPLETE, sim_time)
            if len(lost_target_ids) > 0:
                self._transition(MissionPhase.LOST_TARGET_RECOVERY, sim_time)

        elif phase == MissionPhase.LOST_TARGET_RECOVERY:
            if len(lost_target_ids) == 0:
                self._transition(MissionPhase.HUNT, sim_time)
            if (sim_time - self.state.phase_start_time) > 15.0:
                self._transition(MissionPhase.HUNT, sim_time)

        return self.state.phase

    def get_status_summary(self) -> Dict:
        """Return mission state summary for telemetry."""
        return {
            "phase": self.state.phase.name,
            "phase_time": round(self.state.phase_duration, 1),
            "transitions": self.state.total_phase_transitions,
            "targets_tracked": list(self.state.targets_under_track),
            "targets_lost": list(self.state.lost_target_ids),
            "cumulative_track_s": {
                tid: round(t, 1) for tid, t in self.state.cumulative_track_time.items()
            },
        }
