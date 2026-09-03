# -*- coding: utf-8 -*-
"""
swarm_brain.py — Swarm Tactical Brain & In-Loop Decision Sub-Agent

Runs inside the simulation loop at 2 Hz - 5 Hz to dynamically:
- Allocate roles: EXPLORER, TRACKER, FLANKER/PINCER, RELAY, LOST_TARGET_SWEEP.
- Coordinate multi-drone pincer maneuvers to box in evasive targets.
- Launch localized spiral sweeps when targets break line-of-sight behind buildings.
- Maintain high-altitude RF mesh connectivity.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from .perception import VoxelUncertaintyGrid, LineOfSightSensor


class TacticalRole(Enum):
    EXPLORER = "EXPLORER"
    TRACKER = "TRACKER"
    FLANKER = "FLANKER"
    RELAY = "RELAY"
    LOST_TARGET_SWEEP = "LOST_TARGET_SWEEP"
    BASE_RECOVERY = "BASE_RECOVERY"


@dataclass
class SwarmDirective:
    agent_id: int
    role: TacticalRole
    goal_position: np.ndarray
    desired_speed: float
    target_id: Optional[int] = None
    lookat_heading: Optional[float] = None
    reasoning: str = ""


@dataclass
class TargetBelief:
    target_id: int
    last_position: np.ndarray
    last_velocity: np.ndarray
    last_seen_time: float
    confidence: float
    assigned_tracker: Optional[int] = None
    assigned_flanker: Optional[int] = None
    lost: bool = False


class SwarmTacticalBrain:
    """
    Decentralized in-loop cognitive coordinator for the 4-drone reconnaissance fleet.
    """

    def __init__(self, obstacles: List[Dict[str, Any]]):
        self.obstacles = obstacles
        self.target_beliefs: Dict[int, TargetBelief] = {}
        self.spiral_phase: Dict[int, float] = {i: 0.0 for i in range(4)}
        self.last_decision_time = 0.0

    def evaluate(
        self,
        sim_time: float,
        drone_telemetry: Dict[int, Dict[str, Any]],
        sensor_sightings: Dict[int, Dict[int, float]],  # drone_id -> {target_id: conf}
        target_positions: Dict[int, np.ndarray],
        target_velocities: Dict[int, np.ndarray],
        uncertainty_grid: VoxelUncertaintyGrid,
    ) -> Dict[int, SwarmDirective]:
        """
        Main cognitive evaluation step.
        """
        # 1. Update Target Beliefs from Sensor Sightings
        all_spotted_targets = set()
        for d_id, sightings in sensor_sightings.items():
            for t_id, conf in sightings.items():
                all_spotted_targets.add(t_id)
                t_pos = target_positions[t_id].copy()
                t_vel = target_velocities[t_id].copy()

                if t_id not in self.target_beliefs:
                    self.target_beliefs[t_id] = TargetBelief(
                        target_id=t_id,
                        last_position=t_pos,
                        last_velocity=t_vel,
                        last_seen_time=sim_time,
                        confidence=conf,
                        lost=False,
                    )
                else:
                    tb = self.target_beliefs[t_id]
                    tb.last_position = t_pos
                    tb.last_velocity = t_vel
                    tb.last_seen_time = sim_time
                    tb.confidence = conf
                    tb.lost = False

        # Mark lost targets (unseen for > 1.5 seconds)
        for t_id, tb in self.target_beliefs.items():
            if t_id not in all_spotted_targets and (sim_time - tb.last_seen_time) > 1.5:
                tb.lost = True
                tb.confidence = max(0.0, tb.confidence - 0.2)

        # 2. Dynamic Role Arbitration & Directives Synthesis
        directives: Dict[int, SwarmDirective] = {}

        # Sort targets by priority (visible targets first)
        active_targets = [tb for tb in self.target_beliefs.values() if not tb.lost and tb.confidence > 0.25]
        lost_targets = [tb for tb in self.target_beliefs.values() if tb.lost and (sim_time - tb.last_seen_time) < 8.0]

        # Reset assignments
        for tb in self.target_beliefs.values():
            tb.assigned_tracker = None
            tb.assigned_flanker = None

        # ── Drone 3: High-Altitude Comms & Relay Anchor ────────────────────────
        # Positions at Z=9.5m above the central corridor to bridge mesh across high-rises
        p3 = drone_telemetry[3]["position"]
        other_positions = [drone_telemetry[i]["position"] for i in range(3)]
        swarm_centroid = np.mean(other_positions, axis=0) if other_positions else np.zeros(3)
        relay_goal = np.array([swarm_centroid[0] * 0.4, swarm_centroid[1] * 0.4, 9.5])

        directives[3] = SwarmDirective(
            agent_id=3,
            role=TacticalRole.RELAY,
            goal_position=relay_goal,
            desired_speed=2.2,
            reasoning=f"Loitering at high altitude (Z=9.5m) to maintain RF mesh bridge across skyscraper corridor",
        )

        # ── Tactical Directives for Tactical Drones (0, 1, 2) ──────────────────
        # Priority A: If an active target is spotted, coordinate TRACKER and FLANKER pincer
        if len(active_targets) > 0:
            primary_hvt = active_targets[0]

            # Choose closest drone to be direct TRACKER
            best_tracker_id = None
            min_dist = float("inf")
            for d_id in [0, 1, 2]:
                d_pos = drone_telemetry[d_id]["position"]
                d = float(np.linalg.norm(d_pos[:2] - primary_hvt.last_position[:2]))
                if d < min_dist:
                    min_dist = d
                    best_tracker_id = d_id

            primary_hvt.assigned_tracker = best_tracker_id

            # TRACKER Directive: Orbit/shadow target at 4.0m camera standoff
            t_pos = primary_hvt.last_position
            t_vel = primary_hvt.last_velocity
            tracker_goal = np.array([t_pos[0] + 3.0, t_pos[1] + 3.0, 4.0])
            heading_to_tgt = math.atan2(t_pos[1] - tracker_goal[1], t_pos[0] - tracker_goal[0])

            directives[best_tracker_id] = SwarmDirective(
                agent_id=best_tracker_id,
                role=TacticalRole.TRACKER,
                goal_position=tracker_goal,
                desired_speed=3.2,
                target_id=primary_hvt.target_id,
                lookat_heading=heading_to_tgt,
                reasoning=f"Engaging visual lock on HVT-{primary_hvt.target_id} at 4.0m standoff",
            )

            # Choose FLANKER drone to cut off escape vector ahead of target
            remaining_drones = [d_id for d_id in [0, 1, 2] if d_id != best_tracker_id]
            if len(remaining_drones) > 0:
                flanker_id = remaining_drones[0]
                primary_hvt.assigned_flanker = flanker_id

                # Pincer Lead Calculation (lead 3.0s ahead along velocity vector)
                vel_norm = np.linalg.norm(t_vel[:2])
                lead_time = 3.0
                if vel_norm > 0.2:
                    lead_pos = t_pos[:2] + (t_vel[:2] / vel_norm) * min(12.0, vel_norm * lead_time + 4.0)
                else:
                    lead_pos = t_pos[:2] + np.array([6.0, 0.0])

                flanker_goal = np.array([lead_pos[0], lead_pos[1], 3.2])

                directives[flanker_id] = SwarmDirective(
                    agent_id=flanker_id,
                    role=TacticalRole.FLANKER,
                    goal_position=flanker_goal,
                    desired_speed=4.2,  # Sprint speed
                    target_id=primary_hvt.target_id,
                    reasoning=f"Executing pincer flank ahead of HVT-{primary_hvt.target_id} to block escape corridor",
                )

            # Third drone handles secondary target or continues exploration
            other_id = remaining_drones[1] if len(remaining_drones) > 1 else None
            if other_id is not None:
                if len(active_targets) > 1:
                    sec_hvt = active_targets[1]
                    directives[other_id] = SwarmDirective(
                        agent_id=other_id,
                        role=TacticalRole.TRACKER,
                        goal_position=np.array([sec_hvt.last_position[0] - 3.5, sec_hvt.last_position[1] + 3.5, 4.2]),
                        desired_speed=3.0,
                        target_id=sec_hvt.target_id,
                        reasoning=f"Tracking secondary target HVT-{sec_hvt.target_id}",
                    )
                else:
                    # Autonomous Frontier Explorer
                    p_other = drone_telemetry[other_id]["position"]
                    best_frontier, gain = uncertainty_grid.get_best_frontier(p_other, cruise_altitude=4.5)
                    directives[other_id] = SwarmDirective(
                        agent_id=other_id,
                        role=TacticalRole.EXPLORER,
                        goal_position=best_frontier,
                        desired_speed=2.6,
                        reasoning=f"Searching high-uncertainty urban frontiers (expected gain: {gain:.1f})",
                    )

        # Priority B: If target was recently lost behind a building, launch localized spiral sweep
        elif len(lost_targets) > 0:
            lost_hvt = lost_targets[0]
            for idx, d_id in enumerate([0, 1, 2]):
                self.spiral_phase[d_id] += 0.4
                spiral_r = 4.0 + (idx * 3.0) + (self.spiral_phase[d_id] * 0.3 % 8.0)
                theta = self.spiral_phase[d_id] + (idx * 2.0 * np.pi / 3.0)

                sp_x = lost_hvt.last_position[0] + spiral_r * math.cos(theta)
                sp_y = lost_hvt.last_position[1] + spiral_r * math.sin(theta)
                sp_goal = np.array([sp_x, sp_y, 4.0 + idx * 0.8])

                directives[d_id] = SwarmDirective(
                    agent_id=d_id,
                    role=TacticalRole.LOST_TARGET_SWEEP,
                    goal_position=sp_goal,
                    desired_speed=3.0,
                    target_id=lost_hvt.target_id,
                    reasoning=f"Target HVT-{lost_hvt.target_id} broke line of sight behind building; executing spiral search (r={spiral_r:.1f}m)",
                )

        # Priority C: Normal Autonomous Frontier Exploration (Information-Gain Driven)
        else:
            for d_id in [0, 1, 2]:
                d_pos = drone_telemetry[d_id]["position"]
                best_frontier, gain = uncertainty_grid.get_best_frontier(
                    d_pos, cruise_altitude=3.8 + d_id * 0.6
                )
                directives[d_id] = SwarmDirective(
                    agent_id=d_id,
                    role=TacticalRole.EXPLORER,
                    goal_position=best_frontier,
                    desired_speed=2.8,
                    reasoning=f"Autonomous frontier search in sector (uncertainty gain: {gain:.1f})",
                )

        self.last_decision_time = sim_time
        return directives
