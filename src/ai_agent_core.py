# -*- coding: utf-8 -*-
"""
ai_agent_core.py — Heterogeneous Swarm AI Cognitive Engine & Decentralized Multi-Agent Core

Provides:
- Heterogeneous Drone Specs: Heavy Scout, Fast Interceptor, Thermal Surveyor, Comms Relay
- Explicit AI Cognitive Command Engine: Structured Tool Calls, Natural Language Reasoning Chains, and Command Logs
- Multi-hop Gossip Mesh with Relay Node amplification and Bayesian belief state fusion
- Dynamic Tactical Events: Inter-Sector Target Handover, Battery Relief on Station, Adaptive Comms
- Consensus-Based Bundle Algorithm (CBBA) for dynamic target interception auctions
- Wind Gust Compensation (Dryden turbulence) & 3D APF Obstacle Avoidance
"""

from __future__ import annotations
import math
import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any, Set
import numpy as np

from .physics import GRAVITY, quat_to_rotation_matrix
from .controller import CascadedQuadrotorController
from .sensors import SensorSuite, BatteryModel, TargetObservation
from .gossip import GossipNode, GossipMessage, MessageType, TargetEstimate


class DroneClass(Enum):
    HEAVY_SCOUT = "HEAVY_SCOUT"
    FAST_INTERCEPTOR = "FAST_INTERCEPTOR"
    THERMAL_SURVEYOR = "THERMAL_SURVEYOR"
    COMMS_RELAY = "COMMS_RELAY"


class AIRole(Enum):
    AREA_SURVEYOR = "AREA_SURVEYOR"
    RAPID_INTERCEPTOR = "RAPID_INTERCEPTOR"
    TARGET_SHADOW = "TARGET_SHADOW"
    COMMS_ANCHOR = "COMMS_ANCHOR"
    RELIEF_PATROL = "RELIEF_PATROL"
    BASE_RECOVERY = "BASE_RECOVERY"


@dataclass
class AICommand:
    """A formal tool execution command issued by an autonomous AI Agent."""
    timestamp: float
    agent_id: int
    drone_class: str
    role: str
    reasoning: str
    tool_name: str
    tool_args: Dict[str, Any]
    status: str = "EXECUTED"


@dataclass
class DroneSpecs:
    """Heterogeneous physical and payload specifications."""
    drone_class: DroneClass
    mass: float                    # kg
    arm_length: float              # m
    battery_capacity_wh: float     # Wh
    max_speed: float               # m/s
    cruise_altitude: float         # m
    camera_fov_deg: float          # degrees
    max_sensor_range: float        # m
    comm_range: float              # m (RF transmission distance)
    thrust_margin: float           # Max thrust / weight


HETEROGENEOUS_SPECS: Dict[int, DroneSpecs] = {
    0: DroneSpecs(
        drone_class=DroneClass.HEAVY_SCOUT,
        mass=0.65,
        arm_length=0.15,
        battery_capacity_wh=8.5,
        max_speed=2.4,
        cruise_altitude=4.5,
        camera_fov_deg=45.0,  # Narrow zoom lens
        max_sensor_range=35.0,
        comm_range=18.0,
        thrust_margin=2.0,
    ),
    1: DroneSpecs(
        drone_class=DroneClass.FAST_INTERCEPTOR,
        mass=0.28,
        arm_length=0.065,
        battery_capacity_wh=3.2,
        max_speed=4.5,  # High speed dash
        cruise_altitude=3.5,
        camera_fov_deg=95.0,  # Wide angle
        max_sensor_range=22.0,
        comm_range=15.0,
        thrust_margin=2.6,
    ),
    2: DroneSpecs(
        drone_class=DroneClass.THERMAL_SURVEYOR,
        mass=0.42,
        arm_length=0.10,
        battery_capacity_wh=5.5,
        max_speed=2.8,
        cruise_altitude=3.8,
        camera_fov_deg=75.0,  # Multispectral
        max_sensor_range=28.0,
        comm_range=18.0,
        thrust_margin=2.2,
    ),
    3: DroneSpecs(
        drone_class=DroneClass.COMMS_RELAY,
        mass=0.52,
        arm_length=0.12,
        battery_capacity_wh=6.8,
        max_speed=2.2,
        cruise_altitude=5.5,  # High altitude relay
        camera_fov_deg=65.0,
        max_sensor_range=25.0,
        comm_range=32.0,  # Extended high-gain RF dish
        thrust_margin=2.1,
    ),
}


class HeterogeneousSwarmAgent:
    """
    Autonomous intelligent agent operating a heterogeneous reconnaissance quadrotor
    with an explicit cognitive tool-calling engine and natural language reasoning chains.
    """

    def __init__(
        self,
        agent_id: int,
        search_quadrant: Tuple[float, float, float, float],
        home_position: np.ndarray,
    ):
        self.agent_id = agent_id
        self.specs = HETEROGENEOUS_SPECS[agent_id]
        self.quadrant = search_quadrant
        self.home_pos = home_position.copy()

        # Core subsystems
        self.controller = CascadedQuadrotorController(mass=self.specs.mass)
        self.battery = BatteryModel(initial_capacity_wh=self.specs.battery_capacity_wh)
        self.gossip = GossipNode(agent_id=agent_id, broadcast_interval=0.10)

        # AI Cognitive State
        self.role = AIRole.AREA_SURVEYOR if agent_id != 3 else AIRole.COMMS_ANCHOR
        self.assigned_target_id: Optional[int] = None
        self.current_waypoint: np.ndarray = home_position.copy()
        self.current_waypoint[2] = self.specs.cruise_altitude
        self.active_tool_str: str = "recon_area_search()"

        # Lawnmower sector patrol waypoints
        self.sweep_waypoints = self._build_quadrant_sweep()
        self.sweep_idx = 0

        # Orbit tracking parameters
        self.orbit_center = np.zeros(3)
        self.orbit_radius = 3.5
        self.orbit_speed = min(2.5, self.specs.max_speed * 0.7)
        self.orbit_phase = float(agent_id * (np.pi / 2.0))

        # Real-World Tactical Events State
        self.handover_in_progress = False
        self.relief_requested = False
        self.relief_active = False
        self.last_cognitive_tick = 0.0

        # Telemetry & Command History
        self.flight_trail: List[np.ndarray] = []
        self.total_distance_flown = 0.0
        self.last_pos = home_position.copy()
        self.command_log: List[AICommand] = []

    def _build_quadrant_sweep(self) -> List[np.ndarray]:
        x_min, y_min, x_max, y_max = self.quadrant
        alt = self.specs.cruise_altitude
        wps = []
        y_lines = np.linspace(y_min + 3.0, y_max - 3.0, 5)
        for i, y in enumerate(y_lines):
            if i % 2 == 0:
                wps.append(np.array([x_min + 3.0, y, alt]))
                wps.append(np.array([x_max - 3.0, y, alt]))
            else:
                wps.append(np.array([x_max - 3.0, y, alt]))
                wps.append(np.array([x_min + 3.0, y, alt]))
        return wps

    @property
    def battery_pct(self) -> float:
        return float((self.battery.remaining_wh / self.battery.initial_capacity_wh) * 100.0)

    # ── Formal AI Tool Execution Methods ───────────────────────────────────────
    def recon_area_search(self, bounds: Tuple[float, float, float, float], speed: float, sim_time: float, reasoning: str) -> AICommand:
        cmd = AICommand(
            timestamp=sim_time,
            agent_id=self.agent_id,
            drone_class=self.specs.drone_class.value,
            role=self.role.value,
            reasoning=reasoning,
            tool_name="recon_area_search",
            tool_args={"bounds": list(bounds), "speed_ms": speed, "pattern": "lawnmower"},
        )
        self.active_tool_str = f"recon_area_search(quad={self.quadrant[0]:.0f},{self.quadrant[1]:.0f})"
        self.command_log.append(cmd)
        return cmd

    def recon_fly_to(self, x: float, y: float, z: float, velocity_limit: float, sim_time: float, reasoning: str) -> AICommand:
        self.current_waypoint = np.array([x, y, z])
        cmd = AICommand(
            timestamp=sim_time,
            agent_id=self.agent_id,
            drone_class=self.specs.drone_class.value,
            role=self.role.value,
            reasoning=reasoning,
            tool_name="recon_fly_to",
            tool_args={"target_pos": [x, y, z], "v_limit": velocity_limit},
        )
        self.active_tool_str = f"recon_fly_to([{x:.1f},{y:.1f},{z:.1f}])"
        self.command_log.append(cmd)
        return cmd

    def recon_orbit_point(self, center_x: float, center_y: float, radius: float, speed: float, altitude: float, sim_time: float, reasoning: str) -> AICommand:
        self.orbit_center = np.array([center_x, center_y, altitude])
        self.orbit_radius = radius
        self.orbit_speed = speed
        cmd = AICommand(
            timestamp=sim_time,
            agent_id=self.agent_id,
            drone_class=self.specs.drone_class.value,
            role=self.role.value,
            reasoning=reasoning,
            tool_name="recon_orbit_point",
            tool_args={"center": [center_x, center_y], "radius": radius, "speed": speed, "altitude": altitude},
        )
        self.active_tool_str = f"recon_orbit_point(HVT-{self.assigned_target_id}, r={radius}m)"
        self.command_log.append(cmd)
        return cmd

    def recon_capture_target_intel(self, target_id: int, target_pos: np.ndarray, confidence: float, sim_time: float, reasoning: str) -> AICommand:
        cmd = AICommand(
            timestamp=sim_time,
            agent_id=self.agent_id,
            drone_class=self.specs.drone_class.value,
            role=self.role.value,
            reasoning=reasoning,
            tool_name="recon_capture_target_intel",
            tool_args={"target_id": target_id, "pos": target_pos.tolist(), "conf": confidence},
        )
        self.command_log.append(cmd)
        return cmd

    def gossip_broadcast_handover(self, target_id: int, recipient_id: int, sim_time: float, reasoning: str) -> AICommand:
        cmd = AICommand(
            timestamp=sim_time,
            agent_id=self.agent_id,
            drone_class=self.specs.drone_class.value,
            role=self.role.value,
            reasoning=reasoning,
            tool_name="gossip_broadcast_handover",
            tool_args={"target_id": target_id, "recipient_id": recipient_id},
        )
        self.active_tool_str = f"gossip_handover(T{target_id}->D{recipient_id})"
        self.command_log.append(cmd)
        return cmd

    def gossip_request_relief(self, sim_time: float, reasoning: str) -> AICommand:
        cmd = AICommand(
            timestamp=sim_time,
            agent_id=self.agent_id,
            drone_class=self.specs.drone_class.value,
            role=self.role.value,
            reasoning=reasoning,
            tool_name="gossip_request_relief",
            tool_args={"sector": list(self.quadrant), "battery_pct": self.battery_pct},
        )
        self.active_tool_str = "gossip_request_relief(RTB)"
        self.command_log.append(cmd)
        return cmd

    def comms_relay_reposition(self, centroid: np.ndarray, sim_time: float, reasoning: str) -> AICommand:
        self.current_waypoint = np.array([centroid[0] * 0.5, centroid[1] * 0.5, self.specs.cruise_altitude])
        cmd = AICommand(
            timestamp=sim_time,
            agent_id=self.agent_id,
            drone_class=self.specs.drone_class.value,
            role=self.role.value,
            reasoning=reasoning,
            tool_name="comms_relay_reposition",
            tool_args={"new_center": centroid[:2].tolist(), "alt": self.specs.cruise_altitude},
        )
        self.active_tool_str = f"comms_reposition([{centroid[0]*0.5:.1f},{centroid[1]*0.5:.1f}])"
        self.command_log.append(cmd)
        return cmd

    # ── Optical Perception & Sensing ──────────────────────────────────────────
    def perceive_environment(
        self,
        current_pos: np.ndarray,
        current_quat: np.ndarray,
        ground_targets: Dict[int, np.ndarray],
        sim_time: float,
        rng: np.random.Generator,
    ) -> List[GossipMessage]:
        out_messages: List[GossipMessage] = []
        R_b2w = quat_to_rotation_matrix(current_quat)
        cam_fwd = R_b2w[:, 0]

        for t_id, t_pos in ground_targets.items():
            delta = t_pos - current_pos
            dist = float(np.linalg.norm(delta))

            if dist <= self.specs.max_sensor_range:
                cos_angle = np.dot(cam_fwd, delta / (dist + 1e-6))
                fov_limit = np.cos(np.deg2rad(self.specs.camera_fov_deg / 2.0))

                if cos_angle > fov_limit:
                    range_factor = 1.0 - (dist / self.specs.max_sensor_range)
                    angle_factor = (cos_angle - fov_limit) / (1.0 - fov_limit + 1e-6)
                    conf = float(np.clip(range_factor * 0.6 + angle_factor * 0.4, 0.35, 0.99))

                    noise = rng.normal(0.0, 0.08 * (dist / 10.0), 3)
                    noisy_est = t_pos + noise

                    intel_msg = self.gossip.update_local_target_observation(
                        target_id=t_id,
                        pos=noisy_est,
                        vel=np.zeros(3),
                        conf=conf,
                        sim_time=sim_time,
                    )
                    out_messages.append(intel_msg)

                    # Log occasional optical detection tool call
                    if sim_time - self.last_cognitive_tick > 2.0:
                        self.recon_capture_target_intel(
                            target_id=t_id,
                            target_pos=noisy_est,
                            confidence=conf,
                            sim_time=sim_time,
                            reasoning=f"Visual optical line-of-sight confirmed on HVT-{t_id} (conf={conf:.2f}, dist={dist:.1f}m)",
                        )

        return out_messages

    # ── Cognitive Deliberation & Reasoning Loop ────────────────────────────────
    def evaluate_ai_deliberation(
        self,
        current_pos: np.ndarray,
        current_vel: np.ndarray,
        obstacles: List[Dict[str, Any]],
        peer_positions: Dict[int, np.ndarray],
        sim_time: float,
    ) -> Optional[AICommand]:
        """
        Executes real-time AI Agent reasoning and issues formal tool commands.
        """
        # 1. Process gossip inbox
        self.gossip.process_inbox(sim_time)

        # 2. Update occupancy coverage grid
        gx = int(np.clip((current_pos[0] + 30.0) / 3.0, 0, 19))
        gy = int(np.clip((current_pos[1] + 30.0) / 3.0, 0, 19))
        self.gossip.coverage_grid[gx, gy] = min(1.0, self.gossip.coverage_grid[gx, gy] + 0.15)

        dispatched_cmd: Optional[AICommand] = None
        should_tick = (sim_time - self.last_cognitive_tick) >= 1.0

        # 3. Dynamic Real-World Tactical Events
        # Event A: Drone 1 (Fast Interceptor) triggers Battery Relief on Station at t >= 65.0s
        if self.agent_id == 1 and sim_time >= 65.0 and not self.relief_requested:
            self.relief_requested = True
            self.role = AIRole.BASE_RECOVERY
            self.current_waypoint = self.home_pos.copy()
            dispatched_cmd = self.gossip_request_relief(
                sim_time=sim_time,
                reasoning=f"High-speed sprint budget exhausted ({self.battery_pct:.1f}% SoC); requesting patrol relief for NE sector",
            )
            # Broadcast relief over Gossip
            self.gossip.outbox.append(
                self.gossip.create_message(
                    msg_type=MessageType.ALERT,
                    payload={"event": "RTB_RELIEF_REQUEST", "depleted_drone": 1, "sector": self.quadrant},
                    sim_time=sim_time,
                )
            )
            self.last_cognitive_tick = sim_time
            return dispatched_cmd

        # Event B: Drone 2 (Thermal Surveyor) assumes Relief on Station for Drone 1's quadrant
        if self.agent_id == 2 and sim_time >= 66.0 and not self.relief_active:
            self.relief_active = True
            self.role = AIRole.RELIEF_PATROL
            self.quadrant = (0.0, 0.0, 25.0, 25.0)
            self.sweep_waypoints = self._build_quadrant_sweep()
            self.sweep_idx = 0
            dispatched_cmd = self.recon_area_search(
                bounds=self.quadrant,
                speed=self.specs.max_speed * 0.8,
                sim_time=sim_time,
                reasoning="Acknowledged D1 relief request via Gossip mesh; re-routing to assume NE sector patrol",
            )
            self.last_cognitive_tick = sim_time
            return dispatched_cmd

        # Event C: Drone 3 Adaptive Comms Relay Repositioning (track peer centroid)
        if self.agent_id == 3 and should_tick:
            other_positions = [pos for did, pos in peer_positions.items() if did != 3]
            if other_positions:
                centroid = np.mean(other_positions, axis=0)
                dist_to_centroid = float(np.linalg.norm(self.current_waypoint[:2] - centroid[:2] * 0.5))
                if dist_to_centroid > 2.0:
                    dispatched_cmd = self.comms_relay_reposition(
                        centroid=centroid,
                        sim_time=sim_time,
                        reasoning=f"Optimizing mesh connectivity; relocating high-gain RF dish to centroid [{centroid[0]*0.5:.1f}, {centroid[1]*0.5:.1f}]",
                    )
                    self.last_cognitive_tick = sim_time
                    return dispatched_cmd

        # Event D: Inter-Sector Target Handover (between Drone 1 and Drone 0 at t ≈ 45.0s)
        if self.agent_id == 1 and self.role == AIRole.TARGET_SHADOW and self.assigned_target_id is not None:
            est = self.gossip.target_beliefs.get(self.assigned_target_id)
            if est and est.position[0] < -2.0 and not self.handover_in_progress:
                self.handover_in_progress = True
                dispatched_cmd = self.gossip_broadcast_handover(
                    target_id=self.assigned_target_id,
                    recipient_id=0,
                    sim_time=sim_time,
                    reasoning=f"Target HVT-{self.assigned_target_id} crossed into Western sector (X={est.position[0]:.1f}m); transferring custody to D0",
                )
                self.gossip.outbox.append(
                    self.gossip.create_message(
                        msg_type=MessageType.TASK_BID,
                        payload={"task_id": f"shadow_hvt_{self.assigned_target_id}", "bidder_id": 0, "bid_value": 999.0},
                        sim_time=sim_time,
                    )
                )
                self.role = AIRole.AREA_SURVEYOR
                self.assigned_target_id = None
                self.last_cognitive_tick = sim_time
                return dispatched_cmd

        # 4. Distributed Consensus Auction (CBBA) for Target Shadowing
        if self.role not in [AIRole.BASE_RECOVERY, AIRole.COMMS_ANCHOR, AIRole.RELIEF_PATROL]:
            for t_id, belief in self.gossip.target_beliefs.items():
                task_key = f"shadow_hvt_{t_id}"
                dist = float(np.linalg.norm(current_pos[:2] - belief.position[:2]))
                class_multiplier = 1.4 if self.specs.drone_class == DroneClass.FAST_INTERCEPTOR else 1.0
                utility = class_multiplier * ((self.battery_pct / 100.0) * 60.0 - dist * 1.5)

                if sim_time - self.gossip.last_broadcast_time > 0.3:
                    bid_msg = self.gossip.submit_task_bid(task_key, utility, sim_time)
                    self.gossip.outbox.append(bid_msg)

                assignment = self.gossip.task_assignments.get(task_key)
                if assignment and assignment["winner_id"] == self.agent_id:
                    if self.role not in [AIRole.RAPID_INTERCEPTOR, AIRole.TARGET_SHADOW]:
                        self.role = AIRole.RAPID_INTERCEPTOR
                        self.assigned_target_id = t_id
                        dispatched_cmd = self.recon_fly_to(
                            x=belief.position[0], y=belief.position[1], z=self.specs.cruise_altitude,
                            velocity_limit=self.specs.max_speed, sim_time=sim_time,
                            reasoning=f"Won CBBA auction for HVT-{t_id} (utility={utility:.1f}); initiating rapid intercept vector",
                        )
                        self.last_cognitive_tick = sim_time
                        return dispatched_cmd

        # 5. Role FSM Trajectory Dispatch
        if self.role in [AIRole.AREA_SURVEYOR, AIRole.RELIEF_PATROL]:
            target_wp = self.sweep_waypoints[self.sweep_idx]
            if np.linalg.norm(current_pos - target_wp) < 1.5:
                self.sweep_idx = (self.sweep_idx + 1) % len(self.sweep_waypoints)
            self.current_waypoint = self.sweep_waypoints[self.sweep_idx].copy()
            if should_tick:
                self.active_tool_str = f"recon_area_search(wp={self.sweep_idx+1}/{len(self.sweep_waypoints)})"

        elif self.role == AIRole.RAPID_INTERCEPTOR:
            if self.assigned_target_id in self.gossip.target_beliefs:
                est = self.gossip.target_beliefs[self.assigned_target_id]
                self.orbit_center = est.position.copy()
                dist_to_hvt = float(np.linalg.norm(current_pos[:2] - est.position[:2]))
                if dist_to_hvt < 4.5:
                    self.role = AIRole.TARGET_SHADOW
                    dispatched_cmd = self.recon_orbit_point(
                        center_x=est.position[0], center_y=est.position[1],
                        radius=3.5, speed=self.orbit_speed, altitude=self.specs.cruise_altitude,
                        sim_time=sim_time, reasoning=f"Closed range to HVT-{self.assigned_target_id} (dist={dist_to_hvt:.1f}m); transitioning to circular surveillance orbit",
                    )
                    self.last_cognitive_tick = sim_time
                    return dispatched_cmd
                else:
                    self.current_waypoint = np.array([est.position[0], est.position[1], self.specs.cruise_altitude])

        elif self.role == AIRole.TARGET_SHADOW:
            if self.assigned_target_id in self.gossip.target_beliefs:
                est = self.gossip.target_beliefs[self.assigned_target_id]
                self.orbit_center = est.position.copy()
                self.orbit_phase += self.orbit_speed * 0.01 / self.orbit_radius
                ox = self.orbit_center[0] + self.orbit_radius * np.cos(self.orbit_phase)
                oy = self.orbit_center[1] + self.orbit_radius * np.sin(self.orbit_phase)
                self.current_waypoint = np.array([ox, oy, self.specs.cruise_altitude])
                self.active_tool_str = f"recon_orbit_point(HVT-{self.assigned_target_id}, r=3.5m)"

        elif self.role == AIRole.BASE_RECOVERY:
            self.current_waypoint = self.home_pos.copy()
            self.active_tool_str = f"recon_fly_to(BASE_PAD)"

        if should_tick:
            self.last_cognitive_tick = sim_time

        return dispatched_cmd

    # ── Physical Flight Control with 3D APF & Wind Feedforward ─────────────────
    def compute_motor_control(
        self,
        current_pos: np.ndarray,
        current_vel: np.ndarray,
        current_quat: np.ndarray,
        current_omega: np.ndarray,
        peer_positions: Dict[int, np.ndarray],
        obstacles: List[Dict[str, Any]],
        wind_vel: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        v_rep = np.zeros(3, dtype=np.float64)

        # 1. Peer Separation (d_sep = 2.8m)
        d_sep = 2.8
        k_sep = 4.0
        for peer_id, peer_pos in peer_positions.items():
            if peer_id == self.agent_id:
                continue
            delta = current_pos - peer_pos
            dist = float(np.linalg.norm(delta))
            if 0.05 < dist < d_sep:
                n = delta / dist
                v_rep += k_sep * (1.0 / dist - 1.0 / d_sep) * (1.0 / (dist**2)) * n

        # 2. Obstacle Avoidance (8 urban structures)
        k_obs = 6.0
        d_margin = 2.2
        for obs in obstacles:
            ox, oy = obs["pos"][:2]
            hw, hl = obs.get("size", [3.0, 3.0])[:2]
            obs_h = obs.get("height", 8.0)

            if current_pos[2] < obs_h + 1.2:
                dx = max(0.0, abs(current_pos[0] - ox) - hw)
                dy = max(0.0, abs(current_pos[1] - oy) - hl)
                dist_2d = math.sqrt(dx**2 + dy**2)

                if dist_2d < d_margin:
                    dir_x = (current_pos[0] - ox)
                    dir_y = (current_pos[1] - oy)
                    norm_dir = math.sqrt(dir_x**2 + dir_y**2) + 1e-6
                    n_2d = np.array([dir_x / norm_dir, dir_y / norm_dir, 0.4])
                    v_rep += k_obs * (1.0 / max(0.1, dist_2d) - 1.0 / d_margin) * n_2d

        norm_rep = float(np.linalg.norm(v_rep))
        if norm_rep > 3.0:
            v_rep = (v_rep / norm_rep) * 3.0

        # 3. Wind gust feedforward compensation
        target_pos_adj = self.current_waypoint + v_rep * 0.6
        delta_p = target_pos_adj - current_pos
        dist_p = float(np.linalg.norm(delta_p))
        if dist_p > 1e-3:
            v_des = (delta_p / dist_p) * min(self.specs.max_speed, dist_p * 1.6) + v_rep - 0.5 * wind_vel
        else:
            v_des = v_rep - 0.5 * wind_vel

        motor_cmds, info = self.controller.compute_control(
            position=current_pos,
            velocity=current_vel,
            quaternion=current_quat,
            angular_velocity=current_omega,
            target_position=target_pos_adj,
            target_velocity=v_des,
            dt=dt,
        )

        step_dist = float(np.linalg.norm(current_pos - self.last_pos))
        self.total_distance_flown += step_dist
        self.last_pos = current_pos.copy()

        if len(self.flight_trail) == 0 or np.linalg.norm(current_pos - self.flight_trail[-1]) > 0.25:
            self.flight_trail.append(current_pos.copy())
            if len(self.flight_trail) > 250:
                self.flight_trail.pop(0)

        thrust_N = info.get("thrust_N", self.specs.mass * GRAVITY)
        thrust_ratio = float(thrust_N / (self.specs.mass * GRAVITY))
        self.battery.update(thrust_ratio=thrust_ratio, dt=dt)

        return motor_cmds
