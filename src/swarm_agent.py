# -*- coding: utf-8 -*-
"""
swarm_agent.py — Autonomous Decentralized Drone Agent

Features:
- Finite State Machine: DISPATCH, SECTOR_SEARCH, INTERCEPT_TARGET, ORBIT_SURVEILLANCE, COMM_RELAY, RETURN_TO_BASE
- Real-time 3D Artificial Potential Field (APF) collision & obstacle avoidance
- Distributed Gossip consensus integration (Task Bidding & Target Belief Fusion)
- Cascaded SE(3) control & onboard sensor/battery mechanics
"""

from __future__ import annotations
import math
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from .physics import DRONE_MASS, GRAVITY
from .controller import CascadedQuadrotorController
from .sensors import SensorSuite, BatteryState
from .gossip import GossipNode, MessageType, TargetEstimate


class AgentState(Enum):
    DISPATCH = "DISPATCH"
    SECTOR_SEARCH = "SECTOR_SEARCH"
    INTERCEPT_TARGET = "INTERCEPT_TARGET"
    ORBIT_SURVEILLANCE = "ORBIT_SURVEILLANCE"
    COMM_RELAY = "COMM_RELAY"
    RETURN_TO_BASE = "RETURN_TO_BASE"


class SwarmDroneAgent:
    """
    Autonomous intelligent agent operating an individual nano-quadrotor.
    """

    def __init__(
        self,
        agent_id: int,
        search_sector_bounds: Tuple[float, float, float, float],  # (x_min, y_min, x_max, y_max)
        home_position: np.ndarray,
        cruise_altitude: float = 3.0,
        cruise_speed: float = 2.0,
    ):
        self.agent_id = agent_id
        self.sector_bounds = search_sector_bounds
        self.home_pos = home_position.copy()
        self.cruise_altitude = cruise_altitude
        self.cruise_speed = cruise_speed

        # Subsystems
        self.controller = CascadedQuadrotorController(mass=DRONE_MASS)
        self.sensors = SensorSuite(drone_id=agent_id)
        self.battery = self.sensors.battery
        self.gossip = GossipNode(agent_id=agent_id)

        # Agent state
        self.state = AgentState.SECTOR_SEARCH
        self.assigned_target_id: Optional[int] = None
        self.current_waypoint: np.ndarray = home_position.copy()
        self.current_waypoint[2] = cruise_altitude

        # Sector sweep plan
        self.sweep_waypoints = self._generate_sector_waypoints()
        self.sweep_idx = 0

        # Orbit surveillance state
        self.orbit_center = np.zeros(3)
        self.orbit_radius = 3.0
        self.orbit_speed = 1.8
        self.orbit_phase = 0.0

        # Telemetry logging
        self.flight_trail: List[np.ndarray] = []
        self.total_distance_flown = 0.0
        self.last_pos = home_position.copy()

    @property
    def battery_pct(self) -> float:
        return float((self.battery.remaining_wh / self.battery.initial_capacity_wh) * 100.0)

    def _generate_sector_waypoints(self) -> List[np.ndarray]:
        """Generates lawnmower sweep waypoints within assigned quadrant."""
        x_min, y_min, x_max, y_max = self.sector_bounds
        alt = self.cruise_altitude
        wps = []
        
        # 4-pass lawnmower
        y_steps = np.linspace(y_min + 2.0, y_max - 2.0, 4)
        for i, y in enumerate(y_steps):
            if i % 2 == 0:
                wps.append(np.array([x_min + 2.0, y, alt]))
                wps.append(np.array([x_max - 2.0, y, alt]))
            else:
                wps.append(np.array([x_max - 2.0, y, alt]))
                wps.append(np.array([x_min + 2.0, y, alt]))
        return wps

    def update_decision_loop(
        self,
        current_pos: np.ndarray,
        current_vel: np.ndarray,
        obstacles: List[Dict[str, Any]],
        sim_time: float,
    ) -> None:
        """Executes autonomous decision logic and task auction over gossip."""
        # 1. Process gossip inbox
        self.gossip.process_inbox(sim_time)

        # 2. Update spatial coverage grid
        grid_x = int(np.clip((current_pos[0] + 20.0) / 2.0, 0, 19))
        grid_y = int(np.clip((current_pos[1] + 20.0) / 2.0, 0, 19))
        self.gossip.coverage_grid[grid_x, grid_y] = min(1.0, self.gossip.coverage_grid[grid_x, grid_y] + 0.1)

        # 3. Check Battery Fail-Safe
        if self.battery_pct < 20.0 and self.state != AgentState.RETURN_TO_BASE:
            self.state = AgentState.RETURN_TO_BASE
            self.current_waypoint = self.home_pos.copy()
            return

        # 4. Distributed Task Allocation & Target Evaluation
        # Inspect target beliefs received via gossip or direct sensor
        for t_id, belief in self.gossip.target_beliefs.items():
            task_key = f"track_target_{t_id}"
            
            # Compute local bid utility (lower distance to target + higher battery)
            dist = float(np.linalg.norm(current_pos[:2] - belief.position[:2]))
            utility = (self.battery_pct / 100.0) * 50.0 - dist
            
            # Submit bid periodically
            if sim_time - self.gossip.last_broadcast_time > 0.3:
                self.gossip.outbox.append(self.gossip.submit_task_bid(task_key, utility, sim_time))

            # Check if this agent is the winning bidder
            assignment = self.gossip.task_assignments.get(task_key)
            if assignment and assignment["winner_id"] == self.agent_id:
                if self.state not in [AgentState.INTERCEPT_TARGET, AgentState.ORBIT_SURVEILLANCE]:
                    self.state = AgentState.INTERCEPT_TARGET
                    self.assigned_target_id = t_id
                    break

        # 5. State Machine Execution
        if self.state == AgentState.SECTOR_SEARCH:
            target_wp = self.sweep_waypoints[self.sweep_idx]
            dist_to_wp = float(np.linalg.norm(current_pos - target_wp))
            if dist_to_wp < 1.2:
                self.sweep_idx = (self.sweep_idx + 1) % len(self.sweep_waypoints)
            self.current_waypoint = self.sweep_waypoints[self.sweep_idx].copy()

        elif self.state == AgentState.INTERCEPT_TARGET:
            if self.assigned_target_id in self.gossip.target_beliefs:
                est = self.gossip.target_beliefs[self.assigned_target_id]
                self.orbit_center = est.position.copy()
                dist_to_target = float(np.linalg.norm(current_pos[:2] - est.position[:2]))
                
                if dist_to_target < 4.0:
                    self.state = AgentState.ORBIT_SURVEILLANCE
                else:
                    self.current_waypoint = np.array([est.position[0], est.position[1], self.cruise_altitude])
            else:
                self.state = AgentState.SECTOR_SEARCH

        elif self.state == AgentState.ORBIT_SURVEILLANCE:
            if self.assigned_target_id in self.gossip.target_beliefs:
                est = self.gossip.target_beliefs[self.assigned_target_id]
                self.orbit_center = est.position.copy()
                
                # Advance orbit angle
                self.orbit_phase += self.orbit_speed * 0.01 / self.orbit_radius
                ox = self.orbit_center[0] + self.orbit_radius * np.cos(self.orbit_phase)
                oy = self.orbit_center[1] + self.orbit_radius * np.sin(self.orbit_phase)
                self.current_waypoint = np.array([ox, oy, 2.5])
            else:
                self.state = AgentState.SECTOR_SEARCH

        elif self.state == AgentState.RETURN_TO_BASE:
            self.current_waypoint = self.home_pos.copy()

    def compute_control_with_avoidance(
        self,
        current_pos: np.ndarray,
        current_vel: np.ndarray,
        current_quat: np.ndarray,
        current_omega: np.ndarray,
        peer_positions: Dict[int, np.ndarray],
        obstacles: List[Dict[str, Any]],
        dt: float,
    ) -> np.ndarray:
        """
        Computes motor commands incorporating 3D APF obstacle & peer collision repulsion.
        """
        # APF Repulsive velocity vector
        v_rep = np.zeros(3, dtype=np.float64)

        # 1. Inter-Drone Separation (R_sep = 2.5m)
        d_sep = 2.5
        k_rep_drone = 3.5
        for peer_id, peer_pos in peer_positions.items():
            if peer_id == self.agent_id:
                continue
            delta = current_pos - peer_pos
            dist = float(np.linalg.norm(delta))
            if 0.05 < dist < d_sep:
                n = delta / dist
                v_rep += k_rep_drone * (1.0 / dist - 1.0 / d_sep) * (1.0 / (dist**2)) * n

        # 2. Obstacle / Building Avoidance (Cylinder / Box envelope)
        k_rep_obs = 5.0
        d_obs_margin = 2.0
        for obs in obstacles:
            ox, oy = obs["pos"][:2]
            half_w, half_l = obs.get("size", [2.0, 2.0])[:2]
            obs_h = obs.get("height", 6.0)

            # Check if drone is within vertical range of obstacle
            if current_pos[2] < obs_h + 1.0:
                dx = max(0.0, abs(current_pos[0] - ox) - half_w)
                dy = max(0.0, abs(current_pos[1] - oy) - half_l)
                dist_2d = math.sqrt(dx**2 + dy**2)

                if dist_2d < d_obs_margin:
                    dir_x = (current_pos[0] - ox)
                    dir_y = (current_pos[1] - oy)
                    norm_dir = math.sqrt(dir_x**2 + dir_y**2) + 1e-6
                    n_2d = np.array([dir_x / norm_dir, dir_y / norm_dir, 0.5])
                    v_rep += k_rep_obs * (1.0 / max(0.1, dist_2d) - 1.0 / d_obs_margin) * n_2d

        # Clamp max repulsive correction
        norm_rep = float(np.linalg.norm(v_rep))
        if norm_rep > 2.5:
            v_rep = (v_rep / norm_rep) * 2.5

        # 3. Cascaded SE(3) Controller
        # Apply repulsive bias to target position
        target_pos_adjusted = self.current_waypoint + v_rep * 0.5
        
        # Desired velocity towards target
        delta_p = target_pos_adjusted - current_pos
        dist_p = float(np.linalg.norm(delta_p))
        if dist_p > 1e-3:
            v_des = (delta_p / dist_p) * min(self.cruise_speed, dist_p * 1.5) + v_rep
        else:
            v_des = v_rep

        motor_cmds, info = self.controller.compute_control(
            position=current_pos,
            velocity=current_vel,
            quaternion=current_quat,
            angular_velocity=current_omega,
            target_position=target_pos_adjusted,
            target_velocity=v_des,
            dt=dt,
        )

        # 4. Update battery & flight logging
        step_dist = float(np.linalg.norm(current_pos - self.last_pos))
        self.total_distance_flown += step_dist
        self.last_pos = current_pos.copy()
        
        # Keep flight trail buffer
        if len(self.flight_trail) == 0 or np.linalg.norm(current_pos - self.flight_trail[-1]) > 0.2:
            self.flight_trail.append(current_pos.copy())
            if len(self.flight_trail) > 120:
                self.flight_trail.pop(0)

        # Update battery power consumption
        thrust_N = info.get("thrust_N", DRONE_MASS * GRAVITY)
        thrust_ratio = float(thrust_N / (DRONE_MASS * GRAVITY))
        self.battery.update(thrust_ratio=thrust_ratio, dt=dt)

        return motor_cmds
