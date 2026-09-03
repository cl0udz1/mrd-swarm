"""
agent_interface.py — AI Agent Tool-Calling Interface for MRD-Swarm

Provides strict JSON schemas and Python wrapper functions for LLM/function-calling
agent integration. Each tool maps to a specific drone command with validation.
"""

from __future__ import annotations

import json
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass

from .swarm import SwarmEnvironment, DroneState, GroundTarget


# ═══════════════════════════════════════════════════════════════════════════════
# JSON SCHEMAS FOR FUNCTION CALLING
# ═══════════════════════════════════════════════════════════════════════════════

TOOL_SCHEMAS = {
    "recon_get_telemetry": {
        "name": "recon_get_telemetry",
        "description": "Get current telemetry for a drone: position, velocity, battery, heading, active target locks.",
        "parameters": {
            "type": "object",
            "properties": {
                "drone_id": {
                    "type": "integer",
                    "description": "Drone identifier (0-indexed)",
                    "minimum": 0,
                }
            },
            "required": ["drone_id"],
        },
    },
    "recon_fly_to": {
        "name": "recon_fly_to",
        "description": "Command a drone to fly to a specific 3D waypoint.",
        "parameters": {
            "type": "object",
            "properties": {
                "drone_id": {
                    "type": "integer",
                    "description": "Drone identifier (0-indexed)",
                    "minimum": 0,
                },
                "x": {"type": "number", "description": "Target X coordinate (m)"},
                "y": {"type": "number", "description": "Target Y coordinate (m)"},
                "z": {"type": "number", "description": "Target Z coordinate / altitude (m)"},
                "velocity_limit": {
                    "type": "number",
                    "description": "Maximum velocity (m/s)",
                    "default": 2.0,
                    "minimum": 0.1,
                    "maximum": 10.0,
                },
                "altitude_mode": {
                    "type": "string",
                    "enum": ["absolute", "relative"],
                    "description": "Whether z is absolute or relative to current altitude",
                    "default": "absolute",
                },
            },
            "required": ["drone_id", "x", "y", "z"],
        },
    },
    "recon_orbit_point": {
        "name": "recon_orbit_point",
        "description": "Command a drone to orbit around a specified point.",
        "parameters": {
            "type": "object",
            "properties": {
                "drone_id": {
                    "type": "integer",
                    "description": "Drone identifier (0-indexed)",
                    "minimum": 0,
                },
                "center_x": {"type": "number", "description": "Orbit center X (m)"},
                "center_y": {"type": "number", "description": "Orbit center Y (m)"},
                "radius": {
                    "type": "number",
                    "description": "Orbit radius (m)",
                    "minimum": 0.5,
                    "maximum": 20.0,
                },
                "speed": {
                    "type": "number",
                    "description": "Orbital speed (m/s)",
                    "default": 1.5,
                    "minimum": 0.1,
                    "maximum": 5.0,
                },
                "altitude": {
                    "type": "number",
                    "description": "Orbit altitude (m)",
                    "default": 3.0,
                },
            },
            "required": ["drone_id", "center_x", "center_y", "radius"],
        },
    },
    "recon_area_search": {
        "name": "recon_area_search",
        "description": "Assign multiple drones to search a rectangular area using lawnmower or grid partition pattern.",
        "parameters": {
            "type": "object",
            "properties": {
                "drone_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "List of drone IDs to assign",
                    "minItems": 1,
                },
                "bounding_box": {
                    "type": "object",
                    "properties": {
                        "x_min": {"type": "number"},
                        "y_min": {"type": "number"},
                        "x_max": {"type": "number"},
                        "y_max": {"type": "number"},
                    },
                    "required": ["x_min", "y_min", "x_max", "y_max"],
                    "description": "Search area bounds (m)",
                },
                "pattern": {
                    "type": "string",
                    "enum": ["lawnmower", "grid_partition"],
                    "description": "Search pattern type",
                    "default": "lawnmower",
                },
                "altitude": {
                    "type": "number",
                    "description": "Search altitude (m)",
                    "default": 3.0,
                },
                "speed": {
                    "type": "number",
                    "description": "Search speed (m/s)",
                    "default": 1.5,
                },
            },
            "required": ["drone_ids", "bounding_box"],
        },
    },
    "recon_capture_target_intel": {
        "name": "recon_capture_target_intel",
        "description": "Command a drone to capture intelligence on a detected target: coordinates and camera snapshot.",
        "parameters": {
            "type": "object",
            "properties": {
                "drone_id": {
                    "type": "integer",
                    "description": "Drone identifier (0-indexed)",
                    "minimum": 0,
                },
                "target_id": {
                    "type": "integer",
                    "description": "Target identifier to capture",
                    "minimum": 0,
                },
            },
            "required": ["drone_id", "target_id"],
        },
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════════════════════

class ReconAgentTools:
    """
    Implements the AI agent tool-calling interface for the recon swarm.

    Each method validates inputs, executes the command on the swarm environment,
    and returns a structured JSON response.
    """

    def __init__(self, env: SwarmEnvironment):
        self.env = env
        self._tool_map: Dict[str, Callable] = {
            "recon_get_telemetry": self.recon_get_telemetry,
            "recon_fly_to": self.recon_fly_to,
            "recon_orbit_point": self.recon_orbit_point,
            "recon_area_search": self.recon_area_search,
            "recon_capture_target_intel": self.recon_capture_target_intel,
        }

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Dispatch a tool call by name with validated arguments.

        Returns a structured response dict with 'status', 'data', and optional 'error'.
        """
        if tool_name not in self._tool_map:
            return {"status": "error", "error": f"Unknown tool: {tool_name}"}

        try:
            result = self._tool_map[tool_name](**arguments)
            return {"status": "success", "data": result}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def recon_get_telemetry(self, drone_id: int) -> Dict[str, Any]:
        """
        Get current telemetry for a drone.

        Returns
        -------
        dict with keys: position, velocity, battery_pct, heading, detected_targets,
                        is_active, mission_phase
        """
        self._validate_drone_id(drone_id)
        drone = self.env.get_drone_state(drone_id)

        return {
            "drone_id": drone_id,
            "position": {
                "x": round(float(drone.position[0]), 3),
                "y": round(float(drone.position[1]), 3),
                "z": round(float(drone.position[2]), 3),
            },
            "velocity": {
                "x": round(float(drone.velocity[0]), 3),
                "y": round(float(drone.velocity[1]), 3),
                "z": round(float(drone.velocity[2]), 3),
                "magnitude": round(float(np.linalg.norm(drone.velocity)), 3),
            },
            "battery": {
                "percentage": round(float(drone.battery.percentage), 1),
                "voltage": round(float(drone.battery.voltage), 2),
                "is_critical": drone.battery.is_critical,
            },
            "heading_deg": round(float(np.degrees(drone.heading)), 1),
            "detected_targets": drone.detected_targets,
            "is_active": drone.is_active,
            "mission_phase": drone.mission_phase,
        }

    def recon_fly_to(
        self,
        drone_id: int,
        x: float,
        y: float,
        z: float,
        velocity_limit: float = 2.0,
        altitude_mode: str = "absolute",
    ) -> Dict[str, Any]:
        """
        Command a drone to fly to a specific waypoint.

        Generates a direct trajectory from current position to target.
        """
        self._validate_drone_id(drone_id)
        drone = self.env.get_drone_state(drone_id)

        target = np.array([x, y, z], dtype=np.float64)
        if altitude_mode == "relative":
            target[2] += drone.position[2]

        # Clamp altitude
        target[2] = np.clip(target[2], 0.3, 30.0)

        # Generate trajectory
        self.env.set_trajectory(
            drone_id,
            [drone.position.copy(), target],
            speeds=[velocity_limit],
        )
        drone.mission_phase = "flying_to_waypoint"

        return {
            "drone_id": drone_id,
            "command": "fly_to",
            "target": {"x": x, "y": y, "z": float(target[2])},
            "velocity_limit": velocity_limit,
            "altitude_mode": altitude_mode,
            "estimated_distance": round(float(np.linalg.norm(target - drone.position)), 2),
        }

    def recon_orbit_point(
        self,
        drone_id: int,
        center_x: float,
        center_y: float,
        radius: float,
        speed: float = 1.5,
        altitude: float = 3.0,
    ) -> Dict[str, Any]:
        """
        Command a drone to orbit around a point.

        Generates circular waypoints around the center point.
        """
        self._validate_drone_id(drone_id)
        drone = self.env.get_drone_state(drone_id)

        center = np.array([center_x, center_y, altitude])

        # Generate circular orbit waypoints (16 points)
        n_points = 16
        waypoints = []
        for k in range(n_points + 1):  # +1 to close the loop
            angle = 2 * np.pi * k / n_points
            wx = center_x + radius * np.cos(angle)
            wy = center_y + radius * np.sin(angle)
            waypoints.append(np.array([wx, wy, altitude]))

        # Start from current position → first orbit point → orbit
        full_path = [drone.position.copy(), waypoints[0]] + waypoints
        segment_speeds = [speed] * (len(full_path) - 1)

        self.env.set_trajectory(drone_id, full_path, speeds=segment_speeds)
        drone.mission_phase = "orbiting"

        return {
            "drone_id": drone_id,
            "command": "orbit",
            "center": {"x": center_x, "y": center_y},
            "radius": radius,
            "speed": speed,
            "altitude": altitude,
            "n_waypoints": n_points,
        }

    def recon_area_search(
        self,
        drone_ids: List[int],
        bounding_box: Dict[str, float],
        pattern: str = "lawnmower",
        altitude: float = 3.0,
        speed: float = 1.5,
    ) -> Dict[str, Any]:
        """
        Assign drones to search a rectangular area.

        Lawnmower pattern: parallel sweep lines covering the area.
        Grid partition: divide area among drones, each searches its sub-region.
        """
        for did in drone_ids:
            self._validate_drone_id(did)

        x_min = bounding_box["x_min"]
        y_min = bounding_box["y_min"]
        x_max = bounding_box["x_max"]
        y_max = bounding_box["y_max"]

        if x_max <= x_min or y_max <= y_min:
            raise ValueError("Invalid bounding box: x_max > x_min and y_max > y_min required")

        n_drones = len(drone_ids)
        assignments = {}

        if pattern == "lawnmower":
            # Divide the area into horizontal strips, one per drone
            strip_width = (y_max - y_min) / n_drones
            for idx, did in enumerate(drone_ids):
                y_lo = y_min + idx * strip_width
                y_hi = y_lo + strip_width
                waypoints = self._generate_lawnmower(
                    x_min, x_max, y_lo, y_hi, altitude, strip_width * 0.8
                )
                drone = self.env.get_drone_state(did)
                full_path = [drone.position.copy()] + waypoints
                speeds = [speed] * (len(full_path) - 1)
                self.env.set_trajectory(did, full_path, speeds=speeds)
                drone.mission_phase = "area_search"
                assignments[did] = {
                    "sub_area": {"x_min": x_min, "y_min": y_lo, "x_max": x_max, "y_max": y_hi},
                    "n_waypoints": len(waypoints),
                }

        elif pattern == "grid_partition":
            # Divide into grid cells, assign nearest drone to each
            grid_size = int(np.ceil(np.sqrt(n_drones)))
            cell_w = (x_max - x_min) / grid_size
            cell_h = (y_max - y_min) / grid_size
            for idx, did in enumerate(drone_ids):
                gi = idx // grid_size
                gj = idx % grid_size
                cx_min = x_min + gj * cell_w
                cx_max = cx_min + cell_w
                cy_min = y_min + gi * cell_h
                cy_max = cy_min + cell_h
                waypoints = self._generate_lawnmower(
                    cx_min, cx_max, cy_min, cy_max, altitude, cell_h * 0.8
                )
                drone = self.env.get_drone_state(did)
                full_path = [drone.position.copy()] + waypoints
                speeds = [speed] * (len(full_path) - 1)
                self.env.set_trajectory(did, full_path, speeds=speeds)
                drone.mission_phase = "area_search"
                assignments[did] = {
                    "sub_area": {"x_min": cx_min, "y_min": cy_min, "x_max": cx_max, "y_max": cy_max},
                    "n_waypoints": len(waypoints),
                }
        else:
            raise ValueError(f"Unknown pattern: {pattern}")

        return {
            "command": "area_search",
            "pattern": pattern,
            "bounding_box": bounding_box,
            "altitude": altitude,
            "speed": speed,
            "assignments": assignments,
        }

    def recon_capture_target_intel(
        self,
        drone_id: int,
        target_id: int,
    ) -> Dict[str, Any]:
        """
        Capture intelligence on a target.

        Returns target coordinates, estimated position, and synthetic camera data.
        """
        self._validate_drone_id(drone_id)
        if target_id not in self.env.targets:
            raise ValueError(f"Unknown target: {target_id}")

        drone = self.env.get_drone_state(drone_id)
        target = self.env.get_target_state(target_id)

        # Check if drone has detected this target
        has_detected = target_id in drone.detected_targets

        # Compute relative geometry
        delta = target.position - drone.position
        distance = float(np.linalg.norm(delta))
        bearing = float(np.arctan2(delta[1], delta[0]))

        # Generate synthetic camera observation
        from .sensors import ReconCamera
        camera = self.env.sensor_suites[drone_id].camera
        obs = camera.project_target(
            drone.position, drone.quaternion,
            target.position, target.radius,
        )

        intel = {
            "drone_id": drone_id,
            "target_id": target_id,
            "target_world_coords": {
                "x": round(float(target.position[0]), 2),
                "y": round(float(target.position[1]), 2),
                "z": round(float(target.position[2]), 2),
            },
            "drone_to_target": {
                "distance_m": round(distance, 2),
                "bearing_deg": round(float(np.degrees(bearing)), 1),
            },
            "has_line_of_sight": has_detected,
            "camera_observation": None,
        }

        if obs is not None:
            intel["camera_observation"] = {
                "pixel_x": obs.pixel_x,
                "pixel_y": obs.pixel_y,
                "bbox_width": obs.pixel_width,
                "bbox_height": obs.pixel_height,
                "confidence": round(obs.confidence, 3),
                "in_fov": obs.in_fov,
            }

        # Command drone to approach target if not close
        if distance > 5.0:
            approach_pos = target.position.copy()
            approach_pos[2] = 2.0  # approach at 2m altitude
            self.env.set_trajectory(
                drone_id,
                [drone.position.copy(), approach_pos],
                speeds=[1.5],
            )
            intel["action"] = "approaching_target"
        else:
            intel["action"] = "holding_position"

        return intel

    def _validate_drone_id(self, drone_id: int) -> None:
        if drone_id not in self.env.drones:
            raise ValueError(f"Invalid drone_id: {drone_id}. Valid range: 0-{self.env.n_drones-1}")

    @staticmethod
    def _generate_lawnmower(
        x_min: float, x_max: float,
        y_min: float, y_max: float,
        altitude: float,
        line_spacing: float,
    ) -> List[np.ndarray]:
        """
        Generate lawnmower sweep waypoints for a rectangular area.

        Pattern: alternating left-to-right and right-to-left sweeps
        with spacing = line_spacing.
        """
        waypoints = []
        n_lines = max(1, int(np.ceil((y_max - y_min) / line_spacing)))
        y_lines = np.linspace(y_min, y_max, n_lines + 1)

        for i, y in enumerate(y_lines):
            if i % 2 == 0:
                waypoints.append(np.array([x_min, y, altitude]))
                waypoints.append(np.array([x_max, y, altitude]))
            else:
                waypoints.append(np.array([x_max, y, altitude]))
                waypoints.append(np.array([x_min, y, altitude]))

        return waypoints


def get_all_tool_schemas() -> List[Dict[str, Any]]:
    """Return all tool schemas for LLM function calling registration."""
    return list(TOOL_SCHEMAS.values())
