"""
swarm.py — Multi-Agent Swarm Environment for MuJoCo

Manages N reconnaissance drones in a single MuJoCo physics scene with:
    - Per-drone controller and sensor suite
    - Collision avoidance (Artificial Potential Fields)
    - Asynchronous or lockstep simulation stepping
    - Target management and detection
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import mujoco

from .physics import (
    DRONE_MASS, GRAVITY, quat_to_rotation_matrix,
    ground_effect_factor, downwash_force,
    translational_drag_force, rotational_drag_torque,
    MAX_THRUST_PER_MOTOR, normalized_to_thrust,
    RANGEFINDER_MAX_RANGE,
)
from .controller import CascadedQuadrotorController, TrajectoryGenerator
from .sensors import (
    SensorSuite, IMUReading, ProximityArray,
    TargetObservation, BatteryState,
)


@dataclass
class DroneState:
    """Full state of a single drone."""
    drone_id: int
    position: NDArray[np.float64]          # (3,) world position
    velocity: NDArray[np.float64]          # (3,) world velocity
    quaternion: NDArray[np.float64]        # (4,) [w, x, y, z]
    angular_velocity: NDArray[np.float64]  # (3,) body angular velocity
    heading: float                         # yaw angle (rad)
    battery: BatteryState
    is_active: bool = True
    mission_phase: str = "idle"
    detected_targets: List[int] = field(default_factory=list)


@dataclass
class GroundTarget:
    """Static ground target for reconnaissance."""
    target_id: int
    position: NDArray[np.float64]  # (3,) world position
    radius: float = 0.5            # bounding sphere radius (m)
    is_detected: bool = False
    detected_by: List[int] = field(default_factory=list)


@dataclass
class CollisionAvoidanceConfig:
    """Parameters for Artificial Potential Field collision avoidance."""
    # Repulsive field parameters
    repulsive_gain: float = 2.0           # gain for repulsive potential
    influence_distance: float = 3.0       # distance at which repulsion activates (m)
    # Drone-drone minimum separation
    min_separation: float = 1.5           # minimum distance between drones (m)
    # Terrain avoidance
    min_altitude: float = 0.3             # minimum altitude (m)
    max_altitude: float = 30.0            # maximum altitude (m)
    # Smoothing
    avoidance_weight: float = 0.5         # blend weight with nominal control


class ArtificialPotentialField:
    """
    Decentralized collision avoidance using Artificial Potential Fields.

    Each drone computes a repulsive velocity correction based on:
        1. Proximity to other drones (inter-agent repulsion)
        2. Altitude constraints (ground/ceiling repulsion)
        3. Static obstacles (if obstacle positions are known)

    The repulsive force follows an inverse-distance law:
        F_rep = η * (1/d - 1/d₀) * (1/d²) * ∇d
    where d is distance to obstacle, d₀ is influence distance, η is gain.
    """

    def __init__(self, config: CollisionAvoidanceConfig | None = None):
        self.config = config or CollisionAvoidanceConfig()

    def compute_avoidance_velocity(
        self,
        drone_id: int,
        drone_position: NDArray[np.float64],
        all_positions: Dict[int, NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        """
        Compute repulsive velocity correction for collision avoidance.

        Parameters
        ----------
        drone_id : ID of the drone being controlled
        drone_position : (3,) current position
        all_positions : dict of {drone_id: position} for all active drones

        Returns
        -------
        v_avoid : (3,) velocity correction to add to nominal command
        """
        cfg = self.config
        v_avoid = np.zeros(3)

        # ── Inter-agent repulsion ────────────────────────────────────────────
        for other_id, other_pos in all_positions.items():
            if other_id == drone_id:
                continue
            delta = drone_position - other_pos
            dist = np.linalg.norm(delta)
            if dist < 1e-3:
                # Exact overlap: push in random direction
                delta = np.array([0.1, 0.0, 0.05])
                dist = 0.1

            if dist < cfg.influence_distance and dist < cfg.min_separation * 2:
                # Repulsive force magnitude
                if dist < cfg.min_separation:
                    # Strong repulsion when very close
                    strength = cfg.repulsive_gain * (1.0 / dist - 1.0 / cfg.min_separation) / (dist**2)
                    strength = min(strength, 5.0)  # cap
                else:
                    # Weaker repulsion in influence zone
                    strength = cfg.repulsive_gain * 0.5 / (dist**2)

                direction = delta / dist
                v_avoid += strength * direction

        # ── Altitude constraints ─────────────────────────────────────────────
        z = drone_position[2]
        if z < cfg.min_altitude:
            v_avoid[2] += cfg.repulsive_gain * (cfg.min_altitude - z) * 2.0
        elif z > cfg.max_altitude:
            v_avoid[2] -= cfg.repulsive_gain * (z - cfg.max_altitude) * 0.5

        # Clamp maximum avoidance velocity
        speed = np.linalg.norm(v_avoid)
        max_avoid_speed = 3.0  # m/s
        if speed > max_avoid_speed:
            v_avoid = v_avoid * (max_avoid_speed / speed)

        return v_avoid


class SwarmEnvironment:
    """
    Multi-agent swarm simulation environment wrapping MuJoCo.

    Manages N drones in a single physics scene with:
        - Per-drone controller and sensor suite
        - Lockstep or asynchronous stepping
        - Collision avoidance
        - Target tracking
        - Telemetry collection
    """

    def __init__(
        self,
        model_path: str,
        n_drones: int = 4,
        n_targets: int = 2,
        scene_size: float = 20.0,
        dt: float = 0.001,
        control_dt: float = 0.01,       # control runs at 100 Hz
        seed: int = 42,
    ):
        """
        Parameters
        ----------
        model_path : path to recon_quadrotor.xml
        n_drones : number of drones to spawn
        n_targets : number of ground targets
        scene_size : arena half-size (m)
        dt : physics timestep
        control_dt : control loop timestep
        """
        self.n_drones = n_drones
        self.n_targets = n_targets
        self.scene_size = scene_size
        self.dt = dt
        self.control_dt = control_dt
        self.rng = np.random.default_rng(seed)

        # Load MuJoCo model template
        self.model_template = mujoco.MjModel.from_xml_path(model_path)
        self.model_template.opt.timestep = dt

        # Per-drone MuJoCo instances (each drone has its own physics)
        self.models: Dict[int, mujoco.MjModel] = {}
        self.datas: Dict[int, mujoco.MjData] = {}

        # ── Initialize Drones ────────────────────────────────────────────────
        self.drones: Dict[int, DroneState] = {}
        self.controllers: Dict[int, CascadedQuadrotorController] = {}
        self.sensor_suites: Dict[int, SensorSuite] = {}
        self.trajectories: Dict[int, TrajectoryGenerator | None] = {}

        # Spawn drones in a grid pattern
        spacing = 2.0
        for i in range(n_drones):
            row = i // 2
            col = i % 2
            x = -spacing + col * 2 * spacing
            y = -spacing + row * 2 * spacing
            z = 1.5

            pos = np.array([x, y, z])
            quat = np.array([1.0, 0.0, 0.0, 0.0])  # identity quaternion

            self.drones[i] = DroneState(
                drone_id=i,
                position=pos.copy(),
                velocity=np.zeros(3),
                quaternion=quat.copy(),
                angular_velocity=np.zeros(3),
                heading=0.0,
                battery=BatteryState(
                    capacity_wh=4.5, percentage=100.0,
                    voltage=4.2, is_critical=False,
                    total_energy_consumed_wh=0.0,
                ),
            )

            self.controllers[i] = CascadedQuadrotorController()
            self.sensor_suites[i] = SensorSuite(drone_id=i, seed=seed + i)
            self.trajectories[i] = None

            # Create per-drone MuJoCo instance
            self.models[i] = mujoco.MjModel.from_xml_path(model_path)
            self.models[i].opt.timestep = dt
            self.datas[i] = mujoco.MjData(self.models[i])

        # ── Initialize Targets ───────────────────────────────────────────────
        self.targets: Dict[int, GroundTarget] = {}
        for j in range(n_targets):
            tx = self.rng.uniform(-scene_size * 0.6, scene_size * 0.6)
            ty = self.rng.uniform(-scene_size * 0.6, scene_size * 0.6)
            self.targets[j] = GroundTarget(
                target_id=j,
                position=np.array([tx, ty, 0.0]),
                radius=0.5,
            )

        # ── Collision Avoidance ──────────────────────────────────────────────
        self.apf = ArtificialPotentialField()

        # ── Simulation State ─────────────────────────────────────────────────
        self.step_count = 0
        self.sim_time = 0.0
        self.control_step_interval = int(control_dt / dt)

        # Telemetry log
        self.telemetry_log: List[Dict[str, Any]] = []
        self.detection_log: List[Dict[str, Any]] = []

        # Initialize per-drone MuJoCo data
        for i in self.drones:
            self._sync_state_to_mujoco(i)
            mujoco.mj_forward(self.models[i], self.datas[i])

    def _sync_state_to_mujoco(self, drone_id: int) -> None:
        """Write drone state into its MuJoCo data structure."""
        drone = self.drones[drone_id]
        if not drone.is_active:
            return
        data = self.datas[drone_id]
        data.qpos[0:3] = drone.position
        data.qpos[3:7] = drone.quaternion
        data.qvel[0:3] = drone.velocity
        data.qvel[3:6] = drone.angular_velocity

    def _read_state_from_mujoco(self, drone_id: int) -> None:
        """Read drone state from its MuJoCo data structure."""
        drone = self.drones[drone_id]
        if not drone.is_active:
            return
        data = self.datas[drone_id]
        drone.position = data.qpos[0:3].copy()
        drone.quaternion = data.qpos[3:7].copy()
        drone.velocity = data.qvel[0:3].copy()
        drone.angular_velocity = data.qvel[3:6].copy()

        R = quat_to_rotation_matrix(drone.quaternion)
        euler = np.arctan2(R[1, 0], R[0, 0])
        drone.heading = euler

    def set_trajectory(self, drone_id: int, waypoints: List[NDArray[np.float64]],
                       speeds: List[float] | None = None) -> None:
        """Assign a waypoint trajectory to a drone."""
        self.trajectories[drone_id] = TrajectoryGenerator(waypoints, speeds)

    def set_hover(self, drone_id: int, position: NDArray[np.float64]) -> None:
        """Set drone to hover at a fixed position."""
        self.trajectories[drone_id] = TrajectoryGenerator([position, position])

    def step(self) -> Dict[str, Any]:
        """
        Advance simulation by one control timestep.

        Returns
        -------
        telemetry : dict with per-drone state and sensor data
        """
        # ── Control Loop ─────────────────────────────────────────────────────
        all_positions = {
            i: d.position for i, d in self.drones.items() if d.is_active
        }

        for i, drone in self.drones.items():
            if not drone.is_active:
                continue

            # Get reference trajectory
            traj = self.trajectories.get(i)
            if traj is not None:
                pos_ref, vel_ref, acc_ref = traj.get_reference(self.sim_time)
            else:
                pos_ref = drone.position.copy()
                vel_ref = np.zeros(3)
                acc_ref = np.zeros(3)

            # Collision avoidance correction
            v_avoid = self.apf.compute_avoidance_velocity(
                i, drone.position, all_positions
            )
            # Blend avoidance into position reference
            if np.linalg.norm(v_avoid) > 0.01:
                pos_ref = pos_ref + v_avoid * self.control_dt

            # Compute control
            ctrl, ctrl_info = self.controllers[i].compute_control(
                position=drone.position,
                velocity=drone.velocity,
                quaternion=drone.quaternion,
                angular_velocity=drone.angular_velocity,
                target_position=pos_ref,
                target_velocity=vel_ref,
                target_acceleration=acc_ref,
                dt=self.control_dt,
            )

            # Apply control to this drone's MuJoCo instance
            data = self.datas[i]
            model = self.models[i]
            data.ctrl[0:4] = ctrl

            # ── Physics Stepping ─────────────────────────────────────────────
            for _ in range(self.control_step_interval):
                mujoco.mj_step(model, data)

            # ── Read State Back ──────────────────────────────────────────────
            self._read_state_from_mujoco(i)

            # ── Update Battery ───────────────────────────────────────────────
            thrust_ratio = ctrl_info["total_thrust"] / (DRONE_MASS * GRAVITY)
            drone.battery = self.sensor_suites[i].battery.update(
                thrust_ratio, self.control_dt
            )

            # ── Target Detection ─────────────────────────────────────────────
            for t_id, target in self.targets.items():
                obs = self.sensor_suites[i].camera.project_target(
                    drone.position, drone.quaternion,
                    target.position, target.radius,
                )
                if obs is not None:
                    obs.target_id = t_id
                    if t_id not in drone.detected_targets:
                        drone.detected_targets.append(t_id)
                        target.is_detected = True
                        if i not in target.detected_by:
                            target.detected_by.append(i)
                    self.detection_log.append({
                        "time": self.sim_time,
                        "drone_id": i,
                        "target_id": t_id,
                        "distance": obs.distance,
                        "pixel": [obs.pixel_x, obs.pixel_y],
                        "confidence": obs.confidence,
                    })

            # ── Telemetry ────────────────────────────────────────────────────
            self.telemetry_log.append({
                "step": self.step_count,
                "time": self.sim_time,
                "drone_id": i,
                "position": drone.position.tolist(),
                "velocity": drone.velocity.tolist(),
                "heading": drone.heading,
                "battery_pct": drone.battery.percentage,
                "motor_commands": ctrl_info["motor_commands"],
                "thrust": ctrl_info["total_thrust"],
                "targets_detected": drone.detected_targets.copy(),
            })

        self.step_count += 1
        self.sim_time += self.control_dt

        return self._collect_telemetry()

    def _collect_telemetry(self) -> Dict[str, Any]:
        """Collect current telemetry snapshot for all drones."""
        telemetry = {
            "time": self.sim_time,
            "step": self.step_count,
            "drones": {},
            "targets": {},
        }
        for i, drone in self.drones.items():
            telemetry["drones"][i] = {
                "position": drone.position.tolist(),
                "velocity": drone.velocity.tolist(),
                "heading": drone.heading,
                "battery_pct": drone.battery.percentage,
                "is_active": drone.is_active,
                "detected_targets": drone.detected_targets.copy(),
            }
        for t_id, target in self.targets.items():
            telemetry["targets"][t_id] = {
                "position": target.position.tolist(),
                "is_detected": target.is_detected,
                "detected_by": target.detected_by.copy(),
            }
        return telemetry

    def get_drone_state(self, drone_id: int) -> DroneState:
        """Get full state of a specific drone."""
        return self.drones[drone_id]

    def get_target_state(self, target_id: int) -> GroundTarget:
        """Get state of a specific target."""
        return self.targets[target_id]

    def get_all_detections(self) -> List[Dict[str, Any]]:
        """Return all target detection events."""
        return self.detection_log.copy()

    def get_mission_summary(self) -> Dict[str, Any]:
        """Generate end-of-mission summary."""
        total_targets = len(self.targets)
        detected_targets = sum(1 for t in self.targets.values() if t.is_detected)

        return {
            "total_sim_time": self.sim_time,
            "total_steps": self.step_count,
            "n_drones": self.n_drones,
            "total_targets": total_targets,
            "detected_targets": detected_targets,
            "detection_rate": detected_targets / max(total_targets, 1),
            "per_drone": {
                i: {
                    "final_position": d.position.tolist(),
                    "battery_remaining_pct": d.battery.percentage,
                    "targets_detected": len(d.detected_targets),
                    "energy_consumed_wh": d.battery.total_energy_consumed_wh,
                }
                for i, d in self.drones.items()
            },
            "total_detections": len(self.detection_log),
        }
