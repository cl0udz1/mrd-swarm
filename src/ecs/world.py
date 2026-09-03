# -*- coding: utf-8 -*-
"""
world.py — ECS World Container & System Pipeline Manager with Black Box Logging

Orchestrates data-oriented components, executes discrete stateless systems
at 100 Hz, and streams aerospace evaluation telemetry at 60 Hz.
"""

from __future__ import annotations
import math
from typing import Dict, List, Tuple, Optional, Any, Set
import numpy as np

from ..controller import GeometricSE3Controller, CascadedQuadrotorController
from ..perception import VoxelUncertaintyGrid, LineOfSightSensor
from ..navigation import APFReactiveNavigator
from ..ai_agent_core import HETEROGENEOUS_SPECS
from ..physics import DrydenTurbulenceModel
from ..flight_recorder import FlightDataRecorder
from .components import (
    TransformComponent, PhysicsBodyComponent, SensorComponent,
    BatteryComponent, TacticalComponent, RFMeshComponent, TargetEntityComponent,
    LaserDesignatorComponent, EWJammingField, TacticalRoleID, TargetStateID,
)
from .systems import (
    evasion_system, perception_system, rf_mesh_system,
    laser_designation_system, brain_decision_system,
    apf_navigation_system, se3_control_system, battery_discharge_system,
)
from .mission_state import MissionPhase, MissionStateManager
from .target_tracker import KalmanTargetTracker, EKFTargetTracker
from .doctrines import TacticalDoctrineID, get_doctrine_config
from ..ai_commander import DeepSeekSwarmCommander
from ..ai_vision_recon import DeepSeekVisionRecon


class ECSWorld:
    """
    ECS-inspired modular simulation architecture with Black Box Flight Data Recording.
    """

    def __init__(self, obstacles: List[Dict[str, Any]], seed: int = 42):
        self.obstacles = obstacles
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.sim_time = 0.0
        self.dt = 0.01  # 100 Hz step

        # Atmospheric Turbulence Model (MIL-F-8785C Dryden Filter)
        self.dryden_wind = DrydenTurbulenceModel(dt=self.dt, altitude_m=10.0, wind_speed_20m=3.0, seed=seed)

        # Sub-Agent Helper Systems
        self.uncertainty_grid = VoxelUncertaintyGrid(obstacles=obstacles)
        self.los_sensor = LineOfSightSensor(obstacles)
        self.navigator = APFReactiveNavigator()
        self.recorder = FlightDataRecorder(max_buffer_size=60000)
        self.mission_mgr = MissionStateManager(n_targets=3, cruise_altitude=1.0)
        self.target_tracker = KalmanTargetTracker(n_targets=3)

        # Perception & Detection Canonical Metrics
        self.total_detection_events: int = 0
        self.total_visible_target_frames: int = 0
        self.unique_targets_detected: Set[int] = set()
        self.confirmed_track_events: int = 0

        # DeepSeek AI Cognitive Layer (Async LLM Tactical Commander & Vision Recon)
        self.ai_commander = DeepSeekSwarmCommander()
        self.vision_recon = DeepSeekVisionRecon()

        # Swarm Tactical Doctrine
        self.current_doctrine = TacticalDoctrineID.DEEPSEEK_ADAPTIVE

        # Dynamic Electronic Warfare Jamming Field (Sector 2 East Corridor)
        self.ew_field = EWJammingField(
            active=False,
            center=np.array([14.0, 14.0, 4.0]),
            radius=15.0,
            intensity=0.85,
        )

        # Rooftop Helipads
        self.helipads = [
            {"id": 0, "name": "Helipad Alpha (Complex Bravo)", "pos": [-14.0, 12.0, 6.2], "size": [3.0, 3.0]},
            {"id": 1, "name": "Helipad Bravo (Depot Delta)", "pos": [15.0, -15.0, 5.2], "size": [3.0, 3.0]},
        ]

        # ── 1. Initialize Drone Components (Entities 0, 1, 2, 3) ───────────────
        self.drone_transforms: Dict[int, TransformComponent] = {
            0: TransformComponent(np.array([-8.0, 8.0, 1.5]), np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3), np.zeros(3)),
            1: TransformComponent(np.array([8.0, 8.0, 1.5]), np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3), np.zeros(3)),
            2: TransformComponent(np.array([-8.0, -8.0, 1.5]), np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3), np.zeros(3)),
            3: TransformComponent(np.array([0.0, 0.0, 9.5]), np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3), np.zeros(3)),
        }

        self.physics: Dict[int, PhysicsBodyComponent] = {
            i: PhysicsBodyComponent(
                mass=HETEROGENEOUS_SPECS[i].mass,
                arm_length=HETEROGENEOUS_SPECS[i].arm_length,
                thrust_margin=HETEROGENEOUS_SPECS[i].thrust_margin,
                motor_thrusts=np.zeros(4),
            )
            for i in range(4)
        }

        self.sensors: Dict[int, SensorComponent] = {
            i: SensorComponent(
                camera_fov_deg=HETEROGENEOUS_SPECS[i].camera_fov_deg,
                max_sensor_range=HETEROGENEOUS_SPECS[i].max_sensor_range,
                has_thermal_ir=(i == 2),
            )
            for i in range(4)
        }

        self.batteries: Dict[int, BatteryComponent] = {
            i: BatteryComponent(
                capacity_wh=HETEROGENEOUS_SPECS[i].battery_capacity_wh,
                remaining_wh=HETEROGENEOUS_SPECS[i].battery_capacity_wh,
            )
            for i in range(4)
        }

        self.tacticals: Dict[int, TacticalComponent] = {
            i: TacticalComponent(
                role=TacticalRoleID.EXPLORER if i != 3 else TacticalRoleID.RELAY,
                goal_position=self.drone_transforms[i].position.copy(),
            )
            for i in range(4)
        }

        self.lasers: Dict[int, LaserDesignatorComponent] = {
            i: LaserDesignatorComponent() for i in range(4)
        }

        self.meshes: Dict[int, RFMeshComponent] = {
            i: RFMeshComponent(comm_range_m=32.0 if i == 3 else 18.0)
            for i in range(4)
        }

        self.controllers: Dict[int, CascadedQuadrotorController] = {
            i: CascadedQuadrotorController(mass=HETEROGENEOUS_SPECS[i].mass)
            for i in range(4)
        }

        # ── 2. Initialize Target Components (Entities 0, 1, 2) ─────────────────
        self.target_transforms: Dict[int, TransformComponent] = {
            0: TransformComponent(np.array([16.0, -8.0, 0.3]), np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3), np.zeros(3)),
            1: TransformComponent(np.array([-22.0, 10.0, 0.3]), np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3), np.zeros(3)),
            2: TransformComponent(np.array([8.0, 24.0, 0.3]), np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3), np.zeros(3)),
        }

        self.targets: Dict[int, TargetEntityComponent] = {
            0: TargetEntityComponent(
                target_id=0, name="Convoy Alpha",
                waypoints=[
                    np.array([16.0, -8.0, 0.35]), np.array([4.0, -14.0, 0.35]),
                    np.array([-8.0, -8.0, 0.35]), np.array([-20.0, 8.0, 0.35]),
                    np.array([8.0, 20.0, 0.35]), np.array([22.0, 6.0, 0.35]),
                ],
                base_speed=2.4, evasion_speed=4.2,
            ),
            1: TargetEntityComponent(
                target_id=1, name="Fast Interceptor Bravo",
                waypoints=[
                    np.array([-22.0, 10.0, 0.30]), np.array([-12.0, 22.0, 0.30]),
                    np.array([10.0, 16.0, 0.30]), np.array([-4.0, -2.0, 0.30]),
                    np.array([-18.0, -14.0, 0.30]), np.array([-6.0, -18.0, 0.30]),
                ],
                base_speed=2.8, evasion_speed=4.8,
            ),
            2: TargetEntityComponent(
                target_id=2, name="Shadow Asset Charlie",
                waypoints=[
                    np.array([8.0, 24.0, 0.30]), np.array([22.0, 16.0, 0.30]),
                    np.array([14.0, -10.0, 0.30]), np.array([-6.0, -18.0, 0.30]),
                    np.array([-16.0, 14.0, 0.30]), np.array([4.0, 6.0, 0.30]),
                ],
                base_speed=2.2, evasion_speed=3.8,
            ),
        }

        self.active_links: List[Tuple[int, int]] = []
        self.detected_target_ids: Set[int] = set()

    def trigger_jamming(self, toggle: Optional[bool] = None) -> bool:
        self.ew_field.active = not self.ew_field.active if toggle is None else toggle
        return self.ew_field.active

    def trigger_smoke(self, target_id: int = 0) -> bool:
        if target_id in self.targets:
            t = self.targets[target_id]
            t.smoke_active = True
            t.smoke_timer = 7.0
            t.smoke_position = self.target_transforms[target_id].position.copy()
            t.state = TargetStateID.SMOKE_SCREEN_EVASION
            return True
        return False

    def trigger_pincer(self) -> bool:
        if len(self.detected_target_ids) > 0:
            tid = list(self.detected_target_ids)[0]
            self.tacticals[1].role = TacticalRoleID.TRACKER
            self.tacticals[1].assigned_target_id = tid
            self.tacticals[2].role = TacticalRoleID.FLANKER
            self.tacticals[2].assigned_target_id = tid
            return True
        return False

    def trigger_rtb(self, drone_id: int = 1) -> bool:
        if drone_id in self.tacticals:
            pad = self.helipads[0]["pos"]
            self.tacticals[drone_id].role = TacticalRoleID.RTB_RECOVERY
            self.tacticals[drone_id].goal_position = np.array([pad[0], pad[1], pad[2] + 0.8])
            self.tacticals[drone_id].desired_speed = 12.0
            self.tacticals[drone_id].active_tool = "emergency_rtb(Helipad Alpha)"
            self.tacticals[drone_id].reasoning = "Initiating emergency RTB tactical recovery at rooftop pad"
            return True
        return False

    def set_tactical_doctrine(self, doctrine: str | TacticalDoctrineID) -> str:
        """Set swarm tactical doctrine (AGGRESSIVE_PINCER, WOLFPACK_CONTAINMENT, STEALTH_SHADOW, DEEPSEEK_ADAPTIVE)."""
        if isinstance(doctrine, str):
            for doc_id in TacticalDoctrineID:
                if doc_id.name == doctrine.upper() or doc_id.value == doctrine.upper():
                    self.current_doctrine = doc_id
                    break
        else:
            self.current_doctrine = doctrine
        print(f"[ECS WORLD] Swarm Tactical Doctrine set to: {self.current_doctrine.value}")
        return self.current_doctrine.value

    def export_csv_logs(self, filepath: str) -> None:
        self.recorder.export_csv(filepath)

    def _generate_recon_camera_frame(self, drone_id: int = 1) -> np.ndarray:
        """
        Synthesizes an optical/thermal FPV camera frame for DeepSeek Vision Recon.
        Projects urban obstacles, ground targets, smoke screens, and laser markers.
        """
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (256, 256), color=(25, 30, 36))
        draw = ImageDraw.Draw(img)

        # Ground plane & horizon
        draw.rectangle([0, 120, 256, 256], fill=(35, 42, 50))
        for y in range(120, 256, 24):
            draw.line([(0, y), (256, y)], fill=(45, 55, 65), width=1)

        d_pos = self.drone_transforms[drone_id].position
        d_vel = self.drone_transforms[drone_id].velocity
        d_yaw = math.atan2(d_vel[1], d_vel[0] + 1e-6)

        # Render urban buildings
        for obs in self.obstacles:
            bx, by, bz = obs["pos"]
            rel_x = bx - d_pos[0]
            rel_y = by - d_pos[1]
            dist = math.hypot(rel_x, rel_y)
            if 2.0 < dist < 35.0:
                angle_to_b = math.atan2(rel_y, rel_x) - d_yaw
                angle_to_b = (angle_to_b + math.pi) % (2 * math.pi) - math.pi
                if abs(angle_to_b) < math.radians(45.0):
                    scr_x = 128 + int(angle_to_b / math.radians(45.0) * 128)
                    scale = 60.0 / max(dist, 3.0)
                    w = int(obs.get("size", [3.0, 3.0, 4.0])[0] * scale * 3.5)
                    obs_h = obs.get("height", obs.get("size", [3.0, 3.0, 4.0])[2] if len(obs.get("size", [])) > 2 else 4.0)
                    h = int(obs_h * scale * 3.5)
                    top = max(10, 180 - h)
                    draw.rectangle([scr_x - w//2, top, scr_x + w//2, 180], fill=(55, 65, 78), outline=(85, 100, 120))

        # Render ground targets
        for tid, t_trans in self.target_transforms.items():
            tx, ty, tz = t_trans.position
            rel_x = tx - d_pos[0]
            rel_y = ty - d_pos[1]
            dist = math.hypot(rel_x, rel_y)
            if dist < 32.0:
                angle_to_t = math.atan2(rel_y, rel_x) - d_yaw
                angle_to_t = (angle_to_t + math.pi) % (2 * math.pi) - math.pi
                if abs(angle_to_t) < math.radians(50.0):
                    scr_x = 128 + int(angle_to_t / math.radians(50.0) * 120)
                    scr_y = min(220, max(130, 160 + int(dist * 2.0)))
                    draw.rectangle([scr_x - 12, scr_y - 8, scr_x + 12, scr_y + 8], fill=(220, 45, 45), outline=(255, 255, 255))
                    draw.line([(scr_x - 18, scr_y), (scr_x + 18, scr_y)], fill=(0, 255, 180), width=1)
                    draw.line([(scr_x, scr_y - 14), (scr_x, scr_y + 14)], fill=(0, 255, 180), width=1)
                    if self.targets[tid].smoke_active:
                        draw.ellipse([scr_x - 28, scr_y - 28, scr_x + 28, scr_y + 28], outline=(200, 200, 210), width=2)

        # HUD reticle
        draw.line([(118, 128), (138, 128)], fill=(0, 255, 200), width=1)
        draw.line([(128, 118), (128, 138)], fill=(0, 255, 200), width=1)
        draw.text((8, 8), f"D{drone_id} RECON // EO/IR", fill=(0, 255, 200))
        draw.text((8, 20), f"ALT:{d_pos[2]:.1f}m SPD:{np.linalg.norm(d_vel):.1f}m/s", fill=(180, 200, 220))
        return np.array(img)

    def step(self) -> Dict[str, Any]:
        """
        Executes one full tick of all ECS systems (100 Hz).
        """
        t = self.sim_time

        # 1. Wind Turbulence Model (MIL-F-8785C Dryden Stochastic Filter)
        wind_vel = self.dryden_wind.step()

        # 2. Perception System
        self.detected_target_ids = perception_system(
            drone_transforms=self.drone_transforms,
            sensors=self.sensors,
            target_transforms=self.target_transforms,
            targets=self.targets,
            los_sensor=self.los_sensor,
            uncertainty_grid=self.uncertainty_grid,
            rng=self.rng,
        )
        if self.detected_target_ids:
            self.total_detection_events += len(self.detected_target_ids)
            self.unique_targets_detected.update(self.detected_target_ids)
        frame_vis_pairs = sum(len(s.visible_targets) for s in self.sensors.values())
        self.total_visible_target_frames += frame_vis_pairs
        confirmed_ids = self.target_tracker.get_confirmed_ids()
        self.confirmed_track_events += len(confirmed_ids)

        # 3. Evasion System
        evasion_system(
            targets=self.targets,
            target_transforms=self.target_transforms,
            drone_transforms=self.drone_transforms,
            detected_target_ids=self.detected_target_ids,
            obstacles=self.obstacles,
            dt=self.dt,
            sim_time=t,
        )

        # 4. RF Mesh System
        self.active_links = rf_mesh_system(
            drone_transforms=self.drone_transforms,
            meshes=self.meshes,
            ew_field=self.ew_field,
        )

        # 5. Laser Designation System
        laser_designation_system(
            drone_transforms=self.drone_transforms,
            tacticals=self.tacticals,
            target_transforms=self.target_transforms,
            lasers=self.lasers,
        )

        # 6. DeepSeek Cognitive Layer Triggers (Async non-blocking)
        step_idx = int(round(self.sim_time * 100))
        if step_idx % 30 == 0:  # Check tactical commander every 0.3s
            active_smokes_ids = [tid for tid, t_obj in self.targets.items() if t_obj.smoke_active]
            if hasattr(self.ai_commander, "request_tactical_evaluation"):
                self.ai_commander.request_tactical_evaluation(
                    sim_time=self.sim_time,
                    telemetry={
                        "drones": {
                            did: {
                                "pos": self.drone_transforms[did].position.tolist(),
                                "speed": float(np.linalg.norm(self.drone_transforms[did].velocity)),
                                "battery": float(self.batteries[did].soc_pct),
                                "role": self.tacticals[did].role.name,
                            }
                            for did in self.drone_transforms
                        },
                        "uncertainty_pct": float(self.uncertainty_grid.get_mean_uncertainty()),
                        "phase": self.mission_mgr.phase.name,
                    },
                    known_target_ids=set(self.target_tracker.get_tracked_ids()),
                )

        # Trigger Vision Reconnaissance every 2.5s or on sighting
        if step_idx % 250 == 0 or (len(self.detected_target_ids) > 0 and step_idx % 120 == 0):
            frame = self._generate_recon_camera_frame(drone_id=1)
            self.vision_recon.request_vision_analysis(frame, drone_id=1, camera_mode="RGB_EO")

        # 7. Brain Decision System (with Kalman tracking, utility allocation, AI Commander directive, and closed-loop vision)
        latest_ai_directive = self.ai_commander.get_latest_directive()
        active_vision_obs = self.vision_recon.get_active_vision_observation() if hasattr(self.vision_recon, "get_active_vision_observation") else None
        brain_decision_system(
            drone_transforms=self.drone_transforms,
            sensors=self.sensors,
            tacticals=self.tacticals,
            batteries=self.batteries,
            target_transforms=self.target_transforms,
            detected_target_ids=self.detected_target_ids,
            uncertainty_grid=self.uncertainty_grid,
            sim_time=t,
            mission_mgr=self.mission_mgr,
            tracker=self.target_tracker,
            ai_directive=latest_ai_directive,
            doctrine=self.current_doctrine,
            targets=self.targets,
            vision_observation=active_vision_obs,
        )

        # 7. APF Navigation System
        setpoints = apf_navigation_system(
            drone_transforms=self.drone_transforms,
            tacticals=self.tacticals,
            obstacles=self.obstacles,
            navigator=self.navigator,
        )

        # 8. Geometric SE(3) Control System
        se3_control_system(
            drone_transforms=self.drone_transforms,
            physics=self.physics,
            setpoints=setpoints,
            controllers=self.controllers,
            wind_vel=wind_vel,
            dt=self.dt,
        )

        # 9. Battery Discharge System
        battery_discharge_system(
            physics=self.physics,
            batteries=self.batteries,
            dt=self.dt,
        )

        # 10. Black Box Flight Data Recording (100 Hz)
        uncertainty_pct = float(self.uncertainty_grid.get_mean_uncertainty())
        self.recorder.record_step(
            sim_time=self.sim_time,
            drone_transforms=self.drone_transforms,
            physics=self.physics,
            batteries=self.batteries,
            tacticals=self.tacticals,
            setpoints=setpoints,
            target_transforms=self.target_transforms,
            targets=self.targets,
            detected_target_ids=self.detected_target_ids,
            active_links=self.active_links,
            ew_active=self.ew_field.active,
            uncertainty_pct=uncertainty_pct,
        )

        self.sim_time += self.dt

        # Active Smoke Screens telemetry
        active_smokes = []
        for tid, t_obj in self.targets.items():
            if t_obj.smoke_active:
                active_smokes.append({
                    "target_id": tid,
                    "pos": [round(float(p), 2) for p in t_obj.smoke_position],
                    "timer": round(t_obj.smoke_timer, 1),
                    "radius": 4.5,
                })

        # Active Laser Targeting Beams
        active_lasers = []
        for did, laser in self.lasers.items():
            if laser.active:
                active_lasers.append({
                    "drone_id": did,
                    "target_id": laser.target_id,
                    "origin": [round(float(p), 2) for p in self.drone_transforms[did].position],
                    "target_pos": [round(float(p), 2) for p in laser.target_pos],
                    "color": laser.laser_color,
                })

        # Construct Telemetry Payload with Live Metrics
        telemetry = {
            "type": "TELEMETRY_UPDATE",
            "time": round(self.sim_time, 3),
            "uncertainty_pct": round(uncertainty_pct, 1),
            "wind": [round(float(w), 2) for w in wind_vel],
            "drones": {
                did: {
                    "id": did,
                    "class": HETEROGENEOUS_SPECS[did].drone_class.value,
                    "pos": [round(float(p), 3) for p in self.drone_transforms[did].position],
                    "quat": [round(float(q), 4) for q in self.drone_transforms[did].quaternion],
                    "vel": [round(float(v), 2) for v in self.drone_transforms[did].velocity],
                    "speed": round(float(np.linalg.norm(self.drone_transforms[did].velocity)), 2),
                    "battery": round(float(self.batteries[did].soc_pct), 1),
                    "role": self.tacticals[did].role.name,
                    "active_tool": self.tacticals[did].active_tool,
                    "reasoning": self.tacticals[did].reasoning,
                    "visible_targets": self.sensors[did].visible_targets,
                    "laser_active": self.lasers[did].active,
                    "is_jammed": self.meshes[did].jammed,
                    "formation_angle_deg": round(self.tacticals[did].formation_angle_deg, 1),
                    "tti_seconds": round(self.tacticals[did].tti_seconds, 2) if self.tacticals[did].tti_seconds < 1e6 else None,
                    "threat_score": round(self.tacticals[did].threat_score, 3),
                }
                for did in self.drone_transforms
            },
            "targets": {
                tid: {
                    "id": tid,
                    "name": self.targets[tid].name,
                    "pos": [round(float(p), 3) for p in self.target_transforms[tid].position],
                    "vel": [round(float(v), 2) for v in self.target_transforms[tid].velocity],
                    "state": self.targets[tid].state.name,
                    "is_spotted": tid in self.detected_target_ids,
                    "smoke_active": self.targets[tid].smoke_active,
                }
                for tid in self.targets
            },
            "perception": {
                "detected_targets": sorted(list(self.detected_target_ids)),
                "num_detected": len(self.detected_target_ids),
                "visible_target_pairs": frame_vis_pairs,
                "drone_detections": {
                    did: list(s.visible_targets.keys()) for did, s in self.sensors.items()
                },
                "total_detection_events": self.total_detection_events,
                "total_visible_target_frames": self.total_visible_target_frames,
                "unique_targets_detected": sorted(list(self.unique_targets_detected)),
                "confirmed_track_events": self.confirmed_track_events,
                "mean_uncertainty_pct": round(uncertainty_pct, 2),
                "coverage_pct": round(float(self.uncertainty_grid.get_coverage_pct()), 2),
            },
            "rf_mesh": {
                "active_links": self.active_links,
                "total_links": len(self.active_links),
                "ew_jamming_active": self.ew_field.active,
            },
            "mission_state": self.mission_mgr.get_status_summary(),
            "target_tracks": self.target_tracker.get_telemetry(),
            "tactical_doctrine": self.current_doctrine.value,
            "ai_commander": self.ai_commander.get_latest_directive().to_dict(),
            "vision_recon": self.vision_recon.get_latest_card().to_dict(),
            "combat_effects": {
                "active_smokes": active_smokes,
                "active_lasers": active_lasers,
                "ew_jamming": {
                    "active": self.ew_field.active,
                    "center": [round(float(p), 2) for p in self.ew_field.center],
                    "radius": self.ew_field.radius,
                },
                "helipads": self.helipads,
            },
            "evaluation_metrics": self.recorder.get_live_metrics_summary(),
        }

        return telemetry
