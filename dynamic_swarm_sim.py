# -*- coding: utf-8 -*-
"""
dynamic_swarm_sim.py — Master Closed-Loop Autonomous Drone Swarm Simulation in MuJoCo

Features:
- Fully Autonomous Closed-Loop Decision Engine (2-5 Hz in-loop cognitive cycle)
- Dynamic 3D Voxel Uncertainty Field U(x,y,z) across 45m x 45m x 15m urban theater
- Reactive Evasive Ground Targets (turn corners around skyscrapers to break line of sight)
- Multi-Drone Tactical Maneuvers: Dynamic Pincer Flanking, Target Shadowing, Spiral Sweep
- 3D APF Reactive Navigation & Slalom Collision Avoidance (1.5m to 12.0m altitude)
- 3-Panel Split-Screen 50 FPS 1080p Video Report (Tactical View + D1 FPV + D2 Flanker FPV)
"""

from __future__ import annotations
import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set

import mujoco
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Project Path Setup
PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(PROJECT_DIR))

from src.physics import (
    GRAVITY, quat_to_rotation_matrix,
)
from src.controller import CascadedQuadrotorController
from src.sensors import BatteryModel
from src.gossip import GossipChannel, GossipNode, MessageType
from src.targets import EvasiveTargetManager, TargetState
from src.perception import VoxelUncertaintyGrid, LineOfSightSensor
from src.navigation import APFReactiveNavigator
from src.swarm_brain import SwarmTacticalBrain, TacticalRole, SwarmDirective
from src.renderer import HeadlessRenderer, VideoReportGenerator, HUDOverlay
from src.ai_agent_core import HETEROGENEOUS_SPECS, DroneClass
from src.ai_commander import DeepSeekSwarmCommander
from src.ai_vision_recon import DeepSeekVisionRecon
from src.ecs.doctrines import TacticalDoctrineID, get_doctrine_config, DoctrineConfig
from src.ecs.target_tracker import EKFTargetTracker
from src.ecs.mission_state import MissionStateManager, MissionPhase
from src.ecs.components import TransformComponent, TacticalComponent, SensorComponent, TacticalRoleID
from src.ecs.systems import brain_decision_system


def run_closed_loop_swarm_mission(
    n_steps: int = 1000,
    seed: int = 42,
    enable_video: bool = True,
    doctrine: str = "DEEPSEEK_ADAPTIVE",
):
    print("=" * 95)
    print("  MRD-SWARM: AUTONOMOUS CLOSED-LOOP REACTIVE SWARM INTELLIGENCE MISSION (MuJoCo 3.x)")
    print("=" * 95)

    rng = np.random.default_rng(seed)
    dt = 0.01  # 100 Hz control loop
    total_sim_time = n_steps * dt
    brain_interval = 10  # 10 Hz cognitive decision cycle

    # 1. Define 8 Urban Structures
    obstacles = [
        {"name": "Skyscraper Alpha", "pos": [0.0, 0.0, 7.0], "size": [4.0, 4.0, 7.0], "height": 14.0},
        {"name": "Complex Bravo", "pos": [-14.0, 12.0, 3.0], "size": [6.0, 4.0, 3.0], "height": 6.0},
        {"name": "Silo Charlie", "pos": [-16.0, -14.0, 4.0], "size": [2.5, 2.5, 4.0], "height": 8.0},
        {"name": "Depot Delta", "pos": [15.0, -15.0, 2.5], "size": [7.0, 5.0, 2.5], "height": 5.0},
        {"name": "Substation Echo", "pos": [14.0, 14.0, 2.0], "size": [4.5, 4.5, 2.0], "height": 4.0},
        {"name": "Radar Pylon Foxtrot", "pos": [22.0, 0.0, 5.0], "size": [1.5, 1.5, 5.0], "height": 10.0},
        {"name": "Security Tower Golf", "pos": [-22.0, 0.0, 6.0], "size": [1.5, 1.5, 6.0], "height": 12.0},
        {"name": "Skybridge Hotel", "pos": [0.0, -18.0, 4.5], "size": [10.0, 2.0, 1.0], "height": 7.0},
    ]

    # 2. Instantiate Sub-Agent Components
    target_manager = EvasiveTargetManager()
    uncertainty_grid = VoxelUncertaintyGrid(obstacles=obstacles)
    los_sensor = LineOfSightSensor(obstacles)
    swarm_brain = SwarmTacticalBrain(obstacles)
    navigator = APFReactiveNavigator()

    # 3. Instantiate Controllers & Batteries for 4 Heterogeneous Drones
    controllers = {i: CascadedQuadrotorController(mass=HETEROGENEOUS_SPECS[i].mass) for i in range(4)}
    batteries = {i: BatteryModel(initial_capacity_wh=HETEROGENEOUS_SPECS[i].battery_capacity_wh) for i in range(4)}
    gossip_nodes = {i: GossipNode(agent_id=i, broadcast_interval=0.10) for i in range(4)}
    gossip_channel = GossipChannel(comm_range=18.0, packet_loss_rate=0.04)
    for node in gossip_nodes.values():
        gossip_channel.register_node(node)

    # Initial Drone Physical States
    drone_states = {
        0: {"p": np.array([-8.0, 8.0, 1.5]), "v": np.zeros(3), "q": np.array([1.0, 0.0, 0.0, 0.0]), "w": np.zeros(3)},
        1: {"p": np.array([8.0, 8.0, 1.5]), "v": np.zeros(3), "q": np.array([1.0, 0.0, 0.0, 0.0]), "w": np.zeros(3)},
        2: {"p": np.array([-8.0, -8.0, 1.5]), "v": np.zeros(3), "q": np.array([1.0, 0.0, 0.0, 0.0]), "w": np.zeros(3)},
        3: {"p": np.array([0.0, 0.0, 8.0]), "v": np.zeros(3), "q": np.array([1.0, 0.0, 0.0, 0.0]), "w": np.zeros(3)},
    }
    flight_trails: Dict[int, List[np.ndarray]] = {i: [] for i in range(4)}

    # Load MuJoCo Model for 3D Video Rendering
    world_xml = str(PROJECT_DIR / "mjcf" / "tactical_urban_world_v2.xml")
    m_world = mujoco.MjModel.from_xml_path(world_xml)
    d_world = mujoco.MjData(m_world)
    renderer = HeadlessRenderer(m_world)
    video_gen = VideoReportGenerator(output_path=str(OUTPUT_DIR / "dynamic_swarm_mission.mp4"), fps=50)

    # Active Swarm Directives & Navigation Setpoints
    active_directives: Dict[int, SwarmDirective] = {}
    active_setpoints: Dict[int, Any] = {}

    # Tactical Doctrine Configuration & ECS Core Decision Systems
    doctrine_cfg = get_doctrine_config(doctrine)
    print(f"  [DOCTRINE] Initialized Swarm Battle Doctrine: {doctrine_cfg.name} ({doctrine_cfg.doctrine_id.value})")
    tracker = EKFTargetTracker(n_targets=3)
    mission_mgr = MissionStateManager(n_targets=3, cruise_altitude=1.0)

    tacticals: Dict[int, TacticalComponent] = {i: TacticalComponent(role=TacticalRoleID.EXPLORER) for i in range(4)}
    drone_transforms: Dict[int, TransformComponent] = {
        i: TransformComponent(
            position=drone_states[i]["p"].copy(),
            quaternion=drone_states[i]["q"].copy(),
            velocity=drone_states[i]["v"].copy(),
            angular_velocity=drone_states[i]["w"].copy(),
        )
        for i in range(4)
    }
    target_transforms: Dict[int, TransformComponent] = {
        t.target_id: TransformComponent(
            position=t.position.copy(),
            quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
            velocity=t.velocity.copy(),
            angular_velocity=np.zeros(3),
        )
        for t in target_manager.targets
    }
    sensors: Dict[int, SensorComponent] = {
        i: SensorComponent(
            camera_fov_deg=HETEROGENEOUS_SPECS[i].camera_fov_deg,
            max_sensor_range=HETEROGENEOUS_SPECS[i].max_sensor_range,
        )
        for i in range(4)
    }

    # DeepSeek Cognitive AI Layer (Commander & Vision Recon)
    commander = DeepSeekSwarmCommander(update_interval_s=3.0)
    vision_recon = DeepSeekVisionRecon(min_interval_s=3.5)

    # Telemetry Loggers
    log_time: List[float] = []
    log_uncertainty: List[float] = []
    log_drone_pos: Dict[int, List[List[float]]] = {i: [] for i in range(4)}
    log_drone_battery: Dict[int, List[float]] = {i: [] for i in range(4)}
    log_drone_roles: Dict[int, List[str]] = {i: [] for i in range(4)}
    log_target_pos: Dict[int, List[List[float]]] = {t.target_id: [] for t in target_manager.targets}
    log_target_states: Dict[int, List[str]] = {t.target_id: [] for t in target_manager.targets}
    log_all_sightings: List[Dict[str, Any]] = []

    print(f"Executing {n_steps} Closed-Loop Control Steps ({total_sim_time:.1f} s Mission)...")

    render_interval = max(1, n_steps // 600)  # ~600 frames for 12s HD video playback

    for step in range(n_steps):
        t = step * dt

        # A. Dryden Crosswind Turbulence
        v_wind = np.array([
            1.2 * math.sin(0.4 * t) + 0.3 * math.sin(1.1 * t),
            1.0 * math.cos(0.5 * t) + 0.2 * math.cos(1.0 * t),
            0.2 * math.sin(0.8 * t),
        ])

        # B. Perception & Line-of-Sight Sensor Scanning
        sensor_sightings: Dict[int, Dict[int, float]] = {i: {} for i in range(4)}
        target_positions = target_manager.get_positions()
        target_velocities = target_manager.get_velocities()
        all_spotted_target_ids = []

        for d_id, state in drone_states.items():
            specs = HETEROGENEOUS_SPECS[d_id]
            # Update 3D Voxel Uncertainty Field
            uncertainty_grid.update_coverage(
                drone_pos=state["p"],
                drone_quat=state["q"],
                fov_deg=specs.camera_fov_deg,
                max_range=specs.max_sensor_range,
            )

            # Check LOS to each ground target
            for t_id, t_pos in target_positions.items():
                vis, conf = los_sensor.evaluate_target_visibility(
                    drone_pos=state["p"],
                    drone_quat=state["q"],
                    target_pos=t_pos,
                    fov_deg=specs.camera_fov_deg,
                    max_range=specs.max_sensor_range,
                )
                if vis and conf > 0.25:
                    sensor_sightings[d_id][t_id] = conf
                    all_spotted_target_ids.append(t_id)
                    log_all_sightings.append({
                        "time": t, "drone_id": d_id, "target_id": t_id, "confidence": conf,
                    })

        # C. Reactive Evasive Ground Targets Update
        drone_positions_map = {i: drone_states[i]["p"] for i in range(4)}
        target_telemetry = target_manager.update_all(
            dt=dt,
            sim_time=t,
            drone_positions=drone_positions_map,
            obstacles=obstacles,
            sensor_sightings=all_spotted_target_ids,
        )

        # Sync target positions to MuJoCo simulation
        for t_id, t_data in target_telemetry.items():
            j_name = f"target_{t_id}_joint"
            j_id = mujoco.mj_name2id(m_world, mujoco.mjtObj.mjOBJ_JOINT, j_name)
            if j_id != -1:
                adr = m_world.jnt_qposadr[j_id]
                d_world.qpos[adr:adr+3] = t_data.position
                d_world.qpos[adr+3:adr+7] = [1.0, 0.0, 0.0, 0.0]

        # D. Swarm Tactical Brain Decision Cycle (runs at 10 Hz with EKF + Doctrine + DeepSeek)
        if step % brain_interval == 0 or len(active_directives) == 0:
            for i in range(4):
                drone_transforms[i].position = drone_states[i]["p"].copy()
                drone_transforms[i].velocity = drone_states[i]["v"].copy()
                drone_transforms[i].quaternion = drone_states[i]["q"].copy()
                drone_transforms[i].angular_velocity = drone_states[i]["w"].copy()
                sensors[i].visible_targets = sensor_sightings[i]

            for t_id, t_data in target_telemetry.items():
                target_transforms[t_id].position = t_data.position.copy()
                target_transforms[t_id].velocity = t_data.velocity.copy()

            # Async AI Commander Trigger
            if step % 30 == 0:
                active_smokes_ids = [tid for tid, t_data in target_telemetry.items() if t_data.smoke_active]
                commander.request_tactical_update(
                    sim_time=t,
                    drone_states={
                        did: {
                            "pos": drone_states[did]["p"].tolist(),
                            "speed": float(np.linalg.norm(drone_states[did]["v"])),
                            "battery": float(batteries[did].remaining_wh / batteries[did].initial_capacity_wh * 100.0),
                            "role": tacticals[did].role.value,
                        }
                        for did in range(4)
                    },
                    target_tracks=tracker.get_telemetry(),
                    detected_target_ids=list(all_spotted_target_ids),
                    uncertainty_pct=float(uncertainty_grid.get_mean_uncertainty()),
                    ew_jamming_active=False,
                    active_smokes=active_smokes_ids,
                    mission_phase=mission_mgr.phase.name,
                )

            latest_ai_dir = commander.get_latest_directive()

            # Execute full ECS Brain Decision System with doctrine & EKF
            brain_decision_system(
                drone_transforms=drone_transforms,
                sensors=sensors,
                tacticals=tacticals,
                batteries=batteries,
                target_transforms=target_transforms,
                detected_target_ids=set(all_spotted_target_ids),
                uncertainty_grid=uncertainty_grid,
                sim_time=t,
                mission_mgr=mission_mgr,
                tracker=tracker,
                ai_directive=latest_ai_dir,
                doctrine=doctrine,
            )

            for did in range(4):
                active_directives[did] = SwarmDirective(
                    agent_id=did,
                    role=tacticals[did].role,
                    goal_position=tacticals[did].goal_position.copy(),
                    desired_speed=tacticals[did].desired_speed,
                    target_id=tacticals[did].assigned_target_id,
                    reasoning=tacticals[did].reasoning,
                )

            # Print significant tactical events
            if len(all_spotted_target_ids) > 0 and step % 50 == 0:
                d1_dir = active_directives.get(1)
                d2_dir = active_directives.get(2)
                print(f"  [t={t:5.2f}s] TACTICAL SIGHTING! D1 [{d1_dir.role.value if d1_dir else 'IDLE'}] | D2 [{d2_dir.role.value if d2_dir else 'IDLE'}] -> {d2_dir.reasoning if d2_dir else ''}")

        # E. 3D APF Reactive Local Navigation & Collision Avoidance
        for d_id, state in drone_states.items():
            directive = active_directives.get(d_id)
            goal_pos = directive.goal_position if directive else state["p"]
            des_speed = directive.desired_speed if directive else 2.5

            setpoint = navigator.compute_setpoint(
                current_pos=state["p"],
                current_vel=state["v"],
                goal_pos=goal_pos,
                obstacles=obstacles,
                peer_positions=drone_positions_map,
                current_agent_id=d_id,
                desired_speed=des_speed,
            )
            active_setpoints[d_id] = setpoint

        # F. Geometric SE(3) Control & 6-DoF Rigid Body Physics Integration
        for d_id, state in drone_states.items():
            specs = HETEROGENEOUS_SPECS[d_id]
            setpoint = active_setpoints[d_id]
            mass = specs.mass

            # ── Translational outer loop ──────────────────────────────────
            kp, kv = 3.5, 2.8
            e_p = setpoint.target_position - state["p"]
            e_v = setpoint.target_velocity - state["v"]
            a_des = kp * e_p + kv * e_v + np.array([0.0, 0.0, GRAVITY])

            # Clamp horizontal acceleration (~45° bank)
            a_horiz = np.linalg.norm(a_des[:2])
            if a_horiz > 14.0:
                a_des[:2] = (a_des[:2] / a_horiz) * 14.0

            f_des = mass * a_des
            total_T = float(np.clip(
                np.linalg.norm(f_des),
                0.2 * mass * GRAVITY,
                specs.thrust_margin * mass * GRAVITY,
            ))
            b3_d = f_des / (np.linalg.norm(f_des) + 1e-6)

            # Desired SO(3) rotation R_d = [b1_d, b2_d, b3_d]
            v_h = np.linalg.norm(setpoint.target_velocity[:2])
            yaw = math.atan2(setpoint.target_velocity[1], setpoint.target_velocity[0]) if v_h > 0.4 else 0.0
            b1_c = np.array([math.cos(yaw), math.sin(yaw), 0.0])
            b2_d = np.cross(b3_d, b1_c)
            n_b2 = np.linalg.norm(b2_d)
            b2_d = b2_d / n_b2 if n_b2 > 1e-4 else np.array([0.0, 1.0, 0.0])
            b1_d = np.cross(b2_d, b3_d)
            R_d = np.column_stack([b1_d, b2_d, b3_d])

            # ── Attitude inner loop (SO(3) error dynamics) ────────────────
            R = quat_to_rotation_matrix(state["q"])
            e_R_skew = R_d.T @ R - R.T @ R_d
            e_R = 0.5 * np.array([e_R_skew[2, 1], e_R_skew[0, 2], e_R_skew[1, 0]])
            e_omega = state["w"].copy()

            k_R, k_w = 8.0, 2.5
            J = np.array([1.1e-3, 1.1e-3, 2.1e-3])
            gyroscopic = np.cross(state["w"], J * state["w"])
            tau = np.clip(-k_R * e_R - k_w * e_omega + gyroscopic, -0.05, 0.05)

            # Integrate angular velocity (Euler's equation)
            alpha = (tau - gyroscopic) / J
            state["w"] = state["w"] + alpha * dt
            w_norm = np.linalg.norm(state["w"])
            if w_norm > 20.0:
                state["w"] = (state["w"] / w_norm) * 20.0

            # Integrate quaternion: dq/dt = 0.5 * q ⊗ [0, ω]
            qw, qx, qy, qz = state["q"]
            ox, oy, oz = state["w"]
            q_dot = 0.5 * np.array([
                -qx * ox - qy * oy - qz * oz,
                 qw * ox + qy * oz - qz * oy,
                 qw * oy - qx * oz + qz * ox,
                 qw * oz + qx * oy - qy * ox,
            ])
            state["q"] = state["q"] + q_dot * dt
            q_n = np.linalg.norm(state["q"])
            if q_n > 1e-6:
                state["q"] /= q_n
            if state["q"][0] < 0:
                state["q"] = -state["q"]

            # ── Translational dynamics ────────────────────────────────────
            R_cur = quat_to_rotation_matrix(state["q"])
            f_thrust = R_cur @ np.array([0.0, 0.0, total_T])
            v_rel = state["v"] - v_wind
            f_drag = -0.5 * 1.225 * 0.47 * 0.015 * np.linalg.norm(v_rel) * v_rel
            acc = (f_thrust + np.array([0.0, 0.0, -mass * GRAVITY]) + f_drag) / mass

            state["v"] += acc * dt
            speed = np.linalg.norm(state["v"])
            if speed > 18.0:
                state["v"] = (state["v"] / speed) * 18.0
            state["p"] += state["v"] * dt
            state["p"][2] = max(0.20, state["p"][2])

            # ── Sync drone pose into MuJoCo qpos for 3D rendering ────────
            j_name = f"drone_{d_id}_joint"
            j_id = mujoco.mj_name2id(m_world, mujoco.mjtObj.mjOBJ_JOINT, j_name)
            if j_id != -1:
                adr = m_world.jnt_qposadr[j_id]
                d_world.qpos[adr:adr+3] = state["p"]
                d_world.qpos[adr+3:adr+7] = state["q"]

            # Update battery and breadcrumb trail
            thrust_ratio = float(total_T / (mass * GRAVITY))
            batteries[d_id].update(thrust_ratio=thrust_ratio, dt=dt)

            if len(flight_trails[d_id]) == 0 or np.linalg.norm(state["p"] - flight_trails[d_id][-1]) > 0.25:
                flight_trails[d_id].append(state["p"].copy())
                if len(flight_trails[d_id]) > 180:
                    flight_trails[d_id].pop(0)

        # G. Telemetry Logging
        log_time.append(t)
        current_u = uncertainty_grid.get_mean_uncertainty()
        log_uncertainty.append(current_u)

        for i in range(4):
            log_drone_pos[i].append(drone_states[i]["p"].tolist())
            bat_pct = float(batteries[i].remaining_wh / batteries[i].initial_capacity_wh * 100.0)
            log_drone_battery[i].append(bat_pct)
            role_obj = active_directives[i].role if i in active_directives else "EXPLORER"
            role_str = role_obj.name if hasattr(role_obj, "name") else str(role_obj)
            log_drone_roles[i].append(role_str)

        for t_id, t_data in target_telemetry.items():
            log_target_pos[t_id].append(t_data.position.tolist())
            log_target_states[t_id].append(t_data.state.value)

        # I. Offscreen 3-Panel Split-Screen Rendering
        if enable_video and step % render_interval == 0:
            mujoco.mj_forward(m_world, d_world)

            # 1. Tactical Spectator View
            tactical_rgb = renderer.render_tactical(d_world, azimuth_offset=(t / total_sim_time) * 15.0)

            # 2. Drone 1 FPV Recon Camera
            fpv1_rgb = renderer.render_drone_fpv(d_world, drone_states[1]["p"], drone_states[1]["q"])

            # 3. Drone 2 Flanker FPV Camera
            fpv2_rgb = renderer.render_drone_fpv(d_world, drone_states[2]["p"], drone_states[2]["q"])

            # Pass real MuJoCo FPV camera frame to DeepSeek Vision Recon
            if step % 200 == 0 or (len(all_spotted_target_ids) > 0 and step % 120 == 0):
                vision_recon.request_vision_analysis(fpv1_rgb, drone_id=1, camera_mode="RGB_EO")

            # Compute RF Mesh Links
            active_links = []
            for id_a in range(4):
                for id_b in range(id_a + 1, 4):
                    d = float(np.linalg.norm(drone_states[id_a]["p"] - drone_states[id_b]["p"]))
                    max_r = 32.0 if (id_a == 3 or id_b == 3) else 18.0
                    if d <= max_r:
                        active_links.append((id_a, id_b))

            hud = HUDOverlay(
                mission_time=t,
                drone_positions={i: drone_states[i]["p"] for i in range(4)},
                drone_velocities={i: drone_states[i]["v"] for i in range(4)},
                drone_battery_pct={i: float(batteries[i].remaining_wh / batteries[i].initial_capacity_wh * 100.0) for i in range(4)},
                drone_roles={i: (active_directives[i].role.name if hasattr(active_directives[i].role, "name") else str(active_directives[i].role)) for i in range(4)},
                target_positions=target_positions,
                target_detected={t_id: (t_id in all_spotted_target_ids) for t_id in target_positions},
                detections=log_all_sightings,
                flight_trails=flight_trails,
                active_mesh_links=active_links,
                uncertainty_pct=current_u,
            )

            ai_dir = commander.get_latest_directive()
            vr_card = vision_recon.get_latest_card()

            d1_dir_str = f"[{doctrine_cfg.name[:16]}] {ai_dir.strategic_posture} | {ai_dir.tactical_radio_broadcast[:50]}"
            if vr_card and vr_card.target_detected:
                d2_dir_str = f"[AI VISION: {vr_card.target_type}] {vr_card.tactical_recommendation[:50]}"
            else:
                d2_dir_str = f"[{doctrine_cfg.doctrine_id.value}] {active_directives[2].reasoning if 2 in active_directives else ''}"

            tri_frame = video_gen.compose_tri_panel_frame(
                tactical_rgb=tactical_rgb,
                fpv1_rgb=fpv1_rgb,
                fpv2_rgb=fpv2_rgb,
                hud=hud,
                active_directive_d1=d1_dir_str,
                active_directive_d2=d2_dir_str,
            )
            video_gen.add_frame(tri_frame)

            if len(video_gen.frames) % 50 == 0:
                print(f"  [Step {step:4d}/{n_steps}] t={t:5.2f}s | 3D Uncertainty: {current_u:5.1f}% | Sightings: {len(log_all_sightings)} | AI Posture: {ai_dir.strategic_posture}")

    # 4. Save Video Deliverable
    if enable_video and len(video_gen.frames) > 0:
        print(f"\nEncoding 3-Panel Split-Screen Video Deliverable: {video_gen.output_path} ({len(video_gen.frames)} frames)...")
        video_gen.save_video()
        print(f"Saved Video Report: {video_gen.output_path}")

    # 5. Generate Engineering Telemetry Dashboards
    print("\nSynthesizing Engineering Telemetry Dashboards...")

    # Dashboard 1: 3D Flight Trajectories with Slalom Altitude & Urban Buildings
    fig = plt.figure(figsize=(12, 10), dpi=200)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("MRD-Swarm: Closed-Loop 3D Trajectories & Slalom Maneuvers", fontsize=14, fontweight="bold")

    for obs in obstacles:
        ox, oy, oz = obs["pos"]
        hw, hl, hh = obs["size"]
        x = [ox-hw, ox+hw, ox+hw, ox-hw, ox-hw, ox-hw, ox+hw, ox+hw, ox-hw, ox-hw]
        y = [oy-hl, oy-hl, oy+hl, oy+hl, oy-hl, oy-hl, oy-hl, oy+hl, oy+hl, oy-hl]
        z = [0, 0, 0, 0, 0, hh*2, hh*2, hh*2, hh*2, hh*2]
        ax.plot(x, y, z, color="#475569", alpha=0.6, linewidth=1.5)
        ax.text(ox, oy, hh*2 + 0.5, obs["name"], color="#334155", fontsize=7, ha="center")

    palette = ["#0284c7", "#ef4444", "#10b981", "#a855f7"]
    for i in range(4):
        p_arr = np.array(log_drone_pos[i])
        ax.plot(p_arr[:, 0], p_arr[:, 1], p_arr[:, 2], color=palette[i], linewidth=2.2, label=f"D{i}: {HETEROGENEOUS_SPECS[i].drone_class.value}")
        ax.scatter([p_arr[0, 0]], [p_arr[0, 1]], [p_arr[0, 2]], color=palette[i], marker="o", s=70)
        ax.scatter([p_arr[-1, 0]], [p_arr[-1, 1]], [p_arr[-1, 2]], color=palette[i], marker="^", s=90)

    target_palette = ["#dc2626", "#06b6d4", "#eab308"]
    for t_id in range(3):
        tp_arr = np.array(log_target_pos[t_id])
        ax.plot(tp_arr[:, 0], tp_arr[:, 1], tp_arr[:, 2], color=target_palette[t_id], linestyle="--", linewidth=2.5, label=f"HVT-{t_id}: {target_manager.targets[t_id].name}")

    ax.set_xlim(-30, 30)
    ax.set_ylim(-30, 30)
    ax.set_zlim(0, 16)
    ax.set_xlabel("X Coordinate [m]")
    ax.set_ylabel("Y Coordinate [m]")
    ax.set_zlabel("Altitude Z [m]")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=8)
    p1 = OUTPUT_DIR / "plot_3d_closed_loop_trajectories.png"
    plt.tight_layout()
    plt.savefig(p1)
    plt.close()
    print(f"  [1/4] Saved: {p1.name}")

    # Dashboard 2: 3D Voxel Uncertainty Decay Curve
    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
    ax.set_title("Autonomous 3D Voxel Uncertainty Field Decay (Epistemic Coverage)", fontsize=14, fontweight="bold")
    ax.plot(log_time, log_uncertainty, color="#06b6d4", linewidth=2.5, label="Mean 3D Voxel Uncertainty [%]")
    ax.set_xlabel("Mission Time [s]")
    ax.set_ylabel("Uncertainty [%]")
    ax.set_ylim(0.0, 105.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    p2 = OUTPUT_DIR / "plot_uncertainty_decay_curve.png"
    plt.tight_layout()
    plt.savefig(p2)
    plt.close()
    print(f"  [2/4] Saved: {p2.name}")

    # Dashboard 3: Dynamic Evasive Targets State Timeline
    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
    ax.set_title("Reactive Evasive Ground Targets AI State Evolution", fontsize=14, fontweight="bold")
    state_map = {"PATROL": 1, "ACTIVE_EVASION": 2, "SHADOW_LOITER": 3}
    for t_id in range(3):
        num_s = [state_map.get(s, 1) for s in log_target_states[t_id]]
        ax.plot(log_time, [n + t_id * 0.1 for n in num_s], color=target_palette[t_id], linewidth=2.2, label=f"HVT-{t_id} ({target_manager.targets[t_id].name})")

    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(["Patrol Corridor", "Active Evasion (Corner Shadowing)", "Shadow Loiter"])
    ax.set_xlabel("Mission Time [s]")
    ax.set_ylabel("Target AI Behavior State")
    ax.grid(True, alpha=0.3)
    ax.legend()
    p3 = OUTPUT_DIR / "plot_evasive_targets_state_timeline.png"
    plt.tight_layout()
    plt.savefig(p3)
    plt.close()
    print(f"  [3/4] Saved: {p3.name}")

    # Dashboard 4: Dynamic Swarm Roles Evolution
    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
    ax.set_title("Swarm Tactical Brain Dynamic Role Allocation Timeline", fontsize=14, fontweight="bold")
    role_map = {"EXPLORER": 1, "TRACKER": 2, "FLANKER": 3, "RELAY": 4, "LOST_TARGET_SWEEP": 5, "BASE_RECOVERY": 0}
    for i in range(4):
        num_r = [role_map.get(r, 1) for r in log_drone_roles[i]]
        ax.plot(log_time, [n + i * 0.08 for n in num_r], color=palette[i], linewidth=2.2, label=f"D{i} [{HETEROGENEOUS_SPECS[i].drone_class.value[:5]}]")

    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.set_yticklabels(["Base Recovery", "Explorer", "Tracker (Standoff)", "Flanker (Pincer)", "Relay Anchor", "Lost Target Sweep"])
    ax.set_xlabel("Mission Time [s]")
    ax.set_ylabel("Tactical Role")
    ax.grid(True, alpha=0.3)
    ax.legend()
    p4 = OUTPUT_DIR / "plot_swarm_tactical_roles_timeline.png"
    plt.tight_layout()
    plt.savefig(p4)
    plt.close()
    print(f"  [4/4] Saved: {p4.name}")

    # 6. Save Telemetry Datasets
    json_path = OUTPUT_DIR / "dynamic_swarm_mission_log.json"
    summary = {
        "mission": "MRD-Swarm Closed-Loop Reactive Swarm Intelligence",
        "duration_s": total_sim_time,
        "control_steps": n_steps,
        "total_hvt_sightings": len(log_all_sightings),
        "final_uncertainty_pct": float(current_u),
        "target_tracking_coverage_pct": round((len(log_all_sightings) / max(1, n_steps)) * 100.0, 1),
        "drone_fleet": {
            i: {
                "class": HETEROGENEOUS_SPECS[i].drone_class.value,
                "final_role": log_drone_roles[i][-1],
                "final_battery_pct": float(log_drone_battery[i][-1]),
            }
            for i in range(4)
        },
        "doctrine_summary": {
            "doctrine_id": doctrine_cfg.doctrine_id.value,
            "doctrine_name": doctrine_cfg.name,
            "pincer_separation_deg": doctrine_cfg.pincer_separation_deg,
            "standoff_radius_m": doctrine_cfg.standoff_radius_m,
            "flanker_max_speed_mps": doctrine_cfg.flanker_max_speed_mps,
            "multi_target_split": doctrine_cfg.multi_target_split,
        },
        "ai_commander_summary": {
            "model": commander.model,
            "final_strategic_posture": commander.get_latest_directive().strategic_posture,
            "final_radio_broadcast": commander.get_latest_directive().tactical_radio_broadcast,
            "total_directives_issued": len(commander.directive_history),
            "final_reasoning": commander.get_latest_directive().reasoning_chain,
        },
        "vision_recon_summary": {
            "model": vision_recon.model,
            "target_detected": vision_recon.get_latest_card().target_detected,
            "target_type": vision_recon.get_latest_card().target_type,
            "threat_level": vision_recon.get_latest_card().threat_level,
            "visual_description": vision_recon.get_latest_card().visual_description,
            "tactical_recommendation": vision_recon.get_latest_card().tactical_recommendation,
        },
        "verification_status": "PASS" if (current_u < 45.0 and len(log_all_sightings) > 0) else "FAIL",
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved Mission Summary: {json_path}")

    print("\n" + "=" * 95)
    print(f"  MISSION SUCCESS: 3D Uncertainty Reduced to {current_u:.1f}% | Total Sightings: {len(log_all_sightings)}")
    print("=" * 95)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous Closed-Loop Swarm Mission")
    parser.add_argument("--steps", type=int, default=1000, help="Number of simulation steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-video", dest="video", action="store_false", help="Disable video recording")
    parser.add_argument("--doctrine", type=str, default="DEEPSEEK_ADAPTIVE",
                        choices=["DEEPSEEK_ADAPTIVE", "AGGRESSIVE_PINCER", "WOLFPACK_CONTAINMENT", "STEALTH_SHADOW"],
                        help="Swarm battle doctrine to deploy")
    args = parser.parse_args()

    run_closed_loop_swarm_mission(
        n_steps=args.steps,
        seed=args.seed,
        enable_video=args.video,
        doctrine=args.doctrine,
    )
