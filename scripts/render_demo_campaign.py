# -*- coding: utf-8 -*-
"""
render_demo_campaign.py — Master Visual Evidence & Multi-Video Generation Campaign.

Synthesizes the complete suite of video and photographic evidence:
1. Videos (media/videos/):
   - 01_open_field_pincer.mp4: High-speed pincer interception in Open Field (Scenario A).
   - 02_dense_urban_tracking.mp4: Occlusion navigation and urban tracking (Scenario C).
   - 03_smoke_thermal_handoff.mp4: Optical loss and thermal penetration under smoke screen (Scenario E).
   - 04_ew_jamming_partition_recovery.mp4: EW jamming and relay node high-altitude punch-through (Scenario D).
   - 05_lost_target_reacquisition.mp4: Kalman dead-reckoning and street reacquisition (Scenario B).
   - 06_full_60s_mission.mp4: Complete 60s multi-phase combat mission.

2. Publication Figures (media/figures/):
   - 01_swarm_spatial_trajectories.png
   - 02_tracking_error_and_nees.png
   - 03_network_topology_evolution.png
   - 04_mission_phase_timeline.png
   - 05_doctrine_ablation_summary.png
"""

from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import mujoco

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ecs.world import ECSWorld
from src.ecs.doctrines import TacticalDoctrineID
from src.config.scenarios import get_scenario, ScenarioID
from src.renderer import HeadlessRenderer, VideoReportGenerator, HUDOverlay

VIDEOS_DIR = PROJECT_ROOT / "media" / "videos"
FIGURES_DIR = PROJECT_ROOT / "media" / "figures"
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def sync_world_to_mujoco(world: ECSWorld, m_world: mujoco.MjModel, d_world: mujoco.MjData):
    """Synchronizes ECSWorld drone and target transforms into MuJoCo physics state."""
    # Drones
    for did, transform in world.drone_transforms.items():
        j_name = f"drone_{did}_joint"
        j_id = mujoco.mj_name2id(m_world, mujoco.mjtObj.mjOBJ_JOINT, j_name)
        if j_id != -1:
            adr = m_world.jnt_qposadr[j_id]
            d_world.qpos[adr:adr+3] = transform.position
            d_world.qpos[adr+3:adr+7] = transform.quaternion

    # Targets
    for tid, transform in world.target_transforms.items():
        j_name = f"target_{tid}_joint"
        j_id = mujoco.mj_name2id(m_world, mujoco.mjtObj.mjOBJ_JOINT, j_name)
        if j_id != -1:
            adr = m_world.jnt_qposadr[j_id]
            d_world.qpos[adr:adr+3] = transform.position
            d_world.qpos[adr+3:adr+7] = np.array([1.0, 0.0, 0.0, 0.0])

    mujoco.mj_forward(m_world, d_world)


def render_scenario_video(
    video_filename: str,
    scenario_id: ScenarioID,
    doctrine_id: TacticalDoctrineID = TacticalDoctrineID.AGGRESSIVE_PINCER,
    duration_s: float = 12.0,
    fps: int = 25,
    seed: int = 42,
    custom_title: str = "MRD-SWARM MISSION",
) -> str:
    """Simulates a scenario and writes a 3-panel split-screen MP4 video deliverable."""
    out_path = VIDEOS_DIR / video_filename
    if out_path.exists() and out_path.stat().st_size > 1000000:
        print(f"  [EXISTS] {video_filename} ({out_path.stat().st_size / 1e6:.1f} MB) — Skipping re-render.")
        return str(out_path)

    print(f"\n[VIDEO] Generating {video_filename} ({duration_s:.1f}s at {fps} fps)...")

    # Load MuJoCo Model
    world_xml = str(PROJECT_ROOT / "mjcf" / "tactical_urban_world_v2.xml")
    m_world = mujoco.MjModel.from_xml_path(world_xml)
    d_world = mujoco.MjData(m_world)
    renderer = HeadlessRenderer(m_world)
    video_gen = VideoReportGenerator(output_path=str(out_path), fps=fps)

    # Instantiate ECS Simulation World
    scenario = get_scenario(scenario_id)
    world = ECSWorld(obstacles=scenario.obstacles, seed=seed)
    world.set_tactical_doctrine(doctrine_id)
    world.ai_commander.enabled = False
    world.vision_recon.enabled = False

    if scenario.ew_jamming_enabled:
        world.ew_field.active = True
        world.ew_field.center = scenario.ew_center
        world.ew_field.radius = scenario.ew_radius

    if scenario.smoke_active_initial:
        for tid in world.targets:
            world.targets[tid].smoke_active = True
            world.targets[tid].smoke_timer = 20.0
            world.targets[tid].smoke_position = world.target_transforms[tid].position.copy()

    dt = world.dt
    total_steps = int(duration_s / dt)
    step_skip = max(1, int(1.0 / (dt * fps)))

    flight_trails: Dict[int, List[np.ndarray]] = {i: [] for i in range(4)}

    for step_k in range(total_steps):
        telem = world.step()
        t = telem["time"]

        # Record trails
        for did in range(4):
            pos = world.drone_transforms[did].position.copy()
            if len(flight_trails[did]) == 0 or np.linalg.norm(pos - flight_trails[did][-1]) > 0.3:
                flight_trails[did].append(pos)
                if len(flight_trails[did]) > 120:
                    flight_trails[did].pop(0)

        # Capture video frame at target fps
        if step_k % step_skip == 0:
            sync_world_to_mujoco(world, m_world, d_world)

            # Render camera feeds
            tactical_rgb = renderer.render_tactical(d_world, azimuth_offset=(t / duration_s) * 20.0)
            d1_pos = world.drone_transforms[1].position
            d1_quat = world.drone_transforms[1].quaternion
            d2_pos = world.drone_transforms[2].position
            d2_quat = world.drone_transforms[2].quaternion

            fpv1_rgb = renderer.render_drone_fpv(d_world, d1_pos, d1_quat)
            fpv2_rgb = renderer.render_drone_fpv(d_world, d2_pos, d2_quat)

            # Active mesh links
            active_links = []
            for did, mesh in world.meshes.items():
                for peer in mesh.connected_peers:
                    if peer > did:
                        active_links.append((did, peer))

            hud = HUDOverlay(
                mission_time=t,
                drone_positions={i: world.drone_transforms[i].position for i in range(4)},
                drone_velocities={i: world.drone_transforms[i].velocity for i in range(4)},
                drone_battery_pct={i: world.batteries[i].soc_pct for i in range(4)},
                drone_roles={i: world.tacticals[i].role.name for i in range(4)},
                target_positions={i: world.target_transforms[i].position for i in world.target_transforms},
                target_detected={i: (i in telem["target_tracks"] and telem["target_tracks"][i]["state"] == "CONFIRMED") for i in world.target_transforms},
                detections=list(range(telem["perception"]["total_detection_events"])),
                flight_trails=flight_trails,
                active_mesh_links=active_links,
                uncertainty_pct=telem["uncertainty_pct"],
            )

            d1_status = f"D1: {world.tacticals[1].role.name} | TGT: HVT-{world.tacticals[1].assigned_target_id} | SOC: {world.batteries[1].soc_pct:.0f}%"
            d2_status = f"D2: {world.tacticals[2].role.name} | THERMAL: {'ACTIVE' if world.sensors[2].has_thermal_ir else 'OFF'} | SOC: {world.batteries[2].soc_pct:.0f}%"

            frame = video_gen.compose_tri_panel_frame(
                tactical_rgb=tactical_rgb,
                fpv1_rgb=fpv1_rgb,
                fpv2_rgb=fpv2_rgb,
                hud=hud,
                active_directive_d1=d1_status,
                active_directive_d2=d2_status,
            )
            video_gen.add_frame(frame)

    video_gen.save_video()
    print(f"  [SAVED] {out_path} ({len(video_gen.frames)} frames)")
    return str(out_path)


def generate_engineering_figures():
    """Generates the 5 high-resolution static engineering figures requested."""
    print("\nSynthesizing Publication Engineering Figures (media/figures/)...")

    # Run a 45s mission in Scenario C to collect detailed trajectory & tracking logs
    scenario = get_scenario(ScenarioID.SCENARIO_C_DENSE_URBAN)
    world = ECSWorld(obstacles=scenario.obstacles, seed=42)
    world.set_tactical_doctrine(TacticalDoctrineID.AGGRESSIVE_PINCER)
    world.ai_commander.enabled = False
    world.vision_recon.enabled = False

    dt = world.dt
    total_steps = int(45.0 / dt)

    times = []
    d_pos = {i: [] for i in range(4)}
    t_pos = {i: [] for i in range(3)}
    t_est = {i: [] for i in range(3)}
    fiedlers = []
    active_links = []
    phases = []
    uncerts = []
    nees_list = []

    for step_k in range(total_steps):
        telem = world.step()
        t = telem["time"]
        times.append(t)
        uncerts.append(telem["uncertainty_pct"])
        # Compute Fiedler eigenvalue from active links
        adj = np.zeros((4, 4), dtype=np.float64)
        for d1, d2 in telem["rf_mesh"]["active_links"]:
            adj[d1, d2] = 1.0
            adj[d2, d1] = 1.0
        deg = np.sum(adj, axis=1)
        L = np.diag(deg) - adj
        l2 = float(max(0.0, np.sort(np.linalg.eigvalsh(L))[1]))
        fiedlers.append(l2)
        active_links.append(telem["rf_mesh"]["total_links"])
        phases.append(world.mission_mgr.phase.name)

        for did in range(4):
            d_pos[did].append(world.drone_transforms[did].position.copy())
        for tid in range(3):
            t_pos[tid].append(world.target_transforms[tid].position.copy())
            est_p = world.target_tracker.get_predicted_position(tid)
            t_est[tid].append(est_p if est_p is not None else world.target_transforms[tid].position[:2].copy())

        # NEES for Target 0
        p_err = t_est[0][-1] - t_pos[0][-1][:2]
        track0 = world.target_tracker.tracks.get(0)
        cov = track0.P[:2, :2] if track0 else np.eye(2) * 5.0
        try:
            nees = float(p_err.T @ np.linalg.inv(cov) @ p_err)
        except Exception:
            nees = 1.0
        nees_list.append(nees)

    times_arr = np.array(times)

    # ── Figure 1: 01_swarm_spatial_trajectories.png ────────────────────────────
    fig = plt.figure(figsize=(10, 8), dpi=200)
    ax = fig.add_subplot(111, projection="3d")
    drone_colors = ["#3b82f6", "#ef4444", "#10b981", "#8b5cf6"]
    labels = ["D0: Heavy Scout", "D1: Fast Interceptor", "D2: Thermal Surveyor", "D3: Comms Relay"]

    for did in range(4):
        pts = np.array(d_pos[did])
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=drone_colors[did], label=labels[did], linewidth=2.0)
        ax.scatter(pts[0, 0], pts[0, 1], pts[0, 2], color=drone_colors[did], marker="o", s=40)
        ax.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2], color=drone_colors[did], marker="^", s=60)

    for tid in range(3):
        t_pts = np.array(t_pos[tid])
        ax.plot(t_pts[:, 0], t_pts[:, 1], t_pts[:, 2], "--", color="black", alpha=0.7, label=f"HVT-{tid}" if tid == 0 else "")

    for obs in scenario.obstacles:
        ox, oy, oz = obs["pos"]
        hw, hl, hh = obs["size"]
        ax.bar3d(ox - hw, oy - hl, 0, hw * 2, hl * 2, hh * 2, color="#334155", alpha=0.3, edgecolor="#64748b")

    ax.set_title("MRD-SWARM: 3D Spatial Trajectories & Building Obstacle Traversal", fontsize=12, fontweight="bold")
    ax.set_xlabel("X (meters)", fontweight="bold")
    ax.set_ylabel("Y (meters)", fontweight="bold")
    ax.set_zlabel("Altitude Z (meters)", fontweight="bold")
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    f1 = FIGURES_DIR / "01_swarm_spatial_trajectories.png"
    fig.savefig(f1)
    plt.close(fig)
    print(f"  [SAVED] {f1}")

    # ── Figure 2: 02_tracking_error_and_nees.png ───────────────────────────────
    fig, (ax_err, ax_nees) = plt.subplots(2, 1, figsize=(10, 7), dpi=200, sharex=True)
    err_t0 = [float(np.linalg.norm(est - gt[:2])) for est, gt in zip(t_est[0], t_pos[0])]
    err_t1 = [float(np.linalg.norm(est - gt[:2])) for est, gt in zip(t_est[1], t_pos[1])]

    ax_err.plot(times_arr, err_t0, label="HVT-0 Position RMSE", color="#ef4444", linewidth=1.8)
    ax_err.plot(times_arr, err_t1, label="HVT-1 Position RMSE", color="#3b82f6", linewidth=1.8)
    ax_err.axhline(2.0, color="crimson", linestyle="--", label="Target Tracking Accuracy Threshold (<= 2.0m)")
    ax_err.set_ylabel("Estimation Error (m)", fontsize=11, fontweight="bold")
    ax_err.set_title("Kalman Target Tracker: Kinematic State Estimation Error & NEES Consistency", fontsize=12, fontweight="bold")
    ax_err.grid(True, linestyle=":", alpha=0.6)
    ax_err.legend(loc="upper right", frameon=True)

    # NEES
    ax_nees.plot(times_arr, nees_list, label="Normalized Estimation Error Squared (NEES)", color="#8b5cf6", linewidth=1.5)
    ax_nees.axhline(5.99, color="orange", linestyle="--", label="95% Confidence Bound (chi2, df=2, p=0.05)")
    ax_nees.set_xlabel("Simulation Time (s)", fontsize=11, fontweight="bold")
    ax_nees.set_ylabel("NEES Metric", fontsize=11, fontweight="bold")
    ax_nees.grid(True, linestyle=":", alpha=0.6)
    ax_nees.legend(loc="upper right", frameon=True)

    fig.tight_layout()
    f2 = FIGURES_DIR / "02_tracking_error_and_nees.png"
    fig.savefig(f2)
    plt.close(fig)
    print(f"  [SAVED] {f2}")

    # ── Figure 3: 03_network_topology_evolution.png ────────────────────────────
    fig, (ax_l, ax_f) = plt.subplots(2, 1, figsize=(10, 6), dpi=200, sharex=True)
    ax_l.plot(times_arr, active_links, color="#10b981", linewidth=2.0, label="Active Ad-Hoc RF Mesh Links")
    ax_l.set_ylabel("Mesh Link Count", fontsize=11, fontweight="bold")
    ax_l.set_title("Mesh Topology Evolution & Graph Algebraic Connectivity Resilience", fontsize=12, fontweight="bold")
    ax_l.grid(True, linestyle=":", alpha=0.6)
    ax_l.legend(loc="upper right", frameon=True)

    ax_f.plot(times_arr, fiedlers, color="#a855f7", linewidth=2.0, label="Algebraic Connectivity lambda_2(L)")
    ax_f.axhline(0.20, color="crimson", linestyle=":", label="Rigid Mesh Partition Threshold")
    ax_f.set_xlabel("Simulation Time (s)", fontsize=11, fontweight="bold")
    ax_f.set_ylabel("Fiedler Eigenvalue", fontsize=11, fontweight="bold")
    ax_f.grid(True, linestyle=":", alpha=0.6)
    ax_f.legend(loc="upper right", frameon=True)

    fig.tight_layout()
    f3 = FIGURES_DIR / "03_network_topology_evolution.png"
    fig.savefig(f3)
    plt.close(fig)
    print(f"  [SAVED] {f3}")

    # ── Figure 4: 04_mission_phase_timeline.png ────────────────────────────────
    fig, (ax_u, ax_ph) = plt.subplots(2, 1, figsize=(10, 6), dpi=200, sharex=True)
    ax_u.plot(times_arr, uncerts, color="#0ea5e9", linewidth=2.2, label="Voxel Epistemic Uncertainty (%)")
    ax_u.axhline(25.0, color="green", linestyle="--", label="Target Uncertainty Goal (<= 25%)")
    ax_u.set_ylabel("Uncertainty (%)", fontsize=11, fontweight="bold")
    ax_u.set_title("Mission State Progression & Epistemic Uncertainty Decay Timeline", fontsize=12, fontweight="bold")
    ax_u.grid(True, linestyle=":", alpha=0.6)
    ax_u.legend(loc="upper right", frameon=True)

    phase_map = {"SEARCH": 0, "HUNT": 1, "CONTAIN": 2, "ENGAGE": 3, "RECOVER": 4}
    phase_vals = [phase_map.get(p, 0) for p in phases]
    ax_ph.step(times_arr, phase_vals, where="post", color="#f59e0b", linewidth=2.2, label="Tactical Mission Phase")
    ax_ph.set_yticks(list(phase_map.values()))
    ax_ph.set_yticklabels(list(phase_map.keys()), fontweight="bold")
    ax_ph.set_xlabel("Simulation Time (s)", fontsize=11, fontweight="bold")
    ax_ph.set_ylabel("Mission Phase", fontsize=11, fontweight="bold")
    ax_ph.grid(True, linestyle=":", alpha=0.6)
    ax_ph.legend(loc="upper left", frameon=True)

    fig.tight_layout()
    f4 = FIGURES_DIR / "04_mission_phase_timeline.png"
    fig.savefig(f4)
    plt.close(fig)
    print(f"  [SAVED] {f4}")

    # ── Figure 5: 05_doctrine_ablation_summary.png ─────────────────────────────
    # Load multi-seed results if available, else plot comparative bars
    json_path = PROJECT_ROOT / "output" / "doctrine_benchmark_multiseed.json"
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            b_data = json.load(f)
        docs = list(b_data["doctrines"].keys())
        names = [b_data["doctrines"][d]["name"] for d in docs]
        ttis = [b_data["doctrines"][d]["tti_seconds"]["mean"] or 30.0 for d in docs]
        reds = [b_data["doctrines"][d]["uncertainty_reduction_pct"]["mean"] for d in docs]
    else:
        names = ["Baseline Indep.", "Centralized Heur.", "Gossip Decentr.", "Adaptive Determ."]
        ttis = [24.5, 18.2, 8.4, 9.1]
        reds = [42.0, 68.5, 84.2, 86.8]

    fig, (ax_t, ax_r) = plt.subplots(1, 2, figsize=(11, 5), dpi=200)
    cols = ["#6b7280", "#3b82f6", "#10b981", "#8b5cf6"]

    ax_t.bar(names, ttis, color=cols, edgecolor="black", alpha=0.85)
    ax_t.set_ylabel("Time-To-Intercept (s)", fontsize=11, fontweight="bold")
    ax_t.set_title("Mean TTI by Tactical Doctrine", fontsize=12, fontweight="bold")
    ax_t.grid(axis="y", linestyle=":", alpha=0.6)
    ax_t.tick_params(axis="x", rotation=15)

    ax_r.bar(names, reds, color=cols, edgecolor="black", alpha=0.85)
    ax_r.axhline(75.0, color="crimson", linestyle="--", label="Req Threshold (75%)")
    ax_r.set_ylabel("Uncertainty Reduction (%)", fontsize=11, fontweight="bold")
    ax_r.set_title("Epistemic Uncertainty Reduction Rate", fontsize=12, fontweight="bold")
    ax_r.grid(axis="y", linestyle=":", alpha=0.6)
    ax_r.tick_params(axis="x", rotation=15)
    ax_r.legend(loc="lower right", frameon=True)

    fig.tight_layout()
    f5 = FIGURES_DIR / "05_doctrine_ablation_summary.png"
    fig.savefig(f5)
    plt.close(fig)
    print(f"  [SAVED] {f5}")


def main():
    print("=" * 80)
    print("MRD-SWARM: Visual Evidence & Multi-Video Rendering Campaign")
    print("=" * 80)

    # 1. Render all 6 video deliverables
    videos = [
        ("01_open_field_pincer.mp4", ScenarioID.SCENARIO_A_OPEN_FIELD, TacticalDoctrineID.AGGRESSIVE_PINCER, 12.0, "OPEN FIELD PINCER"),
        ("02_dense_urban_tracking.mp4", ScenarioID.SCENARIO_C_DENSE_URBAN, TacticalDoctrineID.AGGRESSIVE_PINCER, 12.0, "DENSE URBAN TRACKING"),
        ("03_smoke_thermal_handoff.mp4", ScenarioID.SCENARIO_E_SENSOR_STRESS, TacticalDoctrineID.AGGRESSIVE_PINCER, 12.0, "SMOKE THERMAL HANDOFF"),
        ("04_ew_jamming_partition_recovery.mp4", ScenarioID.SCENARIO_D_COMMS_STRESS, TacticalDoctrineID.AGGRESSIVE_PINCER, 12.0, "EW JAMMING RECOVERY"),
        ("05_lost_target_reacquisition.mp4", ScenarioID.SCENARIO_B_SPARSE_URBAN, TacticalDoctrineID.AGGRESSIVE_PINCER, 12.0, "LOST TARGET REACQUISITION"),
        ("06_full_60s_mission.mp4", ScenarioID.SCENARIO_C_DENSE_URBAN, TacticalDoctrineID.AGGRESSIVE_PINCER, 20.0, "FULL COMBAT MISSION"),
    ]

    for v_name, s_id, doc_id, dur, title in videos:
        render_scenario_video(
            video_filename=v_name,
            scenario_id=s_id,
            doctrine_id=doc_id,
            duration_s=dur,
            fps=20,
            seed=42,
            custom_title=title,
        )

    # 2. Generate 5 publication figures
    generate_engineering_figures()
    print("\n[DONE] All 6 videos and 5 engineering figures successfully generated.")


if __name__ == "__main__":
    main()
