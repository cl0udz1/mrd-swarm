#!/usr/bin/env python3
"""
recon_swarm_sim.py — End-to-End MRD-Swarm Mission Simulation

Demonstrates the full lifecycle:
    1. Spawns a 4-drone recon swarm and 2 ground targets in MuJoCo
    2. AI Agent Controller assigns mission objectives:
       - Drone 0 & 1: Perimeter Sweep
       - Drone 2 & 3: Target Intercept and Orbit
    3. Runs headless simulation for 600 control timesteps
    4. Produces:
       - Structured JSON mission log
       - Encoded mission_recon_report.mp4 with FPV/Tactical split-screen + HUD

Usage:
    python recon_swarm_sim.py [--output-dir ./output] [--steps 600] [--seed 42]
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path

# Don't force osmesa — let mujoco pick available backend
# os.environ.setdefault("MUJOCO_GL", "osmesa")

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.physics import (
    DRONE_MASS, GRAVITY, quat_to_rotation_matrix,
    MAX_THRUST_PER_MOTOR,
)
from src.controller import CascadedQuadrotorController, TrajectoryGenerator
from src.sensors import SensorSuite, BatteryState
from src.swarm import SwarmEnvironment, DroneState, GroundTarget
from src.agent_interface import ReconAgentTools, get_all_tool_schemas
from src.renderer import HeadlessRenderer, VideoReportGenerator, CameraConfig, HUDOverlay, RENDERING_AVAILABLE

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("[WARN] OpenCV not available. HUD overlays will be skipped.")

if not RENDERING_AVAILABLE:
    print("[WARN] MuJoCo rendering not available (no OSMesa/EGL). Video will use blank frames with HUD only.")


def generate_perimeter_sweep_waypoints(
    center: tuple[float, float] = (0.0, 0.0),
    half_size: float = 8.0,
    altitude: float = 3.0,
    n_laps: int = 2,
) -> list[np.ndarray]:
    """Generate waypoints for a rectangular perimeter sweep."""
    cx, cy = center
    corners = [
        np.array([cx + half_size, cy + half_size, altitude]),
        np.array([cx + half_size, cy - half_size, altitude]),
        np.array([cx - half_size, cy - half_size, altitude]),
        np.array([cx - half_size, cy + half_size, altitude]),
    ]
    waypoints = []
    for _ in range(n_laps):
        waypoints.extend(corners)
    # Close the loop
    waypoints.append(waypoints[0].copy())
    return waypoints


def generate_orbit_waypoints(
    center: tuple[float, float],
    radius: float = 3.0,
    altitude: float = 2.5,
    n_points: int = 24,
) -> list[np.ndarray]:
    """Generate circular orbit waypoints."""
    cx, cy = center
    waypoints = []
    for k in range(n_points + 1):
        angle = 2 * np.pi * k / n_points
        x = cx + radius * np.cos(angle)
        y = cy + radius * np.sin(angle)
        waypoints.append(np.array([x, y, altitude]))
    return waypoints


def build_hud_overlay(env: SwarmEnvironment, step: int) -> HUDOverlay:
    """Construct HUD overlay data from current environment state."""
    hud = HUDOverlay()
    hud.mission_time = env.sim_time
    hud.total_drones = env.n_drones
    hud.active_drones = sum(1 for d in env.drones.values() if d.is_active)

    for i, drone in env.drones.items():
        hud.drone_positions[i] = drone.position.copy()
        hud.drone_velocities[i] = drone.velocity.copy()
        hud.drone_battery_pct[i] = drone.battery.percentage
        hud.drone_headings[i] = drone.heading

    for t_id, target in env.targets.items():
        hud.target_positions[t_id] = target.position.copy()
        hud.target_detected[t_id] = target.is_detected

    hud.detections = env.detection_log[-10:]  # last 10 detections

    # Build flight trails from telemetry
    trails: dict[int, list] = {}
    for entry in env.telemetry_log:
        did = entry["drone_id"]
        if did not in trails:
            trails[did] = []
        trails[did].append(np.array(entry["position"]))
    hud.flight_trails = trails

    return hud


def run_mission(args: argparse.Namespace) -> None:
    """Execute the full reconnaissance mission."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Locate MJCF model
    model_path = Path(__file__).parent / "mjcf" / "recon_quadrotor.xml"
    if not model_path.exists():
        print(f"[ERROR] MJCF model not found at {model_path}")
        sys.exit(1)

    print("=" * 70)
    print("  MRD-SWARM: Autonomous Reconnaissance Quadrotor Mission")
    print("=" * 70)
    print(f"  Model:     {model_path}")
    print(f"  Drones:    4")
    print(f"  Targets:   2")
    print(f"  Steps:     {args.steps}")
    print(f"  Output:    {output_dir}")
    print("=" * 70)

    # ── Initialize Environment ────────────────────────────────────────────────
    print("\n[1/6] Initializing swarm environment...")
    env = SwarmEnvironment(
        model_path=str(model_path),
        n_drones=4,
        n_targets=2,
        scene_size=20.0,
        dt=0.001,
        control_dt=0.01,
        seed=args.seed,
    )

    # ── Initialize Agent Interface ────────────────────────────────────────────
    print("[2/6] Initializing AI agent interface...")
    agent = ReconAgentTools(env)

    # Print tool schemas (for LLM function calling registration)
    schemas = get_all_tool_schemas()
    print(f"  Registered {len(schemas)} agent tools:")
    for s in schemas:
        print(f"    - {s['name']}")

    # ── Assign Mission Objectives ─────────────────────────────────────────────
    print("[3/6] Assigning mission objectives...")

    # Get target positions for orbit assignment
    target_0_pos = env.targets[0].position
    target_1_pos = env.targets[1].position

    # Drone 0 & 1: Perimeter Sweep
    print("  Drone 0 & 1: Perimeter Sweep (20m x 20m area)")
    sweep_result = agent.call_tool("recon_area_search", {
        "drone_ids": [0, 1],
        "bounding_box": {"x_min": -10, "y_min": -10, "x_max": 10, "y_max": 10},
        "pattern": "lawnmower",
        "altitude": 3.0,
        "speed": 2.0,
    })
    print(f"    -> Sweep assigned: {sweep_result['status']}")

    # Drone 2: Orbit Target 0
    print(f"  Drone 2: Intercept and orbit Target 0 at {target_0_pos[:2]}")
    orbit_result_2 = agent.call_tool("recon_orbit_point", {
        "drone_id": 2,
        "center_x": float(target_0_pos[0]),
        "center_y": float(target_0_pos[1]),
        "radius": 3.0,
        "speed": 1.5,
        "altitude": 2.5,
    })
    print(f"    -> Orbit assigned: {orbit_result_2['status']}")

    # Drone 3: Orbit Target 1
    print(f"  Drone 3: Intercept and orbit Target 1 at {target_1_pos[:2]}")
    orbit_result_3 = agent.call_tool("recon_orbit_point", {
        "drone_id": 3,
        "center_x": float(target_1_pos[0]),
        "center_y": float(target_1_pos[1]),
        "radius": 3.0,
        "speed": 1.5,
        "altitude": 2.5,
    })
    print(f"    -> Orbit assigned: {orbit_result_3['status']}")

    # ── Initialize Renderer ───────────────────────────────────────────────────
    print("[4/6] Initializing headless renderer...")
    renderer = HeadlessRenderer(
        model=env.model_template,
        tactical_cam=CameraConfig(
            name="tactical",
            width=1280,
            height=720,
            lookat=np.array([0.0, 0.0, 1.0]),
            distance=30.0,
            azimuth=45.0,
            elevation=-55.0,
        ),
        fpv_width=640,
        fpv_height=480,
    )

    video_gen = VideoReportGenerator(
        output_path=str(output_dir / "mission_recon_report.mp4"),
        fps=30,
    )

    # ── Run Simulation ────────────────────────────────────────────────────────
    print(f"[5/6] Running simulation for {args.steps} steps...")
    start_time = time.time()

    frame_interval = max(1, args.steps // 150)  # capture ~150 frames for video
    all_frames = []

    for step in range(args.steps):
        # Step the simulation
        telemetry = env.step()

        # Periodic telemetry output
        if step % 100 == 0:
            summary = {
                "step": step,
                "time": env.sim_time,
                "drones": {},
            }
            for i, drone in env.drones.items():
                summary["drones"][i] = {
                    "pos": [round(float(x), 2) for x in drone.position],
                    "bat": round(drone.battery.percentage, 1),
                    "targets": drone.detected_targets,
                }
            print(f"  Step {step:4d}/{args.steps} | "
                  f"t={env.sim_time:.2f}s | "
                  f"Detections: {len(env.detection_log)}")

        # Capture frames for video
        if step % frame_interval == 0:
            hud = build_hud_overlay(env, step)

            # Tactical view (use drone 0's data for scene rendering)
            tactical_frame = renderer.render_tactical_view(env.datas[0], hud)

            # FPV from drone 2 (orbiting target 0)
            drone_2 = env.drones[2]
            fpv_frame = renderer.render_fpv_view(
                env.datas[2],
                drone_2.position,
                drone_2.quaternion,
                hud,
            )

            # Compose split-screen
            composed = video_gen.compose_split_screen(tactical_frame, fpv_frame)
            all_frames.append(composed)

    elapsed = time.time() - start_time
    print(f"  Simulation complete in {elapsed:.1f}s ({args.steps/elapsed:.0f} steps/s)")

    # ── Generate Outputs ──────────────────────────────────────────────────────
    print("[6/6] Generating mission outputs...")

    # Save video
    video_path = video_gen.save_video(all_frames)
    print(f"  Video saved: {video_path}")

    # Save mission log
    mission_summary = env.get_mission_summary()
    mission_log = {
        "mission": "MRD-Swarm Reconnaissance",
        "config": {
            "n_drones": env.n_drones,
            "n_targets": env.n_targets,
            "total_steps": args.steps,
            "sim_time_s": env.sim_time,
            "dt": env.control_dt,
        },
        "summary": mission_summary,
        "telemetry_sample": env.telemetry_log[:50],  # first 50 entries
        "detections": env.get_all_detections(),
        "agent_tool_schemas": schemas,
    }

    log_path = output_dir / "mission_log.json"
    with open(log_path, "w") as f:
        json.dump(mission_log, f, indent=2, default=str)
    print(f"  Mission log saved: {log_path}")

    # Print final summary
    print("\n" + "=" * 70)
    print("  MISSION COMPLETE")
    print("=" * 70)
    print(f"  Simulation time:  {mission_summary['total_sim_time']:.1f}s")
    print(f"  Total steps:      {mission_summary['total_steps']}")
    print(f"  Targets detected: {mission_summary['detected_targets']}/{mission_summary['total_targets']}")
    print(f"  Detection rate:   {mission_summary['detection_rate']:.0%}")
    print(f"  Total detections: {mission_summary['total_detections']}")
    print()
    for i, info in mission_summary["per_drone"].items():
        print(f"  Drone {i}: pos=[{', '.join(f'{x:.1f}' for x in info['final_position'])}] "
              f"bat={info['battery_remaining_pct']:.1f}% "
              f"targets={info['targets_detected']} "
              f"energy={info['energy_consumed_wh']:.3f}Wh")
    print("=" * 70)

    # Cleanup
    renderer.close()
    video_gen.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MRD-Swarm: Autonomous Reconnaissance Quadrotor Mission Simulation"
    )
    parser.add_argument("--output-dir", type=str, default="./output",
                       help="Output directory for mission artifacts")
    parser.add_argument("--steps", type=int, default=600,
                       help="Number of simulation steps")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    args = parser.parse_args()

    run_mission(args)


if __name__ == "__main__":
    main()
