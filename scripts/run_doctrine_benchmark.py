# -*- coding: utf-8 -*-
"""
run_doctrine_benchmark.py — Multi-Doctrine Swarm Tactical Benchmark Suite

Empirically compares 3 distinct swarm battle doctrines under identical initial conditions:
1. AGGRESSIVE_PINCER: Sprint corridor cutoff, 160° separation, max thrust.
2. WOLFPACK_CONTAINMENT: 120° symmetric enclosure, multi-target split, high resilience.
3. DEEPSEEK_ADAPTIVE: Autonomous cognitive switching driven by DeepSeek LLM & Vision.
"""

from __future__ import annotations
import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_DIR = Path("c:/cheetah/mrd-swarm")
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(PROJECT_DIR))

from src.ecs.world import ECSWorld
from src.ecs.doctrines import TacticalDoctrineID, get_doctrine_config

OBSTACLES = [
    {"name": "Skyscraper Alpha", "pos": [0.0, 0.0, 7.0], "size": [4.0, 4.0, 7.0], "height": 14.0},
    {"name": "Complex Bravo", "pos": [-14.0, 12.0, 3.0], "size": [6.0, 4.0, 3.0], "height": 6.0},
    {"name": "Silo Charlie", "pos": [-16.0, -14.0, 4.0], "size": [2.5, 2.5, 4.0], "height": 8.0},
    {"name": "Depot Delta", "pos": [15.0, -15.0, 2.5], "size": [7.0, 5.0, 2.5], "height": 5.0},
    {"name": "Substation Echo", "pos": [14.0, 14.0, 2.0], "size": [4.5, 4.5, 2.0], "height": 4.0},
    {"name": "Radar Pylon Foxtrot", "pos": [22.0, 0.0, 5.0], "size": [1.5, 1.5, 5.0], "height": 10.0},
    {"name": "Security Tower Golf", "pos": [-22.0, 0.0, 6.0], "size": [1.5, 1.5, 6.0], "height": 12.0},
    {"name": "Skybridge Hotel", "pos": [0.0, -18.0, 4.5], "size": [10.0, 2.0, 1.0], "height": 7.0},
]


def evaluate_doctrine(doctrine_id: TacticalDoctrineID, n_steps: int = 1200, seed: int = 42) -> Dict[str, Any]:
    print(f"\n{'='*75}")
    print(f"  BENCHMARK TRIAL: {doctrine_id.value}")
    print(f"{'='*75}")

    world = ECSWorld(obstacles=OBSTACLES, seed=seed)
    world.set_tactical_doctrine(doctrine_id)

    # Disable expensive remote LLM calls during fast comparative benchmark loops if needed
    if doctrine_id != TacticalDoctrineID.DEEPSEEK_ADAPTIVE:
        world.ai_commander.enabled = False

    tti = None
    enclosure_angles = []
    velocities = []
    sightings_count = 0
    uncertainty_timeline = []
    energy_used = {i: 0.0 for i in range(4)}
    initial_batteries = {i: world.batteries[i].remaining_wh for i in range(4)}

    for step in range(n_steps):
        telem = world.step()
        t = world.sim_time
        u = float(world.uncertainty_grid.get_mean_uncertainty())
        uncertainty_timeline.append(u)

        # Track sightings
        detected = telem.get("perception", {}).get("detected_targets", [])
        if detected:
            sightings_count += len(detected)
            if tti is None and len(detected) >= 2:
                tti = t  # First multi-target/dual-drone lock

        # Track enclosure angles
        for did in range(3):
            tac = telem["drones"][did]
            if tac.get("role") == "FLANKER":
                angle = tac.get("formation_angle_deg", 0.0)
                if angle > 0.0:
                    enclosure_angles.append(angle)

        # Track combat velocities
        for did in range(3):
            v = np.linalg.norm(world.drone_transforms[did].velocity)
            velocities.append(float(v))

        if step % 200 == 0:
            print(f"  [Step {step:4d}/{n_steps}] t={t:5.2f}s | Uncertainty: {u:5.1f}% | Sightings: {sightings_count}")

    # Compute final metrics
    final_batteries = {i: world.batteries[i].remaining_wh for i in range(4)}
    total_energy_wh = sum(initial_batteries[i] - final_batteries[i] for i in range(4))

    mean_enclosure = float(np.mean(enclosure_angles)) if enclosure_angles else 0.0
    mean_velocity = float(np.mean(velocities)) if velocities else 0.0
    tti_val = tti if tti is not None else float(n_steps * 0.01)

    res = {
        "doctrine": doctrine_id.value,
        "name": get_doctrine_config(doctrine_id).name,
        "tti_seconds": round(tti_val, 2),
        "mean_enclosure_deg": round(mean_enclosure, 1),
        "mean_velocity_mps": round(mean_velocity, 2),
        "total_sightings": sightings_count,
        "final_uncertainty_pct": round(uncertainty_timeline[-1], 2),
        "energy_consumed_wh": round(total_energy_wh, 3),
        "uncertainty_timeline": uncertainty_timeline,
        "enclosure_angles": enclosure_angles,
    }
    print(f"\n  [RESULTS] TTI: {res['tti_seconds']}s | Enclosure: {res['mean_enclosure_deg']}° | Velocity: {res['mean_velocity_mps']}m/s | Energy: {res['energy_consumed_wh']}Wh")
    return res


def run_benchmark():
    print("==========================================================================")
    print("  MRD-SWARM: COMPARATIVE TACTICAL DOCTRINE BENCHMARK")
    print("==========================================================================")

    doctrines_to_test = [
        TacticalDoctrineID.AGGRESSIVE_PINCER,
        TacticalDoctrineID.WOLFPACK_CONTAINMENT,
        TacticalDoctrineID.DEEPSEEK_ADAPTIVE,
    ]

    benchmark_results = {}
    for doc in doctrines_to_test:
        res = evaluate_doctrine(doc, n_steps=1200, seed=42)
        benchmark_results[doc.value] = res

    # ── Synthesize Comparative Evaluation Plots ────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("#0b0f19")
    for ax in axes.flat:
        ax.set_facecolor("#0f172a")
        ax.tick_params(colors="#94a3b8")
        ax.grid(True, color="#334155", alpha=0.3)
        for spine in ax.spines.values():
            spine.set_color("#334155")

    colors = {
        TacticalDoctrineID.AGGRESSIVE_PINCER.value: "#f43f5e",
        TacticalDoctrineID.WOLFPACK_CONTAINMENT.value: "#38bdf8",
        TacticalDoctrineID.DEEPSEEK_ADAPTIVE.value: "#a855f7",
    }

    names = [r["name"] for r in benchmark_results.values()]
    d_keys = list(benchmark_results.keys())

    # 1. Bar Chart: Time-to-Intercept (TTI)
    ax1 = axes[0, 0]
    ttis = [benchmark_results[k]["tti_seconds"] for k in d_keys]
    bars1 = ax1.bar(names, ttis, color=[colors[k] for k in d_keys], alpha=0.85, edgecolor="#ffffff", linewidth=0.8)
    ax1.set_title("Time-to-Intercept (TTI) [Lower is Faster]", color="#f8fafc", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Seconds (s)", color="#94a3b8")
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f"{yval:.1f}s", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=9)

    # 2. Bar Chart: Mean Enclosure Angle (°)
    ax2 = axes[0, 1]
    angles = [benchmark_results[k]["mean_enclosure_deg"] for k in d_keys]
    bars2 = ax2.bar(names, angles, color=[colors[k] for k in d_keys], alpha=0.85, edgecolor="#ffffff", linewidth=0.8)
    ax2.set_title("Pincer Angular Enclosure Quality [Target: 120°-160°]", color="#f8fafc", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Mean Separation (°)", color="#94a3b8")
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 2.0, f"{yval:.0f}°", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=9)

    # 3. Uncertainty Decay Curves
    ax3 = axes[1, 0]
    for k in d_keys:
        t_arr = np.linspace(0, 12.0, len(benchmark_results[k]["uncertainty_timeline"]))
        ax3.plot(t_arr, benchmark_results[k]["uncertainty_timeline"], label=k, color=colors[k], linewidth=2.0)
    ax3.set_title("3D Uncertainty Decay Profile Over Mission", color="#f8fafc", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Time (s)", color="#94a3b8")
    ax3.set_ylabel("Uncertainty (%)", color="#94a3b8")
    ax3.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#f8fafc")

    # 4. Velocity vs Energy Consumption
    ax4 = axes[1, 1]
    vels = [benchmark_results[k]["mean_velocity_mps"] for k in d_keys]
    energies = [benchmark_results[k]["energy_consumed_wh"] for k in d_keys]
    for i, k in enumerate(d_keys):
        ax4.scatter(vels[i], energies[i], color=colors[k], s=250, label=k, edgecolor="#ffffff", linewidth=1.5, zorder=5)
        ax4.annotate(f"{names[i]}\n({vels[i]}m/s, {energies[i]}Wh)", (vels[i]+0.15, energies[i]), color="#f8fafc", fontsize=8)
    ax4.set_title("Tactical Sprint Velocity vs Energy Expenditure", color="#f8fafc", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Mean Velocity (m/s)", color="#94a3b8")
    ax4.set_ylabel("Energy Consumed (Wh)", color="#94a3b8")
    ax4.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#f8fafc")

    plot_path = OUTPUT_DIR / "plot_tactical_doctrines_comparison.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=180)
    plt.close()
    print(f"\nSaved Comparative Telemetry Plot: {plot_path}")

    # Save JSON summary
    summary_path = OUTPUT_DIR / "doctrine_benchmark_summary.json"
    clean_summary = {
        k: {
            "name": v["name"],
            "tti_seconds": v["tti_seconds"],
            "mean_enclosure_deg": v["mean_enclosure_deg"],
            "mean_velocity_mps": v["mean_velocity_mps"],
            "total_sightings": v["total_sightings"],
            "energy_consumed_wh": v["energy_consumed_wh"],
        }
        for k, v in benchmark_results.items()
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(clean_summary, f, indent=2)
    print(f"Saved Benchmark Summary JSON: {summary_path}")


if __name__ == "__main__":
    run_benchmark()
