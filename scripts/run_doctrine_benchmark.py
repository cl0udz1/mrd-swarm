# -*- coding: utf-8 -*-
"""
run_doctrine_benchmark.py — Multi-Seed Tactical Doctrine Benchmark & Ablation Campaign

Empirically benchmarks swarm battle doctrines across randomized initial conditions & seeds:
1. AGGRESSIVE_PINCER: Rapid corridor interception, 160° separation, maximum sprint thrust.
2. WOLFPACK_CONTAINMENT: 120° symmetric enclosure, multi-target split capability.
3. STEALTH_SHADOW: Low-speed observation, boundary-preserving surveillance.
4. DEEPSEEK_ADAPTIVE: Autonomous cognitive switching via DeepSeek AI Commander & Vision Recon.

Evaluated strictly according to docs/METRICS_SPEC.md:
- Formal continuous-window TTI (standoff <= 6.0m, enclosure >= 60°, duration >= 1.5s).
- Statistical aggregation: Mean, Std, Median, Min, Max, 95% Confidence Interval.
- No hardcoded defaults, no arbitrary fallback TTIs.
"""

from __future__ import annotations
import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(PROJECT_DIR))

from src.server import OBSTACLES
from src.ecs.world import ECSWorld
from src.ecs.doctrines import TacticalDoctrineID, get_doctrine_config


def evaluate_single_run(
    doctrine_id: TacticalDoctrineID,
    seed: int,
    n_steps: int = 2000,
    enable_ai: bool = True,
) -> Dict[str, Any]:
    """
    Executes a single simulation trial and computes formal KPIs.
    """
    world = ECSWorld(obstacles=OBSTACLES, seed=seed)
    world.set_tactical_doctrine(doctrine_id)

    if not enable_ai or doctrine_id != TacticalDoctrineID.DEEPSEEK_ADAPTIVE:
        world.ai_commander.enabled = False
        world.vision_recon.enabled = False

    dt = world.dt
    enclosure_angles = []
    velocities = []
    sightings_count = 0
    uncertainty_timeline = []
    initial_batteries = {i: world.batteries[i].remaining_wh for i in range(4)}

    # Formal TTI tracking: requires continuous holding window >= 1.5s (150 steps)
    tti_hold_required = int(1.5 / dt)
    tti_hold_counter = 0
    tti_candidate_time = None
    confirmed_tti = None

    for step in range(n_steps):
        telem = world.step()
        t = world.sim_time
        u = float(world.uncertainty_grid.get_mean_uncertainty())
        uncertainty_timeline.append(u)

        detected = telem.get("perception", {}).get("detected_targets", [])
        if detected:
            sightings_count += len(detected)

        # Check formal TTI condition against Target 0 (primary HVT)
        tgt_pos = world.target_transforms[0].position[:2]
        d_pos = [world.drone_transforms[i].position[:2] for i in range(3)]
        dists = [float(np.linalg.norm(d_pos[i] - tgt_pos)) for i in range(3)]

        # Find pair of drones meeting standoff radius <= 6.0m
        close_drones = [i for i in range(3) if dists[i] <= 6.0]
        pincer_condition_met = False
        current_max_angle = 0.0

        if len(close_drones) >= 2:
            for i_idx in range(len(close_drones)):
                for j_idx in range(i_idx + 1, len(close_drones)):
                    id_a, id_b = close_drones[i_idx], close_drones[j_idx]
                    vec_a = d_pos[id_a] - tgt_pos
                    vec_b = d_pos[id_b] - tgt_pos
                    norm_a = np.linalg.norm(vec_a)
                    norm_b = np.linalg.norm(vec_b)
                    if norm_a > 1e-3 and norm_b > 1e-3:
                        cos_ang = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
                        ang_deg = float(np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0))))
                        if ang_deg > current_max_angle:
                            current_max_angle = ang_deg
                        if ang_deg >= 60.0:
                            pincer_condition_met = True

        if current_max_angle > 0.0:
            enclosure_angles.append(current_max_angle)

        for did in range(3):
            v = np.linalg.norm(world.drone_transforms[did].velocity)
            velocities.append(float(v))

        # Update continuous holding window
        if confirmed_tti is None:
            if pincer_condition_met:
                if tti_hold_counter == 0:
                    tti_candidate_time = t
                tti_hold_counter += 1
                if tti_hold_counter >= tti_hold_required:
                    confirmed_tti = tti_candidate_time
            else:
                tti_hold_counter = 0
                tti_candidate_time = None

    final_batteries = {i: world.batteries[i].remaining_wh for i in range(4)}
    total_energy_wh = sum(initial_batteries[i] - final_batteries[i] for i in range(4))

    mean_enclosure = float(np.mean(enclosure_angles)) if enclosure_angles else 0.0
    mean_velocity = float(np.mean(velocities)) if velocities else 0.0
    u_reduction = float(uncertainty_timeline[0] - uncertainty_timeline[-1])

    return {
        "seed": seed,
        "doctrine": doctrine_id.value,
        "tti_seconds": round(confirmed_tti, 2) if confirmed_tti is not None else None,
        "interception_success": confirmed_tti is not None,
        "mean_enclosure_deg": round(mean_enclosure, 1),
        "mean_velocity_mps": round(mean_velocity, 2),
        "total_sightings": sightings_count,
        "final_uncertainty_pct": round(uncertainty_timeline[-1], 2),
        "uncertainty_reduction_pct": round(u_reduction, 2),
        "energy_consumed_wh": round(total_energy_wh, 3),
    }


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """Calculates mean, std, median, min, max, and 95% confidence intervals."""
    if not values:
        return {"mean": 0.0, "std": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "ci_95": 0.0}
    arr = np.array(values, dtype=np.float64)
    n = len(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    ci_95 = float(1.96 * std / np.sqrt(n)) if n > 1 else 0.0
    return {
        "mean": round(mean, 2),
        "std": round(std, 2),
        "median": round(float(np.median(arr)), 2),
        "min": round(float(np.min(arr)), 2),
        "max": round(float(np.max(arr)), 2),
        "ci_95": round(ci_95, 2),
    }


def run_benchmark_campaign(num_seeds: int = 20, steps_per_run: int = 1800, enable_ai: bool = False):
    print("=" * 80)
    print(f"  MRD-SWARM: MULTI-SEED DOCTRINE BENCHMARK CAMPAIGN ({num_seeds} SEEDS)")
    print("=" * 80)

    doctrines = [
        TacticalDoctrineID.AGGRESSIVE_PINCER,
        TacticalDoctrineID.WOLFPACK_CONTAINMENT,
        TacticalDoctrineID.STEALTH_SHADOW,
        TacticalDoctrineID.DEEPSEEK_ADAPTIVE,
    ]

    campaign_data: Dict[str, Any] = {
        "metadata": {
            "timestamp": time.time(),
            "num_seeds": num_seeds,
            "steps_per_run": steps_per_run,
            "duration_s": steps_per_run * 0.01,
            "remote_ai_enabled": enable_ai,
        },
        "raw_trials": {},
        "summary_statistics": {},
    }

    seeds = [100 + i * 17 for i in range(num_seeds)]

    for doc in doctrines:
        doc_key = doc.value
        print(f"\n[EVALUATION] Benchmarking {doc_key} over {num_seeds} seeds...", flush=True)
        trials = []
        for s_idx, seed in enumerate(seeds):
            res = evaluate_single_run(doc, seed=seed, n_steps=steps_per_run, enable_ai=enable_ai)
            trials.append(res)
            status_str = f"TTI={res['tti_seconds']}s" if res['tti_seconds'] is not None else "NOT_OBSERVED"
            if (s_idx + 1) % 5 == 0 or (s_idx + 1) == num_seeds:
                print(f"  Seed {s_idx + 1:2d}/{num_seeds} (seed={seed}): {status_str} | Enclosure={res['mean_enclosure_deg']}° | dU={res['uncertainty_reduction_pct']}%", flush=True)

        campaign_data["raw_trials"][doc_key] = trials

        # Compute aggregate statistics
        ttis = [t["tti_seconds"] for t in trials if t["tti_seconds"] is not None]
        enclosures = [t["mean_enclosure_deg"] for t in trials]
        velocities = [t["mean_velocity_mps"] for t in trials]
        uncertainty_reds = [t["uncertainty_reduction_pct"] for t in trials]
        energies = [t["energy_consumed_wh"] for t in trials]
        success_rate = (len(ttis) / num_seeds) * 100.0

        campaign_data["summary_statistics"][doc_key] = {
            "name": get_doctrine_config(doc).name,
            "success_rate_pct": round(success_rate, 1),
            "num_successful_intercepts": len(ttis),
            "num_trials": num_seeds,
            "tti_seconds": compute_statistics(ttis),
            "mean_enclosure_deg": compute_statistics(enclosures),
            "mean_velocity_mps": compute_statistics(velocities),
            "uncertainty_reduction_pct": compute_statistics(uncertainty_reds),
            "energy_consumed_wh": compute_statistics(energies),
        }

    # Save complete JSON record
    out_json = OUTPUT_DIR / "doctrine_benchmark_multiseed.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(campaign_data, f, indent=2)
    print(f"\n[BENCHMARK] Saved raw trial data and statistics to: {out_json}")

    # Generate Statistical Distribution Plots
    _generate_distribution_plots(campaign_data)


def _generate_distribution_plots(data: Dict[str, Any]):
    stats = data["summary_statistics"]
    doc_keys = list(stats.keys())
    names = [stats[k]["name"] for k in doc_keys]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    fig.patch.set_facecolor("#ffffff")

    colors = ["#f43f5e", "#0284c7", "#10b981", "#8b5cf6"]

    # 1. Interception Success Rate (%)
    ax = axes[0, 0]
    rates = [stats[k]["success_rate_pct"] for k in doc_keys]
    bars = ax.bar(names, rates, color=colors, alpha=0.85, edgecolor="#1e293b", linewidth=1.0)
    ax.set_title("Target Interception Success Rate (% of Trials)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle="--", alpha=0.5)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # 2. Time-to-Intercept (TTI) Boxplot / Error Bars
    ax = axes[0, 1]
    tti_means = [stats[k]["tti_seconds"]["mean"] for k in doc_keys]
    tti_cis = [stats[k]["tti_seconds"]["ci_95"] for k in doc_keys]
    ax.bar(names, tti_means, yerr=tti_cis, capsize=6, color=colors, alpha=0.85, edgecolor="#1e293b", linewidth=1.0)
    ax.set_title("Mean Time-to-Intercept (s) with 95% CI [Lower is Faster]", fontsize=11, fontweight="bold")
    ax.set_ylabel("Seconds (s)")
    ax.grid(True, linestyle="--", alpha=0.5)

    # 3. Pincer Enclosure Angle (deg)
    ax = axes[1, 0]
    enc_means = [stats[k]["mean_enclosure_deg"]["mean"] for k in doc_keys]
    enc_cis = [stats[k]["mean_enclosure_deg"]["ci_95"] for k in doc_keys]
    ax.bar(names, enc_means, yerr=enc_cis, capsize=6, color=colors, alpha=0.85, edgecolor="#1e293b", linewidth=1.0)
    ax.set_title("Mean Multi-Drone Enclosure Angle (°) with 95% CI", fontsize=11, fontweight="bold")
    ax.set_ylabel("Enclosure Angle (°)")
    ax.grid(True, linestyle="--", alpha=0.5)

    # 4. Total Energy Consumed (Wh)
    ax = axes[1, 1]
    e_means = [stats[k]["energy_consumed_wh"]["mean"] for k in doc_keys]
    e_cis = [stats[k]["energy_consumed_wh"]["ci_95"] for k in doc_keys]
    ax.bar(names, e_means, yerr=e_cis, capsize=6, color=colors, alpha=0.85, edgecolor="#1e293b", linewidth=1.0)
    ax.set_title("Total Energy Consumed (Wh) with 95% CI", fontsize=11, fontweight="bold")
    ax.set_ylabel("Watt-Hours (Wh)")
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle(f"Figure: Multi-Seed Tactical Doctrine Benchmark ({data['metadata']['num_seeds']} Randomized Seeds)", fontsize=13, fontweight="bold")
    fig.tight_layout()

    plot_path = OUTPUT_DIR / "plot_tactical_doctrines_comparison.png"
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"[BENCHMARK] Generated comparative plot: {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Seed Swarm Doctrine Benchmark")
    parser.add_argument("--seeds", type=int, default=20, help="Number of Monte Carlo seeds")
    parser.add_argument("--steps", type=int, default=1800, help="Simulation steps per trial (18s)")
    parser.add_argument("--enable-remote-ai", action="store_true", help="Enable live cloud LLM queries during benchmark")
    args = parser.parse_args()

    run_benchmark_campaign(num_seeds=args.seeds, steps_per_run=args.steps, enable_ai=args.enable_remote_ai)
