# -*- coding: utf-8 -*-
"""
run_doctrine_benchmark.py — Master Deterministic Multi-Seed Tactical Doctrine Benchmark (V2).

Implements Part 3 & Part 4 of MRD-SWARM Hardening:
1. Evaluates 4 Distinct Doctrines:
   - BASELINE_INDEPENDENT: Uncoordinated local greedy hunt.
   - CENTRALIZED_HEURISTIC: Fixed closest-assignment containment.
   - GOSSIP_DECENTRALIZED: Multi-hop utility auction pincer.
   - ADAPTIVE_DETERMINISTIC: Deterministic cognitive state machine (Offline DeepSeek Architecture).
2. Deterministic Monte Carlo Campaign across 20 distinct random seeds (seeds 1 to 20).
3. Evaluates formal metrics via src.evaluation.metrics:
   - Continuous-window TTI (standoff <= 6.0m, enclosure >= 60°, duration >= 1.5s).
   - Rich failure diagnostics: closest_distance_m, max_enclosure_deg, longest_hold_s, failure_reason.
   - Uncertainty reduction %, T90 time.
   - Tracking continuity: uptime %, track losses, reacquisition time.
   - Network connectivity: mean active links, jamming resilience.
   - Energy efficiency: total Wh, Wh/m, Wh/% reduction.
4. Statistical Hypothesis Testing:
   - Code-computed 95% confidence intervals.
   - Paired Wilcoxon signed-rank tests (p-values) against baseline.
   - Cohen's d effect sizes.
5. Saves machine-readable output/doctrine_benchmark_multiseed.json and media/figures/.
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
from scipy import stats

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
FIGURES_DIR = PROJECT_DIR / "media" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(PROJECT_DIR))

from src.ecs.world import ECSWorld
from src.ecs.doctrines import TacticalDoctrineID, get_doctrine_config
from src.config.scenarios import get_scenario, ScenarioID
from src.evaluation.metrics import (
    evaluate_tti,
    evaluate_coverage,
    evaluate_tracking_ratio,
    evaluate_position_rmse,
    TTIResult,
)

DOCTRINES = [
    {
        "id": "BASELINE_INDEPENDENT",
        "doctrine_enum": TacticalDoctrineID.STEALTH_SHADOW,
        "name": "Baseline Independent",
        "description": "Uncoordinated greedy search without swarm consensus.",
    },
    {
        "id": "CENTRALIZED_HEURISTIC",
        "doctrine_enum": TacticalDoctrineID.WOLFPACK_CONTAINMENT,
        "name": "Centralized Heuristic",
        "description": "Centralized closest-drone assignment and geometric containment.",
    },
    {
        "id": "GOSSIP_DECENTRALIZED",
        "doctrine_enum": TacticalDoctrineID.AGGRESSIVE_PINCER,
        "name": "Gossip Decentralized",
        "description": "Multi-hop RF gossip with distributed utility auction pincer.",
    },
    {
        "id": "ADAPTIVE_DETERMINISTIC",
        "doctrine_enum": TacticalDoctrineID.DEEPSEEK_ADAPTIVE,
        "name": "Adaptive Deterministic",
        "description": "Deterministic cognitive state machine mirroring AI commander.",
    },
]


def evaluate_single_trial(
    doctrine_entry: Dict[str, Any],
    seed: int,
    duration_s: float = 30.0,
) -> Dict[str, Any]:
    """Executes a single 30s-60s simulation run and logs rigorous metrics."""
    scenario = get_scenario(ScenarioID.SCENARIO_C_DENSE_URBAN)
    world = ECSWorld(obstacles=scenario.obstacles, seed=seed)
    world.set_tactical_doctrine(doctrine_entry["doctrine_enum"])

    # Strict token conservation: Remote AI disabled for benchmarking
    world.ai_commander.enabled = False
    world.vision_recon.enabled = False

    dt = world.dt
    n_steps = int(duration_s / dt)

    times: List[float] = []
    drone_positions: Dict[int, List[np.ndarray]] = {0: [], 1: [], 2: [], 3: []}
    target_0_positions: List[np.ndarray] = []
    uncertainty_history: List[float] = []
    active_links_history: List[int] = []
    tracked_history: List[bool] = []
    total_dist_traveled: Dict[int, float] = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
    prev_pos: Dict[int, np.ndarray] = {}

    for step_k in range(n_steps):
        telem = world.step()
        t = telem["time"]
        times.append(t)
        uncertainty_history.append(telem["uncertainty_pct"])
        active_links_history.append(telem["rf_mesh"]["total_links"])

        # Target 0 ground truth position
        t0_pos = np.array(telem["targets"][0]["pos"][:2], dtype=np.float64)
        target_0_positions.append(t0_pos)

        # Drone positions & distance traveled
        for did in range(4):
            d_pos = np.array(telem["drones"][did]["pos"][:2], dtype=np.float64)
            drone_positions[did].append(d_pos)
            if did in prev_pos:
                total_dist_traveled[did] += float(np.linalg.norm(d_pos - prev_pos[did]))
            prev_pos[did] = d_pos

        # Track status of Target 0 in Kalman filter
        t0_track = telem["target_tracks"].get("0", {})
        tracked_history.append(t0_track.get("state") == "CONFIRMED")

    # 1. Continuous Time-To-Intercept (TTI) via Production Metrics Module
    tti_res: TTIResult = evaluate_tti(
        timestamps=times,
        drone_positions=drone_positions,
        target_positions=target_0_positions,
        holding_window_s=1.5,
        standoff_threshold_m=6.0,
        enclosure_threshold_deg=60.0,
    )

    # 2. Epistemic Coverage
    cov_res = evaluate_coverage(
        uncertainty_history=uncertainty_history,
        timestamps=times,
        required_reduction_pct=75.0,
        required_t90_s=18.0,
    )

    # 3. Tracking Continuity
    track_res = evaluate_tracking_ratio(
        is_tracked_per_frame=tracked_history,
        sim_dt=dt,
    )

    # 4. Energy Metrics
    total_energy_wh = sum(world.batteries[did].total_energy_consumed_wh for did in range(4))
    total_dist_m = sum(total_dist_traveled.values())
    energy_per_second_w = (total_energy_wh * 3600.0) / duration_s
    energy_per_meter = total_energy_wh / max(1.0, total_dist_m)
    energy_per_pct_red = total_energy_wh / max(1.0, cov_res.reduction_pct)

    # 5. Perception Counters
    p_data = telem["perception"]

    return {
        "seed": seed,
        "interception_success": tti_res.interception_success,
        "tti_seconds": tti_res.tti_seconds,
        "closest_distance_m": tti_res.closest_distance_m,
        "max_enclosure_deg": tti_res.max_enclosure_deg,
        "longest_hold_duration_s": tti_res.longest_hold_duration_s,
        "failure_reason": tti_res.failure_reason,
        "uncertainty_reduction_pct": cov_res.reduction_pct,
        "time_to_90pct_s": cov_res.time_to_90pct_coverage_s,
        "tracking_uptime_pct": track_res["uptime_pct"],
        "track_losses": track_res["loss_count"],
        "mean_reacquisition_s": track_res["mean_reacquisition_s"],
        "mean_active_links": round(float(np.mean(active_links_history)), 2),
        "total_energy_wh": round(total_energy_wh, 3),
        "energy_per_meter_wh_m": round(energy_per_meter, 4),
        "energy_per_pct_wh": round(energy_per_pct_red, 4),
        "total_detection_events": p_data["total_detection_events"],
        "total_visible_target_frames": p_data["total_visible_target_frames"],
    }


def compute_statistics(values: List[float]) -> Dict[str, Any]:
    """Computes Mean, Std, Median, Min, Max, and 95% Confidence Interval."""
    if not values:
        return {"mean": None, "std": None, "median": None, "min": None, "max": None, "ci_95": None, "n": 0}

    arr = np.array(values, dtype=np.float64)
    n = len(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    median = float(np.median(arr))
    val_min = float(np.min(arr))
    val_max = float(np.max(arr))

    # 95% Student's t Confidence Interval
    if n > 1 and std > 1e-6:
        ci_half = float(stats.t.ppf(0.975, df=n - 1) * (std / math.sqrt(n)))
    else:
        ci_half = 0.0

    return {
        "mean": round(mean, 2),
        "std": round(std, 2),
        "median": round(median, 2),
        "min": round(val_min, 2),
        "max": round(val_max, 2),
        "ci_95": round(ci_half, 2),
        "n": n,
    }


def compute_paired_statistics(
    baseline_series: List[float],
    comparison_series: List[float],
) -> Dict[str, Any]:
    """Computes Wilcoxon signed-rank paired p-value and Cohen's d effect size."""
    n = len(baseline_series)
    if n != len(comparison_series) or n < 5:
        return {"p_value": None, "cohens_d": None, "statistically_significant": False}

    diffs = np.array(comparison_series) - np.array(baseline_series)
    mean_diff = float(np.mean(diffs))

    # Cohen's d for paired samples
    sd_diff = float(np.std(diffs, ddof=1))
    cohens_d = round(mean_diff / sd_diff, 3) if sd_diff > 1e-6 else 0.0

    # Wilcoxon signed-rank test
    try:
        if np.all(diffs == 0):
            p_val = 1.0
        else:
            w_res = stats.wilcoxon(diffs, alternative="two-sided")
            p_val = float(w_res.pvalue)
    except Exception:
        p_val = 1.0

    return {
        "mean_difference": round(mean_diff, 2),
        "cohens_d": cohens_d,
        "p_value": round(p_val, 4),
        "statistically_significant": bool(p_val < 0.05),
    }


def run_benchmark(n_seeds: int = 20, duration_s: float = 30.0) -> Dict[str, Any]:
    print("=" * 80)
    print(f"MRD-SWARM: Master Deterministic Benchmark Campaign ({n_seeds} Seeds x {len(DOCTRINES)} Doctrines)")
    print(f"Duration per Trial: {duration_s:.1f}s | Environment: Dense Urban (Scenario C)")
    print("=" * 80)

    results: Dict[str, Any] = {
        "metadata": {
            "timestamp": time.time(),
            "n_seeds": n_seeds,
            "duration_s": duration_s,
            "backend": "ADAPTIVE_DETERMINISTIC_OFFLINE",
        },
        "doctrines": {},
        "paired_comparisons": {},
    }

    raw_trials: Dict[str, List[Dict[str, Any]]] = {}

    for doc in DOCTRINES:
        doc_id = doc["id"]
        print(f"\nEvaluating Doctrine: {doc['name']} ({doc_id})...")
        raw_trials[doc_id] = []

        t_start = time.time()
        for s_idx in range(1, n_seeds + 1):
            seed = s_idx * 100 + 42
            trial_res = evaluate_single_trial(doc, seed=seed, duration_s=duration_s)
            raw_trials[doc_id].append(trial_res)

        elapsed = time.time() - t_start
        print(f"  Completed {n_seeds} trials in {elapsed:.1f}s ({elapsed / n_seeds:.2f}s/trial)")

        # Aggregate metrics
        ttis = [t["tti_seconds"] for t in raw_trials[doc_id] if t["tti_seconds"] is not None]
        successes = [1.0 if t["interception_success"] else 0.0 for t in raw_trials[doc_id]]
        reds = [t["uncertainty_reduction_pct"] for t in raw_trials[doc_id]]
        uptimes = [t["tracking_uptime_pct"] for t in raw_trials[doc_id]]
        links = [t["mean_active_links"] for t in raw_trials[doc_id]]
        energies = [t["total_energy_wh"] for t in raw_trials[doc_id]]
        detections = [t["total_detection_events"] for t in raw_trials[doc_id]]

        results["doctrines"][doc_id] = {
            "name": doc["name"],
            "description": doc["description"],
            "success_rate_pct": round(float(np.mean(successes)) * 100.0, 1),
            "tti_seconds": compute_statistics(ttis),
            "uncertainty_reduction_pct": compute_statistics(reds),
            "tracking_uptime_pct": compute_statistics(uptimes),
            "mean_active_links": compute_statistics(links),
            "total_energy_wh": compute_statistics(energies),
            "total_detection_events": compute_statistics(detections),
            "trials": raw_trials[doc_id],
        }

        print(f"  Success Rate:        {results['doctrines'][doc_id]['success_rate_pct']}%")
        print(f"  TTI (Mean +/- 95%CI): {results['doctrines'][doc_id]['tti_seconds']['mean']} +/- {results['doctrines'][doc_id]['tti_seconds']['ci_95']} s")
        print(f"  Uncertainty Red:     {results['doctrines'][doc_id]['uncertainty_reduction_pct']['mean']}%")
        print(f"  Tracking Uptime:     {results['doctrines'][doc_id]['tracking_uptime_pct']['mean']}%")
        print(f"  Total Energy:        {results['doctrines'][doc_id]['total_energy_wh']['mean']} Wh")

    # ── Paired Hypothesis Testing against Baseline ─────────────────────────────
    base_id = "BASELINE_INDEPENDENT"
    base_reds = [t["uncertainty_reduction_pct"] for t in raw_trials[base_id]]
    base_uptimes = [t["tracking_uptime_pct"] for t in raw_trials[base_id]]
    base_energies = [t["total_energy_wh"] for t in raw_trials[base_id]]

    for doc_id in ["CENTRALIZED_HEURISTIC", "GOSSIP_DECENTRALIZED", "ADAPTIVE_DETERMINISTIC"]:
        comp_reds = [t["uncertainty_reduction_pct"] for t in raw_trials[doc_id]]
        comp_uptimes = [t["tracking_uptime_pct"] for t in raw_trials[doc_id]]
        comp_energies = [t["total_energy_wh"] for t in raw_trials[doc_id]]

        results["paired_comparisons"][f"{doc_id}_vs_{base_id}"] = {
            "uncertainty_reduction": compute_paired_statistics(base_reds, comp_reds),
            "tracking_uptime": compute_paired_statistics(base_uptimes, comp_uptimes),
            "energy_consumption": compute_paired_statistics(base_energies, comp_energies),
        }

    # Save JSON
    json_path = OUTPUT_DIR / "doctrine_benchmark_multiseed.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[DONE] Benchmark saved to {json_path}")

    # ── Generate Publication Figures ───────────────────────────────────────────
    print("\nGenerating Figure: Doctrine Benchmark Comparison Dashboards...")
    generate_comparison_plots(results)
    generate_radar_tradeoff(results)

    return results


def generate_comparison_plots(results: Dict[str, Any]):
    """Generates a 4-panel publication-grade benchmark plot."""
    doc_keys = list(results["doctrines"].keys())
    labels = [results["doctrines"][k]["name"] for k in doc_keys]
    colors = ["#6b7280", "#3b82f6", "#10b981", "#8b5cf6"]

    fig, axs = plt.subplots(2, 2, figsize=(13, 10))

    # Panel 1: Interception Success Rate
    rates = [results["doctrines"][k]["success_rate_pct"] for k in doc_keys]
    bars1 = axs[0, 0].bar(labels, rates, color=colors, alpha=0.85, edgecolor="black", linewidth=1.2)
    axs[0, 0].set_ylabel("Success Rate (%)", fontsize=11, fontweight="bold")
    axs[0, 0].set_title("(A) Target Interception Success Rate (20 Seeds)", fontsize=12, fontweight="bold")
    axs[0, 0].set_ylim(0, 110)
    axs[0, 0].grid(axis="y", linestyle=":", alpha=0.6)
    for bar in bars1:
        yval = bar.get_height()
        axs[0, 0].text(bar.get_x() + bar.get_width() / 2.0, yval + 2, f"{yval:.1f}%", ha="center", va="bottom", fontweight="bold")

    # Panel 2: Mean Time-to-Intercept (TTI) with 95% CIs
    tti_means = [results["doctrines"][k]["tti_seconds"]["mean"] or 0.0 for k in doc_keys]
    tti_cis = [results["doctrines"][k]["tti_seconds"]["ci_95"] or 0.0 for k in doc_keys]
    bars2 = axs[0, 1].bar(labels, tti_means, yerr=tti_cis, capsize=5, color=colors, alpha=0.85, edgecolor="black", linewidth=1.2)
    axs[0, 1].set_ylabel("TTI (seconds)", fontsize=11, fontweight="bold")
    axs[0, 1].set_title("(B) Mean Time-to-Intercept (95% CI)", fontsize=12, fontweight="bold")
    axs[0, 1].grid(axis="y", linestyle=":", alpha=0.6)
    for bar, m in zip(bars2, tti_means):
        axs[0, 1].text(bar.get_x() + bar.get_width() / 2.0, m + 0.5, f"{m:.2f}s", ha="center", va="bottom", fontweight="bold")

    # Panel 3: Epistemic Uncertainty Reduction
    u_means = [results["doctrines"][k]["uncertainty_reduction_pct"]["mean"] for k in doc_keys]
    u_cis = [results["doctrines"][k]["uncertainty_reduction_pct"]["ci_95"] for k in doc_keys]
    bars3 = axs[1, 0].bar(labels, u_means, yerr=u_cis, capsize=5, color=colors, alpha=0.85, edgecolor="black", linewidth=1.2)
    axs[1, 0].axhline(75.0, color="crimson", linestyle="--", linewidth=1.5, label="Requirement Threshold (>= 75%)")
    axs[1, 0].set_ylabel("Reduction (%)", fontsize=11, fontweight="bold")
    axs[1, 0].set_title("(C) Uncertainty Reduction Rate (95% CI)", fontsize=12, fontweight="bold")
    axs[1, 0].set_ylim(0, 105)
    axs[1, 0].grid(axis="y", linestyle=":", alpha=0.6)
    axs[1, 0].legend(loc="lower right", frameon=True)

    # Panel 4: Fleet Energy Consumption
    e_means = [results["doctrines"][k]["total_energy_wh"]["mean"] for k in doc_keys]
    e_cis = [results["doctrines"][k]["total_energy_wh"]["ci_95"] for k in doc_keys]
    bars4 = axs[1, 1].bar(labels, e_means, yerr=e_cis, capsize=5, color=colors, alpha=0.85, edgecolor="black", linewidth=1.2)
    axs[1, 1].set_ylabel("Energy (Wh)", fontsize=11, fontweight="bold")
    axs[1, 1].set_title("(D) Swarm Total Energy Consumption (95% CI)", fontsize=12, fontweight="bold")
    axs[1, 1].grid(axis="y", linestyle=":", alpha=0.6)

    for ax in axs.flat:
        ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "doctrine_benchmark_comparison.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"  [SAVED] {fig_path}")


def generate_radar_tradeoff(results: Dict[str, Any]):
    """Generates a multi-axis radar chart showing tactical trade-offs across doctrines."""
    categories = [
        "Success Rate",
        "TTI Speed",
        "Uncertainty Red.",
        "Tracking Continuity",
        "Energy Efficiency",
    ]
    n_vars = len(categories)
    angles = [n / float(n_vars) * 2 * math.pi for n in range(n_vars)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    doc_keys = list(results["doctrines"].keys())
    colors = ["#6b7280", "#3b82f6", "#10b981", "#8b5cf6"]

    for k, col in zip(doc_keys, colors):
        d_data = results["doctrines"][k]
        # Normalize metrics to [0, 100]
        s_rate = d_data["success_rate_pct"]
        # Invert TTI speed: faster is higher score
        tti_m = d_data["tti_seconds"]["mean"] or 30.0
        tti_score = max(10.0, 100.0 - (tti_m / 30.0) * 80.0)
        u_score = d_data["uncertainty_reduction_pct"]["mean"]
        track_score = d_data["tracking_uptime_pct"]["mean"]
        # Invert energy: lower Wh is higher efficiency score
        e_m = d_data["total_energy_wh"]["mean"]
        e_score = max(10.0, 100.0 - (e_m / 15.0) * 50.0)

        values = [s_rate, tti_score, u_score, track_score, e_score]
        values += values[:1]

        ax.plot(angles, values, linewidth=2, linestyle="solid", label=d_data["name"], color=col)
        ax.fill(angles, values, color=col, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_title("MRD-SWARM Multi-Criteria Tactical Trade-off", fontsize=12, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), frameon=True)

    plt.tight_layout()
    radar_path = FIGURES_DIR / "doctrine_radar_tradeoff.png"
    plt.savefig(radar_path, dpi=200)
    plt.close()
    print(f"  [SAVED] {radar_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MRD-SWARM Doctrine Benchmark V2")
    parser.add_argument("--seeds", type=int, default=20, help="Number of Monte Carlo seeds (default 20)")
    parser.add_argument("--duration", type=float, default=30.0, help="Duration in seconds (default 30.0)")
    args = parser.parse_args()

    run_benchmark(n_seeds=args.seeds, duration_s=args.duration)
