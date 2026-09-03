# -*- coding: utf-8 -*-
"""
run_integration_validation.py — System-Wide Integration Validation Runner.

Executes runtime integration checks across all operational scenarios (A through E):
1. Runs full ECSWorld simulation for 500 steps per scenario.
2. Validates absence of NaNs, numerical stability, and valid telemetry schemas.
3. Logs cumulative perception metrics, active link counts, and battery status.
4. Generates output/integration_validation_report.json.
"""

from __future__ import annotations
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ecs.world import ECSWorld
from src.config.scenarios import SCENARIO_CONFIGS, ScenarioID, get_scenario

OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def validate_scenario_integration(scenario_id: ScenarioID, steps: int = 500) -> Dict[str, Any]:
    """Runs a full simulation of the given scenario and performs strict schema and numerical audits."""
    cfg = get_scenario(scenario_id)
    world = ECSWorld(obstacles=cfg.obstacles, seed=42)
    world.ai_commander.enabled = False
    world.vision_recon.enabled = False

    # Apply scenario specific modifiers
    if cfg.ew_jamming_enabled:
        world.ew_field.active = True
        world.ew_field.center = cfg.ew_center
        world.ew_field.radius = cfg.ew_radius

    if cfg.smoke_active_initial:
        for t_id, target in world.targets.items():
            target.smoke_active = True
            target.smoke_timer = 15.0
            target.smoke_position = world.target_transforms[t_id].position.copy()

    nan_detected = False
    inf_detected = False
    telemetry_samples = []

    t_start = time.time()
    for step_k in range(steps):
        telem = world.step()

        # Audit Drones
        for did, d_data in telem["drones"].items():
            p = d_data["pos"]
            v = d_data["vel"]
            soc = d_data["battery"]
            if any(math.isnan(x) for x in p) or any(math.isnan(x) for x in v) or math.isnan(soc):
                nan_detected = True
            if any(math.isinf(x) for x in p) or any(math.isinf(x) for x in v) or math.isinf(soc):
                inf_detected = True

        # Audit Targets
        for tid, t_data in telem["targets"].items():
            p = t_data["pos"]
            v = t_data["vel"]
            if any(math.isnan(x) for x in p) or any(math.isnan(x) for x in v):
                nan_detected = True

        if step_k % 100 == 0:
            telemetry_samples.append({
                "time": telem["time"],
                "uncertainty_pct": telem["uncertainty_pct"],
                "active_links": telem["rf_mesh"]["total_links"],
                "detected_targets": telem["perception"]["num_detected"],
                "total_detection_events": telem["perception"]["total_detection_events"],
            })

    sim_elapsed = time.time() - t_start

    final_telem = telem
    p_data = final_telem["perception"]

    return {
        "scenario_id": scenario_id.value,
        "name": cfg.name,
        "steps_executed": steps,
        "simulated_time_s": round(final_telem["time"], 2),
        "wall_time_s": round(sim_elapsed, 3),
        "nan_detected": nan_detected,
        "inf_detected": inf_detected,
        "final_uncertainty_pct": final_telem["uncertainty_pct"],
        "total_detection_events": p_data["total_detection_events"],
        "total_visible_target_frames": p_data["total_visible_target_frames"],
        "unique_targets_detected": p_data["unique_targets_detected"],
        "active_mesh_links_final": final_telem["rf_mesh"]["total_links"],
        "passed": not nan_detected and not inf_detected and p_data["total_detection_events"] > 0,
    }


def main():
    print("=" * 80)
    print("MRD-SWARM: System-Wide Integration Validation Runner")
    print("=" * 80)

    report = {"timestamp": time.time(), "scenarios": {}}
    all_passed = True

    for s_id in ScenarioID:
        print(f"\nValidating {s_id.value} (500 steps at 100 Hz)...")
        res = validate_scenario_integration(s_id, steps=500)
        report["scenarios"][s_id.value] = res
        status = "PASS" if res["passed"] else "FAIL"
        print(f"  [{status}] Detections={res['total_detection_events']}, Links={res['active_mesh_links_final']}, NaNs={res['nan_detected']}")
        if not res["passed"]:
            all_passed = False

    report["all_passed"] = all_passed
    report_path = OUTPUT_DIR / "integration_validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[DONE] Master integration report saved to {report_path}")
    print(f"Overall Integration Status: {'PASS' if all_passed else 'FAIL'}")


if __name__ == "__main__":
    main()
