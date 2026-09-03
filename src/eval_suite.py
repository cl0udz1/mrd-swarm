# -*- coding: utf-8 -*-
"""
eval_suite.py — Scientific Aerospace Evaluation & Mission Benchmark Suite

Computes formal aerospace, robotics, and multi-agent metrics from Black Box logs.
"""

from __future__ import annotations
import math
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


class SwarmMissionEvaluator:
    """
    Computes rigorous mission metrics from FlightDataRecorder records.
    """

    def __init__(self, records: List[Dict[str, Any]]):
        self.records = records

    def compute_all_metrics(self) -> Dict[str, Any]:
        if not self.records:
            return {"error": "No records available for evaluation"}

        duration = self.records[-1]["sim_time"] - self.records[0]["sim_time"]

        # 1. Epistemic Uncertainty & Exploration
        u_init = self.records[0]["uncertainty_pct"]
        u_final = self.records[-1]["uncertainty_pct"]
        t_90 = None  # Time when uncertainty <= 10%
        for r in self.records:
            if r["uncertainty_pct"] <= 10.0 and t_90 is None:
                t_90 = r["sim_time"]

        # 2. Tracking Precision (RMSE) & Speeds
        drone_stats = {}
        for did in range(4):
            pos_errs = []
            vel_errs = []
            speeds = []
            distances = 0.0
            prev_pos = None
            final_soc = 100.0

            for r in self.records:
                d = r["drones"].get(did)
                if d:
                    pos_errs.append(d["pos_err"])
                    vel_errs.append(d["vel_err"])
                    speeds.append(d["speed"])
                    final_soc = d["soc_pct"]
                    if prev_pos is not None:
                        distances += float(np.linalg.norm(d["pos"] - prev_pos))
                    prev_pos = d["pos"].copy()

            drone_stats[did] = {
                "rmse_pos_m": float(np.sqrt(np.mean(np.square(pos_errs)))) if pos_errs else 0.0,
                "max_pos_err_m": float(np.max(pos_errs)) if pos_errs else 0.0,
                "rmse_vel_mps": float(np.sqrt(np.mean(np.square(vel_errs)))) if vel_errs else 0.0,
                "mean_speed_mps": float(np.mean(speeds)) if speeds else 0.0,
                "max_speed_mps": float(np.max(speeds)) if speeds else 0.0,
                "total_distance_m": round(distances, 2),
                "final_soc_pct": round(final_soc, 1),
            }

        # 3. Target Acquisition & Track Maintenance
        target_acq_times = {}
        target_tracked_frames = {tid: 0 for tid in range(3)}
        for tid in range(3):
            for r in self.records:
                t = r["targets"].get(tid)
                if t and t["is_spotted"]:
                    if tid not in target_acq_times:
                        target_acq_times[tid] = r["sim_time"]
                    target_tracked_frames[tid] += 1

        tmr_pct = {
            tid: (target_tracked_frames[tid] / max(1, len(self.records))) * 100.0
            for tid in range(3)
        }

        # 4. Pincer Enclosure Angles (D1 and D2 around HVT-0 when tracked)
        pincer_angles = []
        for r in self.records:
            t0 = r["targets"].get(0)
            d1 = r["drones"].get(1)
            d2 = r["drones"].get(2)
            if t0 and d1 and d2 and t0["is_spotted"]:
                r1 = d1["pos"][:2] - t0["pos"][:2]
                r2 = d2["pos"][:2] - t0["pos"][:2]
                n1 = np.linalg.norm(r1)
                n2 = np.linalg.norm(r2)
                if n1 > 0.5 and n2 > 0.5:
                    cos_theta = float(np.dot(r1, r2) / (n1 * n2))
                    cos_theta = np.clip(cos_theta, -1.0, 1.0)
                    angle_deg = float(np.degrees(np.arccos(cos_theta)))
                    pincer_angles.append(angle_deg)

        mean_pincer_angle = float(np.mean(pincer_angles)) if pincer_angles else 0.0

        # 5. Network Resilience & Algebraic Connectivity (Fiedler Value)
        lambda_nominal = []
        lambda_jammed = []
        for r in self.records:
            if r["ew_active"]:
                lambda_jammed.append(r["lambda_2_fiedler"])
            else:
                lambda_nominal.append(r["lambda_2_fiedler"])

        mean_lambda_nom = float(np.mean(lambda_nominal)) if lambda_nominal else 0.0
        mean_lambda_jam = float(np.mean(lambda_jammed)) if lambda_jammed else 0.0

        return {
            "mission_duration_s": round(duration, 2),
            "epistemic_uncertainty": {
                "initial_pct": u_init,
                "final_pct": u_final,
                "reduction_pct": round(u_init - u_final, 1),
                "time_to_90pct_coverage_s": round(t_90, 2) if t_90 is not None else "N/A (> duration)",
            },
            "per_drone_kinematics": drone_stats,
            "tactical_interception": {
                "acquisition_times_s": target_acq_times,
                "track_maintenance_ratio_pct": tmr_pct,
                "mean_pincer_enclosure_deg": round(mean_pincer_angle, 1),
            },
            "network_resilience": {
                "nominal_fiedler_lambda_2": round(mean_lambda_nom, 4),
                "jammed_fiedler_lambda_2": round(mean_lambda_jam, 4),
                "algebraic_connectivity_retention_pct": round((mean_lambda_jam / (mean_lambda_nom + 1e-6)) * 100.0, 1) if mean_lambda_nom > 0 else 100.0,
            }
        }
