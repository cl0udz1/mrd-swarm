# -*- coding: utf-8 -*-
"""
flight_recorder.py — High-Frequency (100 Hz) Black Box Flight Data Recorder

Records synchronous SE(3) kinematic states, control tracking errors, motor inputs,
electrochemical energy depletion, Graph Laplacian network topology, and target locks.
"""

from __future__ import annotations
import csv
import io
import time
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


class FlightDataRecorder:
    """
    High-frequency circular and continuous telemetry buffer with zero-copy recording.
    """

    def __init__(self, max_buffer_size: int = 60000):  # 600 seconds at 100 Hz
        self.max_buffer_size = max_buffer_size
        self.records: List[Dict[str, Any]] = []
        self.start_wall_time = time.time()
        self.reset()

    def reset(self) -> None:
        self.records.clear()
        self.start_wall_time = time.time()

    def record_step(
        self,
        sim_time: float,
        drone_transforms: Dict[int, Any],
        physics: Dict[int, Any],
        batteries: Dict[int, Any],
        tacticals: Dict[int, Any],
        setpoints: Dict[int, Any],
        target_transforms: Dict[int, Any],
        targets: Dict[int, Any],
        detected_target_ids: set,
        active_links: List[Tuple[int, int]],
        ew_active: bool,
        uncertainty_pct: float,
    ) -> None:
        """Records one synchronous timestep across all subsystems."""
        # 1. Compute Graph Laplacian & Fiedler Eigenvalue lambda_2
        num_nodes = len(drone_transforms)
        adj_mat = np.zeros((num_nodes, num_nodes), dtype=np.float64)
        for a, b in active_links:
            if a < num_nodes and b < num_nodes:
                adj_mat[a, b] = 1.0
                adj_mat[b, a] = 1.0

        deg_mat = np.diag(np.sum(adj_mat, axis=1))
        laplacian = deg_mat - adj_mat
        eigenvals = np.sort(np.linalg.eigvalsh(laplacian))
        lambda_2 = float(eigenvals[1]) if len(eigenvals) > 1 else 0.0
        mean_degree = float(np.mean(np.sum(adj_mat, axis=1)))

        # 2. Extract per-drone tracking errors
        drone_data = {}
        for did, trans in drone_transforms.items():
            sp = setpoints.get(did)
            if sp:
                pos_err = float(np.linalg.norm(trans.position - sp.target_position))
                vel_err = float(np.linalg.norm(trans.velocity - sp.target_velocity))
            else:
                pos_err, vel_err = 0.0, 0.0

            drone_data[did] = {
                "pos": trans.position.copy(),
                "vel": trans.velocity.copy(),
                "speed": float(np.linalg.norm(trans.velocity)),
                "quat": trans.quaternion.copy(),
                "thrust_N": float(physics[did].total_thrust_N),
                "soc_pct": float(batteries[did].soc_pct),
                "pos_err": pos_err,
                "vel_err": vel_err,
                "role": tacticals[did].role.name,
            }

        # 3. Extract target states
        target_data = {}
        for tid, t_trans in target_transforms.items():
            target_data[tid] = {
                "pos": t_trans.position.copy(),
                "vel": t_trans.velocity.copy(),
                "speed": float(np.linalg.norm(t_trans.velocity)),
                "state": targets[tid].state.name,
                "is_spotted": tid in detected_target_ids,
                "smoke_active": targets[tid].smoke_active,
            }

        row = {
            "sim_time": round(sim_time, 3),
            "uncertainty_pct": round(uncertainty_pct, 2),
            "num_active_links": len(active_links),
            "mean_degree": round(mean_degree, 2),
            "lambda_2_fiedler": round(lambda_2, 4),
            "ew_active": ew_active,
            "drones": drone_data,
            "targets": target_data,
        }

        self.records.append(row)
        if len(self.records) > self.max_buffer_size:
            self.records.pop(0)

    def get_live_metrics_summary(self) -> Dict[str, Any]:
        """Calculates live windowed statistical metrics for HUD streaming."""
        if not self.records:
            return {
                "rmse_pos": 0.0,
                "rmse_vel": 0.0,
                "mean_lambda_2": 0.0,
                "mean_speed": 0.0,
                "track_ratio": 0.0,
                "total_energy_wh": 0.0,
            }

        # Recent 300 samples (last 3 seconds)
        window = self.records[-300:] if len(self.records) >= 300 else self.records
        all_pos_errs = []
        all_vel_errs = []
        all_speeds = []
        lambda_vals = []
        tracked_count = 0

        for r in window:
            lambda_vals.append(r["lambda_2_fiedler"])
            for did, d in r["drones"].items():
                all_pos_errs.append(d["pos_err"])
                all_vel_errs.append(d["vel_err"])
                all_speeds.append(d["speed"])
            is_any_tracked = any(t["is_spotted"] for t in r["targets"].values())
            if is_any_tracked:
                tracked_count += 1

        rmse_pos = float(np.sqrt(np.mean(np.square(all_pos_errs)))) if all_pos_errs else 0.0
        rmse_vel = float(np.sqrt(np.mean(np.square(all_vel_errs)))) if all_vel_errs else 0.0
        mean_speed = float(np.mean(all_speeds)) if all_speeds else 0.0
        mean_l2 = float(np.mean(lambda_vals)) if lambda_vals else 0.0
        track_ratio = (tracked_count / len(window)) * 100.0 if window else 0.0

        return {
            "rmse_pos": round(rmse_pos, 3),
            "rmse_vel": round(rmse_vel, 3),
            "mean_lambda_2": round(mean_l2, 3),
            "mean_speed": round(mean_speed, 2),
            "track_ratio": round(track_ratio, 1),
        }

    def export_csv(self, filepath: str) -> None:
        """Exports all recorded telemetry rows to a flattened CSV file."""
        if not self.records:
            return

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            
            # Header
            header = [
                "sim_time", "uncertainty_pct", "num_links", "mean_degree", "lambda_2_fiedler", "ew_active"
            ]
            for did in range(4):
                header.extend([
                    f"d{did}_x", f"d{did}_y", f"d{did}_z",
                    f"d{did}_vx", f"d{did}_vy", f"d{did}_vz", f"d{did}_speed",
                    f"d{did}_qw", f"d{did}_qx", f"d{did}_qy", f"d{did}_qz",
                    f"d{did}_thrust_N", f"d{did}_soc_pct", f"d{did}_pos_err", f"d{did}_role"
                ])
            for tid in range(3):
                header.extend([
                    f"t{tid}_x", f"t{tid}_y", f"t{tid}_z",
                    f"t{tid}_vx", f"t{tid}_vy", f"t{tid}_speed",
                    f"t{tid}_state", f"t{tid}_is_spotted", f"t{tid}_smoke_active"
                ])
            writer.writerow(header)

            for r in self.records:
                row = [
                    r["sim_time"], r["uncertainty_pct"], r["num_active_links"],
                    r["mean_degree"], r["lambda_2_fiedler"], int(r["ew_active"])
                ]
                for did in range(4):
                    d = r["drones"].get(did)
                    if d:
                        row.extend([
                            round(d["pos"][0], 3), round(d["pos"][1], 3), round(d["pos"][2], 3),
                            round(d["vel"][0], 3), round(d["vel"][1], 3), round(d["vel"][2], 3), round(d["speed"], 2),
                            round(d["quat"][0], 4), round(d["quat"][1], 4), round(d["quat"][2], 4), round(d["quat"][3], 4),
                            round(d["thrust_N"], 2), round(d["soc_pct"], 1), round(d["pos_err"], 3), d["role"]
                        ])
                    else:
                        row.extend([0]*15)

                for tid in range(3):
                    t = r["targets"].get(tid)
                    if t:
                        row.extend([
                            round(t["pos"][0], 3), round(t["pos"][1], 3), round(t["pos"][2], 3),
                            round(t["vel"][0], 3), round(t["vel"][1], 3), round(t["speed"], 2),
                            t["state"], int(t["is_spotted"]), int(t["smoke_active"])
                        ])
                    else:
                        row.extend([0]*9)

                writer.writerow(row)

    def export_csv_string(self) -> str:
        """Returns CSV string for direct web browser download."""
        output = io.StringIO()
        writer = csv.writer(output)
        if not self.records:
            return ""

        header = ["sim_time", "uncertainty_pct", "num_links", "lambda_2", "ew_active"]
        for did in range(4):
            header.extend([f"d{did}_x", f"d{did}_y", f"d{did}_z", f"d{did}_spd", f"d{did}_soc", f"d{did}_err"])
        for tid in range(3):
            header.extend([f"t{tid}_x", f"t{tid}_y", f"t{tid}_spd", f"t{tid}_spotted"])
        writer.writerow(header)

        for r in self.records:
            row = [r["sim_time"], r["uncertainty_pct"], r["num_active_links"], r["lambda_2_fiedler"], int(r["ew_active"])]
            for did in range(4):
                d = r["drones"].get(did)
                if d:
                    row.extend([round(d["pos"][0], 2), round(d["pos"][1], 2), round(d["pos"][2], 2), round(d["speed"], 1), round(d["soc_pct"], 1), round(d["pos_err"], 2)])
                else:
                    row.extend([0]*6)
            for tid in range(3):
                t = r["targets"].get(tid)
                if t:
                    row.extend([round(t["pos"][0], 2), round(t["pos"][1], 2), round(t["speed"], 1), int(t["is_spotted"])])
                else:
                    row.extend([0]*4)
            writer.writerow(row)

        return output.getvalue()
