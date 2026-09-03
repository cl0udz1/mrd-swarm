# -*- coding: utf-8 -*-
"""
metrics.py — Authoritative Production Evaluation Metrics Engine for MRD-SWARM.

Single source of truth for benchmark evaluations and automated unit tests.
Implements mathematically formal criteria defined in docs/METRICS_SPEC.md.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Sequence
import numpy as np


@dataclass(frozen=True)
class RequirementResult:
    """Formal boolean evaluation result against an engineering threshold."""
    metric_name: str
    measured_value: float
    threshold_value: float
    comparison_operator: str  # "<=", ">=", "<", ">", "=="
    passed: bool
    status_label: str = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "status_label", "PASS" if self.passed else "FAIL")


@dataclass
class TTIResult:
    """Comprehensive diagnostic result for Time-to-Intercept (TTI)."""
    tti_seconds: Optional[float]
    interception_success: bool
    closest_distance_m: float
    max_enclosure_deg: float
    longest_hold_duration_s: float
    partial_intercept_attempts: int
    failure_reason: str  # "SUCCESS", "TARGET_NEVER_REACHED", "INSUFFICIENT_HOLD_DURATION", "ENCLOSURE_ANGLE_TOO_NARROW"


@dataclass
class CoverageResult:
    """Epistemic uncertainty reduction and coverage rate."""
    initial_pct: float
    final_pct: float
    reduction_pct: float
    time_to_90pct_coverage_s: Optional[float]
    passed_reduction: bool
    passed_t90: bool


def evaluate_enclosure(
    p1: np.ndarray,
    p2: np.ndarray,
    p_target: np.ndarray,
) -> float:
    """
    Computes angular enclosure separation Phi in degrees between two drones relative to target.
    Phi = arccos( (p1 - pt) . (p2 - pt) / (|p1 - pt| * |p2 - pt|) ).
    Returns angle in degrees in range [0.0, 180.0].
    """
    v1 = np.asarray(p1)[:2] - np.asarray(p_target)[:2]
    v2 = np.asarray(p2)[:2] - np.asarray(p_target)[:2]
    d1 = float(np.linalg.norm(v1))
    d2 = float(np.linalg.norm(v2))
    if d1 < 1e-4 or d2 < 1e-4:
        return 0.0
    dot = float(np.dot(v1, v2))
    cos_angle = float(np.clip(dot / (d1 * d2), -1.0, 1.0))
    return float(math.degrees(math.acos(cos_angle)))


def evaluate_tti(
    timestamps: Sequence[float],
    drone_positions: Dict[int, Sequence[np.ndarray]],
    target_positions: Sequence[np.ndarray],
    holding_window_s: float = 1.5,
    standoff_threshold_m: float = 6.0,
    enclosure_threshold_deg: float = 60.0,
) -> TTIResult:
    """
    Evaluates continuous-window Time-to-Intercept (TTI).
    Interception is confirmed ONLY when distance <= standoff_threshold_m AND
    enclosure >= enclosure_threshold_deg continuously for at least holding_window_s.
    """
    n_steps = len(timestamps)
    if n_steps == 0:
        return TTIResult(None, False, float("inf"), 0.0, 0.0, 0, "NO_DATA")

    d_min = float("inf")
    phi_max = 0.0
    hold_start_t: Optional[float] = None
    longest_hold = 0.0
    partial_attempts = 0
    in_partial = False
    confirmed_tti: Optional[float] = None

    for k in range(n_steps):
        t = timestamps[k]
        pt = target_positions[k]

        # Distances from each drone to target
        d_list = []
        p_list = []
        for did, p_hist in drone_positions.items():
            if k < len(p_hist):
                p_d = p_hist[k]
                dist = float(np.linalg.norm(np.asarray(p_d)[:2] - np.asarray(pt)[:2]))
                d_list.append(dist)
                p_list.append(p_d)

        if len(d_list) < 2:
            continue

        # Sort two closest drones
        idx_sorted = np.argsort(d_list)
        d_close = d_list[idx_sorted[0]]
        if d_close < d_min:
            d_min = d_close

        p_close1 = p_list[idx_sorted[0]]
        p_close2 = p_list[idx_sorted[1]]
        phi = evaluate_enclosure(p_close1, p_close2, pt)
        if phi > phi_max:
            phi_max = phi

        # Check interception condition
        condition_met = (d_close <= standoff_threshold_m) and (phi >= enclosure_threshold_deg)

        if condition_met:
            if not in_partial:
                partial_attempts += 1
                in_partial = True
            if hold_start_t is None:
                hold_start_t = t
            current_hold = t - hold_start_t
            if current_hold > longest_hold:
                longest_hold = current_hold
            if current_hold >= holding_window_s and confirmed_tti is None:
                confirmed_tti = round(hold_start_t, 2)
        else:
            in_partial = False
            hold_start_t = None

    # Determine failure reason
    if confirmed_tti is not None:
        reason = "SUCCESS"
        success = True
    elif d_min > standoff_threshold_m:
        reason = "TARGET_NEVER_REACHED"
        success = False
    elif phi_max < enclosure_threshold_deg:
        reason = "ENCLOSURE_ANGLE_TOO_NARROW"
        success = False
    else:
        reason = "INSUFFICIENT_HOLD_DURATION"
        success = False

    return TTIResult(
        tti_seconds=confirmed_tti,
        interception_success=success,
        closest_distance_m=round(d_min, 3),
        max_enclosure_deg=round(phi_max, 1),
        longest_hold_duration_s=round(longest_hold, 2),
        partial_intercept_attempts=partial_attempts,
        failure_reason=reason,
    )


def evaluate_coverage(
    uncertainty_history: Sequence[float],
    timestamps: Sequence[float],
    required_reduction_pct: float = 75.0,
    required_t90_s: float = 18.0,
    threshold_coverage_pct: float = 90.0,
) -> CoverageResult:
    """Evaluates uncertainty reduction and T90 coverage time."""
    if not uncertainty_history:
        return CoverageResult(100.0, 100.0, 0.0, None, False, False)

    u_init = float(uncertainty_history[0])
    u_final = float(uncertainty_history[-1])
    reduction = float(max(0.0, (u_init - u_final) / max(1e-4, u_init) * 100.0))

    # T90: time when uncertainty drops below 10% (i.e. 90% explored)
    t90: Optional[float] = None
    target_u_thresh = u_init * 0.10
    for u_val, t_val in zip(uncertainty_history, timestamps):
        if u_val <= target_u_thresh:
            t90 = round(t_val, 2)
            break

    pass_red = reduction >= required_reduction_pct
    pass_t90 = (t90 is not None) and (t90 <= required_t90_s)

    return CoverageResult(
        initial_pct=round(u_init, 2),
        final_pct=round(u_final, 2),
        reduction_pct=round(reduction, 2),
        time_to_90pct_coverage_s=t90,
        passed_reduction=pass_red,
        passed_t90=pass_t90,
    )


def evaluate_position_rmse(
    true_positions: Sequence[np.ndarray],
    estimated_positions: Sequence[np.ndarray],
    threshold_rmse_m: float = 0.85,
) -> Tuple[float, bool]:
    """Computes Euclidean Position Root Mean Squared Error (RMSE)."""
    if len(true_positions) == 0 or len(true_positions) != len(estimated_positions):
        return float("inf"), False

    diffs = np.array(true_positions) - np.array(estimated_positions)
    sq_err = np.sum(diffs[:, :2] ** 2, axis=1)
    rmse = float(np.sqrt(np.mean(sq_err)))
    return round(rmse, 4), (rmse <= threshold_rmse_m)


def evaluate_tracking_ratio(
    is_tracked_per_frame: Sequence[bool],
    sim_dt: float = 0.01,
) -> Dict[str, Any]:
    """Computes tracking uptime %, track losses, and mean reacquisition time."""
    n_frames = len(is_tracked_per_frame)
    if n_frames == 0:
        return {"uptime_pct": 0.0, "loss_count": 0, "mean_reacquisition_s": 0.0}

    tracked_count = sum(1 for v in is_tracked_per_frame if v)
    uptime_pct = (tracked_count / n_frames) * 100.0

    # Transitions from True -> False denote track loss
    losses = 0
    reacq_times: List[float] = []
    current_loss_duration = 0.0
    in_loss = False

    for v in is_tracked_per_frame:
        if not v:
            if not in_loss:
                losses += 1
                in_loss = True
            current_loss_duration += sim_dt
        else:
            if in_loss:
                reacq_times.append(current_loss_duration)
                current_loss_duration = 0.0
                in_loss = False

    mean_reacq = float(np.mean(reacq_times)) if reacq_times else 0.0

    return {
        "uptime_pct": round(uptime_pct, 2),
        "loss_count": losses,
        "mean_reacquisition_s": round(mean_reacq, 2),
    }


def evaluate_network_retention(
    nominal_fiedler: float,
    jammed_fiedler: float,
    threshold_retention_pct: float = 50.0,
) -> Tuple[float, bool]:
    """Computes algebraic connectivity retention under EW jamming."""
    if nominal_fiedler <= 1e-6:
        return 0.0, False
    retention = float(np.clip((jammed_fiedler / nominal_fiedler) * 100.0, 0.0, 100.0))
    passed = retention >= threshold_retention_pct
    return round(retention, 2), passed


def evaluate_requirement(
    metric_name: str,
    measured_value: float,
    threshold_value: float,
    comparison: str = "<=",
) -> RequirementResult:
    """Evaluates an arbitrary scalar requirement against a standard threshold."""
    if comparison == "<=":
        passed = measured_value <= threshold_value
    elif comparison == ">=":
        passed = measured_value >= threshold_value
    elif comparison == "<":
        passed = measured_value < threshold_value
    elif comparison == ">":
        passed = measured_value > threshold_value
    elif comparison == "==":
        passed = math.isclose(measured_value, threshold_value, abs_tol=1e-5)
    else:
        raise ValueError(f"Unknown comparison operator: {comparison}")

    return RequirementResult(
        metric_name=metric_name,
        measured_value=measured_value,
        threshold_value=threshold_value,
        comparison_operator=comparison,
        passed=passed,
    )
