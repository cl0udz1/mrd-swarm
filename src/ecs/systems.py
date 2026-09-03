# -*- coding: utf-8 -*-
"""
systems.py — Data-Oriented Entity Component System (ECS) Systems

Stateless system pipelines for high-stakes tactical combat operations:
- Dynamic Electronic Warfare (EW) Jamming
- Thermal Smoke Countermeasures & IR Penetration
- Laser Target Designation
- Coordinated Pincer Ambush & Rooftop Helipad RTB
"""

from __future__ import annotations
import math
from typing import Dict, List, Tuple, Optional, Any, Set
import numpy as np

from ..physics import GRAVITY, quat_to_rotation_matrix
from ..controller import CascadedQuadrotorController
from ..perception import VoxelUncertaintyGrid, LineOfSightSensor
from ..navigation import APFReactiveNavigator, NavSetpoint
from .components import (
    TransformComponent, PhysicsBodyComponent, SensorComponent,
    BatteryComponent, TacticalComponent, RFMeshComponent, TargetEntityComponent,
    LaserDesignatorComponent, EWJammingField, TacticalRoleID, TargetStateID,
)
from .mission_state import MissionPhase
from .doctrines import TacticalDoctrineID, DoctrineConfig, get_doctrine_config


def evasion_system(
    targets: Dict[int, TargetEntityComponent],
    target_transforms: Dict[int, TransformComponent],
    drone_transforms: Dict[int, TransformComponent],
    detected_target_ids: Set[int],
    obstacles: List[Dict[str, Any]],
    dt: float,
    sim_time: float,
) -> None:
    """Updates dynamic ground vehicle evasion, corner seeking, and smoke countermeasures."""
    for t_id, target in targets.items():
        trans = target_transforms[t_id]
        is_spotted = t_id in detected_target_ids

        # Update smoke canister timer
        if target.smoke_active:
            target.smoke_timer -= dt
            if target.smoke_timer <= 0.0:
                target.smoke_active = False

        # Find distance to nearest drone
        min_drone_dist = float("inf")
        threat_pos = None
        for did, d_trans in drone_transforms.items():
            d = float(np.linalg.norm(trans.position[:2] - d_trans.position[:2]))
            if d < min_drone_dist:
                min_drone_dist = d
                threat_pos = d_trans.position.copy()

        # State transitions
        if is_spotted:
            target.state = TargetStateID.ACTIVE_EVASION
            target.time_in_evasion = 0.0

            # Automatically deploy smoke screen if drone is within 5.5m and smoke ready
            if min_drone_dist < 5.5 and not target.smoke_active:
                target.smoke_active = True
                target.smoke_timer = 6.0
                target.smoke_position = trans.position.copy()
                target.state = TargetStateID.SMOKE_SCREEN_EVASION

        elif target.state in [TargetStateID.ACTIVE_EVASION, TargetStateID.SMOKE_SCREEN_EVASION]:
            target.time_in_evasion += dt
            if target.time_in_evasion > 5.0 and not target.smoke_active:
                target.state = TargetStateID.PATROL
                target.evasion_goal = None

        if target.state in [TargetStateID.ACTIVE_EVASION, TargetStateID.SMOKE_SCREEN_EVASION]:
            # Seek skyscraper corner shadow
            if target.evasion_goal is None or np.linalg.norm(trans.position[:2] - target.evasion_goal[:2]) < 1.2:
                best_shadow = _find_shadow_corner(trans.position, threat_pos, obstacles, target.waypoints, target.current_wp_idx)
                target.evasion_goal = best_shadow

            diff = target.evasion_goal - trans.position
            dist = float(np.linalg.norm(diff[:2]))
            if dist > 0.1:
                direction = diff[:2] / dist
                speed = target.evasion_speed * (1.0 + 0.15 * math.sin(2.0 * sim_time + t_id))
                trans.velocity[:2] = direction * speed
            else:
                trans.velocity[:2] = np.zeros(2)
        else:
            # Normal Road Patrol
            curr_wp = target.waypoints[target.current_wp_idx]
            diff = curr_wp - trans.position
            dist = float(np.linalg.norm(diff[:2]))

            if dist < 1.2:
                target.current_wp_idx = (target.current_wp_idx + 1) % len(target.waypoints)
                curr_wp = target.waypoints[target.current_wp_idx]
                diff = curr_wp - trans.position
                dist = float(np.linalg.norm(diff[:2]))

            if dist > 0.1:
                direction = diff[:2] / (dist + 1e-6)
                speed = target.base_speed * (1.0 + 0.20 * math.sin(0.8 * sim_time + t_id))
                trans.velocity[:2] = direction * speed
            else:
                trans.velocity[:2] = np.zeros(2)

        trans.position[:2] += trans.velocity[:2] * dt
        trans.position[2] = 0.30
        trans.position[0] = float(np.clip(trans.position[0], -27.0, 27.0))
        trans.position[1] = float(np.clip(trans.position[1], -27.0, 27.0))


def perception_system(
    drone_transforms: Dict[int, TransformComponent],
    sensors: Dict[int, SensorComponent],
    target_transforms: Dict[int, TransformComponent],
    targets: Dict[int, TargetEntityComponent],
    los_sensor: LineOfSightSensor,
    uncertainty_grid: VoxelUncertaintyGrid,
) -> Set[int]:
    """Evaluates optical/thermal line of sight with 3D buildings and smoke screen attenuation."""
    all_spotted_targets: Set[int] = set()

    for d_id, trans in drone_transforms.items():
        sensor = sensors[d_id]
        sensor.visible_targets.clear()

        # Update 3D Voxel Uncertainty Field
        uncertainty_grid.update_coverage(
            drone_pos=trans.position,
            drone_quat=trans.quaternion,
            fov_deg=sensor.camera_fov_deg,
            max_range=sensor.max_sensor_range,
        )

        # Check visibility for each ground target
        for t_id, t_trans in target_transforms.items():
            target = targets[t_id]
            vis, conf = los_sensor.evaluate_target_visibility(
                drone_pos=trans.position,
                drone_quat=trans.quaternion,
                target_pos=t_trans.position,
                fov_deg=sensor.camera_fov_deg,
                max_range=sensor.max_sensor_range,
            )

            # Check if active smoke screen occludes target
            if vis and target.smoke_active:
                smoke_d = float(np.linalg.norm(t_trans.position[:2] - target.smoke_position[:2]))
                if smoke_d < 4.5:
                    if sensor.has_thermal_ir:
                        # Thermal Surveyor penetrates smoke with slight IR attenuation
                        conf *= 0.85
                    else:
                        # Optical camera completely occluded by thick smoke aerosol
                        vis = False
                        conf = 0.0

            if vis and conf > 0.25:
                sensor.visible_targets[t_id] = conf
                all_spotted_targets.add(t_id)

    return all_spotted_targets


def rf_mesh_system(
    drone_transforms: Dict[int, TransformComponent],
    meshes: Dict[int, RFMeshComponent],
    ew_field: EWJammingField,
) -> List[Tuple[int, int]]:
    """Calculates active RF mesh links taking Electronic Warfare jamming into account."""
    active_links: List[Tuple[int, int]] = []
    d_ids = list(drone_transforms.keys())

    # Relay drone high altitude punch-through
    relay_high = drone_transforms[3].position[2] >= 9.0

    for i in range(len(d_ids)):
        id_a = d_ids[i]
        meshes[id_a].connected_peers.clear()
        pa = drone_transforms[id_a].position

        # Check EW Jamming Field
        dist_to_jam = float(np.linalg.norm(pa[:2] - ew_field.center[:2]))
        in_jam_zone = ew_field.active and (dist_to_jam < ew_field.radius)
        meshes[id_a].jammed = in_jam_zone and not relay_high
        meshes[id_a].signal_quality_pct = 25.0 if meshes[id_a].jammed else 100.0

        for j in range(i + 1, len(d_ids)):
            id_b = d_ids[j]
            pb = drone_transforms[id_b].position
            dist = float(np.linalg.norm(pa - pb))

            base_range = 32.0 if (id_a == 3 or id_b == 3) else 18.0
            if in_jam_zone and not relay_high:
                base_range *= (1.0 - ew_field.intensity)

            if dist <= base_range:
                active_links.append((id_a, id_b))
                meshes[id_a].connected_peers.append(id_b)
                meshes[id_b].connected_peers.append(id_a)

    return active_links


def laser_designation_system(
    drone_transforms: Dict[int, TransformComponent],
    tacticals: Dict[int, TacticalComponent],
    target_transforms: Dict[int, TransformComponent],
    lasers: Dict[int, LaserDesignatorComponent],
) -> None:
    """Projects high-precision tactical laser targeting vectors onto locked targets."""
    for did, trans in drone_transforms.items():
        tac = tacticals[did]
        laser = lasers[did]

        if tac.role in [TacticalRoleID.TRACKER, TacticalRoleID.FLANKER] and tac.assigned_target_id is not None:
            t_id = tac.assigned_target_id
            t_pos = target_transforms[t_id].position
            laser.active = True
            laser.target_id = t_id
            laser.target_pos = t_pos.copy()
            laser.laser_color = "#ef4444" if tac.role == TacticalRoleID.TRACKER else "#22c55e"
        else:
            laser.active = False
            laser.target_id = None


def _score_target_priority(
    target_id: int,
    track,  # TargetTrack
    drone_transforms: Dict[int, TransformComponent],
    boundary_max: float = 25.0,
) -> float:
    """
    Multi-factor threat assessment score for target prioritization.

    S = w1*speed + w2*(1 - d_boundary/d_max) + w3*confidence + w4*evasion_factor
    Higher score = higher priority for tracking.
    """
    if track.state.name == "UNINITIALIZED":
        return -1.0

    speed = track.speed
    pos = track.position

    # Proximity to map boundary (targets near edge may escape)
    d_boundary = boundary_max - max(abs(pos[0]), abs(pos[1]))
    boundary_urgency = max(0.0, 1.0 - d_boundary / boundary_max)

    # Track confidence (inverse of age)
    confidence = max(0.0, 1.0 - track.time_since_update / track.LOST_THRESHOLD)

    # Speed threat (faster targets harder to contain)
    speed_factor = min(speed / 5.0, 1.0)

    # Weighted sum
    score = (
        0.30 * speed_factor +
        0.25 * boundary_urgency +
        0.25 * confidence +
        0.20 * (1.0 if track.state.name == "CONFIRMED" else 0.3)
    )
    return float(score)


def _get_battery_soc_pct(bat: Any) -> float:
    """Safely extract state-of-charge percentage from BatteryModel or BatteryComponent."""
    if hasattr(bat, "capacity_wh") and bat.capacity_wh > 0:
        return float(bat.remaining_wh / bat.capacity_wh * 100.0)
    if hasattr(bat, "initial_capacity_wh") and bat.initial_capacity_wh > 0:
        return float(bat.remaining_wh / bat.initial_capacity_wh * 100.0)
    return float(getattr(bat, "soc_pct", 100.0))


def _compute_pincer_geometry(
    tracker_pos: np.ndarray,
    flanker_pos: np.ndarray,
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    standoff_radius: float = 4.0,
    target_sep_deg: float = 150.0,
    lead_time: float = 3.0,
    flanker_alt: float = 3.2,
    sprint_speed: float = 16.0,
) -> tuple:
    """
    Compute coordinated pincer enclosure geometry parameterized by tactical doctrine.

    Returns:
        flanker_goal: (3,) position for flanker to achieve angular enclosure
        angular_sep_deg: angular separation between tracker and flanker around target
        tti_flanker: estimated time-to-intercept for flanker
    """
    d_tracker = tracker_pos[:2] - target_pos[:2]
    d_flanker = flanker_pos[:2] - target_pos[:2]

    angle_tracker = np.arctan2(d_tracker[1], d_tracker[0])
    angle_flanker = np.arctan2(d_flanker[1], d_flanker[0])

    delta_angle = angle_flanker - angle_tracker
    delta_angle = (delta_angle + np.pi) % (2 * np.pi) - np.pi
    angular_sep_deg = float(np.degrees(abs(delta_angle)))

    target_sep = np.radians(target_sep_deg)
    desired_flanker_angle = angle_tracker + target_sep

    v_norm = np.linalg.norm(target_vel[:2])
    if v_norm > 0.3:
        lead_offset = min(14.0, v_norm * lead_time)
        pred_pos = target_pos[:2] + (target_vel[:2] / v_norm) * lead_offset
        flanker_goal_2d = pred_pos + standoff_radius * np.array([
            np.cos(desired_flanker_angle),
            np.sin(desired_flanker_angle),
        ])
    else:
        flanker_goal_2d = target_pos[:2] + standoff_radius * np.array([
            np.cos(desired_flanker_angle),
            np.sin(desired_flanker_angle),
        ])

    dist_to_goal = float(np.linalg.norm(flanker_goal_2d - flanker_pos[:2]))
    tti_flanker = dist_to_goal / max(sprint_speed, 1.0)

    flanker_goal = np.array([flanker_goal_2d[0], flanker_goal_2d[1], flanker_alt])
    return flanker_goal, angular_sep_deg, tti_flanker


def _compute_task_utility(
    drone_id: int,
    task: str,
    drone_pos: np.ndarray,
    drone_vel: np.ndarray,
    target_pos: np.ndarray,
    battery_pct: float,
    has_thermal_ir: bool,
    smoke_active: bool,
    max_speed: float,
    target_speed: float,
) -> float:
    """
    Capability-weighted utility score for (drone, task) pair.

    U = α/dist + β*SoC + γ*sensor_match + δ*speed_advantage
    Higher utility = better fit for this drone-task assignment.
    """
    dist = max(float(np.linalg.norm(drone_pos[:2] - target_pos[:2])), 0.5)

    alpha = 0.35  # proximity
    beta = 0.20   # battery
    gamma = 0.25  # sensor match
    delta = 0.20  # speed

    proximity = 1.0 / (1.0 + dist / 10.0)
    battery_factor = battery_pct / 100.0
    speed_advantage = min(max_speed / (target_speed + 1.0), 2.0) / 2.0

    # Sensor match: thermal IR is critical when smoke is active
    if task == "TRACKER" and smoke_active and has_thermal_ir:
        sensor_match = 1.0
    elif task == "TRACKER" and smoke_active and not has_thermal_ir:
        sensor_match = 0.1  # penalize non-thermal drones in smoke
    elif task == "FLANKER":
        sensor_match = speed_advantage  # flanker needs speed
    else:
        sensor_match = 0.5

    utility = alpha * proximity + beta * battery_factor + gamma * sensor_match + delta * speed_advantage
    return float(utility)


def _compute_pnr_energy(
    drone_pos: np.ndarray,
    helipad_pos: np.ndarray,
    cruise_speed: float = 8.0,
    cruise_power_w: float = 6.0,
    reserve_wh: float = 0.3,
) -> float:
    """
    Compute Point-of-No-Return (PNR) energy threshold in Watt-hours.

    E_rtb = (d_helipad / v_cruise) * P_cruise / 3600 + E_reserve
    """
    dist = float(np.linalg.norm(drone_pos[:2] - helipad_pos[:2]))
    time_to_rtb_s = dist / max(cruise_speed, 1.0)
    energy_rtb_wh = (time_to_rtb_s * cruise_power_w) / 3600.0 + reserve_wh
    return energy_rtb_wh


def _expanding_square_waypoint(center: np.ndarray, leg_index: int, leg_spacing: float = 5.0) -> np.ndarray:
    """
    Generate waypoints for an expanding square search pattern.
    The pattern spirals outward from the center position.
    """
    # Direction sequence: East, North, West, South (repeating with increasing legs)
    directions = [
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([-1.0, 0.0]),
        np.array([0.0, -1.0]),
    ]
    # Each pair of legs is one "ring", increasing distance by leg_spacing
    ring = leg_index // 4
    dir_idx = leg_index % 4
    dist = (ring + 1) * leg_spacing

    offset = directions[dir_idx] * dist
    return np.array([center[0] + offset[0], center[1] + offset[1], 3.5])


# Drone heterogeneous specs for utility computation
_DRONE_MAX_SPEEDS = {0: 12.0, 1: 18.0, 2: 14.0, 3: 6.0}
_DRONE_HAS_THERMAL = {0: False, 1: False, 2: True, 3: False}
_HELIPAD_POS = np.array([0.0, 0.0])  # launch pad at center


def brain_decision_system(
    drone_transforms: Dict[int, TransformComponent],
    sensors: Dict[int, SensorComponent],
    tacticals: Dict[int, TacticalComponent],
    batteries: Dict[int, Any],
    target_transforms: Dict[int, TransformComponent],
    detected_target_ids: Set[int],
    uncertainty_grid: VoxelUncertaintyGrid,
    sim_time: float,
    mission_mgr: Any,   # MissionStateManager
    tracker: Any,        # EKFTargetTracker
    ai_directive: Optional[Any] = None, # TacticalDirective from DeepSeek AI Commander
    doctrine: TacticalDoctrineID | str = TacticalDoctrineID.DEEPSEEK_ADAPTIVE,
) -> None:
    """
    Enhanced tactical brain with 8 intelligence systems & parameterized doctrines:
    1. Multi-phase mission state machine
    2. EKF target track persistence
    3. Battery-aware RTB planning
    4. Coordinated pincer geometry (angular enclosure)
    5. Target priority scoring with AI Commander weighting
    6. Lost-target expanding square recovery
    7. Utility-based task allocation
    8. Swarm Tactical Doctrine Engine (Pincer vs Wolfpack vs Stealth vs Adaptive)
    """
    dt = 0.01  # decision system runs at brain_interval

    # ══════════════════════════════════════════════════════════════════════════
    # 0. RESOLVE TACTICAL DOCTRINE
    # ══════════════════════════════════════════════════════════════════════════
    resolved_doctrine = doctrine
    if doctrine == TacticalDoctrineID.DEEPSEEK_ADAPTIVE and ai_directive:
        posture = getattr(ai_directive, "strategic_posture", "")
        if "PINCER" in posture:
            resolved_doctrine = TacticalDoctrineID.AGGRESSIVE_PINCER
        elif "CONTAINMENT" in posture or "WOLFPACK" in posture:
            resolved_doctrine = TacticalDoctrineID.WOLFPACK_CONTAINMENT
        elif "SHADOW" in posture or "SWEEP" in posture:
            resolved_doctrine = TacticalDoctrineID.STEALTH_SHADOW
    doctrine_cfg = get_doctrine_config(resolved_doctrine)

    # ══════════════════════════════════════════════════════════════════════════
    # 1. EKF PREDICTION + MEASUREMENT UPDATE
    # ══════════════════════════════════════════════════════════════════════════
    tracker.predict(dt * 10)  # brain runs at 10Hz, predict over 0.1s

    # Feed sensor measurements into EKF
    for did, sensor in sensors.items():
        for tid, conf in sensor.visible_targets.items():
            if conf > 0.25 and tid in target_transforms:
                tracker.update(tid, target_transforms[tid].position[:2])

    tracked_ids = tracker.get_tracked_ids()
    lost_ids = tracker.get_lost_ids()

    # ══════════════════════════════════════════════════════════════════════════
    # 2. MISSION PHASE TRANSITIONS
    # ══════════════════════════════════════════════════════════════════════════
    drone_altitudes = {did: float(t.position[2]) for did, t in drone_transforms.items()}
    uncertainty_pct = uncertainty_grid.get_mean_uncertainty()

    phase = mission_mgr.evaluate_transitions(
        sim_time=sim_time,
        drone_altitudes=drone_altitudes,
        detected_target_ids=detected_target_ids,
        tracked_target_ids=tracked_ids,
        lost_target_ids=lost_ids,
        uncertainty_pct=uncertainty_pct,
        dt=dt * 10,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # 3. BATTERY-AWARE RTB CHECK
    # ══════════════════════════════════════════════════════════════════════════
    rtb_drones: Set[int] = set()
    for did in range(4):
        bat = batteries[did]
        remaining_wh = bat.remaining_wh
        pnr_wh = _compute_pnr_energy(drone_transforms[did].position, _HELIPAD_POS)
        soc_pct = _get_battery_soc_pct(bat)

        if remaining_wh <= pnr_wh * 1.3 or soc_pct <= 15.0:
            rtb_drones.add(did)
            tacticals[did].role = TacticalRoleID.RTB_RECOVERY
            tacticals[did].goal_position = np.array([0.0, 0.0, 2.0])
            tacticals[did].desired_speed = 6.0
            tacticals[did].active_tool = "rtb_emergency()"
            tacticals[did].reasoning = f"RTB PNR | SoC={soc_pct:.0f}% E_rem={remaining_wh:.2f}Wh E_rtb={pnr_wh:.2f}Wh"

    # ══════════════════════════════════════════════════════════════════════════
    # 4. DRONE 3: COMMS RELAY (always, unless RTB)
    # ══════════════════════════════════════════════════════════════════════════
    if 3 not in rtb_drones:
        other_pos = [drone_transforms[i].position for i in range(3) if i not in rtb_drones]
        centroid = np.mean(other_pos, axis=0) if other_pos else np.zeros(3)
        tacticals[3].role = TacticalRoleID.RELAY
        tacticals[3].goal_position = np.array([centroid[0] * 0.35, centroid[1] * 0.35, 10.5])
        tacticals[3].desired_speed = 5.5
        tacticals[3].active_tool = "comms_relay_anchor(Z=10.5m)"
        mesh_count = sum(1 for i in range(3) if i not in rtb_drones)
        tacticals[3].reasoning = f"RELAY | alt=10.5m mesh_nodes={mesh_count} phase={phase.name}"

    # Available combat drones (not RTB, not RELAY)
    combat_drones = [did for did in range(3) if did not in rtb_drones]
    if len(combat_drones) == 0:
        return

    # ══════════════════════════════════════════════════════════════════════════
    # 5. TARGET PRIORITY SCORING (WITH AI COMMANDER WEIGHTING)
    # ══════════════════════════════════════════════════════════════════════════
    target_scores: list = []
    prio_list = getattr(ai_directive, "target_priority", []) if ai_directive else []
    for tid, track in tracker.tracks.items():
        score = _score_target_priority(tid, track, drone_transforms)
        # DeepSeek AI Commander priority boost
        if prio_list and tid in prio_list:
            rank = prio_list.index(tid)
            score += max(0.0, 0.40 - rank * 0.15)
        if score > 0:
            target_scores.append((tid, score, track))
    target_scores.sort(key=lambda x: x[1], reverse=True)

    # ══════════════════════════════════════════════════════════════════════════
    # 6. PHASE-DEPENDENT BEHAVIOR
    # ══════════════════════════════════════════════════════════════════════════

    # ── LAUNCH / AREA_SWEEP: Pure frontier exploration ────────────────────
    if phase in (MissionPhase.LAUNCH, MissionPhase.AREA_SWEEP):
        for did in combat_drones:
            frontier_p, gain = uncertainty_grid.get_best_frontier(
                drone_transforms[did].position, cruise_altitude=3.8 + did * 0.8
            )
            frontier_dist = float(np.linalg.norm(frontier_p[:2] - drone_transforms[did].position[:2]))
            tacticals[did].role = TacticalRoleID.EXPLORER
            tacticals[did].assigned_target_id = None
            tacticals[did].goal_position = frontier_p
            tacticals[did].desired_speed = 8.5
            tacticals[did].active_tool = "recon_area_search()"
            tacticals[did].reasoning = (
                f"SWEEP phase={phase.name} | gain={gain:.0f} dist={frontier_dist:.1f}m "
                f"unc={uncertainty_pct:.1f}%"
            )
        return

    # ── LOST_TARGET_RECOVERY: Expanding square search ─────────────────────
    if phase == MissionPhase.LOST_TARGET_RECOVERY and len(lost_ids) > 0:
        lost_tid = list(lost_ids)[0]
        predicted_pos = tracker.get_predicted_position(lost_tid)
        escape_r = tracker.get_escape_radius(lost_tid)

        if predicted_pos is not None:
            # Assign nearest combat drone to LOST_TARGET_SWEEP
            dists = [(did, float(np.linalg.norm(drone_transforms[did].position[:2] - predicted_pos)))
                     for did in combat_drones]
            dists.sort(key=lambda x: x[1])
            sweep_did = dists[0][0]

            # Compute expanding square leg index from time in phase
            phase_time = sim_time - mission_mgr.state.phase_start_time
            leg_idx = int(phase_time / 2.0)  # new leg every 2 seconds
            search_wp = _expanding_square_waypoint(
                np.array([predicted_pos[0], predicted_pos[1]]),
                leg_idx, leg_spacing=min(escape_r, 8.0),
            )

            tacticals[sweep_did].role = TacticalRoleID.LOST_TARGET_SWEEP
            tacticals[sweep_did].assigned_target_id = lost_tid
            tacticals[sweep_did].goal_position = search_wp
            tacticals[sweep_did].desired_speed = 10.0
            track = tracker.tracks[lost_tid]
            tacticals[sweep_did].active_tool = f"expanding_square_search(HVT-{lost_tid})"
            tacticals[sweep_did].reasoning = (
                f"SWEEP LOST HVT-{lost_tid} | age={track.time_since_update:.1f}s "
                f"esc_r={escape_r:.1f}m leg={leg_idx}"
            )

            # Other combat drones continue with normal behavior
            remaining = [did for did in combat_drones if did != sweep_did]
            combat_drones_for_hunt = remaining
        else:
            combat_drones_for_hunt = combat_drones
    else:
        combat_drones_for_hunt = combat_drones

    # ── HUNT / CONTAIN: Target engagement with utility allocation ─────────
    if len(target_scores) > 0 and len(combat_drones_for_hunt) > 0:
        primary_tid, primary_score, primary_track = target_scores[0]

        # Get target state (use EKF predicted pos if available, else transform)
        p_tgt_2d = tracker.get_predicted_position(primary_tid)
        v_tgt_2d = tracker.get_predicted_velocity(primary_tid)
        if p_tgt_2d is None and primary_tid in target_transforms:
            p_tgt_2d = target_transforms[primary_tid].position[:2]
            v_tgt_2d = target_transforms[primary_tid].velocity[:2]
        if p_tgt_2d is None:
            p_tgt_2d = np.zeros(2)
            v_tgt_2d = np.zeros(2)

        p_tgt = np.array([p_tgt_2d[0], p_tgt_2d[1], target_transforms[primary_tid].position[2] if primary_tid in target_transforms else 0.5])
        v_tgt = np.array([v_tgt_2d[0], v_tgt_2d[1], 0.0])
        target_speed = float(np.linalg.norm(v_tgt[:2]))

        # Check if any target has active smoke
        smoke_active = any(
            hasattr(target_transforms.get(tid), 'velocity') and tid in detected_target_ids
            for tid in tracker.get_confirmed_ids()
        )

        # 7. UTILITY-BASED TASK ALLOCATION
        # Score each combat drone for TRACKER role
        tracker_utilities = []
        for did in combat_drones_for_hunt:
            bat = batteries[did]
            soc = _get_battery_soc_pct(bat)
            u = _compute_task_utility(
                did, "TRACKER", drone_transforms[did].position, drone_transforms[did].velocity,
                p_tgt, soc, _DRONE_HAS_THERMAL.get(did, False), smoke_active,
                _DRONE_MAX_SPEEDS.get(did, 10.0), target_speed,
            )
            tracker_utilities.append((did, u))
        tracker_utilities.sort(key=lambda x: x[1], reverse=True)

        # Best drone → TRACKER (parameterized by doctrine)
        tracker_did = tracker_utilities[0][0]
        standoff_d = float(np.linalg.norm(drone_transforms[tracker_did].position[:2] - p_tgt[:2]))
        bearing_deg = float(np.degrees(np.arctan2(
            p_tgt[1] - drone_transforms[tracker_did].position[1],
            p_tgt[0] - drone_transforms[tracker_did].position[0],
        )))

        b_rad = np.radians(bearing_deg)
        tracker_goal = np.array([
            p_tgt[0] - doctrine_cfg.standoff_radius_m * np.cos(b_rad),
            p_tgt[1] - doctrine_cfg.standoff_radius_m * np.sin(b_rad),
            doctrine_cfg.tracker_altitude_m,
        ])

        tacticals[tracker_did].role = TacticalRoleID.TRACKER
        tacticals[tracker_did].assigned_target_id = primary_tid
        tacticals[tracker_did].goal_position = tracker_goal
        tacticals[tracker_did].desired_speed = doctrine_cfg.tracker_desired_speed_mps
        tacticals[tracker_did].active_tool = f"laser_designate_hvt(HVT-{primary_tid})"
        tacticals[tracker_did].threat_score = primary_score
        tacticals[tracker_did].reasoning = (
            f"TRACK HVT-{primary_tid} [{doctrine_cfg.name[:14]}] | dist={standoff_d:.1f}m "
            f"spd={target_speed:.1f}m/s prio={primary_score:.2f} trk={primary_track.state.name}"
        )

        # Remaining combat drones → FLANKER (with doctrine pincer geometry)
        remaining = [did for did in combat_drones_for_hunt if did != tracker_did]
        if len(remaining) > 0:
            flanker_utilities = []
            for did in remaining:
                bat = batteries[did]
                soc = _get_battery_soc_pct(bat)
                u = _compute_task_utility(
                    did, "FLANKER", drone_transforms[did].position, drone_transforms[did].velocity,
                    p_tgt, soc, _DRONE_HAS_THERMAL.get(did, False), smoke_active,
                    _DRONE_MAX_SPEEDS.get(did, 10.0), target_speed,
                )
                flanker_utilities.append((did, u))
            flanker_utilities.sort(key=lambda x: x[1], reverse=True)
            flanker_did = flanker_utilities[0][0]

            # COORDINATED PINCER GEOMETRY (DOCTRINE PARAMETERIZED)
            flanker_goal, angular_sep, tti = _compute_pincer_geometry(
                tracker_pos=drone_transforms[tracker_did].position,
                flanker_pos=drone_transforms[flanker_did].position,
                target_pos=p_tgt,
                target_vel=v_tgt,
                standoff_radius=doctrine_cfg.standoff_radius_m,
                target_sep_deg=doctrine_cfg.pincer_separation_deg,
                lead_time=doctrine_cfg.lead_time_s,
                flanker_alt=doctrine_cfg.flanker_altitude_m,
                sprint_speed=doctrine_cfg.flanker_max_speed_mps,
            )

            tacticals[flanker_did].role = TacticalRoleID.FLANKER
            tacticals[flanker_did].assigned_target_id = primary_tid
            tacticals[flanker_did].goal_position = flanker_goal
            tacticals[flanker_did].desired_speed = doctrine_cfg.flanker_max_speed_mps
            tacticals[flanker_did].active_tool = f"execute_pincer_ambush(HVT-{primary_tid})"
            tacticals[flanker_did].formation_angle_deg = angular_sep
            tacticals[flanker_did].tti_seconds = tti
            tacticals[flanker_did].reasoning = (
                f"PINCER HVT-{primary_tid} [{doctrine_cfg.doctrine_id.value}] | dTheta={angular_sep:.0f}deg TTI={tti:.1f}s "
                f"spd={doctrine_cfg.flanker_max_speed_mps:.1f}m/s prio={primary_score:.2f}"
            )

            # Third drone: Secondary target split OR tri-axis containment
            other_remaining = [did for did in remaining if did != flanker_did]
            if len(other_remaining) > 0:
                explorer_did = other_remaining[0]

                # If doctrine allows multi-target split and secondary exists:
                if doctrine_cfg.multi_target_split and len(target_scores) > 1:
                    sec_tid, sec_score, sec_track = target_scores[1]
                    sec_pos = tracker.get_predicted_position(sec_tid)
                    if sec_pos is not None:
                        sec_dist = float(np.linalg.norm(drone_transforms[explorer_did].position[:2] - sec_pos))
                        tacticals[explorer_did].role = TacticalRoleID.TRACKER
                        tacticals[explorer_did].assigned_target_id = sec_tid
                        tacticals[explorer_did].goal_position = np.array([sec_pos[0], sec_pos[1], doctrine_cfg.tracker_altitude_m])
                        tacticals[explorer_did].desired_speed = doctrine_cfg.tracker_desired_speed_mps
                        tacticals[explorer_did].active_tool = f"split_track_secondary(HVT-{sec_tid})"
                        tacticals[explorer_did].threat_score = sec_score
                        tacticals[explorer_did].reasoning = (
                            f"SPLIT TRACK HVT-{sec_tid} | dist={sec_dist:.1f}m prio={sec_score:.2f} "
                            f"trk={sec_track.state.name}"
                        )
                        return
                elif not doctrine_cfg.multi_target_split:
                    # AGGRESSIVE_PINCER: Tri-axis pincer enclosure around primary HVT
                    tri_angle = np.arctan2(drone_transforms[tracker_did].position[1] - p_tgt[1], drone_transforms[tracker_did].position[0] - p_tgt[0]) - np.radians(doctrine_cfg.pincer_separation_deg)
                    tri_goal_2d = p_tgt[:2] + doctrine_cfg.standoff_radius_m * np.array([np.cos(tri_angle), np.sin(tri_angle)])
                    tacticals[explorer_did].role = TacticalRoleID.FLANKER
                    tacticals[explorer_did].assigned_target_id = primary_tid
                    tacticals[explorer_did].goal_position = np.array([tri_goal_2d[0], tri_goal_2d[1], doctrine_cfg.flanker_altitude_m + 0.4])
                    tacticals[explorer_did].desired_speed = doctrine_cfg.flanker_max_speed_mps * 0.9
                    tacticals[explorer_did].active_tool = f"tri_axis_enclosure(HVT-{primary_tid})"
                    tacticals[explorer_did].threat_score = primary_score
                    tacticals[explorer_did].reasoning = (
                        f"TRI-AXIS CUTOFF HVT-{primary_tid} | standoff={doctrine_cfg.standoff_radius_m:.1f}m "
                        f"spd={doctrine_cfg.flanker_max_speed_mps * 0.9:.1f}m/s"
                    )
                    return

                # Otherwise: continue frontier sweep
                frontier_p, gain = uncertainty_grid.get_best_frontier(
                    drone_transforms[explorer_did].position, cruise_altitude=5.0
                )
                frontier_dist = float(np.linalg.norm(frontier_p[:2] - drone_transforms[explorer_did].position[:2]))
                tacticals[explorer_did].role = TacticalRoleID.EXPLORER
                tacticals[explorer_did].assigned_target_id = None
                tacticals[explorer_did].goal_position = frontier_p
                tacticals[explorer_did].desired_speed = 9.0
                tacticals[explorer_did].active_tool = "thermal_ir_survey()"
                tacticals[explorer_did].reasoning = (
                    f"PATROL | gain={gain:.0f} dist={frontier_dist:.1f}m unc={uncertainty_pct:.1f}%"
                )
    else:
        # No targets scored — pure exploration
        for did in combat_drones_for_hunt:
            frontier_p, gain = uncertainty_grid.get_best_frontier(
                drone_transforms[did].position, cruise_altitude=3.8 + did * 0.8
            )
            frontier_dist = float(np.linalg.norm(frontier_p[:2] - drone_transforms[did].position[:2]))
            tacticals[did].role = TacticalRoleID.EXPLORER
            tacticals[did].assigned_target_id = None
            tacticals[did].goal_position = frontier_p
            tacticals[did].desired_speed = 8.5
            tacticals[did].active_tool = "recon_area_search()"
            tacticals[did].reasoning = (
                f"EXPLORE phase={phase.name} | gain={gain:.0f} dist={frontier_dist:.1f}m "
                f"unc={uncertainty_pct:.1f}%"
            )


def apf_navigation_system(
    drone_transforms: Dict[int, TransformComponent],
    tacticals: Dict[int, TacticalComponent],
    obstacles: List[Dict[str, Any]],
    navigator: APFReactiveNavigator,
) -> Dict[int, NavSetpoint]:
    """Computes collision-free 3D APF setpoints for each quadrotor."""
    setpoints: Dict[int, NavSetpoint] = {}
    positions_map = {did: trans.position for did, trans in drone_transforms.items()}

    for did, trans in drone_transforms.items():
        tac = tacticals[did]
        sp = navigator.compute_setpoint(
            current_pos=trans.position,
            current_vel=trans.velocity,
            goal_pos=tac.goal_position,
            obstacles=obstacles,
            peer_positions=positions_map,
            current_agent_id=did,
            desired_speed=tac.desired_speed,
        )
        setpoints[did] = sp
    return setpoints


def se3_control_system(
    drone_transforms: Dict[int, TransformComponent],
    physics: Dict[int, PhysicsBodyComponent],
    setpoints: Dict[int, NavSetpoint],
    controllers: Dict[int, CascadedQuadrotorController],
    wind_vel: np.ndarray,
    dt: float,
) -> None:
    """
    Geometric Tracking Controller on SE(3) with real SO(3) attitude dynamics.

    Translational outer loop computes desired thrust vector and desired orientation R_d.
    Attitude inner loop computes torque from SO(3) error, integrates angular velocity
    through Euler's rotational equation, and propagates the quaternion.

    References:
        Lee, Leok, McClamroch — "Geometric Tracking Control of a Quadrotor UAV on SE(3)"
        IEEE CDC 2010, DOI: 10.1109/CDC.2010.5717652
    """
    for did, trans in drone_transforms.items():
        phys = physics[did]
        sp = setpoints[did]

        # ═══ 1. TRANSLATIONAL OUTER LOOP ═══════════════════════════════════════
        kp_pos = 3.5
        kv_vel = 2.8
        e_p = sp.target_position - trans.position
        e_v = sp.target_velocity - trans.velocity

        # Desired acceleration in world frame (PD + gravity feedforward)
        a_des = kp_pos * e_p + kv_vel * e_v + np.array([0.0, 0.0, GRAVITY])

        # Clamp horizontal acceleration to ~45° bank limit
        max_a_horiz = 14.0
        norm_a_horiz = float(np.linalg.norm(a_des[:2]))
        if norm_a_horiz > max_a_horiz:
            a_des[:2] = (a_des[:2] / norm_a_horiz) * max_a_horiz

        # Desired thrust vector and magnitude
        f_des = phys.mass * a_des
        total_thrust_N = float(np.linalg.norm(f_des))
        total_thrust_N = float(np.clip(
            total_thrust_N,
            0.2 * phys.mass * GRAVITY,
            phys.thrust_margin * phys.mass * GRAVITY,
        ))
        phys.total_thrust_N = total_thrust_N

        # Desired body z-axis (thrust direction)
        b3_d = f_des / (np.linalg.norm(f_des) + 1e-6)

        # Target heading yaw (from velocity direction)
        v_horiz = float(np.linalg.norm(sp.target_velocity[:2]))
        if v_horiz > 0.4:
            target_yaw = math.atan2(sp.target_velocity[1], sp.target_velocity[0])
        else:
            target_yaw = 0.0

        # Construct desired SO(3) rotation matrix R_d = [b1_d, b2_d, b3_d]
        b1_c = np.array([math.cos(target_yaw), math.sin(target_yaw), 0.0])
        b2_d = np.cross(b3_d, b1_c)
        norm_b2 = float(np.linalg.norm(b2_d))
        if norm_b2 < 1e-4:
            b2_d = np.array([0.0, 1.0, 0.0])
        else:
            b2_d = b2_d / norm_b2
        b1_d = np.cross(b2_d, b3_d)
        R_d = np.column_stack([b1_d, b2_d, b3_d])

        # ═══ 2. ATTITUDE INNER LOOP (SO(3) Error Dynamics) ═════════════════════
        # Current rotation matrix from quaternion
        R = quat_to_rotation_matrix(trans.quaternion)

        # SO(3) attitude error via vee map: e_R = 0.5 * (R_d^T R - R^T R_d)^vee
        e_R_skew = R_d.T @ R - R.T @ R_d
        e_R = 0.5 * np.array([e_R_skew[2, 1], e_R_skew[0, 2], e_R_skew[1, 0]])

        # Angular velocity error (desired angular velocity is zero for hover/tracking)
        e_omega = trans.angular_velocity.copy()

        # Attitude control gains (tuned for stability with small quadrotor inertia)
        k_R = 8.0       # Proportional attitude gain [N·m/rad]
        k_omega = 2.5    # Derivative attitude gain [N·m·s/rad]

        # Desired torque: τ = -k_R * e_R - k_ω * e_ω + ω × Jω (gyroscopic compensation)
        J = phys.inertia_diag
        omega = trans.angular_velocity
        gyroscopic = np.cross(omega, J * omega)
        tau = -k_R * e_R - k_omega * e_omega + gyroscopic

        # Clamp torque to physical limits
        max_torque = 0.05  # N·m (reasonable for nano-quadrotor)
        tau = np.clip(tau, -max_torque, max_torque)

        # ═══ 3. ROTATIONAL DYNAMICS INTEGRATION ════════════════════════════════
        # Euler's rotational equation: J * dω/dt = τ - ω × Jω
        # → dω/dt = J^{-1} * (τ - ω × Jω) = J^{-1} * τ  (gyroscopic already in τ)
        J_inv = 1.0 / J  # diagonal inertia → element-wise inverse
        # Remove double-counting: τ already includes gyroscopic term
        alpha = J_inv * (tau - gyroscopic)  # angular acceleration
        trans.angular_velocity = trans.angular_velocity + alpha * dt

        # Clamp angular velocity to physical limits (~20 rad/s max body rate)
        max_omega = 20.0
        omega_norm = float(np.linalg.norm(trans.angular_velocity))
        if omega_norm > max_omega:
            trans.angular_velocity = (trans.angular_velocity / omega_norm) * max_omega

        # Quaternion integration: dq/dt = 0.5 * q ⊗ [0, ω]
        w, x, y, z = trans.quaternion
        ox, oy, oz = trans.angular_velocity
        q_dot = 0.5 * np.array([
            -x * ox - y * oy - z * oz,
             w * ox + y * oz - z * oy,
             w * oy - x * oz + z * ox,
             w * oz + x * oy - y * ox,
        ])
        trans.quaternion = trans.quaternion + q_dot * dt

        # Re-normalize quaternion to stay on unit sphere
        q_norm = np.linalg.norm(trans.quaternion)
        if q_norm > 1e-6:
            trans.quaternion /= q_norm
        # Ensure canonical form (w > 0)
        if trans.quaternion[0] < 0:
            trans.quaternion = -trans.quaternion

        # ═══ 4. TRANSLATIONAL DYNAMICS ══════════════════════════════════════════
        # Thrust in world frame through CURRENT (not desired) orientation
        R_current = quat_to_rotation_matrix(trans.quaternion)
        f_thrust_world = R_current @ np.array([0.0, 0.0, total_thrust_N])

        # Aerodynamic drag
        v_rel = trans.velocity - wind_vel
        f_drag = -0.5 * 1.225 * 0.47 * 0.015 * float(np.linalg.norm(v_rel)) * v_rel

        # Newton's second law
        acc = (f_thrust_world + np.array([0.0, 0.0, -phys.mass * GRAVITY]) + f_drag) / phys.mass

        # Symplectic Euler integration
        trans.velocity += acc * dt

        # Clamp velocity to max vehicle speed
        speed = float(np.linalg.norm(trans.velocity))
        max_speed = 18.0
        if speed > max_speed:
            trans.velocity = (trans.velocity / speed) * max_speed

        trans.position += trans.velocity * dt
        trans.position[2] = float(np.clip(trans.position[2], 0.25, 14.0))


def battery_discharge_system(
    physics: Dict[int, PhysicsBodyComponent],
    batteries: Dict[int, BatteryComponent],
    dt: float,
) -> None:
    """Updates electrochemical energy depletion."""
    for did, phys in physics.items():
        bat = batteries[did]
        thrust_ratio = phys.total_thrust_N / (phys.mass * GRAVITY + 1e-6)
        power_w = 45.0 + 85.0 * (thrust_ratio ** 1.5)
        energy_wh = (power_w * dt) / 3600.0
        bat.remaining_wh = max(0.0, bat.remaining_wh - energy_wh)
        bat.soc_pct = (bat.remaining_wh / bat.capacity_wh) * 100.0


def _find_shadow_corner(
    current_p: np.ndarray,
    threat_p: Optional[np.ndarray],
    obstacles: List[Dict[str, Any]],
    waypoints: List[np.ndarray],
    wp_idx: int,
) -> np.ndarray:
    """Finds best building corner that occludes threat."""
    if threat_p is None or len(obstacles) == 0:
        return waypoints[(wp_idx + 1) % len(waypoints)]

    threat_dir = current_p[:2] - threat_p[:2]
    threat_unit = threat_dir / (np.linalg.norm(threat_dir) + 1e-6)

    best_corner = None
    best_score = -1e9

    for obs in obstacles:
        ox, oy = obs["pos"][:2]
        hw, hl = obs["size"][:2]
        corners = [
            np.array([ox - hw - 1.2, oy - hl - 1.2, 0.3]),
            np.array([ox + hw + 1.2, oy - hl - 1.2, 0.3]),
            np.array([ox + hw + 1.2, oy + hl + 1.2, 0.3]),
            np.array([ox - hw - 1.2, oy + hl + 1.2, 0.3]),
        ]
        for c in corners:
            d = float(np.linalg.norm(c[:2] - current_p[:2]))
            if d < 18.0:
                align = float(np.dot(c[:2] - np.array([ox, oy]), threat_unit))
                score = align * 2.0 - d * 0.8
                if score > best_score:
                    best_score = score
                    best_corner = c

    return best_corner if best_corner is not None else waypoints[(wp_idx + 1) % len(waypoints)]
