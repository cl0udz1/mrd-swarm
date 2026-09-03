# AUDIT_BASELINE.md: Initial Technical & Scientific Hostile Audit Baseline

**Target System**: MRD-SWARM (Multi-Agent Reconnaissance & Defense Swarm)  
**Auditor**: Principal Robotics, Simulation, Controls & Distributed Systems Reviewer  
**Date**: September 3, 2026  
**Repository**: `https://github.com/cl0udz1/mrd-swarm` / `c:/cheetah/mrd-swarm`  
**Verdict**: **RESEARCH PROTOTYPE WITH SIGNIFICANT TERMINOLOGY INFLATION & ARCHITECTURAL GAPS**

---

## 1. Executive Statement

A rigorous, hostile technical audit was conducted on the entire codebase, mathematical models, control loops, perception systems, distributed networking algorithms, state estimation pipelines, DeepSeek AI integrations, and evaluation suites.

The core diagnosis is that while several algorithmic building blocks exist and function (e.g., 3D APF navigation, basic SO(3) attitude dynamics, real DeepSeek API streaming, Three.js WebGL visualizer), **the scientific and technical documentation significantly overstates the mathematical rigor, physics integration, and architectural reality of the implementation**.

---

## 2. Comprehensive Claim-vs-Reality Audit Matrix

| # | System / Area | Claimed Implementation | Actual Implementation | Valid? | Severity | Required Corrective Action |
|---|---|---|---|:---:|:---:|---|
| **1** | **Physics Engine** | "MuJoCo 3.x Physics (100 Hz SE(3) Rigid-Body Dynamic Pipeline)" | Custom Python forward Euler integrator in `src/ecs/systems.py`. Drones are kinematic bodies in MuJoCo; `d_world.qpos` is overwritten in Python; `mj_step` is never called in master sim. MuJoCo acts solely as an offscreen visualizer. | **NO** | **CRITICAL** | Rename honestly to: *Custom Python 6-DoF rigid-body dynamics with MuJoCo offscreen rendering*. Remove all claims of "MuJoCo aerospace physics". |
| **2** | **Vehicle Specifications** | Authoritative multi-UAV heterogeneous specs (`HEAVY_SCOUT`, `FAST_INTERCEPTOR`, etc.) | Specs are duplicated and contradict each other across files (`src/physics.py` has mass=0.47kg, bat=4.5Wh; `src/ai_agent_core.py` has mass=0.28-0.65kg, bat=22-55Wh; `src/sensors.py` has third set). | **NO** | **HIGH** | Create a single authoritative configuration module (`src/config/airframes.py`). Eliminate all hardcoded duplicate values across the codebase. |
| **3** | **Atmospheric Turbulence** | "Dryden Crosswind Turbulence Model" | Sum of 2 deterministic sine waves: $1.2\sin(0.4t) + 0.3\sin(1.1t)$. Zero stochasticity, zero shaping filter, zero spectral density. | **NO** | **CRITICAL** | Implement a genuine discrete Dryden stochastic gust filter (MIL-F-8785C spectral transfer function driven by band-limited Gaussian noise with seed control), or rename to `SyntheticPeriodicWindDisturbance`. |
| **4** | **State Estimation** | "Extended Kalman Filter (EKF) Target Tracker" (`EKFTargetTracker`) | The state vector is $[px, py, vx, vy]^T$ with linear $F$ and linear $H$. Zero nonlinear functions, zero Jacobians. It is a standard Linear Kalman Filter. | **NO** | **HIGH** | Rename to `KalmanTargetTracker` (or implement nonlinear range/bearing EKF). |
| **5** | **Estimator Sensor Inputs** | Estimator runs on real sensor observations under noise and occlusions | `tracker.update(tid, target_transforms[tid].position[:2])` feeds the simulator's exact ground-truth coordinates directly into the filter with zero measurement noise and zero sensor model. | **NO** | **CRITICAL** | Build an explicit sensor measurement pipeline: Ground Truth $\to$ Sensor FOV/Range $\to$ Noise + Dropout $\to$ Measurement $\to$ Filter $\to$ Track. Ground truth must only be used for evaluation RMSE. |
| **6** | **3D Voxel Uncertainty** | Vectorized camera frustum decay with building occlusion | Frustum decay checks range and spherical angle, but performs NO line-of-sight ray tracing against building geometry. Voxels behind skyscrapers decay as if visible. | **NO** | **HIGH** | Incorporate building occlusion checks into voxel coverage updates or bound ray tracing to free space. |
| **7** | **Flight Controller** | Cascaded $SE(3)$ geometric controller with motor allocation | Multiple disconnected controller implementations exist: `CascadedQuadrotorController` in `src/controller.py` is dead code; `se3_control_system` in `src/ecs/systems.py` directly sets body torques and clamps speeds. | **NO** | **HIGH** | Establish ONE authoritative control pipeline: Goal $\to$ APF Planner $\to$ Position Error $\to$ SO(3) Attitude Error $\to$ Body Torque $\to$ Motor Allocation $\to$ Dynamic Integration. Remove orphaned controller files. |
| **8** | **Software Architecture** | "Data-Oriented ECS / cache-coherent contiguous state updates" | Python dictionaries containing individual `@dataclass` OOP instances (`Dict[int, TransformComponent]`). | **NO** | **MEDIUM** | Rename terminology honestly to: *ECS-inspired modular simulation architecture*. Remove claims of "cache-coherent contiguous memory". |
| **9** | **Mesh Networking** | "Ad-hoc RF Gossip Protocol with multi-hop TTL propagation" | `GossipMessage.ttl = 4` is defined but never read or decremented anywhere in `src/gossip.py`. Nodes do not forward packets; communication is purely single-hop broadcast. | **NO** | **CRITICAL** | Implement actual multi-hop forwarding: deduplication table, TTL decrement, neighbor broadcast, loop prevention, and message aging. |
| **10** | **Belief Fusion** | "Decentralized Bayesian Target Belief Fusion" | `alpha = conf / (curr.conf + conf); pos = (1 - alpha)*pos + alpha*new_pos`. Pure heuristic confidence-weighted averaging. No Gaussian posteriors, no information matrices. | **NO** | **HIGH** | Rename to `ConfidenceWeightedTargetFusion` or implement a formal Covariance Intersection / Information Filter. |
| **11** | **Swarm Coordination** | "Consensus-Based Bundle Algorithm (CBBA)" | `if bid > curr_bid: winner = bidder`. Single-task highest-bid auction. No bundles, no task paths, no marginal bids, no conflict resolution tables. | **NO** | **HIGH** | Rename to `DistributedUtilityAuction` (or implement genuine CBBA with bundle reset). |
| **12** | **DeepSeek AI Authority** | LLM actively directs swarm via `drone_assignments`, `desired_speed`, `target_priority` | Only `strategic_posture` string was read to switch doctrine presets. `drone_assignments` and speeds were discarded by `brain_decision_system`. | **NO** | **CRITICAL** | Formulate an explicit `AI_AUTHORITY_MODEL.md`. Implement a validated arbitration layer where structured LLM proposals pass through deterministic safety/capability filters before influencing setpoints. |
| **13** | **Multimodal Vision** | "Closed-Loop Multimodal Vision Autonomy" | `VisionIntelCard` output was displayed on the HUD and logged to JSON, but never fed into the EKF tracker, world model, or task allocator. | **NO** | **CRITICAL** | Feed structured vision observations (with confidence and provenance) into target belief fusion so optical detection directly initiates or reinforces tracks. |
| **14** | **Benchmark Integrity** | Multi-doctrine empirical validation with aerospace KPIs | Evaluated on a single 12-second run with seed=42. TTI was defined as "first time 2 targets detected", with a 12.0s fallback when unobserved. Summary JSONs contained hardcoded `"PASS"` and `"target_tracking_coverage_pct": 100.0`. | **NO** | **CRITICAL** | Eliminate all hardcoded PASS values. Establish `METRICS_SPEC.md` with unambiguous mathematical definitions. Run multi-seed Monte Carlo trials ($\ge 20$ seeds). Distinguish success from failure honestly. |
| **15** | **Reproducibility** | Universal, cross-platform reproducibility | Machine-specific hardcoded paths (`C:/cheetah/mrd-swarm`) present across multiple files. | **NO** | **HIGH** | Refactor all path references to use `Path(__file__).resolve().parent` relative paths. |

---

## 3. Subsystem-by-Subsystem Technical Audit

### 3.1 Physics & MuJoCo Integration
- **Code Reference**: `dynamic_swarm_sim.py:400-440`, `src/ecs/systems.py:910-953`.
- **Finding**: In `dynamic_swarm_sim.py`, quadrotor poses are computed in Python using symplectic Euler integration, and the resulting $[x, y, z]$ and $[w, x, y, z]$ are written directly into MuJoCo's `d_world.qpos`. `mujoco.mj_forward(m_world, d_world)` is then called to update kinematic transforms for camera rendering. MuJoCo's solver (`mj_step`) is never invoked to integrate forces, moments, contacts, or accelerations.
- **Verdict**: MuJoCo functions as a 3D visualization and offscreen camera rendering environment, NOT the dynamic physics integrator.

### 3.2 Vehicle Dynamics & Controller
- **Code Reference**: `src/physics.py`, `src/controller.py`, `src/ecs/systems.py:820-950`.
- **Finding**: `src/controller.py` contains an unused `CascadedQuadrotorController` with hardcoded mass constants ($0.47\text{ kg}$) that conflict with the heterogeneous specs. The active simulation loop uses `se3_control_system` in `src/ecs/systems.py`. While `se3_control_system` computes an $SO(3)$ attitude error $e_R = \frac{1}{2}(R_d^T R - R^T R_d)^\vee$, it directly integrates angular velocities from a torque clamp without allocating rotor speeds to individual motors ($T_1, T_2, T_3, T_4$).
- **Verdict**: The controller is an $SO(3)$ orientation controller with direct torque integration, not a complete cascaded motor-allocated geometric controller.

### 3.3 State Estimation & Sensor Flow
- **Code Reference**: `src/ecs/target_tracker.py`, `src/ecs/systems.py:465-472`.
- **Finding**: The tracker is titled `EKFTargetTracker`, but the math is 100% linear ($F$ is constant velocity, $H = [I_2, 0]$). There is no Extended Kalman Filter formulation. Crucially, sensor measurement updates bypass any sensor noise model and pass the true target position $p_{\text{target}}$ directly into the filter.
- **Verdict**: A linear Kalman filter operating on noiseless oracle ground-truth states. Must be reconstructed with an explicit synthetic sensor model (range, bearing, Gaussian noise, occlusion dropouts).

### 3.4 Gossip Networking & Distributed Consensus
- **Code Reference**: `src/gossip.py:45-53, 178-200`.
- **Finding**: Packets have a `ttl` attribute, but when received, they are processed locally and never rebroadcast. There is zero multi-hop packet forwarding. Furthermore, "Bayesian Fusion" is a linear interpolation between current and new sightings weighted by scalar confidence values, and "CBBA" is a single-variable greedy auction without bundles or consensus resolution.
- **Verdict**: Single-hop broadcast network with heuristic confidence blending and greedy auction. Requires renaming or proper multi-hop implementation.

### 3.5 AI Control Authority & Multimodal Vision
- **Code Reference**: `src/ai_commander.py`, `src/ai_vision_recon.py`, `src/ecs/systems.py:450-460`.
- **Finding**: The DeepSeek LLM generates detailed military JSON containing `strategic_posture`, `target_priority`, `drone_assignments`, and speeds. However, `brain_decision_system` only parses `strategic_posture` as a string substring to select a doctrine preset. All other fields are ignored. The Multimodal Vision agent analyzes FPV frames, but its output only populates the HUD and summary logs; it does not update the tracker or influence flight control.
- **Verdict**: Asynchronous LLM reasoning is authentic, but its runtime control authority is restricted to doctrine selection, and vision is open-loop.

### 3.6 Evaluation & Benchmark Authenticity
- **Code Reference**: `scripts/run_eval_benchmark.py:238-245`, `scripts/run_doctrine_benchmark.py:35-75`, `dynamic_swarm_sim.py:600-630`.
- **Finding**: Benchmark reports emit hardcoded `"PASS"` tags regardless of whether KPI thresholds were satisfied. Doctrines were evaluated on a single 12-second run with seed 42. TTI was approximated by detection events rather than continuous geometric enclosure.
- **Verdict**: Evaluation artifacts are unvalidated and non-generalizable. A rigorous multi-seed Monte Carlo framework with mathematically defensible metrics must be established.

---

## 4. Required Reconstruction Roadmap

1. **Phase 1: Terminology Normalization & Documentation Truth**: Honest renaming of MuJoCo usage, Kalman filter, and auction logic.
2. **Phase 2: Authoritative Vehicle Configuration**: Centralized airframe specs in `src/config/airframes.py`.
3. **Phase 3: Stochastic Turbulence Model**: Discrete Dryden shaping filter with seed control.
4. **Phase 4 & 5: Perception & Estimation Separation**: Explicit noisy sensor model separating ground truth from observations and state estimates.
5. **Phase 6: Occlusion-Aware 3D Uncertainty Grid**: Ray tracing against building bounding boxes.
6. **Phase 7: Unified Controller & Motor Allocation**: End-to-end flight control path.
7. **Phase 8: ECS Classification**: Honest categorization as modular simulation architecture.
8. **Phase 9: Multi-Hop Gossip Networking**: Forwarding, TTL decrement, deduplication, latency.
9. **Phase 10 & 11: Fusion & Coordination Renaming**: Confidence-weighted fusion and distributed utility auction.
10. **Phase 12 & 13: AI Authority Model & Vision Closed Loop**: Validated AI arbitration layer and closed-loop vision observation fusion.
11. **Phase 14: Comprehensive Failure Mode Handling**: Deterministic fallbacks for API errors and invalid JSON.
12. **Phase 17–20: Rigorous Monte Carlo Benchmarking**: 20+ seeds per doctrine, mathematical metric definitions (`METRICS_SPEC.md`), elimination of hardcoded PASS.
13. **Phase 21 & 22: Automated Test Suite & Path Normalization**: `pytest` test suite covering all subsystems; relative paths throughout.
14. **Phase 27–30: Final Falsification Review & Documentation**: `ARCHITECTURE.md`, `FINAL_TECHNICAL_AUDIT.md`, and scientific `EXPERIMENT_REPORT.md`.
