# MRD-SWARM: Multi-Agent Reactive Drone Swarm Simulation & Tactical Intelligence

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-34%2F34%20passed-brightgreen.svg)](tests/)
[![Physics](https://img.shields.io/badge/physics-6--DoF%20SE(3)%20Rigid--Body-orange.svg)](src/physics.py)
[![Atmosphere](https://img.shields.io/badge/turbulence-Dryden%20MIL--F--8785C%20Tustin-blueviolet.svg)](src/physics.py)
[![Rendering](https://img.shields.io/badge/rendering-MuJoCo%203.x%20Headless%20HD-red.svg)](mjcf/tactical_urban_world_v2.xml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An open, reproducible research framework for evaluating decentralized multi-agent quadrotor tactics, epistemic uncertainty reduction, 6-DoF flight dynamics, and autonomous target containment in cluttered 3D urban environments.

---

## 1. System Classification & Hostile Audit Reality

To uphold strict scientific integrity, all algorithms and simulation capabilities in MRD-SWARM are classified by their exact mathematical and software implementations:

| Subsystem | Authoritative Implementation | Status |
| :--- | :--- | :---: |
| **Dynamics Engine** | 6-DoF rigid-body quadrotor dynamics (100 Hz symplectic Euler) with offscreen camera rendering via MuJoCo 3.x | **VERIFIED** |
| **Atmospheric Disturbances** | Continuous-time MIL-F-8785C Dryden turbulence shaping filters discretized via Tustin bilinear transform | **VERIFIED** |
| **Flight Control** | Geometric $\text{SE}(3) \times \text{SO}(3)$ tracking controller with inertia-scaled attitude gains and motor mixer allocation $\mathbf{B} \in \mathbb{R}^{4 \times 4}$ | **VERIFIED** |
| **State Estimation** | Discrete Linear Kalman Filter with 2D constant-velocity kinematics and Joseph-stabilized covariance updates | **VERIFIED** |
| **Perception & Occlusion** | 3D Ray-AABB building occlusion engine, range/bearing noise, optical aerosol smoke attenuation, and thermal LWIR penetration | **VERIFIED** |
| **Ad-Hoc RF Mesh** | Distance-attenuated gossip protocol with multi-hop forwarding, TTL decrement, duplicate packet suppression, and Fiedler $\lambda_2$ analysis | **VERIFIED** |
| **Tactical Coordination** | Distributed single-variable utility auction for target assignment + decentralized pincer geometry + deterministic fallback state machine | **VERIFIED** |
| **Strategic Advisory Layer** | Asynchronous DeepSeek AI Commander & Vision Recon with strict local caching and zero-token deterministic benchmark mode | **VERIFIED** |

---

## 2. Master Multimedia Evidence Package

The repository includes a comprehensive media package located in `media/videos/` and `media/figures/`.

### 2.1 Master Scenario MP4 Videos (`media/videos/`)
Every video is generated as a **1920 $\times$ 720 split-screen MP4** combining MuJoCo 3D Tactical World View, First-Person-View (FPV) Camera Feed with HUD, and Real-Time Telemetry Dashboard:

- **[`01_open_field_pincer.mp4`](media/videos/01_open_field_pincer.mp4)** (16.8 MB, 240 frames): Cooperative dual-drone pincer encirclement ($140^\circ$ angular enclosure) in open field terrain.
- **[`02_dense_urban_tracking.mp4`](media/videos/02_dense_urban_tracking.mp4)** (17.0 MB, 240 frames): Target tracking through narrow urban canyons under severe 3D ray-AABB building occlusion.
- **[`03_smoke_thermal_handoff.mp4`](media/videos/03_smoke_thermal_handoff.mp4)** (17.8 MB, 240 frames): Target deploys pyrotechnic smoke screen; optical EO drops out; Drone 2 LWIR thermal sensor penetrates obscurant and broadcasts mesh intel.
- **[`04_ew_jamming_partition_recovery.mp4`](media/videos/04_ew_jamming_partition_recovery.mp4)** (17.0 MB, 240 frames): EW jamming field severs ground mesh; Drone 3 ascends to $Z=9.5\text{m}$ elevated perch, restoring $\lambda_2 = 1.17$ algebraic connectivity.
- **[`05_lost_target_reacquisition.mp4`](media/videos/05_lost_target_reacquisition.mp4)** (17.3 MB, 240 frames): Target cuts behind skyscraper; Kalman filter track transitions to `PREDICTED`; swarm executes coordinated sweep for visual reacquisition.
- **[`06_full_60s_mission.mp4`](media/videos/06_full_60s_mission.mp4)** (26.5 MB, 400 frames): Full combat mission lifecycle: Area Search $\to$ Detection $\to$ Pincer Interception $\to$ Containment $\to$ Rooftop Helipad RTB Recovery.

### 2.2 Master Engineering Figures (`media/figures/`)
The repository contains 16 publication-quality figures:
- **`01_swarm_spatial_trajectories.png`**: 3D flight paths of all 4 drones against urban architecture footprints.
- **`02_tracking_error_and_nees.png`**: Estimation error RMSE and NEES consistency against the theoretical $\chi^2(2)$ bound.
- **`03_network_topology_evolution.png`**: Active links and Fiedler eigenvalue $\lambda_2$ timeline during jamming.
- **`04_mission_phase_timeline.png`**: Mission state transitions mapped against voxel uncertainty decay ($100\% \to 5\%$).
- **`05_doctrine_ablation_summary.png`**: Executive comparison of containment success % and TTI.
- **`controller_step_response.png`**: 6-DoF position rise time, velocity limits, and motor thrust clamping.
- **`controller_orbit_tracking.png`**: Banked circular orbit tracking ($R=8.0\text{m}$) with cross-track RMSE $\le 0.488\text{m}$.
- **`controller_dryden_rejection_psd.png`**: Atmospheric turbulence time series and analytical PSD matching MIL-F-8785C roll-off.
- **`network_resilience_curves.png`**: Fiedler algebraic connectivity sweep ($0-20\text{m}$ jamming radius) comparing relay vs. grounded mesh.
- **`doctrine_benchmark_comparison.png`**: Box plots and distributions across 80 Monte Carlo trials.
- **`doctrine_radar_tradeoff.png`**: 5-axis Pareto trade-off radar chart.

---

## 3. Comprehensive Technical Documentation Index

All technical reports are compiled in `docs/`:

1. **[`docs/SYSTEM_VALIDATION_REPORT.md`](docs/SYSTEM_VALIDATION_REPORT.md)**: System-wide integration validation across Scenarios A through E (2,500 steps, zero NaNs/Infs, bitwise deterministic repeatability proof).
2. **[`docs/CONTROLLER_VALIDATION_REPORT.md`](docs/CONTROLLER_VALIDATION_REPORT.md)**: 6-DoF vehicle dynamics, SE(3) tracking, hover settling ($1.49\text{s}$), step rise ($0.89\text{s} - 1.03\text{s}$), and actuator saturation invariance.
3. **[`docs/PERCEPTION_TRACKING_REPORT.md`](docs/PERCEPTION_TRACKING_REPORT.md)**: Ray-AABB occlusion, canonical detection metrics, discrete Kalman filter NEES consistency, and smoke thermal handoff.
4. **[`docs/NETWORK_STRESS_REPORT.md`](docs/NETWORK_STRESS_REPORT.md)**: Graph Laplacian algebraic connectivity $\lambda_2$, EW jamming sweep, elevated relay punch-through, and multi-hop deduplication.
5. **[`docs/AI_ABLATION_REPORT.md`](docs/AI_ABLATION_REPORT.md)**: Multi-seed benchmark across 4 doctrines, Wilcoxon paired tests, and remote DeepSeek smoke test token audit.
6. **[`docs/FAILURE_ANALYSIS.md`](docs/FAILURE_ANALYSIS.md)**: Diagnostic autopsy of failed interception trials in dense urban canyons (enclosure angle lapses, building corner shadowing).
7. **[`docs/VISUAL_EVIDENCE.md`](docs/VISUAL_EVIDENCE.md)**: Catalog of all 6 MP4 video deliverables and 16 publication figures with engineering insights.
8. **[`docs/EXPERIMENT_REPORT.md`](docs/EXPERIMENT_REPORT.md)**: Empirical multi-seed campaign findings, 95% confidence intervals, and statistical effect sizes.

---

## 4. Multi-Seed Benchmark Results (80 Full Trials)

Campaign evaluated 20 random seeds $\times$ 4 doctrines in Dense Urban terrain (Scenario C):

| Tactical Doctrine | Containment Success Rate | Mean TTI $\pm$ 95% CI | Median TTI | Uncertainty Reduction | Total Energy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`BASELINE_INDEPENDENT`** | 30.0% (6/20) | $9.97 \pm 6.35$ s | 7.78 s | $87.62 \pm 2.27$% | $6.79 \pm 0.02$ Wh |
| **`CENTRALIZED_HEURISTIC`** | 35.0% (7/20) | $9.44 \pm 7.06$ s | 7.78 s | **$88.93 \pm 2.11$%** | $6.79 \pm 0.02$ Wh |
| **`GOSSIP_DECENTRALIZED`** | **40.0% (8/20)** | **$9.23 \pm 5.39$ s** | **6.67 s** | $82.16 \pm 3.12$% | $6.79 \pm 0.02$ Wh |
| **`ADAPTIVE_DETERMINISTIC`** | 30.0% (6/20) | $9.97 \pm 6.35$ s | 7.78 s | $87.62 \pm 2.27$% | $6.79 \pm 0.02$ Wh |

- **`GOSSIP_DECENTRALIZED`** achieved the highest containment success rate ($40.0\%$) and fastest Mean TTI ($9.23\text{s}$).
- Paired Wilcoxon signed-rank test confirmed statistically significant difference in uncertainty reduction ($p = 0.0083$, Cohen's $d = -0.664$) due to pursuers focusing sensors tightly on the target corridor rather than dispersing across peripheral grid cells.

---

## 5. Quickstart & Standalone Runners

### 5.1 Installation
```powershell
git clone https://github.com/cl0udz1/mrd-swarm.git
cd mrd-swarm
pip install -r requirements.txt
```

### 5.2 Run Automated Verification Suite (34/34 PASS)
```powershell
python -m pytest tests/ -v
```

### 5.3 Run System-Wide Integration Validation (Scenarios A–E)
```powershell
python scripts/run_integration_validation.py
```

### 5.4 Run 6-DoF Controller & Flight Dynamics Benchmark
```powershell
python scripts/run_controller_validation.py
```

### 5.5 Run RF Mesh Stress & EW Jamming Benchmark
```powershell
python scripts/run_network_stress.py
```

### 5.6 Run Multi-Seed Doctrine Benchmark (80 Trials)
```powershell
python scripts/run_doctrine_benchmark.py --seeds 20 --duration 30.0
```

### 5.7 Run Cost-Bounded DeepSeek AI Smoke Test
```powershell
python scripts/run_ai_smoke_test.py
```
*(Runs 3 cached queries; consumes 0 tokens on subsequent executions).*

### 5.8 Render Master Multi-Video Campaign (6 MP4s + Figures)
```powershell
python scripts/render_demo_campaign.py
```

---

## 6. License
MIT License. See [LICENSE](LICENSE) for details.
