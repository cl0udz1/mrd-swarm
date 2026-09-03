# MRD-SWARM: Visual Evidence & Multi-Media Deliverable Catalog

**Document Version:** 2.0.0  
**Status:** VERIFIED & HARDENED  
**Video Directory:** `media/videos/` (6 MP4 Videos, > 112 MB total)  
**Figure Directory:** `media/figures/` (16 Publication-Quality Figures)  
**Video Synthesis Script:** `python scripts/render_demo_campaign.py`  
**Aerospace Benchmark Script:** `python scripts/run_eval_benchmark.py`

---

## 1. Executive Summary

This catalog documents the complete multimedia evidence package generated for the **MRD-SWARM** multi-robot autonomous drone system. Rather than relying on textual claims, static diagrams, or unverified logs, the system features a dedicated headless multi-camera rendering pipeline built on **MuJoCo 3.12** and **imageio-ffmpeg**.

Every video is generated as a **1920 $\times$ 720 split-screen MP4** combining:
- **Left Panel (1280 $\times$ 720):** MuJoCo 3D Tactical World View rendering full urban geometry, drone airframes, dynamic propeller discs, target ground vehicles, and pyrotechnic smoke effects.
- **Top Right Panel (640 $\times$ 360):** First-Person-View (FPV) Camera Feed with live heads-up display (HUD) crosshairs, lock-on bounding reticles, and multi-spectral mode indicators (Optical EO vs. LWIR Thermal).
- **Bottom Right Panel (640 $\times$ 360):** Real-time telemetry dashboard detailing mission phase, voxel uncertainty %, active RF mesh links, Fiedler algebraic connectivity $\lambda_2$, battery states, and tactical doctrine.

---

## 2. Master Video Evidence Archive (`media/videos/`)

| Video Filename | Scenario Profile | Duration & Framerate | File Size | Primary Engineering Insight Demonstrated |
| :--- | :--- | :---: | :---: | :--- |
| **`01_open_field_pincer.mp4`** | Scenario A (Open Field) | 12.0 s @ 20 fps (240 frames) | 16.8 MB | Unobstructed cooperative pincer closure; dual-drone angular enclosure ($140^\circ$) and sustained standoff containment. |
| **`02_dense_urban_tracking.mp4`** | Scenario C (Dense Urban) | 12.0 s @ 20 fps (240 frames) | 17.0 MB | 3D Ray-AABB building occlusion; Kalman filter track persistence across urban street corridors and skyscraper corners. |
| **`03_smoke_thermal_handoff.mp4`** | Scenario E (Sensor Stress) | 12.0 s @ 20 fps (240 frames) | 17.8 MB | Pyrotechnic aerosol smoke deployment; optical EO extinction on Drone 0; Drone 2 LWIR thermal penetration and mesh handoff. |
| **`04_ew_jamming_partition_recovery.mp4`** | Scenario D (Comms Stress) | 12.0 s @ 20 fps (240 frames) | 17.0 MB | EW jamming attack severing ground links; Drone 3 climb to $Z=9.5\text{m}$ elevated perch, restoring $\lambda_2 = 1.17$ and mesh connectivity. |
| **`05_lost_target_reacquisition.mp4`** | Scenario C (Alley Evasion) | 12.0 s @ 20 fps (240 frames) | 17.3 MB | Target weaving behind building corner; Kalman filter track entering `PREDICTED` mode; coordinated sweep and reacquisition. |
| **`06_full_60s_mission.mp4`** | Complete Operational Flight | 20.0 s @ 20 fps (400 frames) | 26.5 MB | Full multi-phase mission: Area Search $\to$ Detection $\to$ Aggressive Pincer $\to$ Containment $\to$ Helipad RTB Recovery. |

---

## 3. Publication Engineering Figures (`media/figures/`)

The repository contains 16 publication-quality engineering figures across flight dynamics, estimation theory, graph resilience, and benchmark statistics:

### 3.1 Mission Synthesis Figures
1. **`01_swarm_spatial_trajectories.png`**:
   - 2D tactical map displaying 3D flight paths of all 4 heterogeneous drones and ground truth target paths against all 8 urban building footprints.
2. **`02_tracking_error_and_nees.png`**:
   - Dual-panel time series showing target position RMSE (m) and Normalized Estimation Error Squared (NEES) consistency against the theoretical $\chi^2(2)$ $95\%$ confidence boundary.
3. **`03_network_topology_evolution.png`**:
   - Continuous timeline tracking active RF mesh links and graph Laplacian Fiedler eigenvalue $\lambda_2$ under dynamic EW jamming attacks.
4. **`04_mission_phase_timeline.png`**:
   - Mission state machine progression (`SEARCH` $\to$ `HUNT` $\to$ `CONTAINMENT` $\to$ `RTB`) aligned with voxel uncertainty decay curves ($100\% \to 5\%$).
5. **`05_doctrine_ablation_summary.png`**:
   - Executive comparison chart highlighting the $40.0\%$ containment success and $9.23\text{s}$ TTI of `GOSSIP_DECENTRALIZED` vs. uncoordinated search ($30.0\%$).

### 3.2 Flight Dynamics & 6-DoF Control Figures
6. **`controller_step_response.png`**:
   - Multi-panel step response showing position rise time ($t_{10\to90} = 0.89\text{s} - 1.03\text{s}$), velocity saturation envelopes, and motor thrust clamping.
7. **`controller_orbit_tracking.png`**:
   - Banked continuous circular orbit tracking ($R=8.0\text{m}, \omega=0.5\text{ rad/s}$) with cross-track error $\le 0.488\text{m}$.
8. **`controller_dryden_rejection_psd.png`**:
   - MIL-F-8785C low-altitude atmospheric turbulence time-series realization and analytical Power Spectral Density (PSD) matching the theoretical $-20\text{ dB/decade}$ high-frequency roll-off.

### 3.3 Network & RF Resilience Figures
9. **`network_resilience_curves.png`**:
   - Parametric sweep of jamming radius ($0\text{m} - 20\text{m}$) comparing high-altitude relay retention ($100\%$ connectivity) against grounded mesh partitioning at $r = 14\text{m}$ ($\lambda_2 = 0.0$).
10. **`eval_mesh_connectivity.png`**:
    - Step-by-step active RF links and packet propagation history during 60-second flight.

### 3.4 Multi-Seed Doctrine Benchmark Figures
11. **`doctrine_benchmark_comparison.png`**:
    - Comprehensive statistical dashboard comparing TTI distributions, uncertainty reduction %, and active links across 80 Monte Carlo trials.
12. **`doctrine_radar_tradeoff.png`**:
    - 5-axis Pareto radar chart plotting Speed, Containment Success %, Coverage, Mesh Robustness, and Energy Efficiency.

### 3.5 Full Mission Aerospace Evaluation Figures
13. **`eval_3d_trajectories.png`**:
    - 3D perspective trajectory rendering showing altitude profiles, rooftop helipads, and ground target paths.
14. **`eval_kinematics_tracking.png`**:
    - 4-drone kinematic velocity time series vs. structural aerodynamic limits.
15. **`eval_uncertainty_decay.png`**:
    - Voxel grid uncertainty exponential decay curve reaching $5.5\%$ residual.
16. **`eval_target_interception.png`**:
    - Inter-vehicle separation distance $d(t)$ and angular enclosure $\theta_{\text{enc}}(t)$ verifying continuous holding window fulfillment.

---

## 4. How to Inspect Video Deliverables

All video files are standard H.264 MP4 streams playable in VLC, Windows Media Player, QuickTime, or modern web browsers:

```bash
# Play Dense Urban Tracking video via system media player
start media/videos/02_dense_urban_tracking.mp4

# Play Smoke Screen Thermal Handoff video
start media/videos/03_smoke_thermal_handoff.mp4

# Play Full Mission video
start media/videos/06_full_60s_mission.mp4
```

To re-render all videos or generate individual scenario videos with custom parameters:
```bash
python scripts/render_demo_campaign.py
```
