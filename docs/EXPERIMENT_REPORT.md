# MRD-SWARM: Empirical Multi-Seed Experimental Campaign Report

**Experiment Title:** Statistical Evaluation of Decentralized Tactical Doctrines in Cluttered Urban Environments under Dryden Turbulence  
**Campaign Scope:** 20 Independent Randomized Seeds $\times$ 4 Tactical Doctrines = 80 Total Simulation Trials  
**Execution Timestamp:** 2026-09-03  
**Data Artifacts:** `output/doctrine_benchmark_multiseed.json`, `output/plot_tactical_doctrines_comparison.png`, `output/BENCHMARK_EVALUATION_REPORT.md`  

---

## 1. Experimental Methodology & Simulation Conditions

### 1.1 Environmental Setup
- **Theater Dimensions:** $45.0\text{ m} \times 45.0\text{ m} \times 15.0\text{ m}$ ($X \in [-22.5, 22.5]$, $Y \in [-22.5, 22.5]$, $Z \in [0, 15.0]$).
- **Obstacle Density:** 5 high-rise structures (up to 12m height) and 8 urban obstacles causing significant visual and line-of-sight RF blockage.
- **Atmospheric Model:** Discrete Dryden turbulence filter (MIL-F-8785C) operating at 100 Hz ($L_u = L_v = 30.0\text{m}$, $L_w = 10.0\text{m}$, $\sigma = 1.0\text{ m/s}$), driven by seed-controlled Gaussian white noise $\mathcal{N}(0, 1)$.
- **Sensor Modeling:** Range and FOV limitations, ray-box building occlusions, optical aerosol smoke attenuation, additive Gaussian noise ($\sigma_r = 0.35\text{m}$, $\sigma_\theta = 1.5^\circ$), and 5% stochastic packet loss.

### 1.2 Evaluated Tactical Doctrines
1. **AGGRESSIVE_PINCER**: Flankers maneuver at high speed to achieve a $90^\circ$ angular enclosure around the primary target with a $5.0\text{m}$ standoff distance.
2. **WOLFPACK_CONTAINMENT**: Slower, tighter concentric perimeter ($4.0\text{m}$ standoff, $120^\circ$ separation) emphasizing containment and battery conservation.
3. **STEALTH_SHADOW**: Wide standoff ($9.0\text{m}$) utilizing high altitude and obstacle masking to minimize detection.
4. **DEEPSEEK_ADAPTIVE**: Cognitive state machine combining real-time visual smoke detection handoff, target velocity heuristics, and dynamic role re-allocation.

### 1.3 Formal Metrics & Evaluation Criteria (per `docs/METRICS_SPEC.md`)
- **Time to Intercept (TTI)**: First continuous $1.5\text{s}$ window where distance $\le 6.0\text{m}$ and enclosure $\ge 60.0^\circ$.
- **Pincer Enclosure Angle ($\Phi$)**: Angular separation between two nearest tracking drones relative to the target:
  $$\Phi = \arccos\left(\frac{(\mathbf{p}_1 - \mathbf{p}_t) \cdot (\mathbf{p}_2 - \mathbf{p}_t)}{\|\mathbf{p}_1 - \mathbf{p}_t\| \|\mathbf{p}_2 - \mathbf{p}_t\|}\right)$$
- **Epistemic Uncertainty Reduction ($\Delta U$)**: Percentage decrease in unobserved navigable free-space voxels:
  $$\Delta U = \frac{U(0) - U(T)}{U(0)} \times 100\%$$
- **Energy Consumed ($E$)**: Total battery discharge across the fleet calculated via the aerodynamic power demand model.

---

## 2. Multi-Seed Statistical Results (20 Seeds per Doctrine)

All 80 trials were executed under distinct pseudo-random seeds ($S_i = 100 + 17 \cdot i, \; i \in [0, 19]$). The aggregated results are presented below:

### 2.1 Epistemic Uncertainty Reduction ($\Delta U$)

| Doctrine | Mean (%) | Std Dev (%) | Median (%) | Min (%) | Max (%) | 95% Confidence Interval |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **AGGRESSIVE_PINCER** | 55.44 | 4.88 | 55.55 | 45.61 | 63.85 | $[53.30, 57.58]$ |
| **WOLFPACK_CONTAINMENT** | 57.82 | 5.27 | 57.82 | 46.54 | 69.17 | $[55.51, 60.13]$ |
| **STEALTH_SHADOW** | 59.44 | 7.11 | 60.30 | 44.36 | 72.90 | $[56.32, 62.56]$ |
| **DEEPSEEK_ADAPTIVE** | **61.14** | 5.86 | **61.16** | 51.40 | **72.37** | **$[58.57, 63.71]$** |

> **Finding 1 (Exploration Superiority):** `DEEPSEEK_ADAPTIVE` demonstrated the highest epistemic uncertainty reduction (mean $61.14\%$, peak $72.37\%$), statistically outperforming `AGGRESSIVE_PINCER` ($p < 0.01$). This is attributed to its dynamic posture transitions: when no target is locked, the fleet defaults to a coordinated spatial frontier sweep rather than orbiting fixed waypoints.

---

### 2.2 Pincer Enclosure Geometry ($\Phi$)

| Doctrine | Mean Angle (°) | Std Dev (°) | Median (°) | Max Angle (°) | 95% CI (°) |
|---|:---:|:---:|:---:|:---:|:---:|
| **AGGRESSIVE_PINCER** | **44.97** | 49.33 | 29.20 | **135.40** | $[23.38, 66.56]$ |
| **WOLFPACK_CONTAINMENT** | 38.64 | 45.92 | 0.00 | 124.90 | $[18.52, 58.75]$ |
| **STEALTH_SHADOW** | 19.34 | 42.44 | 0.00 | 141.00 | $[0.76, 37.93]$ |
| **DEEPSEEK_ADAPTIVE** | 23.40 | 39.21 | 0.00 | 128.90 | $[6.22, 40.58]$ |

> **Finding 2 (Enclosure Dynamics):** `AGGRESSIVE_PINCER` achieved the highest mean enclosure angle ($44.97^\circ$) and sustained geometric pinching exceeding $100^\circ$ on 6 of 20 seeds. The high standard deviation across all doctrines highlights the severe impact of building occlusion in dense urban environments: when an evasive target turns behind a skyscraper, the line-of-sight between flanking drones is broken, forcing an angular reset.

---

### 2.3 Fleet Energy Consumption ($E$)

| Doctrine | Mean ($Wh$) | Std Dev ($Wh$) | Median ($Wh$) | Min ($Wh$) | Max ($Wh$) |
|---|:---:|:---:|:---:|:---:|:---:|
| **AGGRESSIVE_PINCER** | 2.70 | 0.02 | 2.70 | 2.67 | 2.73 |
| **WOLFPACK_CONTAINMENT** | 2.72 | 0.03 | 2.72 | 2.68 | 2.77 |
| **STEALTH_SHADOW** | 2.72 | 0.03 | 2.72 | 2.68 | 2.77 |
| **DEEPSEEK_ADAPTIVE** | 2.71 | 0.03 | 2.71 | 2.65 | 2.77 |

> **Finding 3 (Aerodynamic Energy Parity):** Fleet energy consumption remained remarkably consistent across doctrines ($\approx 2.71\text{ Wh}$ over a 12.0s burst), primarily driven by the continuous hover requirement of Drone 0 ($0.65\text{kg}$) and Drone 3 ($0.50\text{kg}$). High-speed sprint surges by the Interceptor (Drone 1) contributed $< 4\%$ of total fleet energy due to its lightweight airframe ($0.28\text{kg}$).

---

### 2.4 Time to Intercept (TTI) Analysis & Negative Result Disclosure

Under the strict mathematical criterion defined in `docs/METRICS_SPEC.md` (requiring a continuous $1.5\text{s}$ holding window with distance $\le 6.0\text{m}$ and enclosure $\ge 60.0^\circ$), all four doctrines yielded a **0.0% success rate** within the 12.0-second evaluation horizon.

**Root-Cause Failure Mode Analysis:**
1. **Initial Search Latency:** Drones spawn at perimeter corners ($\approx 12\text{m}$ from the theater center). Traversing through narrow building corridors to the target's initial coordinates requires $4.0 - 7.5\text{s}$.
2. **Evasive Target Maneuvering:** Targets execute sharp $90^\circ$ turns around building corners upon detecting approaching drones, triggering Kalman track state transitions from `CONFIRMED` to `PREDICTED`.
3. **Strict Holding Window:** While drones repeatedly entered the $6.0\text{m}$ standoff zone at enclosure angles $> 70^\circ$, target evasion caused temporary breaks in the condition at $0.8 - 1.2\text{s}$, resetting the continuous holding timer before the $1.5\text{s}$ threshold was satisfied.

> **Scientific Integrity Mandate:** Unlike the previous unvalidated prototype which falsely reported a hardcoded `"100% Interception PASS"`, this benchmark reports the genuine empirical result: **12 seconds is insufficient for guaranteed continuous-window pincer containment in dense urban clutter**. Future mission envelopes must allocate a minimum $30 - 45\text{s}$ operational window for sustained target containment.

---

## 3. Comprehensive 60-Second Aerospace Benchmark

In addition to the 20-seed Monte Carlo sweep, an extended 60.0-second full-stack mission was evaluated via `scripts/run_eval_benchmark.py`:

| Performance Metric | Measured Value | Threshold Requirement | Evaluator Status |
|---|:---:|:---:|:---:|
| **Time to 90% Coverage ($T_{90}$)** | 22.25 s | $\le 18.0\text{ s}$ | **FAIL** |
| **Total Uncertainty Reduction** | 72.3% | $\ge 75.0\%$ | **FAIL** |
| **Interceptor Max Sprint Speed** | 18.00 m/s | $\ge 10.0\text{ m/s}$ | **PASS** |
| **Mean SE(3) Position RMSE** | 10.08 m | $\le 0.85\text{ m}$ | **FAIL** |
| **Mean Pincer Enclosure Angle** | 64.1° | $\ge 50.0^\circ$ | **PASS** |
| **Network Retention under EW** | 0.0% | $\ge 50.0\%$ | **FAIL** |

### Key Observations:
- **Kinematic Limits:** Drone 1 reached its exact physical saturation limit of $18.00\text{ m/s}$ during pursuit sprints, proving motor allocation and velocity clamping are functioning correctly.
- **Enclosure Angle:** During sustained tracking, the fleet maintained an average enclosure of $64.1^\circ$ (exceeding the $50.0^\circ$ requirement), successfully executing coordinated pincer geometry.
- **EW Jamming Vulnerability:** Under active Electronic Warfare jamming at $t \in [25\text{s}, 40\text{s}]$, algebraic connectivity $\lambda_2(L)$ dropped to $0.0$, demonstrating that high-altitude RF relaying alone is insufficient without directional beamforming or cognitive frequency hopping.
