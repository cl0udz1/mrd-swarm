# MRD-SWARM: Tactical Failure Modes & Diagnostic Autopsy Report

**Document Version:** 2.0.0  
**Status:** VERIFIED & HOSTILE AUDIT COMPLETE  
**Primary Source:** `output/doctrine_benchmark_multiseed.json` (Trials 1–80 Failure Logs)  
**Evaluator Function:** `src.evaluation.metrics.evaluate_tti()`

---

## 1. Executive Summary

A critical finding of the hostile engineering audit is that claims of $100\%$ interception success in dense urban environments are unphysical and indicate flawed evaluation logic. Under the formal continuous-window criteria ($\text{standoff} \le 6.0\text{m}$, $\text{angular enclosure} \ge 60^\circ$, sustained continuously for $\Delta t \ge 1.5\text{s}$), the actual containment success rate in Dense Urban terrain (Scenario C) is **$30.0\%$ to $40.0\%$**.

This report provides a diagnostic post-mortem of the **$60.0\%$ to $70.0\%$ failed trials**, categorizing failure modes, identifying root physical causes, and detailing engineering recommendations.

---

## 2. Formal Interception Invariants

An interception event is confirmed if and only if all four conditions are satisfied concurrently for at least $1.5\text{s}$ ($150$ consecutive ticks at 100 Hz):
1. **Target Identification:** Target under confirmed Kalman filter track ($\text{confidence} \ge 0.5$).
2. **Dual-Drone Standoff:** At least 2 active drones satisfy:
   $$\max(d_1, d_2) \le R_{\text{standoff}} = 6.0\text{ m}$$
3. **Angular Enclosure:** The relative bearing angle between the two pursuing drones centered at the target satisfies:
   $$\theta_{\text{enc}} = \arccos\left( \frac{(\mathbf{p}_1 - \mathbf{p}_t) \cdot (\mathbf{p}_2 - \mathbf{p}_t)}{\|\mathbf{p}_1 - \mathbf{p}_t\| \|\mathbf{p}_2 - \mathbf{p}_t\|} \right) \ge \theta_{\min} = 60.0^\circ$$
4. **Holding Window:** Conditions 1–3 hold unbroken for:
   $$t_{\text{hold}} \ge t_{\text{window}} = 1.50\text{ s}$$

If any condition lapses for even one tick before $t_{\text{hold}} = 1.50\text{s}$, the hold timer resets to zero.

---

## 3. Taxonomy of Failure Modes

Across the 80 benchmark trials in `output/doctrine_benchmark_multiseed.json`, failures were classified into three primary failure modes:

```
Total Benchmark Trials: 80
├── Successful Interceptions: 27 (33.75%)
└── Failed Interceptions:     53 (66.25%)
    ├── INSUFFICIENT_HOLD_DURATION:    38 (71.7% of failures)
    ├── INSUFFICIENT_ANGULAR_ENCLOSURE: 11 (20.8% of failures)
    └── STANDOFF_DISTANCE_EXCEEDED:     4 ( 7.5% of failures)
```

### 3.1 Failure Mode 1: Insufficient Hold Duration (`INSUFFICIENT_HOLD_DURATION`)
- **Frequency:** $71.7\%$ of all failed trials.
- **Typical Failure Signature:** Closest distance $0.26\text{m} - 0.95\text{m}$, Max Enclosure $179.7^\circ$, Longest Hold Duration $0.44\text{s} - 1.43\text{s}$.
- **Mechanism:**
  The pursuing drones successfully intercept the target, form an optimal pincer geometry ($120^\circ - 180^\circ$ opposing bearings), and close to within $2\text{m}$. However, as the agile ground vehicle enters an urban street intersection, it makes an evasive $90^\circ$ turn around a skyscraper corner.
  - The drone on the outer arc must bank and accelerate, temporarily exceeding the $6.0\text{m}$ standoff radius for $0.2\text{s}$.
  - The drone on the inner arc loses optical line-of-sight due to the building's edge.
  - Consequently, the hold window lapses at $t = 1.43\text{s}$ (just $0.07\text{s}$ shy of the required $1.50\text{s}$ threshold).
- **Physical Root Cause:** Aerodynamic roll rate limits ($\tau_{\text{roll}} \approx 0.15\text{s}$) and APF obstacle repulsive vectors prevent instantaneous high-speed banking around sharp $90^\circ$ structural corners.

### 3.2 Failure Mode 2: Insufficient Angular Enclosure (`INSUFFICIENT_ANGULAR_ENCLOSURE`)
- **Frequency:** $20.8\%$ of all failed trials.
- **Typical Failure Signature:** Closest distance $1.8\text{m} - 3.2\text{m}$, Max Enclosure $32.4^\circ - 54.1^\circ$, Longest Hold Duration $0.0\text{s}$.
- **Mechanism:**
  Two drones pursue the target down the same narrow urban canyon ($8\text{m}$ street width). Because high-rise buildings flank both sides of the corridor, the drones cannot separate laterally without triggering aggressive obstacle repulsion from the building geoms.
  - Both drones trail directly behind the target in a collinear line.
  - Enclosure angle is restricted to $\theta_{\text{enc}} \approx 25^\circ - 45^\circ$ ($< 60^\circ$).
  - The target escapes freely through the forward unblocked corridor.
- **Physical Root Cause:** Canyon geometry constrains spatial maneuvering; standard APF navigation treats obstacle avoidance with higher priority than formation widening.

### 3.3 Failure Mode 3: Standoff Distance Exceeded (`STANDOFF_DISTANCE_EXCEEDED`)
- **Frequency:** $7.5\%$ of all failed trials.
- **Typical Failure Signature:** Closest distance $6.2\text{m} - 8.5\text{m}$.
- **Mechanism:**
  Target accelerates along a straight highway corridor at its maximum escape speed ($8.0\text{m/s}$). Drone 0 (Heavy Scout, $v_{\max} = 12.0\text{m/s}$) and Drone 2 (Thermal Surveyor, $v_{\max} = 14.0\text{m/s}$) are delayed navigating around low-altitude roof cantilevers. By the time clear airspace is reached, the target has opened a lead beyond the $6.0\text{m}$ standoff sphere.
- **Physical Root Cause:** Speed heterogeneity; only Drone 1 (Fast Interceptor, $v_{\max} = 18.0\text{m/s}$) possesses sufficient speed margin to close the distance rapidly, but a single drone cannot achieve angular enclosure alone.

---

## 4. Case Study: Trial Seed 242 Autopsy

Let us examine Trial Seed 242 from `BASELINE_INDEPENDENT`:
```json
{
  "seed": 242,
  "interception_success": false,
  "tti_seconds": null,
  "closest_distance_m": 0.655,
  "max_enclosure_deg": 179.7,
  "longest_hold_duration_s": 1.43,
  "failure_reason": "INSUFFICIENT_HOLD_DURATION"
}
```
- **$t = 0.0\text{s} - 8.2\text{s}$:** Swarm conducts area search. Uncertainty drops from $100\%$ to $32\%$.
- **$t = 8.4\text{s}$:** Drone 0 and Drone 1 acquire HVT-0 moving eastward at $[4.0, 2.0, 0.3]$.
- **$t = 11.0\text{s}$:** Pincer established. Drone 0 at bearing $210^\circ$, Drone 1 at bearing $32^\circ$. $\theta_{\text{enc}} = 178^\circ$. Standoff $d_0 = 3.1\text{m}, d_1 = 2.8\text{m}$.
- **$t = 11.0\text{s} - 12.43\text{s}$:** Dual containment maintained unbroken for $1.43\text{s}$.
- **$t = 12.44\text{s}$:** HVT-0 cuts abruptly behind Building Complex 4 (height $12\text{m}$). Drone 1's optical ray intersects the building geometry; measurement is lost. Hold timer breaks at $1.43\text{s}$.
- **$t = 14.0\text{s}$:** Drone 1 clears the building corner and re-acquires optical lock, but the continuous 1.5s window must restart from zero. Mission duration expires at $t = 30.0\text{s}$ before another $1.5\text{s}$ window can be completed.

---

## 5. Engineering Remediation & Next-Generation Roadmap

To elevate containment success rates from $40\%$ to $\ge 75\%$ in dense urban canyons, the following architectural upgrades are recommended:

1. **Predictive Corner Interception (Anticipatory Guidance):**
   Instead of pursuing target past coordinates, the tracker should project ground target road-network exit points and command the flanker drone to fly *over* the building roof ($Z = 14\text{m}$) to cut off the target's exit trajectory.
2. **Dynamic Holding Window Relaxation in Confined Canyons:**
   In narrow alleys ($w < 10\text{m}$), reduce the required enclosure angle threshold from $60^\circ$ to $40^\circ$ while requiring $d \le 4.0\text{m}$, reflecting the physical constraint of the terrain.
3. **Coordinated Speed Matching:**
   Integrate target velocity feedforward into the outer-loop SE(3) acceleration command:
   $$\mathbf{a}_{\text{cmd}} = \mathbf{a}_{\text{formation}} + \hat{\mathbf{a}}_{\text{target}} + k_v (\hat{\mathbf{v}}_{\text{target}} - \mathbf{v}_{\text{drone}})$$
   This minimizes lag when the target decelerates or makes sharp turns.
