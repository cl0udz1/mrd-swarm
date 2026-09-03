# METRICS_SPEC.md: Formal Mathematical Metric Definitions & Evaluation Criteria

**System**: MRD-SWARM Autonomous Multi-UAV Architecture  
**Standard**: Scientific Rigor, Empirical Falsifiability & Statistical Reproducibility  
**Version**: 2.0 (Post-Audit Reconstruction)  

---

## 1. Ground Truth vs. Estimation Boundary

All evaluation metrics are computed offline from recorded telemetry.
- **Estimated states** ($\hat{\mathbf{x}}_k, \mathbf{P}_k$) are the internal outputs of onboard filters (`KalmanTargetTracker`).
- **Ground truth states** ($\mathbf{x}_{\text{true}, k}$) are strictly emitted by the simulation environment for validation and are **never** accessible to onboard controllers, navigators, or task allocators.

---

## 2. Formal Metric Definitions

### 2.1 Time-to-Intercept (TTI)

Given ground target planar trajectory $\mathbf{p}_T(t) \in \mathbb{R}^2$ and drone positions $\mathbf{p}_{D, i}(t) \in \mathbb{R}^2$, Time-to-Intercept is the earliest simulation time $t^* \ge 0$ such that for a continuous holding window $\tau \in [t^*, t^* + \Delta t_{\text{hold}}]$ with $\Delta t_{\text{hold}} = 1.5\text{ s}$:

1. At least two active combat drones $i, j \in \{0, 1, 2\}$ satisfy the standoff distance condition:
   $$\|\mathbf{p}_{D, i}(\tau) - \mathbf{p}_T(\tau)\| \le R_{\text{standoff}}, \quad R_{\text{standoff}} = 6.0\text{ m}$$
2. The angular enclosure condition is satisfied:
   $$\Delta\theta_{ij}(\tau) = |\text{wrap}_{[-\pi, \pi]}(\theta_i(\tau) - \theta_j(\tau))| \ge \theta_{\min}, \quad \theta_{\min} = 60^\circ$$
   where $\theta_i(\tau) = \text{atan2}(p_{D, i, y}(\tau) - p_{T, y}(\tau), p_{D, i, x}(\tau) - p_{T, x}(\tau))$.

$$\text{TTI} = \begin{cases} t^* & \text{if } \exists t^* \text{ satisfying (1) and (2) for } \tau \in [t^*, t^* + 1.5\text{s}] \\ \text{NOT\_OBSERVED} & \text{otherwise} \end{cases}$$

> [!CAUTION]
> **Anti-Tampering Rule**: Under no circumstances may TTI be set to an arbitrary default (e.g. simulation duration) if interception was not achieved. If interception fails, the trial reports `NOT_OBSERVED` and counts as a failed interception in statistical success rates.

---

### 2.2 Track Maintenance Ratio (TMR)

The proportion of mission duration during which a ground target is actively tracked with confirmed Kalman Filter confidence and low spatial error:

$$\text{TMR} = \frac{1}{K} \sum_{k=1}^K \mathbb{I}\left(\text{state}_k == \text{CONFIRMED} \;\land\; \|\hat{\mathbf{p}}_k - \mathbf{p}_{\text{true}, k}\| \le 1.5\text{ m}\right) \times 100\%$$

where $K$ is the total number of simulation steps after initial target acquisition, and $\mathbb{I}(\cdot)$ is the indicator function.

---

### 2.3 State Estimation Accuracy (RMSE & NEES)

Target position and velocity estimation errors over confirmed track steps:

$$\text{RMSE}_{\text{pos}} = \sqrt{\frac{1}{M} \sum_{m=1}^M \|\hat{\mathbf{p}}_m - \mathbf{p}_{\text{true}, m}\|^2}$$

$$\text{RMSE}_{\text{vel}} = \sqrt{\frac{1}{M} \sum_{m=1}^M \|\hat{\mathbf{v}}_m - \mathbf{v}_{\text{true}, m}\|^2}$$

**Normalized Estimation Error Squared (NEES)**:
Evaluates filter consistency:
$$\epsilon_k = (\mathbf{x}_{\text{true}, k} - \hat{\mathbf{x}}_k)^T \mathbf{P}_k^{-1} (\mathbf{x}_{\text{true}, k} - \hat{\mathbf{x}}_k)$$
For a 4-state constant-velocity filter, theoretical consistency requires $\mathbb{E}[\epsilon_k] \approx 4.0$.

---

### 2.4 Epistemic Voxel Uncertainty & Exploration

Let $\mathcal{V}_{\text{free}}$ be the set of all free-space voxels (excluding solid building interiors):

$$U_{\text{mean}}(t) = \frac{1}{|\mathcal{V}_{\text{free}}|} \sum_{\mathbf{v} \in \mathcal{V}_{\text{free}}} U(\mathbf{v}, t) \times 100\%$$

$$\Delta U = \frac{U_{\text{mean}}(0) - U_{\text{mean}}(t_{\text{final}})}{U_{\text{mean}}(0)} \times 100\%$$

**Time to 90% Coverage ($T_{90}$)**:
The earliest timestamp $t$ such that the percentage of free-space voxels with $U(\mathbf{v}, t) < 0.15$ exceeds $90.0\%$. If $90\%$ is not reached, $T_{90} = \text{NOT\_REACHED}$.

---

### 2.5 Flight Control Tracking Precision & Actuator Saturation

For each UAV $i$:

$$\text{RMSE}_{\text{pos}, i} = \sqrt{\frac{1}{N} \sum_{k=1}^N \|\mathbf{p}_{i, k} - \mathbf{p}_{\text{des}, i, k}\|^2}$$

$$\text{Saturation Frequency } S_i = \frac{N_{\text{saturated}, i}}{N_{\text{total}}} \times 100\%$$

where $N_{\text{saturated}}$ is the count of control steps where at least one motor thrust reached $T_j = 0$ or $T_j = T_{\max}$.

---

### 2.6 Network Algebraic Connectivity Under Electronic Warfare

Given swarm ad-hoc mesh adjacency matrix $A \in \mathbb{R}^{4 \times 4}$ and degree matrix $D = \text{diag}(\sum_j A_{ij})$, the graph Laplacian is:
$$L = D - A$$
The Fiedler eigenvalue $\lambda_2(L)$ is the second-smallest eigenvalue of $L$.

$$\text{Connectivity Retention} = \frac{\lambda_2(L_{\text{jammed}})}{\lambda_2(L_{\text{nominal}})} \times 100\%$$

---

### 2.7 Energy Expenditure

Total energy consumed across all 4 airframes from battery discharge models:

$$E_{\text{fleet}} = \sum_{i=0}^3 \Delta E_i \quad [\text{Wh}]$$

---

## 3. Strict Verification & Falsification Thresholds

Every metric evaluation in benchmark reports and CI testing must be computed via boolean logic. Hardcoded `"PASS"` strings are strictly prohibited.

```python
# Authoritative evaluation logic:
is_pass = bool(measured_value <= threshold_value) if lower_is_better else bool(measured_value >= threshold_value)
status_string = "PASS" if is_pass else "FAIL"
```

| Metric | Target Requirement | Evaluation Operator | Failure Mode Tag |
|---|---|:---:|---|
| **Time-to-90% Coverage ($T_{90}$)** | $< 18.0\text{ s}$ | $\le$ | `SLOW_EXPLORATION` |
| **Total Uncertainty Reduction ($\Delta U$)** | $> 75.0\%$ | $\ge$ | `INSUFFICIENT_COVERAGE` |
| **Track Maintenance Ratio (TMR)** | $\ge 70.0\%$ | $\ge$ | `TRACK_LOSS` |
| **Position Estimation RMSE** | $< 1.20\text{ m}$ | $\le$ | `ESTIMATION_DIVERGENCE` |
| **SE(3) Flight Tracking RMSE** | $< 0.85\text{ m}$ | $\le$ | `CONTROL_INSTABILITY` |
| **Actuator Saturation Frequency** | $< 15.0\%$ | $\le$ | `ACTUATOR_OVERDRIVE` |
| **Network Retention under EW** | $\ge 50.0\%$ | $\ge$ | `MESH_DISCONNECTION` |
| **Fleet PNR Battery Margin** | Final $\text{SoC} \ge 15.0\%$ for all | $\ge$ | `BATTERY_DEPLETION` |
