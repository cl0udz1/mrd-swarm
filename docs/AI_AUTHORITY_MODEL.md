# AI Authority Model: Cognitive vs. Deterministic Control Boundaries

**System**: MRD-SWARM Autonomous Multi-UAV Architecture  
**Document**: `docs/AI_AUTHORITY_MODEL.md`  
**Standard**: Real-Time Safety-Critical Swarm Systems  

---

## 1. Architectural Principle

In the MRD-Swarm system, Large Language Models (`deepseek-v4-flash`) and Multimodal Vision Models (`deepseek-v4-flash-vision-exp`) are classified as **Non-Real-Time Cognitive Advisory Layers**. 

Under no circumstances may an unvalidated output from a cloud-based API directly dictate motor voltages, bypass obstacle collision avoidance, or violate physical aerodynamic and battery constraints.

```
┌──────────────────────────────────────────────────────────────────┐
│                   DEEPSEEK COGNITIVE LAYER                      │
│   (deepseek-v4-flash LLM + deepseek-v4-flash-vision-exp MLLM)   │
└─────────────────────────────────┬────────────────────────────────┘
                                  │ (Structured Proposal JSON)
                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│            DETERMINISTIC SAFETY & CAPABILITY VALIDATOR           │
│   - Schema Validation & Type Sanitization                        │
│   - Airframe Kinematic Envelope Clamping (v <= v_max, i)         │
│   - Role Compatibility Verification (Sensor payload check)       │
│   - Target ID Hallucination Pruning                              │
└─────────────────────────────────┬────────────────────────────────┘
                                  │ (Validated Tactical Proposals)
                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│            DETERMINISTIC TACTICAL PLANNER & DECISION ECS         │
│   - 6-Phase Mission State Machine                                │
│   - Coordinated Pincer Geometry Solver                           │
│   - Capability-Weighted Utility Task Allocator                   │
│   - Kalman Filter State Estimator                                │
└─────────────────────────────────┬────────────────────────────────┘
                                  │ (Goal Waypoints & Desired Velocities)
                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│                 3D ARTIFICIAL POTENTIAL FIELD (APF)              │
│   - Hard Obstacle Repulsion (Skyscrapers & Buildings)            │
│   - Inter-UAV Peer Collision Avoidance                           │
└─────────────────────────────────┬────────────────────────────────┘
                                  │ (Collision-Free Acceleration Vector)
                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│             SO(3) GEOMETRIC CONTROLLER & MOTOR MIXER             │
│   - Attitude Error & Gyroscopic Cross-Coupling Torque            │
│   - 4-Rotor Thrust Allocation & Actuator Saturation Clamping     │
└─────────────────────────────────┬────────────────────────────────┘
                                  │ (Individual Motor Commands)
                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│                      VEHICLE 6-DoF DYNAMICS                      │
└──────────────────────────────────────────────────────────────────┘
```

---

## 2. Field-by-Field Authority Matrix

Every field produced by the DeepSeek AI Commander has an explicit, mathematically defined level of control authority:

| Generated Field | Type | Authority Level | Enforcement Mechanism | Failure / Fallback Behavior |
|---|---|:---:|---|---|
| **`strategic_posture`** | Enum string | **DIRECTLY EXECUTED** (via Preset Mapping) | Maps to one of 4 formal tactical doctrines (`AGGRESSIVE_PINCER`, `WOLFPACK_CONTAINMENT`, `STEALTH_SHADOW`, `DEEPSEEK_ADAPTIVE`). Directly configures standoff radius, flanker sprint speed, and pincer angle. | If unrecognized, falls back to `COORDINATED_SWEEP` or current active doctrine. |
| **`target_priority`** | List of ints | **VALIDATED ADVISORY** | Target IDs are checked against confirmed Kalman filter tracks. Hallucinated or deleted IDs are pruned. Validated priority list injects up to +0.40 additive utility bonus for highest-priority target. | If empty or invalid, planner prioritizes by target speed, boundary urgency, and track confidence. |
| **`drone_assignments`** | Dict of role dicts | **VALIDATED ADVISORY** | Validates that proposed roles match airframe capabilities (e.g. only Drone 2 assigned thermal pursuit; Drone 3 reserved for Relay). Speed is clamped to $v_{\text{des}} = \min(v_{\text{proposed}}, v_{\max, i})$. | If incompatible, utility-based task allocator overrides with capability-optimal assignment. |
| **`tactical_radio_broadcast`** | String | **INFORMATIONAL ONLY** | Broadcast string is streamed to the live visualizer HUD marquee and recorded in telemetry logs. Zero impact on vehicle kinematics. | Default radio broadcast emitted. |
| **`reasoning_chain`** | String (CoT) | **INFORMATIONAL ONLY** | Extracted from DeepSeek's `reasoning_content` tokens; streamed to the WebGL tactical terminal for operator explainability. | Logged to flight records. |

---

## 3. Multimodal Vision Integration (Closed-Loop)

The Multimodal Vision Agent (`deepseek-v4-flash-vision-exp`) processes 256x256 rendered FPV camera frames:

1. **Optical Smoke Detection**: If `smoke_detected == True` with confidence $\ge 0.70$, the perception pipeline sets `smoke_active = True` for the target in the field of view. This immediately penalizes optical tracking and forces Drone 2 (Thermal Surveyor) to take over tracking.
2. **Visual Target Classification**: If a candidate vehicle is classified as `HIGH_VALUE_VEHICLE` with threat level `HIGH` or `CRITICAL`, its threat priority score in the mission planner is boosted by $+0.50$.
3. **Data Provenance & Staleness**: Every vision observation is tagged with a sensor timestamp. Vision observations older than $4.0\text{s}$ are expired and cannot override local range/bearing sensors.
