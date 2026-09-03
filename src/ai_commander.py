# -*- coding: utf-8 -*-
"""
ai_commander.py — DeepSeek AI Swarm Tactical Commander & Safety Validator

Provides asynchronous, non-blocking tactical reasoning powered by deepseek-v4-flash.
Enforces the formal AI Authority Model (docs/AI_AUTHORITY_MODEL.md):
    LLM Proposal → Schema Validation → Kinematic Clamping → Validated Tactical Directive
"""

from __future__ import annotations
import os
import time
import json
import urllib.request
import urllib.error
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from pathlib import Path

import numpy as np
from .config.airframes import FLEET_CONFIGS


def _load_env() -> None:
    """Load credentials from repository-root .env."""
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


_load_env()

DEFAULT_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEFAULT_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEFAULT_MODEL = os.environ.get("DEEPSEEK_COMMANDER_MODEL", "deepseek-v4-flash")


@dataclass
class ValidatedDirective:
    """
    A sanitized, bounded tactical directive ready for deterministic execution.
    """
    timestamp: float
    sim_time: float
    strategic_posture: str
    target_priority: List[int]
    drone_assignments: Dict[int, Dict[str, Any]]
    tactical_radio_broadcast: str
    reasoning_chain: str
    model: str
    latency_s: float
    token_usage: Dict[str, int]
    is_fallback: bool = False
    operator_override: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": round(self.timestamp, 3),
            "sim_time": round(self.sim_time, 2),
            "strategic_posture": self.strategic_posture,
            "target_priority": self.target_priority,
            "drone_assignments": self.drone_assignments,
            "tactical_radio_broadcast": self.tactical_radio_broadcast,
            "reasoning_chain": self.reasoning_chain,
            "model": self.model,
            "latency_s": round(self.latency_s, 2),
            "token_usage": self.token_usage,
            "is_fallback": self.is_fallback,
            "operator_override": self.operator_override,
        }


# Backwards compatibility alias
TacticalDirective = ValidatedDirective


SYSTEM_PROMPT = """You are the Autonomous Multi-Agent Drone Swarm Tactical Commander for Operation MRD-Swarm.
You command 4 heterogeneous quadrotors in a dense 3D urban environment:
- Drone 0 (Heavy Scout): Long-range optics, wide search envelope.
- Drone 1 (Fast Interceptor): 18 m/s sprint speed, pincer cutoffs.
- Drone 2 (Thermal Surveyor): FLIR sensor, penetrates target smoke.
- Drone 3 (Comms Relay): High-altitude mesh anchor (Z=10.5m), overcomes EW jamming.

TACTICAL RULES:
1. If smoke is active on a target, assign Thermal Surveyor (D2) to TRACKER.
2. If EW jamming active, ensure Comms Relay (D3) maintains altitude centroid.
3. Fast Interceptor (D1) flanks high-priority moving HVTs to cut off escape corridors.

Be concise in your military reasoning. Always output valid JSON only:
{
  "strategic_posture": "CONCENTRIC_CONTAINMENT" | "AGGRESSIVE_PINCER" | "COORDINATED_SWEEP" | "THERMAL_IR_PURSUIT" | "RF_RELAY_PRESERVATION",
  "target_priority": [target_id, ...],
  "drone_assignments": {
    "0": {"role": "EXPLORER" | "TRACKER" | "FLANKER", "target_id": 0, "desired_speed": 10.0, "tactic": "..."},
    "1": {"role": "FLANKER" | "TRACKER", "target_id": 0, "desired_speed": 16.0, "tactic": "..."},
    "2": {"role": "TRACKER" | "EXPLORER", "target_id": 0, "desired_speed": 12.0, "tactic": "..."},
    "3": {"role": "RELAY", "target_id": null, "desired_speed": 5.5, "tactic": "Maintain altitude anchor"}
  },
  "tactical_radio_broadcast": "Military radio transmission string"
}"""


def sanitize_directive(
    raw_data: Dict[str, Any],
    sim_time: float,
    known_target_ids: Set[int],
    reasoning: str = "",
    latency_s: float = 0.0,
    model: str = "deepseek-v4-flash",
    tokens: Optional[Dict[str, int]] = None,
) -> ValidatedDirective:
    """
    Validates and clamps raw LLM JSON output against airframe kinematic envelopes and known targets.
    """
    # 1. Validate Strategic Posture
    valid_postures = {
        "CONCENTRIC_CONTAINMENT", "AGGRESSIVE_PINCER", "COORDINATED_SWEEP",
        "THERMAL_IR_PURSUIT", "RF_RELAY_PRESERVATION", "STEALTH_SHADOW"
    }
    raw_posture = str(raw_data.get("strategic_posture", "COORDINATED_SWEEP")).upper().strip()
    matched_posture = "COORDINATED_SWEEP"
    for vp in valid_postures:
        if vp in raw_posture:
            matched_posture = vp
            break

    # 2. Validate & Prune Target Priorities
    raw_priorities = raw_data.get("target_priority", [])
    valid_priorities: List[int] = []
    if isinstance(raw_priorities, list):
        for tid in raw_priorities:
            if isinstance(tid, (int, float)) and int(tid) in known_target_ids:
                valid_priorities.append(int(tid))
    if not valid_priorities and known_target_ids:
        valid_priorities = sorted(list(known_target_ids))

    # 3. Validate Drone Assignments & Clamp Speeds
    raw_assignments = raw_data.get("drone_assignments", {})
    sanitized_assignments: Dict[int, Dict[str, Any]] = {}
    valid_roles = {"EXPLORER", "TRACKER", "FLANKER", "RELAY", "LOST_TARGET_SWEEP", "RTB_RECOVERY"}

    for did in range(4):
        airframe = FLEET_CONFIGS[did]
        d_cfg = raw_assignments.get(str(did), raw_assignments.get(did, {}))

        role = str(d_cfg.get("role", "EXPLORER")).upper().strip()
        if role not in valid_roles:
            role = "RELAY" if did == 3 else ("FLANKER" if did == 1 else "EXPLORER")

        # Inforce Comms Relay invariant
        if did == 3:
            role = "RELAY"

        # Clamp speed to airframe max capability
        proposed_speed = float(d_cfg.get("desired_speed", airframe.max_speed_mps * 0.7))
        clamped_speed = float(np.clip(proposed_speed, 1.0, airframe.max_speed_mps))

        # Validate target ID
        raw_tid = d_cfg.get("target_id")
        tid_val = int(raw_tid) if (raw_tid is not None and int(raw_tid) in known_target_ids) else None

        sanitized_assignments[did] = {
            "role": role,
            "target_id": tid_val,
            "desired_speed": clamped_speed,
            "tactic": str(d_cfg.get("tactic", "Execute assigned patrol sector")),
        }

    radio_msg = str(raw_data.get("tactical_radio_broadcast", "Swarm Actual: Maintain tactical perimeter. Over."))

    return ValidatedDirective(
        timestamp=time.time(),
        sim_time=sim_time,
        strategic_posture=matched_posture,
        target_priority=valid_priorities,
        drone_assignments=sanitized_assignments,
        tactical_radio_broadcast=radio_msg,
        reasoning_chain=reasoning,
        model=model,
        latency_s=latency_s,
        token_usage=tokens or {},
        is_fallback=False,
    )


class DeepSeekSwarmCommander:
    """
    Non-blocking Tactical AI Commander with background thread execution
    and strict safety validation.
    """

    def __init__(
        self,
        api_key: str = DEFAULT_API_KEY,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        update_interval_s: float = 3.0,
        request_timeout_s: float = 8.0,
        enabled: Optional[bool] = None,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.update_interval_s = update_interval_s
        self.timeout_s = request_timeout_s
        if enabled is not None:
            self.enabled = enabled
        else:
            self.enabled = os.environ.get("ENABLE_REMOTE_AI", "0") in ("1", "true", "True") and bool(self.api_key and self.api_key.startswith("sk-"))

        self.last_query_time = -999.0
        self.is_querying = False
        self.lock = threading.Lock()

        # Initialize default deterministic fallback directive
        self.latest_directive = self._make_default_directive(0.0)
        self.directive_history: List[ValidatedDirective] = [self.latest_directive]
        self.operator_order_queue: List[str] = []

    def _make_default_directive(self, sim_time: float) -> ValidatedDirective:
        """Deterministic safety fallback directive."""
        assignments = {
            0: {"role": "EXPLORER", "target_id": 0, "desired_speed": 10.0, "tactic": "Frontier exploration"},
            1: {"role": "FLANKER", "target_id": 0, "desired_speed": 15.0, "tactic": "Corridor interception"},
            2: {"role": "TRACKER", "target_id": 0, "desired_speed": 12.0, "tactic": "Continuous optical tracking"},
            3: {"role": "RELAY", "target_id": None, "desired_speed": 5.5, "tactic": "Maintain altitude anchor"},
        }
        return ValidatedDirective(
            timestamp=time.time(),
            sim_time=sim_time,
            strategic_posture="COORDINATED_SWEEP",
            target_priority=[0, 1, 2],
            drone_assignments=assignments,
            tactical_radio_broadcast="Falcon-Lead: Area sweep initiated. All elements maintain mesh spacing.",
            reasoning_chain="Deterministic rule-based baseline: establishing perimeter search pattern.",
            model="deterministic-fallback",
            latency_s=0.0,
            token_usage={},
            is_fallback=True,
        )

    def queue_operator_order(self, natural_language_order: str) -> None:
        """Inject human operator command into next evaluation prompt."""
        with self.lock:
            self.operator_order_queue.append(natural_language_order)

    def get_latest_directive(self) -> ValidatedDirective:
        with self.lock:
            return self.latest_directive

    def request_tactical_evaluation(
        self,
        sim_time: float,
        telemetry: Dict[str, Any],
        known_target_ids: Set[int],
    ) -> None:
        """Asynchronously trigger evaluation if update interval elapsed."""
        if not self.enabled:
            return

        if sim_time - self.last_query_time < self.update_interval_s:
            return

        with self.lock:
            if self.is_querying:
                return
            self.is_querying = True

        self.last_query_time = sim_time
        worker = threading.Thread(
            target=self._query_deepseek_worker,
            args=(sim_time, telemetry, known_target_ids),
            daemon=True,
        )
        worker.start()

    def _query_deepseek_worker(
        self,
        sim_time: float,
        telemetry: Dict[str, Any],
        known_target_ids: Set[int],
    ) -> None:
        t_start = time.time()
        try:
            with self.lock:
                orders = list(self.operator_order_queue)
                self.operator_order_queue.clear()

            user_prompt = f"CURRENT BATTLEFIELD TELEMETRY at t={sim_time:.1f}s:\n"
            user_prompt += json.dumps(telemetry, indent=2)
            if orders:
                user_prompt += f"\n\nHUMAN OPERATOR ORDERS:\n" + "\n".join(f"- {o}" for o in orders)

            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.2,
                "max_tokens": 800,
            }

            req = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=self.timeout_s) as response:
                res_data = json.loads(response.read().decode("utf-8"))

            msg_obj = res_data["choices"][0]["message"]
            raw_content = msg_obj.get("content", "{}")
            reasoning = msg_obj.get("reasoning_content", "") or msg_obj.get("reasoning", "")
            tokens = res_data.get("usage", {})
            parsed_json = json.loads(raw_content)

            validated = sanitize_directive(
                raw_data=parsed_json,
                sim_time=sim_time,
                known_target_ids=known_target_ids,
                reasoning=reasoning,
                latency_s=time.time() - t_start,
                model=self.model,
                tokens=tokens,
            )

            with self.lock:
                self.latest_directive = validated
                self.directive_history.append(validated)

        except Exception as e:
            # Deterministic fallback on API error or timeout
            print(f"[AI COMMANDER WARNING] Remote query failed ({time.time() - t_start:.2f}s): {e}")
        finally:
            with self.lock:
                self.is_querying = False
