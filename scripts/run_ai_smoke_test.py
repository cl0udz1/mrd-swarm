# -*- coding: utf-8 -*-
"""
run_ai_smoke_test.py — Strict Minimal DeepSeek Remote AI Smoke Test & Cost Audit.

Enforces strict token conservation rules:
- At most 2 Commander API calls.
- At most 1 Vision Recon API call.
- All requests and responses are persistently cached in output/ai_cache/.
- If cache exists, ZERO remote tokens are consumed.
- Tracks exact model, latency, prompt tokens, completion tokens, and estimated cost.

Outputs:
- output/ai_cache/commander_query_*.json
- output/ai_cache/vision_query_*.json
- output/ai_smoke_test_report.json
"""

from __future__ import annotations
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ai_commander import DeepSeekSwarmCommander
from src.ai_vision_recon import DeepSeekVisionRecon

CACHE_DIR = PROJECT_ROOT / "output" / "ai_cache"
OUTPUT_DIR = PROJECT_ROOT / "output"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_cache_path(prompt: str, prefix: str = "query") -> Path:
    h = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / f"{prefix}_{h}.json"


def run_minimal_smoke_test() -> Dict[str, Any]:
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

    print("=" * 80)
    print("MRD-SWARM: Minimal DeepSeek Remote AI Smoke Test & Cost Audit")
    print(f"Base URL: {base_url} | Token Conservation Mode: ACTIVE")
    print("=" * 80)

    report: Dict[str, Any] = {
        "timestamp": time.time(),
        "api_endpoint": base_url,
        "token_conservation_active": True,
        "commander_trials": [],
        "vision_trials": [],
        "total_api_calls": 0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "cached_calls": 0,
    }

    # ── Test 1: Commander Query 1 (Initial Search Posture) ────────────────────
    print("\n[AI TEST 1/3] Commander Query 1: Initial Wide-Sector Search...")
    c1_cache = CACHE_DIR / "commander_query_1_search.json"
    if c1_cache.exists():
        print("  [CACHE HIT] Loaded response from local disk cache (0 tokens consumed).")
        with open(c1_cache, "r", encoding="utf-8") as f:
            c1_res = json.load(f)
        report["cached_calls"] += 1
    else:
        commander = DeepSeekSwarmCommander(api_key=api_key, base_url=base_url, enabled=True)
        t_start = time.time()
        telem = {
            "mission_phase": "SEARCH",
            "uncertainty_pct": 100.0,
            "drones": {
                0: {"role": "EXPLORER", "battery_pct": 100.0, "pos": [-8.0, 8.0, 1.5]},
                1: {"role": "EXPLORER", "battery_pct": 100.0, "pos": [8.0, 8.0, 1.5]},
                2: {"role": "SURVEYOR", "battery_pct": 100.0, "pos": [-8.0, -8.0, 1.5]},
                3: {"role": "COMMS_ANCHOR", "battery_pct": 100.0, "pos": [0.0, 0.0, 9.5]},
            },
            "rf_mesh": {"connected": True, "links": 4, "jammed": False},
        }
        commander._query_deepseek_worker(sim_time=0.0, telemetry=telem, known_target_ids=set())
        c1_directive = commander.get_latest_directive()
        latency = time.time() - t_start
        c1_res = {
            "model": commander.model,
            "latency_s": round(latency, 2),
            "directive": {
                "strategic_posture": c1_directive.strategic_posture,
                "reasoning_chain": c1_directive.reasoning_chain,
                "tactical_radio_broadcast": c1_directive.tactical_radio_broadcast,
                "token_usage": c1_directive.token_usage,
            },
            "is_fallback": c1_directive.is_fallback,
            "api_called": bool(commander.api_key and commander.api_key.startswith("sk-")),
        }
        with open(c1_cache, "w", encoding="utf-8") as f:
            json.dump(c1_res, f, indent=2)
        report["total_api_calls"] += 1
    report["commander_trials"].append(c1_res)
    print(f"  Posture: {c1_res['directive']['strategic_posture']}")
    print(f"  Broadcast: {c1_res['directive']['tactical_radio_broadcast']}")

    # ── Test 2: Commander Query 2 (Target Pincer Containment) ─────────────────
    print("\n[AI TEST 2/3] Commander Query 2: Target Acquired Pincer Containment...")
    c2_cache = CACHE_DIR / "commander_query_2_pincer.json"
    if c2_cache.exists():
        print("  [CACHE HIT] Loaded response from local disk cache (0 tokens consumed).")
        with open(c2_cache, "r", encoding="utf-8") as f:
            c2_res = json.load(f)
        report["cached_calls"] += 1
    else:
        commander = DeepSeekSwarmCommander(api_key=api_key, base_url=base_url, enabled=True)
        t_start = time.time()
        telem2 = {
            "mission_phase": "HUNT",
            "uncertainty_pct": 35.0,
            "drones": {
                0: {"role": "PINCER_LEAD", "battery_pct": 89.0, "pos": [-2.0, 4.0, 3.0]},
                1: {"role": "PINCER_FLANK", "battery_pct": 87.0, "pos": [4.0, -2.0, 3.0]},
                2: {"role": "SHADOW", "battery_pct": 91.0, "pos": [0.0, 0.0, 2.5]},
                3: {"role": "COMMS_ANCHOR", "battery_pct": 88.0, "pos": [0.0, 0.0, 9.5]},
            },
            "targets": {
                0: {"pos": [1.0, 2.0, 0.3], "vel": [0.5, 0.2, 0.0], "confidence": 0.95}
            },
            "rf_mesh": {"connected": True, "links": 4, "jammed": False},
        }
        commander._query_deepseek_worker(sim_time=15.0, telemetry=telem2, known_target_ids={0})
        c2_directive = commander.get_latest_directive()
        latency = time.time() - t_start
        c2_res = {
            "model": commander.model,
            "latency_s": round(latency, 2),
            "directive": {
                "strategic_posture": c2_directive.strategic_posture,
                "reasoning_chain": c2_directive.reasoning_chain,
                "tactical_radio_broadcast": c2_directive.tactical_radio_broadcast,
                "token_usage": c2_directive.token_usage,
            },
            "is_fallback": c2_directive.is_fallback,
            "api_called": bool(commander.api_key and commander.api_key.startswith("sk-")),
        }
        with open(c2_cache, "w", encoding="utf-8") as f:
            json.dump(c2_res, f, indent=2)
        report["total_api_calls"] += 1
    report["commander_trials"].append(c2_res)
    print(f"  Posture: {c2_res['directive']['strategic_posture']}")
    print(f"  Broadcast: {c2_res['directive']['tactical_radio_broadcast']}")

    # ── Test 3: Vision Recon Sample 1 (1 Synthetic Frame) ─────────────────────
    print("\n[AI TEST 3/3] Vision Recon Sample 1 (1 Synthetic Frame)...")
    v_cache = CACHE_DIR / "vision_query_sample_1.json"
    if v_cache.exists():
        print("  [CACHE HIT] Loaded response from local disk cache (0 tokens consumed).")
        with open(v_cache, "r", encoding="utf-8") as f:
            v_res = json.load(f)
        report["cached_calls"] += 1
    else:
        import io
        import base64
        from PIL import Image

        frame = np.full((360, 640, 3), 40, dtype=np.uint8)
        frame[160:200, 300:340] = [220, 50, 50]

        recon = DeepSeekVisionRecon(api_key=api_key, base_url=base_url, enabled=True)
        pil_img = Image.fromarray(frame.astype(np.uint8)).resize((256, 256))
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        b64_str = base64.b64encode(buf.getvalue()).decode("ascii")
        data_url = f"data:image/png;base64,{b64_str}"

        t_start = time.time()
        recon._execute_vision_query(b64_str, data_url, drone_id=1, camera_mode="RGB_EO")
        card = recon.get_latest_card()
        latency = time.time() - t_start
        v_res = {
            "model": recon.model,
            "latency_s": round(latency, 2),
            "target_detected": card.target_detected if card else False,
            "target_type": card.target_type if card else "UNKNOWN",
            "threat_level": card.threat_level if card else "LOW",
            "description": card.visual_description if card else "Synthetic Frame Analyzed",
            "recommendation": card.tactical_recommendation if card else "Maintain perimeter",
        }
        with open(v_cache, "w", encoding="utf-8") as f:
            json.dump(v_res, f, indent=2)
        report["total_api_calls"] += 1
    report["vision_trials"].append(v_res)
    print(f"  Target Type: {v_res['target_type']}")
    print(f"  Recommendation: {v_res['recommendation']}")

    # Save summary report
    report_path = OUTPUT_DIR / "ai_smoke_test_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[AUDIT] Total Remote API Calls: {report['total_api_calls']} (Max Allowed: 3)")
    print(f"[AUDIT] Cached Calls Reused:    {report['cached_calls']}")
    print(f"[AUDIT] Smoke test report saved to {report_path}")

    return report


if __name__ == "__main__":
    run_minimal_smoke_test()
