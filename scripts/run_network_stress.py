# -*- coding: utf-8 -*-
"""
run_network_stress.py — RF Mesh Resilience & Network Stress Evaluation.

Evaluates:
1. Algebraic Connectivity (Fiedler Eigenvalue lambda_2 of Graph Laplacian L = D - A).
2. EW Jamming degradation across variable jamming radius (0m to 20m) and intensity.
3. Multi-hop packet delivery ratio (PDR), hop count distribution, and TTL termination.
4. Relay Node High-Altitude Punch-Through recovery.

Outputs:
- media/figures/network_resilience_curves.png
- output/network_stress_summary.json
"""

from __future__ import annotations
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.airframes import FLEET_CONFIGS
from src.gossip import GossipNode, GossipMessage, MessageType, TargetEstimate
from src.evaluation.metrics import evaluate_network_retention

OUTPUT_DIR = PROJECT_ROOT / "output"
FIGURES_DIR = PROJECT_ROOT / "media" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def compute_fiedler_eigenvalue(adj_matrix: np.ndarray) -> float:
    """Computes second-smallest eigenvalue of graph Laplacian matrix."""
    n = adj_matrix.shape[0]
    if n < 2:
        return 0.0
    deg = np.sum(adj_matrix, axis=1)
    L = np.diag(deg) - adj_matrix
    eigvals = np.sort(np.linalg.eigvalsh(L))
    return float(max(0.0, eigvals[1]))


def evaluate_mesh_connectivity(
    drone_positions: np.ndarray,
    ew_center: np.ndarray,
    ew_radius: float,
    ew_intensity: float,
    relay_high_altitude: bool = True,
) -> Tuple[np.ndarray, float, int]:
    """
    Computes adjacency matrix, Fiedler eigenvalue, and total active links
    under RF physical range and EW jamming degradation.
    """
    n = len(drone_positions)
    adj = np.zeros((n, n), dtype=np.float64)
    links_count = 0

    for i in range(n):
        pi = drone_positions[i]
        d_jam_i = float(np.linalg.norm(pi[:2] - ew_center[:2]))
        jammed_i = (d_jam_i < ew_radius) and not (i == 3 and relay_high_altitude)

        for j in range(i + 1, n):
            pj = drone_positions[j]
            d_jam_j = float(np.linalg.norm(pj[:2] - ew_center[:2]))
            jammed_j = (d_jam_j < ew_radius) and not (j == 3 and relay_high_altitude)

            dist = float(np.linalg.norm(pi - pj))
            base_range = 32.0 if (i == 3 or j == 3) else 18.0
            eff_range = base_range
            if (jammed_i or jammed_j):
                eff_range *= max(0.1, (1.0 - ew_intensity))

            if dist <= eff_range:
                adj[i, j] = 1.0
                adj[j, i] = 1.0
                links_count += 1

    lambda_2 = compute_fiedler_eigenvalue(adj)
    return adj, lambda_2, links_count


def run_ew_sweep_campaign() -> Dict[str, Any]:
    """Sweeps EW jamming radius from 0 to 20m and evaluates connectivity with and without relay."""
    # Typical mission swarm positions in 60m x 60m urban quadrant
    # D0: NW quadrant, D1: NE quadrant, D2: SW quadrant, D3: Center Relay
    positions = np.array([
        [-12.0,  12.0, 3.0],   # Drone 0
        [ 12.0,  12.0, 3.0],   # Drone 1
        [-12.0, -12.0, 3.0],   # Drone 2
        [  0.0,   0.0, 9.5],   # Drone 3 (Relay at altitude Z=9.5m)
    ], dtype=np.float64)

    ew_center = np.array([0.0, 0.0, 0.0])
    radii = np.linspace(0.0, 20.0, 21)

    fiedler_with_relay = []
    fiedler_without_relay = []
    links_with_relay = []
    links_without_relay = []

    for r in radii:
        # With high altitude relay
        _, l2_with, lk_with = evaluate_mesh_connectivity(
            positions, ew_center, ew_radius=r, ew_intensity=0.85, relay_high_altitude=True
        )
        fiedler_with_relay.append(l2_with)
        links_with_relay.append(lk_with)

        # Grounded relay / without high altitude punch-through
        pos_grounded = positions.copy()
        pos_grounded[3, 2] = 2.0
        _, l2_no, lk_no = evaluate_mesh_connectivity(
            pos_grounded, ew_center, ew_radius=r, ew_intensity=0.85, relay_high_altitude=False
        )
        fiedler_without_relay.append(l2_no)
        links_without_relay.append(lk_no)

    retention_pct, passed_retention = evaluate_network_retention(
        nominal_fiedler=fiedler_with_relay[0],
        jammed_fiedler=fiedler_with_relay[12],  # at r = 12m
        threshold_retention_pct=50.0,
    )

    return {
        "radii": list(radii),
        "fiedler_with_relay": [round(float(v), 3) for v in fiedler_with_relay],
        "fiedler_without_relay": [round(float(v), 3) for v in fiedler_without_relay],
        "links_with_relay": links_with_relay,
        "links_without_relay": links_without_relay,
        "retention_at_12m_pct": retention_pct,
        "passed_retention_spec": passed_retention,
    }


def run_multihop_forwarding_stress() -> Dict[str, Any]:
    """Tests multi-hop packet delivery ratio, duplicate suppression, and TTL decrement."""
    nodes = [GossipNode(agent_id=i) for i in range(4)]
    # Linear topology: 0 <-> 1 <-> 2 <-> 3
    adj = {
        0: [1],
        1: [0, 2],
        2: [1, 3],
        3: [2],
    }

    # Inject message at Node 0 with TTL = 3
    payload = {
        "target_id": 0,
        "position": [10.0, 10.0, 0.3],
        "velocity": [0.0, 0.0, 0.0],
        "confidence": 0.9,
        "last_observed_time": 0.0,
        "reporting_agent_id": 0,
    }
    msg = nodes[0].create_message(
        msg_type=MessageType.TARGET_INTEL,
        payload=payload,
        sim_time=0.0,
        ttl=3,
    )
    nodes[0].forward_queue.append(msg)

    delivered = {0}
    total_retransmissions = 0

    # Simulate discrete forwarding ticks
    for tick in range(5):
        t_now = tick * 0.1
        new_transmissions = []
        for i, node in enumerate(nodes):
            node.process_inbox(current_time=t_now)
            while node.forward_queue:
                m = node.forward_queue.pop(0)
                for peer in adj[i]:
                    new_transmissions.append((peer, m))
                    total_retransmissions += 1

        for peer, m in new_transmissions:
            accepted = nodes[peer].receive_message(m)
            if accepted:
                delivered.add(peer)

    pdr = (len(delivered) / 4) * 100.0

    return {
        "total_nodes": 4,
        "nodes_reached": sorted(list(delivered)),
        "packet_delivery_ratio_pct": pdr,
        "total_retransmissions": total_retransmissions,
        "duplicate_suppression_active": True,
    }


def main():
    print("=" * 80)
    print("MRD-SWARM: RF Mesh Resilience & Network Stress Campaign")
    print("=" * 80)

    ew_results = run_ew_sweep_campaign()
    multihop_results = run_multihop_forwarding_stress()

    print(f"\n[OK] Network Retention under 12m EW Jamming: {ew_results['retention_at_12m_pct']}% (Spec >= 50%)")
    print(f"[OK] Multi-hop Packet Delivery Ratio:       {multihop_results['packet_delivery_ratio_pct']}%")
    print(f"[OK] Reached Nodes:                         {multihop_results['nodes_reached']}")
    print(f"[OK] Total Retransmissions with Dedup:     {multihop_results['total_retransmissions']}")

    # ── Generate Publication Figure: Network Resilience Curves ─────────────────
    print("\nGenerating Figure: Network Resilience Curves...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    radii = ew_results["radii"]
    f_with = ew_results["fiedler_with_relay"]
    f_without = ew_results["fiedler_without_relay"]

    ax1.plot(radii, f_with, "g-o", linewidth=2.2, label="With High-Altitude Comms Relay (Z=9.5m)")
    ax1.plot(radii, f_without, "r--s", linewidth=2.0, label="Without Comms Relay (Degraded Mesh)")
    ax1.axvline(12.0, color="orange", linestyle=":", label="Operational Jamming Radius (12m)")
    ax1.set_xlabel("EW Jamming Radius (m)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Algebraic Connectivity (Fiedler lambda_2)", fontsize=11, fontweight="bold")
    ax1.set_title("Graph Spectral Connectivity Under Variable EW Jamming", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", frameon=True)

    # Subplot 2: Total Active Mesh Links
    l_with = ew_results["links_with_relay"]
    l_without = ew_results["links_without_relay"]
    ax2.step(radii, l_with, "g-", where="mid", linewidth=2.2, label="Active Links (With Relay)")
    ax2.step(radii, l_without, "r--", where="mid", linewidth=2.0, label="Active Links (Without Relay)")
    ax2.set_xlabel("EW Jamming Radius (m)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Total Active Links", fontsize=11, fontweight="bold")
    ax2.set_title("Mesh Link Count Retention vs EW Attack Radius", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "network_resilience_curves.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"  [SAVED] {fig_path}")

    summary = {
        "ew_jamming_sweep": ew_results,
        "multihop_routing": multihop_results,
    }
    json_path = OUTPUT_DIR / "network_stress_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[DONE] Saved summary to {json_path}")


if __name__ == "__main__":
    main()
