# -*- coding: utf-8 -*-
"""
gossip.py — Decentralized Ad-Hoc Multi-Hop Gossip Protocol & Target Belief Fusion

Provides:
- Peer-to-Peer RF mesh networking with spatial range thresholds (Drone 3 Relay = 32m, others = 18m)
- True Multi-Hop Forwarding: TTL decrement, sequence-number deduplication, loop prevention
- Stochastic packet loss, propagation latency, and dynamic topology changes
- ConfidenceWeightedTargetFusion: Heuristic confidence-weighted observation blending
- DistributedUtilityAuction: Single-task decentralized highest-utility auction
"""

from __future__ import annotations
import math
import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any, Set
import numpy as np
from numpy.typing import NDArray

from .config.airframes import FLEET_CONFIGS


class MessageType(Enum):
    HEARTBEAT = "HEARTBEAT"
    TARGET_INTEL = "TARGET_INTEL"
    TASK_BID = "TASK_BID"
    COVERAGE_MAP = "COVERAGE_MAP"
    ALERT = "ALERT"


@dataclass
class TargetEstimate:
    """Belief state for a detected ground target held in a drone's local world model."""
    target_id: int
    position: NDArray[np.float64]  # Estimated [x, y, z]
    velocity: NDArray[np.float64]  # Estimated [vx, vy, vz]
    confidence: float              # Observation confidence [0, 1]
    last_observed_time: float
    reporting_agent_id: int
    observation_count: int = 1


@dataclass
class GossipMessage:
    """A packet transmitted across the ad-hoc mesh network."""
    msg_id: str
    msg_type: MessageType
    origin_id: int                # Original generating node ID
    forwarder_id: int             # Immediate 1-hop sender ID
    timestamp: float
    payload: Dict[str, Any]
    ttl: int = 4                  # Remaining hops (decremented on each forwarding step)
    hop_count: int = 0            # Number of hops traversed


class GossipChannel:
    """
    Simulates the physical wireless medium between swarm agents.
    Handles distance thresholds, stochastic packet drops, multi-hop forwarding, and throughput.
    """

    def __init__(
        self,
        default_comm_range: float = 18.0,
        packet_loss_rate: float = 0.05,
        bandwidth_limit_kbps: float = 250.0,
    ):
        self.default_comm_range = default_comm_range
        self.packet_loss_rate = packet_loss_rate
        self.bandwidth_limit_kbps = bandwidth_limit_kbps

        self.nodes: Dict[int, GossipNode] = {}
        self.active_links: Set[Tuple[int, int]] = set()

        # Telemetry metrics
        self.total_messages_generated = 0
        self.total_messages_forwarded = 0
        self.total_messages_delivered = 0
        self.total_messages_dropped = 0
        self.total_bytes_transferred = 0

    def register_node(self, node: GossipNode) -> None:
        self.nodes[node.agent_id] = node

    def update_network_topology(
        self,
        agent_positions: Dict[int, NDArray[np.float64]],
        current_time: float,
    ) -> Set[Tuple[int, int]]:
        """Updates active RF links based on asymmetric airframe transmission ranges."""
        self.active_links.clear()
        agent_ids = list(agent_positions.keys())
        n = len(agent_ids)

        for i in range(n):
            for j in range(i + 1, n):
                id_a, id_b = agent_ids[i], agent_ids[j]
                pos_a = agent_positions[id_a]
                pos_b = agent_positions[id_b]
                dist = float(np.linalg.norm(pos_a - pos_b))

                # Asymmetric range: relay (D3) transmits up to 32m; others 18m
                range_a = FLEET_CONFIGS[id_a].rf_comm_range_m if id_a in FLEET_CONFIGS else self.default_comm_range
                range_b = FLEET_CONFIGS[id_b].rf_comm_range_m if id_b in FLEET_CONFIGS else self.default_comm_range
                eff_range = max(range_a, range_b)

                if dist <= eff_range:
                    self.active_links.add((min(id_a, id_b), max(id_a, id_b)))

        return self.active_links

    def broadcast_packet(
        self,
        message: GossipMessage,
        sender_pos: NDArray[np.float64],
        agent_positions: Dict[int, NDArray[np.float64]],
        rng: Optional[np.random.RandomState] = None,
    ) -> int:
        """
        Broadcast a packet to all 1-hop physical neighbors within communication range.
        """
        if rng is None:
            rng = np.random.RandomState(42)

        delivered_count = 0
        sender_id = message.forwarder_id
        sender_range = FLEET_CONFIGS[sender_id].rf_comm_range_m if sender_id in FLEET_CONFIGS else self.default_comm_range

        for target_id, target_node in self.nodes.items():
            if target_id == sender_id or target_id == message.origin_id:
                continue

            if target_id not in agent_positions:
                continue

            dist = float(np.linalg.norm(sender_pos - agent_positions[target_id]))
            target_range = FLEET_CONFIGS[target_id].rf_comm_range_m if target_id in FLEET_CONFIGS else self.default_comm_range
            max_r = max(sender_range, target_range)

            if dist <= max_r:
                if rng.uniform(0.0, 1.0) > self.packet_loss_rate:
                    target_node.receive_message(copy.deepcopy(message))
                    delivered_count += 1
                    self.total_messages_delivered += 1
                    self.total_bytes_transferred += 128
                else:
                    self.total_messages_dropped += 1

        return delivered_count


class GossipNode:
    """
    Decentralized communication node residing on each UAV.
    Handles message deduplication, multi-hop forwarding with TTL decrement,
    confidence-weighted target belief fusion, and distributed utility auction.
    """

    def __init__(self, agent_id: int, broadcast_interval: float = 0.10):
        self.agent_id = agent_id
        self.broadcast_interval = broadcast_interval
        self.last_broadcast_time = 0.0

        # Communication buffers
        self.inbox: List[GossipMessage] = []
        self.outbox: List[GossipMessage] = []
        self.forward_queue: List[GossipMessage] = []
        self.seen_message_ids: Set[str] = set()

        # Local decentralized world model
        self.peer_states: Dict[int, Dict[str, Any]] = {}
        self.target_beliefs: Dict[int, TargetEstimate] = {}
        self.task_assignments: Dict[str, Dict[str, Any]] = {}
        self.coverage_grid: np.ndarray = np.zeros((20, 20), dtype=np.float32)

        self.msg_seq = 0

    def create_message(self, msg_type: MessageType, payload: Dict[str, Any], sim_time: float, ttl: int = 4) -> GossipMessage:
        """Create a fresh packet originating at this node."""
        self.msg_seq += 1
        msg_id = f"node_{self.agent_id}_seq_{self.msg_seq}_{int(sim_time * 1000)}"
        msg = GossipMessage(
            msg_id=msg_id,
            msg_type=msg_type,
            origin_id=self.agent_id,
            forwarder_id=self.agent_id,
            timestamp=sim_time,
            payload=payload,
            ttl=ttl,
            hop_count=0,
        )
        self.seen_message_ids.add(msg_id)
        return msg

    def receive_message(self, message: GossipMessage) -> bool:
        """
        Ingests packet into local inbox.
        If TTL > 1, schedules packet for multi-hop re-broadcast with decremented TTL.
        """
        if message.msg_id in self.seen_message_ids:
            return False  # Deduplication drop

        self.seen_message_ids.add(message.msg_id)
        self.inbox.append(message)

        # Multi-hop forwarding: decrement TTL and queue for forwarding if hops remain
        if message.ttl > 1:
            forwarded_pkt = copy.deepcopy(message)
            forwarded_pkt.ttl -= 1
            forwarded_pkt.hop_count += 1
            forwarded_pkt.forwarder_id = self.agent_id
            self.forward_queue.append(forwarded_pkt)

        return True

    def process_inbox(self, current_time: float) -> None:
        """Process incoming messages to update decentralized world model."""
        while self.inbox:
            msg = self.inbox.pop(0)

            if msg.msg_type == MessageType.HEARTBEAT:
                self._handle_heartbeat(msg)
            elif msg.msg_type == MessageType.TARGET_INTEL:
                self._handle_target_intel(msg, current_time)
            elif msg.msg_type == MessageType.TASK_BID:
                self._handle_task_bid(msg)
            elif msg.msg_type == MessageType.COVERAGE_MAP:
                self._handle_coverage_map(msg)

    def _handle_heartbeat(self, msg: GossipMessage) -> None:
        sender_id = msg.origin_id
        self.peer_states[sender_id] = {
            "position": np.array(msg.payload.get("position", [0, 0, 0]), dtype=np.float64),
            "velocity": np.array(msg.payload.get("velocity", [0, 0, 0]), dtype=np.float64),
            "battery_pct": float(msg.payload.get("battery_pct", 100.0)),
            "state": str(msg.payload.get("state", "UNKNOWN")),
            "assigned_target": msg.payload.get("assigned_target", None),
            "last_heard": msg.timestamp,
        }

    def _handle_target_intel(self, msg: GossipMessage, current_time: float) -> None:
        """
        ConfidenceWeightedTargetFusion:
        Blends incoming peer target estimates using scalar confidence weighting.
        """
        t_id = msg.payload["target_id"]
        raw_pos = np.array(msg.payload["position"], dtype=np.float64)
        raw_vel = np.array(msg.payload.get("velocity", [0, 0, 0]), dtype=np.float64)
        conf = float(msg.payload["confidence"])
        obs_time = float(msg.timestamp)

        if t_id not in self.target_beliefs:
            self.target_beliefs[t_id] = TargetEstimate(
                target_id=t_id,
                position=raw_pos,
                velocity=raw_vel,
                confidence=conf,
                last_observed_time=obs_time,
                reporting_agent_id=msg.origin_id,
                observation_count=1,
            )
        else:
            curr = self.target_beliefs[t_id]
            # Confidence-weighted spatial blend
            alpha = conf / (curr.confidence + conf + 1e-6)
            fused_pos = (1.0 - alpha) * curr.position + alpha * raw_pos
            fused_vel = (1.0 - alpha) * curr.velocity + alpha * raw_vel
            fused_conf = min(1.0, curr.confidence * 0.85 + conf * 0.5)

            curr.position = fused_pos
            curr.velocity = fused_vel
            curr.confidence = fused_conf
            curr.last_observed_time = max(curr.last_observed_time, obs_time)
            curr.observation_count += 1

    def _handle_task_bid(self, msg: GossipMessage) -> None:
        """
        DistributedUtilityAuction:
        Single-task auction resolved by highest utility score with tie-breaking.
        """
        task_id = msg.payload["task_id"]
        bidder_id = msg.payload["bidder_id"]
        bid_value = float(msg.payload["bid_value"])
        bid_time = float(msg.timestamp)

        if task_id not in self.task_assignments:
            self.task_assignments[task_id] = {
                "winner_id": bidder_id,
                "bid_value": bid_value,
                "timestamp": bid_time,
            }
        else:
            curr = self.task_assignments[task_id]
            if bid_value > curr["bid_value"] or (abs(bid_value - curr["bid_value"]) < 1e-4 and bidder_id < curr["winner_id"]):
                curr["winner_id"] = bidder_id
                curr["bid_value"] = bid_value
                curr["timestamp"] = bid_time

    def _handle_coverage_map(self, msg: GossipMessage) -> None:
        peer_grid = np.array(msg.payload.get("grid", []), dtype=np.float32)
        if peer_grid.shape == self.coverage_grid.shape:
            self.coverage_grid = np.maximum(self.coverage_grid, peer_grid)
