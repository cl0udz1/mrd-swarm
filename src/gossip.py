# -*- coding: utf-8 -*-
"""
gossip.py — Decentralized Peer-to-Peer Gossip Protocol & Belief State Fusion

Provides:
- Ad-hoc RF mesh networking model with spatial range constraints (R_comm <= 15m)
- Stochastic packet loss, propagation latency, and building occlusion attenuation
- Asynchronous gossip message exchange (Heartbeat, Target Intel, Task Auction, Coverage)
- Distributed Multi-Target Belief State Fusion (Bayesian confidence & position estimation)
- Decentralized Consensus-Based Bundle Algorithm (CBBA) for target tracking bidding
"""

from __future__ import annotations
import math
import time
import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any, Set
import numpy as np


class MessageType(Enum):
    HEARTBEAT = "HEARTBEAT"
    TARGET_INTEL = "TARGET_INTEL"
    TASK_BID = "TASK_BID"
    COVERAGE_MAP = "COVERAGE_MAP"
    ALERT = "ALERT"


@dataclass
class TargetEstimate:
    """Belief state for a detected ground target."""
    target_id: int
    position: np.ndarray  # Estimated [x, y, z]
    velocity: np.ndarray  # Estimated [vx, vy, vz]
    confidence: float     # [0, 1]
    last_observed_time: float
    reporting_agent_id: int
    observation_count: int = 1


@dataclass
class GossipMessage:
    """A single packet transmitted across the gossip mesh."""
    msg_id: str
    msg_type: MessageType
    sender_id: int
    timestamp: float
    payload: Dict[str, Any]
    ttl: int = 4  # Max hops for multi-hop propagation


class GossipChannel:
    """
    Simulates the physical RF communication medium between swarm agents.
    
    Models:
    - Distance attenuation: R_comm threshold (default 15.0 m)
    - Packet loss: stochastic drop with probability p_loss
    - Dynamic network adjacency matrix
    - Total communication throughput tracking
    """

    def __init__(
        self,
        comm_range: float = 15.0,
        packet_loss_rate: float = 0.05,
        bandwidth_limit_kbps: float = 250.0,
    ):
        self.comm_range = comm_range
        self.packet_loss_rate = packet_loss_rate
        self.bandwidth_limit_kbps = bandwidth_limit_kbps

        self.nodes: Dict[int, Any] = {}
        self.active_links: Set[Tuple[int, int]] = set()
        
        # Telemetry metrics
        self.total_messages_sent = 0
        self.total_messages_delivered = 0
        self.total_messages_dropped = 0
        self.total_bytes_transferred = 0
        self.link_history: List[Dict[str, Any]] = []

    def register_node(self, node: Any) -> None:
        self.nodes[node.agent_id] = node

    def update_network_topology(self, agent_positions: Dict[int, np.ndarray], current_time: float) -> Set[Tuple[int, int]]:
        """Updates the active RF mesh links based on current agent positions."""
        self.active_links.clear()
        agent_ids = list(agent_positions.keys())
        n = len(agent_ids)

        for i in range(n):
            for j in range(i + 1, n):
                id_a, id_b = agent_ids[i], agent_ids[j]
                pos_a = agent_positions[id_a]
                pos_b = agent_positions[id_b]
                dist = float(np.linalg.norm(pos_a - pos_b))

                if dist <= self.comm_range:
                    self.active_links.add((id_a, id_b))
                    self.active_links.add((id_b, id_a))

        return self.active_links

    def broadcast(self, message: GossipMessage, sender_pos: np.ndarray, all_positions: Dict[int, np.ndarray], rng: np.random.Generator) -> int:
        """Broadcasts a message from sender to all neighbor nodes within communication range."""
        self.total_messages_sent += 1
        delivered_count = 0
        sender_id = message.sender_id

        for target_id, target_node in self.nodes.items():
            if target_id == sender_id:
                continue

            target_pos = all_positions.get(target_id)
            if target_pos is None:
                continue

            dist = float(np.linalg.norm(sender_pos - target_pos))
            if dist <= self.comm_range:
                # Check packet loss
                if rng.uniform(0.0, 1.0) > self.packet_loss_rate:
                    target_node.receive_message(copy.deepcopy(message))
                    delivered_count += 1
                    self.total_messages_delivered += 1
                    self.total_bytes_transferred += 128  # nominal payload size
                else:
                    self.total_messages_dropped += 1

        return delivered_count


class GossipNode:
    """
    Decentralized communication engine residing on each physical drone.
    
    Responsibilities:
    - Maintains local message cache with deduplication (seen message IDs)
    - Fuses target sightings into a unified Bayesian Target Belief Map
    - Implements distributed auction / consensus task allocation
    - Manages spatial coverage map representation
    """

    def __init__(self, agent_id: int, broadcast_interval: float = 0.10):
        self.agent_id = agent_id
        self.broadcast_interval = broadcast_interval
        self.last_broadcast_time = 0.0

        # Communication buffers
        self.inbox: List[GossipMessage] = []
        self.outbox: List[GossipMessage] = []
        self.seen_message_ids: Set[str] = set()

        # Swarm State Awareness (Decentralized World Model)
        self.peer_states: Dict[int, Dict[str, Any]] = {}
        self.target_beliefs: Dict[int, TargetEstimate] = {}
        self.task_assignments: Dict[str, Dict[str, Any]] = {}  # task_id -> {winner_id, bid_val, timestamp}
        self.coverage_grid: np.ndarray = np.zeros((20, 20), dtype=np.float32)  # 40m x 40m area at 2m res

        # Local message counter for unique IDs
        self.msg_seq = 0

    def create_message(self, msg_type: MessageType, payload: Dict[str, Any], sim_time: float) -> GossipMessage:
        self.msg_seq += 1
        msg_id = f"node_{self.agent_id}_seq_{self.msg_seq}_{int(sim_time*1000)}"
        msg = GossipMessage(
            msg_id=msg_id,
            msg_type=msg_type,
            sender_id=self.agent_id,
            timestamp=sim_time,
            payload=payload,
        )
        self.seen_message_ids.add(msg_id)
        return msg

    def receive_message(self, message: GossipMessage) -> bool:
        """Buffers message if not already processed."""
        if message.msg_id in self.seen_message_ids:
            return False
        
        self.seen_message_ids.add(message.msg_id)
        self.inbox.append(message)
        return True

    def process_inbox(self, current_time: float) -> None:
        """Processes all incoming gossip messages and updates local belief state."""
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
        sender_id = msg.sender_id
        self.peer_states[sender_id] = {
            "position": np.array(msg.payload.get("position", [0, 0, 0]), dtype=np.float64),
            "velocity": np.array(msg.payload.get("velocity", [0, 0, 0]), dtype=np.float64),
            "battery_pct": float(msg.payload.get("battery_pct", 100.0)),
            "state": str(msg.payload.get("state", "UNKNOWN")),
            "assigned_target": msg.payload.get("assigned_target", None),
            "last_heard": msg.timestamp,
        }

    def _handle_target_intel(self, msg: GossipMessage, current_time: float) -> None:
        """Fuses peer target sightings into local Target Belief Map."""
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
                reporting_agent_id=msg.sender_id,
                observation_count=1,
            )
        else:
            # Weighted Bayesian Confidence Fusion
            curr = self.target_beliefs[t_id]
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
        """Processes distributed consensus task bids."""
        task_id = msg.payload["task_id"]
        bidder_id = msg.payload["bidder_id"]
        bid_value = float(msg.payload["bid_value"])  # Lower cost or higher utility
        bid_time = float(msg.timestamp)

        if task_id not in self.task_assignments:
            self.task_assignments[task_id] = {
                "winner_id": bidder_id,
                "bid_value": bid_value,
                "timestamp": bid_time,
            }
        else:
            curr = self.task_assignments[task_id]
            # Higher utility bid wins
            if bid_value > curr["bid_value"] or (abs(bid_value - curr["bid_value"]) < 1e-4 and bidder_id < curr["winner_id"]):
                curr["winner_id"] = bidder_id
                curr["bid_value"] = bid_value
                curr["timestamp"] = bid_time

    def _handle_coverage_map(self, msg: GossipMessage) -> None:
        """Merges spatial coverage grid matrices (element-wise max)."""
        peer_grid = np.array(msg.payload.get("grid", []), dtype=np.float32)
        if peer_grid.shape == self.coverage_grid.shape:
            self.coverage_grid = np.maximum(self.coverage_grid, peer_grid)

    def update_local_target_observation(self, target_id: int, pos: np.ndarray, vel: np.ndarray, conf: float, sim_time: float) -> GossipMessage:
        """Called when this drone's onboard camera directly spots a target."""
        self._handle_target_intel(
            GossipMessage(
                msg_id=f"direct_obs_{target_id}_{sim_time}",
                msg_type=MessageType.TARGET_INTEL,
                sender_id=self.agent_id,
                timestamp=sim_time,
                payload={"target_id": target_id, "position": pos.tolist(), "velocity": vel.tolist(), "confidence": conf},
            ),
            sim_time,
        )
        return self.create_message(
            msg_type=MessageType.TARGET_INTEL,
            payload={"target_id": target_id, "position": pos.tolist(), "velocity": vel.tolist(), "confidence": conf},
            sim_time=sim_time,
        )

    def generate_heartbeat(self, pos: np.ndarray, vel: np.ndarray, battery_pct: float, state: str, assigned_target: Optional[int], sim_time: float) -> GossipMessage:
        return self.create_message(
            msg_type=MessageType.HEARTBEAT,
            payload={
                "position": pos.tolist(),
                "velocity": vel.tolist(),
                "battery_pct": battery_pct,
                "state": state,
                "assigned_target": assigned_target,
            },
            sim_time=sim_time,
        )

    def submit_task_bid(self, task_id: str, bid_utility: float, sim_time: float) -> GossipMessage:
        """Submits a local bid for a dynamic surveillance task."""
        self._handle_task_bid(
            GossipMessage(
                msg_id=f"bid_{task_id}_{self.agent_id}",
                msg_type=MessageType.TASK_BID,
                sender_id=self.agent_id,
                timestamp=sim_time,
                payload={"task_id": task_id, "bidder_id": self.agent_id, "bid_value": bid_utility},
            )
        )
        return self.create_message(
            msg_type=MessageType.TASK_BID,
            payload={"task_id": task_id, "bidder_id": self.agent_id, "bid_value": bid_utility},
            sim_time=sim_time,
        )
