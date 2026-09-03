# -*- coding: utf-8 -*-
"""
test_network.py — Automated Tests for Multi-Hop Gossip Mesh Networking,
TTL Decrement, Deduplication, and Distributed Utility Auction.
"""

import numpy as np
import pytest

from src.gossip import GossipNode, GossipChannel, MessageType, GossipMessage


def test_gossip_message_creation_and_deduplication():
    """Verify unique message generation and deduplication cache."""
    node = GossipNode(agent_id=0)
    msg1 = node.create_message(MessageType.HEARTBEAT, {"state": "OK"}, sim_time=1.0, ttl=4)

    assert msg1.ttl == 4
    assert msg1.origin_id == 0
    assert msg1.forwarder_id == 0
    assert msg1.msg_id in node.seen_message_ids

    # Trying to receive the same message must be rejected (deduplication)
    accepted = node.receive_message(msg1)
    assert accepted is False


def test_multihop_ttl_decrement_and_forwarding():
    """Verify that receiving a message with TTL > 1 queues a forwarded copy with decremented TTL."""
    node_b = GossipNode(agent_id=1)
    incoming_msg = GossipMessage(
        msg_id="origin_0_seq_1",
        msg_type=MessageType.TARGET_INTEL,
        origin_id=0,
        forwarder_id=0,
        timestamp=1.0,
        payload={"target_id": 0, "position": [10.0, 5.0, 0.0], "confidence": 0.9},
        ttl=3,
        hop_count=0,
    )

    accepted = node_b.receive_message(incoming_msg)
    assert accepted is True
    assert len(node_b.inbox) == 1

    # Check forward queue
    assert len(node_b.forward_queue) == 1
    fwd = node_b.forward_queue[0]
    assert fwd.ttl == 2            # TTL decremented from 3 to 2
    assert fwd.hop_count == 1      # Hop count incremented
    assert fwd.forwarder_id == 1   # Forwarder ID updated to Node B


def test_terminal_ttl_does_not_forward():
    """A message arriving with TTL = 1 should be ingested but NOT forwarded."""
    node_c = GossipNode(agent_id=2)
    terminal_msg = GossipMessage(
        msg_id="origin_0_seq_terminal",
        msg_type=MessageType.TARGET_INTEL,
        origin_id=0,
        forwarder_id=1,
        timestamp=2.0,
        payload={"target_id": 0, "position": [10.0, 5.0, 0.0], "confidence": 0.9},
        ttl=1,  # Final hop
        hop_count=2,
    )

    accepted = node_c.receive_message(terminal_msg)
    assert accepted is True
    assert len(node_c.inbox) == 1
    # Should NOT be queued for forwarding because ttl == 1
    assert len(node_c.forward_queue) == 0


def test_distributed_utility_auction_winner():
    """Verify highest utility bid wins the distributed task auction."""
    node = GossipNode(agent_id=0)

    # First bid: Drone 1 bids 0.65
    bid_1 = GossipMessage(
        msg_id="bid_1", msg_type=MessageType.TASK_BID, origin_id=1, forwarder_id=1, timestamp=1.0,
        payload={"task_id": "TRACK_HVT_0", "bidder_id": 1, "bid_value": 0.65},
    )
    node.receive_message(bid_1)
    node.process_inbox(current_time=1.0)
    assert node.task_assignments["TRACK_HVT_0"]["winner_id"] == 1
    assert node.task_assignments["TRACK_HVT_0"]["bid_value"] == 0.65

    # Second bid: Drone 2 bids 0.88 (higher utility)
    bid_2 = GossipMessage(
        msg_id="bid_2", msg_type=MessageType.TASK_BID, origin_id=2, forwarder_id=2, timestamp=1.1,
        payload={"task_id": "TRACK_HVT_0", "bidder_id": 2, "bid_value": 0.88},
    )
    node.receive_message(bid_2)
    node.process_inbox(current_time=1.1)
    # Winner must update to Drone 2
    assert node.task_assignments["TRACK_HVT_0"]["winner_id"] == 2
    assert node.task_assignments["TRACK_HVT_0"]["bid_value"] == 0.88

    # Third bid: Drone 0 bids 0.50 (lower utility)
    bid_3 = GossipMessage(
        msg_id="bid_3", msg_type=MessageType.TASK_BID, origin_id=0, forwarder_id=0, timestamp=1.2,
        payload={"task_id": "TRACK_HVT_0", "bidder_id": 0, "bid_value": 0.50},
    )
    node.receive_message(bid_3)
    node.process_inbox(current_time=1.2)
    # Winner must remain Drone 2
    assert node.task_assignments["TRACK_HVT_0"]["winner_id"] == 2
