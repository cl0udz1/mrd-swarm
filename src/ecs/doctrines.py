# -*- coding: utf-8 -*-
"""
doctrines.py — Swarm Tactical Doctrine Specifications & Parameterized Battle Behaviors

Defines distinct, mathematically grounded tactical doctrines for multi-agent quadrotor fleets:
1. AGGRESSIVE_PINCER: High-speed corridor cutoff, aggressive low-altitude sprint (18 m/s), 160° separation.
2. WOLFPACK_CONTAINMENT: Concentric 120° triangular enclosure, 4.5m standoff, high RF mesh resilience.
3. STEALTH_SHADOW: High-altitude standoff (6.0m radius, 5.5m Z), energy-preserving surveillance.
4. DEEPSEEK_ADAPTIVE: Autonomous real-time doctrine arbitration by the DeepSeek AI Commander.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


class TacticalDoctrineID(Enum):
    AGGRESSIVE_PINCER = "AGGRESSIVE_PINCER"
    WOLFPACK_CONTAINMENT = "WOLFPACK_CONTAINMENT"
    STEALTH_SHADOW = "STEALTH_SHADOW"
    DEEPSEEK_ADAPTIVE = "DEEPSEEK_ADAPTIVE"


@dataclass
class DoctrineConfig:
    """Rigorous operational parameters defining a tactical doctrine."""
    doctrine_id: TacticalDoctrineID
    name: str
    description: str
    pincer_separation_deg: float      # Target angular separation between tracker & flanker (°)
    standoff_radius_m: float          # Distance from target center (m)
    tracker_altitude_m: float         # Cruising altitude for primary tracker (m)
    flanker_altitude_m: float         # Intercept altitude for flanker (m)
    flanker_max_speed_mps: float      # Sprint velocity ceiling for flanker (m/s)
    tracker_desired_speed_mps: float  # Standoff pursuit speed (m/s)
    lead_time_s: float                # Evasive trajectory prediction forward horizon (s)
    multi_target_split: bool          # If True, splits fleet to engage secondary HVTs simultaneously
    relay_altitude_m: float           # Comms relay loiter altitude (m)
    battery_pnr_margin: float         # Point-of-No-Return energy safety factor


DOCTRINE_REGISTRY: Dict[TacticalDoctrineID, DoctrineConfig] = {
    TacticalDoctrineID.AGGRESSIVE_PINCER: DoctrineConfig(
        doctrine_id=TacticalDoctrineID.AGGRESSIVE_PINCER,
        name="Aggressive Multi-Axis Pincer Dash",
        description="Aggressive high-speed sprint to cut off urban escape corridors ahead of moving HVTs.",
        pincer_separation_deg=160.0,
        standoff_radius_m=3.0,
        tracker_altitude_m=3.2,
        flanker_altitude_m=2.6,
        flanker_max_speed_mps=18.0,
        tracker_desired_speed_mps=14.0,
        lead_time_s=4.2,
        multi_target_split=False,     # Concentrates all combat assets on primary HVT
        relay_altitude_m=10.5,
        battery_pnr_margin=1.15,
    ),
    TacticalDoctrineID.WOLFPACK_CONTAINMENT: DoctrineConfig(
        doctrine_id=TacticalDoctrineID.WOLFPACK_CONTAINMENT,
        name="Concentric Wolfpack Containment",
        description="Symmetric 120° triangular perimeter enclosure around targets to minimize escape probability.",
        pincer_separation_deg=120.0,
        standoff_radius_m=4.8,
        tracker_altitude_m=4.2,
        flanker_altitude_m=3.8,
        flanker_max_speed_mps=14.0,
        tracker_desired_speed_mps=10.5,
        lead_time_s=2.5,
        multi_target_split=True,      # Splits combat drones into hunting pairs
        relay_altitude_m=11.0,
        battery_pnr_margin=1.30,
    ),
    TacticalDoctrineID.STEALTH_SHADOW: DoctrineConfig(
        doctrine_id=TacticalDoctrineID.STEALTH_SHADOW,
        name="Stealth High-Altitude Shadow",
        description="Conservative standoff observation preserving battery life and maintaining RF line-of-sight.",
        pincer_separation_deg=90.0,
        standoff_radius_m=6.5,
        tracker_altitude_m=5.8,
        flanker_altitude_m=5.2,
        flanker_max_speed_mps=10.0,
        tracker_desired_speed_mps=8.0,
        lead_time_s=1.8,
        multi_target_split=True,
        relay_altitude_m=12.5,
        battery_pnr_margin=1.45,
    ),
    TacticalDoctrineID.DEEPSEEK_ADAPTIVE: DoctrineConfig(
        doctrine_id=TacticalDoctrineID.DEEPSEEK_ADAPTIVE,
        name="DeepSeek Cognitive Autonomous Adaptive",
        description="Real-time dynamic doctrine arbitration driven by DeepSeek LLM & Vision intelligence.",
        pincer_separation_deg=150.0,
        standoff_radius_m=3.8,
        tracker_altitude_m=3.8,
        flanker_altitude_m=3.2,
        flanker_max_speed_mps=16.5,
        tracker_desired_speed_mps=12.5,
        lead_time_s=3.2,
        multi_target_split=True,
        relay_altitude_m=10.5,
        battery_pnr_margin=1.25,
    ),
}


def get_doctrine_config(doctrine: TacticalDoctrineID | str) -> DoctrineConfig:
    """Retrieve doctrine configuration by ID or string name."""
    if isinstance(doctrine, str):
        doctrine = doctrine.upper()
        for doc_id, cfg in DOCTRINE_REGISTRY.items():
            if doc_id.name == doctrine or doc_id.value == doctrine:
                return cfg
        return DOCTRINE_REGISTRY[TacticalDoctrineID.DEEPSEEK_ADAPTIVE]
    return DOCTRINE_REGISTRY.get(doctrine, DOCTRINE_REGISTRY[TacticalDoctrineID.DEEPSEEK_ADAPTIVE])
