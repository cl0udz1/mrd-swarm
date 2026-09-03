# -*- coding: utf-8 -*-
"""
scenarios.py — Formal Environmental and Operational Scenarios for MRD-SWARM.

Defines standardized operational environments:
- Scenario A: Open Field (0 obstacles, unhindered LOS, baseline kinematics)
- Scenario B: Sparse Urban (4 medium buildings, intermittent occlusion)
- Scenario C: Dense Urban (8 buildings, urban canyon, narrow alleys)
- Scenario D: Comms Stress (Dense urban + continuous EW jamming, RF degradation)
- Scenario E: Sensor Stress (Dense urban + active smoke aerosol + high Dryden turbulence)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional
import numpy as np


class ScenarioID(Enum):
    SCENARIO_A_OPEN_FIELD = "SCENARIO_A_OPEN_FIELD"
    SCENARIO_B_SPARSE_URBAN = "SCENARIO_B_SPARSE_URBAN"
    SCENARIO_C_DENSE_URBAN = "SCENARIO_C_DENSE_URBAN"
    SCENARIO_D_COMMS_STRESS = "SCENARIO_D_COMMS_STRESS"
    SCENARIO_E_SENSOR_STRESS = "SCENARIO_E_SENSOR_STRESS"


@dataclass
class ScenarioConfig:
    scenario_id: ScenarioID
    name: str
    description: str
    obstacles: List[Dict[str, Any]]
    ew_jamming_enabled: bool
    ew_center: np.ndarray
    ew_radius: float
    wind_speed_20m: float
    smoke_active_initial: bool
    mission_duration_s: float = 60.0


# ── Canonical Building Layouts ─────────────────────────────────────────────────

BUILDINGS_SPARSE = [
    {"name": "Building_NW", "pos": [-12.0, 10.0, 0.0], "size": [3.0, 3.0, 5.0]},
    {"name": "Building_NE", "pos": [12.0, 10.0, 0.0], "size": [3.0, 3.0, 5.0]},
    {"name": "Building_SW", "pos": [-10.0, -12.0, 0.0], "size": [3.0, 3.0, 4.0]},
    {"name": "Building_SE", "pos": [10.0, -12.0, 0.0], "size": [3.0, 3.0, 4.0]},
]

BUILDINGS_DENSE = [
    {"name": "Skyscraper_HQ", "pos": [0.0, 0.0, 0.0], "size": [4.0, 4.0, 9.0]},
    {"name": "Tower_Alpha", "pos": [-15.0, 15.0, 0.0], "size": [3.5, 3.5, 6.0]},
    {"name": "Tower_Bravo", "pos": [15.0, 15.0, 0.0], "size": [3.5, 3.5, 6.5]},
    {"name": "Tower_Charlie", "pos": [-15.0, -15.0, 0.0], "size": [3.0, 3.0, 5.0]},
    {"name": "Tower_Delta", "pos": [15.0, -15.0, 0.0], "size": [3.0, 3.0, 5.5]},
    {"name": "Warehouse_East", "pos": [18.0, 0.0, 0.0], "size": [3.0, 5.0, 3.5]},
    {"name": "Depot_West", "pos": [-18.0, 0.0, 0.0], "size": [3.0, 5.0, 3.0]},
    {"name": "Substation_North", "pos": [0.0, 18.0, 0.0], "size": [5.0, 2.5, 3.0]},
]


SCENARIO_CONFIGS: Dict[ScenarioID, ScenarioConfig] = {
    ScenarioID.SCENARIO_A_OPEN_FIELD: ScenarioConfig(
        scenario_id=ScenarioID.SCENARIO_A_OPEN_FIELD,
        name="Scenario A: Open Field",
        description="Zero vertical obstacles, line-of-sight everywhere, benchmark aerodynamic baseline.",
        obstacles=[],
        ew_jamming_enabled=False,
        ew_center=np.array([0.0, 0.0, 0.0]),
        ew_radius=0.0,
        wind_speed_20m=2.0,
        smoke_active_initial=False,
    ),
    ScenarioID.SCENARIO_B_SPARSE_URBAN: ScenarioConfig(
        scenario_id=ScenarioID.SCENARIO_B_SPARSE_URBAN,
        name="Scenario B: Sparse Urban",
        description="4 perimeter buildings, moderate line-of-sight occlusion, low turbulence.",
        obstacles=BUILDINGS_SPARSE,
        ew_jamming_enabled=False,
        ew_center=np.array([0.0, 0.0, 0.0]),
        ew_radius=0.0,
        wind_speed_20m=3.0,
        smoke_active_initial=False,
    ),
    ScenarioID.SCENARIO_C_DENSE_URBAN: ScenarioConfig(
        scenario_id=ScenarioID.SCENARIO_C_DENSE_URBAN,
        name="Scenario C: Dense Urban",
        description="8 buildings with central skyscraper, severe ray occlusion, canyon navigation.",
        obstacles=BUILDINGS_DENSE,
        ew_jamming_enabled=False,
        ew_center=np.array([14.0, 14.0, 4.0]),
        ew_radius=8.0,
        wind_speed_20m=3.5,
        smoke_active_initial=False,
    ),
    ScenarioID.SCENARIO_D_COMMS_STRESS: ScenarioConfig(
        scenario_id=ScenarioID.SCENARIO_D_COMMS_STRESS,
        name="Scenario D: Comms Stress",
        description="Dense urban topology with high-intensity EW jamming field active throughout mission.",
        obstacles=BUILDINGS_DENSE,
        ew_jamming_enabled=True,
        ew_center=np.array([12.0, 12.0, 3.5]),
        ew_radius=12.0,
        wind_speed_20m=3.0,
        smoke_active_initial=False,
    ),
    ScenarioID.SCENARIO_E_SENSOR_STRESS: ScenarioConfig(
        scenario_id=ScenarioID.SCENARIO_E_SENSOR_STRESS,
        name="Scenario E: Sensor Stress",
        description="Dense urban environment with active optical smoke countermeasures and heavy Dryden turbulence (7 m/s).",
        obstacles=BUILDINGS_DENSE,
        ew_jamming_enabled=False,
        ew_center=np.array([0.0, 0.0, 0.0]),
        ew_radius=0.0,
        wind_speed_20m=7.0,
        smoke_active_initial=True,
    ),
}


def get_scenario(scenario_id: ScenarioID) -> ScenarioConfig:
    """Retrieves authoritative configuration for a given scenario ID."""
    return SCENARIO_CONFIGS[scenario_id]
