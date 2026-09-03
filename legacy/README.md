# MRD-SWARM: Legacy / Deprecated Prototypes

This directory preserves early exploratory scripts developed prior to the authoritative ECS-inspired simulation core and multi-seed benchmarking framework.

## Deprecated Scripts
- `sim_gossip_swarm_mission.py`: Initial 45s ad-hoc gossip simulation prototype.
- `sim_advanced_gossip_swarm.py`: 90s standalone script preceding the unified `ECSWorld` pipeline.
- `recon_swarm_sim.py`: Early camera reconnaissance prototype.

## Active Production Entrypoints
For current, tested, and validated simulation and benchmarking workflows, refer to the repository root and `scripts/`:
- `run_swarm_stack.py`: Decoupled physics + 60 Hz WebGL visualizer server.
- `dynamic_swarm_sim.py`: Master closed-loop autonomous simulation harness.
- `scripts/run_doctrine_benchmark.py`: Multi-seed Monte Carlo evaluation campaign.
- `scripts/run_eval_benchmark.py`: Comprehensive aerospace evaluation suite.
- `tests/`: Automated unit and integration test suite (`pytest tests/ -v`).
