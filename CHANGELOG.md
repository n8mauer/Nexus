# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-08-12

First versioned release. The environment code was first uploaded on 2026-03-10 and hardened through 2026-08-12; this release snapshots that work plus the documentation, CI, and demo additions made today.

### Added

- Turn-based multi-agent compute-cluster negotiation environment: state model, 7-phase round loop, job generation with deadlines, and random disruption events (`nexus/state.py`, `nexus/engine.py`, `nexus/events.py`).
- Structured action space with validation and LLM output parsing (`nexus/actions.py`) and natural-language observation rendering with hidden-information masking (`nexus/observations.py`).
- Double-auction market with price impact and commission (`nexus/market.py`), coalition formation with MoE confidence-weighted voting (`nexus/coalitions.py`), and multi-signal reward computation including Sharpe ratio (`nexus/rewards.py`).
- Oversight track: CART behavioral probes, pairwise collusion detection, and a supervisor agent with per-round reports (`nexus/oversight.py`, `agents/supervisor_agent.py`).
- Multi-actor CTO mode with directive parsing and workers that can misinterpret instructions (`nexus/multi_actor.py`, `agents/cto_agent.py`).
- Baseline agents: random, greedy, MCTS/UCB1 strategic planner, and an Anthropic tool-use LLM agent (`agents/`).
- OpenEnv 0.2.x integration: FastAPI HTTP server, typed `NexusEnv` client, and a ProxyAgent bridge that lets an external model control `agent_0` against greedy NPCs (`nexus_openenv/`).
- Typer CLI with `run` and `evaluate` commands, dual-format journaling (JSONL plus Markdown), and results persistence (`scripts/run_simulation.py`, `nexus/journal.py`, `nexus/persistence.py`).
- Test suite of 42 tests across seven modules (`tests/`).
- GRPO fine-tuning setup notebook; setup only, no training runs are published (`notebooks/nexus_training.ipynb`).
- Hugging Face Space staging build script and overlay files (`scripts/build_space.py`, `deploy/space/`).
- Product documentation: PRD, three architecture decision records, roadmap, acceptance criteria, results provenance, and this changelog (`docs/`, `ROADMAP.md`).
- Seeded demo (`demo.py`, tiny preset, greedy vs random, fixed seed) with a journal-asserting test (`tests/test_demo.py`) and a recorded terminal image (`docs/images/demo.svg`).
- OpenEnv server smoke script covering `/health` plus a reset and one-step round trip (`scripts/openenv_smoke.py`).
- Locked dependencies (`requirements.lock`, regenerated via `constraints.in`; `openenv-core` pinned to 0.2.1 to keep the synchronous client working) and a CI workflow enforcing all three acceptance gates on ubuntu-latest with Python 3.11, installing from the lock (`.github/workflows/ci.yml`, see `docs/ACCEPTANCE.md`).

### Changed

- README revised on 2026-08-12 with an accurate Implementation Status section separating Implemented from Planned work.
- Packaging metadata and OpenEnv client type parameters fixed on 2026-08-12.

### Removed

- Committed duplicate package copies for the Hugging Face Space (`hf_space_staging/`); the staging directory is now generated on demand by `scripts/build_space.py` (2026-08-12, see ADR-003).
