# Nexus Roadmap

Last updated: 2026-08-12. Status values: Done, In progress, Planned. Every Done item names its evidence in this repository. This roadmap matches README section 10 (Implementation Status); if the two ever disagree, treat that as a bug.

## Done

| Milestone | Evidence |
|-----------|----------|
| Core simulation engine: state model, 7-phase round loop, job generation, deadlines | `nexus/state.py`, `nexus/engine.py`, `nexus/events.py`, `tests/test_engine.py` |
| Structured action space with validation and LLM output parsing | `nexus/actions.py`, exercised throughout `tests/` |
| Natural-language observations with hidden-information masking | `nexus/observations.py`, `tests/test_observations.py` |
| Double-auction market with price impact and commission | `nexus/market.py`, `tests/test_market.py` |
| Coalition formation with MoE confidence-weighted voting | `nexus/coalitions.py`, `tests/test_coalitions.py` |
| Multi-signal reward computation including Sharpe ratio | `nexus/rewards.py` |
| Oversight: CART behavioral probes, collusion detector, supervisor agent | `nexus/oversight.py`, `agents/supervisor_agent.py`, `tests/test_oversight.py` |
| Multi-actor CTO mode: directives, workers with a misinterpretation model | `nexus/multi_actor.py`, `agents/cto_agent.py` |
| Baseline agents: random, greedy, MCTS/UCB1 strategic, Anthropic tool-use LLM | `agents/random_agent.py`, `agents/greedy_agent.py`, `agents/strategic_agent.py`, `agents/llm_agent.py` |
| OpenEnv 0.2.x integration: HTTP server, typed client, ProxyAgent bridge | `nexus_openenv/`, `tests/test_openenv.py` |
| Typer CLI with `run` and `evaluate` commands, journaling, persistence | `scripts/run_simulation.py`, `nexus/journal.py`, `nexus/persistence.py` |
| 42-test suite passing | `tests/` (seven test modules; `pytest` collects 42 tests, verified 2026-08-12) |
| Seeded demo: tiny preset, greedy vs random, fixed seed, journal written and asserted | `demo.py`, `scripts/demo.py`, `tests/test_demo.py`, `docs/images/demo.svg` (Gate 2 in `docs/ACCEPTANCE.md`) |
| CI workflow enforcing all three acceptance gates on ubuntu-latest, Python 3.11, installing from the lock | `.github/workflows/ci.yml`, `requirements.lock`, `constraints.in` |
| OpenEnv server smoke check: `/health` plus a reset and one-step round trip | `scripts/openenv_smoke.py` (Gate 3 in `docs/ACCEPTANCE.md`; verified locally 2026-08-12) |
| Single canonical package with generated Space staging (removed committed duplicates) | `scripts/build_space.py`, `deploy/space/`, commit 93c999a (2026-08-12) |
| Product documentation: PRD, ADRs, roadmap, acceptance criteria, results provenance, changelog | `docs/PRD.md`, `docs/decisions/`, `ROADMAP.md`, `docs/ACCEPTANCE.md`, `docs/RESULTS.md`, `CHANGELOG.md` (added 2026-08-12) |

## In progress (2026-08-12)

Nothing is in progress as of 2026-08-12; the demo, CI, and smoke-check milestones listed as in progress earlier on this date have landed and moved to Done above.

## Planned

These match the Planned list in README section 10 exactly.

| Milestone | Notes |
|-----------|-------|
| Scripted anomaly injection scenarios (collusion, hoarding) for supervisor training | Config flags exist (`inject_collusion`, `collusion_pairs` in `nexus/config.py`) but no injection logic is implemented yet |
| Gymnasium environment wrapper | Enables classical RL baselines alongside LLM agents |
| Evaluation leaderboard | Standings across agent types and presets |
| Published GRPO training runs with trained-agent evaluation results | `notebooks/nexus_training.ipynb` documents the setup only; no training results exist in this repository |
