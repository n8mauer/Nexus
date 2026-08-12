# Nexus Product Requirements

Date: 2026-08-12

## Persona

**Primary: LLM-agent or RL researcher.** Needs a multi-agent negotiation environment with measurable per-round reward signals (`nexus/rewards.py`), hidden information, and built-in oversight so agents can be trained and evaluated on strategic behavior rather than pattern matching. Wants an environment that speaks the native format of LLMs: text observations in, parseable actions out.

**Secondary: applied AI engineer prototyping procurement or resource-allocation agents.** Wants a simulation testbed with market mechanics (bids, offers, price impact, commission) to evaluate agent strategies before touching real cloud spend.

## User Workflow

1. Clone the repository and install it editable: `pip install -e ".[dev]"` (Python 3.11+).
2. Confirm the install by running the test suite: `pytest` (42 tests across `tests/`).
3. Run a scripted-baseline simulation from the CLI: `python -m scripts.run_simulation run --preset tiny --agents greedy,random --seed 42`.
4. Inspect the outputs in the results directory: `journal.md` (human-readable round narrative), `metrics.jsonl` (machine-readable per-round metrics), and `final_results.json` (final standings), all produced by `nexus/journal.py` and `nexus/persistence.py`.
5. For training loops, start the OpenEnv server: `python -m uvicorn nexus_openenv.server.app:app --port 8000`, then connect with the typed client (`nexus_openenv/client.py`): `reset()` returns a text observation, `step(NexusAction(raw_text=...))` advances one round and returns the round reward. The external model controls `agent_0`; greedy NPCs fill the other seats (`nexus_openenv/server/nexus_environment.py`).
6. Use `notebooks/nexus_training.ipynb` as the GRPO fine-tuning setup reference. Note that it documents the setup only; no training runs are published in this repository.
7. For oversight research, run the `oversight` preset so a `SupervisorAgent` (`agents/supervisor_agent.py`) analyzes each round with CART behavioral probes and reports flags in the round journal.
8. Extend the environment by subclassing `BaseAgent` (`agents/base.py`) or adjusting a `NexusConfig` preset (`nexus/config.py`).

## Problem

Training LLM agents for multi-party negotiation requires an environment that is rich enough to reward strategy (hidden job queues, coalitions, a double-auction market, random disruptions) and structured enough to yield a usable reward signal. Existing options fall short for this use case: classical multi-agent RL suites use tensor observations that are unnatural for LLMs, and LLM multi-agent benchmarks evaluate but do not provide a training loop with per-step rewards. Researchers who also want oversight signals (detecting collusion, hoarding, free-riding) generally have to bolt on their own monitoring. Applied engineers who want to prototype procurement agents have no low-stakes sandbox with market physics.

Nexus addresses this with one codebase: a turn-based cluster-negotiation simulation with natural-language observations, a validated structured action space, multi-signal rewards, a privileged supervisor with interpretable probes, and an OpenEnv-compatible HTTP interface for training frameworks.

## Success Metrics

Each metric maps to an acceptance gate in `docs/ACCEPTANCE.md`.

1. **Test suite health (Gate 1).** `pytest` collects and passes all 42 tests in `tests/` on Python 3.11, locally and in CI.
2. **Reproducible demo (Gate 2).** A seeded demo (tiny preset, greedy vs random, fixed seed) runs to completion with exit code 0 and writes a results journal (`journal.md`, `metrics.jsonl`, `final_results.json`), asserted in CI.
3. **Working integration surface (Gate 3).** The OpenEnv server starts, `/health` reports healthy, and a client `reset()` plus one `step()` round trip succeeds.
4. **Documentation honesty.** `ROADMAP.md` and README section 10 (Implementation Status) agree exactly on what is implemented versus planned, and `docs/RESULTS.md` states plainly that no trained-agent results exist in this repository yet.

## Non-goals

- **Not a production procurement system.** Nexus simulates a resource market; it does not connect to any cloud provider, billing API, or real spend.
- **No trained agents or published training results.** The repository ships scripted and heuristic baselines plus a training-setup notebook. GRPO runs and trained-agent evaluations are roadmap items, not deliverables of this version.
- **No Gymnasium wrapper, evaluation leaderboard, or scripted anomaly-injection scenarios yet.** These are tracked as Planned in `ROADMAP.md`.
- **Not a general-purpose market simulator.** Market mechanics exist to shape negotiation incentives, not to model real spot-price dynamics with fidelity.
- **Not an oversight product.** The supervisor is a research harness for studying interpretable oversight, not a compliance or monitoring tool.
