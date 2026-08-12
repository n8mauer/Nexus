# Results provenance

Date: 2026-08-12

This page states exactly which results connected to Nexus are verifiable from this repository, which are self-reported, and which have no public artifact. Claims are sorted into three tiers; anything not listed here should not be attributed to this project.

## Verified in this repository

Everything in this tier can be checked by cloning the repository and running the listed commands.

- **The environment implementation itself.** The simulation engine, market, coalitions, events, rewards, oversight probes, multi-actor mode, and baseline agents are committed source code under `nexus/`, `agents/`, and `nexus_openenv/`.
- **The 42-test suite.** `python -m pytest` collects 42 tests across seven modules in `tests/` and they pass (verified 2026-08-12 in the repo-local environment and in a fresh Python 3.11 environment installed from `requirements.lock`). This is acceptance Gate 1 in `docs/ACCEPTANCE.md`.
- **Deterministic demo results.** A seeded run (`python demo.py`: `tiny` preset, greedy vs random, seed 42) completes, exits 0, and writes `journal.md`, `metrics.jsonl`, and `final_results.json`. Two consecutive runs on 2026-08-12 produced byte-identical `final_results.json` output (greedy 826 points, random 456 points, Gini 0.144, market volume 892). This is acceptance Gate 2, asserted by `tests/test_demo.py`.
- **OpenEnv round trip.** The in-process reset and step behavior of the OpenEnv adapter is covered by `tests/test_openenv.py`; the HTTP server round trip is acceptance Gate 3, checked by `scripts/openenv_smoke.py` (passed locally on 2026-08-12 with `openenv-core` 0.2.1).

## Self-reported operational results

None at present. No benchmark numbers, throughput figures, training curves, or deployment outcomes are claimed for Nexus beyond what the previous tier covers.

In particular: **no GRPO training results exist in this repository.** `notebooks/nexus_training.ipynb` documents a fine-tuning setup (model, LoRA configuration, reward wiring), not completed runs. Published training runs with trained-agent evaluations are a Planned item in `ROADMAP.md`. No claim of trained-agent performance should be made on behalf of this project until that lands.

## Resume-only and proprietary outcomes

The author's resume references work on multi-agent environments and agentic system delivery in employer settings. Those outcomes have no public artifacts, cannot be verified from this repository, and are not claims of this project. They are listed here only so the boundary is explicit: nothing in this repository substantiates them, and nothing about them should be inferred from this repository.
