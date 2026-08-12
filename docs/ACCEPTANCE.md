# Acceptance Criteria

Date: 2026-08-12

These are the gates a change must pass before it is considered releasable. As of 2026-08-12 all three gates are CI-enforced by `.github/workflows/ci.yml`; Gate 3 keeps a documented local fallback in case it ever proves flaky in CI (see its section). CI targets ubuntu-latest with Python 3.11, invokes Python as `python`, and installs dependencies with `pip install -c requirements.lock -e ".[dev]"` so the committed lock file is exercised on every run.

## Gate 1: Full test suite passes (CI-enforced)

**Statement:** `python -m pytest` collects exactly the repository test suite (42 tests as of 2026-08-12) and every test passes.

- Enforced by: the test modules `tests/test_engine.py`, `tests/test_market.py`, `tests/test_coalitions.py`, `tests/test_observations.py`, `tests/test_openenv.py`, `tests/test_oversight.py`, and `tests/test_demo.py`, run by the CI workflow at `.github/workflows/ci.yml`.
- Command: `python -m pytest` from the repository root after `pip install -e ".[dev]"`.
- Pass condition: exit code 0, zero failures, zero errors. Skips are acceptable only for optional-dependency guards already present in the suite (for example the openenv import guard in `tests/test_openenv.py`).

## Gate 2: Seeded demo completes and writes a results journal (CI-enforced)

**Statement:** A demo simulation using the `tiny` preset with one greedy and one random agent and a fixed seed runs to completion, exits with code 0, and writes a results journal to its output directory: `journal.md`, `metrics.jsonl`, and `final_results.json` (produced by `nexus/journal.py` and `nexus/persistence.py`).

- Enforced by: `tests/test_demo.py`, which invokes the demo as a subprocess and asserts the exit condition and the presence and basic shape of the journal files; CI additionally runs `python demo.py` directly and relies on its exit code. Demo entry point: `demo.py` at the repository root, a thin wrapper around the implementation in `scripts/demo.py`, running the same scenario as `python -m scripts.run_simulation run --preset tiny --agents greedy,random --seed 42 --output <dir>` (the demo additionally seeds the random agent with the run seed, so the two commands do not produce byte-identical journals).
- Pass conditions, written as testable statements:
  1. The process exits with code 0.
  2. `final_results.json` exists, parses as JSON, and contains an `agents` mapping with two entries.
  3. `metrics.jsonl` exists and contains a `simulation_start` event and one `round` event per completed round.
  4. `journal.md` exists and is non-empty.

## Gate 3: OpenEnv server round trip (CI if reliable headless, otherwise local-only)

**Statement:** The OpenEnv HTTP server starts from the installed package, its `/health` endpoint reports healthy, and a client `reset()` followed by one `step()` succeeds.

- Server command (a repo-local `.venv` already exists on the development machine; reuse it locally): `python -m uvicorn nexus_openenv.server.app:app --port 8000`.
- Pass conditions, written as testable statements:
  1. The server process starts without error with `openenv-core` 0.2.x installed.
  2. `GET /health` returns a healthy status.
  3. `NexusEnv(base_url="http://localhost:8000").reset()` (from `nexus_openenv/client.py`) returns an observation with `round_number == 0` and `done == False`.
  4. `step(NexusAction(raw_text="pass"))` returns an observation with `round_number == 1` and a float reward.
- Enforced by: the smoke script at `scripts/openenv_smoke.py`, which starts the server as a subprocess, performs the four checks above, and exits 0 on success, 1 on failure. It accepts `--port` and `--timeout` options.
- CI placement: as of 2026-08-12 this gate runs in `.github/workflows/ci.yml` (step "Gate 3, OpenEnv server round trip"). CI installs `openenv-core` from `requirements.lock`, which pins `openenv-core==0.2.1`, the same version pinned in `deploy/space/uv.lock`. The pin is deliberate: `openenv-core` 0.2.2 and later change `EnvClient` to async by default, which breaks the synchronous `NexusEnv` usage in `nexus_openenv/client.py` (see `constraints.in`). Fallback: if this step ever proves flaky in CI, demote it to local-only (run `python scripts/openenv_smoke.py` before release) and keep CI to Gates 1 and 2. The in-process OpenEnv logic remains CI-covered either way through `tests/test_openenv.py`.

## Out of scope for these gates

Trained-agent performance, GRPO training runs, and leaderboard results are not acceptance criteria; none exist in this repository (see `docs/RESULTS.md` and the Planned section of `ROADMAP.md`).
