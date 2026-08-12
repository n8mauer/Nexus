# ADR-003: One canonical package with a thin OpenEnv adapter and a generated Space staging directory

Date: 2026-08-12
Status: Accepted

## Context

Nexus is consumed three ways: as an installable Python package, as an OpenEnv HTTP environment, and as a Hugging Face Space. The Space requires a flat layout with the OpenEnv modules at the repository root and its own Dockerfile. Earlier, that layout was committed as a `hf_space_staging/` directory that duplicated every package in the repository. The copies drifted from the canonical sources and doubled the maintenance cost of every change.

## Decision

Keep exactly one copy of each package and generate the Space layout on demand (adopted 2026-08-12, commit 93c999a).

- `nexus/` and `agents/` are the only copies of the simulation and agent code.
- `nexus_openenv/` is a thin adapter: `server/nexus_environment.py` wraps `SimulationEngine` behind OpenEnv's `Environment` interface using a `ProxyAgent`, `server/app.py` builds the FastAPI app via `openenv.core`'s `create_app`, and `client.py` provides the typed `NexusEnv` client. Adapter modules use a try/except import shim so they work both in-package and in the Space's flat layout.
- `deploy/space/` holds only Space-specific overlay files: Dockerfile, Space README with frontmatter, pyproject, requirements, and `uv.lock`.
- `scripts/build_space.py` assembles the deployable staging directory into `build/space/` from a declarative copy list, excluding `__pycache__` artifacts. The result is uploaded with `hf upload`; the staging output itself is never committed.

## Consequences

- Code changes happen in one place; the Space cannot silently diverge from the tested sources, and `tests/test_openenv.py` exercises the same modules the Space ships.
- Deployment gains a mandatory build step: forgetting to rerun `python scripts/build_space.py` before `hf upload` ships a stale Space.
- The exact deployed tree is reproducible from the repository but not directly inspectable in it; reviewers must run the build script to see the flat layout.
- The copy list in `build_space.py` must be updated when new top-level modules are added, otherwise they are silently absent from the Space.
