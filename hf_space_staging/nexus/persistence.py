"""Simulation state save/load.

JSON save/load with iteration records for simulation state.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nexus.state import ClusterState


def save_state(state: ClusterState, path: str | Path) -> None:
    """Save simulation state to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = state.to_dict()
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def save_results(results: dict[str, Any], path: str | Path) -> None:
    """Save simulation results to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)


def load_results(path: str | Path) -> dict[str, Any]:
    """Load simulation results from JSON."""
    with open(path) as f:
        return json.load(f)
