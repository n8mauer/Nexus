"""Acceptance test for Gate 2 of docs/ACCEPTANCE.md.

Runs the seeded demo (tiny preset, greedy vs random, fixed seed) as a
subprocess and asserts:
1. The process exits with code 0.
2. final_results.json exists, parses as JSON, and contains an "agents"
   mapping with two entries.
3. metrics.jsonl exists and contains a simulation_start event and one
   round event per completed round.
4. journal.md exists and is non-empty.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_demo(output_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "demo.py"), "--output", str(output_dir)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_demo_exits_zero_and_writes_journal(tmp_path: Path) -> None:
    out = tmp_path / "demo_run"
    proc = _run_demo(out)

    # 1. Exit code 0
    assert proc.returncode == 0, (
        f"demo.py exited {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )

    # 2. final_results.json parses and has two agents
    final_path = out / "final_results.json"
    assert final_path.exists(), "final_results.json was not written"
    results = json.loads(final_path.read_text(encoding="utf-8"))
    agents = results.get("agents")
    assert isinstance(agents, dict), "final_results.json lacks an 'agents' mapping"
    assert len(agents) == 2, f"expected 2 agents, found {len(agents)}"

    # 3. metrics.jsonl has simulation_start and one round event per round
    metrics_path = out / "metrics.jsonl"
    assert metrics_path.exists(), "metrics.jsonl was not written"
    events = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    event_types = [e.get("event") for e in events]
    assert "simulation_start" in event_types, "no simulation_start event logged"
    rounds_played = results.get("rounds_played", 0)
    assert rounds_played > 0, "simulation completed zero rounds"
    round_events = [e for e in events if e.get("event") == "round"]
    assert len(round_events) == rounds_played, (
        f"expected {rounds_played} round events, found {len(round_events)}"
    )

    # 4. journal.md exists and is non-empty
    journal_path = out / "journal.md"
    assert journal_path.exists(), "journal.md was not written"
    assert journal_path.stat().st_size > 0, "journal.md is empty"


def test_demo_scoreboard_output(tmp_path: Path) -> None:
    """The demo prints real computed output: scoreboard, Gini, market volume."""
    proc = _run_demo(tmp_path / "demo_run")
    assert proc.returncode == 0
    assert "Final Scoreboard" in proc.stdout
    assert "Gini coefficient:" in proc.stdout
    assert "Market volume:" in proc.stdout
    assert "Journal written to" in proc.stdout
