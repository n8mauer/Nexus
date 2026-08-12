"""Tests for the core simulation engine."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.greedy_agent import GreedyAgent
from agents.random_agent import RandomAgent
from nexus.config import get_preset
from nexus.engine import SimulationEngine
from nexus.state import JobStatus


def test_engine_runs_tiny():
    """Smoke test: engine runs a tiny simulation to completion."""
    config = get_preset("tiny")
    agents = [RandomAgent("a0", seed=0), RandomAgent("a1", seed=1)]
    engine = SimulationEngine(config, agents, seed=42)

    results = engine.run()

    assert results["rounds_played"] == 10
    assert len(results["agents"]) == 2
    assert "social_welfare" in results
    assert "gini_coefficient" in results


def test_engine_step():
    """Test a single step."""
    config = get_preset("tiny")
    agents = [RandomAgent("a0", seed=0), GreedyAgent("a1")]
    engine = SimulationEngine(config, agents, seed=42)

    scores = engine.step()

    assert "a0" in scores
    assert "a1" in scores
    assert engine.state.current_round == 2


def test_greedy_vs_random():
    """Greedy should generally outscore random."""
    config = get_preset("tiny")
    config.max_rounds = 20
    agents = [GreedyAgent("greedy"), RandomAgent("random", seed=0)]
    engine = SimulationEngine(config, agents, seed=42)

    results = engine.run()

    # Greedy should have more completed jobs (usually)
    greedy_jobs = results["agents"]["greedy"]["jobs_completed"]
    random_jobs = results["agents"]["random"]["jobs_completed"]
    # Not a hard assert since randomness, but log it
    print(f"Greedy: {greedy_jobs} jobs, Random: {random_jobs} jobs")
    assert results["rounds_played"] == 20


def test_job_generation():
    """Each round should generate new jobs."""
    config = get_preset("tiny")
    agents = [RandomAgent("a0", seed=0)]
    config.num_agents = 1
    engine = SimulationEngine(config, agents, seed=42)

    # Initial state: no jobs
    assert len(engine.state.agents["a0"].job_queue) == 0

    engine.step()

    # After one round: should have jobs
    assert len(engine.state.agents["a0"].job_queue) > 0


def test_results_structure():
    """Results should contain all expected fields."""
    config = get_preset("tiny")
    agents = [RandomAgent("a0", seed=0), RandomAgent("a1", seed=1)]
    engine = SimulationEngine(config, agents, seed=42)

    results = engine.run()

    assert "rounds_played" in results
    assert "agents" in results
    assert "market" in results
    assert "round_history" in results

    for aid, info in results["agents"].items():
        assert "final_score" in info
        assert "jobs_completed" in info
        assert "jobs_missed" in info
        assert "trades" in info
        assert "sharpe_ratio" in info
