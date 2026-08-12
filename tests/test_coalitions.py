"""Tests for coalition MoE voting."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nexus.coalitions import CoalitionManager, SplitVote
from nexus.state import (
    AgentState,
    ClusterState,
    CoalitionProposal,
    Job,
    JobPriority,
    JobStatus,
    Resources,
    ResourceType,
)


def _make_coalition_state() -> tuple[ClusterState, Job]:
    state = ClusterState()

    # Collaborative job needing 20 GPU, 40 CPU
    job = Job(
        id="J-coop",
        description="Collaborative training",
        requirements=Resources(gpu=20, cpu=40, memory=100, bandwidth=10),
        deadline=10,
        reward=500.0,
        collaborative=True,
    )

    state.agents["a0"] = AgentState(
        id="a0", team_name="Alpha",
        holdings=Resources(gpu=15, cpu=25, memory=60, bandwidth=8),
    )
    state.agents["a0"].job_queue.append(job)

    state.agents["a1"] = AgentState(
        id="a1", team_name="Beta",
        holdings=Resources(gpu=15, cpu=25, memory=60, bandwidth=8),
    )

    return state, job


def test_coalition_resolve_success():
    state, job = _make_coalition_state()
    manager = CoalitionManager()

    proposal = manager.propose(state, "a0", ["a0", "a1"], "J-coop")

    # Both members vote with splits that cover the job
    votes = [
        SplitVote(
            agent_id="a0",
            proposed_splits={
                "a0": {"gpu": 10, "cpu": 20, "memory": 50, "bandwidth": 5},
                "a1": {"gpu": 10, "cpu": 20, "memory": 50, "bandwidth": 5},
            },
            confidence=0.9,
            is_proposer=True,
        ),
        SplitVote(
            agent_id="a1",
            proposed_splits={
                "a0": {"gpu": 10, "cpu": 20, "memory": 50, "bandwidth": 5},
                "a1": {"gpu": 10, "cpu": 20, "memory": 50, "bandwidth": 5},
            },
            confidence=0.8,
        ),
    ]

    result = manager.resolve(state, proposal, votes)
    assert result.accepted
    assert job.status == JobStatus.COMPLETED


def test_coalition_veto():
    state, job = _make_coalition_state()
    manager = CoalitionManager()

    proposal = manager.propose(state, "a0", ["a0", "a1"], "J-coop")

    votes = [
        SplitVote(
            agent_id="a0",
            proposed_splits={"a0": {"gpu": 10}, "a1": {"gpu": 10}},
            confidence=0.9,
            is_proposer=True,
        ),
        SplitVote(
            agent_id="a1",
            proposed_splits={"a0": {"gpu": 10}, "a1": {"gpu": 10}},
            confidence=0.05,  # Below veto threshold
        ),
    ]

    result = manager.resolve(state, proposal, votes)
    assert not result.accepted
    assert "vetoed" in result.rejection_reason.lower()


def test_coalition_insufficient_resources():
    state, job = _make_coalition_state()
    # Reduce holdings so they can't cover the job
    state.agents["a0"].holdings = Resources(gpu=5, cpu=10, memory=20, bandwidth=2)
    state.agents["a1"].holdings = Resources(gpu=5, cpu=10, memory=20, bandwidth=2)

    manager = CoalitionManager()
    proposal = manager.propose(state, "a0", ["a0", "a1"], "J-coop")

    votes = [
        SplitVote(
            agent_id="a0",
            proposed_splits={
                "a0": {"gpu": 10, "cpu": 20, "memory": 50, "bandwidth": 5},
                "a1": {"gpu": 10, "cpu": 20, "memory": 50, "bandwidth": 5},
            },
            confidence=0.9,
            is_proposer=True,
        ),
        SplitVote(
            agent_id="a1",
            proposed_splits={
                "a0": {"gpu": 10, "cpu": 20, "memory": 50, "bandwidth": 5},
                "a1": {"gpu": 10, "cpu": 20, "memory": 50, "bandwidth": 5},
            },
            confidence=0.8,
        ),
    ]

    result = manager.resolve(state, proposal, votes)
    assert not result.accepted
