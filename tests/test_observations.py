"""Tests for observation rendering."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nexus.observations import render, render_supervisor, _gini
from nexus.state import (
    AgentState,
    ClusterState,
    GlobalEvent,
    Job,
    JobPriority,
    Message,
    Resources,
    ResourceType,
)


def _make_state() -> ClusterState:
    state = ClusterState(
        resources=Resources(gpu=45, cpu=120, memory=256, bandwidth=30),
        current_round=5,
        max_rounds=50,
    )

    job1 = Job(
        id="J-0012",
        description="Train recommendation model",
        requirements=Resources(gpu=20, cpu=40, memory=128, bandwidth=5),
        deadline=8,
        reward=500.0,
        collaborative=True,
    )
    job2 = Job(
        id="J-0015",
        description="Run batch inference",
        requirements=Resources(gpu=5, cpu=10, memory=32, bandwidth=2),
        deadline=6,
        reward=150.0,
    )

    state.agents["alpha"] = AgentState(
        id="alpha",
        team_name="Team Alpha",
        budget=1240.0,
        reputation=78.0,
        holdings=Resources(gpu=12, cpu=30, memory=64, bandwidth=10),
        job_queue=[job1, job2],
        score=450.0,
    )
    state.agents["beta"] = AgentState(
        id="beta",
        team_name="Team Beta",
        budget=900.0,
        reputation=85.0,
        holdings=Resources(gpu=20, cpu=40, memory=100, bandwidth=8),
        score=380.0,
    )

    # Event
    state.events.append(GlobalEvent(
        description="GPU cluster sector B offline -- 20 GPU units unavailable this round",
        round_triggered=5,
    ))

    # Message
    msg = Message(sender_id="beta", recipient_id="alpha", text="Want to split J-0012?", round_sent=5)
    state.agents["alpha"].messages_received.append(msg)
    state.round_messages.append(msg)

    return state


def test_render_basic():
    state = _make_state()
    obs = render(state, "alpha")

    assert "ROUND 5 of 50" in obs
    assert "Team Alpha" in obs
    assert "Budget: $1240" in obs
    assert "J-0012" in obs
    assert "J-0015" in obs
    assert "COLLABORATIVE" in obs
    assert "REPUTATION BOARD" in obs
    assert "ACTIONS REMAINING" in obs


def test_render_hides_others_jobs():
    state = _make_state()
    obs = render(state, "beta")

    # Beta should NOT see Alpha's jobs
    assert "J-0012" not in obs
    assert "J-0015" not in obs


def test_render_shows_messages():
    state = _make_state()
    obs = render(state, "alpha")

    assert "Want to split J-0012?" in obs
    assert "Team Beta" in obs


def test_render_shows_events():
    state = _make_state()
    obs = render(state, "alpha")

    assert "GPU cluster sector B offline" in obs


def test_render_supervisor():
    state = _make_state()
    obs = render_supervisor(state)

    assert "SUPERVISOR VIEW" in obs
    assert "Team Alpha" in obs
    assert "Team Beta" in obs
    # Supervisor sees all jobs
    assert "J-0012" in obs
    assert "ALL MESSAGES" in obs
    assert "AGGREGATE STATISTICS" in obs


def test_gini():
    # Perfect equality
    assert _gini([100, 100, 100, 100]) == 0.0

    # Some inequality
    g = _gini([0, 0, 0, 100])
    assert g > 0.5

    # Edge cases
    assert _gini([]) == 0.0
    assert _gini([0]) == 0.0
