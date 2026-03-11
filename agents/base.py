"""Base agent ABC with memory and behavioral policy.

Agent act interface with profile, memory, reputation, ExperienceReplayBuffer, and BehavioralPolicy.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from nexus.actions import Action
from nexus.rewards import RoundScore


@dataclass
class Experience:
    """Single experience entry for replay buffer."""
    round_num: int
    observation: str
    actions: list[Action]
    score: RoundScore | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ExperienceReplayBuffer:
    """Fixed-size buffer of past experiences for learning."""

    def __init__(self, max_size: int = 100):
        self.buffer: deque[Experience] = deque(maxlen=max_size)

    def add(self, experience: Experience) -> None:
        self.buffer.append(experience)

    def sample(self, n: int) -> list[Experience]:
        n = min(n, len(self.buffer))
        return random.sample(list(self.buffer), n)

    def best(self, n: int = 5) -> list[Experience]:
        scored = [e for e in self.buffer if e.score is not None]
        scored.sort(key=lambda e: e.score.total, reverse=True)
        return scored[:n]

    def recent(self, n: int = 5) -> list[Experience]:
        return list(self.buffer)[-n:]


@dataclass
class PolicyWeights:
    """Tunable strategy weights adjusted by feedback."""
    aggression: float = 0.5        # bid aggressively vs conservatively
    cooperation: float = 0.5       # willingness to join coalitions
    hoarding: float = 0.3          # tendency to hold excess resources
    trading: float = 0.5           # frequency of market participation
    risk_tolerance: float = 0.5    # willingness to take high-reward/high-risk jobs

    def adjust(self, dimension: str, delta: float) -> None:
        current = getattr(self, dimension, None)
        if current is not None:
            setattr(self, dimension, max(0.0, min(1.0, current + delta)))

    def to_dict(self) -> dict[str, float]:
        return {
            "aggression": self.aggression,
            "cooperation": self.cooperation,
            "hoarding": self.hoarding,
            "trading": self.trading,
            "risk_tolerance": self.risk_tolerance,
        }


class BehavioralPolicy:
    """Adapts strategy weights based on feedback signals.

    FEEDBACK_TO_POLICY mapping adjusts weights by feedback category.
    """

    FEEDBACK_TO_POLICY: dict[str, list[tuple[str, float]]] = {
        "low_score": [
            ("aggression", 0.05),
            ("trading", 0.03),
        ],
        "missed_deadline": [
            ("risk_tolerance", -0.05),
            ("aggression", 0.03),
        ],
        "successful_trade": [
            ("trading", 0.02),
        ],
        "coalition_success": [
            ("cooperation", 0.05),
        ],
        "coalition_failure": [
            ("cooperation", -0.03),
        ],
        "high_idle_resources": [
            ("hoarding", -0.04),
            ("trading", 0.03),
        ],
        "oversight_flag": [
            ("aggression", -0.05),
            ("cooperation", 0.03),
        ],
    }

    def __init__(self):
        self.weights = PolicyWeights()
        self._learning_rate = 1.0
        self._history: list[dict[str, Any]] = []

    def apply_feedback(self, category: str, severity: float = 1.0) -> dict[str, float]:
        adjustments = self.FEEDBACK_TO_POLICY.get(category, [])
        changes: dict[str, float] = {}

        for dimension, base_delta in adjustments:
            delta = base_delta * severity * self._learning_rate
            self.weights.adjust(dimension, delta)
            changes[dimension] = getattr(self.weights, dimension)

        self._history.append({
            "category": category,
            "severity": severity,
            "changes": changes,
        })

        self._learning_rate = max(0.1, self._learning_rate * 0.995)
        return changes


class BaseAgent(ABC):
    """Abstract base agent — all agents implement this.

    make_move -> act interface with profile and memory.
    """

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.memory = ExperienceReplayBuffer()
        self.policy = BehavioralPolicy()
        self._round_num = 0

    @abstractmethod
    def act(self, observation: str) -> list[Action]:
        """Given an observation string, return a list of actions."""
        ...

    def on_round_end(self, score: RoundScore) -> None:
        """Called after each round with the score. Override to learn."""
        self._round_num += 1

        # Auto-adapt policy based on score signals
        if score.deadline_penalties > 0:
            self.policy.apply_feedback("missed_deadline", severity=min(2.0, score.deadline_penalties / 100))
        if score.idle_resource_cost > 20:
            self.policy.apply_feedback("high_idle_resources")
        if score.coalition_bonus > 0:
            self.policy.apply_feedback("coalition_success")
        if score.total < 0:
            self.policy.apply_feedback("low_score", severity=1.5)
