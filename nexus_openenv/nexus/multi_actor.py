"""Multi-actor mode: CTO directives + Worker agents with noise model.

CTO issues high-level directives to workers. Workers may misinterpret
or partially execute directives based on noise model.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from nexus.actions import Action, ActionType
from nexus.state import ResourceType


class DirectiveType(str, Enum):
    BID_FOR = "bid_for"
    SELL = "sell"
    ALLOCATE_JOB = "allocate_job"
    FOCUS_PRIORITY = "focus_priority"
    JOIN_COALITION = "join_coalition"
    QUERY = "query"


@dataclass
class Directive:
    type: DirectiveType
    worker_id: str
    params: dict[str, Any] = field(default_factory=dict)
    instruction: str = ""


@dataclass
class WorkerProfile:
    """Each worker has strengths and biases that affect directive execution."""
    worker_id: str
    reliability: float = 0.8  # probability of correct execution
    resource_specialty: ResourceType | None = None  # better at trading this
    speed: float = 1.0  # execution speed multiplier
    bias: str = ""  # "cautious", "aggressive", "lazy"


class DirectiveParser:
    """Parse CTO natural language directives into structured form."""

    def parse(self, text: str, worker_id: str) -> Directive:
        lower = text.lower()

        if "bid" in lower or "buy" in lower:
            return Directive(
                type=DirectiveType.BID_FOR,
                worker_id=worker_id,
                instruction=text,
            )
        elif "sell" in lower or "offer" in lower:
            return Directive(
                type=DirectiveType.SELL,
                worker_id=worker_id,
                instruction=text,
            )
        elif "allocate" in lower or "run" in lower or "execute" in lower:
            return Directive(
                type=DirectiveType.ALLOCATE_JOB,
                worker_id=worker_id,
                instruction=text,
            )
        elif "focus" in lower or "priority" in lower:
            return Directive(
                type=DirectiveType.FOCUS_PRIORITY,
                worker_id=worker_id,
                instruction=text,
            )
        elif "coalition" in lower or "team up" in lower:
            return Directive(
                type=DirectiveType.JOIN_COALITION,
                worker_id=worker_id,
                instruction=text,
            )
        else:
            return Directive(
                type=DirectiveType.QUERY,
                worker_id=worker_id,
                instruction=text,
            )


class WorkerAgent:
    """Semi-autonomous worker that executes CTO directives with noise.

    Workers may misinterpret, delay, or partially execute directives
    based on their profile and the noise model.
    """

    def __init__(self, profile: WorkerProfile, noise: float = 0.2, seed: int | None = None):
        self.profile = profile
        self.noise = noise
        self.rng = random.Random(seed)
        self.pending_directives: list[Directive] = []
        self.execution_log: list[dict[str, Any]] = []

    @property
    def agent_id(self) -> str:
        return self.profile.worker_id

    def receive_directive(self, directive: Directive) -> None:
        self.pending_directives.append(directive)

    def execute_directives(self, observation: str) -> list[Action]:
        """Execute pending directives with potential noise."""
        actions: list[Action] = []

        for directive in self.pending_directives:
            action = self._interpret_directive(directive, observation)
            if action:
                actions.append(action)
                self.execution_log.append({
                    "directive": directive.instruction,
                    "action": action.type.value,
                    "fidelity": self._compute_fidelity(directive, action),
                })

        self.pending_directives.clear()
        return actions or [Action(type=ActionType.PASS, agent_id=self.agent_id)]

    def _interpret_directive(self, directive: Directive, observation: str) -> Action | None:
        """Convert directive to action, with potential misinterpretation."""
        # Check if worker executes correctly
        if self.rng.random() > self.profile.reliability:
            return self._misinterpret(directive)

        # Add noise to parameters
        if directive.type == DirectiveType.BID_FOR:
            quantity = directive.params.get("quantity", 5.0)
            price = directive.params.get("price", 50.0)
            rt = directive.params.get("resource_type", "gpu")

            # Apply noise to quantity and price
            quantity *= 1 + self.rng.gauss(0, self.noise)
            price *= 1 + self.rng.gauss(0, self.noise)

            return Action(
                type=ActionType.BID,
                agent_id=self.agent_id,
                params={
                    "resource_type": rt,
                    "quantity": max(1, quantity),
                    "price": max(1, price),
                },
            )

        elif directive.type == DirectiveType.SELL:
            quantity = directive.params.get("quantity", 5.0)
            price = directive.params.get("price", 50.0)
            rt = directive.params.get("resource_type", "gpu")

            quantity *= 1 + self.rng.gauss(0, self.noise)
            price *= 1 + self.rng.gauss(0, self.noise)

            return Action(
                type=ActionType.OFFER,
                agent_id=self.agent_id,
                params={
                    "resource_type": rt,
                    "quantity": max(1, quantity),
                    "price": max(1, price),
                },
            )

        elif directive.type == DirectiveType.ALLOCATE_JOB:
            job_id = directive.params.get("job_id", "")
            if job_id:
                return Action(
                    type=ActionType.ALLOCATE,
                    agent_id=self.agent_id,
                    params={"job_id": job_id},
                )

        return None

    def _misinterpret(self, directive: Directive) -> Action | None:
        """Worker misinterprets the directive based on bias."""
        bias = self.profile.bias

        if bias == "cautious":
            return Action(type=ActionType.PASS, agent_id=self.agent_id)
        elif bias == "aggressive":
            # Overbid
            return Action(
                type=ActionType.BID,
                agent_id=self.agent_id,
                params={
                    "resource_type": "gpu",
                    "quantity": self.rng.uniform(5, 20),
                    "price": self.rng.uniform(80, 150),
                },
            )
        else:
            # Random action
            return Action(type=ActionType.PASS, agent_id=self.agent_id)

    def _compute_fidelity(self, directive: Directive, action: Action) -> float:
        """Measure how faithfully the directive was executed."""
        if action.type == ActionType.PASS and directive.type != DirectiveType.QUERY:
            return 0.0
        type_match = {
            DirectiveType.BID_FOR: ActionType.BID,
            DirectiveType.SELL: ActionType.OFFER,
            DirectiveType.ALLOCATE_JOB: ActionType.ALLOCATE,
        }
        expected = type_match.get(directive.type)
        if expected and action.type == expected:
            return 0.8 + self.rng.uniform(0, 0.2)
        return 0.3
