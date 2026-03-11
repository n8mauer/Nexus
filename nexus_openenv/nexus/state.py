"""Core state dataclasses for Nexus simulation.

Patterns from: Game.py (state container), solver_state.py (persistence).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ResourceType(str, Enum):
    GPU = "gpu"
    CPU = "cpu"
    MEMORY = "memory"
    BANDWIDTH = "bandwidth"


class JobPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class JobStatus(str, Enum):
    PENDING = "pending"
    ALLOCATED = "allocated"
    RUNNING = "running"
    COMPLETED = "completed"
    MISSED = "missed"


@dataclass
class Resources:
    gpu: float = 0.0
    cpu: float = 0.0
    memory: float = 0.0
    bandwidth: float = 0.0

    def get(self, rtype: ResourceType) -> float:
        return getattr(self, rtype.value)

    def set(self, rtype: ResourceType, value: float) -> None:
        setattr(self, rtype.value, max(0.0, value))

    def add(self, rtype: ResourceType, amount: float) -> None:
        self.set(rtype, self.get(rtype) + amount)

    def subtract(self, rtype: ResourceType, amount: float) -> bool:
        current = self.get(rtype)
        if current >= amount:
            self.set(rtype, current - amount)
            return True
        return False

    def can_afford(self, required: Resources) -> bool:
        for rt in ResourceType:
            if self.get(rt) < required.get(rt):
                return False
        return True

    def consume(self, required: Resources) -> bool:
        if not self.can_afford(required):
            return False
        for rt in ResourceType:
            self.subtract(rt, required.get(rt))
        return True

    def clone(self) -> Resources:
        return Resources(
            gpu=self.gpu, cpu=self.cpu, memory=self.memory, bandwidth=self.bandwidth
        )

    def total(self) -> float:
        return self.gpu + self.cpu + self.memory + self.bandwidth

    def to_dict(self) -> dict[str, float]:
        return {rt.value: self.get(rt) for rt in ResourceType}

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> Resources:
        return cls(**{k: d.get(k, 0.0) for k in ["gpu", "cpu", "memory", "bandwidth"]})


@dataclass
class Job:
    id: str = field(default_factory=lambda: f"J-{uuid.uuid4().hex[:6]}")
    description: str = ""
    requirements: Resources = field(default_factory=Resources)
    deadline: int = 10
    reward: float = 100.0
    priority: JobPriority = JobPriority.MEDIUM
    collaborative: bool = False
    penalty_on_miss: float = 50.0
    status: JobStatus = JobStatus.PENDING
    assigned_round: int | None = None

    def is_expired(self, current_round: int) -> bool:
        return current_round > self.deadline and self.status not in (
            JobStatus.COMPLETED,
            JobStatus.RUNNING,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "requirements": self.requirements.to_dict(),
            "deadline": self.deadline,
            "reward": self.reward,
            "priority": self.priority.value,
            "collaborative": self.collaborative,
            "penalty_on_miss": self.penalty_on_miss,
            "status": self.status.value,
        }


@dataclass
class Bid:
    id: str = field(default_factory=lambda: f"B-{uuid.uuid4().hex[:6]}")
    agent_id: str = ""
    resource_type: ResourceType = ResourceType.GPU
    quantity: float = 0.0
    price_per_unit: float = 0.0
    round_created: int = 0
    accepted: bool = False
    accepted_by: str | None = None


@dataclass
class Offer:
    id: str = field(default_factory=lambda: f"O-{uuid.uuid4().hex[:6]}")
    agent_id: str = ""
    resource_type: ResourceType = ResourceType.GPU
    quantity: float = 0.0
    price_per_unit: float = 0.0
    round_created: int = 0
    accepted: bool = False
    accepted_by: str | None = None


@dataclass
class CoalitionProposal:
    id: str = field(default_factory=lambda: f"C-{uuid.uuid4().hex[:6]}")
    proposer_id: str = ""
    member_ids: list[str] = field(default_factory=list)
    job_id: str = ""
    resource_splits: dict[str, Resources] = field(default_factory=dict)
    votes: dict[str, float] = field(default_factory=dict)  # agent_id -> confidence
    accepted: bool = False
    round_created: int = 0


@dataclass
class Message:
    sender_id: str = ""
    recipient_id: str = ""
    text: str = ""
    round_sent: int = 0


@dataclass
class Trade:
    buyer_id: str = ""
    seller_id: str = ""
    resource_type: ResourceType = ResourceType.GPU
    quantity: float = 0.0
    price_per_unit: float = 0.0
    total_cost: float = 0.0
    commission: float = 0.0
    price_impact: float = 0.0
    round_executed: int = 0


@dataclass
class AgentState:
    id: str = ""
    team_name: str = ""
    budget: float = 1000.0
    reputation: float = 50.0
    holdings: Resources = field(default_factory=Resources)
    job_queue: list[Job] = field(default_factory=list)
    completed_jobs: list[Job] = field(default_factory=list)
    score: float = 0.0
    messages_received: list[Message] = field(default_factory=list)
    trade_history: list[Trade] = field(default_factory=list)
    actions_this_round: int = 0
    max_actions_per_round: int = 3

    def can_act(self) -> bool:
        return self.actions_this_round < self.max_actions_per_round

    def active_jobs(self) -> list[Job]:
        return [j for j in self.job_queue if j.status in (JobStatus.PENDING, JobStatus.ALLOCATED)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "team_name": self.team_name,
            "budget": self.budget,
            "reputation": self.reputation,
            "holdings": self.holdings.to_dict(),
            "score": self.score,
            "job_queue": [j.to_dict() for j in self.job_queue],
            "completed_jobs": [j.to_dict() for j in self.completed_jobs],
        }


@dataclass
class MarketState:
    active_bids: list[Bid] = field(default_factory=list)
    active_offers: list[Offer] = field(default_factory=list)
    completed_trades: list[Trade] = field(default_factory=list)
    price_history: dict[ResourceType, list[float]] = field(default_factory=dict)
    base_prices: dict[ResourceType, float] = field(
        default_factory=lambda: {
            ResourceType.GPU: 100.0,
            ResourceType.CPU: 30.0,
            ResourceType.MEMORY: 20.0,
            ResourceType.BANDWIDTH: 40.0,
        }
    )

    def current_price(self, rtype: ResourceType) -> float:
        history = self.price_history.get(rtype, [])
        if history:
            return history[-1]
        return self.base_prices[rtype]


@dataclass
class GlobalEvent:
    description: str = ""
    round_triggered: int = 0
    resource_effects: Resources = field(default_factory=Resources)
    is_positive: bool = True


@dataclass
class ClusterState:
    resources: Resources = field(
        default_factory=lambda: Resources(gpu=100, cpu=200, memory=512, bandwidth=50)
    )
    current_round: int = 1
    max_rounds: int = 50
    agents: dict[str, AgentState] = field(default_factory=dict)
    market: MarketState = field(default_factory=MarketState)
    events: list[GlobalEvent] = field(default_factory=list)
    coalitions: list[CoalitionProposal] = field(default_factory=list)
    round_messages: list[Message] = field(default_factory=list)

    def is_finished(self) -> bool:
        return self.current_round > self.max_rounds

    def agent_ids(self) -> list[str]:
        return list(self.agents.keys())

    def to_dict(self) -> dict[str, Any]:
        return {
            "resources": self.resources.to_dict(),
            "current_round": self.current_round,
            "max_rounds": self.max_rounds,
            "agents": {aid: a.to_dict() for aid, a in self.agents.items()},
            "events": [{"description": e.description, "round": e.round_triggered} for e in self.events],
        }
