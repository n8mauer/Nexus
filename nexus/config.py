"""Environment configuration and presets."""

from __future__ import annotations

from dataclasses import dataclass, field

from nexus.state import Resources


@dataclass
class NexusConfig:
    num_agents: int = 4
    max_rounds: int = 50
    max_actions_per_round: int = 3
    starting_budget: float = 1000.0
    starting_reputation: float = 50.0

    # Cluster resources (refreshed each round)
    cluster_resources: Resources = field(
        default_factory=lambda: Resources(gpu=100, cpu=200, memory=512, bandwidth=50)
    )

    # Initial agent holdings
    initial_holdings: Resources = field(
        default_factory=lambda: Resources(gpu=20, cpu=50, memory=128, bandwidth=12)
    )

    # Per-round resource refresh (agents receive this each round)
    round_refresh: Resources = field(
        default_factory=lambda: Resources(gpu=8, cpu=16, memory=40, bandwidth=4)
    )

    # Market parameters (commission, price impact)
    commission_rate: float = 0.02  # 2% commission on trades
    price_impact: float = 0.005  # 0.5% price impact per trade

    # Job generation
    jobs_per_round_min: int = 1
    jobs_per_round_max: int = 3
    job_reward_range: tuple[float, float] = (50.0, 500.0)
    job_deadline_range: tuple[int, int] = (3, 15)
    collaborative_job_chance: float = 0.2

    # Event generation
    event_chance_per_round: float = 0.2

    # Reward weights (rule-based reward model)
    reward_weights: dict[str, float] = field(
        default_factory=lambda: {
            "job_completion": 0.4,
            "efficiency": 0.2,
            "reputation": 0.15,
            "coalition": 0.15,
            "fairness": 0.1,
        }
    )

    # Oversight
    enable_oversight: bool = False
    inject_collusion: bool = False
    collusion_pairs: list[tuple[str, str]] = field(default_factory=list)

    # Multi-actor
    enable_multi_actor: bool = False
    num_workers: int = 3
    worker_noise: float = 0.2  # probability of misinterpreting directive


PRESETS: dict[str, NexusConfig] = {
    "tiny": NexusConfig(
        num_agents=2,
        max_rounds=10,
        cluster_resources=Resources(gpu=40, cpu=80, memory=200, bandwidth=20),
        initial_holdings=Resources(gpu=10, cpu=20, memory=64, bandwidth=6),
        round_refresh=Resources(gpu=5, cpu=10, memory=25, bandwidth=3),
        jobs_per_round_min=1,
        jobs_per_round_max=2,
        event_chance_per_round=0.1,
    ),
    "standard": NexusConfig(
        num_agents=4,
        max_rounds=50,
    ),
    "large": NexusConfig(
        num_agents=8,
        max_rounds=100,
        cluster_resources=Resources(gpu=200, cpu=400, memory=1024, bandwidth=100),
    ),
    "oversight": NexusConfig(
        num_agents=4,
        max_rounds=50,
        enable_oversight=True,
    ),
    "multi_actor": NexusConfig(
        num_agents=4,
        max_rounds=50,
        enable_multi_actor=True,
        num_workers=3,
    ),
}


def get_preset(name: str) -> NexusConfig:
    if name not in PRESETS:
        raise ValueError(f"Unknown preset '{name}'. Available: {list(PRESETS.keys())}")
    return PRESETS[name]
