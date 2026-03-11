"""Random event generation: outages, surges, bonus jobs.

Adds stochasticity and forces adaptive strategy.
"""

from __future__ import annotations

import random

from nexus.state import (
    ClusterState,
    GlobalEvent,
    Job,
    JobPriority,
    Resources,
    ResourceType,
)
from nexus.config import NexusConfig


class EventGenerator:
    """Generates random events and new jobs each round."""

    def __init__(self, config: NexusConfig, seed: int | None = None):
        self.config = config
        self.rng = random.Random(seed)
        self._job_counter = 0
        self._job_descriptions = [
            "Train recommendation model",
            "Run batch inference pipeline",
            "Fine-tune language model",
            "Process video transcoding queue",
            "Execute distributed map-reduce",
            "Run Monte Carlo simulations",
            "Compile large codebase",
            "Deploy containerized microservices",
            "Index document embeddings",
            "Run automated test suite",
            "Generate synthetic training data",
            "Optimize database queries",
            "Process real-time data stream",
            "Train image classification model",
            "Run genomics analysis pipeline",
        ]
        self._event_templates = [
            ("GPU cluster sector {sector} offline -- {n} GPU units unavailable this round", ResourceType.GPU, False),
            ("Memory bank failure -- {n} GB unavailable", ResourceType.MEMORY, False),
            ("Network congestion -- bandwidth reduced by {n} Gbps", ResourceType.BANDWIDTH, False),
            ("Surplus capacity released -- {n} extra CPU units available", ResourceType.CPU, True),
            ("Emergency GPU allocation from reserve -- {n} GPU units added", ResourceType.GPU, True),
            ("Bandwidth upgrade -- {n} Gbps extra capacity", ResourceType.BANDWIDTH, True),
            ("Demand surge -- all resource prices increase 20% this round", None, False),
            ("Efficiency bonus -- all completed jobs pay 10% extra this round", None, True),
        ]

    def generate_events(self, state: ClusterState) -> list[GlobalEvent]:
        """Generate random events for the current round."""
        events: list[GlobalEvent] = []

        if self.rng.random() < self.config.event_chance_per_round:
            template, rtype, is_positive = self.rng.choice(self._event_templates)
            n = self.rng.randint(5, 25)
            sector = self.rng.choice(["A", "B", "C", "D"])
            description = template.format(n=n, sector=sector)

            effects = Resources()
            if rtype:
                amount = float(n) if is_positive else -float(n)
                effects.set(rtype, amount)

            event = GlobalEvent(
                description=description,
                round_triggered=state.current_round,
                resource_effects=effects,
                is_positive=is_positive,
            )
            events.append(event)

            # Apply resource effects to cluster
            if rtype:
                current = state.resources.get(rtype)
                new_val = max(0, current + (n if is_positive else -n))
                state.resources.set(rtype, new_val)

        return events

    def generate_jobs(self, state: ClusterState) -> dict[str, list[Job]]:
        """Generate new jobs for each agent. Returns {agent_id: [jobs]}."""
        jobs_by_agent: dict[str, list[Job]] = {}

        for agent_id in state.agent_ids():
            num_jobs = self.rng.randint(
                self.config.jobs_per_round_min, self.config.jobs_per_round_max
            )
            jobs: list[Job] = []
            for _ in range(num_jobs):
                job = self._make_job(state.current_round)
                jobs.append(job)
            jobs_by_agent[agent_id] = jobs

        return jobs_by_agent

    def _make_job(self, current_round: int) -> Job:
        self._job_counter += 1
        description = self.rng.choice(self._job_descriptions)
        priority = self.rng.choice(list(JobPriority))

        # Scale requirements and reward by priority
        priority_mult = {"low": 0.5, "medium": 1.0, "high": 1.5, "critical": 2.5}
        mult = priority_mult[priority.value]

        min_reward, max_reward = self.config.job_reward_range
        reward = self.rng.uniform(min_reward, max_reward) * mult
        penalty = reward * self.rng.uniform(0.2, 0.6)

        min_dl, max_dl = self.config.job_deadline_range
        deadline = current_round + self.rng.randint(min_dl, max_dl)

        requirements = Resources(
            gpu=self.rng.uniform(2, 20) * mult,
            cpu=self.rng.uniform(5, 40) * mult,
            memory=self.rng.uniform(8, 128) * mult,
            bandwidth=self.rng.uniform(1, 10) * mult,
        )

        collaborative = self.rng.random() < self.config.collaborative_job_chance

        return Job(
            id=f"J-{self._job_counter:04d}",
            description=description,
            requirements=requirements,
            deadline=deadline,
            reward=reward,
            priority=priority,
            collaborative=collaborative,
            penalty_on_miss=penalty,
        )
