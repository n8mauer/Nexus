"""Random baseline agent — picks valid actions uniformly at random."""

from __future__ import annotations

import random
import re

from agents.base import BaseAgent
from nexus.actions import Action, ActionType
from nexus.state import ResourceType


class RandomAgent(BaseAgent):
    """Selects random valid actions each round."""

    def __init__(self, agent_id: str, seed: int | None = None):
        super().__init__(agent_id)
        self.rng = random.Random(seed)

    def act(self, observation: str) -> list[Action]:
        actions: list[Action] = []

        # Parse remaining actions from observation
        remaining = self._parse_remaining(observation)
        if remaining <= 0:
            return [Action(type=ActionType.PASS, agent_id=self.agent_id)]

        # Parse available jobs
        job_ids = re.findall(r"\[(J-\w+)\]", observation)

        # Parse available bids/offers
        bid_ids = re.findall(r"BID (B-\w+)", observation)
        offer_ids = re.findall(r"OFFER (O-\w+)", observation)

        for _ in range(min(remaining, self.rng.randint(1, 3))):
            action = self._random_action(job_ids, bid_ids, offer_ids)
            actions.append(action)

        return actions or [Action(type=ActionType.PASS, agent_id=self.agent_id)]

    def _random_action(
        self, job_ids: list[str], bid_ids: list[str], offer_ids: list[str]
    ) -> Action:
        choices = [ActionType.PASS]

        if job_ids:
            choices.append(ActionType.ALLOCATE)
        choices.extend([ActionType.BID, ActionType.OFFER])
        if bid_ids:
            choices.append(ActionType.ACCEPT_BID)
        if offer_ids:
            choices.append(ActionType.ACCEPT_OFFER)

        action_type = self.rng.choice(choices)

        if action_type == ActionType.ALLOCATE and job_ids:
            return Action(
                type=ActionType.ALLOCATE,
                agent_id=self.agent_id,
                params={"job_id": self.rng.choice(job_ids)},
            )

        if action_type == ActionType.BID:
            rt = self.rng.choice(list(ResourceType))
            return Action(
                type=ActionType.BID,
                agent_id=self.agent_id,
                params={
                    "resource_type": rt.value,
                    "quantity": self.rng.uniform(1, 10),
                    "price": self.rng.uniform(10, 100),
                },
            )

        if action_type == ActionType.OFFER:
            rt = self.rng.choice(list(ResourceType))
            return Action(
                type=ActionType.OFFER,
                agent_id=self.agent_id,
                params={
                    "resource_type": rt.value,
                    "quantity": self.rng.uniform(1, 5),
                    "price": self.rng.uniform(10, 100),
                },
            )

        if action_type == ActionType.ACCEPT_BID and bid_ids:
            return Action(
                type=ActionType.ACCEPT_BID,
                agent_id=self.agent_id,
                params={"bid_id": self.rng.choice(bid_ids)},
            )

        if action_type == ActionType.ACCEPT_OFFER and offer_ids:
            return Action(
                type=ActionType.ACCEPT_OFFER,
                agent_id=self.agent_id,
                params={"offer_id": self.rng.choice(offer_ids)},
            )

        return Action(type=ActionType.PASS, agent_id=self.agent_id)

    def _parse_remaining(self, observation: str) -> int:
        match = re.search(r"ACTIONS REMAINING:\s*(\d+)", observation)
        return int(match.group(1)) if match else 3
