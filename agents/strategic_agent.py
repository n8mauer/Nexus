"""Strategic agent using MCTS for negotiation planning.

MCTS strategy search with UCB1 for negotiation planning.
MCTS tree search plans negotiation moves under hidden information,
with UCB1 balancing exploring new tactics vs exploiting proven ones.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field
from typing import Any

from agents.base import BaseAgent
from nexus.actions import Action, ActionType
from nexus.state import ResourceType


@dataclass
class StrategyNode:
    """Node in the MCTS negotiation strategy tree.

    MCTS ThoughtNode for tree search.
    """
    action: Action | None = None
    description: str = ""
    parent: StrategyNode | None = None
    children: list[StrategyNode] = field(default_factory=list)
    visits: int = 0
    total_reward: float = 0.0
    depth: int = 0

    @property
    def average_reward(self) -> float:
        return self.total_reward / self.visits if self.visits > 0 else 0.0

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0


class MCTSStrategySearch:
    """MCTS-based negotiation strategy planner.

    UCB1 selection balances exploration vs exploitation.
    Rollouts simulate opponent responses to estimate action value.
    """

    def __init__(
        self,
        iterations: int = 50,
        exploration_constant: float = 1.414,
        max_depth: int = 3,
        seed: int | None = None,
    ):
        self.iterations = iterations
        self.C = exploration_constant
        self.max_depth = max_depth
        self.rng = random.Random(seed)

    def search(
        self,
        available_actions: list[Action],
        context: dict[str, Any],
    ) -> Action:
        """Find best action via MCTS. Returns the best action."""
        if not available_actions:
            return Action(type=ActionType.PASS)

        root = StrategyNode(description="root")
        root.visits = 1

        # Expand root with all available actions
        for action in available_actions:
            child = StrategyNode(
                action=action,
                description=f"{action.type.value}",
                parent=root,
                depth=1,
            )
            root.children.append(child)

        for _ in range(self.iterations):
            # 1. Selection
            leaf = self._select(root)

            # 2. Expansion
            if leaf.visits > 0 and leaf.depth < self.max_depth:
                child = self._expand(leaf, context)
            else:
                child = leaf

            # 3. Simulation (rollout)
            reward = self._simulate(child, context)

            # 4. Backpropagation
            self._backpropagate(child, reward)

        # Pick best child by visit count (robust selection)
        best = max(root.children, key=lambda c: c.visits)
        return best.action or Action(type=ActionType.PASS)

    def _select(self, node: StrategyNode) -> StrategyNode:
        """Select leaf using UCB1."""
        current = node
        while current.children:
            unvisited = [c for c in current.children if c.visits == 0]
            if unvisited:
                return self.rng.choice(unvisited)
            current = self._best_child_ucb1(current)
        return current

    def _best_child_ucb1(self, node: StrategyNode) -> StrategyNode:
        """UCB1 = exploitation + C * sqrt(ln(N) / n)"""
        log_parent = math.log(node.visits) if node.visits > 0 else 0
        best_ucb = float("-inf")
        best_child = node.children[0]

        for child in node.children:
            if child.visits == 0:
                return child
            exploitation = child.total_reward / child.visits
            exploration = self.C * math.sqrt(log_parent / child.visits)
            ucb = exploitation + exploration
            if ucb > best_ucb:
                best_ucb = ucb
                best_child = child

        return best_child

    def _expand(self, node: StrategyNode, context: dict[str, Any]) -> StrategyNode:
        """Expand a leaf node with possible opponent responses."""
        # Simulate opponent's best response to our action
        responses = ["accept", "reject", "counter"]
        response = self.rng.choice(responses)

        child = StrategyNode(
            description=f"opponent_{response}",
            parent=node,
            depth=node.depth + 1,
        )
        node.children.append(child)
        return child

    def _simulate(self, node: StrategyNode, context: dict[str, Any]) -> float:
        """Rollout: estimate value of reaching this node.

        Uses simple heuristic scoring based on action type and context.
        """
        # Walk up to find the root action
        current = node
        while current.parent and current.parent.action is None and current.parent.parent:
            current = current.parent

        action = current.action
        if not action:
            return 0.0

        # Heuristic scoring
        budget = context.get("budget", 1000)
        num_jobs = context.get("num_jobs", 0)
        holdings_total = context.get("holdings_total", 0)
        needs_total = context.get("needs_total", 0)

        base = 0.0

        if action.type == ActionType.ALLOCATE:
            # High value — completing jobs is the primary goal
            base = 0.8 + self.rng.uniform(0, 0.2)

        elif action.type == ActionType.BID:
            # Value depends on whether we need the resource
            if needs_total > holdings_total:
                base = 0.5 + self.rng.uniform(0, 0.3)
            else:
                base = 0.2

        elif action.type == ActionType.OFFER:
            # Selling excess is moderately valuable
            if holdings_total > needs_total:
                base = 0.4 + self.rng.uniform(0, 0.2)
            else:
                base = 0.1

        elif action.type in (ActionType.ACCEPT_BID, ActionType.ACCEPT_OFFER):
            base = 0.5 + self.rng.uniform(0, 0.2)

        elif action.type == ActionType.PROPOSE_COALITION:
            base = 0.4 + self.rng.uniform(0, 0.3)

        elif action.type == ActionType.PASS:
            base = 0.05

        return base

    def _backpropagate(self, node: StrategyNode, reward: float) -> None:
        """Propagate reward back to root."""
        current: StrategyNode | None = node
        while current is not None:
            current.visits += 1
            current.total_reward += reward
            current = current.parent


class StrategicAgent(BaseAgent):
    """Agent that uses MCTS to plan negotiation strategies."""

    def __init__(self, agent_id: str, mcts_iterations: int = 50, seed: int | None = None):
        super().__init__(agent_id)
        self.mcts = MCTSStrategySearch(iterations=mcts_iterations, seed=seed)
        self.rng = random.Random(seed)

    def act(self, observation: str) -> list[Action]:
        remaining = self._parse_remaining(observation)
        if remaining <= 0:
            return [Action(type=ActionType.PASS, agent_id=self.agent_id)]

        context = self._build_context(observation)
        candidates = self._generate_candidates(observation)

        actions: list[Action] = []
        for _ in range(min(remaining, 3)):
            if not candidates:
                break
            best = self.mcts.search(candidates, context)
            actions.append(best)
            # Remove chosen action from candidates
            candidates = [c for c in candidates if c is not best]

        return actions or [Action(type=ActionType.PASS, agent_id=self.agent_id)]

    def _generate_candidates(self, observation: str) -> list[Action]:
        """Generate all plausible actions from the observation."""
        candidates: list[Action] = []

        # Parse jobs we can allocate
        for m in re.finditer(r"\[(J-\w+)\]", observation):
            candidates.append(Action(
                type=ActionType.ALLOCATE,
                agent_id=self.agent_id,
                params={"job_id": m.group(1)},
            ))

        # Bids for each resource type
        for rt in ResourceType:
            candidates.append(Action(
                type=ActionType.BID,
                agent_id=self.agent_id,
                params={
                    "resource_type": rt.value,
                    "quantity": self.rng.uniform(2, 10),
                    "price": self.rng.uniform(20, 80),
                },
            ))

        # Accept available bids/offers
        for m in re.finditer(r"BID (B-\w+)", observation):
            candidates.append(Action(
                type=ActionType.ACCEPT_BID,
                agent_id=self.agent_id,
                params={"bid_id": m.group(1)},
            ))
        for m in re.finditer(r"OFFER (O-\w+)", observation):
            candidates.append(Action(
                type=ActionType.ACCEPT_OFFER,
                agent_id=self.agent_id,
                params={"offer_id": m.group(1)},
            ))

        candidates.append(Action(type=ActionType.PASS, agent_id=self.agent_id))
        return candidates

    def _build_context(self, observation: str) -> dict[str, Any]:
        budget = 1000.0
        m = re.search(r"Budget:\s*\$([\d.]+)", observation)
        if m:
            budget = float(m.group(1))

        num_jobs = len(re.findall(r"\[J-\w+\]", observation))

        m = re.search(
            r"Holdings:\s*([\d.]+)\s+GPU\s*\|\s*([\d.]+)\s+CPU\s*\|\s*([\d.]+)\s+GB\s+RAM\s*\|\s*([\d.]+)\s+Gbps",
            observation,
        )
        holdings_total = 0.0
        if m:
            holdings_total = sum(float(m.group(i)) for i in range(1, 5))

        return {
            "budget": budget,
            "num_jobs": num_jobs,
            "holdings_total": holdings_total,
            "needs_total": holdings_total * 0.8,  # estimated
        }

    def _parse_remaining(self, obs: str) -> int:
        m = re.search(r"ACTIONS REMAINING:\s*(\d+)", obs)
        return int(m.group(1)) if m else 3
