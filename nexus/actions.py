"""Action types, validation, and LLM output parsing.

Agent action interface: make_move -> act pattern.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from nexus.state import AgentState, ClusterState, ResourceType


class ActionType(str, Enum):
    ALLOCATE = "allocate"
    BID = "bid"
    OFFER = "offer"
    ACCEPT_BID = "accept_bid"
    ACCEPT_OFFER = "accept_offer"
    PROPOSE_COALITION = "propose_coalition"
    ACCEPT_COALITION = "accept_coalition"
    REJECT_COALITION = "reject_coalition"
    SEND_MESSAGE = "send_message"
    PASS = "pass"


@dataclass
class Action:
    type: ActionType
    agent_id: str = ""
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def job_id(self) -> str:
        return self.params.get("job_id", "")

    @property
    def resource_type(self) -> ResourceType | None:
        rt = self.params.get("resource_type")
        if rt:
            return ResourceType(rt) if isinstance(rt, str) else rt
        return None

    @property
    def quantity(self) -> float:
        return float(self.params.get("quantity", 0))

    @property
    def price(self) -> float:
        return float(self.params.get("price", 0))

    @property
    def target_id(self) -> str:
        return self.params.get("target_id", self.params.get("agent_id", ""))

    @property
    def bid_id(self) -> str:
        return self.params.get("bid_id", "")

    @property
    def offer_id(self) -> str:
        return self.params.get("offer_id", "")

    @property
    def proposal_id(self) -> str:
        return self.params.get("proposal_id", "")

    @property
    def message_text(self) -> str:
        return self.params.get("text", "")

    @property
    def member_ids(self) -> list[str]:
        return self.params.get("member_ids", [])


def validate_action(action: Action, agent: AgentState, state: ClusterState) -> tuple[bool, str]:
    """Validate an action against the current state. Returns (valid, reason)."""
    if not agent.can_act():
        return False, f"Agent {agent.id} has used all {agent.max_actions_per_round} actions this round"

    if action.type == ActionType.PASS:
        return True, ""

    if action.type == ActionType.ALLOCATE:
        job = next((j for j in agent.job_queue if j.id == action.job_id), None)
        if not job:
            return False, f"Job {action.job_id} not in agent's queue"
        if not agent.holdings.can_afford(job.requirements):
            return False, f"Insufficient resources for job {action.job_id}"
        return True, ""

    if action.type == ActionType.BID:
        if not action.resource_type:
            return False, "Bid must specify resource_type"
        if action.quantity <= 0:
            return False, "Bid quantity must be positive"
        if action.price <= 0:
            return False, "Bid price must be positive"
        total_cost = action.quantity * action.price
        if agent.budget < total_cost:
            return False, f"Insufficient budget: need ${total_cost:.0f}, have ${agent.budget:.0f}"
        return True, ""

    if action.type == ActionType.OFFER:
        if not action.resource_type:
            return False, "Offer must specify resource_type"
        if action.quantity <= 0:
            return False, "Offer quantity must be positive"
        held = agent.holdings.get(action.resource_type)
        if held < action.quantity:
            return False, f"Insufficient {action.resource_type.value}: have {held}, offering {action.quantity}"
        return True, ""

    if action.type == ActionType.ACCEPT_BID:
        bid = next((b for b in state.market.active_bids if b.id == action.bid_id and not b.accepted), None)
        if not bid:
            return False, f"Bid {action.bid_id} not found or already accepted"
        if bid.agent_id == agent.id:
            return False, "Cannot accept own bid"
        held = agent.holdings.get(bid.resource_type)
        if held < bid.quantity:
            return False, f"Insufficient {bid.resource_type.value} to fill bid"
        return True, ""

    if action.type == ActionType.ACCEPT_OFFER:
        offer = next((o for o in state.market.active_offers if o.id == action.offer_id and not o.accepted), None)
        if not offer:
            return False, f"Offer {action.offer_id} not found or already accepted"
        if offer.agent_id == agent.id:
            return False, "Cannot accept own offer"
        total_cost = offer.quantity * offer.price_per_unit
        if agent.budget < total_cost:
            return False, f"Insufficient budget to accept offer"
        return True, ""

    if action.type == ActionType.PROPOSE_COALITION:
        if not action.job_id:
            return False, "Coalition proposal must specify job_id"
        job = next((j for j in agent.job_queue if j.id == action.job_id), None)
        if not job:
            return False, f"Job {action.job_id} not in agent's queue"
        if not job.collaborative:
            return False, f"Job {action.job_id} is not collaborative"
        if not action.member_ids:
            return False, "Coalition must have members"
        return True, ""

    if action.type in (ActionType.ACCEPT_COALITION, ActionType.REJECT_COALITION):
        proposal = next(
            (c for c in state.coalitions if c.id == action.proposal_id and not c.accepted),
            None,
        )
        if not proposal:
            return False, f"Coalition proposal {action.proposal_id} not found"
        if agent.id not in proposal.member_ids:
            return False, "Agent not a member of this coalition"
        return True, ""

    if action.type == ActionType.SEND_MESSAGE:
        if not action.target_id:
            return False, "Message must specify target agent"
        if action.target_id not in state.agents:
            return False, f"Target agent {action.target_id} not found"
        return True, ""

    return False, f"Unknown action type: {action.type}"


def parse_llm_output(raw: str, agent_id: str) -> list[Action]:
    """Parse LLM text output into actions. Handles JSON and natural language."""
    actions: list[Action] = []

    # Try JSON array first
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            for item in parsed:
                actions.append(_parse_action_dict(item, agent_id))
            return actions
        if isinstance(parsed, dict):
            return [_parse_action_dict(parsed, agent_id)]
    except (json.JSONDecodeError, KeyError, ValueError):
        pass

    # Try extracting JSON blocks from markdown
    json_blocks = re.findall(r"```(?:json)?\s*(\{[^`]+\})\s*```", raw, re.DOTALL)
    for block in json_blocks:
        try:
            parsed = json.loads(block)
            actions.append(_parse_action_dict(parsed, agent_id))
        except (json.JSONDecodeError, KeyError, ValueError):
            continue

    if actions:
        return actions

    # Fallback: keyword parsing
    lower = raw.lower()
    if "pass" in lower or "do nothing" in lower:
        actions.append(Action(type=ActionType.PASS, agent_id=agent_id))
    elif "allocate" in lower:
        job_match = re.search(r"(?:allocate|run|execute)\s+(?:job\s+)?([Jj]-?\w+)", raw)
        if job_match:
            actions.append(
                Action(
                    type=ActionType.ALLOCATE,
                    agent_id=agent_id,
                    params={"job_id": job_match.group(1)},
                )
            )
    elif "bid" in lower:
        bid_match = re.search(
            r"bid\s+(\d+)\s+(\w+)\s+@?\s*\$?(\d+(?:\.\d+)?)", raw, re.IGNORECASE
        )
        if bid_match:
            actions.append(
                Action(
                    type=ActionType.BID,
                    agent_id=agent_id,
                    params={
                        "quantity": float(bid_match.group(1)),
                        "resource_type": bid_match.group(2).lower(),
                        "price": float(bid_match.group(3)),
                    },
                )
            )

    return actions or [Action(type=ActionType.PASS, agent_id=agent_id)]


def _parse_action_dict(d: dict, agent_id: str) -> Action:
    action_type = ActionType(d["type"])
    params = {k: v for k, v in d.items() if k not in ("type", "agent_id")}
    return Action(type=action_type, agent_id=agent_id, params=params)
