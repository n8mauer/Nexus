"""LLM-backed agent using Anthropic tool-use API.

Tool-use API with TOOL_DEFINITIONS and tool dispatch. Hybrid RAG retrieval for agent memory.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import anthropic

from agents.base import BaseAgent, Experience
from nexus.actions import Action, ActionType, parse_llm_output

logger = logging.getLogger(__name__)

TOOL_DEFINITIONS = [
    {
        "name": "allocate",
        "description": "Assign held resources to run a job from your queue. The job must be in your queue and you must have sufficient resources.",
        "input_schema": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string", "description": "The job ID to allocate resources to (e.g., 'J-0001')"},
            },
            "required": ["job_id"],
        },
    },
    {
        "name": "bid",
        "description": "Place a public bid to BUY resources from other agents.",
        "input_schema": {
            "type": "object",
            "properties": {
                "resource_type": {"type": "string", "enum": ["gpu", "cpu", "memory", "bandwidth"]},
                "quantity": {"type": "number", "description": "Number of units to buy"},
                "price": {"type": "number", "description": "Price per unit you're willing to pay"},
            },
            "required": ["resource_type", "quantity", "price"],
        },
    },
    {
        "name": "offer",
        "description": "Offer to SELL resources to other agents on the market.",
        "input_schema": {
            "type": "object",
            "properties": {
                "resource_type": {"type": "string", "enum": ["gpu", "cpu", "memory", "bandwidth"]},
                "quantity": {"type": "number", "description": "Number of units to sell"},
                "price": {"type": "number", "description": "Asking price per unit"},
            },
            "required": ["resource_type", "quantity", "price"],
        },
    },
    {
        "name": "accept_bid",
        "description": "Accept another agent's bid — you sell them the resources they want.",
        "input_schema": {
            "type": "object",
            "properties": {
                "bid_id": {"type": "string", "description": "The bid ID to accept (e.g., 'B-abc123')"},
            },
            "required": ["bid_id"],
        },
    },
    {
        "name": "accept_offer",
        "description": "Accept another agent's sell offer — you buy their resources.",
        "input_schema": {
            "type": "object",
            "properties": {
                "offer_id": {"type": "string", "description": "The offer ID to accept (e.g., 'O-abc123')"},
            },
            "required": ["offer_id"],
        },
    },
    {
        "name": "propose_coalition",
        "description": "Propose splitting a collaborative job with other agents.",
        "input_schema": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string", "description": "The collaborative job to split"},
                "member_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Agent IDs to include in the coalition",
                },
            },
            "required": ["job_id", "member_ids"],
        },
    },
    {
        "name": "send_message",
        "description": "Send a free-text negotiation message to another agent.",
        "input_schema": {
            "type": "object",
            "properties": {
                "target_id": {"type": "string", "description": "Agent ID to message"},
                "text": {"type": "string", "description": "Message content"},
            },
            "required": ["target_id", "text"],
        },
    },
    {
        "name": "pass_turn",
        "description": "Do nothing this turn. Use when no beneficial action is available.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
]

SYSTEM_PROMPT = """You are an AI agent competing in Nexus, a multi-agent compute cluster negotiation game.

Your goal is to maximize your score by:
1. Completing jobs from your queue before their deadlines (primary income)
2. Trading resources on the market to fill shortfalls or sell excess
3. Forming coalitions for collaborative jobs
4. Managing your budget and reputation

Strategy tips:
- Prioritize high-reward jobs with tight deadlines
- Buy resources you need at fair prices; sell resources you won't use
- Build reputation through reliable trades and job completion
- Track other agents' patterns to negotiate better

You have up to 3 actions per round. Use your tools to take actions.
Think step-by-step about which actions maximize expected value."""


class LLMAgent(BaseAgent):
    """Agent backed by Claude API with tool-use for structured actions."""

    def __init__(
        self,
        agent_id: str,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1024,
    ):
        super().__init__(agent_id)
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self._conversation_history: list[dict] = []

    def act(self, observation: str) -> list[Action]:
        # Build context from memory
        memory_context = self._build_memory_context()

        messages = [
            {
                "role": "user",
                "content": f"{memory_context}\n\n{observation}\n\nChoose your actions wisely. Use the available tools.",
            }
        ]

        actions: list[Action] = []

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                tools=TOOL_DEFINITIONS,
                messages=messages,
            )

            for content in response.content:
                if content.type == "tool_use":
                    action = self._dispatch_tool(content.name, content.input)
                    if action:
                        actions.append(action)

            # If no tool calls, try parsing text response
            if not actions:
                for content in response.content:
                    if content.type == "text" and content.text.strip():
                        actions = parse_llm_output(content.text, self.agent_id)

        except anthropic.APIError as e:
            logger.warning("LLM API error for %s: %s", self.agent_id, e)
        except Exception as e:
            logger.error("Unexpected error in LLM agent %s: %s", self.agent_id, e)

        # Store experience
        self.memory.add(Experience(
            round_num=self._round_num,
            observation=observation,
            actions=actions,
        ))

        return actions or [Action(type=ActionType.PASS, agent_id=self.agent_id)]

    def _dispatch_tool(self, tool_name: str, tool_input: dict[str, Any]) -> Action | None:
        """Convert tool call to Action."""
        type_map = {
            "allocate": ActionType.ALLOCATE,
            "bid": ActionType.BID,
            "offer": ActionType.OFFER,
            "accept_bid": ActionType.ACCEPT_BID,
            "accept_offer": ActionType.ACCEPT_OFFER,
            "propose_coalition": ActionType.PROPOSE_COALITION,
            "send_message": ActionType.SEND_MESSAGE,
            "pass_turn": ActionType.PASS,
        }

        action_type = type_map.get(tool_name)
        if not action_type:
            logger.warning("Unknown tool: %s", tool_name)
            return None

        return Action(
            type=action_type,
            agent_id=self.agent_id,
            params=tool_input,
        )

    def _build_memory_context(self) -> str:
        """Build context from past experiences (hybrid retrieval pattern)."""
        best = self.memory.best(3)
        recent = self.memory.recent(2)

        if not best and not recent:
            return ""

        parts = ["PAST EXPERIENCE (for context):"]

        if best:
            parts.append("Best rounds:")
            for exp in best:
                if exp.score:
                    parts.append(f"  Round {exp.round_num}: score={exp.score.total:.0f}")

        if recent:
            parts.append("Recent rounds:")
            for exp in recent:
                if exp.score:
                    parts.append(f"  Round {exp.round_num}: score={exp.score.total:.0f}")

        return "\n".join(parts)
