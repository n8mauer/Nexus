"""Data models for the Nexus OpenEnv environment.

Defines Action and Observation types for the OpenEnv protocol.
The LLM sends raw text (natural language or JSON) as its action,
and receives a natural language observation of the game state.
"""

from pydantic import Field

from openenv.core.env_server.types import Action, Observation


class NexusAction(Action):
    """Action for the Nexus negotiation environment.

    The LLM sends raw text which is parsed into structured game actions.
    Supported formats:
    - JSON: {"type": "allocate", "job_id": "J-001"}
    - JSON array: [{"type": "bid", ...}, {"type": "allocate", ...}]
    - Natural language: "allocate job J-001", "bid 5 gpu @ $100", "pass"
    """

    raw_text: str = Field(
        ..., description="Natural language or JSON action string"
    )


class NexusObservation(Observation):
    """Observation from the Nexus environment after one round.

    Contains a natural language rendering of the game state from the
    perspective of agent_0 (the LLM-controlled agent), plus numeric
    metadata for training signal.
    """

    observation_text: str = Field(
        default="", description="Natural language observation of game state"
    )
    round_number: int = Field(default=0, description="Current round number")
    total_rounds: int = Field(
        default=10, description="Total rounds in this episode"
    )
    score: float = Field(
        default=0.0, description="Cumulative score so far"
    )
    round_reward: float = Field(
        default=0.0, description="Reward earned this round"
    )
