"""Nexus OpenEnv Client.

Connects to a running Nexus OpenEnv server (local or HF Spaces)
and provides a typed Python API for interacting with the environment.
"""

from typing import Dict

from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from openenv.core import EnvClient

from .models import NexusAction, NexusObservation


class NexusEnv(EnvClient[NexusAction, NexusObservation]):
    """Client for the Nexus negotiation environment.

    Example:
        >>> with NexusEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.observation_text[:100])
        ...     result = client.step(NexusAction(raw_text="pass"))
        ...     print(f"Round reward: {result.reward}")
    """

    def _step_payload(self, action: NexusAction) -> Dict:
        return {"raw_text": action.raw_text}

    def _parse_result(self, payload: Dict) -> StepResult[NexusObservation]:
        obs_data = payload.get("observation", {})
        observation = NexusObservation(
            observation_text=obs_data.get("observation_text", ""),
            round_number=obs_data.get("round_number", 0),
            total_rounds=obs_data.get("total_rounds", 10),
            score=obs_data.get("score", 0.0),
            round_reward=obs_data.get("round_reward", 0.0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
