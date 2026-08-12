"""Nexus OpenEnv Environment — wraps SimulationEngine for single-agent OpenEnv API.

Uses the ProxyAgent pattern: the external LLM controls agent_0 via OpenEnv's
step() interface, while NPC GreedyAgents handle the remaining agents internally.
"""

from __future__ import annotations

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from models import NexusAction, NexusObservation
except ImportError:
    from nexus_openenv.models import NexusAction, NexusObservation

from nexus.actions import Action, ActionType, parse_llm_output
from nexus.config import NexusConfig
from nexus.observations import render
from nexus.rewards import RoundScore
from nexus.state import Resources

from agents.greedy_agent import GreedyAgent


# Action types that belong in the negotiate phase
_NEGOTIATE_TYPES = {
    ActionType.SEND_MESSAGE,
    ActionType.BID,
    ActionType.OFFER,
    ActionType.PROPOSE_COALITION,
}


class ProxyAgent:
    """Agent that returns externally-injected actions instead of computing them.

    The SimulationEngine calls act() twice per round:
    1. Negotiate phase — returns message/bid/offer/coalition actions
    2. Final phase — returns allocate/accept/reject actions

    The ProxyAgent splits the injected actions accordingly.
    """

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self._pending_actions: list[Action] = []
        self._call_count = 0

    def set_actions(self, actions: list[Action]) -> None:
        """Inject actions to be returned by the next act() calls."""
        self._pending_actions = actions
        self._call_count = 0

    def act(self, observation: str) -> list[Action]:
        self._call_count += 1
        if self._call_count == 1:
            # Negotiate phase: return message/bid/offer/coalition actions
            negotiate = [
                a for a in self._pending_actions
                if a.type in _NEGOTIATE_TYPES
            ]
            return negotiate or [
                Action(type=ActionType.PASS, agent_id=self.agent_id)
            ]
        else:
            # Final phase: return everything else
            self._call_count = 0
            final = [
                a for a in self._pending_actions
                if a.type not in _NEGOTIATE_TYPES
            ]
            self._pending_actions = []
            return final or [
                Action(type=ActionType.PASS, agent_id=self.agent_id)
            ]

    def on_round_end(self, score: RoundScore) -> None:
        pass


class NexusEnvironment(Environment):
    """OpenEnv Environment wrapping the Nexus SimulationEngine.

    One episode = one full simulation (4 agents, 10 rounds).
    The external LLM controls agent_0. Agents 1-3 are greedy NPCs.
    Each step() call advances one round.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        # Auto-initialize so the env is ready for step() even without reset().
        # OpenEnv HTTP endpoints create a new instance per request via factory,
        # so step() may be called on a freshly-constructed instance.
        self._init_engine()

    def _init_engine(self) -> None:
        """Create a fresh SimulationEngine with ProxyAgent + Greedy NPCs."""
        from nexus.engine import SimulationEngine

        config = NexusConfig(
            num_agents=4,
            max_rounds=10,
            max_actions_per_round=5,
            starting_budget=1000.0,
            cluster_resources=Resources(
                gpu=60, cpu=120, memory=300, bandwidth=30
            ),
            initial_holdings=Resources(gpu=15, cpu=30, memory=80, bandwidth=8),
            round_refresh=Resources(gpu=6, cpu=12, memory=30, bandwidth=3),
            jobs_per_round_min=1,
            jobs_per_round_max=2,
            event_chance_per_round=0.15,
        )

        self._proxy = ProxyAgent("agent_0")
        npcs = [GreedyAgent(f"agent_{i}") for i in range(1, config.num_agents)]
        agents = [self._proxy] + npcs
        self._engine = SimulationEngine(config, agents, seed=42)

    def reset(self) -> NexusObservation:
        self._init_engine()
        self._state = State(episode_id=str(uuid4()), step_count=0)

        obs_text = render(self._engine.state, "agent_0")

        return NexusObservation(
            observation_text=obs_text,
            round_number=0,
            total_rounds=self._engine.config.max_rounds,
            score=0.0,
            round_reward=0.0,
            done=False,
            reward=0.0,
        )

    def step(self, action: NexusAction) -> NexusObservation:
        # Parse LLM text into structured Nexus actions
        actions = parse_llm_output(action.raw_text, "agent_0")
        self._proxy.set_actions(actions)

        # Run one round — engine calls proxy.act() internally
        round_scores = self._engine.step()

        self._state.step_count += 1
        done = self._engine.state.is_finished()

        obs_text = render(self._engine.state, "agent_0")
        agent_score = round_scores.get("agent_0", RoundScore())
        cumulative = self._engine.state.agents["agent_0"].score

        return NexusObservation(
            observation_text=obs_text,
            round_number=self._state.step_count,
            total_rounds=self._engine.config.max_rounds,
            score=cumulative,
            round_reward=agent_score.total,
            done=done,
            reward=agent_score.total,
        )

    @property
    def state(self) -> State:
        return self._state
