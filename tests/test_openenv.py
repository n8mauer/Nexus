"""Tests for OpenEnv integration — ProxyAgent and NexusEnvironment."""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nexus.actions import Action, ActionType
from nexus.rewards import RoundScore


class TestProxyAgent:
    """Test the ProxyAgent action-splitting logic."""

    def _make_proxy(self):
        from nexus_openenv.server.nexus_environment import ProxyAgent
        return ProxyAgent("agent_0")

    def test_empty_actions_returns_pass(self):
        proxy = self._make_proxy()
        proxy.set_actions([])
        # First call (negotiate) — should PASS
        result1 = proxy.act("obs")
        assert len(result1) == 1
        assert result1[0].type == ActionType.PASS
        # Second call (final) — should PASS
        result2 = proxy.act("obs")
        assert len(result2) == 1
        assert result2[0].type == ActionType.PASS

    def test_negotiate_actions_first_call(self):
        proxy = self._make_proxy()
        actions = [
            Action(type=ActionType.SEND_MESSAGE, agent_id="agent_0",
                   params={"target_id": "agent_1", "text": "hello"}),
            Action(type=ActionType.BID, agent_id="agent_0",
                   params={"resource_type": "gpu", "quantity": 5, "price": 100}),
            Action(type=ActionType.ALLOCATE, agent_id="agent_0",
                   params={"job_id": "J-001"}),
        ]
        proxy.set_actions(actions)

        # First call returns negotiate actions
        result1 = proxy.act("obs")
        types1 = {a.type for a in result1}
        assert ActionType.SEND_MESSAGE in types1
        assert ActionType.BID in types1
        assert ActionType.ALLOCATE not in types1

    def test_final_actions_second_call(self):
        proxy = self._make_proxy()
        actions = [
            Action(type=ActionType.BID, agent_id="agent_0",
                   params={"resource_type": "gpu", "quantity": 5, "price": 100}),
            Action(type=ActionType.ALLOCATE, agent_id="agent_0",
                   params={"job_id": "J-001"}),
        ]
        proxy.set_actions(actions)

        proxy.act("obs")  # negotiate
        result2 = proxy.act("obs")  # final
        types2 = {a.type for a in result2}
        assert ActionType.ALLOCATE in types2
        assert ActionType.BID not in types2

    def test_call_counter_resets(self):
        proxy = self._make_proxy()
        proxy.set_actions([Action(type=ActionType.PASS, agent_id="agent_0")])
        proxy.act("obs")  # call 1
        proxy.act("obs")  # call 2, resets counter

        # New set of actions
        proxy.set_actions([
            Action(type=ActionType.BID, agent_id="agent_0",
                   params={"resource_type": "gpu", "quantity": 5, "price": 100}),
        ])
        result = proxy.act("obs")  # should be call 1 again (negotiate)
        assert any(a.type == ActionType.BID for a in result)

    def test_on_round_end_no_error(self):
        proxy = self._make_proxy()
        proxy.on_round_end(RoundScore())  # should not raise


class TestNexusEnvironmentDirect:
    """Test NexusEnvironment without HTTP (direct instantiation).

    These tests require openenv to be installed. They test the
    environment logic without the HTTP server layer.
    """

    def _make_env(self):
        try:
            from nexus_openenv.server.nexus_environment import NexusEnvironment
            return NexusEnvironment()
        except ImportError:
            import pytest
            pytest.skip("openenv not installed")

    def _make_action(self):
        from nexus_openenv.models import NexusAction
        return NexusAction

    def test_reset(self):
        env = self._make_env()
        obs = env.reset()
        assert obs.round_number == 0
        assert obs.total_rounds == 10
        assert obs.score == 0.0
        assert obs.done is False
        assert "ROUND" in obs.observation_text
        assert "YOUR STATE" in obs.observation_text

    def test_step_pass(self):
        env = self._make_env()
        NexusAction = self._make_action()
        env.reset()
        obs = env.step(NexusAction(raw_text="pass"))
        assert obs.round_number == 1
        assert obs.done is False
        assert isinstance(obs.reward, float)

    def test_full_episode(self):
        env = self._make_env()
        NexusAction = self._make_action()
        env.reset()
        for i in range(10):
            obs = env.step(NexusAction(raw_text="pass"))
            assert obs.round_number == i + 1
        assert obs.done is True

    def test_step_with_json_action(self):
        env = self._make_env()
        NexusAction = self._make_action()
        env.reset()
        obs = env.step(NexusAction(raw_text='{"type": "pass"}'))
        assert obs.round_number == 1

    def test_step_with_allocate(self):
        env = self._make_env()
        NexusAction = self._make_action()
        env.reset()
        # Even if the job doesn't exist, it should not crash
        obs = env.step(NexusAction(raw_text='{"type": "allocate", "job_id": "J-nonexistent"}'))
        assert obs.round_number == 1

    def test_state_property(self):
        env = self._make_env()
        env.reset()
        state = env.state
        assert state.step_count == 0
        assert state.episode_id is not None

    def test_multiple_resets(self):
        env = self._make_env()
        env.reset()
        ep1 = env.state.episode_id
        env.reset()
        ep2 = env.state.episode_id
        assert ep1 != ep2  # New episode each reset
