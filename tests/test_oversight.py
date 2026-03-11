"""Tests for oversight and CART behavioral probing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nexus.oversight import (
    BehavioralFeatures,
    BehaviorProbe,
    CollusionDetector,
    extract_features,
)
from nexus.state import (
    AgentState,
    ClusterState,
    Resources,
    ResourceType,
    Trade,
)


def test_behavioral_features_to_array():
    features = BehavioralFeatures(
        agent_id="test",
        trade_frequency=1.5,
        avg_price_deviation=0.2,
        coalition_frequency=0.3,
        message_volume=4.0,
        resource_flow_asymmetry=0.5,
        hoarding_ratio=0.1,
        deadline_miss_rate=0.0,
        budget_velocity=100.0,
    )
    arr = features.to_array()
    assert len(arr) == 8
    assert arr[0] == 1.5  # trade_frequency


def test_behavior_probe_rule_based():
    """Test fallback rule-based detection when CART isn't trained."""
    probe = BehaviorProbe()

    # Normal behavior
    normal = BehavioralFeatures(
        trade_frequency=0.5,
        avg_price_deviation=0.1,
        resource_flow_asymmetry=0.2,
        hoarding_ratio=0.1,
    )
    prob, imp = probe.predict(normal)
    assert prob < 0.5

    # Suspicious behavior
    suspicious = BehavioralFeatures(
        trade_frequency=3.0,
        avg_price_deviation=0.5,
        resource_flow_asymmetry=0.8,
        hoarding_ratio=0.7,
        message_volume=10.0,
    )
    prob, imp = probe.predict(suspicious)
    assert prob > 0.3
    assert len(imp) > 0


def test_extract_features():
    state = ClusterState()
    state.agents["test"] = AgentState(
        id="test",
        team_name="Test",
        budget=500.0,
        holdings=Resources(gpu=20, cpu=40, memory=100, bandwidth=10),
    )
    state.current_round = 5

    features = extract_features(state, "test")

    assert features.agent_id == "test"
    assert features.trade_frequency == 0.0  # No trades yet
    assert features.hoarding_ratio > 0  # Has resources but no jobs


def test_collusion_detector():
    state = ClusterState()
    state.agents["a0"] = AgentState(id="a0", team_name="Alpha")
    state.agents["a1"] = AgentState(id="a1", team_name="Beta")

    # Add suspicious trades at off-market prices
    for i in range(5):
        trade = Trade(
            buyer_id="a0",
            seller_id="a1",
            resource_type=ResourceType.GPU,
            quantity=10,
            price_per_unit=200.0,  # Way above market
            total_cost=2000.0,
            round_executed=i + 1,
        )
        state.market.completed_trades.append(trade)
        state.agents["a0"].trade_history.append(trade)
        state.agents["a1"].trade_history.append(trade)

    # Set low market price
    state.market.price_history[ResourceType.GPU] = [100.0]

    detector = CollusionDetector(price_deviation_threshold=0.25)
    flags = detector.detect(state)

    assert len(flags) > 0
    assert "collusion" in flags[0].reason.lower()
