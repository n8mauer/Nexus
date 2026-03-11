"""Tests for the resource market."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nexus.market import MarketConfig, ResourceMarket
from nexus.state import AgentState, ClusterState, Resources, ResourceType


def _make_state() -> ClusterState:
    state = ClusterState()
    state.agents["buyer"] = AgentState(
        id="buyer", team_name="Buyer", budget=5000.0,
        holdings=Resources(gpu=10, cpu=20, memory=64, bandwidth=5),
    )
    state.agents["seller"] = AgentState(
        id="seller", team_name="Seller", budget=5000.0,
        holdings=Resources(gpu=50, cpu=100, memory=256, bandwidth=20),
    )
    return state


def test_place_bid():
    market = ResourceMarket()
    state = _make_state()

    bid = market.place_bid(state, "buyer", ResourceType.GPU, 10, 100.0)

    assert bid.agent_id == "buyer"
    assert bid.quantity == 10
    assert len(state.market.active_bids) == 1


def test_place_offer():
    market = ResourceMarket()
    state = _make_state()

    offer = market.place_offer(state, "seller", ResourceType.GPU, 20, 90.0)

    assert offer.agent_id == "seller"
    assert offer.quantity == 20
    assert len(state.market.active_offers) == 1


def test_accept_bid():
    """Seller accepts buyer's bid — resources and money transfer."""
    market = ResourceMarket(MarketConfig(commission_rate=0.0, price_impact=0.0))
    state = _make_state()

    bid = market.place_bid(state, "buyer", ResourceType.GPU, 10, 100.0)
    trade = market.accept_bid(state, "seller", bid.id)

    assert trade is not None
    assert trade.buyer_id == "buyer"
    assert trade.seller_id == "seller"
    assert trade.quantity == 10

    # Buyer got GPUs, seller lost them
    assert state.agents["buyer"].holdings.gpu == 20  # 10 + 10
    assert state.agents["seller"].holdings.gpu == 40  # 50 - 10

    # Budget transfer
    assert state.agents["buyer"].budget < 5000
    assert state.agents["seller"].budget > 5000


def test_accept_offer():
    """Buyer accepts seller's offer."""
    market = ResourceMarket(MarketConfig(commission_rate=0.0, price_impact=0.0))
    state = _make_state()

    offer = market.place_offer(state, "seller", ResourceType.CPU, 30, 25.0)
    trade = market.accept_offer(state, "buyer", offer.id)

    assert trade is not None
    assert state.agents["buyer"].holdings.cpu == 50  # 20 + 30
    assert state.agents["seller"].holdings.cpu == 70  # 100 - 30


def test_commission():
    """Trades should incur commission."""
    market = ResourceMarket(MarketConfig(commission_rate=0.05, price_impact=0.0))
    state = _make_state()

    bid = market.place_bid(state, "buyer", ResourceType.GPU, 10, 100.0)
    trade = market.accept_bid(state, "seller", bid.id)

    assert trade is not None
    assert trade.commission > 0
    # Commission is 5% of total
    expected_commission = 10 * 100 * 0.05
    assert abs(trade.commission - expected_commission) < 0.01


def test_price_impact():
    """Larger trades should have price impact."""
    market = ResourceMarket(MarketConfig(commission_rate=0.0, price_impact=0.01))
    state = _make_state()

    bid = market.place_bid(state, "buyer", ResourceType.GPU, 10, 100.0)
    trade = market.accept_bid(state, "seller", bid.id)

    assert trade is not None
    assert trade.price_per_unit > 100.0  # Impact pushes price up
    assert trade.price_impact > 0


def test_insufficient_budget():
    """Trade should fail if buyer can't afford."""
    market = ResourceMarket()
    state = _make_state()
    state.agents["buyer"].budget = 10.0  # Very low budget

    bid = market.place_bid(state, "buyer", ResourceType.GPU, 10, 100.0)
    trade = market.accept_bid(state, "seller", bid.id)

    assert trade is None


def test_insufficient_resources():
    """Trade should fail if seller doesn't have resources."""
    market = ResourceMarket()
    state = _make_state()

    bid = market.place_bid(state, "buyer", ResourceType.GPU, 100, 50.0)
    trade = market.accept_bid(state, "seller", bid.id)

    assert trade is None  # Seller only has 50 GPU


def test_market_stats():
    market = ResourceMarket(MarketConfig(commission_rate=0.02, price_impact=0.0))
    state = _make_state()

    bid = market.place_bid(state, "buyer", ResourceType.GPU, 5, 80.0)
    market.accept_bid(state, "seller", bid.id)

    stats = market.get_market_stats(state)
    assert stats["num_trades"] == 1
    assert stats["total_volume"] > 0
    assert stats["total_commission"] > 0


def test_clear_expired():
    market = ResourceMarket()
    state = _make_state()
    state.current_round = 1

    market.place_bid(state, "buyer", ResourceType.GPU, 5, 80.0)
    assert len(state.market.active_bids) == 1

    state.current_round = 10
    removed = market.clear_expired(state, max_age=3)

    assert removed == 1
    assert len(state.market.active_bids) == 0
