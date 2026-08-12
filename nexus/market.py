"""Resource market with bid/offer matching, price impact, and commission.

5-step trade pipeline: orders -> prices -> trades -> holdings -> value.
Commission + price impact model creates realistic market dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass

from nexus.state import (
    AgentState,
    Bid,
    ClusterState,
    MarketState,
    Offer,
    ResourceType,
    Trade,
)


@dataclass
class MarketConfig:
    commission_rate: float = 0.02
    price_impact: float = 0.005
    max_price_deviation: float = 3.0  # max multiplier from base price


class ResourceMarket:
    """Manages the bid/offer matching engine with realistic market dynamics."""

    def __init__(self, config: MarketConfig | None = None):
        self.config = config or MarketConfig()

    def place_bid(self, state: ClusterState, agent_id: str, resource_type: ResourceType,
                  quantity: float, price_per_unit: float) -> Bid:
        bid = Bid(
            agent_id=agent_id,
            resource_type=resource_type,
            quantity=quantity,
            price_per_unit=price_per_unit,
            round_created=state.current_round,
        )
        state.market.active_bids.append(bid)
        return bid

    def place_offer(self, state: ClusterState, agent_id: str, resource_type: ResourceType,
                    quantity: float, price_per_unit: float) -> Offer:
        offer = Offer(
            agent_id=agent_id,
            resource_type=resource_type,
            quantity=quantity,
            price_per_unit=price_per_unit,
            round_created=state.current_round,
        )
        state.market.active_offers.append(offer)
        return offer

    def accept_bid(self, state: ClusterState, seller_id: str, bid_id: str) -> Trade | None:
        """Seller accepts a buyer's bid — seller provides resources, buyer pays."""
        bid = next((b for b in state.market.active_bids if b.id == bid_id and not b.accepted), None)
        if not bid:
            return None

        buyer = state.agents.get(bid.agent_id)
        seller = state.agents.get(seller_id)
        if not buyer or not seller:
            return None

        return self._execute_trade(
            state=state,
            buyer=buyer,
            seller=seller,
            resource_type=bid.resource_type,
            quantity=bid.quantity,
            price_per_unit=bid.price_per_unit,
            bid=bid,
        )

    def accept_offer(self, state: ClusterState, buyer_id: str, offer_id: str) -> Trade | None:
        """Buyer accepts a seller's offer — buyer pays, seller provides resources."""
        offer = next((o for o in state.market.active_offers if o.id == offer_id and not o.accepted), None)
        if not offer:
            return None

        buyer = state.agents.get(buyer_id)
        seller = state.agents.get(offer.agent_id)
        if not buyer or not seller:
            return None

        return self._execute_trade(
            state=state,
            buyer=buyer,
            seller=seller,
            resource_type=offer.resource_type,
            quantity=offer.quantity,
            price_per_unit=offer.price_per_unit,
            offer=offer,
        )

    def _execute_trade(
        self,
        state: ClusterState,
        buyer: AgentState,
        seller: AgentState,
        resource_type: ResourceType,
        quantity: float,
        price_per_unit: float,
        bid: Bid | None = None,
        offer: Offer | None = None,
    ) -> Trade | None:
        """Execute a trade with commission and price impact.

        5-step pipeline:
        1. Validate orders
        2. Compute adjusted prices (impact)
        3. Execute trade (transfer resources)
        4. Update holdings
        5. Record and update price history
        """
        # Step 1: Validate
        seller_held = seller.holdings.get(resource_type)
        if seller_held < quantity:
            return None

        # Step 2: Price impact — large trades move the price
        impact = self.config.price_impact * quantity
        adjusted_price = price_per_unit * (1 + impact)
        commission = adjusted_price * quantity * self.config.commission_rate
        total_cost = adjusted_price * quantity + commission

        if buyer.budget < total_cost:
            return None

        # Step 3: Execute
        seller.holdings.subtract(resource_type, quantity)
        buyer.holdings.add(resource_type, quantity)

        # Step 4: Update budgets
        buyer.budget -= total_cost
        seller.budget += adjusted_price * quantity  # seller gets full adjusted price

        # Step 5: Record trade
        trade = Trade(
            buyer_id=buyer.id,
            seller_id=seller.id,
            resource_type=resource_type,
            quantity=quantity,
            price_per_unit=adjusted_price,
            total_cost=total_cost,
            commission=commission,
            price_impact=impact,
            round_executed=state.current_round,
        )

        state.market.completed_trades.append(trade)
        buyer.trade_history.append(trade)
        seller.trade_history.append(trade)

        # Update price history
        if resource_type not in state.market.price_history:
            state.market.price_history[resource_type] = []
        state.market.price_history[resource_type].append(adjusted_price)

        # Mark bid/offer as accepted
        if bid:
            bid.accepted = True
            bid.accepted_by = seller.id
        if offer:
            offer.accepted = True
            offer.accepted_by = buyer.id

        # Reputation boost for successful trade
        buyer.reputation = min(100, buyer.reputation + 0.5)
        seller.reputation = min(100, seller.reputation + 0.5)

        return trade

    def clear_expired(self, state: ClusterState, max_age: int = 3) -> int:
        """Remove bids/offers older than max_age rounds."""
        cutoff = state.current_round - max_age
        before = len(state.market.active_bids) + len(state.market.active_offers)
        state.market.active_bids = [
            b for b in state.market.active_bids
            if not b.accepted and b.round_created >= cutoff
        ]
        state.market.active_offers = [
            o for o in state.market.active_offers
            if not o.accepted and o.round_created >= cutoff
        ]
        after = len(state.market.active_bids) + len(state.market.active_offers)
        return before - after

    def get_market_stats(self, state: ClusterState) -> dict:
        """Compute market statistics for portfolio-style evaluation."""
        trades = state.market.completed_trades
        if not trades:
            return {"total_volume": 0, "total_commission": 0, "avg_price": {}}

        total_volume = sum(t.total_cost for t in trades)
        total_commission = sum(t.commission for t in trades)

        # Average price per resource type
        avg_prices: dict[ResourceType, float] = {}
        for rt in ResourceType:
            rt_trades = [t for t in trades if t.resource_type == rt]
            if rt_trades:
                avg_prices[rt] = sum(t.price_per_unit for t in rt_trades) / len(rt_trades)

        # Sharpe-like ratio: mean return / std of returns per round
        round_volumes: dict[int, float] = {}
        for t in trades:
            round_volumes.setdefault(t.round_executed, 0)
            round_volumes[t.round_executed] += t.total_cost

        volumes = list(round_volumes.values())
        if len(volumes) > 1:
            mean_vol = sum(volumes) / len(volumes)
            std_vol = (sum((v - mean_vol) ** 2 for v in volumes) / len(volumes)) ** 0.5
            market_efficiency = mean_vol / std_vol if std_vol > 0 else float("inf")
        else:
            market_efficiency = 0

        return {
            "total_volume": total_volume,
            "total_commission": total_commission,
            "avg_price": {rt.value: p for rt, p in avg_prices.items()},
            "market_efficiency": market_efficiency,
            "num_trades": len(trades),
        }
