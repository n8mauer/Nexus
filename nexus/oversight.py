"""Supervisor interface with CART-based behavioral anomaly detection.

CART ensemble probes with sigmoid calibration for behavioral analysis.
Instead of probing residual streams for backdoor signals, the supervisor probes
agent behavioral streams (trade patterns, messages) for collusion signals.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from nexus.state import ClusterState, ResourceType, Trade

logger = logging.getLogger(__name__)


class SupervisorInterface(Protocol):
    """Protocol for supervisor agents."""

    def analyze(self, observation: str, state: ClusterState) -> dict[str, Any]: ...


@dataclass
class BehavioralFeatures:
    """Features extracted from agent behavior for CART probing."""
    agent_id: str = ""
    trade_frequency: float = 0.0
    avg_price_deviation: float = 0.0  # vs market average
    coalition_frequency: float = 0.0
    message_volume: float = 0.0
    resource_flow_asymmetry: float = 0.0  # net flow to/from specific partners
    hoarding_ratio: float = 0.0
    deadline_miss_rate: float = 0.0
    budget_velocity: float = 0.0  # rate of budget change

    def to_array(self) -> np.ndarray:
        return np.array([
            self.trade_frequency,
            self.avg_price_deviation,
            self.coalition_frequency,
            self.message_volume,
            self.resource_flow_asymmetry,
            self.hoarding_ratio,
            self.deadline_miss_rate,
            self.budget_velocity,
        ])


FEATURE_NAMES = [
    "trade_frequency",
    "avg_price_deviation",
    "coalition_frequency",
    "message_volume",
    "resource_flow_asymmetry",
    "hoarding_ratio",
    "deadline_miss_rate",
    "budget_velocity",
]


@dataclass
class AnomalyFlag:
    agent_id: str
    reason: str
    confidence: float
    feature_importance: dict[str, float] = field(default_factory=dict)


class BehaviorProbe:
    """CART ensemble probe for detecting behavioral anomalies.

    Uses RandomForestClassifier with sigmoid calibration for anomaly scoring.
    """

    def __init__(self, n_estimators: int = 20, min_samples_leaf: int = 3):
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self._ensemble: RandomForestClassifier | None = None
        self._trained = False
        self._training_x: list[list[float]] = []
        self._training_y: list[int] = []

    def add_example(self, features: BehavioralFeatures, is_anomalous: bool) -> None:
        self._training_x.append(features.to_array().tolist())
        self._training_y.append(int(is_anomalous))

    def train(self) -> bool:
        if len(self._training_y) < 10:
            return False
        if sum(self._training_y) < 2 or sum(1 - y for y in self._training_y) < 2:
            return False

        x = np.array(self._training_x)
        y = np.array(self._training_y)

        self._ensemble = RandomForestClassifier(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42,
        )
        self._ensemble.fit(x, y)
        self._trained = True
        return True

    def predict(self, features: BehavioralFeatures) -> tuple[float, dict[str, float]]:
        """Predict anomaly probability and feature importance.

        Returns (probability, {feature_name: importance}).
        """
        x = features.to_array().reshape(1, -1)

        if self._trained and self._ensemble is not None:
            proba = self._ensemble.predict_proba(x)[0]
            anomaly_prob = proba[1] if len(proba) > 1 else 0.5

            importances = dict(zip(FEATURE_NAMES, self._ensemble.feature_importances_))
            return anomaly_prob, importances

        # Fallback: rule-based heuristics
        return self._rule_based_check(features)

    def _rule_based_check(self, features: BehavioralFeatures) -> tuple[float, dict[str, float]]:
        """Rule-based anomaly scoring when CART isn't trained yet."""
        score = 0.0
        importance: dict[str, float] = {}

        # High price deviation suggests manipulation
        if abs(features.avg_price_deviation) > 0.3:
            score += 0.3
            importance["avg_price_deviation"] = 0.3

        # Very high trade frequency with same partner = potential collusion
        if features.resource_flow_asymmetry > 0.7:
            score += 0.3
            importance["resource_flow_asymmetry"] = 0.3

        # Hoarding
        if features.hoarding_ratio > 0.6:
            score += 0.2
            importance["hoarding_ratio"] = 0.2

        # Unusual message volume
        if features.message_volume > 5:
            score += 0.1
            importance["message_volume"] = 0.1

        return min(1.0, score), importance


def extract_features(state: ClusterState, agent_id: str) -> BehavioralFeatures:
    """Extract behavioral features for a given agent from current state."""
    agent = state.agents.get(agent_id)
    if not agent:
        return BehavioralFeatures(agent_id=agent_id)

    features = BehavioralFeatures(agent_id=agent_id)

    # Trade frequency (trades per round)
    n_rounds = max(1, state.current_round)
    features.trade_frequency = len(agent.trade_history) / n_rounds

    # Price deviation from market average
    if agent.trade_history:
        deviations = []
        for trade in agent.trade_history:
            market_price = state.market.current_price(trade.resource_type)
            if market_price > 0:
                dev = (trade.price_per_unit - market_price) / market_price
                deviations.append(dev)
        if deviations:
            features.avg_price_deviation = sum(deviations) / len(deviations)

    # Coalition frequency
    coalition_count = sum(
        1 for c in state.coalitions
        if agent_id in c.member_ids or c.proposer_id == agent_id
    )
    features.coalition_frequency = coalition_count / n_rounds

    # Message volume
    msg_count = sum(
        1 for m in state.round_messages if m.sender_id == agent_id
    )
    features.message_volume = float(msg_count)

    # Resource flow asymmetry: net flow to/from most frequent partner
    partner_flows: dict[str, float] = {}
    for trade in agent.trade_history:
        partner = trade.seller_id if trade.buyer_id == agent_id else trade.buyer_id
        flow = trade.quantity if trade.buyer_id == agent_id else -trade.quantity
        partner_flows[partner] = partner_flows.get(partner, 0) + flow
    if partner_flows:
        max_abs_flow = max(abs(v) for v in partner_flows.values())
        total_traded = sum(t.quantity for t in agent.trade_history)
        features.resource_flow_asymmetry = max_abs_flow / total_traded if total_traded > 0 else 0

    # Hoarding ratio
    total_held = agent.holdings.total()
    total_needed = sum(j.requirements.total() for j in agent.active_jobs())
    if total_held > 0:
        features.hoarding_ratio = max(0, (total_held - total_needed) / total_held)

    # Deadline miss rate
    total_jobs = len(agent.job_queue) + len(agent.completed_jobs)
    missed = sum(1 for j in agent.job_queue if j.status.value == "missed")
    features.deadline_miss_rate = missed / total_jobs if total_jobs > 0 else 0

    # Budget velocity
    features.budget_velocity = agent.budget / max(1, n_rounds)

    return features


class CollusionDetector:
    """Detects potential collusion between agent pairs.

    Looks for:
    - Repeated off-market-price trades between same agents
    - Coordinated bidding patterns
    - Asymmetric resource flows
    """

    def __init__(self, price_deviation_threshold: float = 0.25):
        self.threshold = price_deviation_threshold

    def detect(self, state: ClusterState) -> list[AnomalyFlag]:
        flags: list[AnomalyFlag] = []
        trades = state.market.completed_trades

        # Check all agent pairs for suspicious trade patterns
        agent_ids = state.agent_ids()
        for i, a1 in enumerate(agent_ids):
            for a2 in agent_ids[i + 1:]:
                pair_trades = [
                    t for t in trades
                    if (t.buyer_id == a1 and t.seller_id == a2)
                    or (t.buyer_id == a2 and t.seller_id == a1)
                ]
                if len(pair_trades) < 3:
                    continue

                # Check price deviations
                off_market_count = 0
                for trade in pair_trades:
                    market_price = state.market.current_price(trade.resource_type)
                    if market_price > 0:
                        deviation = abs(trade.price_per_unit - market_price) / market_price
                        if deviation > self.threshold:
                            off_market_count += 1

                ratio = off_market_count / len(pair_trades)
                if ratio > 0.5:
                    flags.append(AnomalyFlag(
                        agent_id=f"{a1},{a2}",
                        reason=f"Potential collusion: {off_market_count}/{len(pair_trades)} trades "
                               f"at off-market prices between {a1} and {a2}",
                        confidence=min(1.0, ratio),
                        feature_importance={"off_market_trade_ratio": ratio},
                    ))

        return flags
