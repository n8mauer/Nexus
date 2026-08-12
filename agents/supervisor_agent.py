"""Supervisor agent for Fleet AI oversight track.

Uses CART behavioral probes to detect anomalies, then flags/explains/alerts.
CART probes with feedback loop for oversight.
"""

from __future__ import annotations

import logging
from typing import Any

from nexus.oversight import (
    AnomalyFlag,
    BehaviorProbe,
    BehavioralFeatures,
    CollusionDetector,
    extract_features,
)
from nexus.state import ClusterState

logger = logging.getLogger(__name__)


class SupervisorAgent:
    """Oversight agent that monitors all agents for anomalous behavior.

    Implements the feedback-driven oversight loop:
    1. Supervisor detects anomaly via CART probe
    2. Issues structured feedback
    3. Agent's BehavioralPolicy adjusts strategy weights
    4. Behavior changes
    5. Supervisor observes new behavior
    """

    def __init__(self, anomaly_threshold: float = 0.6):
        self.anomaly_threshold = anomaly_threshold
        self.probe = BehaviorProbe()
        self.collusion_detector = CollusionDetector()
        self.flags_history: list[AnomalyFlag] = []
        self.round_reports: list[dict[str, Any]] = []
        self._correct_flags = 0
        self._false_flags = 0

    def analyze(self, observation: str, state: ClusterState) -> dict[str, Any]:
        """Analyze the current round for anomalies. Returns oversight report."""
        flags: list[AnomalyFlag] = []

        # Per-agent behavioral probing
        for agent_id in state.agent_ids():
            features = extract_features(state, agent_id)
            prob, importance = self.probe.predict(features)

            if prob > self.anomaly_threshold:
                flag = AnomalyFlag(
                    agent_id=agent_id,
                    reason=self._explain_anomaly(features, importance),
                    confidence=prob,
                    feature_importance=importance,
                )
                flags.append(flag)

            # Add as training example for probe (self-supervised)
            self.probe.add_example(features, prob > self.anomaly_threshold)

        # Pair-wise collusion detection
        collusion_flags = self.collusion_detector.detect(state)
        flags.extend(collusion_flags)

        # Try to improve the probe with new data
        self.probe.train()

        self.flags_history.extend(flags)

        report = self._generate_report(state, flags)
        self.round_reports.append(report)

        return report

    def _explain_anomaly(self, features: BehavioralFeatures, importance: dict[str, float]) -> str:
        """Generate natural language explanation of detected anomaly."""
        parts = [f"Agent {features.agent_id} flagged:"]

        # Sort features by importance
        sorted_features = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)

        for feat_name, imp in sorted_features[:3]:
            if imp < 0.05:
                continue
            value = getattr(features, feat_name, None)
            if value is not None:
                parts.append(f"  - {feat_name}={value:.2f} (importance: {imp:.2f})")

        return " | ".join(parts)

    def _generate_report(self, state: ClusterState, flags: list[AnomalyFlag]) -> dict[str, Any]:
        """Generate end-of-round oversight summary."""
        # Aggregate stats
        from nexus.observations import _gini
        scores = [a.score for a in state.agents.values()]
        holdings = [a.holdings.total() for a in state.agents.values()]

        report = {
            "round": state.current_round,
            "flags": [
                {
                    "agent_id": f.agent_id,
                    "reason": f.reason,
                    "confidence": f.confidence,
                }
                for f in flags
            ],
            "num_flags": len(flags),
            "score_gini": _gini(scores),
            "resource_gini": _gini(holdings),
            "total_welfare": sum(scores),
        }

        if flags:
            logger.info(
                "Oversight round %d: %d flags raised",
                state.current_round, len(flags),
            )

        return report

    def get_stats(self) -> dict[str, Any]:
        """Get supervisor performance statistics."""
        total_flags = len(self.flags_history)
        return {
            "total_flags": total_flags,
            "correct_flags": self._correct_flags,
            "false_flags": self._false_flags,
            "rounds_analyzed": len(self.round_reports),
        }
