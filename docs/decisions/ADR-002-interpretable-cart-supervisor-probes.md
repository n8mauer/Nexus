# ADR-002: Privileged supervisor with interpretable CART behavioral probes

Date: 2026-08-12
Status: Accepted

## Context

The oversight track requires a supervisor that detects anomalous agent behavior (collusion, hoarding, market manipulation) across all agents. An opaque anomaly detector (for example a neural scorer) would produce flags without explanations, which defeats the purpose of studying scalable oversight: researchers need to see why an agent was flagged, and flagged agents need structured feedback they can respond to.

## Decision

Oversight is built on decision-tree ensembles over hand-defined behavioral features, plus a deterministic pairwise collusion detector.

- `nexus/oversight.py` defines `BehavioralFeatures`, eight named features per agent (trade frequency, average price deviation, coalition frequency, message volume, resource flow asymmetry, hoarding ratio, deadline miss rate, budget velocity), extracted from full simulation state by `extract_features()`.
- `BehaviorProbe` wraps a scikit-learn `RandomForestClassifier` (CART ensemble). `predict()` returns both an anomaly probability and a per-feature importance dictionary. Until the probe has at least 10 examples covering both classes, a transparent rule-based fallback (`_rule_based_check`) scores the same features with fixed thresholds.
- `CollusionDetector` flags agent pairs with three or more mutual trades where more than half deviate from the market price by over 25 percent.
- `agents/supervisor_agent.py` runs the probe over every agent each round, converts feature importances into natural-language explanations (`_explain_anomaly`), and emits a per-round report with flags, Gini coefficients, and total welfare.

## Consequences

- Every flag carries its evidence: the top contributing features and their importances, which makes supervisor behavior auditable and testable (`tests/test_oversight.py`).
- The feature set caps what the probe can see; behaviors outside those eight features are invisible to it, and adding a feature means touching `BehavioralFeatures`, `FEATURE_NAMES`, and `extract_features()` together.
- The probe currently trains on its own thresholded predictions (self-supervised labeling in `SupervisorAgent.analyze`), which can reinforce its initial rule-based biases. Ground-truth labels require the scripted anomaly-injection scenarios that are still Planned in `ROADMAP.md`.
- Choosing shallow interpretable models gives up detection power on subtle, distributed strategies that a larger learned model might catch.
