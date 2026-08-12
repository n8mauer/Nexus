# ADR-001: Natural-language observations with a structured, parseable action space

Date: 2026-08-12
Status: Accepted

## Context

The environment exists to train and evaluate LLM agents. Classical multi-agent RL environments expose tensor observations and integer action indices, which forces an encoding layer between the environment and a language model and discards the semantic content (job descriptions, negotiation messages) that LLMs are good at using. At the same time, fully free-form text actions are ambiguous and hard to score, validate, or replay.

## Decision

Observations are rendered as structured natural language and actions are typed, validated data objects.

- `nexus/observations.py` renders each agent's view as sectioned text (cluster status, own state, market, messages, coalition proposals, reputation board) via `render()`, masking other agents' private job queues. `render_supervisor()` produces the privileged full-visibility view.
- `nexus/actions.py` defines a closed `ActionType` enum (allocate, bid, offer, accept_bid, accept_offer, coalition actions, send_message, pass), an `Action` dataclass with typed parameter accessors, and `validate_action()`, which checks every action against budget, holdings, and market state before execution.
- `parse_llm_output()` bridges the two: it accepts JSON, JSON inside markdown fences, or keyword-style natural language, and falls back to a safe `pass` action when nothing parses.

## Consequences

- LLM agents consume observations directly with no feature engineering, and any model that can emit a small JSON object can act in the environment.
- Every action is validated with a human-readable rejection reason, so invalid moves cannot corrupt state and agents receive usable feedback.
- Scripted agents must parse the observation text; `agents/greedy_agent.py` and `agents/strategic_agent.py` do this with regular expressions, which couples them to the exact rendering format in `nexus/observations.py`. Format changes require updating those parsers and `tests/test_observations.py`.
- There is no vector observation space, so classical RL libraries cannot consume the environment directly; a Gymnasium wrapper is a Planned item in `ROADMAP.md`.
- The keyword fallback in `parse_llm_output()` is intentionally forgiving and can misread free text; malformed output degrades to `pass` rather than raising, which trades silent no-ops for robustness.
