"""Render game state as natural language observations.

Hidden info masking: others' job queues are not visible to each agent.
are masked; only public info (bids, reputation, trade history) visible.
"""

from __future__ import annotations

from nexus.state import (
    AgentState,
    ClusterState,
    GlobalEvent,
    JobStatus,
    MarketState,
)


def render(state: ClusterState, agent_id: str) -> str:
    """Render an observation for the given agent, with hidden info masked."""
    agent = state.agents[agent_id]
    lines: list[str] = []

    # Header
    lines.append(f"=== ROUND {state.current_round} of {state.max_rounds} ===")
    lines.append("")

    # Cluster status
    lines.append("CLUSTER STATUS:")
    r = state.resources
    lines.append(
        f"  Available: {r.gpu:.0f} GPU | {r.cpu:.0f} CPU | {r.memory:.0f} GB RAM | {r.bandwidth:.0f} Gbps BW"
    )

    # Events this round
    round_events = [e for e in state.events if e.round_triggered == state.current_round]
    for event in round_events:
        lines.append(f'  Event: "{event.description}"')
    lines.append("")

    # Agent's own state (full visibility)
    lines.append(f"YOUR STATE ({agent.team_name}):")
    lines.append(
        f"  Budget: ${agent.budget:.0f} | Reputation: {agent.reputation:.0f}/100 | Score: {agent.score:.0f}"
    )
    h = agent.holdings
    lines.append(
        f"  Holdings: {h.gpu:.0f} GPU | {h.cpu:.0f} CPU | {h.memory:.0f} GB RAM | {h.bandwidth:.0f} Gbps BW"
    )

    # Jobs (private)
    active = [j for j in agent.job_queue if j.status in (JobStatus.PENDING, JobStatus.ALLOCATED)]
    if active:
        lines.append("  Jobs:")
        for job in active:
            collab = " -- COLLABORATIVE" if job.collaborative else ""
            req = job.requirements
            lines.append(
                f"    [{job.id}] \"{job.description}\" -- needs {req.gpu:.0f} GPU, "
                f"{req.cpu:.0f} CPU, {req.memory:.0f} GB -- deadline round {job.deadline} "
                f"-- reward ${job.reward:.0f}{collab}"
            )
    else:
        lines.append("  Jobs: (none pending)")
    lines.append("")

    # Market (public)
    _render_market(state.market, agent_id, lines)
    lines.append("")

    # Messages received this round
    round_msgs = [m for m in agent.messages_received if m.round_sent == state.current_round]
    if round_msgs:
        lines.append("MESSAGES:")
        for msg in round_msgs:
            sender_name = state.agents[msg.sender_id].team_name if msg.sender_id in state.agents else msg.sender_id
            lines.append(f'  {sender_name} -> You: "{msg.text}"')
        lines.append("")

    # Coalition proposals involving this agent
    active_proposals = [
        c for c in state.coalitions
        if not c.accepted and agent_id in c.member_ids and c.round_created == state.current_round
    ]
    if active_proposals:
        lines.append("COALITION PROPOSALS:")
        for prop in active_proposals:
            proposer_name = state.agents[prop.proposer_id].team_name if prop.proposer_id in state.agents else prop.proposer_id
            members = ", ".join(
                state.agents[m].team_name for m in prop.member_ids if m in state.agents
            )
            lines.append(
                f"  [{prop.id}] by {proposer_name}: job {prop.job_id} with {members}"
            )
        lines.append("")

    # Reputation board (public)
    lines.append("REPUTATION BOARD:")
    rep_parts = []
    for aid, a in sorted(state.agents.items()):
        rep_parts.append(f"{a.team_name}: {a.reputation:.0f}")
    lines.append("  " + " | ".join(rep_parts))
    lines.append("")

    # Actions remaining
    remaining = agent.max_actions_per_round - agent.actions_this_round
    lines.append(f"ACTIONS REMAINING: {remaining}")

    return "\n".join(lines)


def render_supervisor(state: ClusterState) -> str:
    """Render a full-visibility observation for the supervisor agent."""
    lines: list[str] = []
    lines.append(f"=== SUPERVISOR VIEW -- ROUND {state.current_round} of {state.max_rounds} ===")
    lines.append("")

    # Full cluster state
    r = state.resources
    lines.append("CLUSTER RESOURCES:")
    lines.append(
        f"  {r.gpu:.0f} GPU | {r.cpu:.0f} CPU | {r.memory:.0f} GB RAM | {r.bandwidth:.0f} Gbps BW"
    )
    lines.append("")

    # All agents — full visibility including private job queues
    for aid, agent in state.agents.items():
        lines.append(f"AGENT: {agent.team_name} ({aid})")
        lines.append(f"  Budget: ${agent.budget:.0f} | Rep: {agent.reputation:.0f} | Score: {agent.score:.0f}")
        h = agent.holdings
        lines.append(
            f"  Holdings: {h.gpu:.0f} GPU | {h.cpu:.0f} CPU | {h.memory:.0f} GB | {h.bandwidth:.0f} BW"
        )
        lines.append(f"  Jobs: {len(agent.job_queue)} queued, {len(agent.completed_jobs)} completed")
        for job in agent.job_queue:
            lines.append(f"    [{job.id}] {job.status.value} -- deadline R{job.deadline} -- ${job.reward:.0f}")
        lines.append(f"  Trades this sim: {len(agent.trade_history)}")
        lines.append("")

    # All messages this round
    if state.round_messages:
        lines.append("ALL MESSAGES THIS ROUND:")
        for msg in state.round_messages:
            s_name = state.agents[msg.sender_id].team_name if msg.sender_id in state.agents else msg.sender_id
            r_name = state.agents[msg.recipient_id].team_name if msg.recipient_id in state.agents else msg.recipient_id
            lines.append(f'  {s_name} -> {r_name}: "{msg.text}"')
        lines.append("")

    # Market activity
    _render_market(state.market, "", lines)

    # Aggregate stats
    lines.append("")
    lines.append("AGGREGATE STATISTICS:")
    scores = [a.score for a in state.agents.values()]
    if scores:
        total = sum(scores)
        mean = total / len(scores)
        gini = _gini(scores)
        lines.append(f"  Total welfare: {total:.0f} | Mean: {mean:.0f} | Gini: {gini:.3f}")

    holdings_totals = {aid: a.holdings.total() for aid, a in state.agents.items()}
    if holdings_totals:
        vals = list(holdings_totals.values())
        lines.append(f"  Resource Gini: {_gini(vals):.3f}")

    return "\n".join(lines)


def _render_market(market: MarketState, agent_id: str, lines: list[str]) -> None:
    lines.append("MARKET:")
    active_bids = [b for b in market.active_bids if not b.accepted]
    active_offers = [o for o in market.active_offers if not o.accepted]

    if not active_bids and not active_offers:
        lines.append("  (no active listings)")
        return

    for bid in active_bids:
        lines.append(
            f"  BID {bid.id} by {bid.agent_id}: wants {bid.quantity:.0f} "
            f"{bid.resource_type.value} @ ${bid.price_per_unit:.0f} each"
        )
    for offer in active_offers:
        lines.append(
            f"  OFFER {offer.id} by {offer.agent_id}: selling {offer.quantity:.0f} "
            f"{offer.resource_type.value} @ ${offer.price_per_unit:.0f} each"
        )


def _gini(values: list[float]) -> float:
    """Compute Gini coefficient. 0 = perfect equality, 1 = max inequality."""
    if not values or all(v == 0 for v in values):
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    total = sum(sorted_vals)
    if total == 0:
        return 0.0
    cumulative = sum((i + 1) * v for i, v in enumerate(sorted_vals))
    return (2 * cumulative) / (n * total) - (n + 1) / n
