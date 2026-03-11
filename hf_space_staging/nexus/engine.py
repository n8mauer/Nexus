"""Core simulation engine: 7-phase round loop.

State machine round loop with job processing phases.
Phases: EVENT -> OBSERVE -> NEGOTIATE -> ACTION -> EXECUTE -> SCORE -> OVERSIGHT
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol

from nexus.actions import Action, ActionType, validate_action
from nexus.config import NexusConfig
from nexus.events import EventGenerator
from nexus.market import MarketConfig, ResourceMarket
from nexus.observations import render, render_supervisor
from nexus.rewards import RewardComputer, RoundScore
from nexus.state import (
    AgentState,
    ClusterState,
    CoalitionProposal,
    JobStatus,
    Message,
    Resources,
)

if TYPE_CHECKING:
    from nexus.oversight import SupervisorInterface

logger = logging.getLogger(__name__)


class AgentInterface(Protocol):
    """Protocol that all agents must implement."""

    agent_id: str

    def act(self, observation: str) -> list[Action]: ...
    def on_round_end(self, score: RoundScore) -> None: ...


class SimulationEngine:
    """Runs the 7-phase simulation loop."""

    def __init__(
        self,
        config: NexusConfig,
        agents: list[AgentInterface],
        supervisor: SupervisorInterface | None = None,
        seed: int | None = None,
    ):
        self.config = config
        self.agent_map: dict[str, AgentInterface] = {a.agent_id: a for a in agents}
        self.supervisor = supervisor
        self.event_gen = EventGenerator(config, seed=seed)
        self.market = ResourceMarket(
            MarketConfig(
                commission_rate=config.commission_rate,
                price_impact=config.price_impact,
            )
        )
        self.reward_computer = RewardComputer(config.reward_weights)
        self.round_history: list[dict[str, Any]] = []

        # Initialize cluster state
        self.state = self._init_state()

    def _init_state(self) -> ClusterState:
        state = ClusterState(
            resources=self.config.cluster_resources.clone(),
            max_rounds=self.config.max_rounds,
        )

        team_names = [
            "Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta"
        ]
        for i, agent_id in enumerate(self.agent_map.keys()):
            agent_state = AgentState(
                id=agent_id,
                team_name=f"Team {team_names[i % len(team_names)]}",
                budget=self.config.starting_budget,
                reputation=self.config.starting_reputation,
                holdings=self.config.initial_holdings.clone(),
                max_actions_per_round=self.config.max_actions_per_round,
            )
            state.agents[agent_id] = agent_state

        return state

    def run(self) -> dict[str, Any]:
        """Run the full simulation. Returns final results."""
        logger.info(
            "Starting simulation: %d agents, %d rounds",
            len(self.agent_map),
            self.config.max_rounds,
        )

        while not self.state.is_finished():
            self.step()

        return self._compile_results()

    def step(self) -> dict[str, RoundScore]:
        """Execute one round (7 phases). Returns round scores."""
        round_num = self.state.current_round
        logger.info("=== Round %d/%d ===", round_num, self.state.max_rounds)

        # Save pre-round reputations
        prev_reputations = {
            aid: a.reputation for aid, a in self.state.agents.items()
        }

        # Reset per-round state
        self.state.round_messages = []
        for agent in self.state.agents.values():
            agent.actions_this_round = 0
            agent.messages_received = []

        # Per-round resource refresh (PLAN.md: resources refreshed each round)
        refresh = self.config.round_refresh
        for agent in self.state.agents.values():
            for rt in ["gpu", "cpu", "memory", "bandwidth"]:
                current = getattr(agent.holdings, rt)
                setattr(agent.holdings, rt, current + getattr(refresh, rt))

        # Phase 1: EVENTS
        events = self.event_gen.generate_events(self.state)
        self.state.events.extend(events)

        # Generate new jobs
        new_jobs = self.event_gen.generate_jobs(self.state)
        for agent_id, jobs in new_jobs.items():
            self.state.agents[agent_id].job_queue.extend(jobs)

        # Phase 2: OBSERVE + Phase 3: NEGOTIATE (combined — agents see state then act)
        # First sub-round: negotiation actions (messages, bids, offers, coalition proposals)
        observations_1: dict[str, str] = {}
        negotiate_actions: dict[str, list[Action]] = {}
        for agent_id, agent_impl in self.agent_map.items():
            obs = render(self.state, agent_id)
            observations_1[agent_id] = obs
            actions = agent_impl.act(obs)
            negotiate_actions[agent_id] = actions

        # Execute negotiation actions
        for agent_id, actions in negotiate_actions.items():
            for action in actions:
                action.agent_id = agent_id
                self._execute_negotiate_action(action)

        # Phase 4: ACTION — agents submit final actions after seeing negotiation results
        observations_2: dict[str, str] = {}
        final_actions: dict[str, list[Action]] = {}
        for agent_id, agent_impl in self.agent_map.items():
            obs = render(self.state, agent_id)
            observations_2[agent_id] = obs
            actions = agent_impl.act(obs)
            final_actions[agent_id] = actions

        # Phase 5: EXECUTE — resolve all final actions
        for agent_id, actions in final_actions.items():
            for action in actions:
                action.agent_id = agent_id
                self._execute_action(action)

        # Phase 6: SCORE
        scores = self.reward_computer.compute_round_rewards(self.state, prev_reputations)
        for agent_id, score in scores.items():
            self.state.agents[agent_id].score += score.total
            agent_impl = self.agent_map.get(agent_id)
            if agent_impl:
                agent_impl.on_round_end(score)

        # Phase 7: OVERSIGHT
        oversight_result = None
        if self.supervisor and self.config.enable_oversight:
            supervisor_obs = render_supervisor(self.state)
            oversight_result = self.supervisor.analyze(supervisor_obs, self.state)

        # Cleanup
        self.market.clear_expired(self.state)

        # Record round history
        self.round_history.append({
            "round": round_num,
            "scores": {aid: s.total for aid, s in scores.items()},
            "events": [e.description for e in events],
            "num_trades": len([
                t for t in self.state.market.completed_trades
                if t.round_executed == round_num
            ]),
            "oversight": oversight_result,
        })

        self.state.current_round += 1
        return scores

    def _execute_negotiate_action(self, action: Action) -> None:
        """Execute a negotiation-phase action (messages, bids, offers, proposals)."""
        agent = self.state.agents.get(action.agent_id)
        if not agent:
            return

        valid, reason = validate_action(action, agent, self.state)
        if not valid:
            logger.debug("Invalid negotiate action from %s: %s", action.agent_id, reason)
            return

        if action.type == ActionType.SEND_MESSAGE:
            msg = Message(
                sender_id=action.agent_id,
                recipient_id=action.target_id,
                text=action.message_text,
                round_sent=self.state.current_round,
            )
            self.state.round_messages.append(msg)
            target = self.state.agents.get(action.target_id)
            if target:
                target.messages_received.append(msg)
            agent.actions_this_round += 1

        elif action.type == ActionType.BID:
            self.market.place_bid(
                self.state, action.agent_id,
                action.resource_type, action.quantity, action.price,
            )
            agent.actions_this_round += 1

        elif action.type == ActionType.OFFER:
            self.market.place_offer(
                self.state, action.agent_id,
                action.resource_type, action.quantity, action.price,
            )
            agent.actions_this_round += 1

        elif action.type == ActionType.PROPOSE_COALITION:
            proposal = CoalitionProposal(
                proposer_id=action.agent_id,
                member_ids=action.member_ids,
                job_id=action.job_id,
                round_created=self.state.current_round,
            )
            self.state.coalitions.append(proposal)
            agent.actions_this_round += 1

        elif action.type == ActionType.PASS:
            pass  # No action cost for passing

    def _execute_action(self, action: Action) -> None:
        """Execute a final-phase action."""
        agent = self.state.agents.get(action.agent_id)
        if not agent:
            return

        valid, reason = validate_action(action, agent, self.state)
        if not valid:
            logger.debug("Invalid action from %s: %s", action.agent_id, reason)
            return

        if action.type == ActionType.ALLOCATE:
            job = next((j for j in agent.job_queue if j.id == action.job_id), None)
            if job and agent.holdings.consume(job.requirements):
                job.status = JobStatus.COMPLETED
                job.assigned_round = self.state.current_round
                agent.completed_jobs.append(job)
                agent.reputation = min(100, agent.reputation + 1.0)
            agent.actions_this_round += 1

        elif action.type == ActionType.ACCEPT_BID:
            self.market.accept_bid(self.state, action.agent_id, action.bid_id)
            agent.actions_this_round += 1

        elif action.type == ActionType.ACCEPT_OFFER:
            self.market.accept_offer(self.state, action.agent_id, action.offer_id)
            agent.actions_this_round += 1

        elif action.type == ActionType.ACCEPT_COALITION:
            proposal = next(
                (c for c in self.state.coalitions if c.id == action.proposal_id),
                None,
            )
            if proposal:
                proposal.votes[action.agent_id] = 1.0
                # Check if all members voted
                if all(m in proposal.votes for m in proposal.member_ids):
                    self._resolve_coalition(proposal)
            agent.actions_this_round += 1

        elif action.type == ActionType.REJECT_COALITION:
            proposal = next(
                (c for c in self.state.coalitions if c.id == action.proposal_id),
                None,
            )
            if proposal:
                proposal.votes[action.agent_id] = 0.0
            agent.actions_this_round += 1

        elif action.type == ActionType.PASS:
            pass

        # Re-execute negotiate-type actions that arrive in final phase
        elif action.type in (ActionType.BID, ActionType.OFFER, ActionType.SEND_MESSAGE):
            self._execute_negotiate_action(action)

    def _resolve_coalition(self, proposal: CoalitionProposal) -> None:
        """Resolve a coalition — split job across members using MoE voting."""
        if any(v == 0 for v in proposal.votes.values()):
            return  # Someone rejected

        job = None
        proposer = self.state.agents.get(proposal.proposer_id)
        if proposer:
            job = next((j for j in proposer.job_queue if j.id == proposal.job_id), None)

        if not job:
            return

        # Check if combined resources suffice
        combined = Resources()
        for member_id in proposal.member_ids:
            member = self.state.agents.get(member_id)
            if member:
                for rt in ["gpu", "cpu", "memory", "bandwidth"]:
                    current = getattr(combined, rt)
                    setattr(combined, rt, current + getattr(member.holdings, rt))

        if not combined.can_afford(job.requirements):
            return

        # Split costs proportionally
        n_members = len(proposal.member_ids)
        for member_id in proposal.member_ids:
            member = self.state.agents.get(member_id)
            if member:
                share = Resources(
                    gpu=job.requirements.gpu / n_members,
                    cpu=job.requirements.cpu / n_members,
                    memory=job.requirements.memory / n_members,
                    bandwidth=job.requirements.bandwidth / n_members,
                )
                member.holdings.consume(share)

        # Complete the job
        job.status = JobStatus.COMPLETED
        job.assigned_round = self.state.current_round
        proposal.accepted = True

        # Split reward
        reward_per_member = job.reward / n_members
        for member_id in proposal.member_ids:
            member = self.state.agents.get(member_id)
            if member:
                member.completed_jobs.append(job)
                member.reputation = min(100, member.reputation + 1.5)

    def _compile_results(self) -> dict[str, Any]:
        """Compile final simulation results."""
        agent_results = {}
        for aid, agent in self.state.agents.items():
            round_scores = [
                rh["scores"].get(aid, 0)
                for rh in self.round_history
            ]
            sharpe = self.reward_computer.compute_sharpe_ratio(round_scores)
            agent_results[aid] = {
                "team_name": agent.team_name,
                "final_score": agent.score,
                "budget": agent.budget,
                "reputation": agent.reputation,
                "jobs_completed": len(agent.completed_jobs),
                "jobs_missed": len([j for j in agent.job_queue if j.status == JobStatus.MISSED]),
                "trades": len(agent.trade_history),
                "sharpe_ratio": sharpe,
            }

        market_stats = self.market.get_market_stats(self.state)

        # Social welfare metrics
        all_scores = [a.score for a in self.state.agents.values()]
        total_welfare = sum(all_scores)
        from nexus.observations import _gini
        gini = _gini(all_scores)

        return {
            "rounds_played": self.state.current_round - 1,
            "agents": agent_results,
            "market": market_stats,
            "social_welfare": total_welfare,
            "gini_coefficient": gini,
            "round_history": self.round_history,
        }
