"""Coalition manager with MoE voting on resource splits.

MoE per-option voting for coalition resource splits.
Coalition members vote on splits; confidence-weighted aggregation with 1.5x primary weight.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nexus.state import (
    AgentState,
    ClusterState,
    CoalitionProposal,
    JobStatus,
    Resources,
    ResourceType,
)


@dataclass
class SplitVote:
    """A single member's vote on how to split resources for a coalition job."""
    agent_id: str
    proposed_splits: dict[str, dict[str, float]]  # {member_id: {resource_type: amount}}
    confidence: float = 0.5  # 0-1 confidence in this split
    is_proposer: bool = False


@dataclass
class CoalitionResult:
    accepted: bool = False
    final_splits: dict[str, Resources] = field(default_factory=dict)
    votes_for: int = 0
    votes_against: int = 0
    rejection_reason: str = ""


class CoalitionManager:
    """Manages coalition voting using MoE per-option confidence-weighted aggregation.

    Each coalition member is an "expert" who votes on resource splits.
    The proposer gets 1.5x weight (primary expert).
    An elimination pre-pass acts as a veto round.
    """

    PROPOSER_WEIGHT = 1.5
    MEMBER_WEIGHT = 1.0
    ACCEPTANCE_THRESHOLD = 0.10  # min weighted average score per option

    def propose(
        self,
        state: ClusterState,
        proposer_id: str,
        member_ids: list[str],
        job_id: str,
    ) -> CoalitionProposal:
        """Create a new coalition proposal."""
        proposal = CoalitionProposal(
            proposer_id=proposer_id,
            member_ids=member_ids,
            job_id=job_id,
            round_created=state.current_round,
        )
        state.coalitions.append(proposal)
        return proposal

    def vote(
        self,
        proposal: CoalitionProposal,
        vote: SplitVote,
    ) -> None:
        """Record a member's vote on the coalition split."""
        proposal.votes[vote.agent_id] = vote.confidence
        if not proposal.resource_splits:
            proposal.resource_splits = {}
        # Store the proposed split
        for member_id, split in vote.proposed_splits.items():
            if member_id not in proposal.resource_splits:
                proposal.resource_splits[member_id] = Resources()

    def resolve(
        self,
        state: ClusterState,
        proposal: CoalitionProposal,
        member_votes: list[SplitVote],
    ) -> CoalitionResult:
        """Resolve a coalition using MoE voting aggregation.

        1. Elimination pre-pass (veto round)
        2. Confidence-weighted per-option voting
        3. Reconciliation with proposer's splits
        """
        result = CoalitionResult()

        # Find the job
        proposer = state.agents.get(proposal.proposer_id)
        if not proposer:
            result.rejection_reason = "Proposer not found"
            return result

        job = next((j for j in proposer.job_queue if j.id == proposal.job_id), None)
        if not job:
            result.rejection_reason = "Job not found"
            return result

        # Phase 1: Elimination pre-pass (veto round)
        for vote in member_votes:
            if vote.confidence < 0.1:
                result.rejection_reason = f"Agent {vote.agent_id} vetoed (confidence {vote.confidence:.2f})"
                result.votes_against += 1
                return result
            if vote.confidence >= 0.5:
                result.votes_for += 1
            else:
                result.votes_against += 1

        if result.votes_against > result.votes_for:
            result.rejection_reason = "Majority rejected"
            return result

        # Phase 2: Confidence-weighted per-option voting on splits
        # Each member proposes a split; aggregate with confidence weights
        n_members = len(member_votes)
        aggregated_splits: dict[str, dict[ResourceType, float]] = {
            mid: {rt: 0.0 for rt in ResourceType}
            for mid in proposal.member_ids
        }

        for vote in member_votes:
            weight = self.PROPOSER_WEIGHT if vote.is_proposer else self.MEMBER_WEIGHT
            weighted_conf = weight * vote.confidence

            for member_id, split in vote.proposed_splits.items():
                if member_id not in aggregated_splits:
                    continue
                for rt_str, amount in split.items():
                    try:
                        rt = ResourceType(rt_str)
                        aggregated_splits[member_id][rt] += weighted_conf * amount
                    except ValueError:
                        continue

        # Normalize by total weight
        total_weight = sum(
            (self.PROPOSER_WEIGHT if v.is_proposer else self.MEMBER_WEIGHT) * v.confidence
            for v in member_votes
        )
        if total_weight > 0:
            for member_id in aggregated_splits:
                for rt in ResourceType:
                    aggregated_splits[member_id][rt] /= total_weight

        # Phase 3: Validate splits cover job requirements
        total_split = {rt: 0.0 for rt in ResourceType}
        for member_id, splits in aggregated_splits.items():
            for rt, amount in splits.items():
                total_split[rt] += amount

        for rt in ResourceType:
            if total_split[rt] < job.requirements.get(rt) * 0.95:  # 5% tolerance
                result.rejection_reason = f"Splits don't cover {rt.value} requirement"
                return result

        # Phase 4: Check members can afford their splits
        final_splits: dict[str, Resources] = {}
        for member_id in proposal.member_ids:
            member = state.agents.get(member_id)
            if not member:
                result.rejection_reason = f"Member {member_id} not found"
                return result
            split_res = Resources()
            for rt in ResourceType:
                split_res.set(rt, aggregated_splits[member_id][rt])
            if not member.holdings.can_afford(split_res):
                result.rejection_reason = f"Member {member_id} can't afford split"
                return result
            final_splits[member_id] = split_res

        # Execute: deduct resources, complete job
        for member_id, split_res in final_splits.items():
            member = state.agents.get(member_id)
            if member:
                member.holdings.consume(split_res)

        job.status = JobStatus.COMPLETED
        job.assigned_round = state.current_round
        proposal.accepted = True

        result.accepted = True
        result.final_splits = final_splits
        return result
