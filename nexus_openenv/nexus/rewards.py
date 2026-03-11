"""Reward computation with weighted multi-signal scoring.

Rule-based reward model with Sharpe ratio for risk-adjusted evaluation.
Per-component scoring: each job/trade evaluated independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from nexus.state import AgentState, ClusterState, JobStatus


@dataclass
class RoundScore:
    job_completion: float = 0.0
    deadline_penalties: float = 0.0
    efficiency: float = 0.0
    reputation_delta: float = 0.0
    coalition_bonus: float = 0.0
    idle_resource_cost: float = 0.0
    total: float = 0.0
    components: dict[str, float] = field(default_factory=dict)


class RewardComputer:
    """Compute rewards using weighted multi-signal model.

    Weights follow a rule-based reward model:
    - job_completion: 0.4
    - efficiency: 0.2
    - reputation: 0.15
    - coalition: 0.15
    - fairness: 0.1
    """

    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = weights or {
            "job_completion": 0.4,
            "efficiency": 0.2,
            "reputation": 0.15,
            "coalition": 0.15,
            "fairness": 0.1,
        }

    def compute_round_rewards(
        self, state: ClusterState, prev_reputations: dict[str, float]
    ) -> dict[str, RoundScore]:
        """Compute per-agent rewards for the current round."""
        scores: dict[str, RoundScore] = {}

        # Compute fairness (Gini) across all agents
        all_scores_list = [a.score for a in state.agents.values()]
        gini = _gini(all_scores_list)
        fairness_bonus = max(0, (1 - gini) * 50)  # reward equality

        for agent_id, agent in state.agents.items():
            score = RoundScore()

            # 1. Job completion rewards
            for job in agent.job_queue:
                if job.status == JobStatus.COMPLETED and job.assigned_round == state.current_round:
                    score.job_completion += job.reward

            # 2. Deadline penalties
            for job in agent.job_queue:
                if job.is_expired(state.current_round) and job.status == JobStatus.PENDING:
                    score.deadline_penalties += job.penalty_on_miss
                    job.status = JobStatus.MISSED

            # 3. Efficiency: resource utilization ratio
            total_held = agent.holdings.total()
            total_needed = sum(j.requirements.total() for j in agent.active_jobs())
            if total_held > 0:
                utilization = min(1.0, total_needed / total_held)
                score.efficiency = utilization * 100
            else:
                score.efficiency = 0

            # 4. Reputation delta
            prev_rep = prev_reputations.get(agent_id, 50.0)
            score.reputation_delta = agent.reputation - prev_rep

            # 5. Coalition bonus (from completed collaborative jobs)
            for job in agent.job_queue:
                if job.collaborative and job.status == JobStatus.COMPLETED:
                    score.coalition_bonus += job.reward * 0.1  # 10% bonus

            # 6. Idle resource cost
            idle = max(0, total_held - total_needed) if total_held > 0 else 0
            score.idle_resource_cost = idle * 0.05

            # Weighted total
            score.components = {
                "job_completion": score.job_completion,
                "deadline_penalties": -score.deadline_penalties,
                "efficiency": score.efficiency,
                "reputation": score.reputation_delta * 10,
                "coalition": score.coalition_bonus,
                "fairness": fairness_bonus,
                "idle_cost": -score.idle_resource_cost,
            }

            score.total = (
                self.weights["job_completion"] * score.job_completion
                - score.deadline_penalties
                + self.weights["efficiency"] * score.efficiency
                + self.weights["reputation"] * score.reputation_delta * 10
                + self.weights["coalition"] * score.coalition_bonus
                + self.weights["fairness"] * fairness_bonus
                - score.idle_resource_cost
            )

            scores[agent_id] = score

        return scores

    def compute_supervisor_reward(
        self,
        correct_flags: int,
        false_flags: int,
        total_anomalies: int,
        explanations_given: int,
    ) -> float:
        """Compute supervisor reward (Fleet AI track)."""
        recall = correct_flags / total_anomalies if total_anomalies > 0 else 0
        precision = correct_flags / (correct_flags + false_flags) if (correct_flags + false_flags) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return (
            correct_flags * 50
            - false_flags * 30
            + f1 * 200
            + explanations_given * 10
        )

    def compute_cto_reward(
        self,
        team_total_score: float,
        num_directives: int,
        worker_utilization: float,
        miscommunications: int,
    ) -> float:
        """Compute CTO reward (Halluminate track)."""
        return (
            team_total_score
            - num_directives * 2  # penalize micromanagement
            + worker_utilization * 100
            - miscommunications * 20
        )

    def compute_sharpe_ratio(self, round_scores: list[float]) -> float:
        """Sharpe-like ratio for risk-adjusted agent performance."""
        if len(round_scores) < 2:
            return 0.0
        mean = sum(round_scores) / len(round_scores)
        variance = sum((s - mean) ** 2 for s in round_scores) / len(round_scores)
        std = variance ** 0.5
        if std == 0:
            return 0.0
        return mean / std


def _gini(values: list[float]) -> float:
    if not values or all(v == 0 for v in values):
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    total = sum(sorted_vals)
    if total == 0:
        return 0.0
    cumulative = sum((i + 1) * v for i, v in enumerate(sorted_vals))
    return (2 * cumulative) / (n * total) - (n + 1) / n
