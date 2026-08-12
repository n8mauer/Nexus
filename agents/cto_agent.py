"""CTO agent for multi-actor mode (Halluminate track).

Issues high-level directives to worker agents. Must discover
each worker's strengths/biases and adapt management style.
"""

from __future__ import annotations

import re
from typing import Any

from agents.base import BaseAgent
from nexus.actions import Action, ActionType
from nexus.multi_actor import Directive, DirectiveType, WorkerAgent


class CTOAgent(BaseAgent):
    """CTO that manages workers via directives rather than direct actions."""

    def __init__(self, agent_id: str, workers: list[WorkerAgent]):
        super().__init__(agent_id)
        self.workers = {w.agent_id: w for w in workers}
        self._worker_performance: dict[str, list[float]] = {
            w.agent_id: [] for w in workers
        }
        self._directive_count = 0

    def act(self, observation: str) -> list[Action]:
        """Analyze observation and issue directives to workers."""
        # Parse key information
        jobs = self._parse_jobs(observation)
        holdings = self._parse_holdings(observation)
        budget = self._parse_budget(observation)

        # Strategy: assign jobs to workers based on their reliability
        directives = self._plan_directives(jobs, holdings, budget)

        # Issue directives to workers
        all_actions: list[Action] = []
        for directive in directives:
            worker = self.workers.get(directive.worker_id)
            if worker:
                worker.receive_directive(directive)
                actions = worker.execute_directives(observation)
                all_actions.extend(actions)
                self._directive_count += 1

        return all_actions or [Action(type=ActionType.PASS, agent_id=self.agent_id)]

    def _plan_directives(
        self,
        jobs: list[dict],
        holdings: dict[str, float],
        budget: float,
    ) -> list[Directive]:
        """Plan directives based on current state and worker performance."""
        directives: list[Directive] = []
        worker_ids = list(self.workers.keys())

        if not worker_ids:
            return directives

        # Sort jobs by urgency
        jobs.sort(key=lambda j: j.get("reward", 0) / max(1, j.get("deadline_left", 1)), reverse=True)

        # Assign jobs to workers round-robin, best workers first
        sorted_workers = sorted(
            worker_ids,
            key=lambda wid: self._avg_performance(wid),
            reverse=True,
        )

        for i, job in enumerate(jobs[:len(worker_ids)]):
            worker_id = sorted_workers[i % len(sorted_workers)]

            # Check if we can allocate
            can_allocate = True
            for rt in ["gpu", "cpu", "memory", "bandwidth"]:
                if holdings.get(rt, 0) < job.get(f"req_{rt}", 0):
                    can_allocate = False
                    break

            if can_allocate:
                directives.append(Directive(
                    type=DirectiveType.ALLOCATE_JOB,
                    worker_id=worker_id,
                    params={"job_id": job["id"]},
                    instruction=f"Execute job {job['id']} ({job.get('description', '')})",
                ))
            elif budget > 100:
                # Bid for missing resources
                for rt in ["gpu", "cpu", "memory", "bandwidth"]:
                    shortfall = job.get(f"req_{rt}", 0) - holdings.get(rt, 0)
                    if shortfall > 0:
                        directives.append(Directive(
                            type=DirectiveType.BID_FOR,
                            worker_id=worker_id,
                            params={
                                "resource_type": rt,
                                "quantity": shortfall,
                                "price": 50.0,
                            },
                            instruction=f"Buy {shortfall:.0f} {rt} for job {job['id']}",
                        ))
                        break

        return directives

    def _avg_performance(self, worker_id: str) -> float:
        perf = self._worker_performance.get(worker_id, [])
        return sum(perf) / len(perf) if perf else 0.5

    def _parse_jobs(self, obs: str) -> list[dict]:
        jobs = []
        current_round = 1
        rm = re.search(r"ROUND (\d+)", obs)
        if rm:
            current_round = int(rm.group(1))

        for m in re.finditer(
            r'\[(J-\w+)\]\s*"([^"]+)"\s*--\s*needs\s+([\d.]+)\s+GPU,\s*([\d.]+)\s+CPU,\s*([\d.]+)\s+GB\s*--\s*deadline round (\d+)\s*--\s*reward \$([\d.]+)',
            obs,
        ):
            jobs.append({
                "id": m.group(1),
                "description": m.group(2),
                "req_gpu": float(m.group(3)),
                "req_cpu": float(m.group(4)),
                "req_memory": float(m.group(5)),
                "deadline": int(m.group(6)),
                "deadline_left": int(m.group(6)) - current_round,
                "reward": float(m.group(7)),
            })
        return jobs

    def _parse_holdings(self, obs: str) -> dict[str, float]:
        m = re.search(
            r"Holdings:\s*([\d.]+)\s+GPU\s*\|\s*([\d.]+)\s+CPU\s*\|\s*([\d.]+)\s+GB\s+RAM\s*\|\s*([\d.]+)\s+Gbps\s+BW",
            obs,
        )
        if m:
            return {"gpu": float(m.group(1)), "cpu": float(m.group(2)),
                    "memory": float(m.group(3)), "bandwidth": float(m.group(4))}
        return {"gpu": 0, "cpu": 0, "memory": 0, "bandwidth": 0}

    def _parse_budget(self, obs: str) -> float:
        m = re.search(r"Budget:\s*\$([\d.]+)", obs)
        return float(m.group(1)) if m else 0
