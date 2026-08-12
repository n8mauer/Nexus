"""Greedy heuristic agent — allocates highest-reward affordable jobs, bids for shortfalls."""

from __future__ import annotations

import re

from agents.base import BaseAgent
from nexus.actions import Action, ActionType
from nexus.state import ResourceType


class GreedyAgent(BaseAgent):
    """Greedy strategy: prioritize completing highest-reward jobs."""

    def act(self, observation: str) -> list[Action]:
        actions: list[Action] = []
        remaining = self._parse_remaining(observation)
        if remaining <= 0:
            return [Action(type=ActionType.PASS, agent_id=self.agent_id)]

        # Parse own state from observation
        jobs = self._parse_jobs(observation)
        holdings = self._parse_holdings(observation)
        budget = self._parse_budget(observation)
        bids = self._parse_bids(observation)
        offers = self._parse_offers(observation)

        # Sort jobs by reward/deadline ratio (urgency * value)
        jobs.sort(key=lambda j: j["reward"] / max(1, j["deadline_left"]), reverse=True)

        # Strategy 1: Allocate jobs we can afford
        for job in jobs:
            if remaining <= 0:
                break
            if self._can_afford_job(job, holdings):
                actions.append(Action(
                    type=ActionType.ALLOCATE,
                    agent_id=self.agent_id,
                    params={"job_id": job["id"]},
                ))
                # Deduct from tracked holdings
                for rt in ["gpu", "cpu", "memory", "bandwidth"]:
                    holdings[rt] = holdings.get(rt, 0) - job.get(f"req_{rt}", 0)
                remaining -= 1

        # Strategy 2: Accept offers for resources we need
        for job in jobs:
            if remaining <= 0:
                break
            shortfall = self._compute_shortfall(job, holdings)
            if not shortfall:
                continue
            for rt, needed in shortfall.items():
                if remaining <= 0:
                    break
                # Find cheapest offer for this resource
                matching = [o for o in offers if o["resource_type"] == rt]
                matching.sort(key=lambda o: o["price"])
                for offer in matching:
                    if offer["quantity"] >= needed and budget >= offer["price"] * offer["quantity"]:
                        actions.append(Action(
                            type=ActionType.ACCEPT_OFFER,
                            agent_id=self.agent_id,
                            params={"offer_id": offer["id"]},
                        ))
                        holdings[rt] = holdings.get(rt, 0) + offer["quantity"]
                        budget -= offer["price"] * offer["quantity"]
                        remaining -= 1
                        break

        # Strategy 3: Bid for resources we need but can't find offers for
        for job in jobs:
            if remaining <= 0:
                break
            shortfall = self._compute_shortfall(job, holdings)
            if not shortfall:
                continue
            for rt, needed in shortfall.items():
                if remaining <= 0:
                    break
                price = self._estimate_price(rt, observation)
                if budget >= price * needed:
                    actions.append(Action(
                        type=ActionType.BID,
                        agent_id=self.agent_id,
                        params={
                            "resource_type": rt,
                            "quantity": needed,
                            "price": price,
                        },
                    ))
                    budget -= price * needed
                    remaining -= 1

        # Strategy 4: Sell excess resources we don't need
        if remaining > 0:
            needed_total = self._total_needed(jobs)
            for rt in ["gpu", "cpu", "memory", "bandwidth"]:
                if remaining <= 0:
                    break
                excess = holdings.get(rt, 0) - needed_total.get(rt, 0)
                if excess > 2:
                    price = self._estimate_price(rt, observation) * 1.1
                    actions.append(Action(
                        type=ActionType.OFFER,
                        agent_id=self.agent_id,
                        params={
                            "resource_type": rt,
                            "quantity": excess * 0.5,
                            "price": price,
                        },
                    ))
                    remaining -= 1

        return actions or [Action(type=ActionType.PASS, agent_id=self.agent_id)]

    def _parse_remaining(self, obs: str) -> int:
        m = re.search(r"ACTIONS REMAINING:\s*(\d+)", obs)
        return int(m.group(1)) if m else 3

    def _parse_jobs(self, obs: str) -> list[dict]:
        jobs = []
        current_round = 1
        rm = re.search(r"ROUND (\d+) of (\d+)", obs)
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
            return {
                "gpu": float(m.group(1)),
                "cpu": float(m.group(2)),
                "memory": float(m.group(3)),
                "bandwidth": float(m.group(4)),
            }
        return {"gpu": 0, "cpu": 0, "memory": 0, "bandwidth": 0}

    def _parse_budget(self, obs: str) -> float:
        m = re.search(r"Budget:\s*\$([\d.]+)", obs)
        return float(m.group(1)) if m else 0

    def _parse_bids(self, obs: str) -> list[dict]:
        bids = []
        for m in re.finditer(
            r"BID (B-\w+) by (\w+): wants ([\d.]+) (\w+) @ \$([\d.]+)", obs
        ):
            bids.append({
                "id": m.group(1),
                "agent": m.group(2),
                "quantity": float(m.group(3)),
                "resource_type": m.group(4),
                "price": float(m.group(5)),
            })
        return bids

    def _parse_offers(self, obs: str) -> list[dict]:
        offers = []
        for m in re.finditer(
            r"OFFER (O-\w+) by (\w+): selling ([\d.]+) (\w+) @ \$([\d.]+)", obs
        ):
            offers.append({
                "id": m.group(1),
                "agent": m.group(2),
                "quantity": float(m.group(3)),
                "resource_type": m.group(4),
                "price": float(m.group(5)),
            })
        return offers

    def _can_afford_job(self, job: dict, holdings: dict[str, float]) -> bool:
        for rt in ["gpu", "cpu", "memory", "bandwidth"]:
            if holdings.get(rt, 0) < job.get(f"req_{rt}", 0):
                return False
        return True

    def _compute_shortfall(self, job: dict, holdings: dict[str, float]) -> dict[str, float]:
        shortfall = {}
        for rt in ["gpu", "cpu", "memory", "bandwidth"]:
            needed = job.get(f"req_{rt}", 0) - holdings.get(rt, 0)
            if needed > 0:
                shortfall[rt] = needed
        return shortfall

    def _total_needed(self, jobs: list[dict]) -> dict[str, float]:
        totals: dict[str, float] = {"gpu": 0, "cpu": 0, "memory": 0, "bandwidth": 0}
        for job in jobs:
            for rt in totals:
                totals[rt] += job.get(f"req_{rt}", 0)
        return totals

    def _estimate_price(self, resource_type: str, obs: str) -> float:
        base_prices = {"gpu": 100, "cpu": 30, "memory": 20, "bandwidth": 40}
        return base_prices.get(resource_type, 50)
