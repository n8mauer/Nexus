"""Metric computation and evaluation for completed simulations.

Portfolio-style metrics and reward model evaluation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def compute_metrics(results: dict[str, Any]) -> dict[str, Any]:
    """Compute comprehensive evaluation metrics from simulation results."""
    agents = results.get("agents", {})
    round_history = results.get("round_history", [])

    metrics: dict[str, Any] = {}

    # Per-agent metrics
    for aid, info in agents.items():
        total_jobs = info.get("jobs_completed", 0) + info.get("jobs_missed", 0)
        completion_rate = info["jobs_completed"] / total_jobs if total_jobs > 0 else 0

        # Round-by-round scores for this agent
        round_scores = [rh["scores"].get(aid, 0) for rh in round_history]
        sharpe = _sharpe(round_scores)

        metrics[aid] = {
            "team_name": info.get("team_name", aid),
            "final_score": info["final_score"],
            "job_completion_rate": completion_rate,
            "sharpe_ratio": sharpe,
            "trades_per_round": info["trades"] / max(1, results.get("rounds_played", 1)),
            "final_reputation": info["reputation"],
        }

    # Aggregate metrics
    scores = [a["final_score"] for a in agents.values()]
    metrics["aggregate"] = {
        "social_welfare": sum(scores),
        "gini_coefficient": results.get("gini_coefficient", 0),
        "mean_score": sum(scores) / len(scores) if scores else 0,
        "score_std": _std(scores),
        "market_volume": results.get("market", {}).get("total_volume", 0),
        "total_trades": results.get("market", {}).get("num_trades", 0),
    }

    return metrics


def _sharpe(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    std = _std(values)
    return mean / std if std > 0 else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5


def compare_results(*result_paths: str) -> None:
    """Compare multiple simulation results."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="Simulation Comparison")
    table.add_column("Metric")

    results_list = []
    for path in result_paths:
        with open(path) as f:
            results_list.append(json.load(f))
        table.add_column(Path(path).stem)

    metric_names = ["social_welfare", "gini_coefficient", "total_trades"]
    for name in metric_names:
        row = [name]
        for results in results_list:
            metrics = compute_metrics(results)
            val = metrics["aggregate"].get(name, 0)
            row.append(f"{val:.2f}")
        table.add_row(*row)

    console.print(table)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <results1.json> [results2.json ...]")
        sys.exit(1)

    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            results = json.load(f)
        metrics = compute_metrics(results)
        print(json.dumps(metrics, indent=2))
    else:
        compare_results(*sys.argv[1:])
