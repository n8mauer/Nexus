"""Seeded Nexus demo: tiny preset, greedy vs random, fixed seed.

Implements Gate 2 of docs/ACCEPTANCE.md. Runs a deterministic simulation
through the existing engine, prints the final scoreboard, Gini coefficient,
and market volume, and writes the results journal (journal.md,
metrics.jsonl, final_results.json) to the output directory.

Usage:
    python demo.py [--seed 42] [--output demo_output] [--svg PATH]

Exits 0 on success, 1 on failure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rich.console import Console
from rich.table import Table

from agents.greedy_agent import GreedyAgent
from agents.random_agent import RandomAgent
from nexus.config import get_preset
from nexus.engine import SimulationEngine
from nexus.journal import DualFormatJournal
from nexus.persistence import save_results

JOURNAL_FILES = ("journal.md", "metrics.jsonl", "final_results.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the seeded Nexus demo (tiny preset, greedy vs random)."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
    parser.add_argument(
        "--output", default="demo_output", help="Output directory (default demo_output)"
    )
    parser.add_argument(
        "--svg",
        default=None,
        help="Optional path to also export the console output as an SVG image",
    )
    args = parser.parse_args(argv)

    console = Console(record=bool(args.svg), width=100 if args.svg else None)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Start from a clean journal so repeated runs do not append to old files.
    for name in JOURNAL_FILES:
        stale = output_dir / name
        if stale.exists():
            stale.unlink()

    config = get_preset("tiny")
    agent_list = [GreedyAgent("agent_0"), RandomAgent("agent_1", seed=args.seed)]
    config.num_agents = len(agent_list)

    journal = DualFormatJournal(output_dir=str(output_dir))
    journal.log_simulation_start(
        {
            "preset": "tiny",
            "num_agents": config.num_agents,
            "max_rounds": config.max_rounds,
            "agents": "greedy,random",
            "seed": args.seed,
        }
    )

    engine = SimulationEngine(config, agent_list, seed=args.seed)

    console.print(
        f"[bold]Nexus demo[/bold]: preset=tiny, agents=greedy vs random, seed={args.seed}"
    )
    console.print(
        f"Running {config.max_rounds} rounds with {config.num_agents} agents...\n"
    )

    results = engine.run()

    for rh in results.get("round_history", []):
        journal.log_round(
            round_num=rh["round"],
            scores=rh["scores"],
            events=rh["events"],
            num_trades=rh["num_trades"],
            oversight=rh.get("oversight"),
        )
    journal.log_final_results(results)
    save_results(results, output_dir / "final_results.json")

    agent_types = {"agent_0": "greedy", "agent_1": "random"}
    table = Table(title="Final Scoreboard")
    table.add_column("Agent", style="cyan")
    table.add_column("Policy")
    table.add_column("Score", justify="right", style="green")
    table.add_column("Jobs Done", justify="right")
    table.add_column("Jobs Missed", justify="right", style="red")
    table.add_column("Trades", justify="right")
    table.add_column("Budget", justify="right")
    table.add_column("Reputation", justify="right")

    agents = results.get("agents", {})
    for aid, info in sorted(
        agents.items(), key=lambda x: x[1].get("final_score", 0), reverse=True
    ):
        table.add_row(
            info.get("team_name", aid),
            agent_types.get(aid, "?"),
            f"{info.get('final_score', 0):.0f}",
            str(info.get("jobs_completed", 0)),
            str(info.get("jobs_missed", 0)),
            str(info.get("trades", 0)),
            f"${info.get('budget', 0):.0f}",
            f"{info.get('reputation', 0):.0f}",
        )
    console.print(table)

    market = results.get("market", {})
    console.print(f"\nRounds played: {results.get('rounds_played', 0)}")
    console.print(f"Social welfare: {results.get('social_welfare', 0):.0f}")
    console.print(f"Gini coefficient: {results.get('gini_coefficient', 0):.3f}")
    console.print(
        f"Market volume: ${market.get('total_volume', 0):.0f} "
        f"across {market.get('num_trades', 0)} trades"
    )
    console.print(
        f"Journal written to {output_dir} ({', '.join(JOURNAL_FILES)})"
    )

    if args.svg:
        svg_path = Path(args.svg)
        svg_path.parent.mkdir(parents=True, exist_ok=True)
        console.save_svg(str(svg_path), title="python demo.py")
        print(f"SVG saved to {svg_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
