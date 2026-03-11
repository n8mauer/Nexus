"""Typer CLI for running Nexus simulations.

Typer CLI with subcommands for run, evaluate.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.greedy_agent import GreedyAgent
from agents.random_agent import RandomAgent
from nexus.config import get_preset, NexusConfig
from nexus.engine import SimulationEngine
from nexus.journal import DualFormatJournal
from nexus.persistence import save_results

app = typer.Typer(help="Nexus -- Multi-Agent Compute Cluster Negotiation")
console = Console()


def _build_agents(agent_specs: str, config: NexusConfig) -> list:
    """Parse agent spec string like 'random,greedy,greedy' into agent instances."""
    agents = []
    specs = [s.strip() for s in agent_specs.split(",")]

    for i, spec in enumerate(specs):
        agent_id = f"agent_{i}"
        if spec == "random":
            agents.append(RandomAgent(agent_id, seed=i))
        elif spec == "greedy":
            agents.append(GreedyAgent(agent_id))
        elif spec == "llm":
            try:
                from agents.llm_agent import LLMAgent
                agents.append(LLMAgent(agent_id))
            except ImportError:
                console.print(f"[yellow]LLM agent not available, using greedy for slot {i}[/yellow]")
                agents.append(GreedyAgent(agent_id))
        else:
            console.print(f"[yellow]Unknown agent type '{spec}', using random[/yellow]")
            agents.append(RandomAgent(agent_id, seed=i))

    return agents


@app.command("run")
def run(
    preset: str = typer.Option("standard", help="Config preset: tiny, standard, large, oversight, multi_actor"),
    agents: str = typer.Option("greedy,greedy,greedy,greedy", help="Comma-separated agent types: random, greedy, llm"),
    seed: Optional[int] = typer.Option(None, help="Random seed"),
    output: str = typer.Option("results", help="Output directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    inject_collusion: bool = typer.Option(False, help="Inject collusion scenarios (oversight mode)"),
) -> None:
    """Run a full Nexus simulation."""
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)

    config = get_preset(preset)
    if inject_collusion:
        config.enable_oversight = True
        config.inject_collusion = True

    agent_list = _build_agents(agents, config)
    config.num_agents = len(agent_list)

    supervisor = None
    if config.enable_oversight:
        try:
            from agents.supervisor_agent import SupervisorAgent
            supervisor = SupervisorAgent()
        except ImportError:
            console.print("[yellow]Supervisor agent not available[/yellow]")

    journal = DualFormatJournal(output_dir=output)
    journal.log_simulation_start({
        "preset": preset,
        "num_agents": config.num_agents,
        "max_rounds": config.max_rounds,
        "agents": agents,
    })

    engine = SimulationEngine(config, agent_list, supervisor=supervisor, seed=seed)

    console.print(f"\n[bold]Nexus Simulation[/bold] -- preset={preset}, agents={agents}")
    console.print(f"Running {config.max_rounds} rounds with {config.num_agents} agents...\n")

    results = engine.run()

    # Log round history to journal
    for rh in results.get("round_history", []):
        journal.log_round(
            round_num=rh["round"],
            scores=rh["scores"],
            events=rh["events"],
            num_trades=rh["num_trades"],
            oversight=rh.get("oversight"),
        )

    journal.log_final_results(results)
    save_results(results, Path(output) / "final_results.json")

    _print_results(results)
    console.print(f"\n[green]Results saved to {output}/[/green]")


@app.command("evaluate")
def evaluate(
    results_path: str = typer.Argument(help="Path to final_results.json"),
) -> None:
    """Evaluate a completed simulation's results."""
    from nexus.persistence import load_results

    results = load_results(results_path)
    _print_results(results)


def _print_results(results: dict) -> None:
    """Pretty-print simulation results with rich tables."""
    table = Table(title="Final Standings")
    table.add_column("Agent", style="cyan")
    table.add_column("Score", justify="right", style="green")
    table.add_column("Jobs Done", justify="right")
    table.add_column("Jobs Missed", justify="right", style="red")
    table.add_column("Trades", justify="right")
    table.add_column("Budget", justify="right")
    table.add_column("Reputation", justify="right")
    table.add_column("Sharpe", justify="right")

    agents = results.get("agents", {})
    for aid, info in sorted(agents.items(), key=lambda x: x[1].get("final_score", 0), reverse=True):
        table.add_row(
            info.get("team_name", aid),
            f"{info.get('final_score', 0):.0f}",
            str(info.get("jobs_completed", 0)),
            str(info.get("jobs_missed", 0)),
            str(info.get("trades", 0)),
            f"${info.get('budget', 0):.0f}",
            f"{info.get('reputation', 0):.0f}",
            f"{info.get('sharpe_ratio', 0):.2f}",
        )

    console.print(table)
    console.print(f"\nSocial Welfare: {results.get('social_welfare', 0):.0f}")
    console.print(f"Gini Coefficient: {results.get('gini_coefficient', 0):.3f}")

    market = results.get("market", {})
    if market:
        console.print(f"Market Volume: ${market.get('total_volume', 0):.0f}")
        console.print(f"Total Trades: {market.get('num_trades', 0)}")


if __name__ == "__main__":
    app()
