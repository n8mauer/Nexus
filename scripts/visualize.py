"""Rich terminal visualization for simulation replay.

Live-updating panels for real-time simulation state display.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

console = Console()


def replay(results_path: str, speed: float = 0.5) -> None:
    """Replay a simulation from saved results."""
    with open(results_path) as f:
        results = json.load(f)

    agents = results.get("agents", {})
    round_history = results.get("round_history", [])

    console.print(f"\n[bold]Nexus Simulation Replay[/bold] -- {len(agents)} agents, {len(round_history)} rounds\n")

    cumulative_scores: dict[str, float] = {aid: 0.0 for aid in agents}

    for rh in round_history:
        round_num = rh["round"]
        scores = rh.get("scores", {})
        events = rh.get("events", [])
        num_trades = rh.get("num_trades", 0)

        # Update cumulative
        for aid, s in scores.items():
            cumulative_scores[aid] = cumulative_scores.get(aid, 0) + s

        # Build round display
        table = Table(title=f"Round {round_num}", show_header=True)
        table.add_column("Agent", style="cyan")
        table.add_column("Round Score", justify="right")
        table.add_column("Total Score", justify="right", style="green")

        for aid in sorted(cumulative_scores, key=lambda x: cumulative_scores[x], reverse=True):
            team_name = agents.get(aid, {}).get("team_name", aid)
            round_score = scores.get(aid, 0)
            total = cumulative_scores[aid]
            table.add_row(team_name, f"{round_score:.0f}", f"{total:.0f}")

        console.print(table)

        if events:
            for ev in events:
                console.print(f"  [yellow]Event: {ev}[/yellow]")
        if num_trades > 0:
            console.print(f"  Trades: {num_trades}")

        oversight = rh.get("oversight")
        if oversight and isinstance(oversight, dict):
            flags = oversight.get("flags", [])
            if flags:
                for flag in flags:
                    console.print(f"  [red]FLAG: {flag.get('reason', '')}[/red]")

        console.print()
        time.sleep(speed)

    # Final summary
    console.print("[bold]Final Results:[/bold]")
    console.print(f"  Social Welfare: {results.get('social_welfare', 0):.0f}")
    console.print(f"  Gini: {results.get('gini_coefficient', 0):.3f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize.py <results.json> [speed]")
        sys.exit(1)

    speed = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    replay(sys.argv[1], speed=speed)
