"""Dual-format journal: JSONL metrics + Markdown narrative.

Dual-format journal: JSONL plus Markdown for structured logging.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class DualFormatJournal:
    """Logs simulation events in both JSONL (machine) and Markdown (human) formats."""

    def __init__(self, output_dir: str = "results"):
        self.dir = Path(output_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.journal_path = self.dir / "journal.md"
        self.metrics_path = self.dir / "metrics.jsonl"

        # Initialize markdown
        self._append_md(f"# Nexus Simulation Journal\n*Started {_now_iso()}*\n")

    def log_simulation_start(self, config_summary: dict[str, Any]) -> None:
        self._log_jsonl({"event": "simulation_start", **config_summary})
        lines = [
            "## Simulation Configuration",
            "| Parameter | Value |",
            "|-----------|-------|",
        ]
        for k, v in config_summary.items():
            lines.append(f"| `{k}` | `{v}` |")
        self._append_md("\n".join(lines))

    def log_round(
        self,
        round_num: int,
        scores: dict[str, float],
        events: list[str],
        num_trades: int,
        oversight: Any = None,
    ) -> None:
        self._log_jsonl({
            "event": "round",
            "round": round_num,
            "scores": scores,
            "events": events,
            "num_trades": num_trades,
        })

        lines = [f"\n### Round {round_num}"]
        if events:
            for ev in events:
                lines.append(f"- Event: *{ev}*")
        lines.append(f"- Trades: **{num_trades}**")
        lines.append("- Scores: " + ", ".join(f"{k}: {v:.0f}" for k, v in scores.items()))

        if oversight:
            lines.append(f"- Oversight: {oversight}")

        self._append_md("\n".join(lines))

    def log_final_results(self, results: dict[str, Any]) -> None:
        self._log_jsonl({"event": "simulation_end", **_safe_serialize(results)})

        lines = [
            "\n## Final Results",
            f"- Rounds played: **{results.get('rounds_played', 0)}**",
            f"- Social welfare: **{results.get('social_welfare', 0):.0f}**",
            f"- Gini coefficient: **{results.get('gini_coefficient', 0):.3f}**",
            "",
            "### Agent Standings",
            "| Agent | Score | Jobs Done | Trades | Sharpe |",
            "|-------|-------|-----------|--------|--------|",
        ]
        agents = results.get("agents", {})
        for aid, info in sorted(agents.items(), key=lambda x: x[1].get("final_score", 0), reverse=True):
            lines.append(
                f"| {info.get('team_name', aid)} | {info.get('final_score', 0):.0f} "
                f"| {info.get('jobs_completed', 0)} | {info.get('trades', 0)} "
                f"| {info.get('sharpe_ratio', 0):.2f} |"
            )

        market = results.get("market", {})
        if market:
            lines.extend([
                "",
                "### Market Summary",
                f"- Total volume: **${market.get('total_volume', 0):.0f}**",
                f"- Total commission: **${market.get('total_commission', 0):.0f}**",
                f"- Number of trades: **{market.get('num_trades', 0)}**",
            ])

        self._append_md("\n".join(lines))

        # Also write final results as standalone JSON
        final_path = self.dir / "final_results.json"
        with open(final_path, "w") as f:
            json.dump(_safe_serialize(results), f, indent=2, default=str)

    def _log_jsonl(self, entry: dict[str, Any]) -> None:
        entry["_logged_at"] = _now_iso()
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def _append_md(self, text: str) -> None:
        with open(self.journal_path, "a") as f:
            f.write(text + "\n")


def _safe_serialize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_serialize(v) for v in obj]
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    return str(obj)
