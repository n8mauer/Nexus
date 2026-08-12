"""Run the seeded Nexus demo from the repository root.

Thin wrapper around scripts/demo.py (the entry point named in
docs/ACCEPTANCE.md, Gate 2). Runs the tiny preset with one greedy and one
random agent under a fixed seed, prints the final scoreboard, and writes
the results journal.

Usage:
    python demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.demo import main

if __name__ == "__main__":
    raise SystemExit(main())
