"""Smoke test for Gate 3 of docs/ACCEPTANCE.md: OpenEnv server round trip.

Starts the OpenEnv HTTP server as a subprocess, then verifies:
1. The server process starts without error with openenv-core 0.2.x installed.
2. GET /health returns a healthy status.
3. NexusEnv(base_url=...).reset() returns an observation with
   round_number == 0 and done == False.
4. step(NexusAction(raw_text="pass")) returns an observation with
   round_number == 1 and a float reward.

Usage:
    python scripts/openenv_smoke.py [--port 8000] [--timeout 60]

Exits 0 on success, 1 on failure.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def wait_for_health(base_url: str, deadline: float) -> dict:
    """Poll GET /health until it responds or the deadline passes."""
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=5) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, ConnectionError, OSError) as exc:
            last_error = exc
            time.sleep(0.5)
    raise TimeoutError(f"/health never became reachable: {last_error}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OpenEnv server smoke test (Gate 3).")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--timeout", type=float, default=60.0, help="Seconds to wait for server startup"
    )
    args = parser.parse_args(argv)

    base_url = f"http://127.0.0.1:{args.port}"

    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "nexus_openenv.server.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
        ],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        # Check 1 and 2: server starts and /health reports healthy.
        deadline = time.monotonic() + args.timeout
        health = wait_for_health(base_url, deadline)
        status = health.get("status")
        if status != "healthy":
            print(f"FAIL: /health returned {health!r}, expected status 'healthy'")
            return 1
        print(f"PASS: /health status is '{status}'")

        from nexus_openenv.client import NexusEnv
        from nexus_openenv.models import NexusAction

        with NexusEnv(base_url=base_url) as client:
            # Check 3: reset round trip.
            result = client.reset()
            obs = result.observation
            if obs.round_number != 0:
                print(f"FAIL: reset round_number is {obs.round_number}, expected 0")
                return 1
            if obs.done:
                print("FAIL: reset returned done == True, expected False")
                return 1
            print(
                f"PASS: reset returned round_number == 0, done == False "
                f"(observation length {len(obs.observation_text)} chars)"
            )

            # Check 4: one step round trip.
            result = client.step(NexusAction(raw_text="pass"))
            obs = result.observation
            if obs.round_number != 1:
                print(f"FAIL: step round_number is {obs.round_number}, expected 1")
                return 1
            if not isinstance(result.reward, float):
                print(f"FAIL: step reward is {type(result.reward).__name__}, expected float")
                return 1
            print(
                f"PASS: step returned round_number == 1, reward == {result.reward:.2f}"
            )

        print("Smoke test passed: server healthy, reset and step round trips OK.")
        return 0
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}")
        if server.poll() is not None and server.stdout is not None:
            print("--- server output ---")
            print(server.stdout.read())
        return 1
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=10)


if __name__ == "__main__":
    raise SystemExit(main())
