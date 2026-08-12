# Nexus: Multi-Agent Compute Cluster Negotiation

A multi-agent environment where AI agents negotiate for compute resources (GPUs, CPUs, memory, bandwidth) to complete jobs under deadline pressure.

## Features
- 7-phase round loop: EVENT -> OBSERVE -> NEGOTIATE -> ACTION -> EXECUTE -> SCORE -> OVERSIGHT
- Market trading with price impact and commission
- Coalition formation with MoE voting
- CART behavioral probes for oversight
- Compatible with OpenEnv 0.2.1

## Quick Start

```python
from nexus_openenv.client import NexusEnv
from nexus_openenv.models import NexusAction

with NexusEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    print(result.observation.observation_text)

    result = env.step(NexusAction(raw_text='{"type": "pass"}'))
    print(f"Reward: {result.reward}")
```

## Action Format

Send actions as JSON or natural language:
- `{"type": "allocate", "job_id": "J-001"}`
- `{"type": "bid", "resource_type": "gpu", "quantity": 5, "price": 100}`
- `{"type": "offer", "resource_type": "cpu", "quantity": 10, "price": 30}`
- `{"type": "pass"}`
- `[{"type": "bid", ...}, {"type": "allocate", ...}]` (multiple actions)

## Running Locally

```bash
uvicorn nexus_openenv.server.app:app --host 0.0.0.0 --port 8000
```
