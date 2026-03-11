# Nexus: Multi-Agent Compute Cluster Negotiation Environment

## Introduction

Enterprise organizations spend $5M--$50M+ annually on cloud compute, yet procurement decisions remain dominated by static reservation policies and manual spot-bidding rules. Industry analyses consistently show that 10--40% of cloud spend is wasted through over-provisioning, poorly timed purchases, and failure to exploit spot-market volatility [1]. Closing this gap requires procurement agents capable of strategic reasoning: anticipating competitor behavior, forming purchasing coalitions, and adapting to market disruptions in real time. Nexus provides the training ground for developing these capabilities.

Training LLM agents for complex multi-party interactions requires environments that are simultaneously rich enough to produce emergent strategy and structured enough to yield measurable reward signals. Compute cluster resource allocation, where teams compete for GPU time, memory, and bandwidth under budget constraints and shifting demand -- provides a natural testbed. Agents must reason about hidden information (opponents' private job queues), form and dissolve coalitions, negotiate prices in a double-auction market, and manage deadlines, all while being monitored by a supervisory agent tasked with detecting collusion and market manipulation.

Nexus is designed around three complementary tracks that map to open problems in LLM agent research:

1. **Multi-Agent Negotiation.** Agents interact through structured actions (bids, offers, coalition proposals, free-text messages) and must develop theory-of-mind [11] to predict opponents' behavior from partial observations.
2. **Fleet AI / Scalable Oversight.** A privileged supervisor agent observes all actions and must detect anomalous behavior (collusion, hoarding, free-riding) using interpretable machine learning probes, forming a closed feedback loop with the agents it monitors.
3. **Multi-Actor Management (Halluminate).** A CTO agent issues high-level directives to semi-autonomous worker agents that may misinterpret or partially execute instructions, requiring the CTO to learn each worker's reliability profile.
   
---

## 1. Concept

**Nexus** is a turn-based simulation of a shared compute cluster where multiple LLM agents each manage a team's workload. Agents must **negotiate**, **cooperate**, **compete**, and **form coalitions** to allocate scarce resources (GPU, CPU, memory, bandwidth) across competing job queues, all under **partial observability** and monitored by an **oversight agent**.

This scenario exemplifies compute-allocation negotiations with scalable oversight and multi-actor management.

### Why This Environment Matters for LLM Training

- **Theory-of-mind:** Agents must model opponents' hidden job queues and budgets to negotiate effectively.
- **Emergent strategy:** Coalition formation, bluffing, and reputation dynamics emerge naturally.
- **Scalable oversight:** The supervisor agent must detect collusion, resource hoarding, and inefficiency across N agents, a direct Fleet AI challenge.
- **Multi-actor management:** In Halluminate mode, a single "CTO agent" orchestrates multiple worker agents toward organizational goals.

---

## 2. Environment Design

### 2.1 World State

```
ClusterState:
  resources:
    gpu_units: 100          # refreshed each round
    cpu_units: 200
    memory_gb: 512
    bandwidth_gbps: 50
  round: int                # current tick (1..max_rounds)
  max_rounds: 50
  market:                   # public auction board
    active_bids: []
    completed_trades: []
  global_events: []         # random disruptions (outages, surges)
```

### 2.2 Agent State (Per Agent, Partially Observable)

Each agent sees:
- **Own state (full):** job queue, budget, reputation score, resource holdings
- **Others (partial):** only their public bids, reputation scores, and past trade history
- **Cluster (partial):** total remaining resources, but NOT individual allocations of others

```
AgentState:
  id: str
  team_name: str
  budget: float                  # virtual currency
  reputation: float              # 0-100, affects negotiation leverage
  resource_holdings:
    gpu, cpu, memory, bandwidth  # currently held
  job_queue: List[Job]           # private, not visible to others
  completed_jobs: List[Job]
  score: float                   # cumulative reward
```

### 2.3 Job Model

```
Job:
  id: str
  description: str              # natural language task description
  resource_requirements:
    gpu, cpu, memory, bandwidth  # minimum needed
  deadline: int                  # must complete by this round
  reward: float                  # points earned on completion
  priority: "low" | "medium" | "high" | "critical"
  collaborative: bool           # if True, can be split across agents
  penalty_on_miss: float        # score deducted if deadline missed
```

### 2.4 Action Space

Each round, agents submit a structured action (one of):

| Action | Description |
|--------|-------------|
| `allocate(job_id)` | Assign held resources to run a job |
| `bid(resource_type, quantity, price)` | Place a public bid to buy resources |
| `offer(resource_type, quantity, price)` | Offer to sell resources |
| `accept_bid(bid_id)` | Accept another agent's bid |
| `propose_coalition(agent_ids, job_id)` | Propose splitting a collaborative job |
| `accept_coalition(proposal_id)` | Join a proposed coalition |
| `reject_coalition(proposal_id)` | Decline a coalition proposal |
| `send_message(agent_id, text)` | Free-text negotiation message |
| `pass` | Do nothing this round |

Agents can take **up to 3 actions per round** (encouraging prioritization).

### 2.5 Round Flow

```
1. EVENT PHASE     → Random events (GPU outage, demand surge, bonus job drops)
2. OBSERVE PHASE   → Each agent receives their observation (own state + public info)
3. NEGOTIATE PHASE → Agents exchange messages, bids, and coalition proposals (2 sub-rounds)
4. ACTION PHASE    → Agents submit final actions (allocate, trade, etc.)
5. EXECUTE PHASE   → Engine resolves all actions, updates state
6. SCORE PHASE     → Completed jobs earn rewards, missed deadlines incur penalties
7. OVERSIGHT PHASE → Supervisor agent analyzes the round and flags anomalies
```

### 2.6 Observation Format (LLM-Friendly)

Observations are rendered as structured natural language:

```
=== ROUND 5 of 50 ===

CLUSTER STATUS:
  Available: 45 GPU | 120 CPU | 256 GB RAM | 30 Gbps BW
  Event: "GPU cluster sector B offline, 20 GPU units unavailable this round"

YOUR STATE (Team Alpha):
  Budget: $1,240 | Reputation: 78/100 | Score: 450
  Holdings: 12 GPU | 30 CPU | 64 GB RAM | 10 Gbps BW
  Jobs:
    [J-12] "Train recommendation model" needs 20 GPU, 40 CPU, 128 GB — deadline round 8 — reward $500 — COLLABORATIVE
    [J-15] "Run batch inference" needs 5 GPU, 10 CPU, 32 GB — deadline round 6 — reward $150

MARKET:
  BID #201 by Team Beta: wants 10 GPU @ $80 each
  OFFER #202 by Team Gamma: selling 20 CPU @ $15 each

MESSAGES:
  Team Beta → You: "Want to split J-12? I have 15 GPU available."

REPUTATION BOARD:
  Team Alpha: 78 | Team Beta: 85 | Team Gamma: 62 | Team Delta: 71
```

---

## 3. Oversight Agent (Fleet AI Track)

A dedicated **Supervisor Agent** runs in parallel, receiving a privileged view:

### 3.1 Supervisor Observations
- Full visibility into all agent actions, messages, and resource flows
- Aggregated statistics (Gini coefficient of resources, trade volume, coalition patterns)
- Historical behavior patterns per agent

### 3.2 Supervisor Actions

| Action | Description |
|--------|-------------|
| `flag(agent_id, reason)` | Flag suspicious behavior for review |
| `explain(agent_id, summary)` | Generate natural language explanation of an agent's strategy |
| `alert(description)` | Raise a system-wide alert about market manipulation |
| `recommend(action)` | Suggest policy changes (price floors, resource caps) |
| `report()` | Produce an end-of-round oversight summary |

### 3.3 What the Supervisor Learns to Detect
- **Collusion:** Two agents repeatedly trading at off-market prices
- **Resource hoarding:** Agent holding resources they can't use before deadline
- **Free-riding:** Agent in coalition not contributing fair share
- **Market manipulation:** Artificially inflating/deflating resource prices
- **Deadline gaming:** Deliberately missing low-penalty jobs to focus on high-reward ones at others' expense

---

## 4. Multi-Actor Mode (Halluminate Track)

In this variant, a single **CTO Agent** manages 3-4 **Worker Agents**:

```
CTO Agent (the LLM being trained)
  ├── Worker Alpha (semi-autonomous agent)
  ├── Worker Beta  (semi-autonomous agent)
  └── Worker Gamma (semi-autonomous agent)
```

- The CTO sees the full team state but must issue **high-level directives** (not direct actions)
- Workers have their own simple policies and may **misinterpret**, **delay**, or **partially execute** directives
- The CTO must discover each worker's strengths/biases and adapt its management style
- Other teams in the cluster are NPC agents with scripted strategies

### CTO Directive Examples:
```
direct(worker_id, instruction)   → "Alpha, bid for 10 GPUs under $90 each"
reassign(worker_id, job_id)      → Move a worker to a different job
set_priority(worker_id, policy)  → "Focus on high-reward jobs, ignore low-priority"
query(worker_id, question)       → "Beta, what's your current resource utilization?"
```

---

## 5. Reward Design

### Agent Rewards (per round)
```python
reward = (
    + job_completion_rewards          # primary: complete jobs
    - deadline_miss_penalties         # penalty: miss deadlines
    + 0.1 * reputation_delta         # small bonus for reputation growth
    - 0.05 * idle_resource_cost      # holding unused resources costs money
    + coalition_bonus                 # bonus for successful coalition jobs
)
```

### Supervisor Rewards
```python
reward = (
    + correct_flag_bonus             # correctly identified bad behavior
    - false_flag_penalty             # flagged an innocent agent
    + explanation_quality_score      # rated by held-out evaluator
    + anomaly_detection_recall       # caught X% of injected anomalies
)
```

### CTO Rewards (Multi-Actor Mode)
```python
reward = (
    + team_total_score               # overall team performance
    - directive_overhead             # penalty for excessive micromanagement
    + worker_utilization_bonus       # workers stayed busy and effective
    - miscommunication_penalty       # directives that were misunderstood
)
```

---

## 6. Technical Architecture

```
nexus/
├── PLAN.md                      # This file (in repo root)
├── pyproject.toml               # Package config
├── nexus/
│   ├── __init__.py
│   ├── state.py                 # Resource, Job, AgentState, ClusterState, MarketState
│   ├── config.py                # NexusConfig, presets (tiny/standard/oversight/multi_actor)
│   ├── actions.py               # ActionType enum, Action, parse_llm_output, validate
│   ├── observations.py          # render(state, agent_id) -> str, render_supervisor
│   ├── market.py                # ResourceMarket: bid/offer matching, impact, commission
│   ├── coalitions.py            # CoalitionManager: MoE voting on splits
│   ├── events.py                # EventGenerator: disruptions, job spawns
│   ├── rewards.py               # RewardComputer: weighted multi-signal + Sharpe ratio
│   ├── engine.py                # SimulationEngine: 7-phase round loop
│   ├── oversight.py             # SupervisorInterface, BehaviorProbe (CART ensemble)
│   ├── multi_actor.py           # CTOInterface, WorkerAgent, DirectiveParser
│   ├── journal.py               # DualFormatJournal: JSONL + Markdown
│   └── persistence.py           # SimulationState save/load
├── agents/
│   ├── base.py                  # BaseAgent ABC, ExperienceReplayBuffer, BehavioralPolicy
│   ├── random_agent.py          # Random baseline
│   ├── greedy_agent.py          # Greedy heuristic (priority-based job allocation)
│   ├── llm_agent.py             # Anthropic tool-use agent (Claude API)
│   ├── strategic_agent.py       # MCTS/UCB1 negotiation planning
│   ├── supervisor_agent.py      # Oversight: CART probes + collusion detection
│   └── cto_agent.py             # Multi-actor CTO (directive-based management)
├── scripts/
│   ├── run_simulation.py        # Typer CLI: run, evaluate
│   ├── evaluate.py              # Metric computation + comparison
│   └── visualize.py             # Rich terminal replay
└── tests/
    ├── test_engine.py
    ├── test_market.py
    ├── test_coalitions.py
    ├── test_observations.py
    └── test_oversight.py
```

### Dependencies
- Python 3.11+
- `pydantic` — state and action validation
- `gymnasium` — standard RL environment interface
- `anthropic` / `openai` — LLM agent backends (optional)
- `rich` — terminal visualization
- `pytest` — testing

---

## 7. Configuration Presets

| Preset | Agents | Rounds | Resources | Use Case |
|--------|--------|--------|-----------|----------|
| `tiny` | 2 | 10 | Low | Unit testing, debugging |
| `standard` | 4 | 50 | Medium | Normal training runs |
| `large` | 8 | 100 | High | Emergent strategy research |
| `oversight` | 4 + supervisor | 50 | Medium | Fleet AI track |
| `multi_actor` | 1 CTO + 3 workers vs NPCs | 50 | Medium | Halluminate track |

---

## 8. Training Approach

### For LLM Fine-Tuning
1. Run simulations with scripted/heuristic agents to generate trajectories
2. Extract (observation, action, reward) tuples
3. Fine-tune LLMs on high-reward trajectories (rejection sampling / DPO)
4. Self-play: pit fine-tuned agents against each other, repeat

### For In-Context Learning
1. Provide environment rules in system prompt
2. Feed observations as user messages, collect actions as assistant responses
3. Use reward signals to update prompting strategies
4. Evaluate theory-of-mind via held-out negotiation scenarios

---

## 9. Evaluation Metrics

| Metric | What It Measures |
|--------|-----------------|
| **Job Completion Rate** | % of jobs completed before deadline |
| **Negotiation Efficiency** | Average surplus captured in trades |
| **Coalition Success Rate** | % of proposed coalitions that succeed |
| **Theory-of-Mind Score** | Accuracy of predicting other agents' next actions |
| **Oversight Precision/Recall** | Supervisor's ability to detect anomalies |
| **Directive Effectiveness** | CTO directive → worker outcome alignment |
| **Social Welfare** | Total score across all agents (cooperation measure) |
| **Gini Coefficient** | Resource distribution fairness |

---

## 10. Implementation Roadmap

### Phase 1: Core Engine (MVP)
- [ ] State dataclasses (`state.py`)
- [ ] Action parsing and validation (`actions.py`)
- [ ] Round loop and execution engine (`engine.py`)
- [ ] Job generation and deadline tracking
- [ ] Basic observation rendering (`observations.py`)
- [ ] Reward computation (`rewards.py`)
- [ ] Random + greedy baseline agents

### Phase 2: Negotiation & Market
- [ ] Bid/offer matching engine (`market.py`)
- [ ] Free-text message passing between agents
- [ ] Coalition proposal/acceptance flow (`coalitions.py`)
- [ ] Random events (outages, surges) (`events.py`)

### Phase 3: LLM Integration
- [ ] LLM agent with structured action output (`llm_agent.py`)
- [ ] Observation → prompt template
- [ ] Action parsing from LLM output
- [ ] Simulation runner with LLM agents (`run_simulation.py`)

### Phase 4: Oversight (Fleet AI)
- [ ] Supervisor privileged observation view
- [ ] Anomaly injection (scripted collusion, hoarding scenarios)
- [ ] Supervisor action interface and reward
- [ ] Oversight evaluation metrics

### Phase 5: Multi-Actor (Halluminate)
- [ ] CTO directive interface
- [ ] Worker agent with noise/misinterpretation model
- [ ] NPC opponent teams
- [ ] CTO reward function

### Phase 6: Polish
- [ ] Gymnasium env wrapper
- [ ] Terminal visualization with `rich`
- [ ] Evaluation suite and leaderboard
- [ ] Documentation and examples

---

## 11. Differentiation

| Feature | Why It Stands Out |
|---------|------------------|
| **Natural language observations** | LLM-native — no tensor encoding needed |
| **Structured action space** | Parseable but expressive — no free-form action ambiguity |
| **Partial observability built-in** | Forces theory-of-mind, not just pattern matching |
| **Unified env for all modes** | One codebase covers multi-agent negotiation, oversight, and multi-actor management |
| **Scalable agent count** | 2 to 8+ agents with consistent mechanics |
| **Real-world analog** | Compute allocation is a genuine industry problem |

---

## 12. Novel Synthesis

These 5 combinations define the core technical contributions:

1. **MCTS for Negotiation Strategy:** MCTS tree search plans negotiation moves under hidden information, with UCB1 balancing exploring new tactics vs exploiting proven ones.

2. **MoE Coalition Voting:** Coalition members vote on resource splits using confidence-weighted per-option voting. Each member is an "expert"; the elimination pre-pass acts as a veto round.

3. **CART Behavioral Probes for Oversight:** Instead of probing residual streams for backdoor signals, the supervisor probes agent behavioral streams (trade patterns, messages) for collusion signals via CART ensemble with interpretable feature importance.

4. **Market Physics from Portfolio Simulation:** Commission, price impact, and portfolio accounting create realistic market dynamics -- large trades move prices, commissions discourage churn, Sharpe ratio provides risk-adjusted evaluation.

5. **Feedback-Driven Oversight Loop:** Supervisor detects anomaly -> issues structured feedback -> agent's BehavioralPolicy adjusts strategy weights -> behavior changes -> supervisor observes new behavior. Closed oversight loop.
