---

title: OnCallEnv
sdk: docker
app_port: 8000
tags:

* openenv
* rl
* sre
* incident-response

---

# OnCallEnv

OnCallEnv is an OpenEnv-compatible reinforcement learning environment for on-call incident response. It simulates a real operational workflow that human SREs and platform engineers actually perform: interpreting alerts, tracing cascades through a service graph, suppressing false positives, escalating to the right team, and reconstructing incident timelines from noisy evidence.

The environment ships with three deterministic tasks, typed Pydantic models, dense reward shaping, programmatic graders, a FastAPI/OpenEnv server wrapper, Docker deployment, and a baseline inference script.

## Why this is a real-world environment

This project is not a game or toy benchmark. It models realistic incident-management tasks:

* triaging alert severity under ambiguous or misleading monitoring metadata
* deduplicating alert storms caused by a single upstream failure
* labeling incident timelines from shuffled alerts, logs, deploys, pages, and diagnostics

Agents operate over structured production-like artifacts:

* alerts with thresholds, severities, timestamps, labels, and service ownership
* logs with service and level metadata
* a service dependency graph with owner teams and tiers
* scenario metadata and investigation opportunities

## Hackathon requirements checklist

* Real-world task simulation: yes, on-call incident response
* Full OpenEnv interface: yes, typed `Observation` and `Action`, plus `reset()`, `step()`, `state()`, and [`openenv.yaml`](/media/sathish/New%20Volume1/merged_partition_content/Meta-env-Hackathon/openenv.yaml)
* Minimum 3 tasks with agent graders: yes, tasks 1-3 with deterministic graders in [`env/grader.py`](/media/sathish/New%20Volume1/merged_partition_content/Meta-env-Hackathon/env/grader.py)
* Meaningful reward shaping: yes, dense per-step reward in [`env/reward.py`](/media/sathish/New%20Volume1/merged_partition_content/Meta-env-Hackathon/env/reward.py)
* Baseline inference script with fixed seeds: yes, [`inference.py`](/media/sathish/New%20Volume1/merged_partition_content/Meta-env-Hackathon/inference.py)
* Hugging Face Spaces deployment path: yes, repo metadata + Dockerfile in [`server/Dockerfile`](/media/sathish/New%20Volume1/merged_partition_content/Meta-env-Hackathon/server/Dockerfile)
* README with environment, spaces, setup, and usage: yes

Hackathon note: The baseline inference path uses the OpenAI API client with `OPENAI_API_KEY`, and [`inference.py`](/media/sathish/New%20Volume1/merged_partition_content/Meta-env-Hackathon/inference.py) is documented here in the required hackathon format.

## Environment overview

The core environment lives in [`env/env.py`](/media/sathish/New%20Volume1/merged_partition_content/Meta-env-Hackathon/env/env.py). Episodes are generated deterministically from seed-controlled YAML templates and scenario files in [`data/`](/media/sathish/New%20Volume1/merged_partition_content/Meta-env-Hackathon/data).

Public API:

```python
class OnCallEnv:
    def reset(self, task_id: int, seed: int) -> Observation: ...
    def step(self, action: Action) -> tuple[Observation, float, bool, dict]: ...
    def state(self) -> dict: ...
```

Step contract:

* `step(action)` returns `(observation, reward, done, info)`
* `info` includes `valid`, `error`, and `step`
* `state()` exposes hidden episode state and ground truth for graders, not agents

Default episode budgets:

* Task 1: 20
* Task 2: 60
* Task 3: 40

## Tasks

### Task 1: Severity Classification

Difficulty: easy

The agent receives 5-8 alerts and must classify each alert into `P0`, `P1`, `P2`, or `P3`.

Realistic complications:

* two alerts may be correlated and jointly represent a `P0`
* one alert is time-of-day adjusted and should be treated as lower severity
* one alert is a known false positive
* raw monitoring severity can be misleading

Generation and grading:

* generated in [`env/generator.py`](/media/sathish/New%20Volume1/merged_partition_content/Meta-env-Hackathon/env/generator.py)
* graded in [`env/grader.py`](/media/sathish/New%20Volume1/merged_partition_content/Meta-env-Hackathon/env/grader.py)

### Task 2: Alert Storm Deduplication

Difficulty: medium

The agent sees a root-cause alert, downstream cascade alerts, and unrelated alerts in a service dependency graph. It must:

1. label the root cause
2. label downstream symptoms
3. identify false positives and unrelated alerts
4. silence false positives
5. escalate to the correct owning team with severity `P0`

Completion logic now requires all of the above workflow stages:

* all event labels assigned
* all `false_positive` alerts silenced
* escalation recorded

Generation and grading:

* generated in [`env/generator.py`](/media/sathish/New%20Volume1/merged_partition_content/Meta-env-Hackathon/env/generator.py)
* graded in [`env/grader.py`](/media/sathish/New%20Volume1/merged_partition_content/Meta-env-Hackathon/env/grader.py)

### Task 3: Incident Timeline Labelling

Difficulty: hard

The agent reconstructs a shuffled incident timeline from scenario templates such as bad deploys, cert expiry, DNS misconfiguration, failover problems, traffic spikes, and dependency outages.

Event labels include:

* `root_cause`
* `symptom`
* `contributing_factor`
* `noise`
* `unrelated`
* `false_positive`

The agent can also use `investigate` actions tied to scenario diagnostics. Repeated investigation IDs are penalized.

Scenario templates live in [`data/scenarios/`](/media/sathish/New%20Volume1/merged_partition_content/Meta-env-Hackathon/data/scenarios).

## Observation space

Observation is a typed Pydantic model defined in [`env/models.py`](/media/sathish/New%20Volume1/merged_partition_content/Meta-env-Hackathon/env/models.py).

```python
class Observation(OpenEnvObservation):
    step: int
    alerts: list[Alert]
    logs: list[LogLine]
    service_graph: list[ServiceNode]
    active_incidents: list[str]
    budget_remaining: int
    context: dict[str, str]
    done: bool
    reward: float | None
```

Important sub-models:

* `Alert`: `id`, `service`, `metric`, `value`, `threshold`, `severity_raw`, `timestamp`, `labels`
* `LogLine`: `timestamp`, `service`, `level`, `message`, `trace_id`
* `ServiceNode`: `name`, `depends_on`, `team`, `tier`

## Action space

Action is a typed Pydantic model defined in [`env/models.py`](/media/sathish/New%20Volume1/merged_partition_content/Meta-env-Hackathon/env/models.py).

```python
class Action(OpenEnvAction):
    action_type: ActionType
    target_id: str
    severity: Severity | None
    event_label: EventLabel | None
    team: str | None
    remediation: RemediationAction | None
    investigation_id: str | None
```

Supported `action_type` values:

* `classify_alert`
* `label_event`
* `silence_alert`
* `escalate`
* `remediate`
* `investigate`
* `hold`

Validation behavior:

* invalid actions do not crash the environment
* invalid actions return a penalty and an `info["error"]` message
* required fields are enforced for `classify_alert`, `label_event`, `escalate`, and `remediate`

## Reward design

Dense reward shaping is implemented in [`env/reward.py`](/media/sathish/New%20Volume1/merged_partition_content/Meta-env-Hackathon/env/reward.py). Final task grading is separate and implemented in [`env/grader.py`](/media/sathish/New%20Volume1/merged_partition_content/Meta-env-Hackathon/env/grader.py).

Trajectory-level reward signals include:

* positive reward for correct severity classification
* penalties for important misclassifications such as missing a `P0`
* positive reward for correct event labels
* positive reward for silencing actual false positives
* penalty for silencing critical alerts
* positive reward for correct escalation team and severity
* positive reward for helpful one-time investigations
* penalty for repeated or unhelpful investigations
* penalty for invalid actions
* penalty for `hold`

Episode bonuses include:

* efficiency bonus for solving within 60% of the available budget
* task-2 bonus for silencing all false positives
* task-2 bonus for correctly identifying the root-cause alert

All final graders return values in `[0.0, 1.0]`.

## Determinism and reproducibility

Episode generation is deterministic for a fixed `(task_id, seed)` pair. Templates and scenarios are loaded from YAML, and all episode branching uses Python `random.Random(seed)`.

Default baseline seeds in [`inference.py`](/media/sathish/New%20Volume1/merged_partition_content/Meta-env-Hackathon/inference.py):

* Task 1: `42`
* Task 2: `123`
* Task 3: `7`

This makes task construction reproducible, even when model outputs vary across providers or timestamps.

## Baseline inference

The repository includes a baseline agent loop in [`inference.py`](/media/sathish/New%20Volume1/merged_partition_content/Meta-env-Hackathon/inference.py). It:

* runs the three tasks with fixed seeds
* prints `[START]`, `[STEP]`, `[INFO]`, and `[END]` lines
* parses raw JSON model output
* applies task-aware fallback behavior
* coerces repetitive low-value actions into the next useful valid action

Target hackathon usage:

```bash
export OPENAI_API_KEY=your_key_here
python3 inference.py
```

If you are using `uv`:

```bash
uv sync
export OPENAI_API_KEY=your_key_here
uv run python inference.py
```

Expected output shape:

```text
[START] task=severity_classification env=oncall-env model=...
[STEP]  step=1 action=... reward=... done=false error=null
[END]   success=true steps=... episode_return=... final_reward=... error=null rewards=...
```

Baseline note: The baseline is reproducible with respect to task seeds, but exact model outputs can still vary across API runs and model versions.

## Local setup

### Using uv

```bash
uv sync
```

### Using pip

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick usage example

```python
from env.env import OnCallEnv
from env.models import Action, ActionType, Severity

env = OnCallEnv()
obs = env.reset(task_id=1, seed=42)

action = Action(
    action_type=ActionType.CLASSIFY_ALERT,
    target_id=obs.alerts[0].id,
    severity=Severity.P2,
)

obs, reward, done, info = env.step(action)
print(reward, done, info)
```

## OpenEnv server

The FastAPI/OpenEnv wrapper lives in [`server/app.py`](/media/sathish/New%20Volume1/merged_partition_content/Meta-env-Hackathon/server/app.py) and [`server/oncall_environment.py`](/media/sathish/New%20Volume1/merged_partition_content/Meta-env-Hackathon/server/oncall_environment.py).

Run locally:

```bash
uv run uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Or via the project script:

```bash
uv run server --host 0.0.0.0 --port 8000
```

## OpenEnv validation

Validate the environment definition:

```bash
uv run openenv validate
```

Validate the running server:

```bash
uv run uvicorn server.app:app --host 127.0.0.1 --port 8000
uv run openenv validate --url http://127.0.0.1:8000
```

## Tests

Install dev dependencies first, then run:

```bash
uv sync --extra dev
uv run python -m pytest
```

The repo includes tests for:

* model schema and serialization
* generator determinism and data integrity
* grader behavior across the three tasks
* reward shaping
* environment lifecycle and action validation
* server wrapper behavior

## Docker and Hugging Face Spaces

The Docker image is defined in [`server/Dockerfile`](/media/sathish/New%20Volume1/merged_partition_content/Meta-env-Hackathon/server/Dockerfile).

Build locally:

```bash
docker build -t oncall-env -f server/Dockerfile .
```

Run locally:

```bash
docker run --rm -p 8000:8000 oncall-env
```

Hugging Face Spaces deployment:

1. Create a new Docker Space.
2. Push this repository as-is.
3. Ensure the repository root contains this README frontmatter and [`openenv.yaml`](/media/sathish/New%20Volume1/merged_partition_content/Meta-env-Hackathon/openenv.yaml).
4. Spaces will build the container from [`server/Dockerfile`](/media/sathish/New%20Volume1/merged_partition_content/Meta-env-Hackathon/server/Dockerfile).
5. The app serves on port `8000`.

## Repository structure

```text
.
├── env/
│   ├── env.py                # Core reset/step/state loop
│   ├── generator.py          # Deterministic episode generation
│   ├── grader.py             # Final task scoring
│   ├── models.py             # Typed Pydantic observation/action models
│   ├── reward.py             # Dense reward shaping
│   └── types.py              # Internal episode state
│
├── data/
│   ├── service_graph.yaml
│   ├── alert_templates.yaml
│   └── scenarios/
│       └── *.yaml
│
├── server/
│   ├── app.py                # FastAPI/OpenEnv app
│   ├── oncall_environment.py
│   └── Dockerfile
│
├── tests/
│   └── ...                   # Unit tests and lifecycle coverage
│
├── inference.py              # Baseline agent loop
└── openenv.yaml              # OpenEnv metadata
```