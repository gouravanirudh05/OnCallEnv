# OnCallEnv

OpenEnv-compliant environment that simulates on-call SRE incident response. Agents receive alerts, logs, and service topology and must classify severities, deduplicate alert storms, and label incident timelines.

## Quick start

```bash
pip install -e .
```

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
```

## Tasks

### Task 1 — Severity classification
Classify the true severity for 5–8 alerts, including correlated alerts, a time-of-day adjusted alert, and a known false positive.

### Task 2 — Alert storm deduplication
Identify the root cause alert within a dependency graph, label downstream symptoms, silence false positives, and escalate to the right team.

### Task 3 — Incident timeline labeling
Label shuffled events as root cause, symptom, contributing factor, or noise. INVESTIGATE actions can reveal diagnostics.

## Observation space

The observation contains:
- `alerts`: structured alerts with severity metadata
- `logs`: recent log lines
- `service_graph`: dependency graph of services
- `budget_remaining`: steps left in the episode
- `context`: scenario metadata

See `env/models.py` for the full schema.

## Action space

Actions are discrete and typed:
- `CLASSIFY_ALERT` with `severity`
- `LABEL_EVENT` with `event_label`
- `SILENCE_ALERT`
- `ESCALATE` with `team` and `severity`
- `REMEDIATE` with `remediation`
- `INVESTIGATE` with `investigation_id`
- `HOLD`

See `env/models.py` for the full schema.

## Baseline inference

Run the baseline script (uses OpenAI client):

```bash
OPENAI_API_KEY=... MODEL_NAME=... python inference.py
```

The script emits the required `[START]`, `[STEP]`, and `[END]` lines for each task.

## Development

Run tests:

```bash
pytest
```

## Server

Run the OpenEnv server locally:

```bash
uv run server
```

Or with uvicorn directly:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## Docker

```bash
docker build -t oncall-env -f server/Dockerfile .
docker run -p 8000:8000 oncall-env
```
