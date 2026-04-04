# OnCallEnv

OnCallEnv is an OpenEnv environment for on call incident response. Agents see alerts, logs, and service topology. They must classify severity, reduce alert storms, and label incident timelines.

This repo follows the OpenEnv hackathon requirements. It includes three tasks, deterministic graders, a baseline inference script, and a server that passes OpenEnv validation.

## Quick start

```bash
uv sync
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

Task 1. Severity classification
Classify the true severity for a small set of alerts. The set includes correlated alerts, a time adjusted alert, and a known false positive.

Task 2. Alert storm deduplication
Identify the root cause alert, label downstream symptoms, silence false positives, and escalate to the right team.

Task 3. Incident timeline labeling
Label shuffled events as root cause, symptom, contributing factor, or noise. INVESTIGATE actions can reveal diagnostics.

## Observation space

Observations include alerts, logs, the service graph, remaining budget, and scenario context. See `env/models.py` for the full schema.

## Action space

Actions are discrete and typed. The main actions are CLASSIFY_ALERT, LABEL_EVENT, SILENCE_ALERT, ESCALATE, REMEDIATE, INVESTIGATE, and HOLD. See `env/models.py` for the full schema.

## Baseline inference

```bash
OPENAI_API_KEY=... MODEL_NAME=... uv run python inference.py
```

The script prints the required [START], [STEP], and [END] lines for each task.

## Tests

```bash
uv run pytest
```

## Server

```bash
uv run server
```

```bash
uv run uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## Docker

```bash
docker build -t oncall-env -f server/Dockerfile .
docker run -p 8000:8000 oncall-env
```

## OpenEnv validation

```bash
uv run openenv validate
```

```bash
uv run uvicorn server.app:app --host 127.0.0.1 --port 8000
```

```bash
uv run openenv validate --url http://127.0.0.1:8000
```
