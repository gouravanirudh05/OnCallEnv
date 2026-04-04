# INSTRUCTIONS.md — Agent Instructions for OnCallEnv

## What this project is

OpenEnv-compliant RL environment simulating on-call SRE incident management.
An agent receives production alerts, logs, and service topology and makes
structured decisions: classify severity, deduplicate alert storms, label incident timelines.

**Hackathon:** OpenEnv (Meta + HuggingFace)
**Judging weights:** Real-world utility 30% · Task quality 25% · Env design 20% · Code 15% · Novelty 10%

---

## Before you write any code — read this first

`reasoning_gym_env/` is included in this repo as a reference implementation.
Read it before implementing anything. It shows exactly how a well-engineered
OpenEnv environment is structured: how models are typed, how step/reset/state
are implemented, how graders are separated from reward, and how the inference
script is wired. Mirror its patterns. Do not reinvent them.

---

## Project layout

> This section is a living document. Every time you create a file, add it here
> with a one-line description. Do not leave this stale.

```
oncall-env/
├── reasoning_gym_env/   # reference implementation — read before building
├── INSTRUCTIONS.md      # agent instructions and project layout
├── DECISIONS.md         # engineering choices and why (you create this)
├── CHANGELOG.md         # one line per meaningful change (you create this)
├── README.md            # user-facing docs (you create and maintain this)
├── openenv.yaml         # OpenEnv metadata (you create this)
├── server/Dockerfile     # must build and run cleanly (you create this)
├── inference.py         # baseline script, OpenAI API (you create this)
├── pyproject.toml        # dependency manifest
├── uv.lock               # dependency lockfile for uv
├── server/               # OpenEnv FastAPI server
├── server/__init__.py
├── server/app.py
├── server/oncall_environment.py
├── CONTEXT.md            # session handoff log (update every session)
├── env/                  # environment core modules
├── env/__init__.py         # env package exports
├── env/env.py              # core reset/step/state loop
├── env/generator.py        # deterministic episode generation
├── env/grader.py           # task scoring logic
├── env/models.py           # Pydantic action/observation models
├── env/reward.py           # dense reward shaping
├── env/types.py            # internal episode state types
├── data/                  # static YAML data
├── data/service_graph.yaml # service dependency graph
├── data/alert_templates.yaml # alert template definitions
├── data/scenarios/        # scenario templates (8 total)
├── data/scenarios/bad_deploy.yaml
├── data/scenarios/db_failover.yaml
├── data/scenarios/traffic_spike.yaml
├── data/scenarios/dependency_outage.yaml
├── data/scenarios/cert_expiry.yaml
├── data/scenarios/config_change.yaml
├── data/scenarios/memory_leak.yaml
├── data/scenarios/dns_misconfig.yaml
├── tasks/                 # task-specific helpers
├── inference.py           # baseline inference script
├── tests/                 # unit tests
├── tutorials/             # reference tutorials (not part of env)
├── tutorials/Hackathon_Instructions.md
├── tutorials/deployement_tutorial.md
├── tutorials/environment_tutorial.md
├── tutorials/scaling_tutorial.md
├── tutorials/training_tutorial.md
├── tests/conftest.py       # pytest config for local imports
├── tests/test_task1_generator.py
├── tests/test_task1_grader.py
├── tests/test_task2_generator.py
├── tests/test_task2_grader.py
├── tests/test_task3_generator.py
├── tests/test_task3_grader.py
│
│   -- fill in every file you create below this line --
│
```

---

## OpenEnv spec — non-negotiable

```python
class OnCallEnv:
    def reset(self, task_id: int, seed: int) -> Observation: ...
    def step(self, action: Action) -> tuple[Observation, float, bool, dict]: ...
    def state(self) -> dict: ...
```

- `step()` returns `(observation, reward, done, info)`
- `info` always has at minimum: `{"valid": bool, "error": str | None, "step": int}`
- `state()` exposes ground truth — used by graders, never by the agent
- All models are Pydantic. No plain dicts as public interfaces.
- Run `openenv validate` before considering any task done.

---

## The three tasks

### Task 1 — Severity Classification (easy)
5–8 alerts, some correlated, some false positives, some with misleading raw severity.
Agent classifies true severity per alert. Grader is per-alert, fully deterministic.
P0 miss = heavy penalty. Correct false positive silence = bonus.

### Task 2 — Alert Storm Deduplication (medium)
Root cause injected at one node of a service dependency graph. 25–40 downstream
alerts cascade. Agent labels each: ROOT / SYMPTOM / UNRELATED / FALSE_POSITIVE,
then escalates to the correct team. Grader: root cause ID (binary, 0.4 weight)
+ per-alert labels + escalation correctness.

### Task 3 — Incident Timeline Labelling (hard)
10–14 timestamped events shuffled + 3–5 noise events injected. Agent labels each:
ROOT_CAUSE / SYMPTOM / CONTRIBUTING_FACTOR / NOISE. INVESTIGATE action costs 1 step
and reveals 1 diagnostic. Grader: per-event labels + INVESTIGATE efficiency.

All graders return float in [0.0, 1.0]. All are pure Python. Zero external calls.

---

## Hard engineering rules

**Actions are always discrete.**
No `str` fields in Action models. Every field is an Enum or typed primitive.
Free-text actions require LLM graders. LLM graders break reproducibility.

**Graders never call anything external.**
Ground truth is generated at episode creation and stored in episode state.
Graders read from `env.state()`. They are pure functions. Test them in isolation.

**Reward and grading are separate concerns.**
Grader = final episode score [0,1] for evaluation.
Reward = per-step dense signal for learning. Keep them in separate modules.

**Seeds make everything reproducible.**
Same seed → identical episode. Hardcode seeds only in inference.py.
Baseline uses fixed seeds [42, 123, 7] for tasks [1, 2, 3].

**One responsibility per module.**
Env orchestrates. Generator builds episodes. Grader scores. Reward shapes.
Business logic does not live in env.py.

**No catch-all exception handling.**
Handle specific exceptions or let them propagate. No bare `except Exception`.
Invalid actions return low reward + error in info dict — they do not raise.

---

## Style & engineering guide

Follow the reference patterns in `reasoning_gym_env/`. Use the same level of
type-safety, modularity, and explicitness.

**Models and typing**
- All public interfaces are Pydantic models. No raw dicts for actions/observations.
- Actions are discrete only. Every action field is an Enum or typed primitive.
- Use explicit field names and types, avoid implicit unions or Any.

**Determinism and reproducibility**
- All episode generation is seed-driven and pure Python.
- Graders never call external services and never read from global mutable state.
- Same seed must produce identical observations and ground truth.

**Module boundaries**
- `models.py`: types only
- `generator.py`: episode creation and ground truth
- `env.py`: orchestrates reset/step/state, no business logic
- `grader.py`: final score only
- `reward.py`: dense per-step reward only

**Error handling**
- Invalid actions return small penalty + explanatory info dict; never raise.
- No bare `except` blocks. Handle specific errors or let them surface.

**Data authoring**
- YAML templates are hand-authored and controlled; avoid keyword-solvable patterns.
- Use consistent naming and required fields for all templates.
- Scenario templates must include root cause, labels, and noise annotations.

---

## Testing expectations

Follow `TASKS.md` as the execution checklist. At minimum:
- Unit tests for models, generator determinism, graders, and reward shaping.
- Graders must be deterministic and return scores in [0.0, 1.0].
- `openenv validate` must pass before any task is considered done.
 - `inference.py` must emit exact stdout format required in `tutorials/Hackathon_Instructions.md`.
- Baseline uses fixed seeds and runs under 20 minutes on 2 vCPU / 8 GB RAM.

---

## Context handoff protocol (compaction history)

To avoid re-explaining context in new sessions, maintain a running handoff file:

**File:** `CONTEXT.md` at repo root

**Update cadence:** After each meaningful work session or milestone.

**New session rule:** Read `CONTEXT.md` first, then `DECISIONS.md`, then `TASKS.md`.

**Template (keep short, bullet-based):**
```
# CONTEXT.md

## Goals
-

## Progress
-

## Open tasks
-

## Recent decisions
-

## Touched files
-

## Pending commands/tests
-

## Known issues/risks
-
```

---

## Models — update as you implement

Define all models before writing logic elsewhere. Minimum required types:

```python
# Update field names here as you define them
class Severity(str, Enum): ...       # P0 P1 P2 P3
class EventLabel(str, Enum): ...     # ROOT_CAUSE SYMPTOM CONTRIBUTING_FACTOR NOISE
class RemediationAction(str, Enum):  # ROLLBACK_DEPLOY RESTART_SERVICE etc.
class ActionType(str, Enum): ...     # CLASSIFY_ALERT LABEL_EVENT SILENCE etc.

class Alert(BaseModel): ...          # fill in fields when defined
class LogLine(BaseModel): ...
class ServiceNode(BaseModel): ...
class Observation(BaseModel): ...
class Action(BaseModel): ...
```

---

## Data to author (not generate procedurally)

**service_graph.yaml** — ~12 services. Each: name, depends_on[], team, tier.

**alert_templates.yaml** — ~20 templates parameterised by service, metric, thresholds.

**scenarios/** — 8 YAML files, one per incident type:
bad_deploy · db_failover · traffic_spike · dependency_outage ·
cert_expiry · config_change · memory_leak · dns_misconfig

Each scenario must define: root_cause_service, event_sequence[], noise_events[], correct_labels{}.
Hand-author these. Procedural generation produces patterns an LLM solves by keyword matching.

---

## Inference script (inference.py)

Exact stdout format required by OpenEnv:
```
[START] task=<n> env=oncall-env model=<model>
[STEP]  step=<n> action=<str> reward=<0.00> done=<bool> error=<msg|null>
[END]   success=<bool> steps=<n> score=<0.000> rewards=<r1,r2,...>
```

Reads: `OPENAI_API_KEY`, `API_BASE_URL`, `MODEL_NAME` from environment.
No retries on API failure — log and continue.
Mirror structure from `reasoning_gym_env/` inference script if one exists.

---

## Files you must create and maintain

**CHANGELOG.md** — one line per meaningful change.
Format: `[YYYY-MM-DD] <feat|fix|refactor|data|infra|docs>: <description>`

**DECISIONS.md** — every non-obvious engineering choice with its reasoning.
If you make a tradeoff, write it here. This is what gets defended to judges.

**README.md** — what the env does, observation space, action space, all three task
descriptions, setup instructions, baseline scores (fill after running inference.py).
Keep accurate to what is actually built — do not copy from this file verbatim.

**openenv.yaml** — name, version, tasks[], action_space, observation_space,
reward_type, reproducible: true, real_world_domain.

**pyproject.toml** — dependencies and server entry point.

**uv.lock** — lockfile for uv-based installs.

---

## Dockerfile

```dockerfile
FROM python:3.11-slim
# install deps, copy source, CMD runs inference.py
```

Must pass: `docker build -t oncall-env -f server/Dockerfile . && docker run oncall-env`
HF Space tag: `openenv`
