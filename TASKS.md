# TASKS.md

## Setup
- [x] Read `reasoning_gym_env/` fully before writing any code
- [x] Scaffold project structure, create directories
- [ ] Set up `pyproject.toml` or `requirements.txt` with dependencies
- [x] Create `CHANGELOG.md` and `DECISIONS.md`

## Models
- [x] Define all Enums for action types, severities, labels
- [x] Define Pydantic models for Observation, Action, Reward
- [ ] Validate models are frozen, typed, and serialisable
- [ ] Update `CLAUDE.md` project layout with files created

## Data
- [x] Author `service_graph.yaml` — services, dependencies, teams, tiers
- [x] Author `alert_templates.yaml` — parameterised alert definitions
- [x] Author 8 incident scenario YAMLs — event sequences, noise, correct labels
- [ ] Verify scenarios are not trivially solvable by keyword matching

## Environment core
- [x] Implement episode generator — deterministic from seed
- [x] Implement `reset()` — clean state from seed + task id
- [x] Implement `step()` — validate action, update state, return obs/reward/done/info
- [x] Implement `state()` — expose full internal state including ground truth
- [x] Handle invalid actions gracefully — penalise, do not raise

## Reward
- [x] Implement per-step dense reward function
- [ ] Ensure reward varies meaningfully across trajectory
- [ ] Penalise idle steps and repeated invalid actions

## Tasks and graders
- [x] Implement Task 1 + grader (severity classification)
- [x] Implement Task 2 + grader (alert storm deduplication)
- [x] Implement Task 3 + grader (timeline labelling)
- [x] Verify all graders are deterministic, pure Python, return float in [0,1]
- [ ] Verify grader scores vary — not trivially 0 or 1 always

## Spec compliance
- [ ] Create `openenv.yaml` with all required metadata
- [ ] Run `openenv validate` — fix until it passes
- [ ] Confirm `reset()` / `step()` / `state()` signatures match spec exactly

## Inference script
- [ ] Implement `inference.py` in repo root
- [ ] Use OpenAI client, read all credentials from env vars
- [ ] Emit exact `[START]` `[STEP]` `[END]` stdout format
- [ ] Run all 3 tasks with fixed seeds, produce reproducible scores
- [ ] Confirm runtime under 20 min on 2 vCPU / 8 GB

## Containerisation
- [ ] Write `Dockerfile` — builds cleanly, runs `inference.py`
- [ ] Test `docker build && docker run` locally
- [ ] Confirm runs within memory and CPU limits

## Deployment
- [ ] Deploy to HuggingFace Space tagged `openenv`
- [ ] Confirm Space responds to `reset()` with HTTP 200
- [ ] Run pre-submission checklist from `HACKATHON.md`

## Documentation
- [ ] Write `README.md` — env description, obs/action spaces, tasks, setup, baseline scores
- [ ] Fill actual baseline scores into README after running inference
- [ ] Update `CLAUDE.md` project layout section to reflect final file structure
