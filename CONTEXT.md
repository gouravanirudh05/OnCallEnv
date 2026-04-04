# CONTEXT.md

## Goals
- Build OnCallEnv per roadmap and OpenEnv spec

## Progress
- Scaffolded repo directories and added context handoff template
- Created DECISIONS.md and CHANGELOG.md
- Defined core Pydantic models and enums in env/models.py
- Removed legacy Decisions.txt
- Implemented initial env core modules (generator, env, reward, grader, types)
- Authored service graph and alert template YAMLs
- Implemented Task 1 generator logic and tests
- Added import fallback for openenv models during tests
- Implemented Task 1 grader and reward logic
- Implemented Task 2 generator and grader logic
- Added Task 2 generator/grader tests
- Authored Task 3 scenario templates
- Implemented Task 3 generator and grader logic
- Added Task 3 generator/grader tests
- Verified Task 2/3 generator and grader tests
- Updated gitignore to exclude reference materials
- Moved tutorials and hackathon instructions into tutorials/
- Tightened Task 3 grading normalization and reward shaping
- Added reward shaping tests
- Added model immutability and serialization tests
- Added determinism coverage for Task 1 generator
- Fixed generator updates for frozen Pydantic models
- Added OpenEnv server wrapper (server app, environment, Dockerfile)
- Added openenv.yaml, pyproject.toml, and uv.lock for OpenEnv validation
- Implemented baseline inference.py with required stdout format
- Expanded README with uv-based setup and usage
- Added server wrapper tests and validated runtime OpenEnv endpoints locally

## Open tasks
- Run baseline inference to capture scores for README
- Confirm docker build/run and resource limits
- Verify scenario templates are not trivially solvable
- Deployment checklist steps (HF Space, reset 200, etc.)

## Recent decisions
- Track decisions in DECISIONS.md and changes in CHANGELOG.md
- Store session handoff in CONTEXT.md at repo root
- Use uv for dependency management and execution
- Keep core OnCallEnv deterministic and expose via server wrapper only

## Touched files
- CONTEXT.md
- DECISIONS.md
- CHANGELOG.md
- INSTRUCTIONS.md
- env/models.py
- env/types.py
- env/generator.py
- env/env.py
- env/reward.py
- env/grader.py
- env/__init__.py
- data/service_graph.yaml
- data/alert_templates.yaml
- tests/test_task1_generator.py
- tests/conftest.py
- tests/test_task1_grader.py
- tests/test_task2_generator.py
- tests/test_task2_grader.py
- data/scenarios/bad_deploy.yaml
- data/scenarios/db_failover.yaml
- data/scenarios/traffic_spike.yaml
- data/scenarios/dependency_outage.yaml
- data/scenarios/cert_expiry.yaml
- data/scenarios/config_change.yaml
- data/scenarios/memory_leak.yaml
- data/scenarios/dns_misconfig.yaml
- tests/test_task3_generator.py
- tests/test_task3_grader.py
- .gitignore
- tutorials/Hackathon_Instructions.md
- tutorials/deployement_tutorial.md
- tutorials/environment_tutorial.md
- tutorials/scaling_tutorial.md
- tutorials/training_tutorial.md
- env/types.py
- env/grader.py
- env/reward.py
- env/env.py
- tests/test_reward_shaping.py
- env/models.py
- tests/test_models.py
- tests/test_task1_generator.py
- env/generator.py
- README.md
- INSTRUCTIONS.md
- TASKS.md
- openenv.yaml
- pyproject.toml
- uv.lock
- server/app.py
- server/oncall_environment.py
- server/Dockerfile
- tests/test_server_wrapper.py
- inference.py

## Pending commands/tests
- Run baseline inference with fixed seeds for README
- Run docker build/run checks

## Known issues/risks
- Initial generator/grader logic is placeholder and needs roadmap-aligned behavior
