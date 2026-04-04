# DECISIONS.md — Engineering Choices & Why

Every non-obvious decision lives here. Update this as you build.
Format: decision → why. Keep it short. This is what you defend to judges.

---

## Discrete-only action space
Free-text actions require LLM graders. LLM graders are non-deterministic and break
the OpenEnv reproducibility requirement. Every action field is an Enum or typed primitive.
Graders are pure Python functions with no external calls.

## Ground truth in episode state, not a separate oracle
Generator bakes correct labels into the episode object at creation time.
Grader reads from `env.state()`. This makes graders stateless and unit-testable
without running a full episode.

## Reward and grader are separate modules
Grader computes final episode score [0,1] — for evaluation.
Reward computes per-step dense signal — for learning.
These are tuned independently and should not be coupled.

## Hand-authored scenario templates
Procedurally generated event sequences produce patterns solvable by keyword matching.
8 hand-authored YAML scenarios give precise control over signal-to-noise ratio per episode.

## info dict always returned from step()
Agents and graders need to distinguish invalid actions from valid-but-low-reward actions.
`info["valid"]` carries this signal. Without it, an agent cannot tell if it did
something wrong or just something unhelpful.

## Server wrapper uses OpenEnv HTTP server
Added `server/` with a minimal wrapper to expose `OnCallEnv` via OpenEnv's FastAPI server.
The core env remains local-only and deterministic; the wrapper only adapts reset/step/state
to the OpenEnv HTTP schema to satisfy validation and deployment requirements.

## Use uv for dependency management
Standardize on uv (`uv sync`, `uv run`) for installs and execution to keep environments
reproducible across machines without local pip state divergence.

---

<!-- Add new decisions below as you make them -->
