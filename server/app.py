"""FastAPI app for the OnCallEnv OpenEnv server."""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "openenv-core is required to run the server. "
        "Install dependencies with `pip install -e .`"
    ) from exc

from env.models import Action, Observation

from .oncall_environment import OnCallEnvironment


app = create_app(
    OnCallEnvironment,
    Action,
    Observation,
    env_name="oncall-env",
    max_concurrent_envs=2,
)


def main() -> None:
    import argparse
    import os
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
