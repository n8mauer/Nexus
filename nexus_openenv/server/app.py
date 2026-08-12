"""FastAPI application for the Nexus OpenEnv Environment.

Exposes the NexusEnvironment over HTTP and WebSocket endpoints,
compatible with EnvClient.

Usage:
    uvicorn nexus_openenv.server.app:app --reload --host 0.0.0.0 --port 8000
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with: pip install openenv>=0.2.1"
    ) from e

try:
    from models import NexusAction, NexusObservation
except ImportError:
    from nexus_openenv.models import NexusAction, NexusObservation
from .nexus_environment import NexusEnvironment


app = create_app(
    NexusEnvironment,
    NexusAction,
    NexusObservation,
    env_name="nexus",
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for direct execution."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
