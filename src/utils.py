import sys

import requests


def detect_model(endpoint: str) -> str:
    """Auto-detect the served model from the /v1/models endpoint."""
    r = requests.get(f"{endpoint.rstrip('/')}/v1/models", timeout=10)
    r.raise_for_status()
    models = [m.get("id") for m in r.json().get("data", []) if m.get("id")]
    if not models:
        print("Error: No models found at endpoint.", file=sys.stderr)
        sys.exit(1)
    if len(models) > 1:
        print(
            f"Multiple models found: {models}. Using first: {models[0]}",
            file=sys.stderr,
        )
    return models[0]


def assert_server_up(endpoint: str, timeout_s: float = 5.0):
    r = requests.get(f"{endpoint.rstrip('/')}/health", timeout=timeout_s)
    r.raise_for_status()
