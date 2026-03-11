"""Plugin registry."""

from plugins.simulator.simulator import run_simulator

REGISTRY = {
    "simulator": run_simulator,
}


def list_all():
    """Return list of all registered plugin names."""
    return sorted(REGISTRY.keys())
