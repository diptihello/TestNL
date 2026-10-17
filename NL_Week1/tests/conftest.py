"""Pytest configuration for the zomato_reco test suite."""

from __future__ import annotations


def pytest_addoption(parser):
    """Register the --run-live option for live API tests."""
    parser.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="Run tests marked as 'live' (requires GROQ_API_KEY).",
    )


def pytest_collection_modifyitems(config, items):
    """Skip live tests unless --run-live is given."""
    if config.getoption("--run-live"):
        return
    skip_live = __import__("pytest").mark.skip(reason="need --run-live option to run")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
