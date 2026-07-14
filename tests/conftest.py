"""Shared pytest fixtures registering the ``--device`` option and ``device`` fixture."""

import contextlib

import pytest
import torch


def pytest_addoption(parser: pytest.Parser) -> None:
    # A sibling conftest may already have registered ``--device``; reuse it.
    with contextlib.suppress(ValueError):
        parser.addoption(
            "--device",
            action="store",
            default="cpu",
            help="Target torch device for compatibility tests, e.g. cpu, cuda, cuda:0, mps.",
        )


@pytest.fixture(scope="session")
def device(request: pytest.FixtureRequest) -> torch.device:
    device_name = str(request.config.getoption("device"))
    target = torch.device(device_name)

    try:
        torch.empty(0, device=target)
    except (RuntimeError, AssertionError) as exc:
        pytest.skip(f"Device {device_name} is not available: {exc}")

    return target
