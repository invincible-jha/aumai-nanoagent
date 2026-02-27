"""Shared test fixtures for aumai-nanoagent."""

from __future__ import annotations

import pytest

from aumai_nanoagent.models import AgentMessage, AgentSpec, EdgeDevice, NanoAgentConfig


@pytest.fixture()
def edge_device() -> EdgeDevice:
    """Return a standard edge device with 512 MB RAM and 4 cores."""
    return EdgeDevice(
        device_id="test-device-4core",
        platform="linux-arm64",
        memory_mb=512,
        cpu_cores=4,
        has_gpu=False,
    )


@pytest.fixture()
def gpu_device() -> EdgeDevice:
    """Return an edge device with GPU and 1024 MB RAM."""
    return EdgeDevice(
        device_id="gpu-device-8core",
        platform="linux-x86_64",
        memory_mb=1024,
        cpu_cores=8,
        has_gpu=True,
    )


@pytest.fixture()
def agent_spec() -> AgentSpec:
    """Return a standard AgentSpec with 256 MB memory limit."""
    return AgentSpec(
        name="test-agent",
        version="0.1.0",
        capabilities=["echo", "status"],
        max_memory_mb=256,
        max_model_size_mb=100,
    )


@pytest.fixture()
def nano_config(agent_spec: AgentSpec, edge_device: EdgeDevice) -> NanoAgentConfig:
    """Return a NanoAgentConfig combining the spec and device fixtures."""
    return NanoAgentConfig(spec=agent_spec, device=edge_device)


@pytest.fixture()
def user_message() -> AgentMessage:
    """Return a simple user message."""
    return AgentMessage(role="user", content="Hello there!")
