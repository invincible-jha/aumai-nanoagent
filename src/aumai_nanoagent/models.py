"""Pydantic models for aumai-nanoagent."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

__all__ = [
    "AgentSpec",
    "EdgeDevice",
    "NanoAgentConfig",
    "AgentMessage",
]


class AgentSpec(BaseModel):
    """Specification for a nano agent."""

    name: str
    version: str = "0.1.0"
    capabilities: list[str] = Field(default_factory=list)
    max_memory_mb: int = Field(default=256, ge=1)
    max_model_size_mb: int = Field(default=100, ge=1)


class EdgeDevice(BaseModel):
    """Describes the hardware capabilities of an edge device."""

    device_id: str
    platform: str = Field(description="OS/platform, e.g. 'linux-arm64', 'rpi4'.")
    memory_mb: int = Field(ge=1)
    cpu_cores: int = Field(ge=1)
    has_gpu: bool = False


class NanoAgentConfig(BaseModel):
    """Full runtime configuration for a nano agent."""

    spec: AgentSpec
    device: EdgeDevice
    model_path: str | None = None


class AgentMessage(BaseModel):
    """A message in a nano agent conversation."""

    role: str = Field(description="'user' or 'assistant' or 'system'.")
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
