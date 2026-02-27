"""Comprehensive tests for aumai-nanoagent core module."""

from __future__ import annotations

import os
import unittest.mock as mock

import pytest

from aumai_nanoagent.core import DeviceProfiler, ModelOptimizer, NanoRuntime, NanoRuntimeError
from aumai_nanoagent.models import AgentMessage, AgentSpec, EdgeDevice, NanoAgentConfig


# ---------------------------------------------------------------------------
# NanoRuntime tests
# ---------------------------------------------------------------------------


class TestNanoRuntime:
    """Tests for NanoRuntime."""

    def test_process_raises_when_not_loaded(self) -> None:
        """process() must raise NanoRuntimeError before load() is called."""
        runtime = NanoRuntime()
        msg = AgentMessage(role="user", content="Hello")
        with pytest.raises(NanoRuntimeError, match="not loaded"):
            runtime.process(msg)

    def test_load_succeeds_when_memory_fits(
        self, nano_config: NanoAgentConfig
    ) -> None:
        """load() succeeds when spec.max_memory_mb <= device.memory_mb."""
        runtime = NanoRuntime()
        runtime.load(nano_config)  # Should not raise

    def test_load_raises_when_memory_exceeded(self) -> None:
        """load() raises NanoRuntimeError when required memory exceeds device capacity."""
        runtime = NanoRuntime()
        spec = AgentSpec(name="heavy-agent", max_memory_mb=1024, max_model_size_mb=512)
        device = EdgeDevice(
            device_id="small-device",
            platform="linux-arm64",
            memory_mb=256,
            cpu_cores=1,
        )
        config = NanoAgentConfig(spec=spec, device=device)
        with pytest.raises(NanoRuntimeError, match="1024 MB"):
            runtime.load(config)

    def test_process_hello_response(self, nano_config: NanoAgentConfig) -> None:
        """process() returns greeting for 'hello' input."""
        runtime = NanoRuntime()
        runtime.load(nano_config)
        msg = AgentMessage(role="user", content="Hello")
        response = runtime.process(msg)
        assert response.role == "assistant"
        assert "Hello" in response.content
        assert "test-agent" in response.content

    def test_process_hi_response(self, nano_config: NanoAgentConfig) -> None:
        """process() handles 'hi' greeting."""
        runtime = NanoRuntime()
        runtime.load(nano_config)
        msg = AgentMessage(role="user", content="hi there")
        response = runtime.process(msg)
        assert "Hello" in response.content

    def test_process_status_response(self, nano_config: NanoAgentConfig) -> None:
        """process() returns device status for 'status' keyword."""
        runtime = NanoRuntime()
        runtime.load(nano_config)
        msg = AgentMessage(role="user", content="status check please")
        response = runtime.process(msg)
        assert "Runtime status: OK" in response.content
        assert "linux-arm64" in response.content
        assert "512MB" in response.content

    def test_process_health_keyword(self, nano_config: NanoAgentConfig) -> None:
        """process() returns status for 'health' keyword."""
        runtime = NanoRuntime()
        runtime.load(nano_config)
        msg = AgentMessage(role="user", content="health check")
        response = runtime.process(msg)
        assert "Runtime status: OK" in response.content

    def test_process_ping_keyword(self, nano_config: NanoAgentConfig) -> None:
        """process() returns status for 'ping' keyword."""
        runtime = NanoRuntime()
        runtime.load(nano_config)
        msg = AgentMessage(role="user", content="ping")
        response = runtime.process(msg)
        assert "Runtime status: OK" in response.content

    def test_process_short_message_requests_context(
        self, nano_config: NanoAgentConfig
    ) -> None:
        """process() requests more context for short messages (<=5 words)."""
        runtime = NanoRuntime()
        runtime.load(nano_config)
        msg = AgentMessage(role="user", content="what is this")
        response = runtime.process(msg)
        assert "Please provide more context" in response.content

    def test_process_long_message_echoes(self, nano_config: NanoAgentConfig) -> None:
        """process() echoes content back for messages with >5 words."""
        runtime = NanoRuntime()
        runtime.load(nano_config)
        content = "please tell me about the weather in London today right now"
        msg = AgentMessage(role="user", content=content)
        response = runtime.process(msg)
        assert "Processed" in response.content
        assert "words" in response.content

    def test_process_long_content_truncated(self, nano_config: NanoAgentConfig) -> None:
        """process() truncates very long content with ellipsis."""
        runtime = NanoRuntime()
        runtime.load(nano_config)
        long_content = "word " * 100  # 500 chars, >80
        msg = AgentMessage(role="user", content=long_content.strip())
        response = runtime.process(msg)
        assert "..." in response.content

    def test_history_accumulates(self, nano_config: NanoAgentConfig) -> None:
        """history returns all messages including assistant replies."""
        runtime = NanoRuntime()
        runtime.load(nano_config)
        runtime.process(AgentMessage(role="user", content="hello"))
        runtime.process(AgentMessage(role="user", content="status check now"))
        # 2 user + 2 assistant = 4
        assert len(runtime.history) == 4

    def test_history_is_copy(self, nano_config: NanoAgentConfig) -> None:
        """history property returns a copy, not the internal list."""
        runtime = NanoRuntime()
        runtime.load(nano_config)
        h = runtime.history
        h.append(AgentMessage(role="user", content="mutated"))
        assert len(runtime.history) == 0

    def test_unload_resets_state(self, nano_config: NanoAgentConfig) -> None:
        """unload() resets config, loaded flag, and history."""
        runtime = NanoRuntime()
        runtime.load(nano_config)
        runtime.process(AgentMessage(role="user", content="hello"))
        runtime.unload()
        assert len(runtime.history) == 0
        msg = AgentMessage(role="user", content="hello again")
        with pytest.raises(NanoRuntimeError):
            runtime.process(msg)

    def test_reload_after_unload(self, nano_config: NanoAgentConfig) -> None:
        """Runtime can be loaded again after unloading."""
        runtime = NanoRuntime()
        runtime.load(nano_config)
        runtime.unload()
        runtime.load(nano_config)
        response = runtime.process(AgentMessage(role="user", content="hello"))
        assert response.role == "assistant"

    def test_memory_boundary_exact_fit(self) -> None:
        """load() succeeds when required memory exactly equals device memory."""
        runtime = NanoRuntime()
        spec = AgentSpec(name="tight-agent", max_memory_mb=512)
        device = EdgeDevice(
            device_id="exact-fit", platform="linux-x86_64", memory_mb=512, cpu_cores=2
        )
        config = NanoAgentConfig(spec=spec, device=device)
        runtime.load(config)  # exact fit should succeed

    def test_response_has_timestamp(self, nano_config: NanoAgentConfig) -> None:
        """Each assistant response has a non-None timestamp."""
        runtime = NanoRuntime()
        runtime.load(nano_config)
        response = runtime.process(AgentMessage(role="user", content="hello"))
        assert response.timestamp is not None


# ---------------------------------------------------------------------------
# DeviceProfiler tests
# ---------------------------------------------------------------------------


class TestDeviceProfiler:
    """Tests for DeviceProfiler."""

    def test_profile_returns_edge_device(self) -> None:
        """profile() returns a valid EdgeDevice."""
        profiler = DeviceProfiler()
        device = profiler.profile()
        assert device.cpu_cores >= 1
        assert device.memory_mb >= 1
        assert device.device_id != ""
        assert device.platform != ""

    def test_profile_device_id_contains_core_count(self) -> None:
        """device_id includes core count in the format '...Ncore'."""
        profiler = DeviceProfiler()
        device = profiler.profile()
        assert "core" in device.device_id

    def test_gpu_detection_with_cuda_env(self) -> None:
        """DeviceProfiler detects GPU when CUDA_VISIBLE_DEVICES is set."""
        profiler = DeviceProfiler()
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
            device = profiler.profile()
        assert device.has_gpu is True

    def test_gpu_detection_with_rocr_env(self) -> None:
        """DeviceProfiler detects GPU when ROCR_VISIBLE_DEVICES is set."""
        profiler = DeviceProfiler()
        with mock.patch.dict(os.environ, {"ROCR_VISIBLE_DEVICES": "0"}):
            device = profiler.profile()
        assert device.has_gpu is True

    def test_no_gpu_without_env(self) -> None:
        """DeviceProfiler returns has_gpu=False without GPU env vars."""
        profiler = DeviceProfiler()
        env = {k: v for k, v in os.environ.items()
               if k not in {"CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"}}
        with mock.patch.dict(os.environ, env, clear=True):
            device = profiler.profile()
        assert device.has_gpu is False

    def test_detect_memory_mb_fallback(self) -> None:
        """_detect_memory_mb() returns 512 as safe default on failure."""
        profiler = DeviceProfiler()
        # Patch platform.system to return something unexpected to trigger fallback
        with mock.patch("aumai_nanoagent.core.platform.system", return_value="FreeBSD"):
            result = profiler._detect_memory_mb()
        assert result == 512


# ---------------------------------------------------------------------------
# ModelOptimizer tests
# ---------------------------------------------------------------------------


class TestModelOptimizer:
    """Tests for ModelOptimizer."""

    def test_quantize_no_quantization_needed(self, edge_device: EdgeDevice) -> None:
        """When model fits target, quantization='none'."""
        optimizer = ModelOptimizer()
        result = optimizer.quantize_config(model_size_mb=100.0, target_mb=200.0)
        assert result["quantization"] == "none"
        assert result["bits"] == 32

    def test_quantize_int8(self, edge_device: EdgeDevice) -> None:
        """When ratio >= 0.5, int8 quantization is recommended."""
        optimizer = ModelOptimizer()
        result = optimizer.quantize_config(model_size_mb=200.0, target_mb=120.0)
        assert result["quantization"] == "int8"
        assert result["bits"] == 8

    def test_quantize_int4(self) -> None:
        """When ratio >= 0.25 but < 0.5, int4 quantization is recommended."""
        optimizer = ModelOptimizer()
        result = optimizer.quantize_config(model_size_mb=400.0, target_mb=120.0)
        assert result["quantization"] == "int4"
        assert result["bits"] == 4

    def test_quantize_int2(self) -> None:
        """When ratio < 0.25, int2 quantization is recommended."""
        optimizer = ModelOptimizer()
        result = optimizer.quantize_config(model_size_mb=1000.0, target_mb=50.0)
        assert result["quantization"] == "int2"
        assert result["bits"] == 2

    def test_quantize_includes_source_and_target(self) -> None:
        """quantize_config result always includes source/target size and compression ratio."""
        optimizer = ModelOptimizer()
        result = optimizer.quantize_config(model_size_mb=100.0, target_mb=50.0)
        assert result["source_size_mb"] == 100.0
        assert result["target_size_mb"] == 50.0
        assert "compression_ratio" in result

    def test_quantize_raises_on_zero_model_size(self) -> None:
        """quantize_config raises ValueError for non-positive model_size_mb."""
        optimizer = ModelOptimizer()
        with pytest.raises(ValueError, match="positive"):
            optimizer.quantize_config(model_size_mb=0.0, target_mb=50.0)

    def test_quantize_raises_on_zero_target(self) -> None:
        """quantize_config raises ValueError for non-positive target_mb."""
        optimizer = ModelOptimizer()
        with pytest.raises(ValueError, match="positive"):
            optimizer.quantize_config(model_size_mb=100.0, target_mb=0.0)

    def test_quantize_raises_on_negative_values(self) -> None:
        """quantize_config raises ValueError for negative inputs."""
        optimizer = ModelOptimizer()
        with pytest.raises(ValueError):
            optimizer.quantize_config(model_size_mb=-10.0, target_mb=50.0)

    def test_estimate_latency_cpu_only(self, edge_device: EdgeDevice) -> None:
        """estimate_latency returns a positive float for CPU-only device."""
        optimizer = ModelOptimizer()
        latency = optimizer.estimate_latency(model_size_mb=100.0, device=edge_device)
        assert latency > 0.0

    def test_estimate_latency_gpu_faster(
        self, edge_device: EdgeDevice, gpu_device: EdgeDevice
    ) -> None:
        """GPU device should have lower latency than CPU-only device."""
        optimizer = ModelOptimizer()
        cpu_latency = optimizer.estimate_latency(model_size_mb=100.0, device=edge_device)
        gpu_latency = optimizer.estimate_latency(model_size_mb=100.0, device=gpu_device)
        assert gpu_latency < cpu_latency

    def test_estimate_latency_more_cores_faster(self) -> None:
        """More CPU cores should reduce latency."""
        optimizer = ModelOptimizer()
        single_core = EdgeDevice(
            device_id="single", platform="linux-x86_64", memory_mb=512, cpu_cores=1
        )
        quad_core = EdgeDevice(
            device_id="quad", platform="linux-x86_64", memory_mb=512, cpu_cores=4
        )
        latency_single = optimizer.estimate_latency(100.0, single_core)
        latency_quad = optimizer.estimate_latency(100.0, quad_core)
        assert latency_quad < latency_single

    def test_estimate_latency_memory_pressure(self) -> None:
        """Large model relative to device memory increases latency."""
        optimizer = ModelOptimizer()
        small_device = EdgeDevice(
            device_id="small", platform="linux-arm64", memory_mb=64, cpu_cores=1
        )
        latency_large_model = optimizer.estimate_latency(200.0, small_device)
        latency_small_model = optimizer.estimate_latency(10.0, small_device)
        assert latency_large_model > latency_small_model
