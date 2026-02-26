"""Core logic for aumai-nanoagent."""

from __future__ import annotations

import os
import platform
import re
from datetime import datetime

from aumai_nanoagent.models import AgentMessage, EdgeDevice, NanoAgentConfig

__all__ = ["NanoRuntime", "DeviceProfiler", "ModelOptimizer"]


class NanoRuntimeError(RuntimeError):
    """Raised when the NanoRuntime encounters an error."""


class NanoRuntime:
    """Lightweight agent runtime optimised for edge devices.

    Deliberately avoids importing large ML frameworks at module load time.
    Processing is done inline to minimize memory footprint.
    """

    def __init__(self) -> None:
        self._config: NanoAgentConfig | None = None
        self._loaded: bool = False
        self._history: list[AgentMessage] = []

    def load(self, config: NanoAgentConfig) -> None:
        """Load the agent with the given configuration.

        Validates memory constraints against the device profile.

        Args:
            config: The NanoAgentConfig to load.

        Raises:
            NanoRuntimeError: If memory requirements exceed device capacity.
        """
        required_mb = config.spec.max_memory_mb
        available_mb = config.device.memory_mb
        if required_mb > available_mb:
            raise NanoRuntimeError(
                f"Agent requires {required_mb} MB but device only has {available_mb} MB."
            )
        self._config = config
        self._loaded = True
        self._history = []

    def process(self, message: AgentMessage) -> AgentMessage:
        """Process an input message and return a response.

        Provides deterministic echo/reflection responses without loading
        an external model, suitable for edge testing.

        Args:
            message: The incoming user message.

        Returns:
            An assistant AgentMessage with a response.

        Raises:
            NanoRuntimeError: If the runtime has not been loaded.
        """
        if not self._loaded or self._config is None:
            raise NanoRuntimeError("Runtime is not loaded. Call load() first.")

        self._history.append(message)

        content = message.content.strip()
        word_count = len(content.split())

        # Simple rule-based response for edge demonstration
        if re.search(r"\bhello\b|\bhi\b|\bhey\b", content, re.IGNORECASE):
            reply = f"Hello from {self._config.spec.name}! How can I assist?"
        elif re.search(r"\bstatus\b|\bhealth\b|\bping\b", content, re.IGNORECASE):
            device = self._config.device
            reply = (
                f"Runtime status: OK. "
                f"Device={device.platform}, "
                f"Memory={device.memory_mb}MB, "
                f"Cores={device.cpu_cores}."
            )
        elif word_count <= 5:
            reply = f"Received: '{content}'. Please provide more context."
        else:
            reply = f"Processed {word_count} words. Echo: {content[:80]}{'...' if len(content) > 80 else ''}"

        response = AgentMessage(
            role="assistant",
            content=reply,
            timestamp=datetime.utcnow(),
        )
        self._history.append(response)
        return response

    def unload(self) -> None:
        """Release resources and reset the runtime state."""
        self._config = None
        self._loaded = False
        self._history = []

    @property
    def history(self) -> list[AgentMessage]:
        """Return a copy of the conversation history."""
        return list(self._history)


class DeviceProfiler:
    """Detect and profile the current host device capabilities."""

    def profile(self) -> EdgeDevice:
        """Profile the current device.

        Uses stdlib `os` and `platform` — no external dependencies.

        Returns:
            An EdgeDevice with the detected hardware specifications.
        """
        system = platform.system().lower()
        machine = platform.machine().lower()
        platform_str = f"{system}-{machine}"

        # Memory detection
        memory_mb = self._detect_memory_mb()
        cpu_cores = os.cpu_count() or 1

        # GPU detection: simple heuristic — check for CUDA env vars
        has_gpu = (
            os.environ.get("CUDA_VISIBLE_DEVICES") is not None
            or os.environ.get("ROCR_VISIBLE_DEVICES") is not None
        )

        return EdgeDevice(
            device_id=f"{platform_str}-{cpu_cores}core",
            platform=platform_str,
            memory_mb=memory_mb,
            cpu_cores=cpu_cores,
            has_gpu=has_gpu,
        )

    def _detect_memory_mb(self) -> int:
        """Attempt to read total system memory. Returns 512 MB if unavailable."""
        try:
            if platform.system() == "Linux":
                with open("/proc/meminfo", encoding="utf-8") as fh:
                    for line in fh:
                        if line.startswith("MemTotal:"):
                            kb = int(line.split()[1])
                            return kb // 1024
            if platform.system() == "Windows":
                import ctypes

                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]

                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(stat)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))  # type: ignore[attr-defined]
                return int(stat.ullTotalPhys // (1024 * 1024))
        except Exception:
            pass
        return 512  # Safe default


class ModelOptimizer:
    """Suggest quantization and optimization settings for edge deployment."""

    def quantize_config(self, model_size_mb: float, target_mb: float) -> dict[str, object]:
        """Suggest quantization configuration to fit model within target size.

        Args:
            model_size_mb: Current model size in MB.
            target_mb: Target model size in MB.

        Returns:
            Dict with recommended quantization settings.

        Raises:
            ValueError: If target_mb <= 0 or model_size_mb <= 0.
        """
        if model_size_mb <= 0 or target_mb <= 0:
            raise ValueError("model_size_mb and target_mb must be positive.")

        ratio = target_mb / model_size_mb
        config: dict[str, object]

        if ratio >= 1.0:
            config = {"quantization": "none", "bits": 32, "notes": "Model already fits target."}
        elif ratio >= 0.5:
            config = {
                "quantization": "int8",
                "bits": 8,
                "expected_size_mb": model_size_mb * 0.25,
                "notes": "INT8 quantization recommended (4x compression).",
            }
        elif ratio >= 0.25:
            config = {
                "quantization": "int4",
                "bits": 4,
                "expected_size_mb": model_size_mb * 0.125,
                "notes": "INT4 quantization recommended (8x compression). Expect some accuracy loss.",
            }
        else:
            config = {
                "quantization": "int2",
                "bits": 2,
                "expected_size_mb": model_size_mb * 0.0625,
                "notes": "INT2 quantization or model pruning required. Significant accuracy degradation expected.",
                "alternative": "Consider a smaller base model architecture.",
            }

        config["source_size_mb"] = model_size_mb
        config["target_size_mb"] = target_mb
        config["compression_ratio"] = round(model_size_mb / max(target_mb, 0.001), 2)
        return config

    def estimate_latency(self, model_size_mb: float, device: EdgeDevice) -> float:
        """Estimate inference latency in milliseconds.

        Uses a simple heuristic: latency scales with model size and inversely
        with CPU cores. GPU provides a 10x speedup estimate.

        Args:
            model_size_mb: Model size in MB.
            device: The target EdgeDevice.

        Returns:
            Estimated latency in milliseconds per inference call.
        """
        # Base: ~1 ms per MB on a single CPU core at ~1 GHz equivalent
        base_latency_ms = model_size_mb * 2.0
        # Scale down with more cores
        core_factor = 1.0 / max(device.cpu_cores ** 0.5, 1.0)
        # GPU provides significant speedup
        gpu_factor = 0.1 if device.has_gpu else 1.0
        # Memory pressure factor
        memory_factor = max(1.0, model_size_mb / (device.memory_mb * 0.5))

        estimated = base_latency_ms * core_factor * gpu_factor * memory_factor
        return round(estimated, 2)
