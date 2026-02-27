"""aumai-nanoagent quickstart examples.

Demonstrates the core capabilities of the NanoAgent runtime:
  - Device profiling
  - Agent configuration and loading
  - Interactive message processing
  - Memory constraint enforcement
  - Model quantization recommendations

Run this file directly to verify your installation:

    python examples/quickstart.py
"""

from __future__ import annotations

from aumai_nanoagent.core import DeviceProfiler, ModelOptimizer, NanoRuntime, NanoRuntimeError
from aumai_nanoagent.models import AgentMessage, AgentSpec, EdgeDevice, NanoAgentConfig


# ---------------------------------------------------------------------------
# Demo 1: Profile the current device
# ---------------------------------------------------------------------------


def demo_device_profiling() -> EdgeDevice:
    """Profile the current host machine and print its capabilities.

    Returns the detected EdgeDevice for use in subsequent demos.
    """
    print("=" * 60)
    print("DEMO 1: Device Profiling")
    print("=" * 60)

    profiler = DeviceProfiler()
    device = profiler.profile()

    print(f"Device ID:   {device.device_id}")
    print(f"Platform:    {device.platform}")
    print(f"Memory:      {device.memory_mb} MB")
    print(f"CPU Cores:   {device.cpu_cores}")
    print(f"GPU:         {'Yes' if device.has_gpu else 'No'}")
    print()

    return device


# ---------------------------------------------------------------------------
# Demo 2: Load an agent and process messages
# ---------------------------------------------------------------------------


def demo_basic_runtime(device: EdgeDevice) -> None:
    """Load a NanoAgent and process a set of example messages.

    Shows how load(), process(), history, and unload() work together.
    """
    print("=" * 60)
    print("DEMO 2: Basic Runtime — Load, Process, Unload")
    print("=" * 60)

    spec = AgentSpec(
        name="quickstart-agent",
        version="0.1.0",
        capabilities=["echo", "status-check"],
        max_memory_mb=min(256, device.memory_mb),  # respect device limits
        max_model_size_mb=50,
    )
    config = NanoAgentConfig(spec=spec, device=device)

    runtime = NanoRuntime()
    runtime.load(config)
    print(f"Agent '{spec.name}' loaded on {device.platform}.")

    # Process a variety of message types
    test_messages = [
        "hello",
        "hi there",
        "status",
        "ping",
        "health check",
        "ok",
        "What is the current processing latency for this edge runtime?",
    ]

    for text in test_messages:
        msg = AgentMessage(role="user", content=text)
        response = runtime.process(msg)
        print(f"  User:  {text!r}")
        print(f"  Agent: {response.content!r}")
        print()

    # Inspect history
    history = runtime.history
    print(f"Total messages in history: {len(history)}")
    print(f"  ({len([m for m in history if m.role == 'user'])} user, "
          f"{len([m for m in history if m.role == 'assistant'])} assistant)")

    runtime.unload()
    print("Runtime unloaded.\n")


# ---------------------------------------------------------------------------
# Demo 3: Memory constraint enforcement
# ---------------------------------------------------------------------------


def demo_memory_constraint(device: EdgeDevice) -> None:
    """Show that NanoRuntime refuses to load when memory requirements exceed capacity.

    Demonstrates fail-fast, fail-clean behavior.
    """
    print("=" * 60)
    print("DEMO 3: Memory Constraint Enforcement")
    print("=" * 60)

    # Agent that fits comfortably
    small_spec = AgentSpec(name="small-agent", max_memory_mb=64)
    small_config = NanoAgentConfig(spec=small_spec, device=device)
    runtime = NanoRuntime()
    runtime.load(small_config)
    print(f"Small agent (64 MB) loaded on {device.memory_mb} MB device: OK")
    runtime.unload()

    # Agent that deliberately exceeds device memory
    oversized_device = EdgeDevice(
        device_id="tiny-mcu",
        platform="linux-arm",
        memory_mb=128,
        cpu_cores=1,
    )
    oversized_spec = AgentSpec(name="fat-agent", max_memory_mb=512)
    oversized_config = NanoAgentConfig(spec=oversized_spec, device=oversized_device)

    try:
        runtime.load(oversized_config)
        print("ERROR: Should have raised NanoRuntimeError!")
    except NanoRuntimeError as err:
        print(f"Correctly rejected: {err}")

    print()


# ---------------------------------------------------------------------------
# Demo 4: Model quantization advisor
# ---------------------------------------------------------------------------


def demo_quantization_advisor(device: EdgeDevice) -> None:
    """Evaluate several models against the current device's memory capacity.

    Shows quantize_config() and estimate_latency() for different model sizes.
    """
    print("=" * 60)
    print("DEMO 4: Model Quantization Advisor")
    print("=" * 60)

    optimizer = ModelOptimizer()
    target_mb = min(256.0, device.memory_mb / 4.0)  # use 25% of device RAM as target

    models = [
        ("TinyLLaMA-1.1B", 640.0),
        ("Phi-2 (2.7B)",   2700.0),
        ("LLaMA-7B",       7000.0),
        ("LLaMA-13B",      13000.0),
    ]

    print(f"Target memory: {target_mb:.0f} MB on {device.platform} "
          f"({device.cpu_cores} cores, {'GPU' if device.has_gpu else 'CPU only'})\n")
    print(f"{'Model':<20} {'Quant':<6} {'Est. size':<12} {'Est. latency':<14} Notes")
    print("-" * 80)

    for model_name, size_mb in models:
        quant = optimizer.quantize_config(size_mb, target_mb)
        quant_type = str(quant["quantization"])
        expected_size = quant.get("expected_size_mb", size_mb)
        latency = optimizer.estimate_latency(
            float(expected_size),  # type: ignore[arg-type]
            device,
        )
        notes = str(quant["notes"])[:45]
        print(f"{model_name:<20} {quant_type:<6} {float(expected_size):<12.1f} {latency:<14.1f} {notes}")

    print()


# ---------------------------------------------------------------------------
# Demo 5: Programmatic configuration from profiled device
# ---------------------------------------------------------------------------


def demo_auto_config() -> None:
    """Auto-configure an agent by profiling the current device at runtime.

    Shows the pattern of profile-then-configure for dynamic deployments.
    """
    print("=" * 60)
    print("DEMO 5: Auto-Configuration from Device Profile")
    print("=" * 60)

    profiler = DeviceProfiler()
    device = profiler.profile()

    # Allocate at most 25% of device RAM and 50 MB for the model
    max_mem = max(64, device.memory_mb // 4)
    max_model = min(50, max_mem // 2)

    spec = AgentSpec(
        name="auto-agent",
        version="0.1.0",
        capabilities=["auto-configured"],
        max_memory_mb=max_mem,
        max_model_size_mb=max_model,
    )
    config = NanoAgentConfig(spec=spec, device=device)

    print(f"Device:        {device.platform} / {device.memory_mb} MB / {device.cpu_cores} cores")
    print(f"Allocated:     {max_mem} MB RAM, {max_model} MB model")

    runtime = NanoRuntime()
    runtime.load(config)

    response = runtime.process(AgentMessage(role="user", content="status"))
    print(f"Status reply:  {response.content}")
    runtime.unload()
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all quickstart demos in sequence."""
    print("\naumai-nanoagent — Quickstart Demo\n")

    device = demo_device_profiling()
    demo_basic_runtime(device)
    demo_memory_constraint(device)
    demo_quantization_advisor(device)
    demo_auto_config()

    print("All demos completed successfully.")


if __name__ == "__main__":
    main()
