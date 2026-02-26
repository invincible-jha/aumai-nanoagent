"""CLI entry point for aumai-nanoagent."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from aumai_nanoagent.core import DeviceProfiler, ModelOptimizer, NanoRuntime
from aumai_nanoagent.models import AgentMessage, AgentSpec, EdgeDevice, NanoAgentConfig

_runtime = NanoRuntime()
_profiler = DeviceProfiler()
_optimizer = ModelOptimizer()


@click.group()
@click.version_option()
def main() -> None:
    """AumAI NanoAgent — Lightweight edge AI runtime CLI."""


@main.command("run")
@click.option(
    "--config",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to agent YAML/JSON config file.",
)
def run(config: Path) -> None:
    """Load and run a nano agent in interactive mode."""
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        click.echo("PyYAML is required. Install with: pip install pyyaml", err=True)
        sys.exit(1)

    raw = config.read_text(encoding="utf-8")
    if config.suffix in {".yaml", ".yml"}:
        data: dict[str, object] = yaml.safe_load(raw)
    else:
        data = json.loads(raw)

    agent_config = NanoAgentConfig.model_validate(data)
    _runtime.load(agent_config)
    click.echo(f"Agent '{agent_config.spec.name}' loaded on {agent_config.device.platform}.")
    click.echo("Type 'quit' to exit.\n")

    while True:
        try:
            user_input = click.prompt("You")
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.strip().lower() in {"quit", "exit", "q"}:
            break
        msg = AgentMessage(role="user", content=user_input)
        response = _runtime.process(msg)
        click.echo(f"Agent: {response.content}")

    _runtime.unload()
    click.echo("Session ended.")


@main.command("profile")
def profile() -> None:
    """Profile the current device and print its capabilities."""
    device = _profiler.profile()
    click.echo(f"Device ID:   {device.device_id}")
    click.echo(f"Platform:    {device.platform}")
    click.echo(f"Memory:      {device.memory_mb} MB")
    click.echo(f"CPU Cores:   {device.cpu_cores}")
    click.echo(f"GPU:         {'Yes' if device.has_gpu else 'No'}")


@main.command("optimize")
@click.option("--model", "model_path", required=True, help="Path to model file (used for size).")
@click.option("--target-mb", required=True, type=float, help="Target model size in MB.")
def optimize(model_path: str, target_mb: float) -> None:
    """Suggest quantization settings for a model to fit a target size."""
    path = Path(model_path)
    if path.exists():
        model_size_mb = path.stat().st_size / (1024 * 1024)
    else:
        click.echo(f"Warning: model file '{model_path}' not found, using target as current size.", err=True)
        model_size_mb = target_mb * 2  # fallback estimate

    device = _profiler.profile()
    quant_config = _optimizer.quantize_config(model_size_mb, target_mb)
    latency = _optimizer.estimate_latency(model_size_mb, device)

    click.echo(f"Model size:      {model_size_mb:.1f} MB")
    click.echo(f"Target size:     {target_mb:.1f} MB")
    click.echo(f"Quantization:    {quant_config['quantization']}")
    click.echo(f"Bits:            {quant_config['bits']}")
    click.echo(f"Est. size after: {quant_config.get('expected_size_mb', model_size_mb):.1f} MB")
    click.echo(f"Est. latency:    {latency:.1f} ms")
    click.echo(f"Notes:           {quant_config.get('notes', '')}")


if __name__ == "__main__":
    main()
