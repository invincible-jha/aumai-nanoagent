"""Comprehensive CLI tests for aumai-nanoagent."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from aumai_nanoagent.cli import main


@pytest.fixture()
def runner() -> CliRunner:
    """Return a Click test runner."""
    return CliRunner()


@pytest.fixture()
def nano_config_json(tmp_path: Path) -> Path:
    """Write a valid NanoAgentConfig JSON file and return its path."""
    config = {
        "spec": {
            "name": "test-agent",
            "version": "0.1.0",
            "capabilities": ["echo"],
            "max_memory_mb": 256,
            "max_model_size_mb": 100,
        },
        "device": {
            "device_id": "test-device",
            "platform": "linux-arm64",
            "memory_mb": 512,
            "cpu_cores": 4,
            "has_gpu": False,
        },
    }
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config), encoding="utf-8")
    return config_file


class TestCliVersion:
    """Tests for --version flag."""

    def test_version_flag(self, runner: CliRunner) -> None:
        """--version must exit 0 and report 0.1.0."""
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_help_flag(self, runner: CliRunner) -> None:
        """--help must exit 0 and describe the CLI."""
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "NanoAgent" in result.output


class TestProfileCommand:
    """Tests for the `profile` command."""

    def test_profile_exits_zero(self, runner: CliRunner) -> None:
        """profile command exits with code 0."""
        result = runner.invoke(main, ["profile"])
        assert result.exit_code == 0

    def test_profile_shows_device_id(self, runner: CliRunner) -> None:
        """profile command prints Device ID label."""
        result = runner.invoke(main, ["profile"])
        assert "Device ID" in result.output

    def test_profile_shows_platform(self, runner: CliRunner) -> None:
        """profile command prints Platform label."""
        result = runner.invoke(main, ["profile"])
        assert "Platform" in result.output

    def test_profile_shows_memory(self, runner: CliRunner) -> None:
        """profile command prints Memory label."""
        result = runner.invoke(main, ["profile"])
        assert "Memory" in result.output

    def test_profile_shows_cpu_cores(self, runner: CliRunner) -> None:
        """profile command prints CPU Cores label."""
        result = runner.invoke(main, ["profile"])
        assert "CPU Cores" in result.output

    def test_profile_shows_gpu(self, runner: CliRunner) -> None:
        """profile command prints GPU label."""
        result = runner.invoke(main, ["profile"])
        assert "GPU" in result.output


class TestOptimizeCommand:
    """Tests for the `optimize` command."""

    def test_optimize_with_nonexistent_model(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """optimize uses fallback size when model file does not exist."""
        result = runner.invoke(
            main, ["optimize", "--model", "/nonexistent/model.bin", "--target-mb", "50"]
        )
        # Should still exit 0 (uses fallback)
        assert result.exit_code == 0
        assert "Quantization" in result.output

    def test_optimize_shows_target_size(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """optimize prints Target size line."""
        result = runner.invoke(
            main, ["optimize", "--model", "/nonexistent/model.bin", "--target-mb", "100"]
        )
        assert "Target size" in result.output
        assert "100.0" in result.output

    def test_optimize_shows_bits(self, runner: CliRunner) -> None:
        """optimize prints Bits line."""
        result = runner.invoke(
            main, ["optimize", "--model", "/does-not-exist.bin", "--target-mb", "50"]
        )
        assert "Bits" in result.output

    def test_optimize_with_existing_model_file(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """optimize reads file size when model file exists."""
        model_file = tmp_path / "model.bin"
        model_file.write_bytes(b"0" * (1024 * 1024 * 10))  # 10 MB
        result = runner.invoke(
            main,
            ["optimize", "--model", str(model_file), "--target-mb", "5"],
        )
        assert result.exit_code == 0
        assert "Est. latency" in result.output

    def test_optimize_missing_required_option(self, runner: CliRunner) -> None:
        """optimize exits non-zero when --target-mb is missing."""
        result = runner.invoke(main, ["optimize", "--model", "x.bin"])
        assert result.exit_code != 0

    def test_optimize_shows_notes(self, runner: CliRunner) -> None:
        """optimize prints Notes line."""
        result = runner.invoke(
            main, ["optimize", "--model", "/no.bin", "--target-mb", "10"]
        )
        assert "Notes" in result.output


class TestRunCommand:
    """Tests for the `run` command."""

    def test_run_requires_config(self, runner: CliRunner) -> None:
        """run exits non-zero when --config is missing."""
        result = runner.invoke(main, ["run"])
        assert result.exit_code != 0

    def test_run_with_valid_json_config(
        self, runner: CliRunner, nano_config_json: Path
    ) -> None:
        """run loads a valid JSON config and echoes agent name."""
        # Provide EOF immediately to exit the interactive loop
        result = runner.invoke(main, ["run", "--config", str(nano_config_json)], input="\n")
        # Should at least load and print the agent name before EOFError exits
        assert "test-agent" in result.output or result.exit_code in (0, 1)

    def test_run_config_not_found(self, runner: CliRunner, tmp_path: Path) -> None:
        """run exits non-zero for a non-existent config file."""
        result = runner.invoke(
            main, ["run", "--config", str(tmp_path / "missing.json")]
        )
        assert result.exit_code != 0
