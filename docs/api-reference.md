# API Reference — aumai-nanoagent

Complete documentation for all public classes, functions, and Pydantic models
in `aumai_nanoagent`.

---

## Module: `aumai_nanoagent.core`

Public exports: `NanoRuntime`, `DeviceProfiler`, `ModelOptimizer`

---

### `class NanoRuntime`

Lightweight agent runtime optimised for edge devices. Deliberately avoids importing
large ML frameworks at module load time. Processing is done inline to minimize memory
footprint.

```python
from aumai_nanoagent.core import NanoRuntime
```

#### `NanoRuntime.__init__(self) -> None`

Initializes the runtime in an unloaded state. No configuration is applied.
`_config` is `None`, `_loaded` is `False`, `_history` is an empty list.

---

#### `NanoRuntime.load(config: NanoAgentConfig) -> None`

Load the agent with the given configuration.

Validates that `config.spec.max_memory_mb <= config.device.memory_mb` before
accepting the configuration. If the check fails, raises `NanoRuntimeError` and
the runtime remains in an unloaded state. On success, clears the history.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `config` | `NanoAgentConfig` | Full runtime configuration. |

**Raises:**

- `NanoRuntimeError` — if `config.spec.max_memory_mb > config.device.memory_mb`.

**Example:**

```python
from aumai_nanoagent.core import NanoRuntime, NanoRuntimeError
from aumai_nanoagent.models import AgentSpec, EdgeDevice, NanoAgentConfig

device = EdgeDevice(device_id="d1", platform="linux-x86_64", memory_mb=4096, cpu_cores=4)
spec = AgentSpec(name="my-agent", max_memory_mb=256)
config = NanoAgentConfig(spec=spec, device=device)

runtime = NanoRuntime()
runtime.load(config)  # succeeds: 256 <= 4096
```

---

#### `NanoRuntime.process(message: AgentMessage) -> AgentMessage`

Process an input message and return a response.

Appends the incoming message to `_history`, applies rule-based response logic,
creates an `AgentMessage` with `role="assistant"` and a UTC timestamp, appends
the response to `_history`, and returns it.

**Rule logic:**

| Condition | Response |
|---|---|
| Content matches `hello`, `hi`, or `hey` (case-insensitive) | Personalized greeting from `spec.name` |
| Content matches `status`, `health`, or `ping` (case-insensitive) | Device diagnostic summary |
| Word count <= 5 | Prompt asking for more context |
| Word count > 5 | Echo of first 80 characters with word count |

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `message` | `AgentMessage` | The incoming user message. |

**Returns:** `AgentMessage` — assistant response with `role="assistant"`, populated `content`, and UTC `timestamp`.

**Raises:**

- `NanoRuntimeError` — if `load()` has not been called successfully.

**Example:**

```python
from aumai_nanoagent.models import AgentMessage

msg = AgentMessage(role="user", content="hello")
response = runtime.process(msg)
print(response.role)      # "assistant"
print(response.content)   # "Hello from my-agent! How can I assist?"
```

---

#### `NanoRuntime.unload(self) -> None`

Release resources and reset the runtime state.

Sets `_config = None`, `_loaded = False`, and clears `_history`. Safe to call
on an already-unloaded runtime.

**Example:**

```python
runtime.unload()
# runtime._loaded is now False
```

---

#### `NanoRuntime.history` (property)

```python
@property
def history(self) -> list[AgentMessage]:
```

Return a shallow copy of the conversation history.

Returns a new `list` — mutations to the returned list do not affect internal state.
Each entry is an `AgentMessage` with either `role="user"` (input) or
`role="assistant"` (output).

**Returns:** `list[AgentMessage]` — copy of all messages processed in the current session.

**Example:**

```python
history = runtime.history
print(len(history))  # Number of messages (user + assistant interleaved)
```

---

### `class NanoRuntimeError`

```python
class NanoRuntimeError(RuntimeError):
    ...
```

Raised by `NanoRuntime.load()` when memory requirements exceed device capacity,
and by `NanoRuntime.process()` when the runtime has not been loaded. Subclasses
`RuntimeError`.

---

### `class DeviceProfiler`

Detect and profile the current host device capabilities. Uses only Python stdlib
(`os`, `platform`, `ctypes`) — no external dependencies.

```python
from aumai_nanoagent.core import DeviceProfiler
```

---

#### `DeviceProfiler.profile(self) -> EdgeDevice`

Profile the current device.

Detects platform string, total RAM, CPU core count, and GPU presence. Returns
an `EdgeDevice` representing the current machine.

**Returns:** `EdgeDevice` with auto-detected `device_id`, `platform`, `memory_mb`,
`cpu_cores`, and `has_gpu`.

**Detection details:**

| Field | Detection method |
|---|---|
| `platform` | `f"{platform.system().lower()}-{platform.machine().lower()}"` |
| `device_id` | `f"{platform_str}-{cpu_cores}core"` |
| `memory_mb` | Linux: `/proc/meminfo`; Windows: `ctypes.windll.kernel32.GlobalMemoryStatusEx`; other: 512 |
| `cpu_cores` | `os.cpu_count() or 1` |
| `has_gpu` | `True` if `CUDA_VISIBLE_DEVICES` or `ROCR_VISIBLE_DEVICES` is set |

**Example:**

```python
profiler = DeviceProfiler()
device = profiler.profile()
print(device.platform)    # e.g. "linux-x86_64"
print(device.memory_mb)   # e.g. 32768
```

---

### `class ModelOptimizer`

Suggest quantization and optimization settings for edge deployment.

```python
from aumai_nanoagent.core import ModelOptimizer
```

---

#### `ModelOptimizer.quantize_config(model_size_mb: float, target_mb: float) -> dict[str, object]`

Suggest quantization configuration to fit a model within a target size.

Computes `ratio = target_mb / model_size_mb` and maps it to a quantization tier.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `model_size_mb` | `float` | Current (unquantized) model size in megabytes. |
| `target_mb` | `float` | Target model size in megabytes after quantization. |

**Returns:** `dict[str, object]` with the following keys:

| Key | Type | Description |
|---|---|---|
| `quantization` | `str` | One of `"none"`, `"int8"`, `"int4"`, `"int2"`. |
| `bits` | `int` | Bit width: 32, 8, 4, or 2. |
| `expected_size_mb` | `float` | Estimated post-quantization size (absent when `quantization="none"`). |
| `notes` | `str` | Human-readable recommendation. |
| `alternative` | `str` | Present only for `int2` — suggests a smaller architecture. |
| `source_size_mb` | `float` | Echo of `model_size_mb` input. |
| `target_size_mb` | `float` | Echo of `target_mb` input. |
| `compression_ratio` | `float` | `model_size_mb / target_mb`, rounded to 2 decimal places. |

**Quantization tiers:**

| Ratio (`target / model`) | Quantization | Approx. output size |
|---|---|---|
| >= 1.0 | `none` (FP32) | unchanged |
| >= 0.5 | `int8` | `model * 0.25` MB |
| >= 0.25 | `int4` | `model * 0.125` MB |
| < 0.25 | `int2` | `model * 0.0625` MB |

**Raises:**

- `ValueError` — if `model_size_mb <= 0` or `target_mb <= 0`.

**Example:**

```python
optimizer = ModelOptimizer()
config = optimizer.quantize_config(model_size_mb=7000.0, target_mb=500.0)
print(config["quantization"])      # "int8"
print(config["expected_size_mb"])  # 1750.0
print(config["compression_ratio"]) # 14.0
```

---

#### `ModelOptimizer.estimate_latency(model_size_mb: float, device: EdgeDevice) -> float`

Estimate inference latency in milliseconds using a heuristic model.

**Formula:**

```
base = model_size_mb * 2.0
core_factor = 1.0 / sqrt(max(cpu_cores, 1))
gpu_factor = 0.1 if has_gpu else 1.0
memory_factor = max(1.0, model_size_mb / (memory_mb * 0.5))
latency = base * core_factor * gpu_factor * memory_factor
```

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `model_size_mb` | `float` | Model size in megabytes. |
| `device` | `EdgeDevice` | Target device for the estimate. |

**Returns:** `float` — estimated latency in milliseconds, rounded to 2 decimal places.

**Example:**

```python
from aumai_nanoagent.models import EdgeDevice

device = EdgeDevice(device_id="d1", platform="linux-aarch64", memory_mb=4096, cpu_cores=4)
latency = optimizer.estimate_latency(model_size_mb=500.0, device=device)
print(f"{latency:.1f} ms")  # e.g. 500.0 ms
```

---

## Module: `aumai_nanoagent.models`

Public exports: `AgentSpec`, `EdgeDevice`, `NanoAgentConfig`, `AgentMessage`

All models use Pydantic v2 (`BaseModel`).

---

### `class AgentSpec`

Specification for a nano agent. Describes memory and model size requirements.

```python
from aumai_nanoagent.models import AgentSpec
```

**Fields:**

| Field | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `name` | `str` | — (required) | — | Human-readable agent name. |
| `version` | `str` | `"0.1.0"` | — | Semantic version string. |
| `capabilities` | `list[str]` | `[]` | — | Capability tags (e.g. `"echo"`, `"status-check"`). |
| `max_memory_mb` | `int` | `256` | `>= 1` | Maximum RAM the agent may occupy. |
| `max_model_size_mb` | `int` | `100` | `>= 1` | Maximum on-device model size in MB. |

**Example:**

```python
spec = AgentSpec(
    name="field-agent",
    version="1.0.0",
    capabilities=["echo", "status-check"],
    max_memory_mb=512,
    max_model_size_mb=128,
)
```

---

### `class EdgeDevice`

Describes the hardware capabilities of an edge device.

```python
from aumai_nanoagent.models import EdgeDevice
```

**Fields:**

| Field | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `device_id` | `str` | — (required) | — | Unique device identifier (e.g. `"rpi4-4core"`). |
| `platform` | `str` | — (required) | — | OS/arch string (e.g. `"linux-aarch64"`, `"windows-amd64"`). |
| `memory_mb` | `int` | — (required) | `>= 1` | Total device RAM in megabytes. |
| `cpu_cores` | `int` | — (required) | `>= 1` | Number of available CPU cores. |
| `has_gpu` | `bool` | `False` | — | Whether a GPU accelerator is available. |

**Example:**

```python
device = EdgeDevice(
    device_id="rpi4",
    platform="linux-aarch64",
    memory_mb=4096,
    cpu_cores=4,
    has_gpu=False,
)
```

---

### `class NanoAgentConfig`

Full runtime configuration for a nano agent. Composes `AgentSpec` and `EdgeDevice`.

```python
from aumai_nanoagent.models import NanoAgentConfig
```

**Fields:**

| Field | Type | Default | Description |
|---|---|---|---|
| `spec` | `AgentSpec` | — (required) | Agent capability and memory requirements. |
| `device` | `EdgeDevice` | — (required) | Target hardware description. |
| `model_path` | `str \| None` | `None` | Path to the on-device model file, if any. |

**YAML validation example:**

```python
import yaml
from aumai_nanoagent.models import NanoAgentConfig

raw = """
spec:
  name: my-agent
  max_memory_mb: 256
device:
  device_id: rpi4
  platform: linux-aarch64
  memory_mb: 4096
  cpu_cores: 4
"""
config = NanoAgentConfig.model_validate(yaml.safe_load(raw))
print(config.spec.name)   # my-agent
```

---

### `class AgentMessage`

A single message in a nano agent conversation.

```python
from aumai_nanoagent.models import AgentMessage
```

**Fields:**

| Field | Type | Default | Description |
|---|---|---|---|
| `role` | `str` | — (required) | Message author: `"user"`, `"assistant"`, or `"system"`. |
| `content` | `str` | — (required) | Message text content. |
| `timestamp` | `datetime` | `datetime.now(tz=timezone.utc)` | UTC timestamp of when the message was created. |

**Example:**

```python
from datetime import datetime, timezone
from aumai_nanoagent.models import AgentMessage

msg = AgentMessage(role="user", content="hello")
print(msg.role)       # "user"
print(msg.content)    # "hello"
print(type(msg.timestamp))  # <class 'datetime.datetime'>
```

---

## Module: `aumai_nanoagent.cli`

The CLI is accessed via the `nanoagent` command installed by the package.
All commands are built with [Click](https://click.palletsprojects.com/).

| Command | Description |
|---|---|
| `nanoagent run --config PATH` | Load an agent from YAML/JSON and start an interactive REPL. |
| `nanoagent profile` | Detect and print the current device's hardware capabilities. |
| `nanoagent optimize --model PATH --target-mb FLOAT` | Suggest quantization settings for a model. |

See the [README](../README.md) for full CLI usage examples with sample output.

---

## Package metadata

```python
import aumai_nanoagent
print(aumai_nanoagent.__version__)  # "0.1.0"
```
