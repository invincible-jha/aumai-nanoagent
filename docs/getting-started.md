# Getting Started with aumai-nanoagent

This guide walks you from a fresh Python environment to a running edge agent
in under ten minutes.

---

## Prerequisites

| Requirement | Minimum version | Notes |
|---|---|---|
| Python | 3.11 | Type hint features used throughout |
| pip | 23.0+ | `pip install aumai-nanoagent` |
| PyYAML | 6.0+ | Optional — required for YAML config files |
| OS | Linux, macOS, Windows | Memory detection works on all three |

No GPU is required. No cloud credentials are required. NanoAgent is designed to work
with whatever hardware is in front of you.

---

## Installation

### Basic install

```bash
pip install aumai-nanoagent
```

### With YAML support (recommended)

YAML config files are more readable than JSON for agent configuration. Install
PyYAML alongside the package:

```bash
pip install aumai-nanoagent pyyaml
```

### Development install (for contributors)

```bash
git clone https://github.com/aumai/aumai-nanoagent
cd aumai-nanoagent
pip install -e ".[dev]"
```

### Verify the install

```bash
nanoagent --version
# aumai-nanoagent, version 0.1.0

nanoagent --help
# Usage: nanoagent [OPTIONS] COMMAND [ARGS]...
#   AumAI NanoAgent — Lightweight edge AI runtime CLI.
# Commands:
#   optimize  Suggest quantization settings for a model to fit a target size.
#   profile   Profile the current device and print its capabilities.
#   run       Load and run a nano agent in interactive mode.
```

---

## Step-by-Step Tutorial

### Step 1: Profile your device

Before configuring an agent you need to know what hardware it will run on.
Run the profiler to detect your current machine's capabilities:

```bash
nanoagent profile
```

On a typical development laptop you will see output like:

```
Device ID:   linux-x86_64-8core
Platform:    linux-x86_64
Memory:      32768 MB
CPU Cores:   8
GPU:         No
```

Note the `Memory` value — you will use it to set device constraints in your
config file.

If you want GPU detection to show `Yes`, set the environment variable before
running:

```bash
CUDA_VISIBLE_DEVICES=0 nanoagent profile
# GPU:         Yes
```

---

### Step 2: Create your first agent config

Create a file called `my-agent.yaml`:

```yaml
spec:
  name: my-first-agent
  version: "0.1.0"
  capabilities:
    - echo
    - status-check
  max_memory_mb: 256
  max_model_size_mb: 50

device:
  device_id: dev-laptop
  platform: linux-x86_64
  memory_mb: 32768
  cpu_cores: 8
  has_gpu: false

model_path: null
```

The `spec.max_memory_mb` field (256) must be less than or equal to
`device.memory_mb` (32768). If it exceeds the device capacity, `NanoRuntime.load()`
will raise `NanoRuntimeError` and refuse to start.

---

### Step 3: Run the agent in interactive mode

```bash
nanoagent run --config my-agent.yaml
```

You should see:

```
Agent 'my-first-agent' loaded on linux-x86_64.
Type 'quit' to exit.

You: hello
Agent: Hello from my-first-agent! How can I assist?
You: status
Agent: Runtime status: OK. Device=linux-x86_64, Memory=32768MB, Cores=8.
You: what is the current temperature outside
Agent: Processed 7 words. Echo: what is the current temperature outside
You: quit
Session ended.
```

The rule-based responder handles three categories:

- **Greetings** (`hello`, `hi`, `hey`) → personalized greeting.
- **Status** (`status`, `health`, `ping`) → device summary.
- **Everything else** → echo with word count.

This is an intentional design: the runtime is a harness. You replace the rule
engine with your own inference backend.

---

### Step 4: Use the Python API directly

Create `run_agent.py`:

```python
from aumai_nanoagent.core import NanoRuntime
from aumai_nanoagent.models import AgentMessage, AgentSpec, EdgeDevice, NanoAgentConfig

device = EdgeDevice(
    device_id="dev-laptop",
    platform="linux-x86_64",
    memory_mb=32768,
    cpu_cores=8,
    has_gpu=False,
)
spec = AgentSpec(
    name="my-first-agent",
    max_memory_mb=256,
)
config = NanoAgentConfig(spec=spec, device=device)

runtime = NanoRuntime()
runtime.load(config)

messages = ["hello", "status", "what can you do?"]
for text in messages:
    msg = AgentMessage(role="user", content=text)
    response = runtime.process(msg)
    print(f"User: {text}")
    print(f"Agent: {response.content}")
    print()

runtime.unload()
```

```bash
python run_agent.py
```

---

### Step 5: Check model fit before deploying

Before deploying a model to a constrained device, use the optimizer to check
whether it fits and what quantization you need:

```bash
nanoagent optimize --model ./models/phi-2.bin --target-mb 256
```

If the model file does not exist yet, the command still works with an estimate:

```bash
nanoagent optimize --model future-model.bin --target-mb 128
# Warning: model file 'future-model.bin' not found, using target as current size.
# Model size:      256.0 MB
# Target size:     128.0 MB
# Quantization:    int8
# Bits:            8
```

---

## Common Patterns and Recipes

### Pattern 1: Auto-profile and build config programmatically

Instead of writing a YAML file by hand, profile the current machine and build the
config in code:

```python
from aumai_nanoagent.core import DeviceProfiler, NanoRuntime
from aumai_nanoagent.models import AgentSpec, NanoAgentConfig

profiler = DeviceProfiler()
device = profiler.profile()

spec = AgentSpec(
    name="auto-configured-agent",
    max_memory_mb=min(512, device.memory_mb // 4),  # Use at most 25% of RAM
)
config = NanoAgentConfig(spec=spec, device=device)

runtime = NanoRuntime()
runtime.load(config)
```

### Pattern 2: Batch processing messages

```python
from aumai_nanoagent.core import NanoRuntime
from aumai_nanoagent.models import AgentMessage, AgentSpec, EdgeDevice, NanoAgentConfig

device = EdgeDevice(
    device_id="worker-01",
    platform="linux-x86_64",
    memory_mb=8192,
    cpu_cores=4,
)
config = NanoAgentConfig(
    spec=AgentSpec(name="batch-worker"),
    device=device,
)

runtime = NanoRuntime()
runtime.load(config)

batch = [
    "hello",
    "status",
    "process this document text",
    "another message for the agent",
]

for text in batch:
    response = runtime.process(AgentMessage(role="user", content=text))
    print(f"[{response.timestamp.isoformat()}] {response.content}")

print(f"Total messages in history: {len(runtime.history)}")
runtime.unload()
```

### Pattern 3: Model optimization pipeline

```python
from aumai_nanoagent.core import DeviceProfiler, ModelOptimizer

profiler = DeviceProfiler()
optimizer = ModelOptimizer()
device = profiler.profile()

models_to_evaluate = [
    ("llama-7b", 7000.0),
    ("phi-2", 2700.0),
    ("tinyllama", 640.0),
]
target_mb = 256.0

print(f"Target: {target_mb} MB on {device.platform} with {device.memory_mb} MB RAM\n")

for name, size_mb in models_to_evaluate:
    quant = optimizer.quantize_config(size_mb, target_mb)
    latency = optimizer.estimate_latency(
        quant.get("expected_size_mb", size_mb), device  # type: ignore[arg-type]
    )
    print(f"{name:12s}  {quant['quantization']:6s}  {latency:8.1f} ms  {quant['notes']}")
```

### Pattern 4: Memory constraint guard in a service

```python
from aumai_nanoagent.core import NanoRuntime, NanoRuntimeError
from aumai_nanoagent.models import AgentSpec, EdgeDevice, NanoAgentConfig


def create_runtime_or_none(
    agent_name: str,
    required_mb: int,
    device: EdgeDevice,
) -> NanoRuntime | None:
    """Return a loaded NanoRuntime, or None if the device cannot support it."""
    spec = AgentSpec(name=agent_name, max_memory_mb=required_mb)
    config = NanoAgentConfig(spec=spec, device=device)
    runtime = NanoRuntime()
    try:
        runtime.load(config)
        return runtime
    except NanoRuntimeError as err:
        print(f"Cannot load '{agent_name}': {err}")
        return None
```

### Pattern 5: Inspect conversation history

```python
from aumai_nanoagent.core import NanoRuntime
from aumai_nanoagent.models import AgentMessage, AgentSpec, EdgeDevice, NanoAgentConfig

device = EdgeDevice(
    device_id="test", platform="linux-x86_64", memory_mb=4096, cpu_cores=2
)
config = NanoAgentConfig(spec=AgentSpec(name="inspector"), device=device)
runtime = NanoRuntime()
runtime.load(config)

runtime.process(AgentMessage(role="user", content="hello"))
runtime.process(AgentMessage(role="user", content="status"))

for msg in runtime.history:
    print(f"[{msg.role}] [{msg.timestamp.isoformat()}] {msg.content}")
```

---

## Troubleshooting FAQ

**Q: `NanoRuntimeError: Agent requires X MB but device only has Y MB.`**

The `spec.max_memory_mb` in your config exceeds the `device.memory_mb`.
Either reduce `max_memory_mb`, increase `device.memory_mb` to match the real
hardware, or run `nanoagent profile` to get the actual device memory.

---

**Q: `ModuleNotFoundError: No module named 'yaml'`**

Install PyYAML: `pip install pyyaml`. Alternatively, convert your config to
JSON (`.json` extension) and the CLI will parse it without PyYAML.

---

**Q: `NanoRuntimeError: Runtime is not loaded. Call load() first.`**

You called `runtime.process()` before `runtime.load(config)`. Always call
`load()` with a valid `NanoAgentConfig` before processing messages.

---

**Q: The profiler shows 512 MB but my machine has much more RAM.**

On macOS or non-Linux Unix systems, the `/proc/meminfo` path does not exist and
the Windows `ctypes` path is not triggered, so the fallback of 512 MB is used.
Set `device.memory_mb` manually in your config to reflect the actual hardware.

---

**Q: GPU shows `No` even though I have a CUDA GPU.**

NanoAgent detects GPU presence via the `CUDA_VISIBLE_DEVICES` environment variable.
Set it before running: `export CUDA_VISIBLE_DEVICES=0`. NanoAgent does not use
`nvidia-smi` or any native GPU library at runtime.

---

**Q: How do I hook in a real inference model?**

`NanoRuntime.process()` currently uses a rule-based responder. To attach a real
model, subclass `NanoRuntime` and override `process()`, or call `runtime.load()`
to manage session state and implement your own inference loop that calls the model
and returns `AgentMessage(role="assistant", content=model_output)`.

---

## Next Steps

- Read the [API Reference](api-reference.md) for full class and method documentation.
- Explore the [examples/quickstart.py](../examples/quickstart.py) for runnable demos.
- Read about [aumai-skillforge](../../aumai-skillforge/README.md) to add composable
  skills to your edge agent.
- Read about [aumai-toolsmith](../../aumai-toolsmith/README.md) to generate tools
  from natural language descriptions.
- Join the [AumAI Discord](https://discord.gg/aumai) for community support.
