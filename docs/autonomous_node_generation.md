# Autonomous Node Generation Guide

This guide is for LLM agents and advanced users who want to generate MiniCortex/AxonForge nodes reliably with minimal trial and error.

Use this together with the [Node Authoring Guide](node_authoring_guide.md). The authoring guide explains the API. This document explains how to turn a node request into a correct implementation.

## Goal

A generated node should be:

- discoverable by the framework
- valid under the current descriptor API
- placed in the right location
- consistent with MiniCortex runtime behavior
- easy to verify manually

## Minimum Required Context

Before generating a node, the agent should know these things:

- what the node is supposed to do
- whether it is a built-in framework node or a project-local node
- expected inputs
- expected outputs
- user-editable settings
- whether it needs persistent state
- whether it needs slow initialization
- where it should appear in the palette

If any of those are missing, the agent can still proceed, but it must make explicit assumptions.

## Non-Negotiable Rules

These are the rules that matter most for correctness.

### 1. A node must subclass `Node`

```python
from axonforge.core.node import Node

class MyNode(Node):
    ...
```

### 2. A node must implement `process(self)`

`process()` is required. It is how the graph computes.

### 3. Outputs are assigned, not returned

Correct:

```python
self.output_data = result
```

Incorrect:

```python
return result
```

### 4. Use descriptors as class attributes

Descriptors such as `InputPort`, `OutputPort`, `Range`, `Text`, `Action`, and `State` belong on the class body, not inside `__init__()`.

### 5. Put the node in a discoverable location

Use one of these:

- built-in node: inside the framework's `axonforge/nodes/` tree
- project-local node: inside the project's `nodes/` tree

### 6. Use `State` for persisted internal values

If a value must survive graph save/load, it should be a `State`, not just a plain attribute.

### 7. Use `@background_init` only for slow setup

Do not use it by default. Use it when `init()` is slow enough to block the UI.

## Decision Workflow

A reliable agent should follow this sequence.

### Step 1. Classify the node

Choose one primary shape:

- stateless transform node
- stateful node
- source/input node
- display/inspection node
- action-driven utility node
- slow-loading dataset/resource node

### Step 2. Choose file location

If generating a built-in node:

- place it under `axonforge/nodes/<category>/...`

If generating a project-local node:

- place it under `<project>/nodes/...`

Discovery comes from the `nodes/` tree, so file placement is part of correctness.

### Step 3. Choose a palette branch

Prefer explicit branch decoration:

```python
from axonforge.core.descriptors import branch

@branch("Utilities/Custom")
class MyNode(Node):
    ...
```

If the branch is omitted, it is inferred from the file path. Explicit is safer for autonomous generation.

### Step 4. Define graph I/O

Ask:

- what comes in from other nodes?
- what goes out to other nodes?
- what types should be declared?

Then declare ports.

Example:

```python
input_data = InputPort("Input", np.ndarray)
output_data = OutputPort("Output", np.ndarray)
```

### Step 5. Define user-editable controls

Use fields only for values the user should edit in the node UI.

Typical mapping:

- bounded numeric setting -> `Range`
- integer setting -> `Integer`
- float setting -> `Float`
- toggle -> `Bool`
- finite mode selection -> `Enum`
- filesystem path -> `FilePath`, `DirectoryPath`, `FilePaths`

### Step 6. Define displays

Use displays only for visualization, not connectivity.

Typical mapping:

- scalar number -> `Numeric`
- vector/time/history -> `Plot`
- scalar array/map -> `Heatmap`
- image tensor -> `Image`
- status/debug text -> `Text`

### Step 7. Decide whether state is needed

Use `State` if the node needs to remember anything between runs or after save/load.

Examples:

- current index
- history buffer
- counter
- cached selected item

### Step 8. Decide whether `init()` is needed

Use `init()` when the node must allocate or load something once.

Examples:

- allocate a history array
- load a dataset
- prepare lookup tables

If the work is slow, add `@background_init`.

### Step 9. Write `process()` around current inputs and current state

Inside `process()`:

- read inputs from `self.<input_name>`
- read fields from `self.<field_name>`
- read state from `self.<state_name>`
- assign outputs to `self.<output_name>`
- assign displays to `self.<display_name>`
- update state as needed

### Step 10. Add a verification pass

Before considering the node complete, check:

- imports are correct
- descriptor names exist in this codebase
- `process()` exists
- outputs are assigned
- no unsupported API is assumed
- placement path is discoverable

## Canonical Templates

These templates are the safest starting points.

## Template 1: Stateless Transform Node

```python
import numpy as np

from axonforge.core.node import Node
from axonforge.core.descriptors import branch
from axonforge.core.descriptors.ports import InputPort, OutputPort
from axonforge.core.descriptors.fields import Range, Bool
from axonforge.core.descriptors.displays import Text


@branch("Utilities/Custom")
class Gain(Node):
    input_data = InputPort("Input", np.ndarray)
    output_data = OutputPort("Output", np.ndarray)

    gain = Range("Gain", 1.0, 0.0, 10.0)
    enabled = Bool("Enabled", True)

    info = Text("Info", default="idle")

    def process(self):
        if self.input_data is None:
            self.info = "No input"
            return

        if not self.enabled:
            self.output_data = self.input_data
            self.info = "Bypassed"
            return

        out = self.input_data * float(self.gain)
        self.output_data = out
        self.info = f"gain={self.gain}"
```

## Template 2: Stateful Node

```python
import numpy as np

from axonforge.core.node import Node
from axonforge.core.descriptors import branch
from axonforge.core.descriptors.ports import InputPort, OutputPort
from axonforge.core.descriptors.fields import Integer
from axonforge.core.descriptors.displays import Plot, Text
from axonforge.core.descriptors.state import State


@branch("Utilities/Custom")
class RunningHistory(Node):
    input_value = InputPort("Input", (int, float))
    output_history = OutputPort("History", np.ndarray)

    max_history = Integer("Max History", default=64)

    history = State("History", default=None)
    history_count = State("History Count", default=0)

    plot = Plot("Plot", style="line")
    info = Text("Info", default="idle")

    def init(self):
        max_len = int(self.max_history)
        if self.history is None:
            self.history = np.full(max_len, np.nan, dtype=np.float32)
        if self.history_count is None:
            self.history_count = 0

    def process(self):
        if self.input_value is None:
            return

        value = float(self.input_value)
        max_len = int(self.max_history)

        if self.history is None or len(self.history) != max_len:
            self.history = np.full(max_len, np.nan, dtype=np.float32)
            self.history_count = 0

        if self.history_count > 0:
            self.history[:-1] = self.history[1:]
        self.history[-1] = value
        self.history_count = min(int(self.history_count) + 1, max_len)

        self.output_history = self.history
        self.plot = self.history
        self.info = f"count={self.history_count}"
```

## Template 3: Slow-Loading Node

```python
from pathlib import Path
import numpy as np

from axonforge.core.node import Node, background_init
from axonforge.core.descriptors import branch
from axonforge.core.descriptors.ports import OutputPort
from axonforge.core.descriptors.fields import DirectoryPath
from axonforge.core.descriptors.displays import Text
from axonforge.core.descriptors.state import State


@branch("Input/Custom")
class DirectoryArraySource(Node):
    output_data = OutputPort("Output", np.ndarray)

    root_dir = DirectoryPath("Root Dir")

    files = State("Files", default=None)
    current_index = State("Current Index", default=0)

    info = Text("Info", default="idle")

    @background_init
    def init(self):
        root = self.root_dir
        if root is None:
            self.files = []
            return

        self.files = sorted(Path(root).glob("*.npy"))
        self.current_index = 0

    def process(self):
        if not self.files:
            self.info = "No files"
            return

        path = self.files[int(self.current_index) % len(self.files)]
        self.output_data = np.load(path)
        self.info = path.name
```

## Generation Checklist

An autonomous agent should not deliver a node until all items below are true.

- The class subclasses `Node`.
- `process(self)` exists.
- Every descriptor is declared on the class body.
- Every `OutputPort` written by the node is assigned in `process()`.
- Any persisted internal value uses `State`.
- Any user-editable value uses a `Field`.
- Any visual-only value uses a `Display`.
- The node has a valid file path under a discoverable `nodes/` tree.
- The node has an explicit `@branch(...)` unless inferred placement is intentional.
- The imports match the actual package structure in this repo.
- Slow setup is in `init()`, not repeated in `process()`.
- If `@background_init` is used, produced state is safe to serialize back to the live node.

## Manual Smoke Test

After generating a node, validate it with this minimal procedure:

1. Place the file in the correct `nodes/` directory.
2. Run the editor.
3. Confirm the node appears in the expected palette branch.
4. Create the node on the canvas.
5. Confirm `init()` does not fail.
6. Connect sample inputs if needed.
7. Change each field once.
8. Trigger each action once.
9. Run one step or start the network.
10. Confirm outputs/displays/state behave as expected.

## Terminal Validation Command

Agents should run the validator before handing off a generated node.

Built-in or repo-local file:

```bash
axonforge validate-node axonforge/nodes/input/load_text.py --json
```

Project-local file from the project root:

```bash
axonforge validate-node nodes/my_node.py --json
```

Project-local file from outside the project root:

```bash
axonforge validate-node /path/to/project/nodes/my_node.py --project-dir /path/to/project --json
```

Optional stricter check:

```bash
axonforge validate-node nodes/my_node.py --run-init --json
```

What the validator checks by default:

- the file exists and is under a discoverable `nodes/` directory
- the module imports successfully
- the file defines one or more `Node` subclasses
- each class can be instantiated
- required methods are present
- schema generation succeeds
- serialization and deserialization round-trip succeeds

What it does not check by default:

- semantic correctness of your algorithm
- whether `process()` produces the right math
- expensive `init()` behavior unless `--run-init` is passed
- GUI rendering quality

## Common Failure Modes

### Node does not appear in the palette

Likely causes:

- file is not under a discoverable `nodes/` directory
- import error inside the module
- class does not subclass `Node`
- class name is private-like or the module failed to import

### Node appears but fails on creation

Likely causes:

- `init()` raised
- invalid default field/state value
- import-time side effect or missing dependency

### Node appears but does nothing

Likely causes:

- `process()` exits early
- no inputs are connected and the node expects them
- output is returned instead of assigned
- display is set but no `OutputPort` is assigned

### State is lost after reload/save

Likely causes:

- value stored in plain instance attribute instead of `State`
- background init created state that is not serializable back to the live node

## Guidance For Prompting An LLM Agent

If you want an LLM to generate a node reliably, give it:

- a short functional description
- expected inputs and outputs
- desired palette branch
- whether it is project-local or built-in
- whether state is needed
- whether slow initialization is allowed
- one reference to the current authoring docs

Good prompt shape:

```text
Create a project-local MiniCortex node under nodes/utilities/ named RunningAverage.
Branch: Utilities/Statistics.
Input: a scalar float.
Output: current running average as scalar float.
Fields: window size integer.
State: rolling history must persist.
Displays: text info and a plot.
Use the current AxonForge descriptor API only.
Do not invent APIs.
```

Bad prompt shape:

```text
Make a cool node for stats.
```

## Delivery Standard For Autonomous Generation

A strong autonomous node generation result should include:

- the new node file
- a short summary of assumptions made
- the palette branch used
- what fields, actions, displays, and state were added
- a manual verification recipe

If any framework behavior is uncertain, the agent should prefer the current codebase over memory and avoid inventing extra abstractions.
