# MiniCortex Node Authoring Guide

This document explains the node framework used in MiniCortex/AxonForge: what a node is, how node lifecycle works, and how to use decorators, fields, displays, actions, and state descriptors correctly.

For agent-oriented implementation workflow, templates, and validation steps, see the [Autonomous Node Generation Guide](autonomous_node_generation.md).

## Framework Overview

MiniCortex is built around **nodes** connected into a computation graph.

A node is a Python class that:

- inherits from `axonforge.core.node.Node`
- declares its UI and I/O using descriptors as class attributes
- implements `process(self)` to do work
- can optionally implement `init(self)` for one-time setup

Typical node ingredients:

- `InputPort`: receives data from upstream nodes
- `OutputPort`: emits data to downstream nodes
- `Field`: editable settings shown in the node body
- `Display`: visual or textual outputs shown in the node body
- `Action`: a clickable button
- `State`: internal values that are persisted with the graph

The framework discovers node classes from the `nodes/` tree, builds the palette, instantiates nodes on the canvas, and calls `process()` during graph propagation or live execution.

## What A Node Is

At authoring time, a node is just a normal Python class:

```python
import numpy as np

from axonforge.core.node import Node, background_init
from axonforge.core.descriptors import branch
from axonforge.core.descriptors.ports import InputPort, OutputPort
from axonforge.core.descriptors.fields import Range, Bool
from axonforge.core.descriptors.displays import Heatmap, Text
from axonforge.core.descriptors.actions import Action
from axonforge.core.descriptors.state import State


@branch("Demo/Examples")
class ExampleNode(Node):
    x = InputPort("X", np.ndarray)
    y = OutputPort("Y", np.ndarray)

    gain = Range("Gain", 1.0, 0.0, 5.0)
    enabled = Bool("Enabled", True)

    preview = Heatmap("Preview", colormap="grayscale")
    info = Text("Info", default="ready")

    reset = Action("Reset", lambda self, params=None: self._reset(params))

    counter = State("Counter", default=0)

    @background_init
    def init(self):
        self.counter = 0

    def process(self):
        if self.x is None or not self.enabled:
            return

        self.y = self.x * float(self.gain)
        self.preview = self.y
        self.counter += 1
        self.info = f"ticks={self.counter}"

    def _reset(self, params=None):
        self.counter = 0
```

Key idea: descriptors are declared on the class, but you use them at runtime like normal instance attributes:

- read input with `self.x`
- read field with `self.gain`
- write output with `self.y = ...`
- write display with `self.preview = ...`
- read/write state with `self.counter`

## Node Lifecycle

### 1. Discovery

Nodes are discovered from the `nodes/` directory.

- If you use `@branch("Some/Category")`, that path is used in the palette.
- If you do not use `@branch`, the framework infers a palette branch from the file's folder path.

Example:

- `nodes/input/dataset/mnist_digit.py` becomes `Input/Dataset`
- `@branch("Hypercolumn/Activation")` forces that exact branch

### 2. Instantiation

When the user creates a node:

- the framework instantiates your class
- assigns it a unique `node_id`
- stores node metadata like `node_type`, `name`, and `position`
- validates that `process()` exists

### 3. `init(self)`

`init()` is an optional one-time setup hook. It is called right after node creation.

Use `init()` for:

- allocating buffers
- loading data
- setting up caches
- initializing `State` values or runtime attributes

If `init()` raises, the node records a loading error and the graph does not propagate from it successfully.

### 4. `process(self)`

`process()` is the required compute method. It is called:

- during live network execution
- when a field changes and the graph re-propagates
- when an action runs and the graph re-propagates
- when connections change and the graph re-propagates
- after successful initialization in some propagation paths

Important behavior:

- `process()` takes no arguments
- `process()` is expected to mutate output/display/state attributes
- `process()` does not need to return anything
- output values are collected from your `OutputPort` attributes after `process()` finishes

In other words, this is correct:

```python
def process(self):
    self.output_data = self.input_data * 2
```

Returning a value from `process()` is not how outputs are emitted.

### 5. Persistence

When a graph is saved:

- all `Field` values are serialized
- all `State` values are serialized
- node name and position are serialized

What is not persisted as part of normal graph save/load:

- `Display` values
- `Display` config state
- arbitrary runtime attributes unless they are also stored as fields/states

## Decorators

### `@branch(path)`

Source: `axonforge.core.descriptors.branch`

Registers a node class into a palette branch.

```python
@branch("Input/Noise")
class MyNoiseNode(Node):
    ...
```

### Parameters

- `path: str`
  The palette category path. Slash-separated values create nested categories.

### What it changes

- sets `cls._node_branch`
- registers the class in the global node branch registry
- controls where the node appears in the palette

### Notes

- If omitted, branch is inferred from the file path under `nodes/`.
- The decorator only affects palette/category registration, not runtime behavior.

### `@background_init`

Source: `axonforge.core.node.background_init`

Marks `init()` so it runs asynchronously in a worker process instead of on the main application thread.

```python
class MyNode(Node):
    @background_init
    def init(self):
        ...
```

### Parameters

- no parameters

### What it changes

- sets a marker on the `init` function
- causes `NodeMeta` to mark the class as using background init
- makes the bridge schedule `init()` in a background worker
- keeps the node in a loading state until the worker finishes

### Important behavior

- This decorator is only meaningful on `init()`.
- While the node is loading, processing is skipped.
- After worker completion, the framework copies exported state back onto the live node.

### What gets copied back

The default export/import path copies instance attributes from `self.__dict__` if they are:

- not reserved runtime fields such as `node_id`, `name`, `_loading`, `_loading_error`
- not callables
- pickle-able

### Practical guidance

Use `@background_init` when `init()` does slow work such as:

- disk I/O
- dataset loading
- expensive preprocessing

Avoid relying on non-pickle-able objects created inside background `init()` unless you override the export/apply logic.

## `init()` And `process()` In Practice

Use `init()` for setup and `process()` for per-propagation work.

Good split:

- `init()`: load MNIST into memory, allocate history arrays, build lookup tables
- `process()`: consume current inputs/fields/state and produce outputs/displays

Bad split:

- doing heavy file loading inside every `process()` call
- relying on `process()` return values instead of output attributes

## Ports

Ports are part of node authoring even though they are not fields or displays.

### `InputPort(label, data_type=None, default=None)`

Receives data from upstream nodes.

### Parameters

- `label: str`
  UI name shown on the node.
- `data_type: Any = None`
  Expected type metadata. Can be `None`, a single type, a string, or a list/tuple/set of types.
- `default: Any = None`
  Default value when the attribute has never been set.

### What it changes

- creates a named input connector on the left side of the node
- exposes type metadata to the UI
- stores the incoming value on the instance

### Notes

- During graph execution, unconnected inputs are set to `None`.
- `data_type` is used for compatibility checks and display, not strict runtime coercion.

### `OutputPort(label, data_type=None, default=None)`

Emits data to downstream nodes.

### Parameters

- `label: str`
  UI name shown on the node.
- `data_type: Any = None`
  Output type metadata.
- `default: Any = None`
  Default value before any output has been assigned.

### What it changes

- creates a named output connector on the right side of the node
- exposes type metadata to the UI
- lets the network collect emitted values after `process()`

## Fields

Fields are user-editable controls rendered inside the node body.

All field descriptors inherit shared behavior from `Field`.

### Shared field behavior

- Field values are stored per node instance.
- Field values are persisted with the graph.
- Assignment goes through `normalize()` and `validate()`.
- If the value changes and `on_change` is set, the callback runs.

### Shared `on_change` behavior

`on_change` can be:

- `None`
- a method name as `str`
- a callable

Callback signatures:

- string method name: `def handler(self, new_value, old_value): ...`
- callable: `def handler(obj, new_value, old_value): ...`

The callback only runs when the normalized value is different from the previous one.

### `Range(label, default, min_val, max_val, step=0.01, scale="linear", on_change=None)`

Numeric slider/range control stored as `float`.

### Parameters

- `label: str`
  Field label in the UI.
- `default: float`
  Initial numeric value.
- `min_val: float`
  Minimum allowed value.
- `max_val: float`
  Maximum allowed value.
- `step: float = 0.01`
  Slider increment.
- `scale: str = "linear"`
  UI scale hint. The current UI uses this to distinguish linear vs. log-style control behavior.
- `on_change`
  Optional change callback.

### What it changes

- constrains the value to `[min_val, max_val]`
- exposes min/max/step/scale metadata to the UI

### `Integer(label, default=0, on_change=None)`

Integer entry field.

### Parameters

- `label: str`
  Field label.
- `default: int = 0`
  Initial value.
- `on_change`
  Optional change callback.

### What it changes

- coerces assigned values with `int(value)`
- rejects values that cannot be converted to integer

### `Float(label, default=0.0, on_change=None)`

Float entry field.

### Parameters

- `label: str`
  Field label.
- `default: float = 0.0`
  Initial value.
- `on_change`
  Optional change callback.

### What it changes

- coerces assigned values with `float(value)`
- rejects values that cannot be converted to float

### `Bool(label, default=False, on_change=None)`

Boolean checkbox/toggle field.

### Parameters

- `label: str`
  Field label.
- `default: bool = False`
  Initial value.
- `on_change`
  Optional change callback.

### What it changes

- stores the value as a boolean-like field value
- exposes a boolean control in the UI

Note: `Bool` does not override normalization itself, so pass actual boolean values from your own code.

### `Enum(label, options, default, on_change=None)`

String selection field.

### Parameters

- `label: str`
  Field label.
- `options: list[str]`
  Allowed values.
- `default: str`
  Initial selected option.
- `on_change`
  Optional change callback.

### What it changes

- converts assigned values to `str`
- validates that the value is one of `options`
- renders a finite-choice selector in the UI

### `DirectoryPath(label, default=None, on_change=None)`

Directory picker field stored at runtime as `Path | None`.

### Parameters

- `label: str`
  Field label.
- `default: str | Path | None = None`
  Initial directory.
- `on_change`
  Optional change callback.

### What it changes

- normalizes values into `Path` or `None`
- serializes project-local paths relative to the project root when possible
- resolves relative paths against the project root when loading
- rejects restored paths that do not exist or are not directories

### Restore behavior

If a saved directory no longer exists, loading raises a `FieldRestoreError` internally and the field falls back to `None`.

### `FilePath(label, default=None, on_change=None)`

Single-file picker field stored at runtime as `Path | None`.

### Parameters

- `label: str`
  Field label.
- `default: str | Path | None = None`
  Initial file path.
- `on_change`
  Optional change callback.

### What it changes

- normalizes values into `Path` or `None`
- serializes project-local paths relative to the project root when possible
- resolves relative paths against the project root when loading
- rejects restored paths that do not exist or are not files

### Restore behavior

If a saved file is missing, the field falls back to `None`.

### `FilePaths(label, default=None, on_change=None)`

Multi-file picker field stored at runtime as `list[Path]`.

### Parameters

- `label: str`
  Field label.
- `default: iterable[str | Path] | None = None`
  Initial file list.
- `on_change`
  Optional change callback.

### What it changes

- normalizes values into `list[Path]`
- accepts iterable input
- serializes paths relative to the project root when possible
- resolves relative paths when loading
- rejects restore if any referenced file is missing

### Restore behavior

If any saved file is missing, the field falls back to an empty list.

## Displays

Displays are read-only outputs rendered inside the node body.

They are for visualization, not graph connectivity.

### Shared display behavior

- Display values are assigned like normal attributes: `self.preview = data`
- Display values are not persisted in normal graph save/load
- Some displays also carry a separate config dictionary stored on the node instance

### Shared config behavior

The base `Display` class supports:

- `default_config()`
- `get_config(obj)`
- `set_config(obj, new_config)`

If config changes and an `on_change` handler exists, the callback receives:

- string method name: `def handler(self, new_config, old_config): ...`
- callable: `def handler(obj, new_config, old_config): ...`

Important: this `on_change` is for **display config changes**, not for assigning display values.

### `Numeric(label, format=".4f")`

Displays a scalar value as formatted text.

### Parameters

- `label: str`
  Display label.
- `format: str = ".4f"`
  Python format spec used when the value is `int` or `float`.

### What it changes

- exposes both raw value and formatted text to the UI

### `Plot(label, style="line", color_positive=None, color_negative=None, scale_mode="auto", vmin=0.0, vmax=1.0, line_width=1.5)`

Displays 1D or 2D numeric arrays as a plot.

### Parameters

- `label: str`
  Display label.
- `style: str = "line"`
  Plot style. In the current codebase, `"line"` and `"bar"` are the intended modes.
- `color_positive: str | None = None`
  Color used for positive values. Defaults to `#3dff6f`.
- `color_negative: str | None = None`
  Color used for negative values. Defaults to `#e94560`.
- `scale_mode: str = "auto"`
  Scaling mode. `"auto"` derives range from data, `"manual"` uses `vmin`/`vmax`.
- `vmin: float = 0.0`
  Lower bound used when manual scaling is active.
- `vmax: float = 1.0`
  Upper bound used when manual scaling is active.
- `line_width: float = 1.5`
  Line thickness for line plots.

### What it changes

- validates that assigned values are 1D or 2D arrays
- stores plot config separately from the display value
- provides config helpers:
  - `change_style(obj, style, line_width=None)`
  - `change_scale(obj, scale_mode, vmin=None, vmax=None)`

### Notes

- The UI reads `style`, `scale_mode`, `vmin`, `vmax`, and `line_width` from display config.
- Plot values themselves are not saved with the graph.

### `Heatmap(label, colormap="grayscale", scale_mode="auto", vmin=0.0, vmax=1.0, on_change=None)`

Displays 1D or 2D numeric arrays as a heatmap.

### Parameters

- `label: str`
  Display label.
- `colormap: str = "grayscale"`
  Initial colormap.
- `scale_mode: str = "auto"`
  Scaling mode.
- `vmin: float = 0.0`
  Lower bound for manual scaling.
- `vmax: float = 1.0`
  Upper bound for manual scaling.
- `on_change`
  Optional config-change callback.

### What it changes

- validates that assigned values are 1D or 2D arrays
- stores display config for colormap and scale range
- provides config helpers:
  - `change_colormap(obj, new_colormap)`
  - `change_scale(obj, scale_mode, vmin=None, vmax=None)`

### Notes

- Current built-in nodes use colormap values such as `"grayscale"`, `"bwr"`, and `"viridis"`.

### `Image(label)`

Displays image data directly.

### Parameters

- `label: str`
  Display label.

### What it changes

- validates that assigned values are either:
  - a 2D image
  - a 3D image with shape `H x W x 3`
  - a 3D image with shape `H x W x 4`

### Notes

- Use `Image` when your data already represents an image.
- Use `Heatmap` when you want scalar-array visualization with colormap/scaling behavior.

### `Text(label, default="", max_height=140.0)`

Displays read-only text.

### Parameters

- `label: str`
  Display label.
- `default: str = ""`
  Fallback text before anything is assigned.
- `max_height: float = 140.0`
  UI height cap for the embedded text widget.

### What it changes

- exposes text content to the UI
- controls how tall the text display can grow before scrolling

## Actions

Actions are buttons rendered in the node body.

### `Action(label, callback, params=None, confirm=False)`

Creates an action button that calls a callback.

### Parameters

- `label: str`
  Button text shown in the node.
- `callback: Callable[[obj, Optional[dict]], Any]`
  Function called when the action runs.
- `params: list[dict] | None = None`
  Optional parameter schema metadata for the action.
- `confirm: bool = False`
  Optional confirmation hint metadata.

### What it changes

- renders a button in the node body
- exposes a callable on the instance, so `self.reset()` works
- runs `callback(self, params)` when triggered

### Callback shape

Common pattern:

```python
reset = Action("Reset", lambda self, params=None: self._reset(params))
```

or

```python
def _reset(self, params=None):
    ...
```

### Notes about `params` and `confirm`

These values are included in the action schema, but in the current Qt frontend:

- action buttons are rendered
- the callback is triggered
- extra parameter entry UI is not implemented
- confirmation handling is not implemented

So today, `params` and `confirm` are mainly useful as forward-compatible metadata unless you extend the UI.

## State

State is for internal values that belong to the node and should survive graph save/load.

State is not a UI control and is not directly editable by the user like a field.

### `State(label=None, default=None)`

Stores persisted internal node state.

### Parameters

- `label: str | None = None`
  Optional human-readable name. If omitted, the attribute name is used.
- `default: Any = None`
  Default initial value.

### What it changes

- stores a value on the instance
- includes that value in graph serialization
- adds state metadata to the node schema

### Good uses for `State`

- counters
- rolling history arrays
- selected dataset index
- cached internal values that should be restored when reopening the graph

### Notes

- State labels are currently shown as a hint line in the node UI, not as editable widgets.
- State values are preserved across graph save/load.
- State is not preserved across hot-reload unless your reload flow explicitly restores it.

## When To Use What

- Use `Field` when the user should edit the value in the node UI.
- Use `State` when the value is internal but must persist with the graph.
- Use `Display` when the value should be shown visually or textually in the node body.
- Use `InputPort` and `OutputPort` when the value participates in graph connections.
- Use `Action` when the user should trigger an imperative event such as reset, next, precompute, or randomize.

## Recommended Pattern

For most nodes:

1. Declare inputs, outputs, fields, displays, actions, and state on the class.
2. Use `init()` to create buffers or load resources.
3. Use `@background_init` if `init()` is slow.
4. Use `process()` to read current inputs/fields/state and assign outputs/displays/state.
5. Keep expensive one-time work out of `process()`.

## Common Pitfalls

- Forgetting to subclass `Node`.
- Forgetting to implement `process()`.
- Returning data from `process()` instead of assigning `OutputPort` attributes.
- Storing important persistent values in plain attributes instead of `State`.
- Expecting `Display` values to be saved with the graph.
- Using `@background_init` for work that creates non-pickle-able runtime objects without custom export/apply handling.
- Assuming `Action.params` or `Action.confirm` already have full UI support in the current frontend.

## Quick Reference

### Decorators

- `@branch(path)`: controls palette branch/category
- `@background_init`: runs `init()` asynchronously in a worker process

### Fields

- `Range`: bounded float slider/range
- `Integer`: integer input
- `Float`: float input
- `Bool`: boolean toggle
- `Enum`: fixed-choice string selector
- `DirectoryPath`: directory picker -> `Path | None`
- `FilePath`: single file picker -> `Path | None`
- `FilePaths`: multi-file picker -> `list[Path]`

### Displays

- `Numeric`: scalar numeric text
- `Plot`: line/bar-style plot for 1D or 2D numeric data
- `Heatmap`: colormapped 1D or 2D scalar data
- `Image`: grayscale/RGB/RGBA image
- `Text`: read-only text block

### Other

- `Action`: button callback
- `State`: persisted internal value
- `InputPort`: incoming graph connection
- `OutputPort`: outgoing graph connection
