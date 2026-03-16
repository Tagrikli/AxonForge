<p align="center">
  <video src="data/media_github/axonforge.mp4" controls width="720">
    <a href="data/media_github/axonforge.mp4">Watch the AxonForge demo</a>
  </video>
</p>

<h1 align="center">AxonForge</h1>

<p align="center">
  <strong>A node-based computational framework with a PySide6 visual editor</strong>
</p>

<p align="center">
  <em>Define computational nodes in Python, connect them visually, and run the graph live.</em>
</p>

---

## Overview

AxonForge is a desktop node editor and runtime for building computational graphs. It is designed as a personal lab environment where research projects live in their own directories, completely separate from the framework source.

Core ideas:

- Declarative node classes using descriptors (`InputPort`, `OutputPort`, `Field`, `State`, `Display`, etc.)
- Real-time PySide6 graph editor
- Topological network execution with runtime error reporting
- CLI-driven project workflow (`axonforge init` / `axonforge run`)
- Project-local custom nodes discovered alongside built-in ones
- Multiple graphs per project, managed via a right-side panel
- Hot reload for dynamic nodes

---

## Documentation

- [Node Authoring Guide](docs/node_authoring_guide.md): comprehensive reference for building nodes, including lifecycle, decorators, fields, displays, actions, and state.
- [Autonomous Node Generation Guide](docs/autonomous_node_generation.md): workflow, templates, and verification checklist for reliably generating nodes with an LLM agent.

For installed-package usage outside this repo:

```bash
axonforge docs agent
axonforge docs template
axonforge docs fields
```

Docs topics:

- `agent`: compact authoring rules and workflow for LLM agents and external users
- `template`: a copy-pasteable starter node
- `fields`: field descriptor reference and constructor summary

For machine-readable output:

```bash
axonforge docs agent --json
axonforge docs fields --json
```

### Validation

You can validate a node definition from the terminal without opening the GUI:

```bash
axonforge validate-node path/to/node.py
```

For machine-readable output:

```bash
axonforge validate-node path/to/node.py --json
```

For project-local nodes, run it from the project root or pass `--project-dir` explicitly.

You can also audit declared package dependencies against current repo imports:

```bash
python utils/audit_dependencies.py
python utils/audit_dependencies.py --fail-on-unused
```

---

## Installation

### Prerequisites

- Python 3.12+
- Linux/macOS/Windows with Qt support for PySide6

### Install

```bash
git clone https://github.com/your-username/AxonForge.git
cd AxonForge

# using uv
uv sync

# or using pip (editable install recommended for development)
pip install -e .
```

The editable install (`-e`) means changes to the framework source are reflected immediately in all environments that installed it -- no reinstall needed.

---

## Workflow

### 1. Initialize a project

```bash
mkdir ~/research/my-experiment
cd ~/research/my-experiment
axonforge init --name "my-experiment"
```

This creates the project scaffold:

```
my-experiment/
├── nodes/           <- place custom node files here
├── graphs/          <- node graphs are saved here
├── data/            <- numpy array storage
└── axonforge.json   <- project metadata
```

Built-in MNIST/Fashion-MNIST input nodes do not store datasets in the project folder. They download and cache dataset files under the user's cache directory on first use.

### 2. Run the editor

```bash
# Open with empty canvas
axonforge run

# Open a specific graph
axonforge run training
```

The app launches scoped to the current directory. It discovers both built-in nodes and any custom nodes in your project's `nodes/` folder.

### 3. Work with graphs

- **Shift+E** toggles the graph panel (right-side drawer)
- Click a graph name to switch to it
- Click **+** or press **Ctrl+N** to create a new unnamed graph
- **Ctrl+S** saves the current graph. Unnamed graphs are named inline from the graph drawer.
- Graphs auto-save when switching or closing the app

---

## Quick Start

1. Initialize a project with `axonforge init`.
2. Run `axonforge run` from the project directory.
3. Add nodes: drag from the palette drawer (`Shift+Q`), or press `Shift+A` to search.
4. Connect ports by dragging between them.
5. Set node properties inline.
6. Start the network with `Ctrl+Space` or step with `Ctrl+Return`.
7. Save your graph with `Ctrl+S`.

---

## Controls

| Shortcut / Action | Behavior |
|---|---|
| `Ctrl+Space` | Start/stop network |
| `Ctrl+Return` | Single network step |
| `Ctrl+S` | Save graph |
| `Shift+E` | Toggle graph panel |
| `Shift+A` | Open quick-add node picker |
| `Shift+D` | Duplicate selected node(s) |
| `Shift+Q` | Toggle node palette drawer |
| `Shift+W` | Toggle console |
| Right-click node | Delete node |
| Right-click connection | Delete connection |
| Mouse wheel | Zoom |
| Middle mouse drag | Pan |

---

## Defining Nodes

Nodes are normal Python classes inheriting from `Node`. Place them in your project's `nodes/` folder and they will be auto-discovered.

```python
import numpy as np

from axonforge.core.node import Node, background_init
from axonforge.core.descriptors import branch
from axonforge.core.descriptors.ports import InputPort, OutputPort
from axonforge.core.descriptors.fields import (
    Bool,
    DirectoryPath,
    Enum,
    FilePath,
    FilePaths,
    Integer,
    Range,
)
from axonforge.core.descriptors.displays import Heatmap, Text
from axonforge.core.descriptors.actions import Action
from axonforge.core.descriptors.state import State


@branch("Demo/Examples")
class ExampleNode(Node):
    # Ports: receive/send graph data
    x = InputPort("X", np.ndarray)
    y = OutputPort("Y", np.ndarray)

    # Fields: user-editable controls in the node
    gain = Range("Gain", 1.0, 0.0, 5.0)
    steps = Integer("Steps", 1)
    enabled = Bool("Enabled", True)
    mode = Enum("Mode", ["a", "b"], "a")
    config_file = FilePath("Config File")
    dataset_dir = DirectoryPath("Dataset Dir")
    inputs = FilePaths("Input Files")

    # Displays: visual outputs rendered in the node body
    preview = Heatmap("Preview", colormap="grayscale", scale_mode="auto")
    info = Text("Info", default="ready")

    # Action button: calls _reset when clicked
    reset = Action("Reset", lambda self, params=None: self._reset(params))

    # State: persistent internal value saved/loaded with graph
    counter = State(default=0)

    # Optional: run in background worker process
    @background_init
    def init(self):
        self.counter = 0

    def process(self):
        if self.x is None or not self.enabled:
            return
        self.y = self.x * float(self.gain)
        self.preview = self.y
        self.counter += 1
        self.info = f"ticks={self.counter}, mode={self.mode}"

    def _reset(self, params=None):
        self.counter = 0
```

`FilePath` and `DirectoryPath` load as `Path | None`, and `FilePaths` loads as `list[Path]`. When saved paths live inside the current project, they are written relative to the project root so moving the project keeps those references valid.

The `@branch("Category/Subcategory")` decorator determines where the node appears in the palette. If omitted, the category is inferred from the folder structure.

---

## Project Structure

Each project directory contains:

```
my-project/
├── axonforge.json       <- project marker and metadata
├── nodes/               <- custom node .py files (auto-discovered)
├── graphs/              <- graph JSON files
│   ├── training.json
│   └── evaluation.json
└── data/                <- shared numpy arrays (.npy)
```

### Graph file format (v5)

```json
{
  "version": 5,
  "kind": "graph",
  "name": "training",
  "viewport": {
    "pan": {"x": 12.3, "y": -45.6},
    "zoom": 1.2
  },
  "editor": {
    "drawer_collapsed": true,
    "drawer_width": 260,
    "console_collapsed": false,
    "console_height": 220,
    "max_hz_enabled": false,
    "max_hz_value": 60
  },
  "nodes": [...],
  "connections": [...]
}
```

Arrays are stored as `.npy` files in `data/` with UUID filenames, shared across all graphs in a project.

---

## Console and Logging

- Console lines are prefixed with `<user>@<project> >` (or `!>` for error stream).
- Multi-line tracebacks are colorized by line type.

---

## Runtime Notes

- Network execution uses topological ordering with cycle fallback semantics.
- Signals are passed by reference (no defensive clone in runtime).
  - Avoid in-place mutation of shared input objects unless intended.
- Process errors stop the network and mark failing nodes.

---

## Architecture

```text
axonforge/                   <- Core framework
  cli.py                     <- CLI entry point (init, run)
  core/
    node.py
    registry.py
    descriptors/
  network/
    network.py
  nodes/                     <- Built-in node modules
    ...

axonforge_qt/                <- PySide6 GUI
  main.py
  main_window.py
  bridge.py
  computation_thread.py
  canvas/
  panels/
    palette.py               <- Left drawer: node palette
    graph_panel.py           <- Right drawer: graph management
    console.py
  themes/
```

---

## License

Provided as-is for research and experimentation.
