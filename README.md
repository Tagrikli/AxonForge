<p align="center">
  <img src="data/media_github/axonforge2.gif" alt="AxonForge demo 2" width="720">
</p>

<h1 align="center">AxonForge</h1>

<p align="center">
  <strong>A node-based computational framework for prototyping algorithms that evolve over time.</strong>
</p>

<p align="center">
  Wire nodes, connect feedback loops, tweak parameters live, and watch your system converge — all without leaving the canvas.
</p>

---

## Why AxonForge?

Most dataflow tools force you into directed acyclic graphs. Most simulators lock you into a specific model family. AxonForge does neither.

It gives you a **live, visual graph** where every node is a plain Python class, connections can form cycles, and execution ticks forward one step at a time — making feedback, recurrence, and iterative learning first-class concepts.

**Designed for computational neuroscience prototyping. General enough for any iterative computation.**

### What sets it apart

- **Feedback-native execution** — Topological sort with cycle-aware semantics. Recurrent connections deliver previous-step values automatically. No delay nodes, no workarounds.
- **Declarative node API** — A node is a Python class with descriptors. Ports, fields, displays, state — all declared as class attributes. Minimal boilerplate, maximum clarity.
- **Live parameter tuning** — Sliders, toggles, enums, and file pickers rendered directly in each node. Change a value, see the effect immediately.
- **Real-time visualization** — Heatmaps, plots, images, and text displays embedded in nodes. Watch weights organize, activations sharpen, and losses drop as the graph runs.
- **Hot-reloadable nodes** — Edit your node's source, hit reload. Field values preserved, code updated, no restart needed.
- **Project-local architecture** — Your research lives in its own directory with its own nodes, graphs, and data. The framework stays separate. Move your project, share it, version it independently.

---

## Use Cases

**Computational neuroscience** — Prototype bio-inspired learning rules, lateral inhibition, population codes, and cortical models with feedback loops between layers.

**Dynamical systems** — Build and observe systems with recurrent dynamics: oscillators, attractors, coupled differential equations, control loops.

**Signal processing** — Chain transforms, filters, and analysis nodes into live pipelines with immediate visual feedback.

**Algorithm design** — Explore iterative algorithms step by step. Decompose complex computations into inspectable, rewirable stages.

---

## Quick Start

```bash
# Install
git clone https://github.com/your-username/AxonForge.git
cd AxonForge
uv sync            # or: pip install -e .

# Create a project
mkdir ~/research/my-experiment && cd ~/research/my-experiment
axonforge init --name "my-experiment"

# Launch the editor
axonforge run
```

This gives you:

```
my-experiment/
├── nodes/           ← your custom node files (auto-discovered)
├── graphs/          ← saved computational graphs
├── data/            ← numpy array storage
└── axonforge.json   ← project metadata
```

### First steps on the canvas

1. Open the node palette with **Shift+Q**, or search nodes with **Shift+A**.
2. Drag nodes onto the canvas and connect ports by dragging between them.
3. Set parameters inline — sliders, toggles, dropdowns right on the node.
4. Run the graph with **Ctrl+Space** (continuous) or **Ctrl+Return** (single step).
5. Save with **Ctrl+S**.

---

## Defining a Node

Nodes are Python classes. Drop them in your project's `nodes/` folder and they appear in the palette automatically.

```python
import numpy as np
from axonforge.core.node import Node
from axonforge.core.descriptors import branch
from axonforge.core.descriptors.ports import InputPort, OutputPort
from axonforge.core.descriptors.fields import Range, Bool
from axonforge.core.descriptors.displays import Heatmap, Text
from axonforge.core.descriptors.actions import Action
from axonforge.core.descriptors.state import State


@branch("Demo")
class LeakyIntegrator(Node):
    """A simple leaky integrator with adjustable decay."""

    signal = InputPort("Signal", np.ndarray)
    output = OutputPort("Output", np.ndarray)

    decay = Range("Decay", default=0.9, min_val=0.0, max_val=1.0, step=0.01)
    enabled = Bool("Enabled", default=True)

    preview = Heatmap("State", colormap="viridis")
    info = Text("Info", default="idle")

    reset = Action("Reset", lambda self, p=None: self._reset(p))
    accumulator = State(default=None)

    def init(self):
        self.accumulator = None

    def process(self):
        if self.signal is None or not self.enabled:
            return
        x = np.asarray(self.signal, dtype=np.float64)
        if self.accumulator is None:
            self.accumulator = np.zeros_like(x)
        self.accumulator = float(self.decay) * self.accumulator + x
        self.output = self.accumulator
        self.preview = self.accumulator
        self.info = f"peak={self.accumulator.max():.3f}"

    def _reset(self, params=None):
        self.accumulator = None
```

That's it. Ports define data flow. Fields give you live controls. Displays show results. State persists across steps. The `@branch` decorator sets the palette category — or omit it and the folder structure is used instead.

---

## Built-in Node Library

AxonForge ships with a few ready to use nodes:

| Category | Nodes |
|---|---|
| **Math** | Arithmetic, comparison, trigonometry |
| **Linear Algebra** | Vector ops, matrix ops, SVD, eigen decomposition, QR |
| **Statistics** | Mean, std, variance, correlation, histogram, argmax/argmin |
| **Information Theory** | Entropy, KL divergence, mutual information, cross-entropy |
| **Noise** | Gaussian, uniform, Perlin, Worley, blue noise, salt & pepper |
| **Input** | Image loader, rotating line, moving shape, MNIST/Fashion-MNIST |
| **Encoding** | One-hot, sparse hash, unit-vector hash, spatial encoders |
| **Display** | Heatmap, vector plot, image viewer, line/bar charts |
| **Utilities** | Softmax, top-K, Jaccard index, range mapping, sliders |

All nodes are composable — connect any compatible ports and build your pipeline.

---

## How Execution Works

Each tick of the graph:

1. **Topological sort** orders nodes by dependency (Kahn's algorithm).
2. Nodes in cycles are placed last — their inputs carry **previous-step values**, giving you implicit feedback with one-step delay.
3. Each node's `process()` runs in order: read inputs, compute, write outputs.
4. Displays update in real-time on the canvas.

This means feedback connections just work. Wire a node's output back to an earlier node's input, and the graph becomes recurrent — no special configuration needed.

---

## Controls

| Shortcut | Action |
|---|---|
| **Ctrl+Space** | Start / stop the graph |
| **Ctrl+Return** | Single step |
| **Ctrl+S** | Save graph |
| **Ctrl+N** | New graph |
| **Shift+A** | Quick-add node search |
| **Shift+Q** | Toggle node palette |
| **Shift+E** | Toggle graph panel |
| **Shift+D** | Duplicate selected nodes |
| **Shift+W** | Toggle console |
| **Shift+X** | Delete selection / hovered connection |
| Mouse wheel | Zoom canvas |
| Middle drag | Pan canvas |
| Scroll on plot | Zoom plot (Y-axis for line/bar, both axes for scatter) |
| Right-drag on plot | Pan plot view |
| Double-click plot | Reset plot to auto-fit |

---

## Documentation

- [Node Authoring Guide](docs/node_authoring_guide.md) — Complete reference for building nodes: lifecycle, descriptors, fields, displays, actions, and state.
- [Autonomous Node Generation Guide](docs/autonomous_node_generation.md) — Templates and workflow for generating nodes with LLM agents.

From the terminal:

```bash
axonforge docs agent       # Authoring rules (compact, LLM-friendly)
axonforge docs template    # Copy-pasteable starter node
axonforge docs fields      # Field descriptor reference
```

Validate a node without opening the editor:

```bash
axonforge validate-node path/to/node.py
```

---

## Project Structure

```
my-project/
├── axonforge.json       ← project metadata
├── nodes/               ← custom nodes (auto-discovered on launch)
├── graphs/              ← graph definitions (JSON, v5)
│   ├── training.json
│   └── evaluation.json
└── data/                ← numpy arrays (.npy, shared across graphs)
```

Graphs are self-contained JSON files. Arrays are stored externally as `.npy` with UUID filenames. Path references inside graphs are project-relative, so you can move or share the whole directory.

---

## Architecture

```
axonforge/               ← Core (pure Python, no GUI dependency)
  core/
    node.py              ← Node base class, metaclass
    registry.py          ← Thread-safe node & connection registry
    descriptors/         ← Ports, fields, displays, actions, state
  network/
    network.py           ← Execution engine, topological sort
  nodes/                 ← Built-in node library
  cli.py                 ← CLI entry point

axonforge_qt/            ← Visual editor (PySide6)
  bridge.py              ← API between engine and GUI
  canvas/                ← Graph canvas and node rendering
  panels/                ← Palette, graph manager, console
```

The core framework has no GUI dependency. The visual editor is a separate layer built with PySide6.

---

## Runtime Notes

- Signals are passed by reference for performance. Avoid in-place mutation of inputs unless intended.
- Process errors stop the network and highlight the failing node with full traceback.
- Background initialization (`@background_init`) runs heavy setup in a worker process without blocking the UI.
- Built-in dataset nodes (MNIST, Fashion-MNIST) cache downloads in the user's cache directory, not in the project.

---

## Requirements

- Python 3.12+
- PySide6 (for the visual editor)
- NumPy

---

## License

Provided as-is for research and experimentation.
