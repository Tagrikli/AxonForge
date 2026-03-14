# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**AxonForge** is a personal lab environment for building and experimenting with **biologically-inspired local learning algorithms**. It provides a node-based PySide6 visual editor where algorithms are wired together as computational graphs and run interactively. The project directory is `MiniCortex/` but the package is `axonforge`.

The primary research focus is the **Hypercolumn** family of nodes (`axonforge/nodes/hypercolumn/`), which model cortical minicolumn dynamics. The current stable algorithm is **GeodesicIMDVv2** (`learning3.py`), documented in `notes/GeodesicIMDVv2.md`. It implements LOO-shadow-inhibited activations with geodesic (Rodrigues rotation) weight updates on the unit hypersphere — a single mechanism that drives both learning specialization and activation sparsity without any separate inhibition parameter.

## Commands

```bash
# Install dependencies (preferred)
uv sync

# Run the application
python -m axonforge_qt.main
# or after install:
axonforge-qt
# or from root:
python main.py
```

There is no test suite currently.

## Architecture

The codebase has two top-level packages:

- **`axonforge/`** — Core framework (nodes, descriptors, execution engine, registry)
- **`axonforge_qt/`** — PySide6 GUI (canvas, panels, bridge, computation thread)

### Core: Descriptor Protocol

Nodes declare their interface using Python data descriptors at the class level. `NodeMeta` (metaclass in `axonforge/core/node.py`) scans the class at definition time and populates `_input_ports`, `_output_ports`, `_properties`, `_actions`, `_outputs`, and `_stores`.

```python
class MyNode(Node):
    x = InputPort("X", np.ndarray)
    y = OutputPort("Y", np.ndarray)
    gain = Range("Gain", 1.0, 0.0, 5.0)
    mode = Enum("Mode", ["a", "b"], "a")
    info = Text("Info", default="ready")
    counter = Store(default=0)
    reset = Action("Reset", callback="_reset")

    def process(self):
        if self.x is not None:
            self.y = self.x * float(self.gain)
```

Descriptor types live in `axonforge/core/descriptors/`:
- `ports.py` — `InputPort`, `OutputPort`
- `properties.py` — `Range`, `Integer`, `Float`, `Bool`, `Enum`
- `displays.py` — `Text`, `Vector2D`, `BarChart`, `LineChart`, etc. (read-only UI outputs)
- `actions.py` — `Action` (button that invokes a method)
- `store.py` — `Store` (persisted per-node state, saved as `.npy` in `data/`)

### Core: Network Execution

`axonforge/network/network.py` runs execution with Kahn's algorithm for topological sort. Cyclic connections are processed after acyclic nodes as feedback from the previous step. Signals flow via `_signals_now` dict: output values are written there after `process()`, then read as inputs on the next node. Node failures mark the node as failed and collect tracebacks without stopping the whole graph.

### Core: Registry

`axonforge/core/registry.py` holds two global thread-safe registries:
- `node_registry` — maps `node_id → Node instance`
- `connection_registry` — maps `connection_id → (src_node_id, src_port, dst_node_id, dst_port)`

### Core: Node Discovery

Nodes are discovered from `axonforge/nodes/` via `discover_nodes()` in `axonforge/core/descriptors/node.py`. The `@branch("Category/Subcategory")` decorator categorizes nodes for the palette. Hot-reload is supported via `axonforge/core/descriptors/dynamic.py`.

### GUI: BridgeAPI

`axonforge_qt/bridge.py` is the central synchronous API (~919 lines) that connects the Qt UI to the framework:
- Node discovery and palette building
- Project load/save (JSON + `data/` folder for numpy arrays)
- Thread-safe display buffer management (lock-protected snapshots from `ComputationThread`)
- Background init job polling (process-based `@background_init` for expensive node setup)
- Network state queries

### GUI: ComputationThread

`axonforge_qt/computation_thread.py` runs `network.execute_step()` in a `QThread` loop. It samples execution Hz, throttles based on user speed settings, snapshots display outputs to a shared buffer, and emits errors to the UI.

### GUI: Canvas

`axonforge_qt/canvas/` contains the QGraphicsScene/View:
- `editor_scene.py` — node/connection graphics management, hit testing, Shift+A quick-add
- `node_item.py` — draggable node widgets with inline property editing
- `connection_item.py` / `port_item.py` — edge drawing and port hit detection

### Project File Format

Saved as `project.json` + `data/` folder. Version 4. Structure:
```json
{
  "version": 4, "kind": "project", "project_name": "...",
  "viewport": {"pan": {"x": 0, "y": 0}, "zoom": 1.0},
  "editor": {"drawer_collapsed": false, "console_collapsed": false, ...},
  "nodes": [...],
  "connections": [...]
}
```
NumPy arrays (from `Store` descriptors) are saved as `.npy` files in `data/`.

## Key Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+Space` | Start/stop network execution |
| `Ctrl+Return` | Single step |
| `Ctrl+S` | Save project |
| `Shift+A` | Quick-add node picker |
| `Shift+D` | Duplicate selected node(s) |
| `Shift+R` | Toggle drawer |
| `Shift+T` | Toggle console |
| Right-click node/connection | Delete |

## Adding New Nodes

1. Create a file under `axonforge/nodes/<category>/`.
2. Subclass `Node`, use descriptors to declare ports/properties/displays.
3. Implement `process(self)` — called every network step.
4. Optionally implement `init(self)` decorated with `@background_init` for expensive setup.
5. Decorate the class with `@branch("Category/Subcategory")`.
6. Refresh the palette (button in UI) or restart to discover the new node.
