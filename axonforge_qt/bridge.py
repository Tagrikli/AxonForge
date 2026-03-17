"""
BridgeAPI — Direct Python interface replacing the REST/WebSocket server layer.

All operations that the web UI performed via HTTP are now synchronous method calls.
The UI calls these directly; no serialization, no network overhead.
"""

import json
import importlib
import multiprocessing
import copy
import sys
import threading
import traceback
import uuid
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from axonforge.core.node import Node
from axonforge.core.registry import (
    register_node,
    unregister_node,
    get_node,
    get_all_nodes,
    get_connections,
    get_connections_for_node,
    add_connection,
    remove_connection,
    clear_node_registry,
)
from axonforge.core.descriptors.ports import _format_data_type
from axonforge.core.descriptors.node import (
    discover_nodes,
    rediscover_nodes,
    build_node_palette,
)
from axonforge.network.network import Network, NetworkError


APP_NAME = "AxonForge"
GRAPH_DATA_DIRNAME = "data"
GRAPH_DIR_NAME = "graphs"
DEFAULT_EDITOR_STATE = {
    "drawer_collapsed": True,
    "drawer_width": 260,
    "console_collapsed": False,
    "console_height": 220,
    "max_hz_enabled": False,
    "max_hz_value": 60,
}


def _run_node_init_process(
    module_name: str,
    class_name: str,
    node_data: Dict[str, Any],
    project_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Process worker for running Node.init() outside the UI process.

    Returns:
        {"ok": bool, "state": dict, "error": str, "traceback": str}
    """
    try:
        module = importlib.import_module(module_name)
        node_class = getattr(module, class_name)
        node = node_class.from_dict(
            node_data,
            project_dir=Path(project_dir) if project_dir else None,
        )
        node.init()
        return {
            "ok": True,
            "state": node.export_background_init_state(),
            "error": "",
            "traceback": "",
        }
    except Exception as exc:
        return {
            "ok": False,
            "state": {},
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


class BridgeAPI:
    """Synchronous API that the Qt UI calls directly."""

    def __init__(self, project_dir: Path) -> None:
        self.project_dir: Path = project_dir
        self.current_graph_name: Optional[str] = None
        self.current_graph_dirty: bool = False
        self.nodes: List[Node] = []
        self._nodes_lock = threading.Lock()
        self.network: Network = Network()
        self.node_classes: Dict[str, Type[Node]] = {}
        self.node_palette: Dict[str, List[Type[Node]]] = {}
        self.viewport: Dict[str, Any] = {"pan": {"x": 0.0, "y": 0.0}, "zoom": 1.0}
        self.editor_state: Dict[str, Any] = dict(DEFAULT_EDITOR_STATE)
        self._apply_editor_state_to_network()

        # Shared display buffer: {node_id: {output_key: value}}
        self._display_buffer: Dict[str, Dict[str, Any]] = {}
        self._buffer_lock = threading.Lock()
        self._dirty = False
        self._node_loading_states: Dict[str, Dict[str, Any]] = {}
        self._loading_lock = threading.Lock()
        self._init_jobs_lock = threading.Lock()
        self._pending_init_jobs: Dict[str, Future] = {}
        self._init_executor = ProcessPoolExecutor(
            max_workers=max(1, min(4, multiprocessing.cpu_count() or 1)),
            mp_context=multiprocessing.get_context("spawn"),
        )
        self._loading_project = False

    # ── Initialisation ───────────────────────────────────────────────────

    def init(self) -> None:
        """Discover built-in and project-local nodes, build palette."""
        discover_nodes()
        local_nodes = self.project_dir / "nodes"
        if local_nodes.exists():
            discover_nodes(nodes_dir=local_nodes, package=None)
        palette = build_node_palette()
        for category, class_list in palette.items():
            for cls in class_list:
                self.node_classes[cls.__name__] = cls
        self.node_palette = palette

    # ── Display buffer (shared with computation thread) ──────────────────

    def snapshot_displays(self) -> None:
        """Write current display outputs of every node into the shared buffer."""
        if self._loading_project:
            return
        with self._nodes_lock:
            nodes_copy = list(self.nodes)

        buf: Dict[str, Dict[str, Any]] = {}
        for node in nodes_copy:
            outputs: Dict[str, Any] = {}
            for name, descriptor in getattr(node.__class__, "_outputs", {}).items():
                try:
                    val = getattr(node, name)
                except Exception:
                    val = None
                output_config = descriptor.get_config(node)
                outputs[name] = {
                    'value': self._clone_display_value(val),
                    'config': dict(output_config),
                }
            buf[node.node_id] = outputs
        with self._buffer_lock:
            self._display_buffer = buf
            self._dirty = True

    def _clone_display_value(self, value: Any) -> Any:
        """Clone display payloads so the UI sees stable snapshots, not live buffers."""
        try:
            import numpy as np
            if isinstance(value, np.ndarray):
                return value.copy()
        except Exception:
            pass

        if isinstance(value, list):
            return [self._clone_display_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._clone_display_value(item) for item in value)
        if isinstance(value, dict):
            return {k: self._clone_display_value(v) for k, v in value.items()}

        try:
            return copy.deepcopy(value)
        except Exception:
            return value

    def read_display_buffer(self) -> Optional[Dict[str, Dict[str, Any]]]:
        """Read the display buffer if dirty. Returns None if clean."""
        with self._buffer_lock:
            if not self._dirty:
                return None
            self._dirty = False
            return self._display_buffer

    def get_node_loading_states(self) -> Dict[str, Dict[str, Any]]:
        """Get per-node background initialization state."""
        with self._loading_lock:
            return dict(self._node_loading_states)

    def _set_node_loading_state(self, node_id: str, loading: bool, error: str = "") -> None:
        with self._loading_lock:
            self._node_loading_states[node_id] = {
                "loading": bool(loading),
                "error": str(error or ""),
            }

    def _start_background_init(self, node: Node) -> None:
        """Schedule node.init() in a worker process and mark node as loading."""
        node_id = node.node_id
        if not node_id:
            return
        node._loading = True
        self._set_node_loading_state(node_id, True, "")
        payload = node.to_dict(project_dir=self.project_dir)
        future = self._init_executor.submit(
            _run_node_init_process,
            node.__class__.__module__,
            node.__class__.__name__,
            payload,
            str(self.project_dir),
        )
        with self._init_jobs_lock:
            self._pending_init_jobs[node_id] = future

    def _cancel_background_init(self, node_id: str) -> None:
        with self._init_jobs_lock:
            future = self._pending_init_jobs.pop(node_id, None)
        if future is not None and not future.done():
            future.cancel()

    def poll_background_init_jobs(self) -> None:
        """Apply finished background init results to live node instances."""
        completed: List[tuple[str, Future]] = []
        with self._init_jobs_lock:
            for node_id, future in list(self._pending_init_jobs.items()):
                if future.done():
                    completed.append((node_id, future))
                    self._pending_init_jobs.pop(node_id, None)

        if not completed:
            return

        has_display_update = False
        for node_id, future in completed:
            node = get_node(node_id)
            if node is None:
                continue

            existing_error = str(getattr(node, "_loading_error", "") or "")
            error = ""
            worker_traceback = ""
            try:
                result = future.result()
            except Exception as exc:
                error = str(exc)
                worker_traceback = traceback.format_exc()
            else:
                if not result.get("ok", False):
                    error = result.get("error", "Background init failed")
                    worker_traceback = result.get("traceback", "")
                else:
                    try:
                        node.apply_background_init_state(result.get("state", {}))
                    except Exception as exc:
                        error = f"Failed to apply init state: {exc}"
                        worker_traceback = traceback.format_exc()
                    else:
                        if self.network and not self.network.running:
                            try:
                                self.network.propagate_from_node(node_id, recompute_start=True)
                            except NetworkError as exc:
                                print(f"Background init propagation error in '{node_id}': {exc}")
                                if exc.traceback:
                                    print(exc.traceback)

            if error:
                print(f"Node init failed for {node.node_type}: {error}")
                if worker_traceback:
                    print(worker_traceback)

            final_error = "\n".join(part for part in (existing_error, error) if part)
            node._loading = False
            node._loading_error = final_error
            self._set_node_loading_state(node_id, False, final_error)
            has_display_update = True

        if has_display_update:
            self.snapshot_displays()

    def shutdown_background_workers(self) -> None:
        """Stop worker processes used by background init."""
        with self._init_jobs_lock:
            self._pending_init_jobs.clear()
        self._init_executor.shutdown(wait=True, cancel_futures=True)

    # ── Node CRUD ────────────────────────────────────────────────────────

    def create_node(self, node_type: str, x: float = 0, y: float = 0) -> Node:
        node_class = self.node_classes.get(node_type)
        if not node_class:
            raise ValueError(f"Unknown node type: {node_type}")
        node = node_class(x=x, y=y)
        register_node(node)
        node.validate_required_methods()
        with self._nodes_lock:
            self.nodes.append(node)

        if getattr(node.__class__, "_uses_background_init", False):
            self._start_background_init(node)
            self.snapshot_displays()
            return node

        init_ok = True
        try:
            node.init()
        except Exception as e:
            init_ok = False
            error = str(e)
            tb = traceback.format_exc()
            node._loading_error = error
            self._set_node_loading_state(node.node_id, False, error)
            print(f"Node init failed for {node_type}: {e}")
            print(tb)
        else:
            node._loading_error = ""
            self._set_node_loading_state(node.node_id, False, "")

        # Propagate current state only when init succeeded.
        if init_ok and self.network:
            try:
                self.network.propagate_current_state()
            except NetworkError as e:
                # Error is stored in network.last_error and node is tracked in failed_nodes
                print(f"Node creation error: {e}")
                if e.traceback:
                    print(e.traceback)
        self.mark_current_graph_dirty()
        self.snapshot_displays()
        return node

    def delete_node(self, node_id: str) -> None:
        if not get_node(node_id):
            raise ValueError(f"Node not found: {node_id}")
        self._cancel_background_init(node_id)
        for conn in get_connections_for_node(node_id):
            remove_connection(
                conn["from_node"], conn["from_output"],
                conn["to_node"], conn["to_input"],
            )
        with self._nodes_lock:
            self.nodes = [n for n in self.nodes if n.node_id != node_id]
        unregister_node(node_id)
        with self._loading_lock:
            self._node_loading_states.pop(node_id, None)
        self.mark_current_graph_dirty()
        self.snapshot_displays()

    def get_node_schema(self, node_id: str) -> dict:
        node = get_node(node_id)
        if not node:
            raise ValueError(f"Node not found: {node_id}")
        return node.get_schema()

    def update_node_position(self, node_id: str, x: float, y: float) -> None:
        node = get_node(node_id)
        if node:
            node.position = {"x": float(x), "y": float(y)}
            self.mark_current_graph_dirty()

    def set_node_name(self, node_id: str, name: str) -> str:
        node = get_node(node_id)
        if not node:
            raise ValueError(f"Node not found: {node_id}")
        clean_name = str(name).strip() or node.node_type
        node.name = clean_name
        self.mark_current_graph_dirty()
        return clean_name

    # ── Fields & Actions ─────────────────────────────────────────────────

    def set_field(self, node_id: str, field_key: str, value: Any) -> None:
        node = get_node(node_id)
        if not node:
            raise ValueError(f"Node not found: {node_id}")
        setattr(node, field_key, value)
        # Only propagate when network is not running (let running network handle processing)
        if self.network and not self.network.running:
            try:
                self.network.propagate_from_node(node_id, recompute_start=True)
            except NetworkError as e:
                # Error is stored in network.last_error and node is tracked in failed_nodes
                print(f"Field set error in '{node_id}.{field_key}': {e}")
                if e.traceback:
                    print(e.traceback)
        self.mark_current_graph_dirty()
        self.snapshot_displays()

    def execute_action(self, node_id: str, action_key: str, params: Optional[dict] = None) -> Any:
        node = get_node(node_id)
        if not node:
            raise ValueError(f"Node not found: {node_id}")
        action = getattr(node, action_key)
        result = action(params)
        if self.network:
            try:
                self.network.propagate_from_node(node_id, recompute_start=True)
            except NetworkError as e:
                # Error is stored in network.last_error and node is tracked in failed_nodes
                print(f"Action execution error in '{node_id}.{action_key}': {e}")
                if e.traceback:
                    print(e.traceback)
        self.mark_current_graph_dirty()
        self.snapshot_displays()
        return result

    def handle_display_event(
        self, node_id: str, display_key: str,
        event_type: str, row: int, col: int,
        button: str = "", delta: int = 0,
    ) -> None:
        node = get_node(node_id)
        if not node:
            return
        descriptor = getattr(node.__class__, "_outputs", {}).get(display_key)
        if descriptor is None:
            return
        handler = getattr(descriptor, f"handle_{event_type}", None)
        if handler is None:
            return
        if event_type == "scroll":
            handler(node, row, col, delta)
        else:
            handler(node, row, col, button)
        if self.network and not self.network.running:
            try:
                self.network.propagate_from_node(node_id, recompute_start=True)
            except NetworkError as e:
                print(f"Display event error in '{node_id}.{display_key}': {e}")
                if e.traceback:
                    print(e.traceback)
        self.mark_current_graph_dirty()
        self.snapshot_displays()

    def set_output_enabled(self, node_id: str, output_key: str, enabled: bool) -> None:
        node = get_node(node_id)
        if not node:
            raise ValueError(f"Node not found: {node_id}")
        if not hasattr(node, "_output_enabled") or node._output_enabled is None:
            node._output_enabled = {}
        node._output_enabled[output_key] = bool(enabled)
        self.mark_current_graph_dirty()

    # ── Connections ──────────────────────────────────────────────────────

    def create_connection(
        self, from_node: str, from_output: str, to_node: str, to_input: str
    ) -> None:
        fn = get_node(from_node)
        tn = get_node(to_node)
        if not fn or not tn:
            raise ValueError("Node not found")

        from_port = next(
            (p for p in getattr(fn.__class__, "_output_ports", {}).values()
             if p.name == from_output), None,
        )
        to_port = next(
            (p for p in getattr(tn.__class__, "_input_ports", {}).values()
             if p.name == to_input), None,
        )
        if not from_port or not to_port:
            raise ValueError("Port not found")

        if not self._is_type_compatible(from_port.data_type, to_port.data_type):
            raise ValueError(
                f"Incompatible types: {_format_data_type(from_port.data_type)} → "
                f"{_format_data_type(to_port.data_type)}"
            )

        # Remove existing connection to the same input
        for conn in get_connections():
            if conn["to_node"] == to_node and conn["to_input"] == to_input:
                remove_connection(
                    conn["from_node"], conn["from_output"],
                    conn["to_node"], conn["to_input"],
                )
                if self.network and hasattr(self.network, "_signals_now"):
                    key = (conn["from_node"], conn["from_output"])
                    self.network._signals_now.pop(key, None)

        if not add_connection(from_node, from_output, to_node, to_input):
            raise ValueError("Connection already exists")

        if self.network:
            try:
                self.network.propagate_from_node(to_node, recompute_start=True)
            except NetworkError as e:
                # Error is stored in network.last_error and node is tracked in failed_nodes
                print(f"Connection creation error from '{from_node}.{from_output}' to '{to_node}.{to_input}': {e}")
                if e.traceback:
                    print(e.traceback)
        self.mark_current_graph_dirty()
        self.snapshot_displays()

    def delete_connection(
        self, from_node: str, from_output: str, to_node: str, to_input: str
    ) -> None:
        if not remove_connection(from_node, from_output, to_node, to_input):
            raise ValueError("Connection not found")
        target = get_node(to_node)
        if target and hasattr(target, to_input):
            setattr(target, to_input, None)
        if self.network:
            key = (from_node, from_output)
            if hasattr(self.network, "_signals_now"):
                self.network._signals_now.pop(key, None)
        self.mark_current_graph_dirty()
        self.snapshot_displays()

    # ── Network Control ──────────────────────────────────────────────────

    def start_network(self) -> None:
        self.network.start()

    def stop_network(self) -> None:
        if self.network:
            self.network.stop()

    def step_network(self) -> None:
        if self.network.running:
            self.network.stop()
        try:
            self.network.execute_step()
        except NetworkError as e:
            print(f"Network step error: {e}")
            if e.traceback:
                print(e.traceback)
            # error stored in network.last_error
        self.snapshot_displays()

    def set_network_speed(self, speed: float) -> None:
        clamped = max(1.0, min(1000.0, float(speed)))
        self.network.speed = clamped
        self.editor_state["max_hz_value"] = int(clamped)

    def set_network_max_speed(self, enabled: bool) -> None:
        value = bool(enabled)
        self.network.max_speed = value
        self.editor_state["max_hz_enabled"] = value

    def get_network_state(self) -> dict:
        return {
            "running": bool(self.network.running),
            "speed": float(self.network.speed),
            "max_speed": bool(getattr(self.network, "max_speed", False)),
            "step": int(self.network.get_step_count()),
            "actual_hz": float(getattr(self.network, "actual_hz", 0.0)),
        }

    # ── Graphs ────────────────────────────────────────────────────────────

    @property
    def project_name(self) -> str:
        """Read the project name from axonforge.json."""
        meta_file = self.project_dir / "axonforge.json"
        if meta_file.exists():
            try:
                data = json.loads(meta_file.read_text(encoding="utf-8"))
                return data.get("name", self.project_dir.name)
            except Exception:
                pass
        return self.project_dir.name

    @property
    def current_graph_display_name(self) -> str:
        return self.current_graph_name or "unsaved"

    def mark_current_graph_dirty(self) -> None:
        self.current_graph_dirty = True

    def mark_current_graph_saved(self) -> None:
        self.current_graph_dirty = False

    def current_graph_has_content(self) -> bool:
        if self.nodes:
            return True
        return bool(get_connections())

    def current_graph_has_unsaved_state(self) -> bool:
        return self.current_graph_dirty or (
            self.current_graph_name is None and self.current_graph_has_content()
        )

    def _clear_graph_state(self) -> None:
        self.stop_network()
        # Wait for any ongoing network step to finish before clearing state
        with self.network._step_lock:
            with self._buffer_lock:
                self._display_buffer = {}
                self._dirty = False
            with self._init_jobs_lock:
                for future in self._pending_init_jobs.values():
                    if not future.done():
                        future.cancel()
                self._pending_init_jobs.clear()
            clear_node_registry()
            with self._nodes_lock:
                self.nodes.clear()
            self.viewport = {"pan": {"x": 0.0, "y": 0.0}, "zoom": 1.0}
            self.editor_state = dict(DEFAULT_EDITOR_STATE)
            self._apply_editor_state_to_network()
            with self._loading_lock:
                self._node_loading_states = {}

    def list_graphs(self) -> List[str]:
        """Return sorted list of graph names (without .json extension)."""
        graphs_dir = self.project_dir / GRAPH_DIR_NAME
        if not graphs_dir.exists():
            return []
        return sorted(
            p.stem for p in graphs_dir.glob("*.json")
        )

    def save_graph(self, name: str) -> None:
        """Save current graph state to graphs/<name>.json."""
        graphs_dir = self.project_dir / GRAPH_DIR_NAME
        graphs_dir.mkdir(parents=True, exist_ok=True)
        data_dir = self.project_dir / GRAPH_DATA_DIRNAME
        data_dir.mkdir(parents=True, exist_ok=True)

        def _array_serializer(array: Any) -> Dict[str, Any]:
            if not hasattr(array, "shape"):
                return {"_type": "unsupported_array"}
            array_name = f"{uuid.uuid4().hex}.npy"
            array_rel_path = Path(GRAPH_DATA_DIRNAME) / array_name
            array_abs_path = self.project_dir / array_rel_path
            import numpy as np
            np.save(array_abs_path, array)
            return {
                "_type": "ndarray_ref",
                "file": str(array_rel_path.as_posix()),
            }

        with self._nodes_lock:
            nodes_copy = list(self.nodes)
        payload = {
            "version": 5,
            "kind": "graph",
            "name": name,
            "viewport": self.viewport,
            "editor": self.get_editor_state(),
            "nodes": [
                n.to_dict(array_serializer=_array_serializer, project_dir=self.project_dir)
                for n in nodes_copy
            ],
            "connections": get_connections(),
        }
        graph_file = graphs_dir / f"{name}.json"
        with open(graph_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        self.current_graph_name = name
        self.mark_current_graph_saved()

    def load_graph(self, name: str) -> dict:
        """Load a graph from graphs/<name>.json. Returns viewport state."""
        self._loading_project = True
        try:
            graph_file = self.project_dir / GRAPH_DIR_NAME / f"{name}.json"
            if not graph_file.exists():
                raise FileNotFoundError(f"Graph not found: {graph_file}")

            with open(graph_file, encoding="utf-8") as f:
                data = json.load(f)

            self._clear_graph_state()

            def _array_loader(ref: dict) -> Any:
                ref_file = str(ref.get("file", "")).strip()
                if not ref_file:
                    raise ValueError("Invalid ndarray_ref with empty file")
                array_path = (self.project_dir / ref_file).resolve()
                try:
                    array_path.relative_to(self.project_dir)
                except ValueError:
                    raise ValueError(f"Invalid ndarray_ref path outside project: {ref_file}")
                if not array_path.exists():
                    raise FileNotFoundError(f"Array file not found: {array_path}")
                import numpy as np
                return np.load(array_path, allow_pickle=False)

            node_id_map: Dict[str, str] = {}
            for nd in data.get("nodes", []):
                cls = self.node_classes.get(nd.get("type"))
                if not cls:
                    continue
                node = cls.from_dict(
                    nd,
                    array_loader=_array_loader,
                    project_dir=self.project_dir,
                )
                new_id = register_node(node)
                node_id_map[nd.get("id")] = new_id
                with self._nodes_lock:
                    self.nodes.append(node)
                if getattr(node.__class__, "_uses_background_init", False):
                    self._start_background_init(node)
                else:
                    restore_error = str(getattr(node, "_loading_error", "") or "")
                    try:
                        node.init()
                    except Exception as e:
                        error = str(e)
                        tb = traceback.format_exc()
                        final_error = "\n".join(
                            part for part in (restore_error, error) if part
                        )
                        node._loading_error = final_error
                        self._set_node_loading_state(new_id, False, final_error)
                        print(f"Node init failed for {node.node_type}: {e}")
                        print(tb)
                    else:
                        node._loading_error = restore_error
                        self._set_node_loading_state(new_id, False, restore_error)

            for conn in data.get("connections", []):
                fn = node_id_map.get(conn["from_node"])
                tn = node_id_map.get(conn["to_node"])
                if fn and tn:
                    add_connection(fn, conn["from_output"], tn, conn["to_input"])

            self.viewport = data.get("viewport", {"pan": {"x": 0, "y": 0}, "zoom": 1.0})
            self.set_editor_state(data.get("editor", {}))
            self.current_graph_name = name
            self.mark_current_graph_saved()
            self.snapshot_displays()
            return self.viewport
        finally:
            self._loading_project = False

    def new_graph(self, name: str | None = None) -> None:
        """Clear state and start a new empty graph."""
        self._clear_graph_state()
        self.current_graph_name = name
        self.current_graph_dirty = name is None
        self.snapshot_displays()

    def rename_graph(self, old_name: str, new_name: str) -> None:
        """Rename a graph file on disk."""
        graphs_dir = self.project_dir / GRAPH_DIR_NAME
        old_file = graphs_dir / f"{old_name}.json"
        new_file = graphs_dir / f"{new_name}.json"
        if old_file.exists():
            old_file.rename(new_file)
        if self.current_graph_name == old_name:
            self.current_graph_name = new_name

    def delete_graph(self, name: str) -> None:
        """Delete a graph file from disk."""
        graph_file = self.project_dir / GRAPH_DIR_NAME / f"{name}.json"
        if graph_file.exists():
            graph_file.unlink()
        if self.current_graph_name == name:
            self._clear_graph_state()
            self.current_graph_name = None
            self.mark_current_graph_saved()
            self.snapshot_displays()

    def clear_graph(self) -> None:
        """Clear the current graph state without changing the graph name."""
        self._clear_graph_state()
        self.mark_current_graph_dirty()
        self.snapshot_displays()

    # ── Hot Reload ───────────────────────────────────────────────────────

    def reload_node(self, node_id: str) -> Node:
        from axonforge.core.registry import _node_registry

        old = get_node(node_id)
        if not old:
            raise ValueError(f"Node not found: {node_id}")
        if not getattr(old.__class__, "dynamic", False):
            raise ValueError("Node is not dynamic")
        self._cancel_background_init(node_id)

        module_path = old.__class__.__module__
        module = sys.modules.get(module_path)
        if not module:
            raise RuntimeError(f"Module not found: {module_path}")

        importlib.reload(module)
        new_class = getattr(module, old.__class__.__name__)
        if not getattr(new_class, "dynamic", False):
            raise ValueError("Reloaded class is no longer dynamic")

        new_inputs = set(getattr(new_class, "_input_ports", {}).keys())
        new_outputs = set(getattr(new_class, "_output_ports", {}).keys())
        for conn in get_connections_for_node(node_id):
            invalid = False
            if conn["from_node"] == node_id and conn["from_output"] not in new_outputs:
                invalid = True
            if conn["to_node"] == node_id and conn["to_input"] not in new_inputs:
                invalid = True
            if invalid:
                remove_connection(
                    conn["from_node"], conn["from_output"],
                    conn["to_node"], conn["to_input"],
                )

        new_node = new_class(x=old.position.get("x", 0), y=old.position.get("y", 0))
        new_node.node_id = node_id
        new_node.name = old.name
        new_node._loading = False
        new_node._loading_error = ""

        # Only copy field values, not state values
        for field_name in getattr(old.__class__, "_fields", {}).keys():
            if hasattr(old, field_name):
                try:
                    setattr(new_node, field_name, getattr(old, field_name))
                except Exception:
                    pass

        if hasattr(old, "_output_enabled"):
            new_node._output_enabled = old._output_enabled.copy()

        with self._nodes_lock:
            for i, n in enumerate(self.nodes):
                if n.node_id == node_id:
                    self.nodes[i] = new_node
                    break

        _node_registry[node_id] = new_node
        if new_class.__name__ in self.node_classes:
            self.node_classes[new_class.__name__] = new_class

        new_node.init()
        self._set_node_loading_state(node_id, False, "")
        self.snapshot_displays()
        return new_node

    def rediscover(self) -> None:
        new_palette = rediscover_nodes()
        local_nodes = self.project_dir / "nodes"
        if local_nodes.exists():
            discover_nodes(nodes_dir=local_nodes, package=None)
            new_palette = build_node_palette()
        for category, classes in new_palette.items():
            for cls in classes:
                self.node_classes[cls.__name__] = cls
        self.node_palette = new_palette

    # ── Topology Snapshot ────────────────────────────────────────────────

    def get_topology(self) -> dict:
        with self._nodes_lock:
            nodes_copy = list(self.nodes)
        return {
            "nodes": [n.get_schema() for n in nodes_copy],
            "connections": get_connections(),
            "viewport": self.viewport,
            "editor": self.get_editor_state(),
        }

    def _normalize_editor_state(self, state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        base = dict(DEFAULT_EDITOR_STATE)
        if not isinstance(state, dict):
            return base
        merged = base.copy()
        merged.update(state)
        hz_value_raw = merged.get("max_hz_value", base["max_hz_value"])
        try:
            hz_value = int(hz_value_raw)
        except Exception:
            hz_value = base["max_hz_value"]
        merged["max_hz_value"] = max(1, min(1000, hz_value))
        drawer_w_raw = merged.get("drawer_width", base["drawer_width"])
        console_h_raw = merged.get("console_height", base["console_height"])
        try:
            drawer_w = int(drawer_w_raw)
        except Exception:
            drawer_w = base["drawer_width"]
        try:
            console_h = int(console_h_raw)
        except Exception:
            console_h = base["console_height"]
        merged["drawer_width"] = max(120, min(2000, drawer_w))
        merged["console_height"] = max(100, min(2000, console_h))
        merged["drawer_collapsed"] = bool(merged.get("drawer_collapsed", base["drawer_collapsed"]))
        merged["console_collapsed"] = bool(merged.get("console_collapsed", base["console_collapsed"]))
        merged["max_hz_enabled"] = bool(merged.get("max_hz_enabled", base["max_hz_enabled"]))
        return merged

    def _apply_editor_state_to_network(self) -> None:
        self.network.speed = float(self.editor_state.get("max_hz_value", 60.0))
        self.network.max_speed = bool(self.editor_state.get("max_hz_enabled", False))

    def set_editor_state(self, state: Dict[str, Any]) -> None:
        current = dict(self.editor_state)
        if isinstance(state, dict):
            current.update(state)
        self.editor_state = self._normalize_editor_state(current)
        self._apply_editor_state_to_network()

    def get_editor_state(self) -> Dict[str, Any]:
        return dict(self.editor_state)

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _is_type_compatible(
        output_type: Optional[Any],
        input_type: Optional[Any]
    ) -> bool:
        """
        Check if an output port can connect to an input port.

        Args:
            output_type: None (any), single Type, or list of Types
            input_type: None (any), single Type, or list of Types

        Returns:
            True if connection is allowed

        Rules:
            - None means "any type" (always compatible)
            - List means "any of these types" (OR logic)
            - Single type uses subclass checking
        """
        def normalize(t: Optional[Any]) -> List[Any]:
            """Normalize to list for uniform handling. Empty list = any type."""
            if t is None:
                return []
            if isinstance(t, (list, tuple, set)):
                return list(t)
            return [t]

        out_types = normalize(output_type)
        in_types = normalize(input_type)

        # None (any type) - always compatible
        if not out_types or not in_types:
            return True

        # Check if ANY output type is compatible with ANY input type
        for out_t in out_types:
            for in_t in in_types:
                if BridgeAPI._check_single_type(out_t, in_t):
                    return True

        return False

    @staticmethod
    def _check_single_type(output_type: Any, input_type: Any) -> bool:
        """Check compatibility between two single types."""
        # Subclass check for type objects
        if isinstance(output_type, type) and isinstance(input_type, type):
            return issubclass(output_type, input_type)
        # String comparison as fallback
        return str(output_type) == str(input_type)
