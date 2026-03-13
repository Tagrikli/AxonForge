"""
BridgeAPI — Direct Python interface replacing the REST/WebSocket server layer.

All operations that the web UI performed via HTTP are now synchronous method calls.
The UI calls these directly; no serialization, no network overhead.
"""

import json
import importlib
import multiprocessing
import sys
import threading
import traceback
import uuid
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from PySide6.QtCore import QStandardPaths

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
WORKSPACES_DIR = Path("workspaces")
PROJECT_FILENAME = "project.json"
PROJECT_DATA_DIRNAME = "data"
APP_CONFIG_FILENAME = "settings.json"
DEFAULT_EDITOR_STATE = {
    "drawer_collapsed": True,
    "drawer_width": 260,
    "console_collapsed": False,
    "console_height": 220,
    "max_hz_enabled": False,
    "max_hz_value": 60,
}


def _run_node_init_process(module_name: str, class_name: str, node_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process worker for running Node.init() outside the UI process.

    Returns:
        {"ok": bool, "state": dict, "error": str, "traceback": str}
    """
    try:
        module = importlib.import_module(module_name)
        node_class = getattr(module, class_name)
        node = node_class.from_dict(node_data)
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

    def __init__(self) -> None:
        self.nodes: List[Node] = []
        self._nodes_lock = threading.Lock()
        self.network: Network = Network()
        self.node_classes: Dict[str, Type[Node]] = {}
        self.node_palette: Dict[str, List[Type[Node]]] = {}
        self.viewport: Dict[str, Any] = {"pan": {"x": 0.0, "y": 0.0}, "zoom": 1.0}
        self.editor_state: Dict[str, Any] = dict(DEFAULT_EDITOR_STATE)
        self.current_workspace: Optional[str] = None
        self.current_project_dir: Optional[Path] = None
        self._recent_projects: List[str] = []
        self._config_dir = self._ensure_config_dir()
        self._config_path = self._config_dir / APP_CONFIG_FILENAME
        self._load_app_config()
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
        """Discover nodes and build palette."""
        discover_nodes()
        palette = build_node_palette()
        for category, class_list in palette.items():
            for cls in class_list:
                self.node_classes[cls.__name__] = cls
        self.node_palette = palette

    def _ensure_config_dir(self) -> Path:
        """Resolve and create the per-user app config directory."""
        base = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.GenericConfigLocation
        )
        if not base:
            base = str(Path.home() / ".config")
        config_dir = Path(base) / APP_NAME
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir

    def _load_app_config(self) -> None:
        """Load persisted UI/app settings."""
        if not self._config_path.exists():
            return
        try:
            with open(self._config_path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Warning: failed to load app config: {e}")
            return

        recent = data.get("recent_projects", [])
        if isinstance(recent, list):
            self._recent_projects = [str(p) for p in recent if str(p).strip()]

    def _save_app_config(self) -> None:
        """Persist UI/app settings (recent projects, etc.)."""
        payload = {
            "version": 1,
            "recent_projects": self._recent_projects[:20],
        }
        try:
            self._config_dir.mkdir(parents=True, exist_ok=True)
            with open(self._config_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            print(f"Warning: failed to save app config: {e}")

    def get_config_dir(self) -> str:
        return str(self._config_dir)

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
                # Include display configuration from descriptor
                output_config = {}
                if hasattr(descriptor, 'color_mode'):
                    output_config['color_mode'] = descriptor.color_mode
                # Include scale configuration for bar/line charts
                if hasattr(descriptor, 'scale_mode'):
                    output_config['scale_mode'] = descriptor.scale_mode
                    output_config['scale_min'] = descriptor.scale_min
                    output_config['scale_max'] = descriptor.scale_max
                outputs[name] = {
                    'value': val,
                    'config': output_config
                }
            buf[node.node_id] = outputs
        with self._buffer_lock:
            self._display_buffer = buf
            self._dirty = True

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
        node._loading_error = ""
        self._set_node_loading_state(node_id, True, "")
        payload = node.to_dict()
        future = self._init_executor.submit(
            _run_node_init_process,
            node.__class__.__module__,
            node.__class__.__name__,
            payload,
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

            node._loading = False
            node._loading_error = error
            self._set_node_loading_state(node_id, False, error)
            has_display_update = True

        if has_display_update:
            self.snapshot_displays()

    def shutdown_background_workers(self) -> None:
        """Stop worker processes used by background init."""
        with self._init_jobs_lock:
            self._pending_init_jobs.clear()
        self._init_executor.shutdown(wait=False, cancel_futures=True)

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

    # ── Properties & Actions ─────────────────────────────────────────────

    def set_property(self, node_id: str, prop_key: str, value: Any) -> None:
        node = get_node(node_id)
        if not node:
            raise ValueError(f"Node not found: {node_id}")
        setattr(node, prop_key, value)
        # Only propagate when network is not running (let running network handle processing)
        if self.network and not self.network.running:
            try:
                self.network.propagate_from_node(node_id, recompute_start=True)
            except NetworkError as e:
                # Error is stored in network.last_error and node is tracked in failed_nodes
                print(f"Property set error in '{node_id}.{prop_key}': {e}")
                if e.traceback:
                    print(e.traceback)
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
        self.snapshot_displays()
        return result

    def set_output_enabled(self, node_id: str, output_key: str, enabled: bool) -> None:
        node = get_node(node_id)
        if not node:
            raise ValueError(f"Node not found: {node_id}")
        if not hasattr(node, "_output_enabled") or node._output_enabled is None:
            node._output_enabled = {}
        node._output_enabled[output_key] = bool(enabled)

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

    # ── Projects ─────────────────────────────────────────────────────────

    @property
    def current_project_name(self) -> str:
        if self.current_project_dir:
            return self.current_project_dir.name
        return "unsaved"

    def _track_recent_project(self, project_dir: Path) -> None:
        resolved = str(project_dir.resolve())
        self._recent_projects = [p for p in self._recent_projects if p != resolved]
        self._recent_projects.insert(0, resolved)
        self._recent_projects = self._recent_projects[:20]
        self._save_app_config()

    def list_recent_projects(self) -> List[str]:
        kept: List[str] = []
        for path in self._recent_projects:
            p = Path(path)
            if (p / PROJECT_FILENAME).exists():
                kept.append(str(p))
        if kept != self._recent_projects:
            self._recent_projects = kept
            self._save_app_config()
        else:
            self._recent_projects = kept
        return list(self._recent_projects)

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

    def new_project(self, project_dir: str | Path) -> None:
        project_path = Path(project_dir).expanduser().resolve()
        project_path.mkdir(parents=True, exist_ok=True)
        self._clear_graph_state()
        self.current_project_dir = project_path
        self.current_workspace = project_path.name
        self._track_recent_project(project_path)
        self.snapshot_displays()

    def save_project(self, project_dir: Optional[str | Path] = None) -> None:
        target_dir: Optional[Path]
        if project_dir is not None:
            target_dir = Path(project_dir).expanduser().resolve()
        else:
            target_dir = self.current_project_dir
        if target_dir is None:
            raise ValueError("Project directory is not set")

        target_dir.mkdir(parents=True, exist_ok=True)
        data_dir = target_dir / PROJECT_DATA_DIRNAME
        data_dir.mkdir(parents=True, exist_ok=True)

        def _array_serializer(array: Any) -> Dict[str, Any]:
            if not hasattr(array, "shape"):
                return {"_type": "unsupported_array"}
            array_name = f"{uuid.uuid4().hex}.npy"
            array_rel_path = Path(PROJECT_DATA_DIRNAME) / array_name
            array_abs_path = target_dir / array_rel_path
            import numpy as np
            np.save(array_abs_path, array)
            return {
                "_type": "ndarray_ref",
                "file": str(array_rel_path.as_posix()),
            }

        with self._nodes_lock:
            nodes_copy = list(self.nodes)
        payload = {
            "version": 4,
            "kind": "project",
            "project_name": target_dir.name,
            "viewport": self.viewport,
            "editor": self.get_editor_state(),
            "nodes": [n.to_dict(array_serializer=_array_serializer) for n in nodes_copy],
            "connections": get_connections(),
        }
        project_file = target_dir / PROJECT_FILENAME
        with open(project_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        self.current_project_dir = target_dir
        self.current_workspace = target_dir.name
        self._track_recent_project(target_dir)

    def load_project(self, project_dir: str | Path) -> dict:
        self._loading_project = True
        try:
            project_path = Path(project_dir).expanduser().resolve()
            project_file = project_path / PROJECT_FILENAME
            if not project_file.exists():
                raise FileNotFoundError(f"Project not found: {project_file}")

            with open(project_file, encoding="utf-8") as f:
                data = json.load(f)

            self._clear_graph_state()

            def _array_loader(ref: dict) -> Any:
                ref_file = str(ref.get("file", "")).strip()
                if not ref_file:
                    raise ValueError("Invalid ndarray_ref with empty file")
                array_path = (project_path / ref_file).resolve()
                try:
                    array_path.relative_to(project_path)
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
                node = cls.from_dict(nd, array_loader=_array_loader)
                new_id = register_node(node)
                node_id_map[nd.get("id")] = new_id
                with self._nodes_lock:
                    self.nodes.append(node)
                if getattr(node.__class__, "_uses_background_init", False):
                    self._start_background_init(node)
                else:
                    try:
                        node.init()
                    except Exception as e:
                        error = str(e)
                        tb = traceback.format_exc()
                        node._loading_error = error
                        self._set_node_loading_state(new_id, False, error)
                        print(f"Node init failed for {node.node_type}: {e}")
                        print(tb)
                    else:
                        node._loading_error = ""
                        self._set_node_loading_state(new_id, False, "")

            for conn in data.get("connections", []):
                fn = node_id_map.get(conn["from_node"])
                tn = node_id_map.get(conn["to_node"])
                if fn and tn:
                    add_connection(fn, conn["from_output"], tn, conn["to_input"])

            self.viewport = data.get("viewport", {"pan": {"x": 0, "y": 0}, "zoom": 1.0})
            self.set_editor_state(data.get("editor", {}))
            self.current_project_dir = project_path
            self.current_workspace = project_path.name
            self._track_recent_project(project_path)
            self.snapshot_displays()
            return self.viewport
        finally:
            self._loading_project = False

    # Legacy compatibility wrappers (workspace -> project)
    def list_workspaces(self) -> List[dict]:
        result = []
        for project_path in self.list_recent_projects():
            path = Path(project_path) / PROJECT_FILENAME
            stat = path.stat()
            result.append({
                "name": Path(project_path).name,
                "created": stat.st_ctime,
                "modified": stat.st_mtime,
            })
        return result

    def save_workspace(self, name: str) -> None:
        WORKSPACES_DIR.mkdir(parents=True, exist_ok=True)
        target = WORKSPACES_DIR / name
        self.save_project(target)

    def load_workspace(self, name: str) -> dict:
        target = WORKSPACES_DIR / name
        return self.load_project(target)

    def delete_workspace(self, name: str) -> None:
        target = (WORKSPACES_DIR / name / PROJECT_FILENAME)
        if not target.exists():
            raise FileNotFoundError(f"Workspace not found: {name}")
        target.unlink()

    def clear_workspace(self) -> None:
        self._clear_graph_state()
        self.current_project_dir = None
        self.current_workspace = None
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

        # Only copy property values, not store values
        for prop_name in getattr(old.__class__, "_properties", {}).keys():
            if hasattr(old, prop_name):
                try:
                    setattr(new_node, prop_name, getattr(old, prop_name))
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
