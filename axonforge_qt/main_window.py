"""
MainWindow — QMainWindow assembling header, palette, canvas, console, and graph panel.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, QPoint, QPointF, QEvent
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QPushButton, QSplitter, QCheckBox, QSpinBox, QMessageBox,
)

from .bridge import BridgeAPI
from .computation_thread import ComputationThread
from .canvas.editor_scene import EditorScene
from .canvas.editor_view import EditorView
from .panels.palette import PalettePanel
from .panels.console import ConsolePanel
from .panels.graph_panel import GraphPanel


class MainWindow(QMainWindow):
    """Top-level application window."""

    def __init__(self, bridge: BridgeAPI, ui_fps: float = 30.0) -> None:
        super().__init__()
        self.bridge = bridge
        self._ui_fps = ui_fps
        self._palette_last_width = 260
        self._palette_collapsed = True
        self._graph_panel_last_width = 200
        self._graph_panel_collapsed = True
        self._console_last_height = 220
        self._console_collapsed = False
        self._syncing_editor_state = False
        self._viewport_apply_token = 0
        self._transition_active = False
        self._window_closing = False
        self._retired_scenes: list[EditorScene] = []
        self._resize_anchor_horizontal: tuple[QPoint, QPointF] | None = None
        self._resize_anchor_vertical: tuple[QPoint, QPointF] | None = None
        self._shortcuts: list[QShortcut] = []

        self.setWindowTitle("AxonForge — Node Editor")
        self.setMinimumSize(1200, 700)
        self._build_menu_bar()

        # ── Central widget ───────────────────────────────────────────────
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ── Header bar ───────────────────────────────────────────────────
        self._header_bar = self._build_header()
        root_layout.addWidget(self._header_bar)

        # ── Body: palette + canvas + graph panel ─────────────────────────
        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        self._palette = PalettePanel(bridge)
        self._scene = EditorScene(bridge)
        self._view = EditorView(self._scene)
        self._graph_panel = GraphPanel(bridge)

        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.addWidget(self._palette)
        self._splitter.addWidget(self._view)
        self._splitter.addWidget(self._graph_panel)
        self._splitter.setStretchFactor(0, 0)  # palette: fixed
        self._splitter.setStretchFactor(1, 1)  # canvas: stretches
        self._splitter.setStretchFactor(2, 0)  # graph panel: fixed
        self._splitter.splitterMoved.connect(self._on_splitter_moved)
        body_layout.addWidget(self._splitter)

        # Connect graph panel signals
        self._graph_panel.graph_switch_requested.connect(self._switch_graph)
        self._graph_panel.graph_new_requested.connect(lambda: self._new_graph(None))
        self._graph_panel.graph_delete_requested.connect(self._delete_graph)
        self._graph_panel.graph_rename_requested.connect(self._rename_graph)

        # ── Console panel ────────────────────────────────────────────────
        self._console = ConsolePanel(bridge)
        self._console.setVisible(True)

        self._main_splitter = QSplitter(Qt.Orientation.Vertical)
        self._main_splitter.addWidget(body)
        self._main_splitter.addWidget(self._console)
        self._main_splitter.setStretchFactor(0, 1)
        self._main_splitter.setStretchFactor(1, 0)
        self._main_splitter.splitterMoved.connect(self._on_main_splitter_moved)
        root_layout.addWidget(self._main_splitter)

        # Enable anchor-preserving behavior while dragging splitter handles.
        h_handle = self._splitter.handle(1)
        if h_handle is not None:
            h_handle.installEventFilter(self)
        v_handle = self._main_splitter.handle(1)
        if v_handle is not None:
            v_handle.installEventFilter(self)

        # Start with drawers collapsed.
        self._set_palette_collapsed(True)
        self._set_graph_panel_collapsed(True)

        # ── Computation thread ───────────────────────────────────────────
        self._comp_thread = ComputationThread(bridge)
        self._comp_thread.network_error.connect(self._on_network_error)
        self._comp_thread.start()

        # ── UI refresh timer ─────────────────────────────────────────────
        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self._on_refresh_tick)
        self._refresh_timer.start(int(1000.0 / self._ui_fps))

        # ── Keyboard shortcuts ───────────────────────────────────────────
        self._add_shortcut("Ctrl+Space", self._toggle_network)
        self._add_shortcut("Ctrl+Return", self._step_network)
        self._add_shortcut("Shift+W", self._toggle_console)
        self._add_shortcut("Shift+Q", self._toggle_palette_drawer)
        self._add_shortcut("Shift+E", self._toggle_graph_panel)
        self._add_shortcut("Ctrl+N", lambda: self._new_graph(None))
        self._add_shortcut("Ctrl+S", self._save_graph)

        # ── Load initial topology ────────────────────────────────────────
        self._load_initial_state()

    # ── Header ───────────────────────────────────────────────────────────

    def _add_shortcut(self, sequence: str, callback) -> None:
        shortcut = QShortcut(QKeySequence(sequence), self)
        shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        shortcut.activated.connect(callback)
        self._shortcuts.append(shortcut)

    def _build_menu_bar(self) -> None:
        """Create application menu bar."""
        self.menuBar().hide()

    def _build_header(self) -> QWidget:
        header = QWidget()
        header.setObjectName("header_bar")
        header.setFixedHeight(44)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(12, 0, 12, 0)
        layout.setSpacing(10)

        # Title
        title = QLabel("AxonForge")
        title.setObjectName("app_title")
        layout.addWidget(title)
        self._workspace_label = QLabel("")
        self._workspace_label.setObjectName("workspace_label")
        layout.addWidget(self._workspace_label)
        self._workspace_dirty_indicator = QWidget()
        self._workspace_dirty_indicator.setObjectName("workspace_dirty_indicator")
        self._workspace_dirty_indicator.setFixedSize(8, 8)
        self._workspace_dirty_indicator.setVisible(False)
        layout.addWidget(self._workspace_dirty_indicator)

        layout.addStretch()

        # Network controls
        self._toggle_btn = QPushButton("Start")
        self._toggle_btn.setObjectName("network_toggle")
        self._toggle_btn.clicked.connect(self._toggle_network)
        layout.addWidget(self._toggle_btn)

        self._iterate_btn = QPushButton("Iterate")
        self._iterate_btn.setObjectName("network_iterate")
        self._iterate_btn.setToolTip("Execute one step")
        self._iterate_btn.clicked.connect(self._step_network)
        layout.addWidget(self._iterate_btn)

        speed_label = QLabel("Max Hz")
        speed_label.setObjectName("speed_label")
        layout.addWidget(speed_label)

        self._max_speed_checkbox = QCheckBox("Max")
        self._max_speed_checkbox.setObjectName("network_max_speed_checkbox")
        self._max_speed_checkbox.setChecked(False)
        self._max_speed_checkbox.toggled.connect(self._on_max_speed_toggled)
        layout.addWidget(self._max_speed_checkbox)

        self._speed_input = QSpinBox()
        self._speed_input.setObjectName("network_speed_input")
        self._speed_input.setRange(1, 1000)
        self._speed_input.setValue(60)
        self._speed_input.valueChanged.connect(self._on_speed_changed)
        layout.addWidget(self._speed_input)

        self._actual_hz = QLabel("0.0 Hz")
        self._actual_hz.setObjectName("actual_hz")
        layout.addWidget(self._actual_hz)

        return header

    # ── Initial state ────────────────────────────────────────────────────

    def _load_initial_state(self) -> None:
        topology = self.bridge.get_topology()
        self._scene.rebuild_from_topology(topology)
        self._apply_viewport_state(topology.get("viewport", {}))
        self._update_header_label()
        self._apply_editor_state()

    def _update_header_label(self) -> None:
        project = self.bridge.project_name
        graph = self.bridge.current_graph_display_name
        self._workspace_label.setText(f"{project} / {graph}")
        self._workspace_dirty_indicator.setVisible(self.bridge.current_graph_has_unsaved_state())

    # ── UI refresh ───────────────────────────────────────────────────────

    def _on_refresh_tick(self) -> None:
        if self._transition_active or self._window_closing:
            return
        self.bridge.poll_background_init_jobs()
        self._update_header_label()
        buf = self.bridge.read_display_buffer()
        if buf is not None:
            self._scene.update_displays(buf)

        loading_states = self.bridge.get_node_loading_states()
        for node_item in self._scene._node_items.values():
            state = loading_states.get(node_item.node_id, {})
            node_item.set_loading_state(bool(state.get("loading", False)))

        # Update error states on nodes (network/process errors + init errors)
        failed_nodes = self.bridge.network.failed_nodes
        failed_node_errors = self.bridge.network.failed_node_errors
        for node_item in self._scene._node_items.values():
            state = loading_states.get(node_item.node_id, {})
            init_error = str(state.get("error", "") or "")
            process_error = failed_node_errors.get(node_item.node_id, "")
            has_error = bool(process_error) or (node_item.node_id in failed_nodes) or bool(init_error)
            message = process_error or init_error
            node_item.set_error_state(has_error, message)

        # Update network state UI
        state = self.bridge.get_network_state()
        running = state["running"]
        max_speed = bool(state.get("max_speed", False))
        speed = int(state.get("speed", 60))
        self._toggle_btn.setText("Stop" if running else "Start")
        self._toggle_btn.setProperty("running", "true" if running else "false")
        self._toggle_btn.style().unpolish(self._toggle_btn)
        self._toggle_btn.style().polish(self._toggle_btn)
        self._actual_hz.setText(f"{state['actual_hz']:.1f} Hz")
        if self._speed_input.value() != speed:
            self._speed_input.blockSignals(True)
            self._speed_input.setValue(speed)
            self._speed_input.blockSignals(False)
        if self._max_speed_checkbox.isChecked() != max_speed:
            self._max_speed_checkbox.blockSignals(True)
            self._max_speed_checkbox.setChecked(max_speed)
            self._max_speed_checkbox.blockSignals(False)
        self._speed_input.setEnabled(not max_speed)

        # Update header bar style based on running state
        self._header_bar.setProperty("running", "true" if running else "false")
        self._header_bar.style().unpolish(self._header_bar)
        self._header_bar.style().polish(self._header_bar)

    # ── Network controls ─────────────────────────────────────────────────

    def _on_splitter_moved(self, pos: int, index: int) -> None:
        sizes = self._splitter.sizes()
        if not sizes:
            return
        palette_width = int(sizes[0])
        if palette_width > 40:
            self._palette_last_width = palette_width
            self._palette_collapsed = False
            if not self._syncing_editor_state:
                self.bridge.set_editor_state({
                    "drawer_collapsed": False,
                    "drawer_width": palette_width,
                })
        elif palette_width <= 2:
            self._palette_collapsed = True
            if not self._syncing_editor_state:
                self.bridge.set_editor_state({"drawer_collapsed": True})
        if self._resize_anchor_horizontal is not None:
            ag, asc = self._resize_anchor_horizontal
            self._restore_view_anchor(ag, asc)

    def _on_main_splitter_moved(self, pos: int, index: int) -> None:
        sizes = self._main_splitter.sizes()
        if len(sizes) < 2:
            return
        console_height = int(sizes[1])
        if console_height > 40:
            self._console_last_height = console_height
            self._console_collapsed = False
            if not self._syncing_editor_state:
                self.bridge.set_editor_state({
                    "console_collapsed": False,
                    "console_height": console_height,
                })
        elif console_height <= 2:
            self._console_collapsed = True
            if not self._syncing_editor_state:
                self.bridge.set_editor_state({"console_collapsed": True})
        if self._resize_anchor_vertical is not None:
            ag, asc = self._resize_anchor_vertical
            self._restore_view_anchor(ag, asc)

    def _set_palette_collapsed(self, collapsed: bool) -> None:
        anchor_global, anchor_scene = self._capture_view_anchor()
        sizes_now = self._splitter.sizes()
        total = max(sum(sizes_now), self._splitter.width(), 1)
        if collapsed:
            sizes = self._splitter.sizes()
            if sizes and sizes[0] > 40:
                self._palette_last_width = int(sizes[0])
            graph_w = sizes[2] if len(sizes) > 2 else 0
            self._splitter.setSizes([0, total - graph_w, graph_w])
            self._palette_collapsed = True
            QTimer.singleShot(
                0,
                lambda ag=anchor_global, asc=anchor_scene: self._restore_view_anchor(ag, asc),
            )
            if not self._syncing_editor_state:
                self.bridge.set_editor_state({
                    "drawer_collapsed": True,
                    "drawer_width": int(self._palette_last_width),
                })
            return

        target = max(200, min(self._palette_last_width, int(total * 0.45)))
        graph_w = sizes_now[2] if len(sizes_now) > 2 else 0
        self._splitter.setSizes([target, max(1, total - target - graph_w), graph_w])
        self._palette_collapsed = False
        QTimer.singleShot(
            0,
            lambda ag=anchor_global, asc=anchor_scene: self._restore_view_anchor(ag, asc),
        )
        if not self._syncing_editor_state:
            self.bridge.set_editor_state({
                "drawer_collapsed": False,
                "drawer_width": int(target),
            })

    def _toggle_palette_drawer(self) -> None:
        self._set_palette_collapsed(not self._palette_collapsed)

    def _set_graph_panel_collapsed(self, collapsed: bool) -> None:
        anchor_global, anchor_scene = self._capture_view_anchor()
        sizes_now = self._splitter.sizes()
        total = max(sum(sizes_now), self._splitter.width(), 1)
        palette_w = sizes_now[0] if sizes_now else 0

        if collapsed:
            if len(sizes_now) > 2 and sizes_now[2] > 40:
                self._graph_panel_last_width = int(sizes_now[2])
            self._splitter.setSizes([palette_w, total - palette_w, 0])
            self._graph_panel_collapsed = True
        else:
            target = max(160, min(self._graph_panel_last_width, int(total * 0.3)))
            self._splitter.setSizes([palette_w, max(1, total - palette_w - target), target])
            self._graph_panel_collapsed = False

        QTimer.singleShot(
            0,
            lambda ag=anchor_global, asc=anchor_scene: self._restore_view_anchor(ag, asc),
        )

    def _toggle_graph_panel(self) -> None:
        self._set_graph_panel_collapsed(not self._graph_panel_collapsed)

    def _capture_view_anchor(self) -> tuple[QPoint, QPointF]:
        """Capture a stable anchor in scene space before layout changes."""
        viewport = self._view.viewport()
        local = viewport.rect().center()
        global_pos = viewport.mapToGlobal(local)
        scene_pos = self._view.mapToScene(local)
        return global_pos, scene_pos

    def _restore_view_anchor(self, global_pos: QPoint, scene_pos: QPointF) -> None:
        """Restore the captured scene anchor at the same screen position."""
        viewport = self._view.viewport()
        local_after = viewport.mapFromGlobal(global_pos)
        if not viewport.rect().contains(local_after):
            local_after = viewport.rect().center()
        scene_after = self._view.mapToScene(local_after)
        delta = scene_after - scene_pos
        self._view.translate(delta.x(), delta.y())

    def _toggle_network(self) -> None:
        if self.bridge.network.running:
            self.bridge.stop_network()
        else:
            # Clear failed nodes when starting fresh
            self.bridge.network.clear_failed_nodes()
            self.bridge.start_network()

    def _step_network(self) -> None:
        self.bridge.step_network()

    def _on_speed_changed(self, value: int) -> None:
        self.bridge.set_network_speed(float(value))
        if not self._syncing_editor_state:
            self.bridge.set_editor_state({"max_hz_value": int(value)})

    def _on_max_speed_toggled(self, enabled: bool) -> None:
        self._speed_input.setEnabled(not enabled)
        self.bridge.set_network_max_speed(enabled)
        if not self._syncing_editor_state:
            self.bridge.set_editor_state({"max_hz_enabled": bool(enabled)})

    def _on_network_error(self, node_id: str, node_name: str, error: str, tb: str) -> None:
        error_msg = f"Network error in '{node_name}': {error}"
        print(error_msg)
        print(tb)
        if hasattr(self, '_console'):
            self._console.append_message(error_msg, is_error=True)
            self._console.append_message(tb, is_error=True)
        node_item = self._scene._node_items.get(node_id)
        if node_item:
            node_item.set_error_state(True, error)

    def _toggle_console(self) -> None:
        """Toggle console panel visibility with Shift+W shortcut."""
        if hasattr(self, '_console'):
            self._set_console_collapsed(not self._console_collapsed)

    # ── Graph switching ──────────────────────────────────────────────────

    def _begin_transition(self) -> None:
        """Stop background activity before tearing down graph state."""
        if self._transition_active:
            return
        self._transition_active = True
        self._viewport_apply_token += 1
        self._refresh_timer.stop()
        self._refresh_timer.blockSignals(True)
        if self._comp_thread.isRunning():
            self._comp_thread.request_stop()
            self._comp_thread.wait()

    def _end_transition(self) -> None:
        """Restart background activity after graph state is ready."""
        if not self._transition_active:
            return
        if self._window_closing:
            old_thread = self._comp_thread
            self._transition_active = False
            try:
                old_thread.deleteLater()
            except Exception:
                pass
            return
        old_thread = self._comp_thread
        self._comp_thread = ComputationThread(self.bridge)
        self._comp_thread.network_error.connect(self._on_network_error)
        self._comp_thread.start()
        self._transition_active = False
        self._refresh_timer.blockSignals(False)
        self._refresh_timer.start(int(1000.0 / self._ui_fps))
        try:
            old_thread.deleteLater()
        except Exception:
            pass

    def _rebuild_scene(self, topology: dict) -> None:
        """Swap in a fresh scene, then dispose the old one after detaching it."""
        self._view.setUpdatesEnabled(False)
        self._view.viewport().setUpdatesEnabled(False)

        old_scene = self._scene
        new_scene = EditorScene(self.bridge)
        new_scene.rebuild_from_topology(topology)
        self._scene = new_scene
        self._view.set_editor_scene(new_scene)

        try:
            old_scene.deactivate()
        except Exception:
            pass
        self._retired_scenes.append(old_scene)

        self._view.viewport().setUpdatesEnabled(True)
        self._view.setUpdatesEnabled(True)
        self._view.viewport().update()
        self._view.update()

    def _autosave_current_graph(self) -> None:
        """Save the current graph if it has a name."""
        if self.bridge.current_graph_name is None:
            return
        if not self.bridge.current_graph_dirty:
            return
        self._snapshot_viewport_state()
        self._snapshot_editor_state()
        self.bridge.save_graph(self.bridge.current_graph_name)

    def _show_graph_name_entry(self) -> None:
        self._set_graph_panel_collapsed(False)
        self._graph_panel.refresh()
        QTimer.singleShot(0, self._graph_panel.begin_rename_current_graph)

    def _confirm_discard_unnamed_graph(self, action_text: str) -> bool:
        if self.bridge.current_graph_name is not None:
            return True
        if not self.bridge.current_graph_has_unsaved_state():
            return True
        result = QMessageBox.question(
            self,
            "Unsaved Graph",
            (
                "The current graph has no name and has not been saved.\n\n"
                f"{action_text}"
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if result == QMessageBox.StandardButton.Yes:
            return True
        self._show_graph_name_entry()
        return False

    def _save_graph_with_name(self, name: str) -> None:
        self._snapshot_viewport_state()
        self._snapshot_editor_state()
        self.bridge.save_graph(name)
        self._update_header_label()
        self._graph_panel.refresh()

    def _switch_graph(self, name: str) -> None:
        """Switch to a different graph."""
        if not self._confirm_discard_unnamed_graph("Switch graphs anyway?"):
            return
        try:
            self._autosave_current_graph()
        except Exception as e:
            print(f"Failed to save graph before switch: {e}")
        self._begin_transition()
        try:
            self.bridge.load_graph(name)
            topology = self.bridge.get_topology()
            self._rebuild_scene(topology)
            self._schedule_viewport_state(topology.get("viewport", {}))
            self._update_header_label()
            self._apply_editor_state()
        except Exception as e:
            print(f"Failed to load graph: {e}")
        finally:
            self._end_transition()
        self._graph_panel.refresh()

    def _new_graph(self, name: str | None = None) -> None:
        """Create a new empty graph and switch to it."""
        if not self._confirm_discard_unnamed_graph("Create another graph anyway?"):
            return
        try:
            self._autosave_current_graph()
        except Exception as e:
            print(f"Failed to save graph before creating new: {e}")
        self._begin_transition()
        try:
            self.bridge.new_graph(name)
            topology = self.bridge.get_topology()
            self._rebuild_scene(topology)
            self._update_header_label()
        finally:
            self._end_transition()
        self._graph_panel.refresh()

    def _delete_graph(self, name: str) -> None:
        """Delete a graph file."""
        is_active = (name == self.bridge.current_graph_name)
        if is_active:
            self._begin_transition()
        try:
            self.bridge.delete_graph(name)
            if is_active:
                topology = self.bridge.get_topology()
                self._rebuild_scene(topology)
                self._update_header_label()
        finally:
            if is_active:
                self._end_transition()
        self._graph_panel.refresh()

    def _rename_graph(self, old_name: object, new_name: str) -> None:
        """Rename a graph file."""
        clean_name = new_name.strip()
        if not clean_name:
            return
        existing = set(self.bridge.list_graphs())
        if clean_name in existing and clean_name != old_name:
            QMessageBox.warning(
                self,
                "Graph Name In Use",
                f"A graph named '{clean_name}' already exists.",
            )
            return

        if old_name is None:
            self._save_graph_with_name(clean_name)
        elif old_name == self.bridge.current_graph_name:
            self._save_graph_with_name(clean_name)
            if old_name != clean_name:
                self.bridge.delete_graph(old_name)
        else:
            self.bridge.rename_graph(old_name, clean_name)
        self._update_header_label()
        self._graph_panel.refresh()

    def _save_graph(self) -> None:
        """Save the current graph (Ctrl+S)."""
        name = self.bridge.current_graph_name
        if name is None:
            self._show_graph_name_entry()
            return
        self._save_graph_with_name(name)

    # ── Viewport & editor state ──────────────────────────────────────────

    def _snapshot_viewport_state(self) -> None:
        viewport = self._view.viewport()
        center_local = viewport.rect().center()
        center_scene = self._view.mapToScene(center_local)
        self.bridge.viewport = {
            "pan": {"x": float(center_scene.x()), "y": float(center_scene.y())},
            "zoom": float(self._view.current_zoom),
        }

    def _schedule_viewport_state(self, viewport_state: dict) -> None:
        """Apply viewport state after the view/scene swap has settled."""
        self._viewport_apply_token += 1
        token = self._viewport_apply_token

        def _apply() -> None:
            if token != self._viewport_apply_token:
                return
            if self._view is None or self._scene is None:
                return
            if self._view.scene() is not self._scene:
                QTimer.singleShot(0, _apply)
                return
            if not self._view.isVisible():
                QTimer.singleShot(50, _apply)
                return
            self._apply_viewport_state(viewport_state)

        QTimer.singleShot(0, _apply)

    def _apply_viewport_state(self, viewport_state: dict) -> None:
        pan = viewport_state.get("pan", {}) if isinstance(viewport_state, dict) else {}
        try:
            pan_x = float(pan.get("x", 0.0))
            pan_y = float(pan.get("y", 0.0))
        except Exception:
            pan_x, pan_y = 0.0, 0.0
        try:
            zoom = float(viewport_state.get("zoom", 1.0))
        except Exception:
            zoom = 1.0
        self._view.reset_view()
        self._view.set_zoom(zoom)
        self._view.centerOn(pan_x, pan_y)

    def _snapshot_editor_state(self) -> None:
        drawer_sizes = self._splitter.sizes()
        drawer_width = int(drawer_sizes[0]) if drawer_sizes else int(self._palette_last_width)
        if drawer_width > 40:
            self._palette_last_width = drawer_width
        console_sizes = self._main_splitter.sizes()
        console_height = int(console_sizes[1]) if len(console_sizes) > 1 else int(self._console_last_height)
        if console_height > 40:
            self._console_last_height = console_height
        self.bridge.set_editor_state({
            "drawer_collapsed": bool(self._palette_collapsed),
            "drawer_width": int(self._palette_last_width),
            "console_collapsed": bool(self._console_collapsed),
            "console_height": int(self._console_last_height),
            "max_hz_enabled": bool(self._max_speed_checkbox.isChecked()),
            "max_hz_value": int(self._speed_input.value()),
        })

    def _apply_editor_state(self) -> None:
        state = self.bridge.get_editor_state()
        speed_value = int(state.get("max_hz_value", 60))
        max_hz_enabled = bool(state.get("max_hz_enabled", False))
        drawer_collapsed = bool(state.get("drawer_collapsed", True))
        console_collapsed = bool(state.get("console_collapsed", False))
        self._palette_last_width = int(state.get("drawer_width", self._palette_last_width))
        self._console_last_height = int(state.get("console_height", self._console_last_height))

        self._syncing_editor_state = True
        try:
            self._speed_input.blockSignals(True)
            self._speed_input.setValue(speed_value)
            self._speed_input.blockSignals(False)

            self._max_speed_checkbox.blockSignals(True)
            self._max_speed_checkbox.setChecked(max_hz_enabled)
            self._max_speed_checkbox.blockSignals(False)
            self._speed_input.setEnabled(not max_hz_enabled)

            self._set_palette_collapsed(drawer_collapsed)
            self._set_console_collapsed(console_collapsed)
        finally:
            self._syncing_editor_state = False

    def _set_console_collapsed(self, collapsed: bool) -> None:
        anchor_global, anchor_scene = self._capture_view_anchor()
        sizes_now = self._main_splitter.sizes()
        total = max(sum(sizes_now), self._main_splitter.height(), 1)
        if collapsed:
            sizes = self._main_splitter.sizes()
            if len(sizes) > 1 and sizes[1] > 40:
                self._console_last_height = int(sizes[1])
            self._console.setVisible(True)
            self._main_splitter.setSizes([total, 0])
            self._console.setVisible(False)
            self._console_collapsed = True
            if not self._syncing_editor_state:
                self.bridge.set_editor_state({
                    "console_collapsed": True,
                    "console_height": int(self._console_last_height),
                })
            QTimer.singleShot(
                0,
                lambda ag=anchor_global, asc=anchor_scene: self._restore_view_anchor(ag, asc),
            )
            return

        self._console.setVisible(True)
        target = max(100, min(self._console_last_height, int(total * 0.45)))
        self._main_splitter.setSizes([max(1, total - target), target])
        self._console_collapsed = False
        if not self._syncing_editor_state:
            self.bridge.set_editor_state({
                "console_collapsed": False,
                "console_height": int(target),
            })
        QTimer.singleShot(
            0,
            lambda ag=anchor_global, asc=anchor_scene: self._restore_view_anchor(ag, asc),
        )

    def eventFilter(self, watched, event):
        event_type = event.type()
        h_handle = self._splitter.handle(1)
        v_handle = self._main_splitter.handle(1)

        if watched is h_handle:
            if event_type == QEvent.Type.MouseButtonPress:
                self._resize_anchor_horizontal = self._capture_view_anchor()
            elif event_type == QEvent.Type.MouseButtonRelease:
                self._resize_anchor_horizontal = None
        elif watched is v_handle:
            if event_type == QEvent.Type.MouseButtonPress:
                self._resize_anchor_vertical = self._capture_view_anchor()
            elif event_type == QEvent.Type.MouseButtonRelease:
                self._resize_anchor_vertical = None

        return super().eventFilter(watched, event)

    # ── Cleanup ──────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        self._window_closing = True
        if not self._confirm_discard_unnamed_graph("Quit anyway?"):
            self._window_closing = False
            event.ignore()
            return

        if self.bridge.current_graph_name is not None:
            try:
                self._save_graph_with_name(self.bridge.current_graph_name)
            except Exception as e:
                print(f"Failed to save graph on close: {e}")

        self._begin_transition()
        self.bridge.shutdown_background_workers()
        super().closeEvent(event)
