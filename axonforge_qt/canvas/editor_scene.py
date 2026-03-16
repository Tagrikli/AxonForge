"""
EditorScene — QGraphicsScene that manages nodes, connections, and interaction.
"""

from __future__ import annotations

from typing import Dict, List, Optional, TYPE_CHECKING

from PySide6.QtCore import Qt, QPointF, QRectF, Signal, QLineF, QTimer
from PySide6.QtGui import QCursor, QPen, QColor
from PySide6.QtWidgets import QGraphicsScene, QGraphicsSceneMouseEvent, QGraphicsProxyWidget, QApplication

from .node_item import NodeItem
from .port_item import PortItem, PortHitZoneItem
from .connection_item import ConnectionItem, ConnectionPreviewItem
from .z_layers import Z_NODE

if TYPE_CHECKING:
    from ..bridge import BridgeAPI


class EditorScene(QGraphicsScene):
    """Scene managing all node and connection items."""

    # Signals
    node_selected = Signal(str)          # node_id
    selection_cleared = Signal()

    PORT_SNAP_DISTANCE = 30.0  # pixels for proximity-based port selection
    NODE_FRONT_STEP = 1e-4

    def __init__(self, bridge: BridgeAPI, parent=None) -> None:
        super().__init__(parent)
        self.bridge = bridge

        self._node_items: Dict[str, NodeItem] = {}
        self._connection_items: List[ConnectionItem] = []

        # Connection drag state
        self._connecting = False
        self._connect_from_port: Optional[PortItem] = None
        self._preview_line: Optional[ConnectionPreviewItem] = None

        self._highlighted_port: Optional[PortItem] = None
        self._node_z_seq: int = 0

    # ── Node management ──────────────────────────────────────────────────

    def add_node_item(self, schema: dict, bring_to_front: bool = True) -> NodeItem:
        item = NodeItem(schema)
        self.addItem(item)
        self._node_items[schema["node_id"]] = item

        # Wire signals
        item.signals.position_changed.connect(self._on_node_moved)
        item.signals.name_change_requested.connect(self._on_name_change_requested)
        item.signals.field_changed.connect(self._on_field_changed)
        item.signals.action_triggered.connect(self._on_action_triggered)
        item.signals.reload_requested.connect(self._on_reload_requested)
        if bring_to_front:
            self.bring_node_to_front(item)

        return item

    def _clear_interactions_for_node(self, node_id: str) -> None:
        if self._highlighted_port and self._highlighted_port.node_id == node_id:
            self._highlighted_port.set_highlighted(False)
            self._highlighted_port = None
        if self._connect_from_port and self._connect_from_port.node_id == node_id:
            if self._preview_line:
                self.removeItem(self._preview_line)
                self._preview_line = None
            self._connecting = False
            self._connect_from_port = None

    def remove_node_item(self, node_id: str) -> None:
        self._clear_interactions_for_node(node_id)
        item = self._node_items.pop(node_id, None)
        dirty_rect = QRectF()
        if item:
            dirty_rect = item.sceneBoundingRect().adjusted(-8.0, -8.0, 8.0, 8.0)
            try:
                item.dispose()
            except Exception:
                pass
            item.setEnabled(False)
            item.setVisible(False)
            for child in item.childItems():
                try:
                    child.setEnabled(False)
                    child.setVisible(False)
                except Exception:
                    pass
            self.removeItem(item)
        # Remove associated connections
        i = 0
        while i < len(self._connection_items):
            c = self._connection_items[i]
            if c.conn_data["from_node"] == node_id or c.conn_data["to_node"] == node_id:
                self.removeItem(c)
                self._connection_items.pop(i)
            else:
                i += 1
        self._update_port_connected_states()
        if not dirty_rect.isNull():
            self.update(dirty_rect)
            for view in self.views():
                view.viewport().update()

    def get_node_item(self, node_id: str) -> Optional[NodeItem]:
        return self._node_items.get(node_id)

    def bring_node_to_front(self, node_item: NodeItem) -> None:
        """
        Bring a node to the top of node-level stacking.

        Keeps node z-values in the node band (around Z_NODE) so other layers
        like ports/widgets/overlays remain above as designed.
        """
        self._node_z_seq += 1
        node_item.setZValue(Z_NODE + self._node_z_seq * self.NODE_FRONT_STEP)

    def dispose(self) -> None:
        """Release timers and interaction state before the scene is destroyed."""
        if self._highlighted_port:
            self._highlighted_port.set_highlighted(False)
            self._highlighted_port = None
        self._connect_from_port = None
        self._preview_line = None
        self._connecting = False
        for item in list(self._node_items.values()):
            try:
                item.dispose()
            except Exception:
                pass
        self.clear()
        self._node_items.clear()
        self._connection_items.clear()

    def deactivate(self) -> None:
        """Freeze scene-owned timers/interactions without destroying the items."""
        if self._highlighted_port:
            self._highlighted_port.set_highlighted(False)
            self._highlighted_port = None
        self._connect_from_port = None
        self._preview_line = None
        self._connecting = False
        for item in list(self._node_items.values()):
            try:
                item.dispose()
            except Exception:
                pass

    # ── Connection management ────────────────────────────────────────────

    def add_connection_item(self, conn_data: dict) -> ConnectionItem:
        item = ConnectionItem(conn_data)
        self.addItem(item)
        self._connection_items.append(item)
        self._update_connection_path(item)
        self._update_port_connected_states()
        return item

    def remove_connection_item(self, conn_data: dict) -> None:
        for i, item in enumerate(self._connection_items):
            d = item.conn_data
            if (d["from_node"] == conn_data["from_node"] and
                d["from_output"] == conn_data["from_output"] and
                d["to_node"] == conn_data["to_node"] and
                d["to_input"] == conn_data["to_input"]):
                self.removeItem(item)
                self._connection_items.pop(i)
                break
        self._update_port_connected_states()

    def _update_connection_path(self, item: ConnectionItem) -> None:
        d = item.conn_data
        from_node = self._node_items.get(d["from_node"])
        to_node = self._node_items.get(d["to_node"])
        if not from_node or not to_node:
            return
        from_port = from_node.get_port(d["from_output"], "output")
        to_port = to_node.get_port(d["to_input"], "input")
        if from_port and to_port:
            item.update_path(from_port.center_scene_pos, to_port.center_scene_pos)

    def update_connections_for_node(self, node_id: str) -> None:
        """Called when a node moves — update all its connections."""
        for item in self._connection_items:
            d = item.conn_data
            if d["from_node"] == node_id or d["to_node"] == node_id:
                self._update_connection_path(item)

    def update_all_connections(self) -> None:
        for item in self._connection_items:
            self._update_connection_path(item)

    def _update_port_connected_states(self) -> None:
        """Update connected visual state on all ports."""
        connected_ports = set()
        for item in self._connection_items:
            d = item.conn_data
            connected_ports.add((d["from_node"], d["from_output"], "output"))
            connected_ports.add((d["to_node"], d["to_input"], "input"))

        for node_item in self._node_items.values():
            for port in node_item.input_ports:
                port.set_connected((port.node_id, port.port_name, "input") in connected_ports)
            for port in node_item.output_ports:
                port.set_connected((port.node_id, port.port_name, "output") in connected_ports)

    def _nearest_port_at(self, scene_pos: QPointF, max_distance: Optional[float] = None) -> Optional[PortItem]:
        """Return the nearest PortItem within max_distance, or None."""
        effective_dist = max_distance if max_distance is not None else self.PORT_SNAP_DISTANCE
        best = None
        best_dist = effective_dist
        for node_item in self._node_items.values():
            if getattr(node_item, "_loading", False):
                continue
            for port in node_item.input_ports + node_item.output_ports:
                port_center = port.center_scene_pos
                dist = QLineF(scene_pos, port_center).length()
                if dist <= best_dist:
                    best = port
                    best_dist = dist
        return best

    def _items_at(self, scene_pos: QPointF):
        """Return items at position in top-to-bottom stacking order."""
        return self.items(
            scene_pos,
            Qt.ItemSelectionMode.IntersectsItemShape,
            Qt.SortOrder.DescendingOrder,
        )

    def _resolve_interactive_item(self, item):
        """Resolve proxy/child items to the nearest interactive canvas item."""
        while item is not None:
            if isinstance(item, (PortItem, PortHitZoneItem, NodeItem, ConnectionItem)):
                return item
            item = item.parentItem()
        return None

    def _has_proxy_ancestor(self, item) -> bool:
        while item is not None:
            if isinstance(item, QGraphicsProxyWidget):
                return True
            item = item.parentItem()
        return False

    def _is_over_proxy_widget(self, scene_pos: QPointF) -> bool:
        for item in self._items_at(scene_pos):
            if self._has_proxy_ancestor(item):
                return True
        return False

    def _port_from_interactive(self, item) -> Optional[PortItem]:
        port: Optional[PortItem] = None
        if isinstance(item, PortItem):
            port = item
        elif isinstance(item, PortHitZoneItem):
            port = item.port
        if port is None:
            return None
        parent = port.parentItem()
        if isinstance(parent, NodeItem) and getattr(parent, "_loading", False):
            return None
        return port

    def interactive_item_at(self, scene_pos: QPointF):
        """Return topmost interactive item (port/node/connection) at scene_pos."""
        for item in self._items_at(scene_pos):
            interactive = self._resolve_interactive_item(item)
            if interactive is not None:
                return interactive
        return None

    def _top_node_at(self, scene_pos: QPointF) -> Optional[NodeItem]:
        """Return the topmost node at scene_pos."""
        for item in self._items_at(scene_pos):
            interactive = self._resolve_interactive_item(item)
            if isinstance(interactive, NodeItem):
                return interactive
            if isinstance(interactive, PortItem):
                parent = interactive.parentItem()
                if isinstance(parent, NodeItem):
                    return parent
            if isinstance(interactive, PortHitZoneItem):
                parent = interactive.parentItem()
                if isinstance(parent, NodeItem):
                    return parent
        return None

    def _update_port_highlight(self, scene_pos: QPointF) -> None:
        """Update port highlighting based on snap proximity and label hover."""
        if QApplication.activePopupWidget() is not None or self._is_over_proxy_widget(scene_pos):
            if self._highlighted_port:
                self._highlighted_port.set_highlighted(False)
                self._highlighted_port = None
            return

        hovered_port = self._port_from_interactive(self.interactive_item_at(scene_pos))
        # Find the nearest port within snap distance or under label
        nearest_snap = self._nearest_port_at(scene_pos)
        label_port = self._port_at_label(scene_pos)

        # Determine which port to highlight
        new_highlight = hovered_port or nearest_snap or label_port
        
        # Don't highlight the source port (the one we're dragging from)
        if new_highlight and self._connect_from_port:
            if (new_highlight.node_id == self._connect_from_port.node_id and
                new_highlight.port_name == self._connect_from_port.port_name and
                new_highlight.port_type == self._connect_from_port.port_type):
                new_highlight = None
        
        # Clear previous highlight if different
        if self._highlighted_port and self._highlighted_port != new_highlight:
            self._highlighted_port.set_highlighted(False)
            self._highlighted_port = None
        
        # Set new highlight
        if new_highlight and new_highlight != self._highlighted_port:
            new_highlight.set_highlighted(True)
            self._highlighted_port = new_highlight

    def _port_at_label(self, scene_pos: QPointF) -> Optional[PortItem]:
        """Return the PortItem for label/corridor hit zones in the I/O row."""
        HEADER_HEIGHT = 22.0
        PORT_ROW_HEIGHT = 18.0
        LABEL_LEFT = 12.0
        LABEL_RIGHT_MARGIN = 14.0
        PORT_GRAB_MARGIN = 6.0

        for node_item in self._node_items.values():
            if getattr(node_item, "_loading", False):
                continue
            # Get node position in scene coordinates
            node_pos = node_item.pos()
            width = node_item._width
            half_width = width / 2

            # Input side: include label area and the corridor between label and port.
            for i, port in enumerate(node_item.input_ports):
                label_x = node_pos.x() - PORT_GRAB_MARGIN
                label_y = node_pos.y() + HEADER_HEIGHT + i * PORT_ROW_HEIGHT
                label_w = half_width - LABEL_RIGHT_MARGIN + PORT_GRAB_MARGIN
                label_h = PORT_ROW_HEIGHT
                label_rect = QRectF(label_x, label_y, label_w, label_h)
                if label_rect.contains(scene_pos):
                    return port

            # Output side: include label area and the corridor between label and port.
            for i, port in enumerate(node_item.output_ports):
                label_x = node_pos.x() + half_width + 2
                label_y = node_pos.y() + HEADER_HEIGHT + i * PORT_ROW_HEIGHT
                label_w = half_width - 2 + PORT_GRAB_MARGIN
                label_h = PORT_ROW_HEIGHT
                label_rect = QRectF(label_x, label_y, label_w, label_h)
                if label_rect.contains(scene_pos):
                    return port

        return None

    # ── Full rebuild ─────────────────────────────────────────────────────

    def rebuild_from_topology(self, topology: dict) -> None:
        """Clear and rebuild all items from a topology snapshot."""
        self.dispose()
        self._node_z_seq = 0

        for schema in topology.get("nodes", []):
            self.add_node_item(schema, bring_to_front=False)

        for conn in topology.get("connections", []):
            self.add_connection_item(conn)

    # ── Display updates ──────────────────────────────────────────────────

    def update_displays(self, display_buffer: Dict[str, Dict]) -> None:
        """Push display data to node items."""
        for node_id, outputs in display_buffer.items():
            item = self._node_items.get(node_id)
            if item:
                item.update_displays(outputs)

    # ── Connection drag ──────────────────────────────────────────────────

    def start_connection_drag(self, port: PortItem, scene_pos: QPointF) -> None:
        self._connecting = True
        self._connect_from_port = port
        self._preview_line = ConnectionPreviewItem()
        self.addItem(self._preview_line)
        self._preview_line.update_path(port.center_scene_pos, scene_pos)

    def update_connection_drag(self, scene_pos: QPointF) -> None:
        if self._preview_line and self._connect_from_port:
            self._preview_line.update_path(self._connect_from_port.center_scene_pos, scene_pos)

    def end_connection_drag(self, target_port: Optional[PortItem]) -> None:
        if self._preview_line:
            self.removeItem(self._preview_line)
            self._preview_line = None

        # Clear port highlight
        if self._highlighted_port:
            self._highlighted_port.set_highlighted(False)
            self._highlighted_port = None

        if not self._connecting or not self._connect_from_port:
            self._connecting = False
            return

        from_port = self._connect_from_port
        self._connecting = False
        self._connect_from_port = None

        if not target_port:
            return
        if from_port.node_id == target_port.node_id:
            return
        if from_port.port_type == target_port.port_type:
            return

        # Determine direction
        if from_port.port_type == "output":
            from_node, from_output = from_port.node_id, from_port.port_name
            to_node, to_input = target_port.node_id, target_port.port_name
        else:
            from_node, from_output = target_port.node_id, target_port.port_name
            to_node, to_input = from_port.node_id, from_port.port_name

        try:
            self.bridge.create_connection(from_node, from_output, to_node, to_input)
            # Rebuild connections from registry
        except Exception as e:
            print(f"Connection failed: {e}")
        
        self._rebuild_connections()

    def cancel_connection_drag(self) -> None:
        if self._preview_line:
            self.removeItem(self._preview_line)
            self._preview_line = None
        # Clear port highlight
        if self._highlighted_port:
            self._highlighted_port.set_highlighted(False)
            self._highlighted_port = None
        self._connecting = False
        self._connect_from_port = None

    @property
    def is_connecting(self) -> bool:
        return self._connecting

    def _rebuild_connections(self) -> None:
        """Remove all connection items and rebuild from registry."""
        for item in self._connection_items:
            self.removeItem(item)
        self._connection_items.clear()

        from axonforge.core.registry import get_connections
        for conn in get_connections():
            self.add_connection_item(conn)

    # ── Selection ────────────────────────────────────────────────────────

    def deselect_all(self) -> None:
        self.clearSelection()
        self.selection_cleared.emit()

    def clear_connection_selection(self) -> None:
        return

    def delete_selected(self) -> None:
        """Delete selected node(s)."""
        selected = [item for item in self.selectedItems() if isinstance(item, NodeItem)]
        self._delete_node_items(selected)

    def _delete_node_items(self, items: List[NodeItem]) -> None:
        """Delete specific node items."""
        for item in items:
            try:
                self.bridge.delete_node(item.node_id)
            except ValueError:
                pass
            self.remove_node_item(item.node_id)
            self._rebuild_connections()

    def _hovered_interactive_item(self):
        """Return the interactive item currently under the cursor, if any."""
        if not self.views():
            return None
        view = self.views()[0]
        view_pos = view.mapFromGlobal(QCursor.pos())
        if not view.viewport().rect().contains(view_pos):
            return None
        scene_pos = view.mapToScene(view_pos)
        if self._is_over_proxy_widget(scene_pos):
            node = self._top_node_at(scene_pos)
            return node
        return self.interactive_item_at(scene_pos)

    def _delete_connection(self, item: ConnectionItem) -> None:
        """Delete a specific connection item directly (no selection state)."""
        d = item.conn_data
        try:
            self.bridge.delete_connection(
                d["from_node"], d["from_output"], d["to_node"], d["to_input"]
            )
        except ValueError:
            pass
        self.remove_connection_item(d)

    # ── Signal handlers ──────────────────────────────────────────────────

    def _on_node_moved(self, node_id: str, x: float, y: float) -> None:
        self.bridge.update_node_position(node_id, x, y)

    def _on_name_change_requested(self, node_id: str, name: str) -> None:
        item = self._node_items.get(node_id)
        try:
            applied_name = self.bridge.set_node_name(node_id, name)
        except Exception as e:
            print(f"Node rename failed: {e}")
            if item:
                item.cancel_title_rename()
            return
        if item:
            item.set_display_name(applied_name)

    def _on_field_changed(self, node_id: str, field_key: str, value: object) -> None:
        try:
            self.bridge.set_field(node_id, field_key, value)
        except Exception as e:
            print(f"Field update failed: {e}")

    def _on_action_triggered(self, node_id: str, action_key: str) -> None:
        try:
            self.bridge.execute_action(node_id, action_key)
        except Exception as e:
            print(f"Action failed: {e}")

    def _on_reload_requested(self, node_id: str) -> None:
        def _do_reload() -> None:
            try:
                new_node = self.bridge.reload_node(node_id)
                # Remove old item and rebuild
                self.remove_node_item(node_id)
                self.add_node_item(new_node.get_schema())
                self._rebuild_connections()
            except Exception as e:
                print(f"Reload failed: {e}")

        QTimer.singleShot(0, _do_reload)

    # ── Mouse events for connection creation ─────────────────────────────

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        # If a popup (e.g., QComboBox list) is active, never intercept with scene logic.
        if QApplication.activePopupWidget() is not None:
            super().mousePressEvent(event)
            return

        # Right button → delete connections only
        if event.button() == Qt.MouseButton.RightButton:
            top_item = self.interactive_item_at(event.scenePos())
            if isinstance(top_item, ConnectionItem):
                self._delete_connection(top_item)
                return
        if event.button() == Qt.MouseButton.LeftButton:
            scene_pos = event.scenePos()

            # Let embedded widgets (and their children) own their clicks.
            if self._is_over_proxy_widget(scene_pos):
                self.clear_connection_selection()
                super().mousePressEvent(event)
                return

            top_item = self.interactive_item_at(scene_pos)

            clicked_port: Optional[PortItem] = self._port_from_interactive(top_item)
            if clicked_port is None:
                # Labels are painted in the node body, so keep this geometric fallback.
                label_port = self._port_at_label(scene_pos)
                if label_port:
                    top_node = self._top_node_at(scene_pos)
                    if top_node is None or top_node.node_id == label_port.node_id:
                        clicked_port = label_port

            if clicked_port is not None:
                parent = clicked_port.parentItem()
                if isinstance(parent, NodeItem):
                    self.bring_node_to_front(parent)
                self.start_connection_drag(clicked_port, scene_pos)
                return

            if isinstance(top_item, ConnectionItem):
                # Connections are hover-highlight only; they are not selectable.
                return

            if top_item is None:
                snapped_port = self._nearest_port_at(scene_pos)
                if snapped_port:
                    parent = snapped_port.parentItem()
                    if isinstance(parent, NodeItem):
                        self.bring_node_to_front(parent)
                    self.start_connection_drag(snapped_port, scene_pos)
                    return
                self.clear_connection_selection()
                super().mousePressEvent(event)
                return

            # Click landed on a node or child widget; let default item handling run.
            self.clear_connection_selection()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if QApplication.activePopupWidget() is not None:
            if self._highlighted_port:
                self._highlighted_port.set_highlighted(False)
                self._highlighted_port = None
            super().mouseMoveEvent(event)
            return

        if self._connecting:
            self.update_connection_drag(event.scenePos())
            self._update_port_highlight(event.scenePos())
            return
        # Also update port highlight when not connecting (for label hover)
        self._update_port_highlight(event.scenePos())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self._connecting:
            scene_pos = event.scenePos()
            top_item = self.interactive_item_at(scene_pos)
            target = self._port_from_interactive(top_item)
            # If no exact port under cursor, try label hit
            if target is None:
                target = self._port_at_label(scene_pos)
            # If no label hit, try proximity-based snapping
            if target is None:
                target = self._nearest_port_at(scene_pos)
            self.end_connection_drag(target)
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:
        # Allow keys to be handled by focused widget (e.g., property input)
        focus = self.focusItem()
        if focus:
            # Check if focus is a proxy widget or a child of a proxy widget
            from PySide6.QtWidgets import QGraphicsProxyWidget
            item = focus
            while item:
                if isinstance(item, QGraphicsProxyWidget):
                    super().keyPressEvent(event)
                    return
                item = item.parentItem()

        # Shift+D → duplicate selected node
        if event.key() == Qt.Key.Key_D and event.modifiers() == Qt.KeyboardModifier.ShiftModifier:
            self.duplicate_selected()
            return

        # Shift+X → delete selected node(s)
        if event.key() == Qt.Key.Key_X and event.modifiers() == Qt.KeyboardModifier.ShiftModifier:
            hovered = self._hovered_interactive_item()
            if isinstance(hovered, ConnectionItem):
                self._delete_connection(hovered)
                event.accept()
                return

            hovered_port = self._port_from_interactive(hovered)
            hovered_node = None
            if isinstance(hovered, NodeItem):
                hovered_node = hovered
            elif hovered_port is not None:
                parent = hovered_port.parentItem()
                if isinstance(parent, NodeItem):
                    hovered_node = parent

            selected_nodes = [item for item in self.selectedItems() if isinstance(item, NodeItem)]
            if hovered_node is not None:
                if hovered_node in selected_nodes and selected_nodes:
                    self._delete_node_items(selected_nodes)
                else:
                    self._delete_node_items([hovered_node])
            else:
                self._delete_node_items(selected_nodes)
            event.accept()
            return

        super().keyPressEvent(event)

    def duplicate_selected(self) -> None:
        """Duplicate selected nodes (Shift+D)."""
        selected = [item for item in self.selectedItems() if isinstance(item, NodeItem)]
        if not selected:
            return

        for item in selected:
            node_type = item.node_type
            # Offset the new node by 30 pixels
            x = item.x() + 60
            y = item.y() + 30
            try:
                new_node = self.bridge.create_node(node_type, x, y)
                new_item = self.add_node_item(new_node.get_schema())
                # Copy fields from original to new node
                self._copy_node_fields(item, new_item)
            except Exception as e:
                print(f"Failed to duplicate node: {e}")

    def _copy_node_fields(self, source_item: NodeItem, target_item: NodeItem) -> None:
        """Copy field values from source node to target node."""
        for field in source_item.schema.get("fields", []):
            key = field.get("key")
            if key and field.get("value") is not None:
                try:
                    self.bridge.set_field(target_item.node_id, key, field["value"])
                except Exception as e:
                    print(f"Failed to copy field {key}: {e}")
