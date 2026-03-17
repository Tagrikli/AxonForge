"""
EditorView — QGraphicsView with pan, zoom, and grid background.
"""

import os
from datetime import datetime
from PySide6.QtCore import Qt, QPointF, Signal, QRectF
from typing import Optional
from PySide6.QtGui import QPainter, QPen, QColor, QWheelEvent, QMouseEvent, QKeyEvent, QCursor, QImage
from PySide6.QtWidgets import QGraphicsView, QGraphicsProxyWidget, QApplication, QPlainTextEdit

from .editor_scene import EditorScene
from .node_quick_add import NodeQuickAddPopup

# Theme
BG_PRIMARY = QColor("#0b0f1a")
GRID_MINOR = QColor("#141a2a")
GRID_MAJOR = QColor("#1c2742")
GRID_MINOR_SIZE = 20
GRID_MAJOR_SIZE = 100

MIN_ZOOM = 0.1
MAX_ZOOM = 5.0
ZOOM_FACTOR = 1.0015  # per pixel of wheel delta


class EditorView(QGraphicsView):
    """Pannable, zoomable view with grid background."""

    zoom_changed = Signal(float)  # current zoom level

    def __init__(self, scene: EditorScene, parent=None) -> None:
        super().__init__(scene, parent)
        self._scene = scene
        self._zoom = 1.0
        self._panning = False
        self._pan_start = QPointF()
        self._quick_add_popup: Optional[NodeQuickAddPopup] = None

        # Rendering
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setBackgroundBrush(BG_PRIMARY)
        self.setFrameShape(QGraphicsView.Shape.NoFrame)

        # Large scene rect so panning doesn't hit edges
        self.setSceneRect(-50000, -50000, 100000, 100000)
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)

    def set_editor_scene(self, scene: EditorScene) -> None:
        """Swap the active editor scene while keeping view state intact."""
        self._scene = scene
        self.setScene(scene)

    # ── Grid background ──────────────────────────────────────────────────

    def drawBackground(self, painter: QPainter, rect) -> None:
        super().drawBackground(painter, rect)

        # Minor grid only (thinner lines for smaller squares)
        left_minor = int(rect.left()) - (int(rect.left()) % GRID_MINOR_SIZE)
        top_minor = int(rect.top()) - (int(rect.top()) % GRID_MINOR_SIZE)
        painter.setPen(QPen(GRID_MINOR, 0.5))
        x = left_minor
        while x < rect.right():
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            x += GRID_MINOR_SIZE
        y = top_minor
        while y < rect.bottom():
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
            y += GRID_MINOR_SIZE

    # ── Zoom ─────────────────────────────────────────────────────────────

    @property
    def current_zoom(self) -> float:
        return self._zoom

    def set_zoom(self, zoom: float, center: Optional[QPointF] = None) -> None:
        zoom = max(MIN_ZOOM, min(MAX_ZOOM, zoom))
        if center is None:
            center = self.viewport().rect().center()
            center = QPointF(center.x(), center.y())

        old_scene = self.mapToScene(int(center.x()), int(center.y()))
        factor = zoom / self._zoom
        self.scale(factor, factor)
        self._zoom = zoom

        new_scene = self.mapToScene(int(center.x()), int(center.y()))
        delta = new_scene - old_scene
        self.translate(delta.x(), delta.y())

        self.zoom_changed.emit(self._zoom)

    def reset_view(self) -> None:
        self.resetTransform()
        self._zoom = 1.0
        self.centerOn(0, 0)
        self.zoom_changed.emit(self._zoom)

    def zoom_in(self) -> None:
        self.set_zoom(self._zoom * 1.2)

    def zoom_out(self) -> None:
        self.set_zoom(self._zoom / 1.2)

    # ── Mouse events ─────────────────────────────────────────────────────

    def wheelEvent(self, event: QWheelEvent) -> None:
        from .node_item import NodeItem

        item = self.itemAt(event.position().toPoint())
        top_item = item
        while top_item is not None:
            if isinstance(top_item, QGraphicsProxyWidget):
                widget = top_item.widget()
                if isinstance(widget, QPlainTextEdit):
                    scrollbar = widget.verticalScrollBar()
                    if scrollbar is not None and scrollbar.maximum() > scrollbar.minimum():
                        super().wheelEvent(event)
                        return
            if isinstance(top_item, NodeItem):
                scene_pos = self.mapToScene(event.position().toPoint())
                local_pos = top_item.mapFromScene(scene_pos)
                # Check if mouse is over an interactive heatmap or plot display
                if (top_item._find_interactive_heatmap_at(local_pos) is not None
                        or top_item._find_plot_key_at(local_pos) is not None):
                    super().wheelEvent(event)
                    return
            top_item = top_item.parentItem()

        delta = event.angleDelta().y()
        factor = ZOOM_FACTOR ** delta
        new_zoom = max(MIN_ZOOM, min(MAX_ZOOM, self._zoom * factor))
        center = QPointF(event.position().x(), event.position().y())
        self.set_zoom(new_zoom, center)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        # Never run view-level deselection while a popup (e.g., combo dropdown) is open.
        if QApplication.activePopupWidget() is not None:
            super().mousePressEvent(event)
            return

        # Middle button → pan
        if event.button() == Qt.MouseButton.MiddleButton:
            self._scene.deselect_all()
            self._start_pan(event)
            return
        if event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            clicked_item = self._scene.interactive_item_at(scene_pos)
            if clicked_item is None and not self._scene.is_connecting:
                self._scene.deselect_all()
                # do not start pan with left button
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._panning:
            delta = event.position() - self._pan_start
            self._pan_start = event.position()
            self.translate(delta.x() / self._zoom, delta.y() / self._zoom)
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._panning:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            # Re-enable mouse tracking after pan ends
            self.viewport().setMouseTracking(True)
            return
        super().mouseReleaseEvent(event)

    def _start_pan(self, event: QMouseEvent) -> None:
        self._panning = True
        self._pan_start = event.position()
        self.setCursor(Qt.CursorShape.ClosedHandCursor)
        # Disable hover events on all items during pan to prevent highlighting
        for item in self._scene.items():
            item.setAcceptHoverEvents(False)
        # Disable mouse tracking during pan
        self.viewport().setMouseTracking(False)

    # ── Context menu ─────────────────────────────────────────────────────


    def _create_node_at(self, node_type: str, x: float, y: float) -> None:
        try:
            node = self._scene.bridge.create_node(node_type, x, y)
            self._scene.add_node_item(node.get_schema())
        except Exception as e:
            print(f"Failed to create node: {e}")

    def _show_quick_add(self, global_pos, scene_pos: QPointF) -> None:
        if self._quick_add_popup is not None:
            self._quick_add_popup.close()
            self._quick_add_popup = None

        popup = NodeQuickAddPopup(self._scene.bridge.node_palette, parent=self)
        popup.node_selected.connect(
            lambda node_type, x=scene_pos.x(), y=scene_pos.y(): self._create_node_at(node_type, x, y)
        )
        popup.destroyed.connect(lambda _=None: setattr(self, "_quick_add_popup", None))
        popup.move(global_pos + QPointF(8.0, 8.0).toPoint())
        popup.show()
        popup.focus_search()
        self._quick_add_popup = popup

    # ── Drag & drop from palette ───────────────────────────────────────

    def dragEnterEvent(self, event) -> None:
        """Handle drag enter events for node palette drops."""
        if event.mimeData().hasFormat("application/x-axonforge-node-type"):
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:
        """Allow drag move events for node palette drops."""
        if event.mimeData().hasFormat("application/x-axonforge-node-type"):
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:
        """Handle drop events to create nodes from palette."""
        data = event.mimeData().data("application/x-axonforge-node-type")
        if not data:
            super().dropEvent(event)
            return
        node_type = bytes(data).decode()
        scene_pos = self.mapToScene(event.position().toPoint())
        try:
            node = self._scene.bridge.create_node(node_type, scene_pos.x(), scene_pos.y())
            self._scene.add_node_item(node.get_schema())
        except Exception as e:
            print(f"Failed to create node: {e}")
        event.acceptProposedAction()

    def _fit_to_view(self) -> None:
        items = [i for i in self._scene.items() if hasattr(i, "node_id")]
        if not items:
            return
        from PySide6.QtCore import QRectF
        rect = QRectF()
        for item in items:
            rect = rect.united(item.sceneBoundingRect())
        rect.adjust(-50, -50, 50, 50)
        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
        self._zoom = self.transform().m11()
        self.zoom_changed.emit(self._zoom)

    def export_screenshot(self) -> None:
        """Capture the entire scene and save it to the project's data/screenshots folder."""
        # 1. Determine save directory
        # Try to get the current project directory from the bridge
        project_dir = getattr(self._scene.bridge, "project_dir", None)
        if project_dir:
            save_dir = os.path.join(str(project_dir), "data", "screenshots")
        else:
            save_dir = os.path.join(os.getcwd(), "data", "screenshots")
            
        os.makedirs(save_dir, exist_ok=True)

        # 2. Generate filename: yyMMdd_HHmmss.png
        filename = datetime.now().strftime("%y%m%d_%H%M%S") + ".png"
        file_path = os.path.join(save_dir, filename)

        # 3. Determine the area containing all nodes
        rect = self.scene().itemsBoundingRect()
        if rect.isEmpty():
            return
        
        # Add padding
        rect.adjust(-50, -50, 50, 50)

        # 4. Create the image at 3x resolution for crisp output
        scale = 3
        img_size = rect.size().toSize() * scale
        image = QImage(img_size, QImage.Format.Format_ARGB32)
        image.fill(BG_PRIMARY)

        # 5. Render
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.scene().render(painter, target=QRectF(image.rect()), source=rect)
        painter.end()

        # 6. Save
        # Passing None for format allows Qt to infer it from the file extension (.png)
        if image.save(file_path):
            print(f"Screenshot saved to: {file_path}")
        else:
            print(f"Failed to save screenshot to: {file_path}")

    # ── Keyboard ─────────────────────────────────────────────────────────

    def keyPressEvent(self, event: QKeyEvent) -> None:
        # If a property widget has focus, let it handle the key
        focus = self._scene.focusItem()
        if focus:
            item = focus
            while item:
                if isinstance(item, QGraphicsProxyWidget):
                    super().keyPressEvent(event)
                    return
                item = item.parentItem()
        # CTRL+ENTER → toggle network
        if event.key() == Qt.Key.Key_Return and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            # Handled by main window
            event.ignore()
            return
        # CTRL+SPACE → ignore (was right arrow)
        if event.key() == Qt.Key.Key_Space and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            event.ignore()
            return
        # CTRL+P -> export screenshot
        if event.key() == Qt.Key.Key_P and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            self.export_screenshot()
            event.accept()
            return

        # Shift+A -> quick add popup at mouse position
        if event.key() == Qt.Key.Key_A and event.modifiers() == Qt.KeyboardModifier.ShiftModifier:
            global_pos = QCursor.pos()
            view_pos = self.mapFromGlobal(global_pos)
            scene_pos = self.mapToScene(view_pos)
            self._show_quick_add(global_pos, scene_pos)
            event.accept()
            return
        super().keyPressEvent(event)
