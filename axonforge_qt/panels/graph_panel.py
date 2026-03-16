"""
GraphPanel — Right-side drawer listing available graphs in the project.

Supports creating, switching, and deleting graphs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QMessageBox, QLineEdit,
)

if TYPE_CHECKING:
    from ..bridge import BridgeAPI


class _GraphEntry(QWidget):
    """Single row in the graph list: name + delete button."""

    clicked = Signal(object)
    delete_clicked = Signal(object)
    rename_requested = Signal(object, str)  # old_name, new_name

    def __init__(
        self,
        name: str,
        graph_key: object,
        active: bool = False,
        deletable: bool = True,
        unsaved: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.graph_name = name
        self.graph_key = graph_key
        self.setFixedHeight(32)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 2, 6, 2)
        layout.setSpacing(6)

        self._dot = QWidget()
        self._dot.setFixedSize(8, 8)
        self._dot.setStyleSheet(
            "QWidget { background: #6f7d9d; border-radius: 4px; }"
        )
        self._dot.setVisible(unsaved)
        self._dot.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        layout.addWidget(self._dot)

        self._label = QLabel(name)
        self._label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        font = QFont()
        font.setPointSize(10)
        if active:
            font.setBold(True)
            self._label.setStyleSheet("color: #8fa4ff;")
        else:
            self._label.setStyleSheet("color: #cfd8ee;")
        self._label.setFont(font)
        layout.addWidget(self._label)

        # Inline rename editor (hidden by default)
        self._edit = QLineEdit()
        self._edit.setFont(font)
        self._edit.setStyleSheet(
            "QLineEdit { background: #1a2236; color: #cfd8ee; border: 1px solid #8fa4ff;"
            " border-radius: 2px; padding: 0 4px; }"
        )
        self._edit.setVisible(False)
        self._edit.returnPressed.connect(self._commit_rename)
        self._edit.editingFinished.connect(self._commit_rename)
        layout.addWidget(self._edit)

        layout.addStretch()

        self._delete_btn = QPushButton("\u00d7")
        self._delete_btn.setFixedSize(20, 20)
        self._delete_btn.setStyleSheet(
            "QPushButton { color: #666; background: transparent; border: none; font-size: 14px; }"
            "QPushButton:hover { color: #ff6b6b; }"
        )
        self._delete_btn.setVisible(deletable)
        self._delete_btn.clicked.connect(lambda: self.delete_clicked.emit(self.graph_key))
        layout.addWidget(self._delete_btn)

        self._active = active
        self._renaming = False
        self._update_style()

    def _update_style(self) -> None:
        if self._active:
            self.setStyleSheet("_GraphEntry { background: rgba(143, 164, 255, 0.1); }")
        else:
            self.setStyleSheet(
                "_GraphEntry { background: transparent; }"
                "_GraphEntry:hover { background: rgba(255, 255, 255, 0.04); }"
            )

    def _start_rename(self) -> None:
        self._renaming = True
        self._edit.setText(self.graph_name)
        self._label.setVisible(False)
        self._edit.setVisible(True)
        self._edit.setFocus()
        self._edit.selectAll()

    def _commit_rename(self) -> None:
        if not self._renaming:
            return
        new_name = self._edit.text().strip()
        self._renaming = False
        self._label.setVisible(True)
        self._edit.setVisible(False)
        if new_name and new_name != self.graph_name:
            self.rename_requested.emit(self.graph_key, new_name)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.graph_key)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._start_rename()
        else:
            super().mouseDoubleClickEvent(event)


class GraphPanel(QWidget):
    """Right-side graph management panel."""

    graph_switch_requested = Signal(str)
    graph_new_requested = Signal()
    graph_delete_requested = Signal(object)
    graph_rename_requested = Signal(object, str)  # old_name, new_name

    def __init__(self, bridge: BridgeAPI, parent=None) -> None:
        super().__init__(parent)
        self.bridge = bridge
        self._entries: dict[object, _GraphEntry] = {}
        self.setObjectName("graph_panel")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = QWidget()
        header.setObjectName("graph_panel_header")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(10, 6, 10, 6)

        title = QLabel("GRAPHS")
        title.setObjectName("graph_panel_title")
        header_layout.addWidget(title)
        header_layout.addStretch()

        add_btn = QPushButton("+")
        add_btn.setObjectName("graph_add_btn")
        add_btn.setFixedSize(24, 24)
        add_btn.setStyleSheet(
            "QPushButton { color: #8fa4ff; background: transparent; border: 1px solid #26324d;"
            " border-radius: 4px; font-size: 16px; font-weight: bold; }"
            "QPushButton:hover { background: rgba(143, 164, 255, 0.15); }"
        )
        add_btn.clicked.connect(self._on_add_graph)
        header_layout.addWidget(add_btn)

        layout.addWidget(header)

        # Scrollable list of graphs
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        self._list_container = QWidget()
        self._list_layout = QVBoxLayout(self._list_container)
        self._list_layout.setContentsMargins(0, 4, 0, 4)
        self._list_layout.setSpacing(0)
        self._list_layout.addStretch()

        scroll.setWidget(self._list_container)
        layout.addWidget(scroll)

        self.refresh()

    def refresh(self) -> None:
        """Rebuild the graph list from disk."""
        # Clear existing entries (keep the stretch at the end)
        self._entries.clear()
        while self._list_layout.count() > 1:
            item = self._list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        active = self.bridge.current_graph_name
        if active is None:
            entry = _GraphEntry(
                self.bridge.current_graph_display_name,
                graph_key=None,
                active=True,
                deletable=False,
                unsaved=self.bridge.current_graph_has_unsaved_state(),
            )
            entry.clicked.connect(self._on_graph_clicked)
            entry.rename_requested.connect(self._on_rename_requested)
            self._entries[None] = entry
            self._list_layout.insertWidget(self._list_layout.count() - 1, entry)

        for name in self.bridge.list_graphs():
            entry = _GraphEntry(
                name,
                graph_key=name,
                active=(name == active),
                deletable=True,
                unsaved=(name == active and self.bridge.current_graph_has_unsaved_state()),
            )
            entry.clicked.connect(self._on_graph_clicked)
            entry.delete_clicked.connect(self._on_delete_clicked)
            entry.rename_requested.connect(self._on_rename_requested)
            self._entries[name] = entry
            self._list_layout.insertWidget(self._list_layout.count() - 1, entry)

    def begin_rename_current_graph(self) -> bool:
        key = self.bridge.current_graph_name
        if key not in self._entries:
            key = None
        entry = self._entries.get(key)
        if entry is None:
            return False
        entry._start_rename()
        return True

    def _on_add_graph(self) -> None:
        self.graph_new_requested.emit()

    def _on_graph_clicked(self, name: object) -> None:
        if name is None:
            return
        if name != self.bridge.current_graph_name:
            self.graph_switch_requested.emit(name)

    def _on_delete_clicked(self, name: object) -> None:
        if name is None:
            return
        result = QMessageBox.question(
            self,
            "Delete Graph",
            f"Delete graph '{name}'? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if result == QMessageBox.StandardButton.Yes:
            self.graph_delete_requested.emit(name)

    def _on_rename_requested(self, old_name: object, new_name: str) -> None:
        self.graph_rename_requested.emit(old_name, new_name)
