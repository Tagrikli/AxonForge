"""
NodeItem — QGraphicsItem representing a single node on the canvas.

Layout (top to bottom):
  Header  (title + reload btn)
  I/O ports row
  Fields (sliders, etc.)
  Actions (buttons)
  Displays (images, text, etc.)

Field widgets are embedded via QGraphicsProxyWidget.
Display outputs are painted directly for performance.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

from PySide6.QtCore import Qt, QRectF, QPointF, QSizeF, Signal, QObject, QTimer, QEvent
from PySide6.QtGui import (
    QCursor, QPen, QBrush, QColor, QPainter, QFont, QFontMetrics, QImage, QPainterPath,
)
from PySide6.QtWidgets import (
    QGraphicsItem, QGraphicsProxyWidget,
    QWidget, QSlider, QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox, QPushButton, QLineEdit,
    QPlainTextEdit,
    QFileDialog, QHBoxLayout, QVBoxLayout, QLabel, QSizePolicy, QToolTip, QMenu,
    QStyleOptionGraphicsItem, QGraphicsSceneMouseEvent, QGraphicsSceneWheelEvent, QProxyStyle, QStyle,
)

from .port_item import PortItem, PortHitZoneItem
from .z_layers import Z_NODE, Z_NODE_WIDGET

if TYPE_CHECKING:
    from .editor_scene import EditorScene

# ── Theme constants ──────────────────────────────────────────────────────

BG_NODE = QColor("#141c2f")
BG_HEADER = QColor("#19213a")
BORDER_COLOR = QColor("#26324d")
BORDER_HOVER = QColor("#555555")
BORDER_SELECTED = QColor("#00f5ff")
TEXT_PRIMARY = QColor("#e6f1ff")
TEXT_SECONDARY = QColor("#8fa4ff")
ACCENT = QColor("#00f5ff")
BG_SECONDARY = QColor("#11172a")
INPUT_PORT_ACCENT = QColor("#3dff6f")
WARNING_COLOR = QColor("#ffd400")
ERROR_COLOR = QColor("#ff3b30")  # Red for error state
ERROR_PANEL_BG = QColor(80, 20, 20, 180)
ERROR_PANEL_BORDER = QColor(140, 40, 40)
ERROR_TEXT = QColor("#ff6b6b")
STATE_TEXT = QColor("#57d37a")
LOADING_OVERLAY_BG = QColor(12, 18, 32, 165)
LOADING_TEXT = QColor("#d6e6ff")

HEADER_HEIGHT = 22.0
PORT_ROW_HEIGHT = 18.0
NODE_MIN_WIDTH = 250.0
PLOT_NODE_MIN_WIDTH = 420.0
PADDING = 6.0
MAX_DISPLAY_DIM = 512
PLOT_DISPLAY_HEIGHT = 180.0
FONT_FAMILY = "Roboto"
MONO_FAMILY = "Roboto Mono"


def _generate_category_color(category: str) -> QColor:
    """Generate a consistent neon-style color for a category using hashing.
    
    Uses the same color generation as the palette panel so that node colors
    match top-level category colors in the palette tree.
    
    The category is a branch path like "Input/Noise"; node tags use the
    top-level part ("Input").
    """
    if not category:
        return QColor()
    
    # Extract top-level category name.
    # e.g., "Input/Noise" -> "Input"
    parts = [part for part in category.split("/") if part]
    branch_name = parts[0] if parts else category
    
    # Stable neon color generation from branch text.
    hash_bytes = hashlib.md5(branch_name.encode()).digest()
    hash_int = int.from_bytes(hash_bytes[:4], "big")
    hue = hash_int % 360
    saturation = 80 + ((hash_int >> 8) % 21)
    brightness = 80 + ((hash_int >> 16) % 21)

    import colorsys
    r, g, b = colorsys.hsv_to_rgb(hue / 360.0, saturation / 100.0, brightness / 100.0)
    hex_color = f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
    return QColor(hex_color)


class _NodeSignals(QObject):
    """Signals emitted by NodeItem (QGraphicsItem can't emit signals directly)."""
    position_changed = Signal(str, float, float)  # node_id, x, y
    name_change_requested = Signal(str, str)      # node_id, new_name
    field_changed = Signal(str, str, object)   # node_id, field_key, value
    action_triggered = Signal(str, str)            # node_id, action_key
    reload_requested = Signal(str)                 # node_id


class _DisableRightClickFilter(QObject):
    """Filter to swallow right-click/context-menu events for input widgets."""

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Type.ContextMenu:
            return True
        if event.type() in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonRelease):
            button = getattr(event, "button", None)
            if callable(button) and button() == Qt.MouseButton.RightButton:
                return True
        return super().eventFilter(watched, event)


class _AbsoluteClickSliderStyle(QProxyStyle):
    """Make left-click on slider groove jump directly to clicked value."""

    def styleHint(self, hint, option=None, widget=None, returnData=None) -> int:
        if hint == QStyle.StyleHint.SH_Slider_AbsoluteSetButtons:
            return Qt.MouseButton.LeftButton.value
        if hint == QStyle.StyleHint.SH_Slider_PageSetButtons:
            return Qt.MouseButton.MiddleButton.value
        return super().styleHint(hint, option, widget, returnData)


class _NodeProxyWidget(QGraphicsProxyWidget):
    """Proxy widget that keeps child popups above node paint layers."""

    def newProxyWidget(self, child):
        proxy = super().newProxyWidget(child)
        if proxy is not None:
            proxy.setZValue(Z_NODE_WIDGET + 100)
        return proxy


class _NodeTextDisplay(QPlainTextEdit):
    """Read-only text display that yields wheel events when it cannot scroll."""

    def wheelEvent(self, event) -> None:
        scrollbar = self.verticalScrollBar()
        if scrollbar is None or scrollbar.maximum() <= scrollbar.minimum():
            event.ignore()
            return
        super().wheelEvent(event)


class NodeItem(QGraphicsItem):
    """Visual representation of a Node on the canvas."""

    def __init__(self, schema: dict, parent=None) -> None:
        super().__init__(parent)
        self.schema = schema
        self.node_id: str = schema["node_id"]
        self.node_type: str = schema["node_type"]
        self._title: str = schema.get("name") or schema["node_type"]
        self._dynamic: bool = schema.get("dynamic", False)
        self._category: str = schema.get("category", "")

        self.signals = _NodeSignals()

        # Ports
        self.input_ports: List[PortItem] = []
        self.output_ports: List[PortItem] = []
        self._port_hit_zones: Dict[str, PortHitZoneItem] = {}

        # Proxy widgets for fields/actions
        self._proxy_widgets: List[QGraphicsProxyWidget] = []
        self._text_display_proxies: Dict[str, QGraphicsProxyWidget] = {}
        self._text_display_widgets: Dict[str, QPlainTextEdit] = {}
        self._popup_menus: List[QMenu] = []
        self._right_click_filters: List[QObject] = []
        self._title_edit_proxy: Optional[QGraphicsProxyWidget] = None
        self._title_edit: Optional[QLineEdit] = None
        self._editing_title = False

        # Display cache: {output_key: value}
        self._display_cache: Dict[str, Any] = {}
        # Display config cache: {output_key: config}
        self._display_config: Dict[str, Dict[str, Any]] = {}
        # Display images cache: {output_key: QImage}
        self._display_images: Dict[str, QImage] = {}
        # Display RGB cache for vectorized painting
        self._display_rgb_cache: Dict[str, np.ndarray] = {}

        # Computed layout metrics
        has_plot_output = any(output.get("type") == "plot" for output in schema.get("outputs", []))
        self._width = max(NODE_MIN_WIDTH, PLOT_NODE_MIN_WIDTH if has_plot_output else NODE_MIN_WIDTH)
        self._height = 0.0
        self._body_y = 0.0
        self._display_rects: Dict[str, QRectF] = {}
        # Plot pan/zoom view state: {key: {"ymin","ymax"} or {"xmin","xmax","ymin","ymax"}}
        self._plot_view_state: Dict[str, dict] = {}
        self._plot_drag_key: Optional[str] = None
        self._plot_drag_last_pos: Optional[QPointF] = None
        self._error_partition_rect: Optional[QRectF] = None
        self._state_hint_rect: Optional[QRectF] = None
        self._state_hint_text: str = ""

        # Fonts
        self._title_font = QFont(FONT_FAMILY, 8)
        self._title_font.setWeight(QFont.Weight.DemiBold)
        self._label_font = QFont(FONT_FAMILY, 7)
        self._port_font = QFont(FONT_FAMILY, 7)
        self._value_font = QFont(MONO_FAMILY, 7)
        self._small_font = QFont(FONT_FAMILY, 6)

        # Flags
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)
        self.setZValue(Z_NODE)
        self._hovered = False

        # Tooltip support for multi-type ports
        self._tooltip_port: Optional[PortItem] = None
        self._tooltip_timer = QTimer()
        self._tooltip_timer.setSingleShot(True)
        self._tooltip_timer.timeout.connect(self._show_port_tooltip)
        self._tooltip_delay = 500  # ms

        # Error state
        self._error_state = False
        self._error_message = ""

        # Background initialization state
        self._loading = bool(schema.get("loading", False))
        self._loading_spinner_angle = 0.0
        self._loading_timer = QTimer()
        self._loading_timer.setInterval(50)
        self._loading_timer.timeout.connect(self._advance_loading_spinner)

        # Position from schema
        pos = schema.get("position", {})
        self.setPos(pos.get("x", 0), pos.get("y", 0))

        # Build
        self._build_title_editor()
        self._build_ports()
        self._build_field_widgets()
        self._build_display_widgets()
        self._apply_loading_ui_state()
        self._layout()
        if self._loading:
            self._loading_timer.start()

    def dispose(self) -> None:
        """Stop auxiliary Qt objects before the graphics item is destroyed."""
        QToolTip.hideText()
        self._tooltip_port = None
        for timer in (self._tooltip_timer, self._loading_timer):
            try:
                timer.stop()
            except Exception:
                pass
            try:
                timer.timeout.disconnect()
            except Exception:
                pass
            try:
                timer.deleteLater()
            except Exception:
                pass

    # ── Port creation ────────────────────────────────────────────────────

    def _build_ports(self) -> None:
        for port_spec in self.schema.get("input_ports", []):
            port = PortItem(
                node_id=self.node_id,
                port_name=port_spec["name"],
                port_type="input",
                data_type=port_spec.get("data_type", "any"),
                data_types=port_spec.get("data_types"),  # List for multi-type ports
                label=port_spec.get("label", port_spec["name"]),
                parent=self,
            )
            self.input_ports.append(port)
            key = f"input:{port.port_name}"
            self._port_hit_zones[key] = PortHitZoneItem(port=port, parent=self)

        for port_spec in self.schema.get("output_ports", []):
            port = PortItem(
                node_id=self.node_id,
                port_name=port_spec["name"],
                port_type="output",
                data_type=port_spec.get("data_type", "any"),
                data_types=port_spec.get("data_types"),  # List for multi-type ports
                label=port_spec.get("label", port_spec["name"]),
                parent=self,
            )
            self.output_ports.append(port)
            key = f"output:{port.port_name}"
            self._port_hit_zones[key] = PortHitZoneItem(port=port, parent=self)

    def _build_title_editor(self) -> None:
        self._title_edit = QLineEdit(self._title)
        self._title_edit.setFont(self._title_font)
        self._title_edit.setStyleSheet(
            "QLineEdit { background: #11172a; color: #e6f1ff; border: 1px solid #00f5ff;"
            " border-radius: 3px; padding: 0 4px; font-size: 10px; }"
        )
        self._disable_widget_right_click(self._title_edit)
        self._title_edit.returnPressed.connect(self._commit_title_rename)
        self._title_edit.editingFinished.connect(self._commit_title_rename)
        self._title_edit_proxy = _NodeProxyWidget(self)
        self._title_edit_proxy.setWidget(self._title_edit)
        self._title_edit_proxy.setZValue(Z_NODE_WIDGET + 1)
        self._title_edit_proxy.setVisible(False)

    # ── Field widgets ─────────────────────────────────────────────────

    def _build_field_widgets(self) -> None:
        """Create embedded QWidgets for each field."""
        for field in self.schema.get("fields", []):
            widget = self._create_field_widget(field)
            if widget:
                proxy = _NodeProxyWidget(self)
                proxy.setWidget(widget)
                proxy.setZValue(Z_NODE_WIDGET)
                self._proxy_widgets.append(proxy)

        # Action buttons
        actions = self.schema.get("actions", [])
        if actions:
            container = QWidget()
            container.setStyleSheet("background: transparent;")
            layout = QHBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(2)
            for action in actions:
                btn = QPushButton(action["label"])
                btn.setFixedHeight(18)
                btn.setStyleSheet(
                    "QPushButton { background: #11172a; border: 1px solid #26324d; "
                    "color: #e6f1ff; font-size: 8px; padding: 1px 6px; }"
                    "QPushButton:hover { background: #00f5ff; border-color: #00f5ff; }"
                )
                key = action["key"]
                btn.clicked.connect(lambda checked=False, k=key: self.signals.action_triggered.emit(self.node_id, k))
                layout.addWidget(btn)
            container.setFixedHeight(22)
            proxy = _NodeProxyWidget(self)
            proxy.setWidget(container)
            proxy.setZValue(Z_NODE_WIDGET)
            self._proxy_widgets.append(proxy)

    def _build_display_widgets(self) -> None:
        """Create embedded widgets for displays that need native interaction."""
        for output in self.schema.get("outputs", []):
            if output.get("type") != "text":
                continue
            key = output.get("key", "")
            editor = _NodeTextDisplay()
            editor.setReadOnly(True)
            editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
            editor.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            editor.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            editor.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            editor.setFont(self._value_font)
            editor.setPlainText(str(output.get("value", output.get("default", "")) or ""))
            editor.setStyleSheet(
                "QPlainTextEdit { background: #11172a; color: #00f5ff; border: 1px solid #26324d; "
                "padding: 4px; font-size: 8px; font-family: 'Roboto Mono'; selection-background-color: #2a3a5f; }"
                "QScrollBar:vertical { background: #11172a; width: 8px; margin: 1px; }"
                "QScrollBar::handle:vertical { background: #8fa4ff; min-height: 18px; border-radius: 3px; }"
                "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }"
            )
            proxy = _NodeProxyWidget(self)
            proxy.setWidget(editor)
            proxy.setZValue(Z_NODE_WIDGET)
            self._text_display_proxies[key] = proxy
            self._text_display_widgets[key] = editor

    def _create_field_widget(self, field: dict) -> Optional[QWidget]:
        ptype = field.get("type", "")
        key = field.get("key", "")

        if ptype == "range":
            return self._make_range_widget(field, key)
        elif ptype == "float":
            return self._make_float_widget(field, key)
        elif ptype == "integer":
            return self._make_integer_widget(field, key)
        elif ptype == "bool":
            return self._make_bool_widget(field, key)
        elif ptype == "enum":
            return self._make_enum_widget(field, key)
        elif ptype == "directory_path":
            return self._make_directory_path_widget(field, key)
        elif ptype == "file_path":
            return self._make_file_path_widget(field, key)
        elif ptype == "file_paths":
            return self._make_file_paths_widget(field, key)
        return None

    def _make_range_widget(self, prop: dict, key: str) -> QWidget:
        # Sanitize inputs to avoid invalid values reaching Qt widgets.
        import math as _m
        def _safe_float(v, default):
            try:
                fv = float(v)
            except Exception:
                return float(default)
            return fv if _m.isfinite(fv) else float(default)

        container = QWidget()
        container.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Label row
        label_row = QWidget()
        label_row.setStyleSheet("background: transparent;")
        lr = QHBoxLayout(label_row)
        lr.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(prop.get("label", key))
        lbl.setStyleSheet("color: #8fa4ff; font-size: 8px; background: transparent;")
        lbl.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        val_lbl = QLabel(self._format_range_value(prop))
        val_lbl.setStyleSheet("color: #00f5ff; font-size: 7px; font-family: 'JetBrains Mono'; background: transparent;")
        val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        lr.addWidget(lbl)
        lr.addWidget(val_lbl)
        layout.addWidget(label_row)

        # Slider
        use_log = prop.get("scale") == "log" and prop.get("min", 0) > 0
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setFixedHeight(14)
        click_style = _AbsoluteClickSliderStyle(slider.style())
        click_style.setParent(slider)
        slider.setStyle(click_style)
        slider.setStyleSheet(
            "QSlider::groove:horizontal { background: #11172a; height: 3px; }"
            "QSlider::handle:horizontal { background: #00f5ff; width: 10px; height: 10px; margin: -4px 0; }"
        )

        pmin = _safe_float(prop.get("min", 0), 0.0)
        pmax = _safe_float(prop.get("max", 1), 1.0)
        step = _safe_float(prop.get("step", 0.01), 0.01)
        value = _safe_float(prop.get("value", prop.get("default", pmin)), pmin)
        if pmax < pmin:
            pmin, pmax = pmax, pmin
        if step <= 0:
            step = 0.01
        value = max(pmin, min(pmax, value))

        # Calculate number of steps based on step property
        span = max(pmax - pmin, 0.0)
        steps = int(span / step) if step > 0 else 1000
        if steps < 1:
            steps = 1
        # Clamp steps to avoid huge slider ranges that can crash Qt.
        if steps > 1_000_000:
            print(
                f"Warning: clamping range steps for {self.node_type}.{key} "
                f"(span={span}, step={step})"
            )
            steps = 1_000_000
        
        slider.setMinimum(0)
        slider.setMaximum(max(steps, 1))

        if use_log:
            import math as _m
            log_min = _m.log10(pmin)
            log_max = _m.log10(pmax)
            log_val = _m.log10(max(value, pmin))
            slider.setValue(int((log_val - log_min) / (log_max - log_min) * steps))
        else:
            if pmax != pmin:
                # Calculate slider position, ensuring proper mapping to step boundaries
                if step > 0:
                    # Normalize and round to get the correct step position
                    normalized = (value - pmin) / (pmax - pmin)
                    step_position = round(normalized * steps)
                    slider.setValue(min(max(step_position, 0), steps))
                else:
                    slider.setValue(int((value - pmin) / (pmax - pmin) * steps))

        def on_slider_change(pos: int) -> None:
            if use_log:
                log_min_ = _m.log10(pmin)
                log_max_ = _m.log10(pmax)
                real = 10 ** (log_min_ + (pos / steps) * (log_max_ - log_min_))
            else:
                # Calculate raw value from position
                if steps > 0:
                    raw = pmin + (pos / steps) * (pmax - pmin)
                else:
                    raw = pmin
                
                # Round to step size using division to avoid floating point errors
                if step > 0:
                    # Use round() on the ratio, not on the result to minimize precision issues
                    raw_steps = round(raw / step - pmin / step)
                    real = pmin + raw_steps * step
                else:
                    real = raw
            
            # Clamp the value to valid range
            real = max(pmin, min(pmax, real))
            if not _m.isfinite(real):
                real = pmin

            val_lbl.setText(f"{real:.4f}" if prop.get("scale") == "log" else f"{real:.3f}")
            self.signals.field_changed.emit(self.node_id, key, real)

        slider.valueChanged.connect(on_slider_change)
        layout.addWidget(slider)

        container.setFixedHeight(30)
        return container

    def _make_integer_widget(self, prop: dict, key: str) -> QWidget:
        container = QWidget()
        container.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        label_row = QWidget()
        label_row.setStyleSheet("background: transparent;")
        lr = QHBoxLayout(label_row)
        lr.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(prop.get("label", key))
        lbl.setStyleSheet("color: #8fa4ff; font-size: 8px; background: transparent;")
        lbl.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        val_lbl = QLabel(str(prop.get("value", prop.get("default", 0))))
        val_lbl.setStyleSheet("color: #00f5ff; font-size: 7px; font-family: 'JetBrains Mono'; background: transparent;")
        val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        lr.addWidget(lbl)
        lr.addWidget(val_lbl)
        layout.addWidget(label_row)

        spin = QSpinBox()
        spin.setFixedHeight(18)
        spin.setStyleSheet(
            "QSpinBox { background: #11172a; border: 1px solid #26324d; color: #e6f1ff; "
            "font-size: 8px; padding: 0 4px; }"
            "QSpinBox:focus { border-color: #00f5ff; }"
            "QSpinBox::up-button, QSpinBox::down-button { width: 0; }"
        )
        if prop.get("min") is not None:
            spin.setMinimum(prop["min"])
        else:
            spin.setMinimum(-999999)
        if prop.get("max") is not None:
            spin.setMaximum(prop["max"])
        else:
            spin.setMaximum(999999)
        spin.setValue(int(prop.get("value", prop.get("default", 0))))
        self._disable_widget_right_click(spin)
        line_edit = spin.lineEdit()
        if line_edit:
            self._disable_widget_right_click(line_edit)

        def on_spin_change(v: int) -> None:
            val_lbl.setText(str(v))
            self.signals.field_changed.emit(self.node_id, key, v)

        spin.valueChanged.connect(on_spin_change)
        layout.addWidget(spin)

        container.setFixedHeight(34)
        return container

    def _make_float_widget(self, prop: dict, key: str) -> QWidget:
        container = QWidget()
        container.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        label_row = QWidget()
        label_row.setStyleSheet("background: transparent;")
        lr = QHBoxLayout(label_row)
        lr.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(prop.get("label", key))
        lbl.setStyleSheet("color: #8fa4ff; font-size: 8px; background: transparent;")
        lbl.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        val_lbl = QLabel(str(prop.get("value", prop.get("default", 0.0))))
        val_lbl.setStyleSheet("color: #00f5ff; font-size: 7px; font-family: 'JetBrains Mono'; background: transparent;")
        val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        lr.addWidget(lbl)
        lr.addWidget(val_lbl)
        layout.addWidget(label_row)

        spin = QDoubleSpinBox()
        spin.setFixedHeight(18)
        spin.setStyleSheet(
            "QDoubleSpinBox { background: #11172a; border: 1px solid #26324d; color: #e6f1ff; "
            "font-size: 8px; padding: 0 4px; }"
            "QDoubleSpinBox:focus { border-color: #00f5ff; }"
            "QDoubleSpinBox::up-button, QDoubleSpinBox::down-button { width: 0; }"
        )
        spin.setMinimum(-999999.0)
        spin.setMaximum(999999.0)
        spin.setDecimals(6)
        spin.setValue(float(prop.get("value", prop.get("default", 0.0))))
        self._disable_widget_right_click(spin)
        line_edit = spin.lineEdit()
        if line_edit:
            self._disable_widget_right_click(line_edit)

        def on_spin_change(v: float) -> None:
            val_lbl.setText(f"{v:.3f}")
            self.signals.field_changed.emit(self.node_id, key, v)

        spin.valueChanged.connect(on_spin_change)
        layout.addWidget(spin)

        container.setFixedHeight(34)
        return container

    def _disable_widget_right_click(self, widget: QWidget) -> None:
        widget.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        blocker = _DisableRightClickFilter(widget)
        widget.installEventFilter(blocker)
        self._right_click_filters.append(blocker)

    def _make_bool_widget(self, prop: dict, key: str) -> QWidget:
        container = QWidget()
        container.setStyleSheet("background: transparent;")
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        cb = QCheckBox(prop.get("label", key))
        cb.setChecked(bool(prop.get("value", prop.get("default", False))))
        cb.setStyleSheet(
            "QCheckBox { color: #8fa4ff; font-size: 8px; background: transparent; }"
            "QCheckBox::indicator { width: 12px; height: 12px; border: 1px solid #26324d; background: #11172a; }"
            "QCheckBox::indicator:checked { background: #00f5ff; border-color: #00f5ff; }"
        )

        def on_check(state: int) -> None:
            self.signals.field_changed.emit(self.node_id, key, state == Qt.CheckState.Checked.value)

        cb.stateChanged.connect(on_check)
        layout.addWidget(cb)

        container.setFixedHeight(18)
        return container

    def _make_enum_widget(self, prop: dict, key: str) -> QWidget:
        container = QWidget()
        container.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        lbl = QLabel(prop.get("label", key))
        lbl.setStyleSheet("color: #8fa4ff; font-size: 8px; background: transparent;")
        layout.addWidget(lbl)

        button = QPushButton()
        button.setFixedHeight(20)
        button.setStyleSheet(
            "QPushButton { background: #11172a; border: 1px solid #26324d; color: #e6f1ff; "
            "font-size: 8px; padding: 0 6px; text-align: left; }"
            "QPushButton:hover { border-color: #00f5ff; }"
        )
        self._disable_widget_right_click(button)

        menu = QMenu()
        menu.setWindowFlag(Qt.WindowType.Popup, True)
        menu.setStyleSheet(
            "QMenu { background: #11172a; border: 1px solid #26324d; color: #e6f1ff; }"
            "QMenu::item { padding: 4px 14px; }"
            "QMenu::item:selected { background: #141c2f; color: #00f5ff; }"
        )
        self._popup_menus.append(menu)

        current = prop.get("value", prop.get("default", ""))
        button.setText(str(current))

        def _apply_value(text: str) -> None:
            button.setText(text)
            self.signals.field_changed.emit(self.node_id, key, text)

        for opt in prop.get("options", []):
            action = menu.addAction(opt)
            action.triggered.connect(lambda checked=False, text=opt: _apply_value(text))

        def _show_menu() -> None:
            menu.exec(QCursor.pos())

        button.clicked.connect(_show_menu)
        layout.addWidget(button)

        container.setFixedHeight(34)
        return container

    def _make_directory_path_widget(self, prop: dict, key: str) -> QWidget:
        container = QWidget()
        container.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        lbl = QLabel(prop.get("label", key))
        lbl.setStyleSheet("color: #8fa4ff; font-size: 8px; background: transparent;")
        layout.addWidget(lbl)

        row = QWidget()
        row.setStyleSheet("background: transparent;")
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(4)

        line_edit = QLineEdit()
        line_edit.setReadOnly(True)
        line_edit.setText(str(prop.get("value", prop.get("default", "")) or ""))
        line_edit.setPlaceholderText("No directory selected")
        line_edit.setToolTip(line_edit.text())
        line_edit.setStyleSheet(
            "QLineEdit { background: #11172a; border: 1px solid #26324d; color: #e6f1ff; "
            "font-size: 8px; padding: 0 4px; }"
        )
        self._disable_widget_right_click(line_edit)

        browse_btn = QPushButton("...")
        browse_btn.setFixedWidth(28)
        browse_btn.setFixedHeight(20)
        browse_btn.setStyleSheet(
            "QPushButton { background: #11172a; border: 1px solid #26324d; color: #e6f1ff; font-size: 8px; }"
            "QPushButton:hover { border-color: #00f5ff; color: #00f5ff; }"
        )
        self._disable_widget_right_click(browse_btn)

        def on_browse() -> None:
            start_dir = line_edit.text().strip() or self._dialog_project_dir()
            selected = QFileDialog.getExistingDirectory(
                None,
                prop.get("label", key),
                start_dir,
            )
            if not selected:
                return
            line_edit.setText(selected)
            line_edit.setToolTip(selected)
            self.signals.field_changed.emit(self.node_id, key, selected)

        browse_btn.clicked.connect(on_browse)
        row_layout.addWidget(line_edit, 1)
        row_layout.addWidget(browse_btn, 0)
        layout.addWidget(row)

        container.setFixedHeight(42)
        return container

    def _make_file_path_widget(self, prop: dict, key: str) -> QWidget:
        container = QWidget()
        container.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        lbl = QLabel(prop.get("label", key))
        lbl.setStyleSheet("color: #8fa4ff; font-size: 8px; background: transparent;")
        layout.addWidget(lbl)

        row = QWidget()
        row.setStyleSheet("background: transparent;")
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(4)

        line_edit = QLineEdit()
        line_edit.setReadOnly(True)
        line_edit.setText(str(prop.get("value", prop.get("default", "")) or ""))
        line_edit.setPlaceholderText("No file selected")
        line_edit.setToolTip(line_edit.text())
        line_edit.setStyleSheet(
            "QLineEdit { background: #11172a; border: 1px solid #26324d; color: #e6f1ff; "
            "font-size: 8px; padding: 0 4px; }"
        )
        self._disable_widget_right_click(line_edit)

        browse_btn = QPushButton("...")
        browse_btn.setFixedWidth(28)
        browse_btn.setFixedHeight(20)
        browse_btn.setStyleSheet(
            "QPushButton { background: #11172a; border: 1px solid #26324d; color: #e6f1ff; font-size: 8px; }"
            "QPushButton:hover { border-color: #00f5ff; color: #00f5ff; }"
        )
        self._disable_widget_right_click(browse_btn)

        def on_browse() -> None:
            start_dir = self._dialog_project_dir()
            current_path = line_edit.text().strip()
            if current_path:
                start_dir = str(Path(current_path).expanduser().parent)
            selected, _ = QFileDialog.getOpenFileName(
                None,
                prop.get("label", key),
                start_dir,
                "All Files (*)",
            )
            if not selected:
                return
            line_edit.setText(selected)
            line_edit.setToolTip(selected)
            self.signals.field_changed.emit(self.node_id, key, selected)

        browse_btn.clicked.connect(on_browse)
        row_layout.addWidget(line_edit, 1)
        row_layout.addWidget(browse_btn, 0)
        layout.addWidget(row)

        container.setFixedHeight(42)
        return container

    def _make_file_paths_widget(self, prop: dict, key: str) -> QWidget:
        container = QWidget()
        container.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        lbl = QLabel(prop.get("label", key))
        lbl.setStyleSheet("color: #8fa4ff; font-size: 8px; background: transparent;")
        layout.addWidget(lbl)

        row = QWidget()
        row.setStyleSheet("background: transparent;")
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(4)

        values = prop.get("value", prop.get("default", [])) or []
        line_edit = QLineEdit()
        line_edit.setReadOnly(True)
        line_edit.setText(self._format_file_paths_text(values))
        line_edit.setPlaceholderText("No files selected")
        line_edit.setToolTip("\n".join(str(item) for item in values))
        line_edit.setStyleSheet(
            "QLineEdit { background: #11172a; border: 1px solid #26324d; color: #e6f1ff; "
            "font-size: 8px; padding: 0 4px; }"
        )
        self._disable_widget_right_click(line_edit)

        browse_btn = QPushButton("...")
        browse_btn.setFixedWidth(28)
        browse_btn.setFixedHeight(20)
        browse_btn.setStyleSheet(
            "QPushButton { background: #11172a; border: 1px solid #26324d; color: #e6f1ff; font-size: 8px; }"
            "QPushButton:hover { border-color: #00f5ff; color: #00f5ff; }"
        )
        self._disable_widget_right_click(browse_btn)

        def on_browse() -> None:
            start_dir = self._dialog_project_dir()
            tooltip_lines = [line for line in line_edit.toolTip().splitlines() if line.strip()]
            if tooltip_lines:
                start_dir = str(Path(tooltip_lines[0]).expanduser().parent)
            selected, _ = QFileDialog.getOpenFileNames(
                None,
                prop.get("label", key),
                start_dir,
                "All Files (*)",
            )
            if not selected:
                return
            line_edit.setText(self._format_file_paths_text(selected))
            line_edit.setToolTip("\n".join(selected))
            self.signals.field_changed.emit(self.node_id, key, selected)

        browse_btn.clicked.connect(on_browse)
        row_layout.addWidget(line_edit, 1)
        row_layout.addWidget(browse_btn, 0)
        layout.addWidget(row)

        container.setFixedHeight(42)
        return container

    def _dialog_project_dir(self) -> str:
        scene = self.scene()
        bridge = getattr(scene, "bridge", None)
        project_dir = getattr(bridge, "project_dir", None)
        return str(project_dir) if project_dir is not None else ""

    def _format_file_paths_text(self, values: List[Any]) -> str:
        if not values:
            return ""
        text_values = [str(item) for item in values]
        if len(text_values) == 1:
            return text_values[0]
        return f"{len(text_values)} files"

    # ── Layout ───────────────────────────────────────────────────────────

    def _layout(self) -> None:
        """Compute positions for all sub-elements."""
        # QGraphicsScene must be notified before this item's bounding rect changes.
        self.prepareGeometryChange()
        y = 0.0

        # Header
        self._layout_title_editor()
        y += HEADER_HEIGHT

        # I/O ports
        n_inputs = len(self.input_ports)
        n_outputs = len(self.output_ports)
        n_port_rows = max(n_inputs, n_outputs)
        if n_port_rows > 0:
            io_top = y
            label_right_margin = 14.0
            center_gap = 2.0
            port_grab_margin = 6.0
            for i, port in enumerate(self.input_ports):
                row_y = io_top + i * PORT_ROW_HEIGHT
                port.setPos(0, row_y + PORT_ROW_HEIGHT / 2)
                zone = self._port_hit_zones.get(f"input:{port.port_name}")
                if zone:
                    zone.set_rect(QRectF(
                        -port_grab_margin,
                        row_y,
                        self._width / 2 - label_right_margin + port_grab_margin,
                        PORT_ROW_HEIGHT,
                    ))
            for i, port in enumerate(self.output_ports):
                row_y = io_top + i * PORT_ROW_HEIGHT
                port.setPos(self._width, row_y + PORT_ROW_HEIGHT / 2)
                zone = self._port_hit_zones.get(f"output:{port.port_name}")
                if zone:
                    zone.set_rect(QRectF(
                        self._width / 2 + center_gap,
                        row_y,
                        self._width / 2 - center_gap + port_grab_margin,
                        PORT_ROW_HEIGHT,
                    ))
            y += n_port_rows * PORT_ROW_HEIGHT

        # Optional process-error partition directly below ports.
        self._error_partition_rect = None
        if self._error_message:
            text_width = max(0.0, self._width - 2 * (PADDING + 4))
            fm = QFontMetrics(self._small_font)
            wrapped = fm.boundingRect(
                0, 0, int(text_width), 0,
                int(Qt.TextFlag.TextWordWrap),
                self._error_message,
            )
            panel_h = max(PORT_ROW_HEIGHT, float(wrapped.height()) + 8.0)
            self._error_partition_rect = QRectF(0, y, self._width, panel_h)
            y += panel_h

        self._body_y = y

        # Field widgets
        for proxy in self._proxy_widgets:
            w = proxy.widget()
            if w:
                proxy.setPos(PADDING, y)
                w.setFixedWidth(int(self._width - 2 * PADDING))
                y += w.height() + 2

        # Display outputs
        self._display_rects.clear()
        for output in self.schema.get("outputs", []):
            key = output.get("key", "")
            enabled = output.get("enabled", True)
            otype = output.get("type", "")

            # Toggle label height
            y += 14

            if not enabled:
                self._display_rects[key] = QRectF(PADDING, y, self._width - 2 * PADDING, 0)
                continue

            if otype in ("heatmap", "image"):
                data = self._display_cache.get(key)
                if data is not None:
                    if isinstance(data, list):
                        data = np.array(data)

                    if hasattr(data, "shape") and len(data.shape) >= 1:
                        if len(data.shape) == 1:
                            rows, cols = 1, data.shape[0]
                        else:
                            rows, cols = data.shape[:2]
                        if cols > 0:
                            aspect = rows / cols
                            h = max(24.0, (self._width - 2 * PADDING) * aspect)
                        else:
                            h = self._width - 2 * PADDING
                    else:
                        h = self._width - 2 * PADDING
                else:
                    h = self._width - 2 * PADDING  # default square
                self._display_rects[key] = QRectF(PADDING, y, self._width - 2 * PADDING, h)
                y += h + 2
            elif otype == "plot":
                h = PLOT_DISPLAY_HEIGHT
                self._display_rects[key] = QRectF(PADDING, y, self._width - 2 * PADDING, h)
                y += h + 2
            elif otype == "numeric":
                h = 16
                self._display_rects[key] = QRectF(PADDING, y, self._width - 2 * PADDING, h)
                y += h + 2
            elif otype == "text":
                text_value = self._display_cache.get(
                    key,
                    output.get("value", output.get("default", "")),
                )
                max_h = max(48.0, float(output.get("max_height", 140.0)))
                content_h = self._calculate_text_height(text_value, self._width - 2 * PADDING - 8)
                h = min(content_h, max_h)
                self._display_rects[key] = QRectF(PADDING, y, self._width - 2 * PADDING, h)
                proxy = self._text_display_proxies.get(key)
                widget = self._text_display_widgets.get(key)
                if proxy and widget:
                    widget_width = int(self._width - 2 * PADDING)
                    content_h = self._calculate_text_widget_height(widget, text_value, widget_width)
                    h = min(content_h, max_h)
                    self._display_rects[key] = QRectF(PADDING, y, self._width - 2 * PADDING, h)
                    proxy.setPos(PADDING, y)
                    widget.setFixedWidth(widget_width)
                    widget.setFixedHeight(max(20, int(h)))
                    needs_scroll = content_h > max_h + 0.5
                    widget.setVerticalScrollBarPolicy(
                        Qt.ScrollBarPolicy.ScrollBarAsNeeded
                        if needs_scroll
                        else Qt.ScrollBarPolicy.ScrollBarAlwaysOff
                    )
                    proxy.setVisible(bool(output.get("enabled", True)) and h > 0)
                y += h + 2

        # State information line at the bottom.
        self._state_hint_rect = None
        self._state_hint_text = ""
        states = self.schema.get("states", [])
        if states:
            labels: List[str] = []
            for entry in states:
                label = str(entry.get("label") or entry.get("key") or "").strip()
                if label:
                    labels.append(label)
            if labels:
                self._state_hint_text = "State: " + ", ".join(labels)
                state_width = max(0.0, self._width - 2 * PADDING)
                fm = QFontMetrics(self._small_font)
                wrapped = fm.boundingRect(
                    0, 0, int(state_width), 0,
                    int(Qt.TextFlag.TextWordWrap),
                    self._state_hint_text,
                )
                state_h = max(12.0, float(wrapped.height()))
                self._state_hint_rect = QRectF(PADDING, y, state_width, state_h)
                y += state_h + 2

        y += PADDING
        self._height = y

    def _layout_title_editor(self) -> None:
        if self._title_edit_proxy is None or self._title_edit is None:
            return
        rect = self._title_text_rect()
        edit_h = max(18.0, HEADER_HEIGHT - 6.0)
        self._title_edit.setFixedSize(max(40, int(rect.width())), int(edit_h))
        self._title_edit_proxy.setPos(rect.x(), (HEADER_HEIGHT - edit_h) / 2.0)

    # ── Geometry ─────────────────────────────────────────────────────────

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self._width, self._height)

    # ── Painting ─────────────────────────────────────────────────────────

    def _apply_loading_ui_state(self) -> None:
        """Enable/disable embedded widgets while loading."""
        enabled = not self._loading
        for proxy in self._proxy_widgets:
            widget = proxy.widget()
            if widget:
                widget.setEnabled(enabled)
        for widget in self._text_display_widgets.values():
            widget.setEnabled(enabled)

    def _advance_loading_spinner(self) -> None:
        self._loading_spinner_angle = (self._loading_spinner_angle + 14.0) % 360.0
        self.update()

    def _title_text_rect(self) -> QRectF:
        reload_reserve = 24.0 if self._dynamic else 8.0
        return QRectF(
            PADDING + 4.0,
            0.0,
            max(40.0, self._width - 2 * PADDING - reload_reserve),
            HEADER_HEIGHT,
        )

    def _begin_title_rename(self) -> None:
        if self._loading or self._title_edit is None or self._title_edit_proxy is None:
            return
        self._editing_title = True
        self._title_edit.setText(self._title)
        self._title_edit_proxy.setVisible(True)
        self._title_edit.setFocus()
        self._title_edit.selectAll()

    def _finish_title_rename(self) -> str:
        self._editing_title = False
        if self._title_edit_proxy is not None:
            self._title_edit_proxy.setVisible(False)
        if self._title_edit is None:
            return self._title
        return self._title_edit.text().strip() or self.node_type

    def _commit_title_rename(self) -> None:
        if not self._editing_title:
            return
        new_name = self._finish_title_rename()
        if new_name != self._title:
            self.signals.name_change_requested.emit(self.node_id, new_name)

    def cancel_title_rename(self) -> None:
        if not self._editing_title:
            return
        self._finish_title_rename()
        if self._title_edit is not None:
            self._title_edit.setText(self._title)

    def set_display_name(self, name: str) -> None:
        self._title = str(name).strip() or self.node_type
        self.schema["name"] = self._title
        if self._title_edit is not None:
            self._title_edit.setText(self._title)
        self.update()

    def set_loading_state(self, is_loading: bool) -> None:
        """Set whether this node is currently background-initializing."""
        if self._loading == is_loading:
            return
        self._loading = is_loading
        self._apply_loading_ui_state()
        if self._title_edit is not None:
            self._title_edit.setEnabled(not self._loading)
        if self._loading:
            self._loading_spinner_angle = 0.0
            self._loading_timer.start()
        else:
            self._loading_timer.stop()
        self.update()

    def set_error_state(self, has_error: bool, message: str = "") -> None:
        """
        Set error state and optional process-error message.

        When message is non-empty, an error partition is rendered below the I/O ports.
        """
        next_message = message if has_error else ""
        if self._error_state != has_error or self._error_message != next_message:
            self._error_state = has_error
            self._error_message = next_message
            self._layout()
            self.update()

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget=None) -> None:
        rect = self.boundingRect()

        # Node background
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(BG_NODE))
        painter.drawRect(rect)

        # Border - error state takes precedence
        if self._error_state:
            pen = QPen(ERROR_COLOR, 2)  # Thicker red border for errors
        elif self.isSelected():
            pen = QPen(BORDER_SELECTED, 1)
        elif self._hovered:
            pen = QPen(BORDER_HOVER, 1)
        else:
            pen = QPen(BORDER_COLOR, 1)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(rect)

        # Header background
        header_rect = QRectF(0, 0, self._width, HEADER_HEIGHT)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(BG_HEADER))
        painter.drawRect(header_rect)

        # Header bottom border
        painter.setPen(QPen(BORDER_COLOR, 1))
        painter.drawLine(QPointF(0, HEADER_HEIGHT), QPointF(self._width, HEADER_HEIGHT))

        # Category colour stripe on header left
        if self._category:
            cat_color = _generate_category_color(self._category)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(cat_color))
            painter.drawRect(QRectF(0, 0, 3, HEADER_HEIGHT))

        # Title text: user-defined name plus node type.
        title_rect = self._title_text_rect()
        type_text = f"({self.node_type})"
        type_gap = 6.0
        type_font = self._small_font
        type_metrics = QFontMetrics(type_font)
        type_width = min(
            type_metrics.horizontalAdvance(type_text),
            max(0, int(title_rect.width() * 0.45)),
        )
        name_rect = QRectF(title_rect)
        if type_width > 0:
            name_rect.setWidth(max(32.0, title_rect.width() - type_width - type_gap))

        painter.setPen(QPen(TEXT_PRIMARY))
        painter.setFont(self._title_font)
        name_metrics = QFontMetrics(self._title_font)
        name_text = name_metrics.elidedText(
            self._title,
            Qt.TextElideMode.ElideRight,
            int(name_rect.width()),
        )
        painter.drawText(
            name_rect,
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            name_text,
        )

        if type_width > 0:
            name_draw_width = min(
                name_metrics.horizontalAdvance(name_text),
                int(name_rect.width()),
            )
            type_x = title_rect.x() + name_draw_width + type_gap
            type_rect = QRectF(
                type_x,
                0.0,
                max(0.0, title_rect.right() - type_x),
                HEADER_HEIGHT,
            )
            painter.setPen(QPen(TEXT_SECONDARY))
            painter.setFont(type_font)
            painter.drawText(
                type_rect,
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                type_metrics.elidedText(
                    type_text,
                    Qt.TextElideMode.ElideRight,
                    int(type_rect.width()),
                ),
            )

        # Dynamic reload button indicator
        if self._dynamic:
            painter.setPen(QPen(TEXT_SECONDARY))
            painter.setFont(self._label_font)
            painter.drawText(
                QRectF(self._width - 22, 0, 18, HEADER_HEIGHT),
                Qt.AlignmentFlag.AlignCenter, "↻"
            )

        # I/O port section
        n_inputs = len(self.input_ports)
        n_outputs = len(self.output_ports)
        n_port_rows = max(n_inputs, n_outputs)
        if n_port_rows > 0:
            io_top = HEADER_HEIGHT
            io_bottom = io_top + n_port_rows * PORT_ROW_HEIGHT

            # Separator line
            painter.setPen(QPen(BORDER_COLOR, 1))
            painter.drawLine(QPointF(0, io_bottom), QPointF(self._width, io_bottom))

            # Port labels
            painter.setFont(self._port_font)
            for i, port in enumerate(self.input_ports):
                label = port.label
                # Show "..." for multi-type ports, otherwise show type
                if port.data_types:
                    type_display = "..."
                elif port.data_type and port.data_type != "any":
                    type_display = port.data_type
                else:
                    type_display = None
                text = f"{label} ({type_display})" if type_display else label
                painter.setPen(QPen(TEXT_SECONDARY))
                r = QRectF(12, io_top + i * PORT_ROW_HEIGHT, self._width / 2 - 14, PORT_ROW_HEIGHT)
                painter.drawText(r, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, text)

            for i, port in enumerate(self.output_ports):
                label = port.label
                # Show "..." for multi-type ports, otherwise show type
                if port.data_types:
                    type_display = "..."
                elif port.data_type and port.data_type != "any":
                    type_display = port.data_type
                else:
                    type_display = None
                text = f"{label} ({type_display})" if type_display else label
                painter.setPen(QPen(TEXT_SECONDARY))
                r = QRectF(self._width / 2 + 2, io_top + i * PORT_ROW_HEIGHT, self._width / 2 - 14, PORT_ROW_HEIGHT)
                painter.drawText(r, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, text)

        # Error partition (process errors) under port rows.
        if self._error_partition_rect:
            panel = self._error_partition_rect
            painter.setPen(QPen(ERROR_PANEL_BORDER, 1))
            painter.setBrush(QBrush(ERROR_PANEL_BG))
            painter.drawRect(panel)
            painter.setPen(QPen(ERROR_TEXT))
            painter.setFont(self._small_font)
            painter.drawText(
                panel.adjusted(PADDING, 2, -PADDING, -2),
                int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap),
                self._error_message,
            )

        # Display outputs
        for output in self.schema.get("outputs", []):
            key = output.get("key", "")
            otype = output.get("type", "")
            enabled = output.get("enabled", True)
            rect_d = self._display_rects.get(key)
            if not rect_d:
                continue

            # Draw toggle label above display
            label_y = rect_d.y() - 14
            painter.setFont(self._small_font)
            painter.setPen(QPen(TEXT_SECONDARY))
            painter.drawText(
                QRectF(PADDING + 14, label_y, self._width - 2 * PADDING - 14, 14),
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                output.get("label", key),
            )
            # Checkbox indicator
            cb_rect = QRectF(PADDING, label_y + 2, 10, 10)
            painter.setPen(QPen(BORDER_COLOR, 1))
            if enabled:
                painter.setBrush(QBrush(ACCENT))
            else:
                painter.setBrush(QBrush(BG_SECONDARY))
            painter.drawRect(cb_rect)

            if not enabled or rect_d.height() == 0:
                continue

            # Draw the actual display
            if otype == "heatmap":
                self._paint_heatmap(painter, rect_d, key, output)
            elif otype == "image":
                self._paint_image(painter, rect_d, key)
            elif otype == "plot":
                self._paint_plot(painter, rect_d, key, output)
            elif otype == "numeric":
                self._paint_numeric(painter, rect_d, key, output)

        # State hint at node bottom.
        if self._state_hint_rect and self._state_hint_text:
            painter.setPen(QPen(STATE_TEXT))
            painter.setFont(self._small_font)
            painter.drawText(
                self._state_hint_rect,
                int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap),
                self._state_hint_text,
            )

        # Loading overlay on top of node contents.
        if self._loading:
            painter.save()
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(LOADING_OVERLAY_BG))
            painter.drawRect(rect)

            # Subtle horizontal streaks to mimic a blurred/frosted layer.
            painter.setPen(QPen(QColor(210, 225, 255, 14), 1))
            y_line = 0.0
            while y_line < rect.height():
                painter.drawLine(QPointF(0, y_line), QPointF(rect.width(), y_line))
                y_line += 5.0

            center = rect.center()
            spinner_radius = 11.0
            spinner_rect = QRectF(
                center.x() - spinner_radius,
                center.y() - spinner_radius - 8.0,
                spinner_radius * 2.0,
                spinner_radius * 2.0,
            )

            painter.setPen(QPen(QColor(130, 160, 210, 140), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(spinner_rect)

            painter.setPen(QPen(ACCENT, 2.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            # 1/16 degree units; negative for clockwise visual motion.
            start = int(-self._loading_spinner_angle * 16.0)
            span = int(-120 * 16)
            painter.drawArc(spinner_rect, start, span)

            painter.setPen(QPen(LOADING_TEXT))
            painter.setFont(self._small_font)
            painter.drawText(
                QRectF(center.x() - 55.0, center.y() + 9.0, 110.0, 18.0),
                Qt.AlignmentFlag.AlignCenter,
                "Loading...",
            )
            painter.restore()

    # ── Display painting ─────────────────────────────────────────────────

    def _paint_heatmap(self, painter: QPainter, rect: QRectF, key: str, spec: dict) -> None:
        """Paint a 1D/2D scalar heatmap as a pixel image."""
        data = self._display_cache.get(key)
        if data is None or not isinstance(data, (np.ndarray, list)):
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(BG_SECONDARY))
            painter.drawRect(rect)
            return

        if isinstance(data, list):
            data = np.array(data)

        if not isinstance(data, np.ndarray):
            return

        if data.ndim not in (1, 2):
            return

        config = self._display_config.get(key, {})
        colormap = config.get("colormap", spec.get("colormap", "grayscale"))
        scale_mode = config.get("scale_mode", spec.get("scale_mode", "auto"))
        vmin = float(config.get("vmin", spec.get("vmin", 0.0)))
        vmax = float(config.get("vmax", spec.get("vmax", 1.0)))

        data = np.asarray(data, dtype=np.float32)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        rows, cols = data.shape

        if rows > MAX_DISPLAY_DIM or cols > MAX_DISPLAY_DIM:
            step_rows = (rows + MAX_DISPLAY_DIM - 1) // MAX_DISPLAY_DIM
            step_cols = (cols + MAX_DISPLAY_DIM - 1) // MAX_DISPLAY_DIM
            step = max(step_rows, step_cols)
            data = data[::step, ::step]
            rows, cols = data.shape

        finite_mask = np.isfinite(data)
        if not np.any(finite_mask):
            t = np.zeros_like(data, dtype=np.float32)
        else:
            finite_values = data[finite_mask]
            if scale_mode == "manual":
                low = vmin
                high = vmax
            else:
                low = float(np.min(finite_values))
                high = float(np.max(finite_values))
            if not np.isfinite(low):
                low = 0.0
            if not np.isfinite(high) or high <= low:
                high = low + 1.0
            t = np.clip((data - low) / (high - low), 0.0, 1.0)
            t = np.where(finite_mask, t, 0.0)

        if colormap == "viridis":
            rc = np.clip(255 * (0.267 + 0.329 * t - 0.868 * t**2 + 1.65 * t**3), 0, 255).astype(np.uint8)
            gc = np.clip(255 * (0.012 + 0.696 * t - 1.16 * t**2 + 0.98 * t**3), 0, 255).astype(np.uint8)
            bc = np.clip(255 * (0.909 - 0.298 * t - 1.45 * t**2 + 2.22 * t**3), 0, 255).astype(np.uint8)
        elif colormap in ("bwr", "diverging"):
            mask = t < 0.5
            k = np.where(mask, t / 0.5, (t - 0.5) / 0.5)
            rc = np.where(mask, 255 * k, 255).astype(np.uint8)
            gc = np.where(mask, 255 * k, 255 * (1 - k)).astype(np.uint8)
            bc = np.where(mask, 255, 255 * (1 - k)).astype(np.uint8)
        else:
            gray = (t * 255).astype(np.uint8)
            rc = gray
            gc = gray
            bc = gray

        rgb = np.stack([rc, gc, bc], axis=2)
        rgb = np.ascontiguousarray(rgb)
        self._display_rgb_cache[key] = rgb

        img = QImage(rgb.data, cols, rows, cols * 3, QImage.Format.Format_RGB888)
        self._display_images[key] = img
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        painter.drawImage(rect, img)
        painter.restore()

    def _paint_image(self, painter: QPainter, rect: QRectF, key: str) -> None:
        """Paint a grayscale/RGB/RGBA image."""
        data = self._display_cache.get(key)
        if data is None or not isinstance(data, (np.ndarray, list)):
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(BG_SECONDARY))
            painter.drawRect(rect)
            return

        image_data = np.asarray(data)
        if image_data.ndim == 2:
            image_data = image_data[:, :, None]
        if image_data.ndim != 3 or image_data.shape[2] not in (1, 3, 4):
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(BG_SECONDARY))
            painter.drawRect(rect)
            return

        rows, cols = image_data.shape[:2]
        if rows > MAX_DISPLAY_DIM or cols > MAX_DISPLAY_DIM:
            step_rows = (rows + MAX_DISPLAY_DIM - 1) // MAX_DISPLAY_DIM
            step_cols = (cols + MAX_DISPLAY_DIM - 1) // MAX_DISPLAY_DIM
            step = max(step_rows, step_cols)
            image_data = image_data[::step, ::step]
            rows, cols = image_data.shape[:2]

        if np.issubdtype(image_data.dtype, np.floating):
            finite = image_data[np.isfinite(image_data)]
            max_value = float(np.max(finite)) if finite.size else 0.0
            min_value = float(np.min(finite)) if finite.size else 0.0
            if 0.0 <= min_value and max_value <= 1.0:
                image_data = np.clip(image_data, 0.0, 1.0) * 255.0
            else:
                image_data = np.clip(image_data, 0.0, 255.0)
        else:
            image_data = np.clip(image_data, 0, 255)

        image_data = np.where(np.isfinite(image_data), image_data, 0)
        image_data = np.ascontiguousarray(image_data.astype(np.uint8))

        if image_data.shape[2] == 1:
            gray = np.repeat(image_data, 3, axis=2)
            self._display_rgb_cache[key] = gray
            img = QImage(gray.data, cols, rows, cols * 3, QImage.Format.Format_RGB888)
        elif image_data.shape[2] == 4:
            self._display_rgb_cache[key] = image_data
            img = QImage(image_data.data, cols, rows, cols * 4, QImage.Format.Format_RGBA8888)
        else:
            self._display_rgb_cache[key] = image_data
            img = QImage(image_data.data, cols, rows, cols * 3, QImage.Format.Format_RGB888)

        self._display_images[key] = img
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        painter.drawImage(rect, img)
        painter.restore()

    # ── Plot pan/zoom helpers ─────────────────────────────────────────────

    def _plot_rect_for_display(self, key: str) -> Optional[QRectF]:
        """Compute the inner plot area from the display rect."""
        rect = self._display_rects.get(key)
        if rect is None:
            return None
        return QRectF(
            rect.x() + 30.0,        # y_label_w
            rect.y() + 12.0,        # top_pad
            rect.width() - 30.0 - 2,
            rect.height() - 12.0 - 10.0,  # top_pad + x_label_h
        )

    def _find_plot_key_at(self, pos: QPointF) -> Optional[str]:
        """Find which plot display contains the given position."""
        for output in self.schema.get("outputs", []):
            if output.get("type") != "plot":
                continue
            key = output.get("key", "")
            rect = self._display_rects.get(key)
            if rect and rect.contains(pos):
                return key
        return None

    def _get_plot_style(self, key: str) -> str:
        """Get the plot style for a display key."""
        config = self._display_config.get(key, {})
        for output in self.schema.get("outputs", []):
            if output.get("key") == key:
                return str(config.get("style", output.get("style", "line")))
        return "line"

    def _auto_range_for_plot(self, key: str) -> Optional[dict]:
        """Compute auto-fit range from cached data."""
        data = self._display_cache.get(key)
        if data is None:
            return None
        data = np.asarray(data, dtype=np.float32)
        style = self._get_plot_style(key)

        if style == "scatter":
            if data.ndim == 1:
                data = data.reshape(-1, 2) if data.size >= 2 else np.empty((0, 2))
            if data.ndim != 2 or data.shape[1] != 2 or data.shape[0] == 0:
                return None
            xs, ys = data[:, 0], data[:, 1]
            xmin, xmax = float(np.nanmin(xs)), float(np.nanmax(xs))
            ymin, ymax = float(np.nanmin(ys)), float(np.nanmax(ys))
            if xmin == xmax:
                xmax = xmin + 1.0
            if ymin == ymax:
                ymax = ymin + 1.0
            xpad = max((xmax - xmin) * 0.05, 1e-6)
            ypad = max((ymax - ymin) * 0.05, 1e-6)
            return {"xmin": xmin - xpad, "xmax": xmax + xpad,
                    "ymin": ymin - ypad, "ymax": ymax + ypad}
        else:
            data = data.flatten()
            if len(data) == 0:
                return None
            dmin, dmax = float(np.nanmin(data)), float(np.nanmax(data))
            if dmin == dmax:
                dmax = dmin + 1.0
            return {"ymin": dmin, "ymax": dmax}

    def _paint_plot(self, painter: QPainter, rect: QRectF, key: str, spec: dict) -> None:
        """Paint a 1D plot with grid, axis ticks, and value labels."""
        data = self._display_cache.get(key)
        if data is None:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(BG_SECONDARY))
            painter.drawRect(rect)
            return

        if isinstance(data, list):
            data = np.array(data, dtype=np.float32)
        if not isinstance(data, np.ndarray):
            data = np.asarray(data, dtype=np.float32)

        config = self._display_config.get(key, {})
        style = str(config.get("style", spec.get("style", "line")))

        # Reserve margin for Y-axis tick labels on the left
        tick_font = QFont(MONO_FAMILY, 5)
        tick_fm = QFontMetrics(tick_font)
        y_label_w = 30.0  # fixed width for y-axis labels
        x_label_h = 10.0  # space for bottom tick labels

        # Background
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(BG_SECONDARY))
        painter.drawRect(rect)

        # Plot area (inset from rect to leave room for labels)
        top_pad = 12.0  # room for value annotation at top
        plot_rect = QRectF(
            rect.x() + y_label_w,
            rect.y() + top_pad,
            rect.width() - y_label_w - 2,
            rect.height() - top_pad - x_label_h,
        )

        if style == "scatter":
            # Scatter expects Nx2 data, don't flatten
            if data.ndim == 1:
                data = data.reshape(-1, 2) if data.size >= 2 else np.empty((0, 2), dtype=np.float32)
            if data.ndim != 2 or data.shape[1] != 2 or data.shape[0] == 0:
                return
            self._paint_plot_scatter(painter, rect, plot_rect, data, config, spec, tick_font, key)
        else:
            data = data.flatten()
            if len(data) == 0:
                return
            if style == "bar":
                self._paint_plot_bar(painter, rect, plot_rect, data, config, spec, tick_font, key)
            else:
                self._paint_plot_line(painter, rect, plot_rect, data, config, spec, tick_font, key)

    @staticmethod
    def _nice_ticks(dmin: float, dmax: float, max_ticks: int = 5) -> list[float]:
        """Compute clean tick positions between dmin and dmax."""
        if dmin == dmax:
            return [dmin]
        rng = dmax - dmin
        rough_step = rng / max(max_ticks - 1, 1)
        # Round to a 'nice' step: 1, 2, 5, 10, 20, 50, ...
        mag = 10 ** math.floor(math.log10(rough_step)) if rough_step > 0 else 1.0
        residual = rough_step / mag
        if residual <= 1.5:
            nice = 1.0
        elif residual <= 3.5:
            nice = 2.0
        elif residual <= 7.5:
            nice = 5.0
        else:
            nice = 10.0
        step = nice * mag

        start = math.floor(dmin / step) * step
        ticks = []
        v = start
        while v <= dmax + step * 0.01:
            if dmin <= v <= dmax:
                ticks.append(v)
            v += step
        return ticks if ticks else [dmin, dmax]

    @staticmethod
    def _format_tick(v: float) -> str:
        """Format a tick value concisely."""
        av = abs(v)
        if av == 0:
            return "0"
        if av >= 1000:
            return f"{v:.0f}"
        if av >= 1:
            return f"{v:.1f}"
        if av >= 0.01:
            return f"{v:.2f}"
        return f"{v:.1e}"

    def _paint_plot_grid_and_ticks(
        self, painter: QPainter, rect: QRectF, plot_rect: QRectF,
        dmin: float, dmax: float, n: int, tick_font: QFont,
    ) -> None:
        """Draw grid lines and axis tick labels for a plot."""
        grid_color = QColor("#26324d")
        tick_color = QColor("#6b7fa0")
        rng = dmax - dmin if dmax != dmin else 1.0

        # Y-axis ticks and horizontal grid lines
        y_ticks = self._nice_ticks(dmin, dmax, max_ticks=5)
        painter.setFont(tick_font)
        grid_pen = QPen(grid_color)
        grid_pen.setWidthF(0.3)
        grid_pen.setStyle(Qt.PenStyle.DotLine)

        for tv in y_ticks:
            ratio = (tv - dmin) / rng if rng != 0 else 0.5
            y = plot_rect.y() + plot_rect.height() * (1.0 - ratio)
            if plot_rect.y() <= y <= plot_rect.y() + plot_rect.height():
                # Grid line
                painter.setPen(grid_pen)
                painter.drawLine(
                    QPointF(plot_rect.x(), y),
                    QPointF(plot_rect.x() + plot_rect.width(), y),
                )
                # Tick label
                painter.setPen(QPen(tick_color))
                label = self._format_tick(tv)
                label_rect = QRectF(
                    rect.x(), y - 5, plot_rect.x() - rect.x() - 2, 10,
                )
                painter.drawText(
                    label_rect,
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                    label,
                )

        # Zero line (solid, slightly brighter) if data spans zero
        if dmin < 0 < dmax:
            zero_ratio = (0 - dmin) / rng
            zero_y = plot_rect.y() + plot_rect.height() * (1.0 - zero_ratio)
            zero_pen = QPen(QColor("#444c66"))
            zero_pen.setWidthF(0.6)
            painter.setPen(zero_pen)
            painter.drawLine(
                QPointF(plot_rect.x(), zero_y),
                QPointF(plot_rect.x() + plot_rect.width(), zero_y),
            )

        # Plot area border (left + bottom only)
        border_pen = QPen(QColor("#26324d"))
        border_pen.setWidthF(0.5)
        painter.setPen(border_pen)
        painter.drawLine(
            QPointF(plot_rect.x(), plot_rect.y()),
            QPointF(plot_rect.x(), plot_rect.y() + plot_rect.height()),
        )
        painter.drawLine(
            QPointF(plot_rect.x(), plot_rect.y() + plot_rect.height()),
            QPointF(plot_rect.x() + plot_rect.width(), plot_rect.y() + plot_rect.height()),
        )

    def _paint_plot_current_value(
        self, painter: QPainter, plot_rect: QRectF,
        data: np.ndarray, dmin: float, dmax: float, color: QColor,
    ) -> None:
        """Draw the current (last) value annotation at top-right."""
        if len(data) == 0:
            return
        last_val = float(data[-1])
        label = self._format_tick(last_val)

        font = QFont(MONO_FAMILY, 6)
        font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(font)
        painter.setPen(QPen(color))

        label_rect = QRectF(
            plot_rect.x() + plot_rect.width() - 60,
            plot_rect.y() + 1,
            58, 10,
        )
        painter.drawText(
            label_rect,
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop,
            label,
        )

    def _paint_plot_bar(
        self, painter: QPainter, rect: QRectF, plot_rect: QRectF,
        data: np.ndarray, config: dict, spec: dict, tick_font: QFont,
        key: str = "",
    ) -> None:
        """Bar chart with grid and tick labels."""
        n = len(data)
        if n == 0:
            return

        view = self._plot_view_state.get(key)
        if view is not None:
            dmin = view["ymin"]
            dmax = view["ymax"]
        else:
            dmin = float(np.nanmin(data))
            dmax = float(np.nanmax(data))
        if dmin == dmax:
            dmax = dmin + 1.0

        # Grid and ticks
        self._paint_plot_grid_and_ticks(painter, rect, plot_rect, dmin, dmax, n, tick_font)

        rng = dmax - dmin
        positive_color = QColor(spec.get("color_positive", "#3dff6f"))
        negative_color = QColor(spec.get("color_negative", "#e94560"))

        # Downsample if needed
        if n > MAX_DISPLAY_DIM:
            step = max(1, n // MAX_DISPLAY_DIM)
            data = data[::step]
            n = len(data)

        bar_w = plot_rect.width() / n
        zero_ratio = max(0.0, min(1.0, (0.0 - dmin) / rng))
        zero_y = plot_rect.y() + plot_rect.height() * (1.0 - zero_ratio)

        painter.setPen(Qt.PenStyle.NoPen)
        for i in range(n):
            v = float(data[i])
            v_ratio = max(0.0, min(1.0, (v - dmin) / rng))
            v_y = plot_rect.y() + plot_rect.height() * (1.0 - v_ratio)

            if v >= 0:
                painter.setBrush(QBrush(positive_color))
                top = min(v_y, zero_y)
                h = abs(zero_y - v_y)
            else:
                painter.setBrush(QBrush(negative_color))
                top = zero_y
                h = abs(v_y - zero_y)

            if h > 0:
                painter.drawRect(QRectF(
                    plot_rect.x() + i * bar_w, top,
                    max(bar_w - 0.5, 0.5), h,
                ))

        # Current value label
        color = positive_color if float(data[-1]) >= 0 else negative_color
        self._paint_plot_current_value(painter, plot_rect, data, dmin, dmax, color)

    def _paint_plot_line(
        self, painter: QPainter, rect: QRectF, plot_rect: QRectF,
        data: np.ndarray, config: dict, spec: dict, tick_font: QFont,
        key: str = "",
    ) -> None:
        """Line chart with grid and tick labels."""
        n = len(data)
        if n < 2:
            return

        view = self._plot_view_state.get(key)
        if view is not None:
            dmin = view["ymin"]
            dmax = view["ymax"]
        else:
            dmin = float(np.nanmin(data))
            dmax = float(np.nanmax(data))
        if dmin == dmax:
            dmax = dmin + 1.0

        rng = dmax - dmin

        # Grid and ticks
        self._paint_plot_grid_and_ticks(painter, rect, plot_rect, dmin, dmax, n, tick_font)

        # Downsample
        if n > MAX_DISPLAY_DIM:
            step = max(1, n // MAX_DISPLAY_DIM)
            data = data[::step]
            n = len(data)

        line_color = QColor(spec.get("color_positive", "#00f5ff"))
        line_width = float(spec.get("line_width", 1.5))

        step_x = plot_rect.width() / max(n - 1, 1)

        # Build points
        points: list[QPointF] = []
        for i in range(n):
            x = plot_rect.x() + i * step_x
            ratio = (float(data[i]) - dmin) / rng if rng != 0 else 0.5
            ratio = max(0.0, min(1.0, ratio))
            y = plot_rect.y() + plot_rect.height() * (1.0 - ratio)
            points.append(QPointF(x, y))

        # Draw line segments
        pen = QPen(line_color)
        pen.setWidthF(line_width)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for i in range(len(points) - 1):
            painter.drawLine(points[i], points[i + 1])

        # Subtle fill under the line
        fill_color = QColor(line_color)
        fill_color.setAlpha(18)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(fill_color))
        fill_path = QPainterPath()
        fill_path.moveTo(QPointF(points[0].x(), plot_rect.y() + plot_rect.height()))
        for p in points:
            fill_path.lineTo(p)
        fill_path.lineTo(QPointF(points[-1].x(), plot_rect.y() + plot_rect.height()))
        fill_path.closeSubpath()
        painter.drawPath(fill_path)

        # Current value label
        self._paint_plot_current_value(painter, plot_rect, data, dmin, dmax, line_color)

    def _paint_plot_scatter(
        self, painter: QPainter, rect: QRectF, plot_rect: QRectF,
        data: np.ndarray, config: dict, spec: dict, tick_font: QFont,
        key: str = "",
    ) -> None:
        """Scatter plot with grid and tick labels on both axes. Data is Nx2."""
        n = data.shape[0]
        if n == 0:
            return

        xs = data[:, 0]
        ys = data[:, 1]

        view = self._plot_view_state.get(key)
        if view is not None and "xmin" in view:
            xmin = view["xmin"]
            xmax = view["xmax"]
            ymin = view["ymin"]
            ymax = view["ymax"]
        else:
            xmin = float(np.nanmin(xs))
            xmax = float(np.nanmax(xs))
            ymin = float(np.nanmin(ys))
            ymax = float(np.nanmax(ys))
            xpad = max((xmax - xmin) * 0.05, 1e-6)
            ypad = max((ymax - ymin) * 0.05, 1e-6)
            xmin -= xpad
            xmax += xpad
            ymin -= ypad
            ymax += ypad
        if xmin == xmax:
            xmax = xmin + 1.0
        if ymin == ymax:
            ymax = ymin + 1.0

        xrng = xmax - xmin
        yrng = ymax - ymin

        # Grid and ticks for Y-axis (reuse existing helper)
        self._paint_plot_grid_and_ticks(painter, rect, plot_rect, ymin, ymax, n, tick_font)

        # X-axis ticks and vertical grid lines
        grid_color = QColor("#26324d")
        tick_color = QColor("#6b7fa0")
        x_ticks = self._nice_ticks(xmin, xmax, max_ticks=5)
        painter.setFont(tick_font)
        grid_pen = QPen(grid_color)
        grid_pen.setWidthF(0.3)
        grid_pen.setStyle(Qt.PenStyle.DotLine)

        for tv in x_ticks:
            ratio = (tv - xmin) / xrng if xrng != 0 else 0.5
            x = plot_rect.x() + plot_rect.width() * ratio
            if plot_rect.x() <= x <= plot_rect.x() + plot_rect.width():
                # Vertical grid line
                painter.setPen(grid_pen)
                painter.drawLine(
                    QPointF(x, plot_rect.y()),
                    QPointF(x, plot_rect.y() + plot_rect.height()),
                )
                # X-axis tick label
                painter.setPen(QPen(tick_color))
                label = self._format_tick(tv)
                label_rect = QRectF(
                    x - 15, plot_rect.y() + plot_rect.height() + 1, 30, 10,
                )
                painter.drawText(
                    label_rect,
                    Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignTop,
                    label,
                )

        # Downsample if too many points
        if n > MAX_DISPLAY_DIM:
            step = max(1, n // MAX_DISPLAY_DIM)
            xs = xs[::step]
            ys = ys[::step]
            n = len(xs)

        dot_color = QColor(spec.get("color_positive", "#00f5ff"))
        point_radius = float(spec.get("line_width", 1.5))

        # Draw points with age-based fading (older = more transparent)
        painter.setPen(Qt.PenStyle.NoPen)
        for i in range(n):
            xv = float(xs[i])
            yv = float(ys[i])
            xr = (xv - xmin) / xrng if xrng != 0 else 0.5
            yr = (yv - ymin) / yrng if yrng != 0 else 0.5
            xr = max(0.0, min(1.0, xr))
            yr = max(0.0, min(1.0, yr))
            px = plot_rect.x() + plot_rect.width() * xr
            py = plot_rect.y() + plot_rect.height() * (1.0 - yr)

            # Fade: oldest point ~30 alpha, newest ~220
            alpha = int(30 + (220 - 30) * (i / max(n - 1, 1)))
            c = QColor(dot_color)
            c.setAlpha(alpha)
            painter.setBrush(QBrush(c))
            painter.drawEllipse(QPointF(px, py), point_radius, point_radius)

        # Draw last point brighter and larger
        if n > 0:
            xv = float(xs[-1])
            yv = float(ys[-1])
            xr = max(0.0, min(1.0, (xv - xmin) / xrng))
            yr = max(0.0, min(1.0, (yv - ymin) / yrng))
            px = plot_rect.x() + plot_rect.width() * xr
            py = plot_rect.y() + plot_rect.height() * (1.0 - yr)
            bright = QColor(dot_color)
            bright.setAlpha(255)
            painter.setBrush(QBrush(bright))
            painter.drawEllipse(QPointF(px, py), point_radius + 1.0, point_radius + 1.0)

            # Value annotation
            font = QFont(MONO_FAMILY, 6)
            font.setWeight(QFont.Weight.DemiBold)
            painter.setFont(font)
            painter.setPen(QPen(dot_color))
            label = f"{xv:.2f}, {yv:.2f}"
            label_rect = QRectF(
                plot_rect.x() + plot_rect.width() - 80,
                plot_rect.y() + 1,
                78, 10,
            )
            painter.drawText(
                label_rect,
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop,
                label,
            )

    def _paint_numeric(self, painter: QPainter, rect: QRectF, key: str, spec: dict) -> None:
        """Paint a numeric value."""
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(BG_SECONDARY))
        painter.drawRect(rect)

        value = self._display_cache.get(key)
        fmt = spec.get("format", ".4f")
        if isinstance(value, (int, float)):
            text = f"{value:{fmt}}"
        else:
            text = str(value) if value is not None else "—"

        painter.setPen(QPen(ACCENT))
        painter.setFont(self._value_font)
        painter.drawText(rect.adjusted(4, 0, -4, 0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, text)

    # ── Display updates ──────────────────────────────────────────────────

    def update_displays(self, outputs: Dict[str, Any]) -> None:
        """Update cached display data and repaint."""
        changed = False
        layout_needed = False
        for key, value in outputs.items():
            if key in self._display_rects:
                # Handle new format: {'value': ..., 'config': ...}
                if isinstance(value, dict) and 'value' in value:
                    display_value = value['value']
                    config = value.get('config', {})
                    # Update config cache if changed.
                    if key in self._display_config:
                        old_config = self._display_config[key]
                        if old_config != config:
                            self._display_config[key] = config
                            changed = True
                    else:
                        self._display_config[key] = config
                else:
                    display_value = value
                    config = {}
                    
                # Check if content changed for re-layout
                if key in self._display_cache:
                    old_value = self._display_cache[key]
                    if str(old_value) != str(display_value):
                        # Check if this output needs re-layout
                        for output in self.schema.get("outputs", []):
                            if output.get("key") == key:
                                otype = output.get("type")
                                if otype == "text":
                                    layout_needed = True
                                elif otype in ("heatmap", "image"):
                                    # Check if shape changed
                                    def get_shape(v):
                                        if hasattr(v, "shape"):
                                            return v.shape
                                        if isinstance(v, list):
                                            # Simple heuristic for 2D list shape
                                            rows = len(v)
                                            cols = len(v[0]) if rows > 0 and isinstance(v[0], list) else 0
                                            return (rows, cols)
                                        return None
                                    
                                    if get_shape(old_value) != get_shape(display_value):
                                        layout_needed = True
                                break
                else:
                    # First time receiving data for this display - always re-layout
                    layout_needed = True

                self._display_cache[key] = display_value
                text_widget = self._text_display_widgets.get(key)
                if text_widget is not None:
                    next_text = str(display_value) if display_value is not None else ""
                    if text_widget.toPlainText() != next_text:
                        text_widget.setPlainText(next_text)
                changed = True
        if changed:
            # Re-layout if content changed to adjust node height
            if layout_needed:
                self._layout()
            self.update()

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _format_range_value(prop: dict) -> str:
        import math as _m
        value = prop.get("value", prop.get("default", 0))
        try:
            value = float(value)
        except Exception:
            value = 0.0
        if not _m.isfinite(value):
            value = 0.0
        if prop.get("scale") == "log":
            return f"{value:.6f}"
        return f"{value:.3f}"

    def _calculate_text_height(self, text: str, width: float) -> float:
        """Calculate the height needed for multiline text."""
        fm = QFontMetrics(self._value_font)
        min_height = max(24.0, float(fm.lineSpacing()) + 12.0)
        if not text:
            return min_height
        # Get bounding rect for the text with word wrap
        rect = fm.boundingRect(
            0,
            0,
            int(max(1.0, width)),
            0,
            int(Qt.TextFlag.TextWordWrap),
            text,
        )
        return max(min_height, rect.height() + 12.0)

    def _calculate_text_widget_height(
        self,
        widget: QPlainTextEdit,
        text: str,
        width: int,
    ) -> float:
        """Estimate the QPlainTextEdit height for wrapped text content."""
        width = max(40, int(width))
        frame = widget.frameWidth() * 2
        margin = int(widget.document().documentMargin() * 2)
        scrollbar_width = widget.verticalScrollBar().sizeHint().width()
        viewport_width = max(20, width - frame - margin - scrollbar_width)
        base_height = self._calculate_text_height(text, viewport_width)
        return max(base_height, 24.0)

    def get_port(self, port_name: str, port_type: str) -> Optional[PortItem]:
        ports = self.input_ports if port_type == "input" else self.output_ports
        for p in ports:
            if p.port_name == port_name:
                return p
        return None

    # ── Events ───────────────────────────────────────────────────────────

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self.signals.position_changed.emit(self.node_id, value.x(), value.y())
            # Notify scene to update connections
            scene = self.scene()
            if scene and hasattr(scene, "update_connections_for_node"):
                scene.update_connections_for_node(self.node_id)
        return super().itemChange(change, value)

    def hoverEnterEvent(self, event) -> None:
        self._hovered = True
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        self._hovered = False
        self._tooltip_timer.stop()
        self._tooltip_port = None
        self.update()
        super().hoverLeaveEvent(event)

    def hoverMoveEvent(self, event) -> None:
        """Track hover over port labels for multi-type tooltip."""
        pos = event.pos()
        n_inputs = len(self.input_ports)
        n_outputs = len(self.output_ports)
        n_port_rows = max(n_inputs, n_outputs)
        io_top = HEADER_HEIGHT

        # Check input ports
        for i, port in enumerate(self.input_ports):
            if port.data_types:  # Only multi-type ports
                label_rect = QRectF(12, io_top + i * PORT_ROW_HEIGHT, self._width / 2 - 14, PORT_ROW_HEIGHT)
                if label_rect.contains(pos):
                    if self._tooltip_port != port:
                        self._tooltip_port = port
                        self._tooltip_timer.start(self._tooltip_delay)
                    return

        # Check output ports
        for i, port in enumerate(self.output_ports):
            if port.data_types:  # Only multi-type ports
                label_rect = QRectF(self._width / 2 + 2, io_top + i * PORT_ROW_HEIGHT, self._width / 2 - 14, PORT_ROW_HEIGHT)
                if label_rect.contains(pos):
                    if self._tooltip_port != port:
                        self._tooltip_port = port
                        self._tooltip_timer.start(self._tooltip_delay)
                    return

        # Not hovering over a multi-type port - don't hide tooltip immediately
        # Only reset if no tooltip is currently shown
        if not QToolTip.text():
            self._tooltip_timer.stop()
            self._tooltip_port = None
        super().hoverMoveEvent(event)

    def _show_port_tooltip(self) -> None:
        """Show tooltip with type list for multi-type ports."""
        if self._tooltip_port and self._tooltip_port.data_types:
            # Format types as bullet list
            types_text = "\n".join(f"• {t}" for t in self._tooltip_port.data_types)
            # Get global position for tooltip - use the label rect center, not port position
            io_top = HEADER_HEIGHT
            
            # Find the label rect for this port
            for i, port in enumerate(self.input_ports):
                if port == self._tooltip_port:
                    label_rect = QRectF(12, io_top + i * PORT_ROW_HEIGHT, self._width / 2 - 14, PORT_ROW_HEIGHT)
                    break
            else:
                for i, port in enumerate(self.output_ports):
                    if port == self._tooltip_port:
                        label_rect = QRectF(self._width / 2 + 2, io_top + i * PORT_ROW_HEIGHT, self._width / 2 - 14, PORT_ROW_HEIGHT)
                        break
                else:
                    label_rect = QRectF(0, 0, self._width, PORT_ROW_HEIGHT)
            
            # Use the center of the label rect for tooltip position
            label_center = label_rect.center()
            scene_pos = self.mapToScene(label_center)
            view = self.scene().views()[0] if self.scene() and self.scene().views() else None
            if view:
                # Convert scene coordinates to viewport coordinates, then to global
                viewport_pos = view.mapFromScene(scene_pos)
                global_pos = view.viewport().mapToGlobal(viewport_pos)
                # Convert label rect to viewport coordinates for the "hover area" that keeps tooltip visible
                scene_label_rect = self.mapToScene(label_rect).boundingRect()
                viewport_label_rect = view.mapFromScene(scene_label_rect).boundingRect()
                # Show tooltip with a rect area - tooltip stays visible while mouse is in this area
                QToolTip.showText(global_pos, f"Types:\n{types_text}", view.viewport(), viewport_label_rect)

    def mouseDoubleClickEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """Handle double-click on reload button, title area, or plot view reset."""
        if self._dynamic:
            if event.pos().x() > self._width - 24 and event.pos().y() < HEADER_HEIGHT:
                self.signals.reload_requested.emit(self.node_id)
                return
        if self._title_text_rect().contains(event.pos()):
            self._begin_title_rename()
            event.accept()
            return
        # Double-click on plot: reset view to auto-fit
        plot_key = self._find_plot_key_at(event.pos())
        if plot_key is not None and plot_key in self._plot_view_state:
            del self._plot_view_state[plot_key]
            self.update()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """Handle click on output toggle checkboxes and plot drag."""
        scene = self.scene()
        if scene and hasattr(scene, "bring_node_to_front"):
            scene.bring_node_to_front(self)

        pos = event.pos()

        # Right-click on plot: start pan drag
        if event.button() == Qt.MouseButton.RightButton:
            plot_key = self._find_plot_key_at(pos)
            if plot_key is not None:
                if plot_key not in self._plot_view_state:
                    auto = self._auto_range_for_plot(plot_key)
                    if auto is not None:
                        self._plot_view_state[plot_key] = auto
                if plot_key in self._plot_view_state:
                    self._plot_drag_key = plot_key
                    self._plot_drag_last_pos = pos
                    event.accept()
                    return

        for output in self.schema.get("outputs", []):
            key = output.get("key", "")
            rect_d = self._display_rects.get(key)
            if not rect_d:
                continue
            cb_rect = QRectF(PADDING, rect_d.y() - 12, 10, 10)
            if cb_rect.contains(pos):
                current = output.get("enabled", True)
                output["enabled"] = not current
                self._layout()
                self.update()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """Handle plot pan drag."""
        if self._plot_drag_key is not None:
            key = self._plot_drag_key
            view = self._plot_view_state.get(key)
            plot_rect = self._plot_rect_for_display(key)

            if view is not None and plot_rect is not None and self._plot_drag_last_pos is not None:
                dx_px = event.pos().x() - self._plot_drag_last_pos.x()
                dy_px = event.pos().y() - self._plot_drag_last_pos.y()
                style = self._get_plot_style(key)

                if style == "scatter":
                    xrng = view["xmax"] - view["xmin"]
                    yrng = view["ymax"] - view["ymin"]
                    dx_data = -dx_px / plot_rect.width() * xrng
                    dy_data = dy_px / plot_rect.height() * yrng
                    view["xmin"] += dx_data
                    view["xmax"] += dx_data
                    view["ymin"] += dy_data
                    view["ymax"] += dy_data
                else:
                    yrng = view["ymax"] - view["ymin"]
                    dy_data = dy_px / plot_rect.height() * yrng
                    view["ymin"] += dy_data
                    view["ymax"] += dy_data

                self._plot_drag_last_pos = event.pos()
                self.update()

            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """End plot pan drag."""
        if self._plot_drag_key is not None and event.button() == Qt.MouseButton.RightButton:
            self._plot_drag_key = None
            self._plot_drag_last_pos = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QGraphicsSceneWheelEvent) -> None:
        """Zoom plot on scroll: 1D for line/bar, 2D for scatter."""
        plot_key = self._find_plot_key_at(event.pos())
        if plot_key is None:
            super().wheelEvent(event)
            return

        event.accept()

        plot_rect = self._plot_rect_for_display(plot_key)
        if plot_rect is None or plot_rect.width() <= 0 or plot_rect.height() <= 0:
            return

        # Initialize view state from auto range if not yet set
        if plot_key not in self._plot_view_state:
            auto = self._auto_range_for_plot(plot_key)
            if auto is None:
                return
            self._plot_view_state[plot_key] = auto

        view = self._plot_view_state[plot_key]
        style = self._get_plot_style(plot_key)

        delta = event.delta()
        factor = 0.85 if delta > 0 else 1.0 / 0.85

        if style == "scatter":
            # 2D zoom centered on mouse position
            mx = (event.pos().x() - plot_rect.x()) / plot_rect.width()
            my = 1.0 - (event.pos().y() - plot_rect.y()) / plot_rect.height()
            mx = max(0.0, min(1.0, mx))
            my = max(0.0, min(1.0, my))

            xmin, xmax = view["xmin"], view["xmax"]
            cx = xmin + mx * (xmax - xmin)
            new_xrng = (xmax - xmin) * factor
            view["xmin"] = cx - mx * new_xrng
            view["xmax"] = cx + (1.0 - mx) * new_xrng

            ymin, ymax = view["ymin"], view["ymax"]
            cy = ymin + my * (ymax - ymin)
            new_yrng = (ymax - ymin) * factor
            view["ymin"] = cy - my * new_yrng
            view["ymax"] = cy + (1.0 - my) * new_yrng
        else:
            # 1D zoom (Y-axis only) centered on mouse Y
            my = 1.0 - (event.pos().y() - plot_rect.y()) / plot_rect.height()
            my = max(0.0, min(1.0, my))

            ymin, ymax = view["ymin"], view["ymax"]
            cy = ymin + my * (ymax - ymin)
            new_yrng = (ymax - ymin) * factor
            view["ymin"] = cy - my * new_yrng
            view["ymax"] = cy + (1.0 - my) * new_yrng

        self.update()
