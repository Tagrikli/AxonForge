"""
NodeItem — QGraphicsItem representing a single node on the canvas.

Layout (top to bottom):
  Header  (title + reload btn)
  I/O ports row
  Properties (sliders, etc.)
  Actions (buttons)
  Displays (images, text, etc.)

Property widgets are embedded via QGraphicsProxyWidget.
Display outputs are painted directly for performance.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

from PySide6.QtCore import Qt, QRectF, QPointF, QSizeF, Signal, QObject, QTimer, QEvent
from PySide6.QtGui import (
    QPen, QBrush, QColor, QPainter, QFont, QFontMetrics, QImage, QPainterPath,
)
from PySide6.QtWidgets import (
    QGraphicsItem, QGraphicsProxyWidget,
    QWidget, QSlider, QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox, QPushButton,
    QHBoxLayout, QVBoxLayout, QLabel, QSizePolicy, QToolTip,
    QStyleOptionGraphicsItem, QGraphicsSceneMouseEvent, QProxyStyle, QStyle,
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
STORES_TEXT = QColor("#57d37a")
LOADING_OVERLAY_BG = QColor(12, 18, 32, 165)
LOADING_TEXT = QColor("#d6e6ff")

HEADER_HEIGHT = 22.0
PORT_ROW_HEIGHT = 18.0
NODE_MIN_WIDTH = 250.0
PADDING = 6.0
MAX_DISPLAY_DIM = 512
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
    property_changed = Signal(str, str, object)   # node_id, prop_key, value
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

        # Proxy widgets for properties
        self._proxy_widgets: List[QGraphicsProxyWidget] = []
        self._right_click_filters: List[QObject] = []

        # Display cache: {output_key: value}
        self._display_cache: Dict[str, Any] = {}
        # Display config cache: {output_key: config}
        self._display_config: Dict[str, Dict[str, Any]] = {}
        # Display images cache: {output_key: QImage}
        self._display_images: Dict[str, QImage] = {}
        # Display RGB cache for vectorized painting
        self._display_rgb_cache: Dict[str, np.ndarray] = {}

        # Computed layout metrics
        self._width = NODE_MIN_WIDTH
        self._height = 0.0
        self._body_y = 0.0
        self._display_rects: Dict[str, QRectF] = {}
        self._error_partition_rect: Optional[QRectF] = None
        self._stores_hint_rect: Optional[QRectF] = None
        self._stores_hint_text: str = ""

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
        self._build_ports()
        self._build_property_widgets()
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

    # ── Property widgets ─────────────────────────────────────────────────

    def _build_property_widgets(self) -> None:
        """Create embedded QWidgets for each property."""
        for prop in self.schema.get("properties", []):
            widget = self._create_property_widget(prop)
            if widget:
                proxy = QGraphicsProxyWidget(self)
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
            proxy = QGraphicsProxyWidget(self)
            proxy.setWidget(container)
            proxy.setZValue(Z_NODE_WIDGET)
            self._proxy_widgets.append(proxy)

    def _create_property_widget(self, prop: dict) -> Optional[QWidget]:
        ptype = prop.get("type", "")
        key = prop.get("key", "")

        if ptype == "range":
            return self._make_range_widget(prop, key)
        elif ptype == "float":
            return self._make_float_widget(prop, key)
        elif ptype == "integer":
            return self._make_integer_widget(prop, key)
        elif ptype == "bool":
            return self._make_bool_widget(prop, key)
        elif ptype == "enum":
            return self._make_enum_widget(prop, key)
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
            self.signals.property_changed.emit(self.node_id, key, real)

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
            self.signals.property_changed.emit(self.node_id, key, v)

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
            self.signals.property_changed.emit(self.node_id, key, v)

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
            self.signals.property_changed.emit(self.node_id, key, state == Qt.CheckState.Checked.value)

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

        combo = QComboBox()
        combo.setFixedHeight(20)
        combo.setStyleSheet(
            "QComboBox { background: #11172a; border: 1px solid #26324d; color: #e6f1ff; "
            "font-size: 8px; padding: 0 6px; }"
            "QComboBox:hover { border-color: #00f5ff; }"
            "QComboBox::drop-down { border: none; width: 16px; }"
            "QComboBox::down-arrow { border-left: 3px solid transparent; border-right: 3px solid transparent; "
            "border-top: 4px solid #8fa4ff; }"
            "QComboBox QAbstractItemView { background: #11172a; border: 1px solid #26324d; "
            "color: #e6f1ff; selection-background-color: #141c2f; selection-color: #00f5ff; }"
        )
        for opt in prop.get("options", []):
            combo.addItem(opt)
        current = prop.get("value", prop.get("default", ""))
        idx = combo.findText(current)
        if idx >= 0:
            combo.setCurrentIndex(idx)

        def on_combo(text: str) -> None:
            self.signals.property_changed.emit(self.node_id, key, text)

        combo.currentTextChanged.connect(on_combo)
        layout.addWidget(combo)

        container.setFixedHeight(34)
        return container

    # ── Layout ───────────────────────────────────────────────────────────

    def _layout(self) -> None:
        """Compute positions for all sub-elements."""
        y = 0.0

        # Header
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

        # Property widgets
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

            if otype == "vector2d":
                data = self._display_cache.get(key)
                if data is not None:
                    if isinstance(data, list):
                        data = np.array(data)
                    
                    if hasattr(data, "shape") and len(data.shape) >= 2:
                        rows, cols = data.shape[:2]
                        if cols > 0:
                            aspect = rows / cols
                            h = (self._width - 2 * PADDING) * aspect
                        else:
                            h = self._width - 2 * PADDING
                    else:
                        h = self._width - 2 * PADDING
                else:
                    h = self._width - 2 * PADDING  # default square
                self._display_rects[key] = QRectF(PADDING, y, self._width - 2 * PADDING, h)
                y += h + 2
            elif otype == "vector1d":
                h = 30
                self._display_rects[key] = QRectF(PADDING, y, self._width - 2 * PADDING, h)
                y += h + 2
            elif otype == "barchart":
                h = 30
                self._display_rects[key] = QRectF(PADDING, y, self._width - 2 * PADDING, h)
                y += h + 2
            elif otype == "linechart":
                h = 30
                self._display_rects[key] = QRectF(PADDING, y, self._width - 2 * PADDING, h)
                y += h + 2
            elif otype == "numeric":
                h = 16
                self._display_rects[key] = QRectF(PADDING, y, self._width - 2 * PADDING, h)
                y += h + 2
            elif otype == "text":
                # Calculate height based on text content
                text_value = self._display_cache.get(key, "")
                h = self._calculate_text_height(text_value, self._width - 2 * PADDING - 8)
                self._display_rects[key] = QRectF(PADDING, y, self._width - 2 * PADDING, h)
                y += h + 2

        # Stores information line at the bottom.
        self._stores_hint_rect = None
        self._stores_hint_text = ""
        stores = self.schema.get("stores", [])
        if stores:
            labels: List[str] = []
            for store in stores:
                label = str(store.get("label") or store.get("key") or "").strip()
                if label:
                    labels.append(label)
            if labels:
                self._stores_hint_text = "Stored: " + ", ".join(labels)
                stores_width = max(0.0, self._width - 2 * PADDING)
                fm = QFontMetrics(self._small_font)
                wrapped = fm.boundingRect(
                    0, 0, int(stores_width), 0,
                    int(Qt.TextFlag.TextWordWrap),
                    self._stores_hint_text,
                )
                stores_h = max(12.0, float(wrapped.height()))
                self._stores_hint_rect = QRectF(PADDING, y, stores_width, stores_h)
                y += stores_h + 2

        y += PADDING
        self._height = y
        self.prepareGeometryChange()

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

    def _advance_loading_spinner(self) -> None:
        self._loading_spinner_angle = (self._loading_spinner_angle + 14.0) % 360.0
        self.update()

    def set_loading_state(self, is_loading: bool) -> None:
        """Set whether this node is currently background-initializing."""
        if self._loading == is_loading:
            return
        self._loading = is_loading
        self._apply_loading_ui_state()
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

        # Title text
        painter.setPen(QPen(TEXT_PRIMARY))
        painter.setFont(self._title_font)
        title_rect = QRectF(PADDING + 4, 0, self._width - 2 * PADDING - 20, HEADER_HEIGHT)
        painter.drawText(title_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self._title)

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
            if otype == "vector2d":
                self._paint_vector2d(painter, rect_d, key, output)
            elif otype == "vector1d":
                self._paint_vector1d(painter, rect_d, key)
            elif otype == "barchart":
                self._paint_barchart(painter, rect_d, key, output)
            elif otype == "linechart":
                self._paint_linechart(painter, rect_d, key, output)
            elif otype == "numeric":
                self._paint_numeric(painter, rect_d, key, output)
            elif otype == "text":
                self._paint_text(painter, rect_d, key)

        # Stores hint at node bottom.
        if self._stores_hint_rect and self._stores_hint_text:
            painter.setPen(QPen(STORES_TEXT))
            painter.setFont(self._small_font)
            painter.drawText(
                self._stores_hint_rect,
                int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap),
                self._stores_hint_text,
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

    def _paint_vector2d(self, painter: QPainter, rect: QRectF, key: str, spec: dict) -> None:
        """Paint a 2D numpy array as a pixel image."""
        data = self._display_cache.get(key)
        if data is None or not isinstance(data, (np.ndarray, list)):
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(BG_SECONDARY))
            painter.drawRect(rect)
            return

        if isinstance(data, list):
            data = np.array(data, dtype=np.float32)

        if data.ndim != 2:
            return

        rows, cols = data.shape
        # Use config from snapshot first, fall back to schema spec
        config = self._display_config.get(key, {})
        color_mode = config.get("color_mode", spec.get("color_mode", "grayscale"))

        # Limit maximum display size for performance
        MAX_DIM = MAX_DISPLAY_DIM
        if rows > MAX_DIM or cols > MAX_DIM:
            step_rows = (rows + MAX_DIM - 1) // MAX_DIM
            step_cols = (cols + MAX_DIM - 1) // MAX_DIM
            step = max(step_rows, step_cols)
            data = data[::step, ::step]
            rows, cols = data.shape

        # Normalize data based on color mode
        if color_mode == "viridis":
            # viridis colormap - expects [-1, 1] range
            t = np.clip((data + 1.0) / 2.0, 0.0, 1.0)
            # Viridis approximation (perceptually uniform, purple to yellow)
            rc = np.clip(255 * (0.267 + 0.329 * t - 0.868 * t**2 + 1.65 * t**3), 0, 255).astype(np.uint8)
            gc = np.clip(255 * (0.012 + 0.696 * t - 1.16 * t**2 + 0.98 * t**3), 0, 255).astype(np.uint8)
            bc = np.clip(255 * (0.909 - 0.298 * t - 1.45 * t**2 + 2.22 * t**3), 0, 255).astype(np.uint8)
        elif color_mode in ("bwr", "diverging"):
            # map from [-1,1] to [0,1]
            t = np.clip((data + 1.0) / 2.0, 0.0, 1.0)
            # compute RGB using vectorized operations
            mask = t < 0.5
            k = np.where(mask, t / 0.5, (t - 0.5) / 0.5)
            rc = np.where(mask, 255 * k, 255).astype(np.uint8)
            gc = np.where(mask, 255 * k, 255 * (1 - k)).astype(np.uint8)
            bc = np.where(mask, 255, 255 * (1 - k)).astype(np.uint8)
        else:
            # grayscale
            v = np.clip(data, 0.0, 1.0)
            gray = (v * 255).astype(np.uint8)
            rc = gray
            gc = gray
            bc = gray

        # Combine channels into a (rows, cols, 3) uint8 array
        rgb = np.stack([rc, gc, bc], axis=2)
        rgb = np.ascontiguousarray(rgb)
        # Keep reference to prevent garbage collection
        self._display_rgb_cache[key] = rgb

        # Create QImage from memory buffer (RGB888, 3 bytes per pixel)
        img = QImage(rgb.data, cols, rows, cols * 3, QImage.Format.Format_RGB888)
        self._display_images[key] = img
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        painter.drawImage(rect, img)
        painter.restore()

    def _paint_vector1d(self, painter: QPainter, rect: QRectF, key: str) -> None:
        """Paint a 1D array as a bar chart."""
        data = self._display_cache.get(key)
        if data is None:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(BG_SECONDARY))
            painter.drawRect(rect)
            return

        if isinstance(data, list):
            data = np.array(data, dtype=np.float32)
        if isinstance(data, np.ndarray):
            data = data.flatten()

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(BG_SECONDARY))
        painter.drawRect(rect)

        n = len(data)
        if n == 0:
            return

        dmin = float(np.nanmin(data))
        dmax = float(np.nanmax(data))
        rng = dmax - dmin if dmax != dmin else 1.0

        # Limit number of bars for performance
        max_bars = MAX_DISPLAY_DIM
        if n > max_bars:
            step = n // max_bars
            data = data[::step]
            n = len(data)

        bar_w = rect.width() / n
        painter.setBrush(QBrush(QColor("#e94560")))
        for i in range(n):
            v = (float(data[i]) - dmin) / rng
            h = v * (rect.height() - 2)
            painter.drawRect(QRectF(
                rect.x() + i * bar_w,
                rect.y() + rect.height() - h,
                max(bar_w - 0.5, 0.5),
                h,
            ))

    def _paint_barchart(self, painter: QPainter, rect: QRectF, key: str, spec: dict) -> None:
        """Paint a 1D array as a bar chart with negative value support."""
        data = self._display_cache.get(key)
        if data is None:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(BG_SECONDARY))
            painter.drawRect(rect)
            return

        if isinstance(data, list):
            data = np.array(data, dtype=np.float32)
        if isinstance(data, np.ndarray):
            data = data.flatten()

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(BG_SECONDARY))
        painter.drawRect(rect)

        n = len(data)
        if n == 0:
            return

        # Get configuration from display cache (set from bridge snapshot)
        config = self._display_config.get(key, {})
        scale_mode = config.get("scale_mode", "auto")
        
        # Determine dmin/dmax based on scale mode
        if scale_mode == "manual":
            dmin = config.get("scale_min", 0.0)
            dmax = config.get("scale_max", 1.0)
        else:
            # Auto mode: calculate from data
            dmin = float(np.nanmin(data))
            dmax = float(np.nanmax(data))
        
        has_negative = dmin < 0
        has_positive = dmax > 0
        
        # Get configuration from spec
        color = spec.get("color", "#e94560")
        show_negative = spec.get("show_negative", True)
        
        # Calculate zero line position
        if has_negative and has_positive:
            # Both positive and negative values - zero line based on relative ranges
            # The total range is dmax - dmin, zero is positioned proportionally
            zero_ratio = -dmin / (dmax - dmin) if dmax != dmin else 0.5
            zero_y = rect.y() + rect.height() * (1 - zero_ratio)
            # Use the full range for scaling
            rng = dmax - dmin
        elif has_negative and not has_positive:
            # All negative - zero at top
            zero_y = rect.y()
            rng = abs(dmin)
        else:
            # All positive - zero at bottom
            zero_y = rect.y() + rect.height()
            rng = dmax - dmin

        # Limit number of bars for performance
        max_bars = MAX_DISPLAY_DIM
        if n > max_bars:
            step = n // max_bars
            data = data[::step]
            n = len(data)

        bar_w = rect.width() / n
        
        # Determine colors for positive and negative values
        positive_color = QColor("#3dff6f")  # Green for positive
        negative_color = QColor("#e94560")  # Red for negative
        
        for i in range(n):
            v = float(data[i])
            
            # Determine bar height based on value relative to zero
            if has_negative and has_positive:
                # Mixed data: normalize from dmin to dmax
                if rng != 0:
                    normalized = (v - dmin) / rng
                else:
                    normalized = 0.0
                normalized = max(0.0, min(1.0, normalized))
                
                if v >= 0:
                    # Positive value - scale from zero upward
                    h = max(0.0, (normalized - zero_ratio) * rect.height())
                    painter.setBrush(QBrush(positive_color))
                    top = zero_y - h
                    clamped_top = max(rect.y(), top)
                    clamped_height = zero_y - clamped_top
                    if clamped_height > 0:
                        painter.drawRect(QRectF(
                            rect.x() + i * bar_w,
                            clamped_top,
                            max(bar_w - 0.5, 0.5),
                            clamped_height,
                        ))
                else:
                    # Negative value - scale from zero downward
                    h = max(0.0, (zero_ratio - normalized) * rect.height())
                    painter.setBrush(QBrush(negative_color))
                    bottom = zero_y + h
                    clamped_bottom = min(rect.y() + rect.height(), bottom)
                    clamped_height = clamped_bottom - zero_y
                    if clamped_height > 0:
                        painter.drawRect(QRectF(
                            rect.x() + i * bar_w,
                            zero_y,
                            max(bar_w - 0.5, 0.5),
                            clamped_height,
                        ))
            elif has_positive and not has_negative:
                # All positive: scale from zero (bottom) using dmax as reference
                if dmax > 0:
                    normalized = v / dmax
                else:
                    normalized = 0.0
                normalized = max(0.0, min(1.0, normalized))
                h = normalized * rect.height()
                painter.setBrush(QBrush(positive_color))
                top = zero_y - h
                clamped_top = max(rect.y(), top)
                clamped_height = zero_y - clamped_top
                if clamped_height > 0:
                    painter.drawRect(QRectF(
                        rect.x() + i * bar_w,
                        clamped_top,
                        max(bar_w - 0.5, 0.5),
                        clamped_height,
                    ))
            else:
                # All negative: scale from zero (top) using abs(dmin) as reference
                if dmin < 0:
                    normalized = abs(v) / abs(dmin)
                else:
                    normalized = 0.0
                normalized = max(0.0, min(1.0, normalized))
                h = normalized * rect.height()
                painter.setBrush(QBrush(negative_color))
                bottom = zero_y + h
                clamped_bottom = min(rect.y() + rect.height(), bottom)
                clamped_height = clamped_bottom - zero_y
                if clamped_height > 0:
                    painter.drawRect(QRectF(
                        rect.x() + i * bar_w,
                        zero_y,
                        max(bar_w - 0.5, 0.5),
                        clamped_height,
                    ))

    def _paint_linechart(self, painter: QPainter, rect: QRectF, key: str, spec: dict) -> None:
        """Paint a 1D array as a line chart."""
        data = self._display_cache.get(key)
        if data is None:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(BG_SECONDARY))
            painter.drawRect(rect)
            return

        if isinstance(data, list):
            data = np.array(data, dtype=np.float32)
        if isinstance(data, np.ndarray):
            data = data.flatten()

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(BG_SECONDARY))
        painter.drawRect(rect)

        n = len(data)
        if n < 2:
            return

        # Get configuration from display cache (set from bridge snapshot)
        config = self._display_config.get(key, {})
        scale_mode = config.get("scale_mode", "auto")
        
        # Determine dmin/dmax based on scale mode
        if scale_mode == "manual":
            dmin = config.get("scale_min", 0.0)
            dmax = config.get("scale_max", 1.0)
        else:
            # Auto mode: calculate from data
            dmin = float(np.nanmin(data))
            dmax = float(np.nanmax(data))
        
        rng = dmax - dmin if dmax != dmin else 1.0

        # Limit number of points for performance
        max_points = MAX_DISPLAY_DIM
        if n > max_points:
            step = n // max_points
            data = data[::step]
            n = len(data)

        # Get configuration from spec
        line_color = spec.get("color", "#00f5ff")
        line_width = spec.get("line_width", 1.5)

        # Draw lines with clipping and intersection calculation
        pen = QPen(QColor(line_color))
        pen.setWidthF(line_width)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        # Calculate points
        step_x = rect.width() / (n - 1)
        
        # Helper function to clip point to chart bounds
        def clip_to_bounds(x: float, y: float) -> tuple:
            """Clip point to chart rectangle bounds."""
            clipped_x = max(rect.x(), min(rect.x() + rect.width(), x))
            clipped_y = max(rect.y(), min(rect.y() + rect.height(), y))
            return (clipped_x, clipped_y)
        
        # Helper function to calculate intersection with chart boundaries
        def get_intersection(x1: float, y1: float, x2: float, y2: float) -> list:
            """Calculate intersection points of line segment with chart boundaries.
            Returns list of intersection points (can be 0, 1, or 2 points).
            """
            intersections = []
            
            # Chart boundaries
            left = rect.x()
            right = rect.x() + rect.width()
            top = rect.y()
            bottom = rect.y() + rect.height()
            
            # Check if segment is completely outside
            if (max(y1, y2) < top or min(y1, y2) > bottom):
                return intersections  # No intersection
            
            # Check if segment is completely horizontal at boundary
            if abs(y2 - y1) < 0.001:
                if y1 >= top and y1 <= bottom:
                    # Horizontal line within bounds - clip both points
                    return [clip_to_bounds(x1, y1), clip_to_bounds(x2, y2)]
                return intersections
            
            # Calculate slope
            slope = (x2 - x1) / (y2 - y1)
            
            # Check intersection with top boundary (y = top)
            if (y1 <= top and y2 >= top) or (y1 >= top and y2 <= top):
                ix = x1 + slope * (top - y1)
                if left <= ix <= right:
                    intersections.append((ix, top))
            
            # Check intersection with bottom boundary (y = bottom)
            if (y1 <= bottom and y2 >= bottom) or (y1 >= bottom and y2 <= bottom):
                ix = x1 + slope * (bottom - y1)
                if left <= ix <= right:
                    intersections.append((ix, bottom))
            
            # Check intersection with left boundary (x = left)
            if (x1 <= left and x2 >= left) or (x1 >= left and x2 <= left):
                iy = y1 + (y2 - y1) * (left - x1) / (x2 - x1) if x2 != x1 else y1
                if top <= iy <= bottom:
                    intersections.append((left, iy))
            
            # Check intersection with right boundary (x = right)
            if (x1 <= right and x2 >= right) or (x1 >= right and x2 <= right):
                iy = y1 + (y2 - y1) * (right - x1) / (x2 - x1) if x2 != x1 else y1
                if top <= iy <= bottom:
                    intersections.append((right, iy))
            
            return intersections
        
        # Build list of valid points and draw segments
        valid_points = []
        
        for i in range(n):
            x = rect.x() + i * step_x
            normalized_y = (float(data[i]) - dmin) / rng if rng != 0 else 0.5
            normalized_y = max(0.0, min(1.0, normalized_y))  # Clamp
            y = rect.y() + rect.height() * (1 - normalized_y)
            
            # Clip to bounds
            x, y = clip_to_bounds(x, y)
            valid_points.append(QPointF(x, y))
        
        # Draw line segments, calculating intersections for out-of-bounds segments
        for i in range(n - 1):
            x1_raw = rect.x() + i * step_x
            normalized_y1_raw = (float(data[i]) - dmin) / rng if rng != 0 else 0.5
            y1_raw = rect.y() + rect.height() * (1 - normalized_y1_raw)
            
            x2_raw = rect.x() + (i + 1) * step_x
            normalized_y2_raw = (float(data[i + 1]) - dmin) / rng if rng != 0 else 0.5
            y2_raw = rect.y() + rect.height() * (1 - normalized_y2_raw)
            
            # Check if segment is completely outside bounds
            if (max(y1_raw, y2_raw) < rect.y() or min(y1_raw, y2_raw) > rect.y() + rect.height()):
                continue  # Skip segments completely outside
            
            # Check if both points are within bounds - draw direct line
            if (rect.y() <= y1_raw <= rect.y() + rect.height() and
                rect.y() <= y2_raw <= rect.y() + rect.height() and
                rect.x() <= x1_raw <= rect.x() + rect.width() and
                rect.x() <= x2_raw <= rect.x() + rect.width()):
                painter.drawLine(valid_points[i], valid_points[i + 1])
            else:
                # Segment crosses boundary - calculate intersections
                intersections = get_intersection(x1_raw, y1_raw, x2_raw, y2_raw)
                if len(intersections) >= 2:
                    # Draw from first to last intersection
                    painter.drawLine(QPointF(intersections[0][0], intersections[0][1]),
                                    QPointF(intersections[-1][0], intersections[-1][1]))
                elif len(intersections) == 1:
                    # One intersection - clip and draw to intersection point
                    painter.drawLine(valid_points[i], QPointF(intersections[0][0], intersections[0][1]))

        # Draw zero line if there are negative values
        if dmin < 0:
            zero_ratio = -dmin / rng
            zero_ratio = max(0.0, min(1.0, zero_ratio))
            zero_y = rect.y() + rect.height() * (1 - zero_ratio)
            pen_zero = QPen(QColor("#555555"))
            pen_zero.setWidthF(0.5)
            pen_zero.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen_zero)
            painter.drawLine(QPointF(rect.x(), zero_y), QPointF(rect.x() + rect.width(), zero_y))

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

    def _paint_text(self, painter: QPainter, rect: QRectF, key: str) -> None:
        """Paint a text value."""
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(BG_SECONDARY))
        painter.drawRect(rect)

        value = self._display_cache.get(key)
        text = str(value) if value is not None else ""

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
                    # Update config cache if changed (color_mode or scale settings)
                    if key in self._display_config:
                        old_config = self._display_config[key]
                        if (old_config.get('color_mode') != config.get('color_mode') or
                            old_config.get('scale_mode') != config.get('scale_mode') or
                            old_config.get('scale_min') != config.get('scale_min') or
                            old_config.get('scale_max') != config.get('scale_max')):
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
                                elif otype == "vector2d":
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
        if not text:
            return 16.0
        fm = QFontMetrics(self._value_font)
        # Get bounding rect for the text with word wrap
        rect = fm.boundingRect(0, 0, int(width), 0, 0, text)
        # Return height with some padding, minimum 16
        return max(16.0, rect.height() + 8)

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
        """Handle double-click on reload button area for dynamic nodes."""
        if self._dynamic:
            if event.pos().x() > self._width - 24 and event.pos().y() < HEADER_HEIGHT:
                self.signals.reload_requested.emit(self.node_id)
                return
        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """Handle click on output toggle checkboxes."""
        scene = self.scene()
        if scene and hasattr(scene, "bring_node_to_front"):
            scene.bring_node_to_front(self)

        pos = event.pos()
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
