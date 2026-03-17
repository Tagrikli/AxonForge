"""Interactive attention mask with gaussian fall-off circle."""

import numpy as np

from ...core.node import Node
from ...core.descriptors.ports import OutputPort
from ...core.descriptors.fields import Integer, Float
from ...core.descriptors.displays import Heatmap, Text
from ...core.descriptors.actions import Action
from ...core.descriptors.state import State


class AttentionMask(Node):
    """Gaussian fall-off circle that follows the mouse.

    Left-click or drag to position the circle.
    Scroll to resize the radius.
    """

    width = Integer("Width", default=28)
    height = Integer("Height", default=28)
    radius = Float("Radius", default=4.0)

    center_row = State(default=None)
    center_col = State(default=None)
    cached_mask = State(default=None)

    output = OutputPort("Mask", np.ndarray)

    canvas = Heatmap(
        "Canvas", colormap="grayscale",
        on_press="_on_place", on_move="_on_place",
        on_scroll="_on_scroll",
    )
    info = Text("Info", default="Click to place | Scroll to resize")

    reset = Action("Reset", lambda self, p=None: self._on_reset(p))

    def _build_mask(self):
        w, h = int(self.width), int(self.height)
        r = float(self.radius)
        if self.center_row is None or self.center_col is None:
            return np.zeros((h, w), dtype=np.float32)
        cr, cc = float(self.center_row), float(self.center_col)
        rows = np.arange(h, dtype=np.float64)
        cols = np.arange(w, dtype=np.float64)
        rr, cc_grid = np.meshgrid(rows, cols, indexing="ij")
        dist_sq = (rr - cr) ** 2 + (cc_grid - cc) ** 2
        sigma = max(r / 2.0, 0.5)
        mask = np.exp(-dist_sq / (2.0 * sigma ** 2))
        self.cached_mask = mask.astype(np.float32)
        return self.cached_mask

    def _on_place(self, row, col, button):
        if button == "left":
            self.center_row = row
            self.center_col = col
            mask = self._build_mask()
            self.canvas = mask
            self.output = mask
            self.info = f"pos: ({row}, {col}) | radius: {float(self.radius):.1f}"

    def _on_scroll(self, row, col, delta):
        r = max(1.0, float(self.radius) + delta)
        self.radius = r
        mask = self._build_mask()
        self.canvas = mask
        self.output = mask
        self.info = f"pos: ({self.center_row}, {self.center_col}) | radius: {r:.1f}"

    def process(self):
        if self.cached_mask is None:
            self.cached_mask = self._build_mask()
        self.canvas = self.cached_mask
        self.output = self.cached_mask
        if self.center_row is not None:
            self.info = f"pos: ({self.center_row}, {self.center_col}) | radius: {float(self.radius):.1f}"

    def _on_reset(self, params=None):
        self.center_row = None
        self.center_col = None
        w, h = int(self.width), int(self.height)
        self.cached_mask = np.zeros((h, w), dtype=np.float32)
        self.canvas = self.cached_mask
        self.output = self.cached_mask
        self.info = "Click to place | Scroll to resize"
        return {"status": "ok"}
