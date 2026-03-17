"""Interactive drawable mask node."""

import numpy as np

from ...core.node import Node
from ...core.descriptors.ports import OutputPort
from ...core.descriptors.fields import Integer
from ...core.descriptors.displays import Heatmap, Text
from ...core.descriptors.actions import Action
from ...core.descriptors.state import State


class DrawableMask(Node):
    """Draw on a grid with mouse. Left-click paints, right-click erases."""

    width = Integer("Width", default=28)
    height = Integer("Height", default=28)

    canvas_state = State(default=None)

    output = OutputPort("Mask", np.ndarray)

    canvas = Heatmap(
        "Canvas", colormap="grayscale",
        on_press="_on_draw", on_move="_on_draw",
    )
    info = Text("Info", default="Left: paint | Right: erase")

    reset = Action("Reset", lambda self, p=None: self._on_reset(p))

    def init(self):
        w, h = int(self.width), int(self.height)
        if self.canvas_state is None or self.canvas_state.shape != (h, w):
            self.canvas_state = np.zeros((h, w), dtype=np.float32)

    def _on_draw(self, row, col, button):
        w, h = int(self.width), int(self.height)
        if self.canvas_state is None or self.canvas_state.shape != (h, w):
            self.canvas_state = np.zeros((h, w), dtype=np.float32)
        if 0 <= row < h and 0 <= col < w:
            self.canvas_state[row, col] = 0.0 if button == "right" else 1.0
            self.canvas = self.canvas_state
            self.output = self.canvas_state

    def process(self):
        w, h = int(self.width), int(self.height)
        if self.canvas_state is None or self.canvas_state.shape != (h, w):
            self.canvas_state = np.zeros((h, w), dtype=np.float32)
        self.canvas = self.canvas_state
        self.output = self.canvas_state
        n_painted = int(np.sum(self.canvas_state > 0))
        self.info = f"{h}x{w} | painted: {n_painted}"

    def _on_reset(self, params=None):
        w, h = int(self.width), int(self.height)
        self.canvas_state = np.zeros((h, w), dtype=np.float32)
        self.canvas = self.canvas_state
        self.output = self.canvas_state
        self.info = f"{h}x{w} | painted: 0"
        return {"status": "ok"}
