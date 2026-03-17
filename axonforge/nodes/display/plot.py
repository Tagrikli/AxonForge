"""Plot nodes with history for MiniCortex."""

import numpy as np

from ...core.node import Node
from ...core.descriptors.ports import InputPort
from ...core.descriptors.state import State
from ...core.descriptors.fields import Integer
from ...core.descriptors.displays import Plot, Text
from ...core.descriptors.actions import Action


class PlotBar(Node):
    """Plot numeric values in a history array using bar style."""

    input_value = InputPort("Input", (int, float))
    max_history = Integer("Max History", default=100)

    history = State("History", default=None)
    history_count = State("History Count", default=0)

    plot = Plot("Plot", style="bar")
    info = Text("Info", default="min: —  max: —")

    reset = Action("Reset", lambda self, params=None: self._on_reset(params))

    def init(self):
        max_len = int(self.max_history)
        if self.history is None:
            self.history = np.full(max_len, np.nan, dtype=np.float32)
        else:
            if len(self.history) != max_len:
                self._resize_history(max_len)
        if self.history_count is None:
            self.history_count = 0

    def _resize_history(self, new_size):
        """Resize history array while preserving the most recent valid samples."""
        old_size = len(self.history)
        new_history = np.full(new_size, np.nan, dtype=np.float32)
        valid_count = int(max(0, min(int(self.history_count), old_size, new_size)))
        if valid_count > 0:
            new_history[-valid_count:] = self.history[-valid_count:]
        self.history = new_history
        self.history_count = valid_count

    def process(self):
        max_len = int(self.max_history)
        if len(self.history) != max_len:
            self._resize_history(max_len)

        if self.input_value is not None:
            float_value = float(self.input_value)
            if self.history_count > 0:
                self.history[:-1] = self.history[1:]
            self.history[-1] = float_value
            if self.history_count < max_len:
                self.history_count += 1

        if self.history_count > 0:
            if self.history_count < max_len:
                valid_values = self.history[-self.history_count:]
            else:
                valid_values = self.history.copy()
            self.plot = valid_values
            dmin = float(np.nanmin(valid_values))
            dmax = float(np.nanmax(valid_values))
            self.info = f"min: {dmin:.2f}  max: {dmax:.2f}"
        else:
            self.plot = np.array([], dtype=np.float32)
            self.info = "min: —  max: —"

    def _on_reset(self, params: dict):
        """Clear the history and reset the display."""
        max_len = int(self.max_history)
        self.history_count = 0
        self.history = np.full(max_len, np.nan, dtype=np.float32)
        self.plot = np.array([], dtype=np.float32)
        self.info = "min: —  max: —"
        return {"status": "ok"}


class PlotLine(Node):
    """Plot numeric values in a history array using line style."""

    input_value = InputPort("Input", (int, float))
    max_history = Integer("Max History", default=100)

    history = State("History", default=None)
    history_count = State("History Count", default=0)

    plot = Plot(
        "Plot",
        style="line",
        color_positive="#00f5ff",
        color_negative="#00f5ff",
        line_width=0.7,
    )
    info = Text("Info", default="min: —  max: —")

    reset = Action("Reset", lambda self, params=None: self._on_reset(params))

    def init(self):
        max_len = int(self.max_history)
        if self.history is None:
            self.history = np.full(max_len, np.nan, dtype=np.float32)
        else:
            if len(self.history) != max_len:
                self._resize_history(max_len)
        if self.history_count is None:
            self.history_count = 0

    def _resize_history(self, new_size):
        """Resize history array while preserving the most recent valid samples."""
        old_size = len(self.history)
        new_history = np.full(new_size, np.nan, dtype=np.float32)
        valid_count = int(max(0, min(int(self.history_count), old_size, new_size)))
        if valid_count > 0:
            new_history[-valid_count:] = self.history[-valid_count:]
        self.history = new_history
        self.history_count = valid_count

    def process(self):
        max_len = int(self.max_history)
        if len(self.history) != max_len:
            self._resize_history(max_len)

        if self.input_value is not None:
            float_value = float(self.input_value)
            if self.history_count > 0:
                self.history[:-1] = self.history[1:]
            self.history[-1] = float_value
            if self.history_count < max_len:
                self.history_count += 1

        if self.history_count > 0:
            if self.history_count < max_len:
                valid_values = self.history[-self.history_count:]
            else:
                valid_values = self.history.copy()
            self.plot = valid_values
            dmin = float(np.nanmin(valid_values))
            dmax = float(np.nanmax(valid_values))
            self.info = f"min: {dmin:.2f}  max: {dmax:.2f}"
        else:
            self.plot = np.array([], dtype=np.float32)
            self.info = "min: —  max: —"

    def _on_reset(self, params: dict):
        """Clear the history and reset the display."""
        max_len = int(self.max_history)
        self.history_count = 0
        self.history = np.full(max_len, np.nan, dtype=np.float32)
        self.plot = np.array([], dtype=np.float32)
        self.info = "min: —  max: —"
        return {"status": "ok"}


class PlotScatter(Node):
    """Aggregate 2D points and display them as a scatter plot with history."""

    input_value = InputPort("Input", np.ndarray)
    max_history = Integer("Max History", default=200)

    history = State("History", default=None)
    history_count = State("History Count", default=0)

    plot = Plot(
        "Plot",
        style="scatter",
        color_positive="#00f5ff",
        color_negative="#00f5ff",
        line_width=1.5,
    )
    info = Text("Info", default="pts: 0")

    reset = Action("Reset", lambda self, params=None: self._on_reset_scatter(params))

    def init(self):
        max_len = int(self.max_history)
        if self.history is None:
            self.history = np.full((max_len, 2), np.nan, dtype=np.float32)
        else:
            if self.history.shape != (max_len, 2):
                self._resize_history(max_len)
        if self.history_count is None:
            self.history_count = 0

    def _resize_history(self, new_size):
        old = self.history
        new_history = np.full((new_size, 2), np.nan, dtype=np.float32)
        valid_count = int(max(0, min(int(self.history_count), old.shape[0], new_size)))
        if valid_count > 0:
            new_history[-valid_count:] = old[-valid_count:]
        self.history = new_history
        self.history_count = valid_count

    def process(self):
        max_len = int(self.max_history)
        if self.history.shape[0] != max_len:
            self._resize_history(max_len)

        if self.input_value is not None:
            arr = np.asarray(self.input_value, dtype=np.float32).flatten()
            if arr.size >= 2:
                point = arr[:2]
                if self.history_count > 0:
                    self.history[:-1] = self.history[1:]
                self.history[-1] = point
                if self.history_count < max_len:
                    self.history_count += 1

        if self.history_count > 0:
            valid = self.history[-self.history_count:]
            self.plot = valid
            self.info = f"pts: {self.history_count}"
        else:
            self.plot = np.empty((0, 2), dtype=np.float32)
            self.info = "pts: 0"

    def _on_reset_scatter(self, params: dict):
        max_len = int(self.max_history)
        self.history_count = 0
        self.history = np.full((max_len, 2), np.nan, dtype=np.float32)
        self.plot = np.empty((0, 2), dtype=np.float32)
        self.info = "pts: 0"
        return {"status": "ok"}
