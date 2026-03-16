"""Plot nodes with history for MiniCortex."""

import numpy as np

from ...core.node import Node
from ...core.descriptors.ports import InputPort
from ...core.descriptors.state import State
from ...core.descriptors.fields import Integer, Enum, Float
from ...core.descriptors.displays import Plot, Text
from ...core.descriptors.actions import Action


class PlotBar(Node):
    """Plot numeric values in a history array using bar style."""

    # Input port for numeric values
    input_value = InputPort("Input", (int, float))
    
    # Field for maximum history length
    max_history = Integer("Max History", default=100)
    
    # Scale mode property - auto or manual
    scale_mode = Enum(
        "Scale Mode",
        options=["auto", "manual"],
        default="auto",
        on_change="_on_scale_changed"
    )
    
    # Manual scale range (used when scale_mode is "manual")
    vmin = Float("V Min", default=0.0, on_change="_on_scale_values_changed")
    vmax = Float("V Max", default=1.0, on_change="_on_scale_values_changed")
    
    # State for history - persists across sessions
    history = State("History", default=None)
    
    # State for current count of valid elements
    history_count = State("History Count", default=0)
    
    # Display for plotting the history as a signed bar plot
    plot = Plot("Plot", style="bar", scale_mode="auto")
    
    # Display for min/max info
    info = Text("Info", default="min: —  max: —")
    
    # Action to reset the history
    reset = Action("Reset", lambda self, params=None: self._on_reset(params))

    def _on_scale_changed(self, new_value, old_value):
        """Called when scale_mode property changes, updates the display config."""
        plot_descriptor = PlotBar.__dict__["plot"]
        if self.scale_mode == "manual":
            # Auto-set min/max to current history values when switching to manual
            if self.history_count > 0:
                max_len = int(self.max_history)
                if self.history_count < max_len:
                    display_array = self.history[-self.history_count:]
                else:
                    display_array = self.history.copy()
                dmin = float(np.nanmin(display_array))
                dmax = float(np.nanmax(display_array))
                self.vmin = dmin
                self.vmax = dmax
            plot_descriptor.change_scale(self, "manual", float(self.vmin), float(self.vmax))
        else:
            plot_descriptor.change_scale(self, "auto")

    def _on_scale_values_changed(self, new_value, old_value):
        """Called when vmin or vmax changes, updates the display config."""
        if self.scale_mode == "manual":
            plot_descriptor = PlotBar.__dict__["plot"]
            plot_descriptor.change_scale(self, "manual", float(self.vmin), float(self.vmax))

    def init(self):
        # Initialize history as numpy array if not already set
        max_len = int(self.max_history)
        if self.history is None:
            self.history = np.full(max_len, np.nan, dtype=np.float32)
        else:
            # Resize if needed when loading from storage
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
            valid_values = self.history[-valid_count:]
            new_history[-valid_count:] = valid_values

        self.history = new_history
        self.history_count = valid_count

    def process(self):
        max_len = int(self.max_history)
        
        # Check if max_history changed
        if len(self.history) != max_len:
            self._resize_history(max_len)
        
        if self.input_value is not None:
            # Convert input to float
            float_value = float(self.input_value)
            
            # Shift content left and add new value at the end
            if self.history_count > 0:
                # Shift existing values left
                self.history[:-1] = self.history[1:]
            
            # Add new value at the end
            self.history[-1] = float_value
            
            # Increment count, but cap at max_len
            if self.history_count < max_len:
                self.history_count += 1
        
        # Get the slice of valid data for display
        if self.history_count > 0:
            if self.history_count < max_len:
                valid_values = self.history[-self.history_count:]
            else:
                valid_values = self.history.copy()
            self.plot = valid_values
            
            # Calculate and display min/max
            dmin = float(np.nanmin(valid_values))
            dmax = float(np.nanmax(valid_values))
            self.info = f"min: {dmin:.2f}  max: {dmax:.2f}"
        else:
            # Empty plot when no history
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

    # Input port for numeric values
    input_value = InputPort("Input", (int, float))
    
    # Field for maximum history length
    max_history = Integer("Max History", default=100)
    
    # Scale mode property - auto or manual
    scale_mode = Enum(
        "Scale Mode",
        options=["auto", "manual"],
        default="auto",
        on_change="_on_scale_changed"
    )
    
    # Manual scale range (used when scale_mode is "manual")
    vmin = Float("V Min", default=0.0, on_change="_on_scale_values_changed")
    vmax = Float("V Max", default=1.0, on_change="_on_scale_values_changed")
    
    # State for history - persists across sessions
    history = State("History", default=None)
    
    # State for current count of valid elements
    history_count = State("History Count", default=0)
    
    # Display for plotting the history as a line plot
    plot = Plot(
        "Plot",
        style="line",
        color_positive="#00f5ff",
        color_negative="#00f5ff",
        line_width=0.7,
        scale_mode="auto",
    )
    
    # Display for min/max info
    info = Text("Info", default="min: —  max: —")
    
    # Action to reset the history
    reset = Action("Reset", lambda self, params=None: self._on_reset(params))

    def _on_scale_changed(self, new_value, old_value):
        """Called when scale_mode property changes, updates the display config."""
        plot_descriptor = PlotLine.__dict__["plot"]
        if self.scale_mode == "manual":
            # Auto-set min/max to current history values when switching to manual
            if self.history_count > 0:
                max_len = int(self.max_history)
                if self.history_count < max_len:
                    display_array = self.history[-self.history_count:]
                else:
                    display_array = self.history.copy()
                dmin = float(np.nanmin(display_array))
                dmax = float(np.nanmax(display_array))
                self.vmin = dmin
                self.vmax = dmax
            plot_descriptor.change_scale(self, "manual", float(self.vmin), float(self.vmax))
        else:
            plot_descriptor.change_scale(self, "auto")

    def _on_scale_values_changed(self, new_value, old_value):
        """Called when vmin or vmax changes, updates the display config."""
        if self.scale_mode == "manual":
            plot_descriptor = PlotLine.__dict__["plot"]
            plot_descriptor.change_scale(self, "manual", float(self.vmin), float(self.vmax))

    def init(self):
        # Initialize history as numpy array if not already set
        max_len = int(self.max_history)
        if self.history is None:
            self.history = np.full(max_len, np.nan, dtype=np.float32)
        else:
            # Resize if needed when loading from storage
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
            valid_values = self.history[-valid_count:]
            new_history[-valid_count:] = valid_values

        self.history = new_history
        self.history_count = valid_count

    def process(self):
        max_len = int(self.max_history)
        
        # Check if max_history changed
        if len(self.history) != max_len:
            self._resize_history(max_len)
        
        if self.input_value is not None:
            # Convert input to float
            float_value = float(self.input_value)
            
            # Shift content left and add new value at the end
            if self.history_count > 0:
                # Shift existing values left
                self.history[:-1] = self.history[1:]
            
            # Add new value at the end
            self.history[-1] = float_value
            
            # Increment count, but cap at max_len
            if self.history_count < max_len:
                self.history_count += 1
        
        # Get the slice of valid data for display
        if self.history_count > 0:
            if self.history_count < max_len:
                valid_values = self.history[-self.history_count:]
            else:
                valid_values = self.history.copy()
            self.plot = valid_values
            
            # Calculate and display min/max
            dmin = float(np.nanmin(valid_values))
            dmax = float(np.nanmax(valid_values))
            self.info = f"min: {dmin:.2f}  max: {dmax:.2f}"
        else:
            # Empty plot when no history
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
