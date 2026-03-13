"""1D Vector display node for MiniCortex."""

import numpy as np

from ...core.node import Node
from ...core.descriptors.ports import InputPort, OutputPort
from ...core.descriptors.properties import Float, Enum, Bool
from ...core.descriptors.displays import BarChart, Text, Vector2D
from ..hypercolumn.utilities import to_display_grid


class DisplayVector(Node):
    """Display 1D numpy array as a bar chart."""

    input_data = InputPort("Input", np.ndarray)
    output_data = OutputPort("Output", np.ndarray)
    display = BarChart("Plot", scale_min=-1.0, scale_max=1.0)
    info = Text("Info", default="No data")

    def process(self):
        if self.input_data is None:
            self.info = "No data"
            return

        # Flatten to 1D
        arr = self.input_data.flatten().astype(np.float32)

        # Set outputs - pass input directly to display
        self.output_data = arr
        self.display = arr
        self.info = f"Shape: {self.input_data.shape} → {arr.shape[0]}"


"""DisplayMatrix node for MiniCortex - unified display for 2D arrays."""


class DisplayMatrix(Node):
    """Display 2D array with selectable colormap.

    Input: 2D numpy array
    - Grayscale: expects [0, 1] range
    - BWR: expects [-1, 1] range
    - Viridis: expects [-1, 1] range
    """

    input_data = InputPort("Input", np.ndarray)
    display = Vector2D("Display", color_mode="grayscale")

    # Colormap selection
    colormap = Enum(
        "Colormap",
        options=["grayscale", "bwr", "viridis"],
        default="grayscale",
        on_change="_on_colormap_changed",
    )
    use_grid_display = Bool("Grid Display")

    def _on_colormap_changed(self, new_value, old_value):
        """Called when colormap property changes, updates the display config."""
        display_descriptor = DisplayMatrix.__dict__["display"]
        display_descriptor.change_color_mode(new_value, obj=self)

    def process(self):
        if self.input_data is not None and hasattr(self.input_data, "ndim"):
            data = self.input_data
            if bool(self.use_grid_display):
                data = to_display_grid(data)
            elif data.ndim == 1:
                # Reshape 1D to 2D (single row)
                data = data.reshape(1, -1)
            self.display = data
