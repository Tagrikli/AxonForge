"""1D Vector display node for MiniCortex."""

import numpy as np

from ...core.node import Node
from ...core.descriptors.ports import InputPort, OutputPort
from ...core.descriptors.fields import Float, Enum, Bool
from ...core.descriptors.displays import Plot, Text, Heatmap, Image
from ..utilities.display_helpers import to_display_grid


class DisplayVector(Node):
    """Display 1D numpy array as a bar chart."""

    input_data = InputPort("Input", np.ndarray)
    output_data = OutputPort("Output", np.ndarray)

    scale_mode = Enum("Scale", options=["auto", "manual"], default="auto")
    vmin = Float("V Min", default=-1.0)
    vmax = Float("V Max", default=1.0)

    display = Plot("Plot", style="bar", scale_mode="auto", vmin=-1.0, vmax=1.0)
    info = Text("Info", default="No data")

    def _sync_display_config(self) -> None:
        display_descriptor = type(self).__dict__["display"]
        display_descriptor.change_scale(
            self,
            str(self.scale_mode),
            float(self.vmin),
            float(self.vmax),
        )

    def process(self):
        self._sync_display_config()
        if self.input_data is None:
            self.info = "No data"
            return

        # Flatten to 1D
        arr = self.input_data.flatten().astype(np.float32)

        # Set outputs - pass input directly to display
        self.output_data = arr
        self.display = arr
        self.info = f"Shape: {self.input_data.shape} → {arr.shape[0]}"


"""DisplayHeatmap node for MiniCortex - unified display for scalar arrays."""


class DisplayHeatmap(Node):
    """Display 1D/2D scalar data as a configurable heatmap.

    Heatmap controls:
    - colormap
    - scale mode
    - vmin / vmax for manual scaling
    """

    input_data = InputPort("Input", np.ndarray)
    display = Heatmap("Display", colormap="grayscale")

    colormap = Enum(
        "Colormap",
        options=["grayscale", "bwr", "viridis"],
        default="grayscale",
    )
    scale_mode = Enum("Scale", options=["auto", "manual"], default="auto")
    vmin = Float("V Min", default=0.0)
    vmax = Float("V Max", default=1.0)
    use_grid_display = Bool("Grid Display")

    def _sync_display_config(self) -> None:
        display_descriptor = type(self).__dict__["display"]
        display_descriptor.change_colormap(self, str(self.colormap))
        display_descriptor.change_scale(
            self,
            str(self.scale_mode),
            float(self.vmin),
            float(self.vmax),
        )

    def process(self):
        self._sync_display_config()
        if self.input_data is not None and hasattr(self.input_data, "ndim"):
            data = self.input_data
            if bool(self.use_grid_display):
                data = to_display_grid(data)
            elif data.ndim == 1:
                # Reshape 1D to 2D (single row)
                data = data.reshape(1, -1)
            self.display = data


class DisplayImage(Node):
    """Display an image array without altering its contents."""

    input_image = InputPort("Input", np.ndarray)
    output_image = OutputPort("Output", np.ndarray)
    display = Image("Display")
    info = Text("Info", default="No image")

    def process(self):
        if self.input_image is None:
            self.info = "No image"
            return

        image = self.input_image
        self.output_image = image
        self.display = image

        shape = getattr(image, "shape", None)
        if shape is not None:
            self.info = f"Shape: {shape}"
        else:
            self.info = "Image"
