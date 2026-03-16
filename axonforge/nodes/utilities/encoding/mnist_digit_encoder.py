"""MNIST digit sparse distributed encoder with vertical gaussian line.

This encoder takes an MNIST label (integer 0-9) as input and outputs a 28x28 matrix
with a vertical scanning line. The line position is determined by the digit value:
- 0 means line at the beginning (left edge)
- 9 means line at the end (right edge)
- Other values interpolate the position

The line has a gaussian vertical distribution with configurable width and falloff (std).
"""

import numpy as np

from ....core.node import Node
from ....core.descriptors.ports import InputPort, OutputPort
from ....core.descriptors.fields import Range, Integer
from ....core.descriptors.displays import Heatmap, Text
from ....core.descriptors.state import State
from ....core.descriptors import branch


class MNISTDigitEncoder(Node):
    """Encode MNIST digit labels into 28x28 matrices with vertical gaussian lines.
    
    The encoder maps integer labels (0-9) to spatial positions on a 28x28 grid.
    A vertical gaussian line is placed at a position interpolated between left (0)
    and right (9) based on the input digit.
    
    Properties:
        width: Width of the gaussian line in pixels (controls the spread)
        falloff: Falloff controls the standard deviation of the gaussian distribution
    """

    # ── Ports ─────────────────────────────────────────────────────────────
    i_label = InputPort("Label", int)
    o_output = OutputPort("Output", np.ndarray)

    # ── Properties ────────────────────────────────────────────────────────
    width = Integer("Width", default=3)
    falloff = Range("Falloff", default=1.0, min_val=0.1, max_val=10.0, scale="linear")

    # ── Displays ───────────────────────────────────────────────────────────
    preview = Heatmap("Preview", colormap="grayscale")
    info = Text("Info", default="No input")

    # ── State ─────────────────────────────────────────────────────────────
    last_label = State(default=None)

    # Output dimensions
    OUTPUT_HEIGHT = 28
    OUTPUT_WIDTH = 28

    def init(self):
        """Initialize the encoder with default values."""
        self.last_label = None

    def process(self):
        """Process the input label and generate the output matrix."""
        if self.i_label is None:
            return

        label = int(self.i_label)
        
        # Clamp label to valid range (0-9)
        label = max(0, min(9, label))

        # Calculate line position: 0 = left edge, 9 = right edge
        # Position is interpolated across the 28-pixel width
        # Using linear interpolation: position = label * (width - 1) / 9
        position = (label / 9.0) * (self.OUTPUT_WIDTH - 1)

        # Generate the vertical gaussian line
        output = self._generate_vertical_line(position)

        # Set outputs
        self.o_output = output
        self.preview = output
        self.last_label = label

        # Update info display
        width_val = int(self.width)
        falloff_val = float(self.falloff)
        self.info = f"Label: {label}, Position: {position:.2f}, Width: {width_val}, Falloff: {falloff_val:.2f}"

    def _generate_vertical_line(self, center_x: float) -> np.ndarray:
        """Generate a 28x28 matrix with a vertical gaussian line.
        
        Args:
            center_x: X-position of the line center (0 to 27)
            
        Returns:
            28x28 numpy array with gaussian line intensity
        """
        width_val = int(self.width)
        falloff_val = float(self.falloff)

        # Create coordinate grids
        y_coords, x_coords = np.meshgrid(
            np.arange(self.OUTPUT_HEIGHT),
            np.arange(self.OUTPUT_WIDTH),
            indexing='ij'
        )

        # Calculate horizontal distance from line center
        # Use gaussian falloff (std = falloff)
        # The width parameter affects how much of the gaussian we sample
        std = falloff_val
        half_width = max(1, width_val // 2)
        
        # Create horizontal gaussian profile
        # Gaussian: exp(-0.5 * ((x - center) / std)^2)
        horizontal_distance = x_coords - center_x
        gaussian_profile = np.exp(-0.5 * (horizontal_distance / std) ** 2)
        
        # Apply width scaling - reduce intensity at edges of the width
        # Use a smooth falloff based on width
        width_scale = np.exp(-0.5 * (horizontal_distance / (std * width_val / 2)) ** 2)
        
        # Combine: the line is brightest at center_x, falls off horizontally
        # Vertically it's constant (full height line)
        line = gaussian_profile * width_scale
        
        # Ensure the line spans the full height
        # The gaussian is applied horizontally, so it covers all y values
        
        return line.astype(np.float32)


class MNISTDigitEncoderSparse(MNISTDigitEncoder):
    """Sparse distributed version of MNIST digit encoder.
    
    This version creates a sparse representation by only keeping the most
    active pixels (those above a threshold), creating a more distributed
    encoding similar to sparse coding in the brain.
    """

    # ── Additional Properties ─────────────────────────────────────────────
    sparsity = Range("Sparsity", default=0.1, min_val=0.01, max_val=1.0, scale="linear")

    def _generate_vertical_line(self, center_x: float) -> np.ndarray:
        """Generate sparse vertical gaussian line."""
        # First generate the full gaussian line
        output = super()._generate_vertical_line(center_x)
        
        # Apply sparsity by keeping only top values
        sparsity_val = float(self.sparsity)
        
        # Calculate threshold to keep only top (1-sparsity)% of values
        threshold = np.percentile(output, (1 - sparsity_val) * 100)
        
        # Create sparse representation
        sparse_output = np.where(output >= threshold, output, 0.0)
        
        return sparse_output.astype(np.float32)


class AngleEncoder(Node):
    """Encode angle values (0-360 degrees) into 28x28 matrices with vertical gaussian lines.
    
    This encoder takes a float angle value (0-360 degrees) as input and outputs a 28x28 matrix
    with a vertical gaussian line. The line position is determined by the angle value:
    - 0 degrees means line at the beginning (left edge)
    - 360 degrees means line at the end (right edge)
    - Other values interpolate the position linearly
    
    This is similar to MNISTDigitEncoder but accepts continuous float values instead of
    discrete integers (0-9), providing higher resolution positioning.
    
    Properties:
        width: Width of the gaussian line in pixels
        falloff: Falloff controls the standard deviation of the gaussian distribution
    """

    # ── Ports ─────────────────────────────────────────────────────────────
    i_angle = InputPort("Angle", float)
    o_output = OutputPort("Output", np.ndarray)

    # ── Properties ───────────────────────────────────────────────────────────
    width = Integer("Width", default=3)
    falloff = Range("Falloff", default=1.0, min_val=0.1, max_val=10.0, scale="linear")

    # ── Displays ────────────────────────────────────────────────────────────
    preview = Heatmap("Preview", colormap="grayscale")
    info = Text("Info", default="No input")

    # ── State ─────────────────────────────────────────────────────────────
    last_angle = State(default=None)

    # Output dimensions
    OUTPUT_HEIGHT = 28
    OUTPUT_WIDTH = 28

    def init(self):
        """Initialize the encoder with default values."""
        self.last_angle = None

    def process(self):
        """Process the input angle and generate the output matrix."""
        if self.i_angle is None:
            return

        angle = float(self.i_angle)
        
        # Normalize angle to 0-360 range
        angle = angle % 360.0
        if angle < 0:
            angle += 360.0

        # Take 180-degree equivalent since vertical line is symmetric
        # 0° and 180° both point left, 90° and 270° both point right
        angle_180 = angle % 180.0

        # Calculate line position: 0 degrees = left edge, 180 degrees = right edge
        # Position is interpolated across the 28-pixel width
        position = (angle_180 / 180.0) * (self.OUTPUT_WIDTH - 1)

        # Generate the vertical gaussian line
        output = self._generate_vertical_line(position)

        # Set outputs
        self.o_output = output
        self.preview = output
        self.last_angle = angle

        # Update info display
        width_val = int(self.width)
        falloff_val = float(self.falloff)
        self.info = f"Angle: {angle:.1f}° → {angle_180:.1f}°, Position: {position:.2f}, Width: {width_val}, Falloff: {falloff_val:.2f}"

    def _generate_vertical_line(self, center_x: float) -> np.ndarray:
        """Generate a 28x28 matrix with a vertical gaussian line.
        
        Args:
            center_x: X-position of the line center (0 to 27)
            
        Returns:
            28x28 numpy array with gaussian line intensity
        """
        width_val = int(self.width)
        falloff_val = float(self.falloff)

        # Create coordinate grids
        y_coords, x_coords = np.meshgrid(
            np.arange(self.OUTPUT_HEIGHT),
            np.arange(self.OUTPUT_WIDTH),
            indexing='ij'
        )

        # Calculate horizontal distance from line center
        # Use gaussian falloff (std = falloff)
        std = falloff_val
        
        # Create horizontal gaussian profile
        # Gaussian: exp(-0.5 * ((x - center) / std)^2)
        horizontal_distance = x_coords - center_x
        gaussian_profile = np.exp(-0.5 * (horizontal_distance / std) ** 2)
        
        # Apply width scaling
        width_scale = np.exp(-0.5 * (horizontal_distance / (std * width_val / 2)) ** 2)
        
        # Combine: the line is brightest at center_x, falls off horizontally
        line = gaussian_profile * width_scale
        
        return line.astype(np.float32)


class AngleEncoderSparse(AngleEncoder):
    """Sparse distributed version of Angle encoder.
    
    This version creates a sparse representation by only keeping the most
    active pixels (those above a threshold).
    """

    # ── Additional Properties ─────────────────────────────────────────────
    sparsity = Range("Sparsity", default=0.1, min_val=0.01, max_val=1.0, scale="linear")

    def _generate_vertical_line(self, center_x: float) -> np.ndarray:
        """Generate sparse vertical gaussian line."""
        # First generate the full gaussian line
        output = super()._generate_vertical_line(center_x)
        
        # Apply sparsity by keeping only top values
        sparsity_val = float(self.sparsity)
        
        # Calculate threshold to keep only top (1-sparsity)% of values
        threshold = np.percentile(output, (1 - sparsity_val) * 100)
        
        # Create sparse representation
        sparse_output = np.where(output >= threshold, output, 0.0)
        
        return sparse_output.astype(np.float32)
