"""Trigonometric operations for MiniCortex."""

import numpy as np

from axonforge.core.node import Node
from axonforge.core.descriptors.ports import InputPort, OutputPort
from axonforge.core.descriptors.fields import Range, Integer
from axonforge.core.descriptors.displays import Text
from axonforge.core.descriptors import branch


class Sin(Node):
    """Compute sine."""

    input_data = InputPort("Input", np.ndarray)
    output = OutputPort("Result", np.ndarray)
    
    info = Text("Info", default="No input")

    def process(self):
        if self.input_data is None:
            return
        
        self.output = np.sin(self.input_data).astype(np.float32)
        self.info = f"Shape: {self.output.shape}"


class Cos(Node):
    """Compute cosine."""

    input_data = InputPort("Input", np.ndarray)
    output = OutputPort("Result", np.ndarray)
    
    info = Text("Info", default="No input")

    def process(self):
        if self.input_data is None:
            return
        
        self.output = np.cos(self.input_data).astype(np.float32)
        self.info = f"Shape: {self.output.shape}"


class Tan(Node):
    """Compute tangent."""

    input_data = InputPort("Input", np.ndarray)
    output = OutputPort("Result", np.ndarray)
    
    info = Text("Info", default="No input")

    def process(self):
        if self.input_data is None:
            return
        
        self.output = np.tan(self.input_data).astype(np.float32)
        self.info = f"Shape: {self.output.shape}"


class Asin(Node):
    """Compute arcsine."""

    input_data = InputPort("Input", np.ndarray)
    output = OutputPort("Result", np.ndarray)
    
    info = Text("Info", default="No input")

    def process(self):
        if self.input_data is None:
            return
        
        # Clip to valid domain for asin
        x = np.clip(self.input_data, -1.0, 1.0)
        self.output = np.arcsin(x).astype(np.float32)
        self.info = f"Shape: {self.output.shape}"


class Acos(Node):
    """Compute arccosine."""

    input_data = InputPort("Input", np.ndarray)
    output = OutputPort("Result", np.ndarray)
    
    info = Text("Info", default="No input")

    def process(self):
        if self.input_data is None:
            return
        
        # Clip to valid domain for acos
        x = np.clip(self.input_data, -1.0, 1.0)
        self.output = np.arccos(x).astype(np.float32)
        self.info = f"Shape: {self.output.shape}"


class Atan(Node):
    """Compute arctangent."""

    input_data = InputPort("Input", np.ndarray)
    output = OutputPort("Result", np.ndarray)
    
    info = Text("Info", default="No input")

    def process(self):
        if self.input_data is None:
            return
        
        self.output = np.arctan(self.input_data).astype(np.float32)
        self.info = f"Shape: {self.output.shape}"


class Atan2(Node):
    """Compute arctangent of two inputs (y, x)."""

    input_y = InputPort("Y", np.ndarray)
    input_x = InputPort("X", np.ndarray)
    output = OutputPort("Result", np.ndarray)
    
    info = Text("Info", default="No input")

    def process(self):
        if self.input_y is None or self.input_x is None:
            self.info = "Waiting for inputs"
            return
        
        self.output = np.arctan2(self.input_y, self.input_x).astype(np.float32)
        self.info = f"Shape: {self.output.shape}"


class DegreesToRadians(Node):
    """Convert degrees to radians."""

    input_data = InputPort("Degrees", np.ndarray)
    output = OutputPort("Radians", np.ndarray)
    
    info = Text("Info", default="No input")

    def process(self):
        if self.input_data is None:
            return
        
        self.output = np.deg2rad(self.input_data).astype(np.float32)
        self.info = f"Shape: {self.output.shape}"


class RadiansToDegrees(Node):
    """Convert radians to degrees."""

    input_data = InputPort("Radians", np.ndarray)
    output = OutputPort("Degrees", np.ndarray)
    
    info = Text("Info", default="No input")

    def process(self):
        if self.input_data is None:
            return
        
        self.output = np.rad2deg(self.input_data).astype(np.float32)
        self.info = f"Shape: {self.output.shape}"
