"""Arithmetic operations for MiniCortex."""

import numpy as np

from axonforge.core.node import Node
from axonforge.core.descriptors.ports import InputPort, OutputPort
from axonforge.core.descriptors.fields import Range, Integer
from axonforge.core.descriptors.displays import Text, Numeric
from axonforge.core.descriptors import branch


class Add(Node):
    """Add two inputs element-wise."""

    input_a = InputPort("A", np.ndarray)
    input_b = InputPort("B", np.ndarray)
    output = OutputPort("Result", np.ndarray)
    
    info = Text("Info", default="No input")

    def process(self):
        if self.input_a is None or self.input_b is None:
            self.info = "Waiting for inputs"
            return
        
        self.output = (self.input_a + self.input_b).astype(np.float32)
        self.info = f"Shape: {self.output.shape}"


class Subtract(Node):
    """Subtract second input from first."""

    input_a = InputPort("A", np.ndarray)
    input_b = InputPort("B", np.ndarray)
    output = OutputPort("Result", np.ndarray)
    
    info = Text("Info", default="No input")

    def process(self):
        if self.input_a is None or self.input_b is None:
            self.info = "Waiting for inputs"
            return
        
        self.output = (self.input_a - self.input_b).astype(np.float32)
        self.info = f"Shape: {self.output.shape}"


class Multiply(Node):
    """Multiply two inputs element-wise."""

    input_a = InputPort("A", np.ndarray)
    input_b = InputPort("B", np.ndarray)
    output = OutputPort("Result", np.ndarray)
    
    info = Text("Info", default="No input")

    def process(self):
        if self.input_a is None or self.input_b is None:
            self.info = "Waiting for inputs"
            return
        
        self.output = (self.input_a * self.input_b).astype(np.float32)
        self.info = f"Shape: {self.output.shape}"


class Divide(Node):
    """Divide first input by second."""

    input_a = InputPort("A", np.ndarray)
    input_b = InputPort("B", np.ndarray)
    output = OutputPort("Result", np.ndarray)
    
    epsilon = Range("Epsilon", 1e-8, 1e-12, 1e-2, scale="log")
    info = Text("Info", default="No input")

    def process(self):
        if self.input_a is None or self.input_b is None:
            self.info = "Waiting for inputs"
            return
        
        eps = float(self.epsilon)
        self.output = (self.input_a / (self.input_b + eps)).astype(np.float32)
        self.info = f"Shape: {self.output.shape}"


class Power(Node):
    """Raise input to a power."""

    input_data = InputPort("Input", np.ndarray)
    output = OutputPort("Result", np.ndarray)
    
    exponent = Range("Exponent", 2.0, 0.1, 10.0, scale="linear")
    info = Text("Info", default="No input")

    def process(self):
        if self.input_data is None:
            return
        
        exp = float(self.exponent)
        self.output = (np.power(self.input_data, exp)).astype(np.float32)
        self.info = f"x^{exp}"


class SquareRoot(Node):
    """Compute square root."""

    input_data = InputPort("Input", np.ndarray)
    output = OutputPort("Result", np.ndarray)
    
    info = Text("Info", default="No input")

    def process(self):
        if self.input_data is None:
            return
        
        self.output = np.sqrt(np.maximum(0, self.input_data)).astype(np.float32)
        self.info = f"Shape: {self.output.shape}"


class Absolute(Node):
    """Compute absolute value."""

    input_data = InputPort("Input", np.ndarray)
    output = OutputPort("Result", np.ndarray)
    
    info = Text("Info", default="No input")

    def process(self):
        if self.input_data is None:
            return
        
        self.output = np.abs(self.input_data).astype(np.float32)
        self.info = f"Shape: {self.output.shape}"


class Negative(Node):
    """Negate input."""

    input_data = InputPort("Input", np.ndarray)
    output = OutputPort("Result", np.ndarray)
    
    info = Text("Info", default="No input")

    def process(self):
        if self.input_data is None:
            return
        
        self.output = (-self.input_data).astype(np.float32)
        self.info = f"Shape: {self.output.shape}"


class Modulo(Node):
    """Compute modulo operation."""

    input_a = InputPort("A", np.ndarray)
    input_b = InputPort("B", np.ndarray)
    output = OutputPort("Result", np.ndarray)
    
    info = Text("Info", default="No input")

    def process(self):
        if self.input_a is None or self.input_b is None:
            self.info = "Waiting for inputs"
            return
        
        # Avoid division by zero
        b = np.where(self.input_b == 0, 1, self.input_b)
        self.output = (self.input_a % b).astype(np.float32)
        self.info = f"Shape: {self.output.shape}"


class Clip(Node):
    """Clip values to a range."""

    input_data = InputPort("Input", np.ndarray)
    output = OutputPort("Result", np.ndarray)
    
    min_val = Range("Min", 0.0, -1.0, 1.0, scale="linear")
    max_val = Range("Max", 1.0, 0.0, 2.0, scale="linear")
    info = Text("Info", default="No input")

    def process(self):
        if self.input_data is None:
            return
        
        min_v = float(self.min_val)
        max_v = float(self.max_val)
        self.output = np.clip(self.input_data, min_v, max_v).astype(np.float32)
        self.info = f"Clipped to [{min_v}, {max_v}]"
