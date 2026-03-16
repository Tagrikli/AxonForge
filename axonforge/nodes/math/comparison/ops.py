"""Comparison operations for MiniCortex."""

import numpy as np

from axonforge.core.node import Node
from axonforge.core.descriptors.ports import InputPort, OutputPort
from axonforge.core.descriptors.fields import Range
from axonforge.core.descriptors.displays import Text
from axonforge.core.descriptors import branch


class GreaterThan(Node):
    """Return 1 where a > b, else 0."""

    input_a = InputPort("A", np.ndarray)
    input_b = InputPort("B", np.ndarray)
    output = OutputPort("Result", np.ndarray)
    
    info = Text("Info", default="No input")

    def process(self):
        if self.input_a is None or self.input_b is None:
            self.info = "Waiting for inputs"
            return
        
        self.output = (self.input_a > self.input_b).astype(np.float32)
        self.info = f"Shape: {self.output.shape}"


class LessThan(Node):
    """Return 1 where a < b, else 0."""

    input_a = InputPort("A", np.ndarray)
    input_b = InputPort("B", np.ndarray)
    output = OutputPort("Result", np.ndarray)
    
    info = Text("Info", default="No input")

    def process(self):
        if self.input_a is None or self.input_b is None:
            self.info = "Waiting for inputs"
            return
        
        self.output = (self.input_a < self.input_b).astype(np.float32)
        self.info = f"Shape: {self.output.shape}"


class GreaterEqual(Node):
    """Return 1 where a >= b, else 0."""

    input_a = InputPort("A", np.ndarray)
    input_b = InputPort("B", np.ndarray)
    output = OutputPort("Result", np.ndarray)
    
    info = Text("Info", default="No input")

    def process(self):
        if self.input_a is None or self.input_b is None:
            self.info = "Waiting for inputs"
            return
        
        self.output = (self.input_a >= self.input_b).astype(np.float32)
        self.info = f"Shape: {self.output.shape}"


class LessEqual(Node):
    """Return 1 where a <= b, else 0."""

    input_a = InputPort("A", np.ndarray)
    input_b = InputPort("B", np.ndarray)
    output = OutputPort("Result", np.ndarray)
    
    info = Text("Info", default="No input")

    def process(self):
        if self.input_a is None or self.input_b is None:
            self.info = "Waiting for inputs"
            return
        
        self.output = (self.input_a <= self.input_b).astype(np.float32)
        self.info = f"Shape: {self.output.shape}"


class Equal(Node):
    """Return 1 where a == b, else 0."""

    input_a = InputPort("A", np.ndarray)
    input_b = InputPort("B", np.ndarray)
    output = OutputPort("Result", np.ndarray)
    
    info = Text("Info", default="No input")

    def process(self):
        if self.input_a is None or self.input_b is None:
            self.info = "Waiting for inputs"
            return
        
        self.output = (self.input_a == self.input_b).astype(np.float32)
        self.info = f"Shape: {self.output.shape}"


class NotEqual(Node):
    """Return 1 where a != b, else 0."""

    input_a = InputPort("A", np.ndarray)
    input_b = InputPort("B", np.ndarray)
    output = OutputPort("Result", np.ndarray)
    
    info = Text("Info", default="No input")

    def process(self):
        if self.input_a is None or self.input_b is None:
            self.info = "Waiting for inputs"
            return
        
        self.output = (self.input_a != self.input_b).astype(np.float32)
        self.info = f"Shape: {self.output.shape}"


class Maximum(Node):
    """Element-wise maximum of two inputs."""

    input_a = InputPort("A", np.ndarray)
    input_b = InputPort("B", np.ndarray)
    output = OutputPort("Result", np.ndarray)
    
    info = Text("Info", default="No input")

    def process(self):
        if self.input_a is None or self.input_b is None:
            self.info = "Waiting for inputs"
            return
        
        self.output = np.maximum(self.input_a, self.input_b).astype(np.float32)
        self.info = f"Shape: {self.output.shape}"


class Minimum(Node):
    """Element-wise minimum of two inputs."""

    input_a = InputPort("A", np.ndarray)
    input_b = InputPort("B", np.ndarray)
    output = OutputPort("Result", np.ndarray)
    
    info = Text("Info", default="No input")

    def process(self):
        if self.input_a is None or self.input_b is None:
            self.info = "Waiting for inputs"
            return
        
        self.output = np.minimum(self.input_a, self.input_b).astype(np.float32)
        self.info = f"Shape: {self.output.shape}"
