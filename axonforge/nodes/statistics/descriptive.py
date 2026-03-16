"""Statistics nodes for MiniCortex."""

import numpy as np

from axonforge.core.node import Node
from axonforge.core.descriptors.ports import InputPort, OutputPort
from axonforge.core.descriptors.displays import Text, Numeric
from axonforge.core.descriptors import branch


class Mean(Node):
    """Compute mean of input array."""

    input_data = InputPort("Input", np.ndarray)
    mean = OutputPort("Mean", float)
    
    display = Numeric("Mean", format=".4f")
    info = Text("Info", default="No input")

    def process(self):
        if self.input_data is None:
            return

        m = float(np.mean(self.input_data))
        self.mean = m
        self.display = m
        self.info = f"Mean: {m:.4f}"


class Std(Node):
    """Compute standard deviation of input array."""

    input_data = InputPort("Input", np.ndarray)
    std = OutputPort("Std Dev", float)
    
    display = Numeric("Std Dev", format=".4f")
    info = Text("Info", default="No input")

    def process(self):
        if self.input_data is None:
            return

        s = float(np.std(self.input_data))
        self.std = s
        self.display = s
        self.info = f"Std Dev: {s:.4f}"


class Variance(Node):
    """Compute variance of input array."""

    input_data = InputPort("Input", np.ndarray)
    variance = OutputPort("Variance", float)
    
    display = Numeric("Variance", format=".4f")
    info = Text("Info", default="No input")

    def process(self):
        if self.input_data is None:
            return

        v = float(np.var(self.input_data))
        self.variance = v
        self.display = v
        self.info = f"Variance: {v:.4f}"


class Sum(Node):
    """Compute sum of input array."""

    input_data = InputPort("Input", np.ndarray)
    sum_out = OutputPort("Sum", float)
    
    display = Numeric("Sum", format=".4f")
    info = Text("Info", default="No input")

    def process(self):
        if self.input_data is None:
            return

        s = float(np.sum(self.input_data))
        self.sum_out = s
        self.display = s
        self.info = f"Sum: {s:.4f}"


class Min(Node):
    """Compute minimum value of input array."""

    input_data = InputPort("Input", np.ndarray)
    min_val = OutputPort("Min", float)
    
    display = Numeric("Min", format=".4f")
    info = Text("Info", default="No input")

    def process(self):
        if self.input_data is None:
            return

        m = float(np.min(self.input_data))
        self.min_val = m
        self.display = m
        self.info = f"Min: {m:.4f}"


class Max(Node):
    """Compute maximum value of input array."""

    input_data = InputPort("Input", np.ndarray)
    max_val = OutputPort("Max", float)
    
    display = Numeric("Max", format=".4f")
    info = Text("Info", default="No input")

    def process(self):
        if self.input_data is None:
            return

        m = float(np.max(self.input_data))
        self.max_val = m
        self.display = m
        self.info = f"Max: {m:.4f}"


class ArgMax(Node):
    """Find index of maximum value."""

    input_data = InputPort("Input", np.ndarray)
    argmax = OutputPort("ArgMax", int)
    
    display = Numeric("ArgMax")
    info = Text("Info", default="No input")

    def process(self):
        if self.input_data is None:
            return

        idx = int(np.argmax(self.input_data))
        self.argmax = idx
        self.display = idx
        self.info = f"ArgMax index: {idx}"


class ArgMin(Node):
    """Find index of minimum value."""

    input_data = InputPort("Input", np.ndarray)
    argmin = OutputPort("ArgMin", int)
    
    display = Numeric("ArgMin")
    info = Text("Info", default="No input")

    def process(self):
        if self.input_data is None:
            return

        idx = int(np.argmin(self.input_data))
        self.argmin = idx
        self.display = idx
        self.info = f"ArgMin index: {idx}"


class Percentile(Node):
    """Compute percentile of input array."""

    input_data = InputPort("Input", np.ndarray)
    percentile = OutputPort("Percentile", float)
    
    percent = ("Percentile", 50.0, 0.0, 100.0)
    display = Numeric("Percentile", format=".4f")
    info = Text("Info", default="No input")

    def process(self):
        if self.input_data is None:
            return

        p = float(np.percentile(self.input_data.flatten(), self.percent))
        self.percentile = p
        self.display = p
        self.info = f"{self.percent}th percentile: {p:.4f}"


class Histogram(Node):
    """Compute histogram of input array."""

    input_data = InputPort("Input", np.ndarray)
    output = OutputPort("Histogram", np.ndarray)
    
    bins = ("Bins", 10, 2, 100)
    info = Text("Info", default="No input")

    def process(self):
        if self.input_data is None:
            return

        hist, edges = np.histogram(self.input_data.flatten(), bins=self.bins)
        self.output = hist.astype(np.float32)
        self.info = f"Histogram: {hist.shape}"


class Covariance(Node):
    """Compute covariance matrix of two inputs."""

    input_a = InputPort("A", np.ndarray)
    input_b = InputPort("B", np.ndarray)
    covariance = OutputPort("Covariance", float)
    
    display = Numeric("Covariance", format=".4f")
    info = Text("Info", default="No input")

    def process(self):
        if self.input_a is None or self.input_b is None:
            self.info = "Waiting for inputs"
            return

        try:
            cov = float(np.cov(self.input_a.flatten(), self.input_b.flatten())[0, 1])
            self.covariance = cov
            self.display = cov
            self.info = f"Covariance: {cov:.4f}"
        except Exception as e:
            self.info = f"Error: {e}"


class Correlation(Node):
    """Compute Pearson correlation coefficient."""

    input_a = InputPort("A", np.ndarray)
    input_b = InputPort("B", np.ndarray)
    correlation = OutputPort("Correlation", float)
    
    display = Numeric("Correlation", format=".4f")
    info = Text("Info", default="No input")

    def process(self):
        if self.input_a is None or self.input_b is None:
            self.info = "Waiting for inputs"
            return

        try:
            corr = float(np.corrcoef(self.input_a.flatten(), self.input_b.flatten())[0, 1])
            self.correlation = corr
            self.display = corr
            self.info = f"Correlation: {corr:.4f}"
        except Exception as e:
            self.info = f"Error: {e}"
