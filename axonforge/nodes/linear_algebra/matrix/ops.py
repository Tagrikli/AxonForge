"""Matrix operations for MiniCortex."""

import numpy as np

from axonforge.core.node import Node
from axonforge.core.descriptors.ports import InputPort, OutputPort
from axonforge.core.descriptors.fields import Integer
from axonforge.core.descriptors.displays import Text
from axonforge.core.descriptors import branch


class MatrixTranspose(Node):
    """Transpose a matrix."""

    input_data = InputPort("Input", np.ndarray)
    output = OutputPort("Output", np.ndarray)
    
    info = Text("Info", default="No input")

    def process(self):
        if self.input_data is None:
            return

        self.output = self.input_data.T.astype(np.float32)
        self.info = f"{self.input_data.shape} → {self.output.shape}"


class MatrixMultiply(Node):
    """Matrix multiplication (A @ B)."""

    input_a = InputPort("A", np.ndarray)
    input_b = InputPort("B", np.ndarray)
    output = OutputPort("Result", np.ndarray)
    
    info = Text("Info", default="No input")

    def process(self):
        if self.input_a is None or self.input_b is None:
            self.info = "Waiting for inputs"
            return

        try:
            self.output = (self.input_a @ self.input_b).astype(np.float32)
            self.info = f"{self.input_a.shape} @ {self.input_b.shape} → {self.output.shape}"
        except Exception as e:
            self.info = f"Error: {e}"


class MatrixDeterminant(Node):
    """Compute determinant of a square matrix."""

    input_data = InputPort("Input", np.ndarray)
    determinant = OutputPort("Determinant", float)
    
    info = Text("Info", default="No input")

    def process(self):
        if self.input_data is None:
            return

        try:
            det = float(np.linalg.det(self.input_data))
            self.determinant = det
            self.info = f"Det: {det:.4f}"
        except Exception as e:
            self.info = f"Error: {e}"


class MatrixInverse(Node):
    """Compute inverse of a square matrix."""

    input_data = InputPort("Input", np.ndarray)
    output = OutputPort("Inverse", np.ndarray)
    
    info = Text("Info", default="No input")

    def process(self):
        if self.input_data is None:
            return

        try:
            inv = np.linalg.inv(self.input_data)
            self.output = inv.astype(np.float32)
            self.info = f"Inverse computed"
        except Exception as e:
            self.info = f"Error: {e}"


class MatrixIdentity(Node):
    """Create an identity matrix."""

    output = OutputPort("Output", np.ndarray)
    
    size = Integer("Size", default=3)
    info = Text("Info", default="No input")

    def process(self):
        n = int(self.size)
        self.output = np.eye(n, dtype=np.float32)
        self.info = f"Identity {n}x{n}"


class MatrixZeros(Node):
    """Create a zeros matrix."""

    output = OutputPort("Output", np.ndarray)
    
    rows = Integer("Rows", default=3)
    cols = Integer("Cols", default=3)
    info = Text("Info", default="No input")

    def process(self):
        r = int(self.rows)
        c = int(self.cols)
        self.output = np.zeros((r, c), dtype=np.float32)
        self.info = f"Zeros {r}x{c}"


class MatrixOnes(Node):
    """Create a ones matrix."""

    output = OutputPort("Output", np.ndarray)
    
    rows = Integer("Rows", default=3)
    cols = Integer("Cols", default=3)
    info = Text("Info", default="No input")

    def process(self):
        r = int(self.rows)
        c = int(self.cols)
        self.output = np.ones((r, c), dtype=np.float32)
        self.info = f"Ones {r}x{c}"


class MatrixReshape(Node):
    """Reshape a matrix to new dimensions."""

    input_data = InputPort("Input", np.ndarray)
    output = OutputPort("Output", np.ndarray)
    
    rows = Integer("Rows", default=-1)
    cols = Integer("Cols", default=-1)
    info = Text("Info", default="No input")

    def process(self):
        if self.input_data is None:
            return

        total = self.input_data.size
        r = int(self.rows)
        c = int(self.cols)

        if r == -1 and c == -1:
            self.info = "Specify at least one dimension"
            return
        
        if r == -1:
            r = total // c
        if c == -1:
            c = total // r

        if r * c != total:
            self.info = f"Cannot reshape {self.input_data.shape} to ({r}, {c})"
            return

        self.output = self.input_data.reshape(r, c).astype(np.float32)
        self.info = f"{self.input_data.shape} → {self.output.shape}"


class MatrixFlatten(Node):
    """Flatten a matrix to 1D."""

    input_data = InputPort("Input", np.ndarray)
    output = OutputPort("Output", np.ndarray)
    
    info = Text("Info", default="No input")

    def process(self):
        if self.input_data is None:
            return

        self.output = self.input_data.flatten().astype(np.float32)
        self.info = f"{self.input_data.shape} → {self.output.shape}"


class MatrixTransposeProjection(Node):
    """Project input onto transposed matrix: output = W.T @ x.

    Useful for reconstruction operations where you need to project
    activations back through transposed weights.
    """

    weights = InputPort("Weights (W)", np.ndarray)
    input_data = InputPort("Input (x)", np.ndarray)
    output = OutputPort("Output (W.T @ x)", np.ndarray)
    
    info = Text("Info", default="No input")

    def process(self):
        if self.weights is None or self.input_data is None:
            self.info = "Waiting for inputs"
            return

        try:
            # Flatten input if needed to ensure 1D vector
            x = self.input_data.flatten()
            # Compute W.T @ x
            result = self.weights.T @ x
            self.output = result.astype(np.float32)
            self.info = f"{self.weights.shape}.T @ {x.shape} → {self.output.shape}"
        except Exception as e:
            self.info = f"Error: {e}"


class MatrixCrop(Node):
    """Crop a matrix according to start and end coordinates."""

    input_data = InputPort("Input", np.ndarray)
    output = OutputPort("Output", np.ndarray)
    
    x_start = Integer("X Start", default=0)
    y_start = Integer("Y Start", default=0)
    x_end = Integer("X End", default=10)
    y_end = Integer("Y End", default=10)
    
    info = Text("Info", default="No input")

    def process(self):
        if self.input_data is None:
            return

        try:
            # In numpy, first dimension is rows (Y), second is columns (X)
            # So we use [y_start:y_end, x_start:x_end]
            y1, y2 = int(self.y_start), int(self.y_end)
            x1, x2 = int(self.x_start), int(self.x_end)
            
            # Basic validation/clamping
            h, w = self.input_data.shape[:2]
            y1 = max(0, min(h, y1))
            y2 = max(0, min(h, y2))
            x1 = max(0, min(w, x1))
            x2 = max(0, min(w, x2))
            
            if y1 >= y2 or x1 >= x2:
                self.info = "Invalid crop range"
                return

            self.output = self.input_data[y1:y2, x1:x2].copy().astype(np.float32)
            self.info = f"{self.input_data.shape} → {self.output.shape}"
        except Exception as e:
            self.info = f"Error: {e}"
