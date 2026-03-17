"""Utility nodes for MiniCortex."""

import numpy as np

from axonforge.core.node import Node
from axonforge.core.descriptors.ports import InputPort, OutputPort
from axonforge.core.descriptors.fields import Integer, Range, Float as FloatProp, Enum
from axonforge.core.descriptors.displays import Text, Numeric
from axonforge.core.descriptors.actions import Action
from axonforge.core.descriptors.state import State


class Uniformity(Node):
    """Calculate non-uniformity of a numpy array using entropy-based measure."""

    input_data = InputPort("Input", np.ndarray)
    non_uniformity = OutputPort("Non-Uniformity", float)
    entropy = OutputPort("Entropy", float)
    display = Numeric("Non-Uniformity", format=".4f")
    info = Text("Info", default="No data")

    def process(self):
        if self.input_data is None:
            return

        # Flatten and L2 normalize
        v = self.input_data.flatten().astype(np.float64)
        norm = np.linalg.norm(v)

        if norm > 0:
            v = v / norm
        else:
            self.non_uniformity = 0.0
            self.entropy = 0.0
            self.display = 0.0
            self.info = "Zero vector"
            return

        # Compute probability distribution (sums to 1 by construction)
        p = v**2

        # Compute entropy
        entropy_val = -np.sum(p * np.log(p + 1e-9))

        # Compute non-uniformity (0 = uniform, 1 = concentrated)
        max_entropy = np.log(v.size)
        non_uniformity_val = 1.0 - entropy_val / max_entropy if max_entropy > 0 else 0.0

        # Set outputs
        self.non_uniformity = float(non_uniformity_val)
        self.entropy = float(entropy_val)
        self.display = float(non_uniformity_val)
        uniformity_val = 1.0 - non_uniformity_val
        self.info = f"Uniformity: {uniformity_val:.4f}\nNon-Uniformity: {non_uniformity_val:.4f}\nEntropy: {entropy_val:.4f}\nMax Entropy: {max_entropy:.4f}"




class TopK(Node):
    """Keep only the top K values in an array, set rest to zero."""

    input_data = InputPort("Input", np.ndarray)
    output_data = OutputPort("Output", np.ndarray)
    k = Integer("K", default=10)

    def process(self):
        if self.input_data is None:
            return

        # State original shape
        original_shape = self.input_data.shape

        # Flatten the array
        flat = self.input_data.flatten().astype(np.float64)

        # Get k value (ensure it doesn't exceed array size)
        k_val = min(int(self.k), len(flat))

        if k_val <= 0:
            self.output_data = np.zeros(original_shape, dtype=np.float32)
            return

        # Find the k-th largest value (threshold)
        # Use partition for efficiency: all values >= threshold are in the top k
        threshold = np.partition(flat, -k_val)[-k_val]

        # Create output: keep top k values, set rest to zero
        result = np.where(flat >= threshold, flat, 0.0)

        # Reshape back to original shape
        self.output_data = result.reshape(original_shape).astype(np.float32)


class JaccardIndex(Node):
    """Calculate Jaccard Index (Intersection over Union) between two arrays.

    Converts positive values to 1 and non-positive values to 0.
    Then computes: intersection / union
    """

    input_true = InputPort("True", np.ndarray)
    input_predicted = InputPort("Predicted", np.ndarray)
    output_iou = OutputPort("IoU", float)

    display = Numeric("Jaccard Index", format=".4f")
    info = Text("Info", default="No data")

    def process(self):
        if self.input_true is None or self.input_predicted is None:
            self.info = "Waiting for inputs"
            return

        # Flatten arrays
        a = self.input_true.flatten().astype(np.float32)
        b = self.input_predicted.flatten().astype(np.float32)

        # Check if arrays are same size
        if len(a) != len(b):
            self.info = f"Size mismatch: {len(a)} vs {len(b)}"
            self.output_iou = 0.0
            self.display = 0.0
            return

        # Convert to binary: positive values -> 1, non-positive -> 0
        a_binary = (a > 0).astype(np.int32)
        b_binary = (b > 0).astype(np.int32)

        # Calculate intersection and union
        intersection = np.sum(a_binary & b_binary)
        union = np.sum(a_binary | b_binary)

        # Calculate Jaccard Index
        jaccard = intersection / union if union > 0 else 0.0

        # Set outputs
        self.output_iou = float(jaccard)
        self.display = float(jaccard)
        self.info = f"Jaccard Index: {jaccard:.4f}\nIntersection: {intersection}\nUnion: {union}"


class Slider(Node):
    """Output a slider value in [0, 1]."""

    value = Range("Value", default=0.0, min_val=0.0, max_val=1.0, step=0.01)
    output = OutputPort("Output", float)

    def process(self):
        self.output = self.value


class SliderLog(Node):
    """Output a log-scaled slider value in [0.0001, 1]."""

    value = Range(
        "Value",
        default=0.01,
        min_val=0.0001,
        max_val=1.0,
        step=0.0001,
        scale="log",
    )
    output = OutputPort("Output", float)

    def process(self):
        self.output = self.value


class Float(Node):
    """Output a float constant."""

    value = FloatProp("Value", default=0.0)
    output = OutputPort("Output", float)

    def process(self):
        self.output = self.value


class RangeMap(Node):
    """Map a 0..1 input to a [min, max] range."""

    input_value = InputPort("Input", float)
    min_value = FloatProp("Min", default=0.0)
    max_value = FloatProp("Max", default=1.0)
    output = OutputPort("Output", float)

    def process(self):
        if self.input_value is None:
            return
        self.output = self.min_value + (self.max_value - self.min_value) * self.input_value


class Softmax(Node):
    """Apply softmax with temperature to input."""

    input_data = InputPort("Input", np.ndarray)
    temperature = InputPort("Temperature", float)
    output = OutputPort("Output", np.ndarray)

    def process(self):
        if self.input_data is None:
            return
        temp = 1.0 if self.temperature is None else self.temperature
        x = self.input_data / temp
        x = x - np.max(x)
        exp = np.exp(x)
        self.output = exp / np.sum(exp)


class FloatMath(Node):
    """Apply a selected math operation to two floats."""

    a = InputPort("A", [float, int])
    b = InputPort("B", [float, int])
    output = OutputPort("Output", float)

    op = Enum(
        "Operation",
        options=["add", "sub", "mul", "div", "min", "max", "pow"],
        default="add",
    )

    def process(self):
        if self.a is None or self.b is None:
            return
        if self.op == "add":
            self.output = self.a + self.b
        elif self.op == "sub":
            self.output = self.a - self.b
        elif self.op == "mul":
            self.output = self.a * self.b
        elif self.op == "div":
            self.output = self.a / (self.b + 1e-8)
        elif self.op == "min":
            self.output = self.a if self.a < self.b else self.b
        elif self.op == "max":
            self.output = self.a if self.a > self.b else self.b
        elif self.op == "pow":
            self.output = self.a ** self.b


class RandomProjectionSignature(Node):
    """Project a normalized array onto a stored random reference vector."""

    input_data = InputPort("Input", np.ndarray)
    output = OutputPort("Output", float)
    reference_vector = OutputPort("Reference Vector", np.ndarray)

    set_reference = Action("Set Reference", lambda self, params=None: self._on_set_reference(params))

    display = Numeric("Signature", format=".4f")
    info = Text("Info", default="Press Set Reference")

    reference_length = State(default=0)
    stored_reference = State(default=None)

    def _normalize_vector(self, array):
        vec = np.asarray(array).flatten().astype(np.float64)
        norm = np.linalg.norm(vec)
        if norm <= 1e-12:
            return None
        return vec / norm

    def _on_set_reference(self, params=None):
        if self.input_data is None:
            self.info = "No input to capture"
            return {"status": "error", "message": "No input to capture"}

        ref = self._normalize_vector(self.input_data)
        if ref is None:
            self.info = "Reference input is zero"
            return {"status": "error", "message": "Reference input is zero"}

        rng = np.random.RandomState()
        random_vec = rng.randn(ref.size).astype(np.float64)
        random_norm = np.linalg.norm(random_vec)
        if random_norm <= 1e-12:
            random_vec = np.ones(ref.size, dtype=np.float64)
            random_norm = np.linalg.norm(random_vec)
        random_vec = random_vec / random_norm

        self.reference_length = int(ref.size)
        self.stored_reference = random_vec.astype(np.float32)
        self.reference_vector = self.stored_reference
        self.info = f"Reference set | Length: {ref.size}"
        return {"status": "ok", "message": f"Reference set with length {ref.size}"}

    def process(self):
        if self.stored_reference is None:
            self.info = "Press Set Reference"
            return

        self.reference_vector = np.asarray(self.stored_reference, dtype=np.float32)

        if self.input_data is None:
            self.info = f"Waiting for input | Ref length: {int(self.reference_length)}"
            return

        vec = self._normalize_vector(self.input_data)
        if vec is None:
            self.output = 0.0
            self.display = 0.0
            self.info = "Input is zero vector"
            return

        ref = np.asarray(self.stored_reference, dtype=np.float64)
        if vec.size != ref.size:
            self.output = 0.0
            self.display = 0.0
            self.info = f"Size mismatch: {vec.size} vs {ref.size}"
            return

        score = float(np.dot(vec, ref))
        self.output = score
        self.display = score
        self.info = f"Signature: {score:.4f} | Length: {vec.size}"


class RandomProjection2D(Node):
    """Project a normalized array onto two random reference vectors to get a 2D point."""

    input_data = InputPort("Input", np.ndarray)
    output = OutputPort("Output", np.ndarray)

    set_reference = Action("Set Reference", lambda self, params=None: self._on_set_reference(params))

    info = Text("Info", default="Press Set Reference")

    stored_ref_a = State(default=None)
    stored_ref_b = State(default=None)
    reference_length = State(default=0)

    def _normalize_vector(self, array):
        vec = np.asarray(array).flatten().astype(np.float64)
        norm = np.linalg.norm(vec)
        if norm <= 1e-12:
            return None
        return vec / norm

    def _make_random_unit(self, size):
        rng = np.random.RandomState()
        vec = rng.randn(size).astype(np.float64)
        norm = np.linalg.norm(vec)
        if norm <= 1e-12:
            vec = np.ones(size, dtype=np.float64)
            norm = np.linalg.norm(vec)
        return (vec / norm).astype(np.float32)

    def _on_set_reference(self, params=None):
        if self.input_data is None:
            self.info = "No input to capture"
            return {"status": "error", "message": "No input to capture"}

        ref = self._normalize_vector(self.input_data)
        if ref is None:
            self.info = "Reference input is zero"
            return {"status": "error", "message": "Reference input is zero"}

        size = ref.size
        self.stored_ref_a = self._make_random_unit(size)
        self.stored_ref_b = self._make_random_unit(size)
        self.reference_length = int(size)
        self.info = f"References set | Length: {size}"
        return {"status": "ok", "message": f"References set with length {size}"}

    def process(self):
        if self.stored_ref_a is None:
            self.info = "Press Set Reference"
            return

        if self.input_data is None:
            self.info = f"Waiting for input | Ref length: {int(self.reference_length)}"
            return

        vec = self._normalize_vector(self.input_data)
        if vec is None:
            self.output = np.zeros(2, dtype=np.float32)
            self.info = "Input is zero vector"
            return

        ref_a = np.asarray(self.stored_ref_a, dtype=np.float64)
        ref_b = np.asarray(self.stored_ref_b, dtype=np.float64)
        if vec.size != ref_a.size:
            self.output = np.zeros(2, dtype=np.float32)
            self.info = f"Size mismatch: {vec.size} vs {ref_a.size}"
            return

        x = float(np.dot(vec, ref_a))
        y = float(np.dot(vec, ref_b))
        self.output = np.array([x, y], dtype=np.float32)
        self.info = f"x: {x:.4f}  y: {y:.4f}"
