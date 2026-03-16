"""Vector operations for MiniCortex."""

import numpy as np

from axonforge.core.node import Node
from axonforge.core.descriptors.ports import InputPort, OutputPort
from axonforge.core.descriptors.displays import Text, Numeric
from axonforge.core.descriptors.fields import Range, Float
from axonforge.core.descriptors import branch


class L2Normalize(Node):
    """L2 normalize a numpy array. Flattens if required, then reshapes to original shape."""

    input_data = InputPort("Input", np.ndarray)
    output_data = OutputPort("Output", np.ndarray)
    
    norm_value = Numeric("L2 Norm", format=".4f")
    info = Text("Info", default="No input")

    def process(self):
        if self.input_data is None:
            return

        # State original shape
        original_shape = self.input_data.shape

        # Flatten the array for L2 normalization
        flat = self.input_data.flatten().astype(np.float64)

        # Compute L2 norm
        l2_norm = np.linalg.norm(flat)

        # Normalize (avoid division by zero)
        if l2_norm > 0:
            normalized = flat / l2_norm
        else:
            normalized = flat

        # Set outputs
        self.output_data = normalized.reshape(original_shape).astype(np.float32)
        self.norm_value = float(l2_norm)
        self.info = f"Norm: {l2_norm:.4f}, Shape: {original_shape}"


class L1Normalize(Node):
    """L1 normalize a numpy array."""

    input_data = InputPort("Input", np.ndarray)
    output_data = OutputPort("Output", np.ndarray)
    
    norm_value = Numeric("L1 Norm", format=".4f")
    info = Text("Info", default="No input")

    def process(self):
        if self.input_data is None:
            return

        original_shape = self.input_data.shape
        flat = self.input_data.flatten().astype(np.float64)
        
        l1_norm = np.sum(np.abs(flat))
        
        if l1_norm > 0:
            normalized = flat / l1_norm
        else:
            normalized = flat
        
        self.output_data = normalized.reshape(original_shape).astype(np.float32)
        self.norm_value = float(l1_norm)
        self.info = f"Norm: {l1_norm:.4f}"


class MinMaxNormalize(Node):
    """Min-max normalize array to [0, 1] range."""

    input_data = InputPort("Input", np.ndarray)
    output_data = OutputPort("Output", np.ndarray)
    
    min_val = Numeric("Min", format=".4f")
    max_val = Numeric("Max", format=".4f")
    info = Text("Info", default="No input")

    def process(self):
        if self.input_data is None:
            return

        original_shape = self.input_data.shape
        flat = self.input_data.flatten().astype(np.float64)
        
        min_v = np.min(flat)
        max_v = np.max(flat)
        
        if max_v > min_v:
            normalized = (flat - min_v) / (max_v - min_v)
        else:
            normalized = flat
        
        self.output_data = normalized.reshape(original_shape).astype(np.float32)
        self.min_val = float(min_v)
        self.max_val = float(max_v)
        self.info = f"Range: [{min_v:.4f}, {max_v:.4f}]"


class MeanCenter(Node):
    """Mean center a numpy array (subtract the mean)."""

    input_data = InputPort("Input", np.ndarray)
    output_data = OutputPort("Output", np.ndarray)

    mean_value = Numeric("Mean", format=".4f")
    info = Text("Info", default="No input")

    def process(self):
        if self.input_data is None:
            self.info = "No input"
            return

        original_shape = self.input_data.shape
        flat = self.input_data.flatten().astype(np.float64)

        mean_v = np.mean(flat)
        centered = flat - mean_v

        self.output_data = centered.reshape(original_shape).astype(np.float32)
        self.mean_value = float(mean_v)
        self.info = f"Mean: {mean_v:.4f}, Shape: {original_shape}"


class VectorScale(Node):
    """Scale a vector by a scalar value."""

    input_vector = InputPort("Input", np.ndarray)
    output_vector = OutputPort("Output", np.ndarray)
    
    scalar = Float("Scalar", default=0.5)
    info = Text("Info", default="No input")

    def process(self):
        if self.input_vector is None:
            self.info = "No input"
            return

        # Scale the vector by the scalar value
        scaled = self.input_vector * self.scalar
        
        self.output_vector = scaled.astype(np.float32)
        self.info = f"Scalar: {self.scalar:.2f}, Shape: {scaled.shape}"


class VectorMagnitude(Node):
    """Compute the L2 magnitude/length of a vector."""

    input_data = InputPort("Input", np.ndarray)
    magnitude = OutputPort("Magnitude", float)
    
    display = Numeric("Magnitude", format=".4f")
    info = Text("Info", default="No input")

    def process(self):
        if self.input_data is None:
            return

        flat = self.input_data.flatten().astype(np.float64)
        mag = float(np.linalg.norm(flat))
        
        self.magnitude = mag
        self.display = mag
        self.info = f"Magnitude: {mag:.4f}"


class VectorDotProduct(Node):
    """Compute dot product of two vectors."""

    input_a = InputPort("A", np.ndarray)
    input_b = InputPort("B", np.ndarray)
    output = OutputPort("Dot Product", float)
    
    display = Numeric("Dot Product", format=".4f")
    info = Text("Info", default="No input")

    def process(self):
        if self.input_a is None or self.input_b is None:
            self.info = "Waiting for inputs"
            return

        a = self.input_a.flatten().astype(np.float64)
        b = self.input_b.flatten().astype(np.float64)
        
        if len(a) != len(b):
            self.info = f"Size mismatch: {len(a)} vs {len(b)}"
            self.output = 0.0
            return

        dot = float(np.dot(a, b))
        self.output = dot
        self.display = dot
        self.info = f"Dot: {dot:.4f}, Size: {len(a)}"


class VectorCrossProduct(Node):
    """Compute cross product of two 3D vectors."""

    input_a = InputPort("A", np.ndarray)
    input_b = InputPort("B", np.ndarray)
    output = OutputPort("Result", np.ndarray)
    
    info = Text("Info", default="No input")

    def process(self):
        if self.input_a is None or self.input_b is None:
            self.info = "Waiting for inputs"
            return

        a = self.input_a.flatten().astype(np.float64)
        b = self.input_b.flatten().astype(np.float64)
        
        if len(a) != 3 or len(b) != 3:
            self.info = "Requires 3D vectors"
            return

        cross = np.cross(a, b)
        self.output = cross.astype(np.float32)
        self.info = f"Cross product computed"


class VectorDistance(Node):
    """Compute Euclidean distance between two vectors."""

    input_a = InputPort("A", np.ndarray)
    input_b = InputPort("B", np.ndarray)
    distance = OutputPort("Distance", float)
    
    display = Numeric("Distance", format=".4f")
    info = Text("Info", default="No input")

    def process(self):
        if self.input_a is None or self.input_b is None:
            self.info = "Waiting for inputs"
            return

        a = self.input_a.flatten().astype(np.float64)
        b = self.input_b.flatten().astype(np.float64)
        
        if len(a) != len(b):
            self.info = f"Size mismatch"
            self.distance = 0.0
            return

        dist = float(np.linalg.norm(a - b))
        self.distance = dist
        self.display = dist
        self.info = f"Distance: {dist:.4f}"


class VectorAngle(Node):
    """Compute angle between two vectors in radians."""

    input_a = InputPort("A", np.ndarray)
    input_b = InputPort("B", np.ndarray)
    angle = OutputPort("Angle (rad)", float)
    
    display = Numeric("Angle", format=".4f")
    info = Text("Info", default="No input")

    def process(self):
        if self.input_a is None or self.input_b is None:
            self.info = "Waiting for inputs"
            return

        a = self.input_a.flatten().astype(np.float64)
        b = self.input_b.flatten().astype(np.float64)
        
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            self.info = "Zero vector"
            self.angle = 0.0
            return

        cos_angle = np.dot(a, b) / (norm_a * norm_b)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = float(np.arccos(cos_angle))
        
        self.angle = angle
        self.display = angle
        self.info = f"Angle: {angle:.4f} rad ({np.degrees(angle):.2f}°)"


class VectorCosineSimilarity(Node):
    """Compute cosine similarity between two vectors."""

    input_a = InputPort("A", np.ndarray)
    input_b = InputPort("B", np.ndarray)
    similarity = OutputPort("Cosine Similarity", float)
    
    display = Numeric("Similarity", format=".4f")
    info = Text("Info", default="No input")

    def process(self):
        if self.input_a is None or self.input_b is None:
            self.info = "Waiting for inputs"
            return

        a = self.input_a.flatten().astype(np.float64)
        b = self.input_b.flatten().astype(np.float64)
        
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            self.info = "Zero vector"
            self.similarity = 0.0
            return

        cos_sim = np.dot(a, b) / (norm_a * norm_b)
        self.similarity = float(cos_sim)
        self.display = float(cos_sim)
        self.info = f"Cosine Similarity: {cos_sim:.4f}"
