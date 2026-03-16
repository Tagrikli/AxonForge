import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path
import os

from ....core.node import Node
from ....core.descriptors.ports import InputPort, OutputPort
from ....core.descriptors.fields import Range, Integer
from ....core.descriptors.displays import Heatmap, Plot, Text, Numeric
from ....core.descriptors.actions import Action
from ....core.descriptors.state import State
from ....core.descriptors import branch



@branch("Utilities/Encoding")
class IntegerFeedbackEncoder(Node):
    """Encode an integer into a sparse feedback vector.
    
    Maps an integer (e.g., a digit 0-9) to a vector of size N, where a configurable
    number of templates are set to 1.0 and the rest to 0.0. The active templates
    are assigned using a sliding window approach, allowing for overlap if the
    number of templates per digit is large relative to the vector size.
    """

    i_value = InputPort("Value", int)
    o_output = OutputPort("Output", np.ndarray)

    vector_size = Integer("Vector Size", default=20)
    templates_per_value = Integer("Templates per Value", default=2)
    max_value = Integer("Max Value", default=9)
    contrast = Range("Contrast", default=1.0, min_val=1.0, max_val=2.0, step=0.01)

    plot = Plot("Vector", style="bar", color_positive="#d6e6ff", color_negative="#d6e6ff")
    info = Text("Info", default="No input")

    def process(self):
        if self.i_value is None:
            return

        val = int(self.i_value)
        size = int(self.vector_size)
        t_per_val = int(self.templates_per_value)
        max_val = int(self.max_value)
        c_val = float(self.contrast)

        # Clamp value
        val = max(0, min(max_val, val))

        # Initialize with the suppressed value
        suppressed_val = 2.0 - c_val
        output = np.full(size, suppressed_val, dtype=np.float32)

        if size > 0 and t_per_val > 0:
            num_values = max_val + 1
            stride = size / float(num_values)
            
            start_index = int(round(val * stride))
            
            for i in range(t_per_val):
                idx = (start_index + i) % size
                output[idx] = c_val

        self.o_output = output
        self.plot = output
        self.info = f"Value: {val}, Active: {t_per_val}/{size}, Contrast: {c_val:.2f}"


class OneHotEncode(Node):

    i_index = InputPort("Category Index", int)

    o_encoding = OutputPort("Encoding", np.ndarray)

    input_length = Integer("Length", 10)
    plot = Plot("Vector", style="bar", color_positive="#d6e6ff", color_negative="#d6e6ff")
    value = Text("Value")

    def init(self):
        return super().init()

    def process(self):

        self.o_encoding = np.zeros((self.input_length))
        self.o_encoding[self.i_index] = 1

        self.plot = self.o_encoding
        self.value = self.i_index




class SparseIndexHashEncode(Node):
    """Hash an integer value to a binary vector with given sparsity.

    Uses deterministic hashing so the same input always produces the same output.
    The output vector has approximately (sparsity)% of its values set to 1.
    Outputs sparsity, vector_size, and max_value for connecting to decoder.
    """

    i_value = InputPort("Value", int)
    o_output = OutputPort("Output", np.ndarray)

    # Output ports for decoder connection
    o_sparsity = OutputPort("Sparsity", float)
    o_vector_size = OutputPort("Vector Size", int)
    o_max_value = OutputPort("Max Value", int)

    # Sparsity as a percentage (0-100)
    sparsity = Range("Sparsity", 50.0, 0.0, 100.0, step=1, scale="linear")

    # Vector size (length of the output binary vector)
    vector_size = Integer("Vector Size", default=1000)

    # Max value (for decoder mapping range)
    max_value = Integer("Max Value", default=100)

    # Display the output vector
    plot = Plot("Vector", style="bar", color_positive="#d6e6ff", color_negative="#d6e6ff")

    # Info text
    info = Text("Info", default="No input")

    def process(self):
        if self.i_value is None:
            return

        # Get parameters
        sparsity_pct = float(self.sparsity)
        size = int(self.vector_size)
        max_val = int(self.max_value)
        input_val = int(self.i_value)

        # Calculate number of ones based on sparsity
        # Sparsity 0% = all ones, Sparsity 100% = all zeros
        num_ones = int(size * (100.0 - sparsity_pct) / 100.0)
        num_ones = max(0, min(size, num_ones))

        # Create deterministic hash from input value
        hash_val = hash(str(input_val))

        # Use the hash to create a seed for random number generator
        rng = np.random.RandomState(hash_val % (2**31))

        # Create binary vector with random positions for ones
        output = np.zeros(size, dtype=np.float32)

        if num_ones > 0:
            indices = rng.choice(size, size=num_ones, replace=False)
            output[indices] = 1.0

        # Set outputs
        self.o_output = output
        self.o_sparsity = sparsity_pct
        self.o_vector_size = size
        self.o_max_value = max_val
        self.plot = output
        self.info = f"Input: {input_val}, Size: {size}, Ones: {num_ones}, Sparsity: {sparsity_pct:.1f}%"


class SparseIndexHashDecode(Node):
    """Reverse mapping node that takes a binary vector and outputs the corresponding integer.

    Uses precomputed mapping to decode sparse binary vectors back to original integer values.
    Receives sparsity, vector_size, and max_value from encoder via input ports.
    """

    i_array = InputPort("Array", np.ndarray)
    i_sparsity = InputPort("Sparsity", float)
    i_vector_size = InputPort("Vector Size", int)
    i_max_value = InputPort("Max Value", int)

    # Precompute button
    precompute = Action("Precompute", lambda self, params=None: self._on_precompute(params))

    # Info text
    info = Text("Info", default="Press Precompute")

    # State for the mapping dictionary
    mapping = State(default=None)

    # State received parameters
    _sparsity = State(default=None)
    _vector_size = State(default=None)
    _max_value = State(default=None)

    def _on_precompute(self, params: dict):
        """Create mapping from integers to binary vectors."""
        # Use received parameters or fall back to defaults
        sparsity_pct = float(self._sparsity) if self._sparsity is not None else 50.0
        size = int(self._vector_size) if self._vector_size is not None else 1000
        max_val = int(self._max_value) if self._max_value is not None else 100

        # Calculate number of ones based on sparsity
        num_ones = int(size * (100.0 - sparsity_pct) / 100.0)
        num_ones = max(0, min(size, num_ones))

        # Create mapping dictionary
        self.mapping = {}

        for i in range(max_val + 1):
            # Create deterministic hash
            hash_val = hash(str(i))
            rng = np.random.RandomState(hash_val % (2**31))

            # Create binary vector
            output = np.zeros(size, dtype=np.float32)
            if num_ones > 0:
                indices = rng.choice(size, size=num_ones, replace=False)
                output[indices] = 1.0

            # Use tuple as key for dictionary lookup
            self.mapping[tuple(output)] = i

        self.info = f"Mapped {max_val + 1} values"
        return {"status": "ok", "message": f"Mapped {max_val + 1} values"}

    def process(self):
        # State received parameters from encoder
        if self.i_sparsity is not None:
            self._sparsity = self.i_sparsity
        if self.i_vector_size is not None:
            self._vector_size = self.i_vector_size
        if self.i_max_value is not None:
            self._max_value = self.i_max_value

        if self.i_array is None:
            return

        if self.mapping is None:
            self.info = "Press Precompute"
            return

        # Flatten input and convert to tuple for lookup
        arr = self.i_array.flatten()
        arr_tuple = tuple(arr)

        # Look up in mapping
        if arr_tuple in self.mapping:
            self.info = f"Decoded: {self.mapping[arr_tuple]}"
        else:
            self.info = "No match found"


class UnitVectorHashEncode(Node):
    """Hash an integer value to a consistent random unit vector.

    Uses deterministic hashing so the same input always produces the same output.
    The output vector has unit length (L2 norm = 1) and is randomly distributed
    on the unit hypersphere. Outputs vector_size and max_value for connecting
    to decoder.
    """

    i_value = InputPort("Value", int)
    o_output = OutputPort("Output", np.ndarray)

    # Output ports for decoder connection
    o_vector_size = OutputPort("Vector Size", int)
    o_max_value = OutputPort("Max Value", int)

    # Vector size (dimension of the output unit vector)
    vector_size = Integer("Vector Size", default=1000)

    # Max value (for decoder mapping range)
    max_value = Integer("Max Value", default=100)

    # Display the output vector
    plot = Plot("Vector", style="bar", scale_mode="auto")

    # Info text
    info = Text("Info", default="No input")

    def process(self):
        if self.i_value is None:
            return

        # Get parameters
        size = int(self.vector_size)
        max_val = int(self.max_value)
        input_val = int(self.i_value)

        # Create deterministic hash from input value
        hash_val = hash(str(input_val))

        # Use the hash to create a seed for random number generator
        rng = np.random.RandomState(hash_val % (2**31))

        # Generate random vector with normal distribution
        output = rng.randn(size).astype(np.float32)

        # Normalize to unit length (L2 norm = 1)
        norm = np.linalg.norm(output)
        if norm > 0:
            output = output / norm

        # Set outputs
        self.o_output = output
        self.o_vector_size = size
        self.o_max_value = max_val
        self.plot = output
        self.info = f"Input: {input_val}, Size: {size}, Norm: {np.linalg.norm(output):.6f}"


class UnitVectorHashDecode(Node):
    """Reverse mapping node that takes a unit vector and outputs the corresponding integer.

    Uses precomputed mapping to decode unit vectors back to original integer values.
    Receives vector_size and max_value from encoder via input ports.
    Finds the closest matching vector using cosine similarity (dot product).
    """

    i_vector = InputPort("Vector", np.ndarray)
    i_vector_size = InputPort("Vector Size", int)
    i_max_value = InputPort("Max Value", int)

    # Output port for decoded value
    o_decoded = OutputPort("Decoded Value", int)

    # Precompute button
    precompute = Action("Precompute", lambda self, params=None: self._on_precompute(params))

    # Info text
    info = Text("Info", default="Press Precompute")

    # State for the mapping dictionary
    mapping = State(default=None)

    # State received parameters
    _vector_size = State(default=None)
    _max_value = State(default=None)

    # State last decoded value
    _last_decoded = State(default=None)

    def _on_precompute(self, params: dict):
        """Create mapping from integers to unit vectors."""
        # Use received parameters or fall back to defaults
        size = int(self._vector_size) if self._vector_size is not None else 1000
        max_val = int(self._max_value) if self._max_value is not None else 100

        # Create mapping dictionary (key: tuple of vector, value: integer)
        self.mapping = {}

        for i in range(max_val + 1):
            # Create deterministic hash
            hash_val = hash(str(i))
            rng = np.random.RandomState(hash_val % (2**31))

            # Generate random unit vector
            output = rng.randn(size).astype(np.float32)
            norm = np.linalg.norm(output)
            if norm > 0:
                output = output / norm

            # Use tuple as key for dictionary lookup
            self.mapping[tuple(output)] = i

        self.info = f"Mapped {max_val + 1} values"
        return {"status": "ok", "message": f"Mapped {max_val + 1} values"}

    def process(self):
        # State received parameters from encoder
        if self.i_vector_size is not None:
            self._vector_size = self.i_vector_size
        if self.i_max_value is not None:
            self._max_value = self.i_max_value

        if self.i_vector is None:
            return

        if self.mapping is None:
            self.info = "Press Precompute"
            return

        # Flatten input and normalize to unit length
        vec = self.i_vector.flatten().astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        # Find closest match using dot product (cosine similarity for unit vectors)
        input_tuple = tuple(vec)

        decoded_value = None

        # Check for exact match first
        if input_tuple in self.mapping:
            decoded_value = self.mapping[input_tuple]
            self.info = f"Decoded: {decoded_value}"
        else:
            # Otherwise find closest by cosine similarity
            best_match = None
            best_similarity = -1.0

            for stored_vec_tuple, value in self.mapping.items():
                stored_vec = np.array(stored_vec_tuple, dtype=np.float32)
                similarity = np.dot(vec, stored_vec)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = value

            if best_match is not None:
                decoded_value = best_match
                self.info = f"Decoded: {best_match} (sim: {best_similarity:.4f})"
            else:
                self.info = "No match found"

        # Set output port
        self.o_decoded = decoded_value


class AccuracyTracker(Node):
    """Track accuracy between predicted and real class labels over time.

    Maintains a rolling window of correct/incorrect predictions and displays
    accumulated average accuracy values in a line chart.
    """

    i_predicted = InputPort("Predicted Class", int)
    i_real = InputPort("Real Class", int)

    # Rolling window size for accuracy calculation (e.g., 100 = accuracy over last 100 inputs)
    window_size = Integer("Window Size", default=100)

    # Number of average accuracy points to display in the chart
    history_size = Integer("History Size", default=50)

    # Line chart to display accumulated average accuracies
    plot = Plot(
        "Accuracy History",
        style="line",
        color_positive="#00f5ff",
        color_negative="#00f5ff",
        line_width=1.5,
        scale_mode="auto",
    )

    # Info display
    info = Text("Info", default="No data")

    # Reset button
    reset = Action("Reset", lambda self, params=None: self._on_reset(params))

    # State for prediction history (1 = correct, 0 = incorrect)
    _prediction_history = State(default=None)

    # State for accuracy history (list of average accuracy values)
    _accuracy_history = State(default=None)

    def init(self):
        """Initialize history stores."""
        self._prediction_history = []
        self._accuracy_history = []
        return super().init()

    def _on_reset(self, params: dict):
        """Reset all history stores."""
        self._prediction_history = []
        self._accuracy_history = []
        self.plot = np.array([], dtype=np.float32)
        self.info = "Reset - No data"
        return {"status": "ok", "message": "History reset"}

    def process(self):
        if self.i_predicted is None or self.i_real is None:
            return

        # Get parameters
        window = int(self.window_size)
        history = int(self.history_size)

        predicted = int(self.i_predicted)
        real = int(self.i_real)

        # Check if prediction is correct
        is_correct = 1 if predicted == real else 0

        # Add to prediction history
        pred_history = list(self._prediction_history) if self._prediction_history is not None else []
        pred_history.append(is_correct)

        # Keep only the last 'window' predictions
        if len(pred_history) > window:
            pred_history = pred_history[-window:]

        self._prediction_history = pred_history

        # Calculate current accuracy over the window
        if len(pred_history) > 0:
            current_accuracy = sum(pred_history) / len(pred_history)
        else:
            current_accuracy = 0.0

        # Add to accuracy history
        acc_history = list(self._accuracy_history) if self._accuracy_history is not None else []
        acc_history.append(current_accuracy)

        # Keep only the last 'history' accuracy points for display
        if len(acc_history) > history:
            acc_history = acc_history[-history:]

        self._accuracy_history = acc_history

        # Update line chart with accuracy history
        self.plot = np.array(acc_history, dtype=np.float32)

        # Update info
        total_samples = len(pred_history)
        total_accuracy_history = len(acc_history)
        self.info = f"Window: {total_samples}/{window}, History: {total_accuracy_history}, Current Acc: {current_accuracy:.2%}"


class CosineTracker(Node):
    """Track cosine similarity between two vectors over time.

    Calculates cosine similarity between two input arrays and displays
    the similarity values in a line chart.
    """

    i_vector_a = InputPort("Vector A", np.ndarray)
    i_vector_b = InputPort("Vector B", np.ndarray)

    # Number of similarity values to display in the chart
    history_size = Integer("History Size", default=100)

    # Line chart to display cosine similarity history
    plot = Plot(
        "Cosine Similarity",
        style="line",
        color_positive="#00f5ff",
        color_negative="#00f5ff",
        line_width=1.5,
        scale_mode="auto",
    )

    # Info display
    info = Text("Info", default="No data")

    # Reset button
    reset = Action("Reset", lambda self, params=None: self._on_reset(params))

    # State for similarity history
    _similarity_history = State(default=None)

    def init(self):
        """Initialize history store."""
        self._similarity_history = []
        return super().init()

    def _on_reset(self, params: dict):
        """Reset similarity history."""
        self._similarity_history = []
        self.plot = np.array([], dtype=np.float32)
        self.info = "Reset - No data"
        return {"status": "ok", "message": "History reset"}

    def process(self):
        if self.i_vector_a is None or self.i_vector_b is None:
            return

        # Get parameters
        history = int(self.history_size)

        # Flatten and convert to float32
        vec_a = self.i_vector_a.flatten().astype(np.float32)
        vec_b = self.i_vector_b.flatten().astype(np.float32)

        # Calculate norms
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        # Calculate cosine similarity
        if norm_a > 0 and norm_b > 0:
            similarity = np.dot(vec_a, vec_b) / (norm_a * norm_b)
        else:
            similarity = 0.0

        # Add to similarity history
        sim_history = list(self._similarity_history) if self._similarity_history is not None else []
        sim_history.append(similarity)

        # Keep only the last 'history' values
        if len(sim_history) > history:
            sim_history = sim_history[-history:]

        self._similarity_history = sim_history

        # Update line chart with similarity history
        self.plot = np.array(sim_history, dtype=np.float32)

        # Update info
        total_history = len(sim_history)
        self.info = f"History: {total_history}/{history}, Current Sim: {similarity:.4f}"
