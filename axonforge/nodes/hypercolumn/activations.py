"""Activation nodes for MiniCortex."""

import numpy as np

from axonforge.core.node import Node
from axonforge.core.descriptors.ports import InputPort, OutputPort
from axonforge.core.descriptors.properties import Range
from axonforge.core.descriptors import branch


@branch("Hypercolumn/Activation")
class LateralInhibition(Node):
    """
    Applies lateral inhibition to pre-computed activations.
    
    Takes raw or rectified activations (e.g., from GeodesicIMDVv1), computes the
    total activity A = Σ a, and applies inhibition to each neuron based on the
    activity of all other neurons:
    
        a'_i = max(a_i - β * (A - a_i), 0)
    """

    activations = InputPort("Activations", np.ndarray)
    
    beta = Range("Beta", default=0.1, min_val=0.0, max_val=1.0, step=0.01)
    
    output = OutputPort("Inhibited Activations", np.ndarray)
    inhibition = OutputPort("Inhibition", np.ndarray)

    def process(self):
        if self.activations is None:
            return

        a = np.asarray(self.activations, dtype=np.float64).ravel()
        
        # Ensure activations are non-negative for the inhibition calculation
        a_rect = np.maximum(a, 0.0)
        A = np.sum(a_rect)
        
        beta_val = float(self.beta)
        
        inh = beta_val * (A - a_rect)
        self.inhibition = inh
        self.output = np.maximum(a - inh, 0.0)


@branch("Hypercolumn/Activation")
class Hyperpolarization(Node):
    """
    Lateral inhibition between active minicolumns.

    Compute raw activations s = W @ x̂, then keep only active neurons for
    inhibition by rectifying s to a = max(s, 0). The total active drive is
    A = Σ a. Each neuron is inhibited by the activity of all other active
    neurons:

        s'_i = max(s_i - β * (A - a_i), 0)

    This makes only active neurons contribute to inhibition while preserving
    rectified outputs.
    """

    input_data = InputPort("Input", np.ndarray)
    weights = InputPort("Weights", np.ndarray)
    beta = InputPort("Beta", float)
    output = OutputPort("Output", np.ndarray)
    inhibition = OutputPort("Inhibition", np.ndarray)

    def process(self):
        if self.input_data is None or self.weights is None:
            return

        x = self.input_data.ravel().astype(np.float64)
        w = self.weights.astype(np.float64)
        norm = np.linalg.norm(x) + 1e-8
        x_norm = x / norm
        s = w @ x_norm

        a = np.maximum(s, 0.0)
        A = np.sum(a)
        beta = 0.0 if self.beta is None else self.beta
        
        inh = beta * (A - a)
        self.inhibition = inh
        self.output = np.maximum(s - inh, 0.0)


@branch("Hypercolumn/Activation")
class HyperpolarizationV2(Node):
    """
    Lateral inhibition between active minicolumns.

    Compute raw activations s = W @ x̂, then keep only active neurons for
    inhibition by rectifying s to a = max(s, 0). The total active drive is
    A = Σ a. Each neuron is inhibited by the activity of all other active
    neurons:

        s'_i = max(s_i - β * (A - a_i), 0)

    This makes only active neurons contribute to inhibition while preserving
    rectified outputs.
    """

    input_data = InputPort("Input", np.ndarray)
    weights = InputPort("Weights", np.ndarray)
    beta = InputPort("Beta", float)
    output = OutputPort("Output", np.ndarray)
    inhibition = OutputPort("Inhibition", np.ndarray)

    def process(self):
        if self.input_data is None or self.weights is None:
            return

        x = self.input_data.ravel().astype(np.float64)
        w = self.weights.astype(np.float64)
        norm = np.linalg.norm(x) + 1e-8
        x_norm = x / norm
        s = w @ x_norm

        a = np.maximum(s, 0.0)
        A = np.sum(a)
        beta = 0.0 if self.beta is None else self.beta
        
        inh = beta * (A - a)
        self.inhibition = inh
        self.output = np.maximum(s - inh, 0.0)


@branch("Hypercolumn/Activation")
class Shunting(Node):
    """
    Shunting (divisive) inhibition.

    Instead of subtracting activity from other neurons, shunting inhibition
    scales each neuron's activation based on the total activity of the
    population. Higher overall activity increases inhibition and reduces
    the gain of all neurons.

    Let s be the activations and a = max(s, 0). The inhibitory signal is

        I = β * Σ_j a_j

    and each neuron is scaled by

        s'_i = s_i / (1 + I)

    This implements divisive normalization: neurons keep their relative
    ordering while the overall activity is reduced as population activity
    grows.
    """

    input_data = InputPort("Input", np.ndarray)
    weights = InputPort("Weights", np.ndarray)
    beta = InputPort("Beta", float)
    output = OutputPort("Output", np.ndarray)

    def process(self):
        if self.input_data is None or self.weights is None:
            return

        x = self.input_data.ravel()
        w = self.weights
        norm = np.linalg.norm(x) + 1e-8
        x_norm = x / norm
        s = w @ x_norm
        a = np.maximum(s, 0.0)
        beta = 0.0 if self.beta is None else self.beta
        I = beta * np.sum(a)
        self.output = np.maximum(s / (1.0 + I), 0.0)


@branch("Hypercolumn/Activation")
class ShuntingV2(Node):
    """
    Shunting (divisive) inhibition with unit-normalized weights.

    Divisive normalization ensures that winners always provide a learning signal,
    preventing 'Mass Extinction' in homeostatic loops.
    """

    input_data = InputPort("Input", np.ndarray)
    weights = InputPort("Weights", np.ndarray)
    beta = InputPort("Beta", float)
    output = OutputPort("Output", np.ndarray)
    inhibition = OutputPort("Inhibition", np.ndarray)

    def process(self):
        if self.input_data is None or self.weights is None:
            return

        # 1. Normalize input x and weights w_i to unit length using np.float64.
        x = self.input_data.ravel().astype(np.float64)
        w = self.weights.astype(np.float64)

        x_norm = x / (np.linalg.norm(x) + 1e-8)
        w_norms = np.linalg.norm(w, axis=1, keepdims=True) + 1e-8
        w_unit = w / w_norms

        # 2. Compute raw similarities: s = W · x̂.
        s = w_unit @ x_norm

        # 3. Rectify: a = max(s, 0).
        a = np.maximum(s, 0.0)

        # 4. Compute total inhibitory signal: I = β · Σ a_j.
        beta = 1.0 if self.beta is None else self.beta
        I = beta * np.sum(a)

        # 5. Compute inhibited output: s'_i = a_i / (1 + I).
        # 6. Output s'_i to output and I to inhibition.
        self.output = a / (1.0 + I)

        # The inhibition output should be a vector of the same shape as output
        # (all elements equal to I) to maintain compatibility.
        self.inhibition = np.full_like(self.output, I)


@branch("Hypercolumn/Activation")
class SoftmaxActivation(Node) :
    input_data = InputPort("Input", np.ndarray)
    weights = InputPort("Weights", np.ndarray)
    temperature = InputPort("Temperature", float)
    output = OutputPort("Output", np.ndarray)

    def process(self):
        if self.input_data is None or self.weights is None:
            return

        x = self.input_data.ravel()
        w = self.weights
        norm = np.linalg.norm(x) + 1e-8
        x_norm = x / norm
        s = w @ x_norm
        temp = 1.0 if self.temperature is None else self.temperature
        s = s / (temp + 1e-8)
        s = s - np.max(s)
        exp = np.exp(s)
        self.output = exp / np.sum(exp)
