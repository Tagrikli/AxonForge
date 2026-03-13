"""Learning nodes for MiniCortex hypercolumn (V2)."""

import numpy as np

from axonforge.core.node import Node
from axonforge.core.descriptors.ports import InputPort, OutputPort
from axonforge.core.descriptors.properties import Range, Integer, Bool
from axonforge.core.descriptors.displays import Vector2D
from axonforge.core.descriptors.actions import Action
from axonforge.core.descriptors import branch
from axonforge.nodes.hypercolumn.utilities import to_display_grid

@branch("Hypercolumn/Learning")
class GeodesicIMDVv1(Node):
    """
    Inhibition-Mediated Data Vacuum (IMDV) learning algorithm.

    Self-Contained LOO (Leave-One-Out) Reconstruction learning on the unit
    hypersphere with dimension-invariant inhibition. Manages its own weights.

    Force Formulation:
        tau_pull,i = geodesic_tangent(w_i -> x_hat)
        tau_push,i = geodesic_tangent(w_i -> x_shadow_hat,i)
        tau_net,i = (1 - balance) * tau_pull,i - (balance * inh_i) * tau_push,i
        where inh_i = (A - a_i) / (A + eps)   (normalized to [0,1])

    Input Ports:
        inputs: Raw input pattern x (np.ndarray)
        feedback: Optional top-down plasticity modulator for pull tangent (np.ndarray)

    Properties:
        amount: Number of templates/minicolumns
        dim: Dimensionality of each template
        balance: E/I balance [0,1]. 0 = pure pull, 1 = pure push (default: 0.5)
        step_fraction: Learning speed [0,1] (default: 0.1)

    Output Ports:
        weights: Current weight matrix W (np.ndarray)
        activations: Internal activities a_i (np.ndarray)
    """
    inputs = InputPort("Input", np.ndarray)
    feedback = InputPort("Feedback", np.ndarray)

    amount = Integer("Amount", default=25)
    dim = Integer("Dim", default=784)
    balance = Range("Balance", default=0.5, min_val=0.0, max_val=1.0, step=0.01)
    step_fraction = Range("Step Fraction", default=0.1, min_val=0.0, max_val=1.0, step=0.01)
    is_learning = Bool("Is Learning", default=True)

    weights = OutputPort("Weights", np.ndarray)
    activations = OutputPort("Activations", np.ndarray)

    weights_preview = Vector2D("Weights", color_mode="bwr")
    activations_preview = Vector2D("Activations", color_mode="grayscale")

    reset = Action("Reset", "_on_reset")

    def init(self):
        self._on_reset()

    def _on_reset(self, params=None):
        cols = int(self.dim)
        rows = int(self.amount)
        w = np.random.randn(rows, cols)
        norms = np.linalg.norm(w, axis=1, keepdims=True) + 1e-8
        self._w = w / norms
        self.weights_preview = to_display_grid(self._w)
        return {"status": "ok"}

    def process(self):
        eps_stable = 1e-8

        if self.inputs is None:
            self.weights = self._w
            return

        balance = self.balance
        step_fraction = self.step_fraction

        # Use float64 for internal calculations
        w = np.asarray(self._w, dtype=np.float64)
        x = np.asarray(self.inputs, dtype=np.float64).ravel()

        n_templates, dim = w.shape

        # 1. Normalize Input: x_hat = x / (||x|| + 1e-8)
        x_norm = np.linalg.norm(x)
        if x_norm < eps_stable:
            self.weights = self._w
            return
        x_hat = x / (x_norm + eps_stable)

        # Ensure weights are unit vectors
        w_norms = np.linalg.norm(w, axis=1, keepdims=True)
        w = w / (w_norms + eps_stable)

        # 2. Compute Similarities & Activities: s = W @ x_hat, a = max(s, 0), A = sum(a)
        s = w @ x_hat
        a = np.maximum(s, 0.0)

        A = np.sum(a)

        # Feedback modulates tangent forces, not activations
        if self.feedback is not None:
            f = np.asarray(self.feedback, dtype=np.float64).ravel()
        else:
            f = None

        # 3. Compute Global Reconstruction: x_recon = a @ W
        x_recon = a @ w

        # 4. LOO Shadow: x_shadow,i = x_recon - a_i * w_i
        x_shadow = x_recon[None, :] - a[:, None] * w

        # Normalize Shadow: x_shadow_hat,i = x_shadow,i / (||x_shadow,i|| + 1e-8)
        x_shadow_norm = np.linalg.norm(x_shadow, axis=1, keepdims=True)
        x_shadow_hat = x_shadow / (x_shadow_norm + eps_stable)

        # Inhibition (normalized to [0,1] for dimension invariance): inh_i = (A - a_i) / (A + eps)
        inh = (A - a) / (A + eps_stable)

        # 5. Compute Tangents
        # tau_pull_hat
        cos_theta_pull = np.sum(w * x_hat[None, :], axis=1, keepdims=True)
        tangent_pull = x_hat[None, :] - cos_theta_pull * w
        norm_pull = np.linalg.norm(tangent_pull, axis=1, keepdims=True)
        tau_pull_hat = np.where(norm_pull < eps_stable, np.zeros_like(w), tangent_pull / norm_pull)

        # tau_push_hat
        cos_theta_push = np.sum(w * x_shadow_hat, axis=1, keepdims=True)
        tangent_push = x_shadow_hat - cos_theta_push * w
        norm_push = np.linalg.norm(tangent_push, axis=1, keepdims=True)
        tau_push_hat = np.where(norm_push < eps_stable, np.zeros_like(w), tangent_push / norm_push)

        # 6. Net Tangent (tau_net,i):
        inactive_mask = (a <= 0)
        zero_pull_mask = (norm_pull[:, 0] < eps_stable) & inactive_mask

        if np.any(zero_pull_mask):
            random_tangent = np.random.randn(np.sum(zero_pull_mask), dim)
            w_zero = w[zero_pull_mask]
            random_tangent -= np.sum(random_tangent * w_zero, axis=1, keepdims=True) * w_zero
            random_norm = np.linalg.norm(random_tangent, axis=1, keepdims=True)
            tau_pull_hat[zero_pull_mask] = random_tangent / (random_norm + eps_stable)

        # Scale pull tangent by feedback (top-down plasticity modulation)
        pull_weight = 1.0 - balance
        push_weight = balance

        pull_scale = pull_weight
        if f is not None:
            pull_scale = pull_weight * f[:, None]

        tau_net = np.where(
            a[:, None] > 0,
            pull_scale * tau_pull_hat - (push_weight * inh[:, None]) * tau_push_hat,
            pull_scale * tau_pull_hat
        )

        if self.is_learning:
            # 7. Torque approach: use ||tau_net|| directly as rotation angle
            tau_net_norm = np.linalg.norm(tau_net, axis=1, keepdims=True)
            rotation_axis = np.where(tau_net_norm < eps_stable, np.zeros_like(w), tau_net / tau_net_norm)

            actual_rotation = step_fraction * tau_net_norm

            # rodrigues_rotation
            cos_a = np.cos(actual_rotation)
            sin_a = np.sin(actual_rotation)
            w_new = w * cos_a + rotation_axis * sin_a

            # Renormalize: Ensure all w_i,new are unit vectors
            w_new_norms = np.linalg.norm(w_new, axis=1, keepdims=True)
            w_new = w_new / (w_new_norms + eps_stable)

            self._w = w_new

        # Store and output
        self.weights = self._w
        a_out = a * x_norm
        self.activations = a_out
        self.weights_preview = to_display_grid(self._w)
        self.activations_preview = to_display_grid(a_out)
