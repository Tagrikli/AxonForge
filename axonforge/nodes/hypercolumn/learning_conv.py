"""Convolutional IMDV learning node for MiniCortex hypercolumn."""

import numpy as np

from axonforge.core.node import Node
from axonforge.core.descriptors.ports import InputPort, OutputPort
from axonforge.core.descriptors.properties import Range, Integer, Bool
from axonforge.core.descriptors.displays import Vector2D
from axonforge.core.descriptors.actions import Action
from axonforge.core.descriptors import branch
from axonforge.nodes.hypercolumn.utilities import to_display_grid


@branch("Hypercolumn/Learning")
class GeodesicIMDVConv(Node):
    """
    Convolutional Geodesic IMDV learning.

    Applies the IMDV learning algorithm to non-overlapping patches of a 2D
    input with shared weights across all patch positions. Stride equals
    patch size (no overlap).

    The template count (amount) must be a perfect square so the per-patch
    activations can be arranged in a square grid, preserving topology in
    the output activation map.

    For a (H, W) input with patch_size P and amount K (sqrt(K) = S):
        - Spatial grid: (H // P) x (W // P) patches
        - Output activation map: (grid_h * S) x (grid_w * S)

    Input Ports:
        inputs: 2D input pattern (np.ndarray, shape H x W)
        feedback: Optional top-down plasticity modulator (np.ndarray)

    Properties:
        amount: Number of templates per patch (must be a perfect square)
        patch_size: Side length of each square patch
        balance: E/I balance [0,1]
        step_fraction: Learning speed [0,1]

    Output Ports:
        weights: Current weight matrix W (amount x patch_size^2)
        activations: 2D activation map
    """

    inputs = InputPort("Input", np.ndarray)
    feedback = InputPort("Feedback", np.ndarray)

    amount = Integer("Amount", default=9)
    patch_size = Integer("Patch Size", default=4)
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
        k = int(self.amount)
        dim = int(self.patch_size) ** 2
        w = np.random.randn(k, dim)
        norms = np.linalg.norm(w, axis=1, keepdims=True) + 1e-8
        self._w = w / norms
        self.weights_preview = to_display_grid(self._w)
        return {"status": "ok"}

    def _imdv_step(self, patches, w, balance, step_fraction, feedback_patches):
        """
        Run one IMDV learning step on a batch of patches with shared weights.

        Parameters:
            patches: (N, dim) array of normalized input patches
            w: (K, dim) weight matrix (shared, will be updated)
            balance: E/I balance scalar
            step_fraction: learning rate scalar
            feedback_patches: (N, K) feedback or None

        Returns:
            w_new: (K, dim) updated weight matrix
            all_activations: (N, K) activations for each patch
        """
        eps = 1e-8
        n_patches, dim = patches.shape
        k = w.shape[0]

        # Ensure weights are unit vectors
        w_norms = np.linalg.norm(w, axis=1, keepdims=True)
        w = w / (w_norms + eps)

        # Accumulate weight updates across all patches
        tau_accum = np.zeros_like(w)
        all_activations = np.zeros((n_patches, k))

        for p in range(n_patches):
            x = patches[p]

            # Normalize input patch
            x_norm = np.linalg.norm(x)
            if x_norm < eps:
                continue
            x_hat = x / (x_norm + eps)

            # Similarities and activities
            s = w @ x_hat
            a = np.maximum(s, 0.0)
            A = np.sum(a)

            all_activations[p] = a * x_norm

            # Global reconstruction and LOO shadow
            x_recon = a @ w
            x_shadow = x_recon[None, :] - a[:, None] * w

            x_shadow_norm = np.linalg.norm(x_shadow, axis=1, keepdims=True)
            x_shadow_hat = x_shadow / (x_shadow_norm + eps)

            # Inhibition
            inh = (A - a) / (A + eps)

            # Pull tangent
            cos_pull = np.sum(w * x_hat[None, :], axis=1, keepdims=True)
            tang_pull = x_hat[None, :] - cos_pull * w
            norm_pull = np.linalg.norm(tang_pull, axis=1, keepdims=True)
            tau_pull = np.where(norm_pull < eps, np.zeros_like(w), tang_pull / norm_pull)

            # Push tangent
            cos_push = np.sum(w * x_shadow_hat, axis=1, keepdims=True)
            tang_push = x_shadow_hat - cos_push * w
            norm_push = np.linalg.norm(tang_push, axis=1, keepdims=True)
            tau_push = np.where(norm_push < eps, np.zeros_like(w), tang_push / norm_push)

            # Random tangent for inactive units with zero pull
            inactive_mask = a <= 0
            zero_pull_mask = (norm_pull[:, 0] < eps) & inactive_mask
            if np.any(zero_pull_mask):
                n_zero = np.sum(zero_pull_mask)
                rand_tang = np.random.randn(n_zero, dim)
                w_zero = w[zero_pull_mask]
                rand_tang -= np.sum(rand_tang * w_zero, axis=1, keepdims=True) * w_zero
                rand_norm = np.linalg.norm(rand_tang, axis=1, keepdims=True)
                tau_pull[zero_pull_mask] = rand_tang / (rand_norm + eps)

            # Net tangent
            pull_weight = 1.0 - balance
            push_weight = balance

            pull_scale = pull_weight
            if feedback_patches is not None:
                f = feedback_patches[p]
                pull_scale = pull_weight * f[:, None]

            tau_net = np.where(
                a[:, None] > 0,
                pull_scale * tau_pull - (push_weight * inh[:, None]) * tau_push,
                pull_scale * tau_pull,
            )

            tau_accum += tau_net

        # Average tangent across patches
        tau_avg = tau_accum / n_patches

        # Rodrigues rotation with averaged tangent
        tau_norm = np.linalg.norm(tau_avg, axis=1, keepdims=True)
        rotation_axis = np.where(tau_norm < eps, np.zeros_like(w), tau_avg / tau_norm)

        angle = step_fraction * tau_norm
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        w_new = w * cos_a + rotation_axis * sin_a

        # Renormalize
        w_new_norms = np.linalg.norm(w_new, axis=1, keepdims=True)
        w_new = w_new / (w_new_norms + eps)

        return w_new, all_activations

    def process(self):
        if self.inputs is None:
            self.weights = self._w
            return

        inp = np.asarray(self.inputs, dtype=np.float64)

        # Ensure 2D
        if inp.ndim == 1:
            side = int(np.sqrt(inp.size))
            inp = inp.reshape(side, side)

        H, W = inp.shape
        P = int(self.patch_size)
        k = int(self.amount)

        # Validate perfect square for amount
        s = int(np.sqrt(k))
        assert s * s == k, f"amount must be a perfect square, got {k}"

        grid_h = H // P
        grid_w = W // P

        # Extract non-overlapping patches: (grid_h * grid_w, P*P)
        patches = inp[:grid_h * P, :grid_w * P].reshape(grid_h, P, grid_w, P)
        patches = patches.transpose(0, 2, 1, 3).reshape(grid_h * grid_w, P * P)

        # Handle feedback
        feedback_patches = None
        if self.feedback is not None:
            fb = np.asarray(self.feedback, dtype=np.float64)
            # Feedback should be shaped so each patch gets (K,) modulation
            # Expected shape: (grid_h * s, grid_w * s) matching output topology
            if fb.ndim == 2 and fb.shape == (grid_h * s, grid_w * s):
                fb_patches = fb.reshape(grid_h, s, grid_w, s)
                fb_patches = fb_patches.transpose(0, 2, 1, 3).reshape(grid_h * grid_w, k)
                feedback_patches = fb_patches

        w = np.asarray(self._w, dtype=np.float64)
        balance = self.balance
        step_fraction = self.step_fraction

        if self.is_learning:
            w_new, all_activations = self._imdv_step(
                patches, w, balance, step_fraction, feedback_patches
            )
            self._w = w_new
        else:
            _, all_activations = self._imdv_step(
                patches, w, balance, step_fraction, feedback_patches
            )

        # Assemble activation map: (grid_h * s, grid_w * s)
        # all_activations is (grid_h * grid_w, k)
        act_grid = all_activations.reshape(grid_h, grid_w, s, s)
        act_map = act_grid.transpose(0, 2, 1, 3).reshape(grid_h * s, grid_w * s)

        self.weights = self._w
        self.activations = act_map
        self.weights_preview = to_display_grid(self._w)
        self.activations_preview = act_map


@branch("Hypercolumn/Learning")
class ReconstructConv(Node):
    """
    Reconstruct a 2D input from convolutional IMDV activations and weights.

    Takes the activation map and shared weight matrix from GeodesicIMDVConv
    and reconstructs the original spatial input by computing a_i @ W for each
    patch position, then assembling the patches back into a 2D image.

    All shapes are inferred from the inputs — no manual configuration needed
    beyond patch_size.

    Input Ports:
        activations: 2D activation map (grid_h * S, grid_w * S)
        weights: Shared weight matrix (K, patch_size^2) where K = S*S

    Properties:
        patch_size: Side length of each square patch (must match the conv node)

    Output Ports:
        output: Reconstructed 2D image (grid_h * P, grid_w * P)
    """

    activations = InputPort("Activations", np.ndarray)
    weights = InputPort("Weights", np.ndarray)

    patch_size = Integer("Patch Size", default=4)

    output = OutputPort("Output", np.ndarray)

    output_preview = Vector2D("Reconstruction", color_mode="grayscale")

    def init(self):
        pass

    def process(self):
        if self.activations is None or self.weights is None:
            return

        act_map = np.asarray(self.activations, dtype=np.float64)
        w = np.asarray(self.weights, dtype=np.float64)

        P = int(self.patch_size)
        k, dim = w.shape

        s = int(np.sqrt(k))

        map_h, map_w = act_map.shape
        grid_h = map_h // s
        grid_w = map_w // s

        # Disassemble activation map into per-patch activations: (N, K)
        # act_map is (grid_h * s, grid_w * s)
        act_patches = (
            act_map.reshape(grid_h, s, grid_w, s)
            .transpose(0, 2, 1, 3)
            .reshape(grid_h * grid_w, k)
        )

        # Reconstruct each patch: (N, dim) = (N, K) @ (K, dim)
        recon_patches = act_patches @ w

        # Assemble patches back into 2D image: (grid_h * P, grid_w * P)
        recon = (
            recon_patches.reshape(grid_h, grid_w, P, P)
            .transpose(0, 2, 1, 3)
            .reshape(grid_h * P, grid_w * P)
        )

        self.output = recon
        self.output_preview = recon
