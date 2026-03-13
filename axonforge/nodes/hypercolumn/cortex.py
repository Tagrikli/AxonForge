import math
import numpy as np
from typing import Optional

from axonforge.core.node import Node
from axonforge.core.descriptors.ports import InputPort, OutputPort
from axonforge.core.descriptors.properties import Range, Integer, Bool
from axonforge.core.descriptors.displays import Vector2D, Text
from axonforge.core.descriptors.actions import Action
from axonforge.core.descriptors import branch
from axonforge.nodes.hypercolumn.utilities import to_display_grid

class Reconstruct(Node):
    """Reconstruct an input from weights and activations: x_reconstructed = W.T @ a.

    Takes the (n, D) weight matrix and the (k, k) activation map from a
    HyperColumn, flattens the activations to (n,), computes W.T @ a → (D,),
    and reshapes back to the original 2-D input shape for output and display.
    """

    # ── Ports ──────────────────────────────────────────────────────────
    weights = InputPort("Weights", np.ndarray)
    activations = InputPort("Activations", np.ndarray)
    output = OutputPort("Output", np.ndarray)

    # ── Displays ───────────────────────────────────────────────────────
    preview = Vector2D("Reconstruction", color_mode="grayscale")
    info = Text("Info", default="Waiting for inputs…")

    def process(self):
        if self.weights is None or self.activations is None:
            return

        W = self.weights  # (n, D)
        a = self.activations  # (k, k) or any shape with n total elements

        n, D = W.shape
        a_flat = a.flatten().astype(np.float64)  # (n,)

        x_recon = W.T @ a_flat

        side = int(np.sqrt(D))
        x_recon_2d = x_recon.reshape(side, side)

        self.output = x_recon_2d
        self.preview = x_recon_2d
        self.info = (
            f"W ({n}×{D}) · a ({n},) → ({x_recon_2d.shape[0]}×{x_recon_2d.shape[1]})"
        )


class Weights(Node):
    """Generate and hold a weight matrix of unit vectors."""

    amount = Integer("Amount", default=25)
    dim = Integer("Dim", default=784)

    update = InputPort("Update", np.ndarray)
    weights = OutputPort("Weights", np.ndarray)
    amount_out = OutputPort("Amount", int)

    preview = Vector2D("Weights", color_mode="bwr")

    reset = Action("Reset", "_on_reset")

    def init(self):
        self._on_reset()
        self.preview = to_display_grid(self._weights)

    def _on_reset(self, params=None):
        cols = int(self.dim)
        rows = int(self.amount)
        w = np.random.randn(rows, cols)
        norms = np.linalg.norm(w, axis=1, keepdims=True) + 1e-8
        self._weights = w / norms
        self.preview = to_display_grid(self._weights)
        self._ignore_update_once = True
        return {"status": "ok"}

    def process(self):
        if getattr(self, "_ignore_update_once", False):
            self._ignore_update_once = False
        elif self.update is not None:
            w = self.update
            norms = np.linalg.norm(w, axis=1, keepdims=True) + 1e-8
            self._weights = w / norms
        self.weights = self._weights
        self.amount_out = self.amount
        self.preview = to_display_grid(self._weights)
