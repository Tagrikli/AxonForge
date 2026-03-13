"""Cortical computation nodes for MiniCortex."""

import math
import numpy as np
from typing import Optional

from axonforge.core.node import Node
from axonforge.core.descriptors.ports import InputPort, OutputPort
from axonforge.core.descriptors.properties import Range, Integer, Bool
from axonforge.core.descriptors.displays import Vector2D, Text
from axonforge.core.descriptors.actions import Action
from axonforge.core.descriptors import branch
from axonforge.nodes.hypercolumn.utilities import to_display_grid, softmax


class HyperColumn(Node):

    # ── Ports ──────────────────────────────────────────────────────────
    i_input = InputPort("Input", np.ndarray)
    i_feedback = InputPort("Feedback", np.ndarray)
    o_output = OutputPort("Output", np.ndarray)
    o_weights = OutputPort("Weights", np.ndarray)
    o_recon = OutputPort("Recon", np.ndarray)

    # Props
    input_len = Integer("Input Size", default=28)
    minicolumn_count = Integer("Minicolumns", default=9)
    temperature = Range("temp", default=1, min_val=0.001, max_val=5, step=0.1)
    is_learning = Bool("Learning")
    a_reset_weights = Action("Reset", "_on_reset")

    # Displays
    d_s_raw = Vector2D("S Raw", color_mode="grayscale")

    def init(self):
        self.weights = np.random.randn(self.minicolumn_count, self.input_len**2)
        self._on_reset()

    def _on_reset(self, params=None):
        weights = np.random.randn(self.minicolumn_count, self.input_len**2)
        norms = np.linalg.norm(weights, axis=1, keepdims=True) + 1e-8
        self.weights = weights = weights / norms

        return {"status": "ok"}

    def process(self):
        if self.i_input is None:
            return

        x: np.ndarray = self.i_input.flatten()
        norm = np.linalg.norm(x) + 1e-8
        x_norm = x / norm

        s_raw = softmax(self.weights @ x_norm, temperature=self.temperature)
        recon = self.weights.T @ s_raw

        ### DISPLAY/OUTPUT UPDATE ###
        self.o_recon = to_display_grid(recon)
        self.d_s_raw = to_display_grid(s_raw)
