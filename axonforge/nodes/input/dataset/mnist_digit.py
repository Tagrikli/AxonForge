"""MNIST digit input node."""

import numpy as np

from ....core.descriptors.actions import Action
from ....core.descriptors.displays import Text, Heatmap
from ....core.descriptors.ports import OutputPort
from ....core.descriptors.fields import Integer, Range
from ....core.descriptors.state import State
from ....core.node import Node, background_init
from ....nodes.utilities.dataset_cache import load_dataset_with_python_mnist


class InputDigitMNIST(Node):
    """Cycle through MNIST digit images."""

    output_pattern = OutputPort("Pattern", np.ndarray)
    output_digit = OutputPort("Digit", int)
    pattern = Heatmap("Pattern", colormap="grayscale")
    digit = Text("Digit", default="0")

    digit_filter = Range("Digit Filter", default=-1, min_val=-1, max_val=9, step=1)
    repeat = Integer("Repeat", default=1)

    prev_button = Action("◀", lambda self, params=None: self.prev_digit(params))
    next_button = Action("▶", lambda self, params=None: self.next_digit(params))

    idx = State(default=0)
    repeat_counter = State(default=0)

    @background_init
    def init(self):
        self._images, self._labels = load_dataset_with_python_mnist("mnist")
        if self._images is None or self._labels is None:
            return

        # Shuffle the entire dataset so sequential access isn't in label order
        perm = np.random.permutation(len(self._images))
        self._images = self._images[perm]
        self._labels = self._labels[perm]

        # Build per-digit index lists (indices into the shuffled arrays)
        self._class_indices = {}
        for d in range(10):
            self._class_indices[d] = np.where(self._labels == d)[0]

        self._update_output()

    def process(self):
        if self._images is None:
            return

        rep = int(self.repeat)
        if rep <= 0:
            # Frozen — only Prev/Next buttons move the index
            self._update_output()
            return

        self.repeat_counter = int(self.repeat_counter) + 1
        if int(self.repeat_counter) >= rep:
            self.repeat_counter = 0
            self._advance(1)

        self._update_output()

    def next_digit(self, params=None):
        if self._images is not None:
            self._advance(1)
            self._update_output()

    def prev_digit(self, params=None):
        if self._images is not None:
            self._advance(-1)
            self._update_output()

    def _advance(self, direction):
        """Move index by `direction` (+1 or -1) within the current filter."""
        filt = int(self.digit_filter)
        if filt == -1:
            # All digits — walk through the shuffled dataset
            self.idx = (int(self.idx) + direction) % len(self._images)
        else:
            # Filtered to a specific digit class
            indices = self._class_indices.get(filt, [])
            if len(indices) == 0:
                return
            # Find where our current idx sits in the class list
            pos = np.searchsorted(indices, int(self.idx))
            pos = (pos + direction) % len(indices)
            self.idx = int(indices[pos])

    def _update_output(self):
        if self._images is None:
            return
        i = int(self.idx)
        self.output_pattern = self._images[i]
        self.pattern = self._images[i]
        self.output_digit = int(self._labels[i])
        self.digit = str(self._labels[i])
