"""Dataset input nodes for MiniCortex."""

import numpy as np

from ....core.descriptors.displays import Text, Vector2D
from ....core.descriptors.ports import OutputPort
from ....core.descriptors.properties import Integer, Range
from ....core.descriptors.store import Store
from ....core.node import Node, background_init
from ....nodes.utilities import _load_dataset_with_python_mnist


class InputDigitMNIST(Node):
    """Cycle through MNIST digit images."""

    output_pattern = OutputPort("Pattern", np.ndarray)
    output_digit = OutputPort("Digit", int)
    pattern = Vector2D("Pattern", color_mode="grayscale")
    digit = Text("Digit", default="0")

    # Property: how many times to repeat each image
    repeat_count = Integer("Repeat", default=1)

    # Property: digit filter (-1 = all, 0-9 = specific digit class)
    digit_filter = Range("Digit Filter", default=-1, min_val=-1, max_val=9, step=1)

    # Store: index and step counter
    idx = Store(default=0)
    repeat_counter = Store(default=0)
    size = Store(default=28)

    @background_init
    def init(self):
        self._images, self._labels = _load_dataset_with_python_mnist("mnist")
        # Build indices for each digit class (0-9)
        self._class_indices = {}
        if self._labels is not None:
            for digit in range(10):
                self._class_indices[digit] = np.where(self._labels == digit)[0]
        self._update_pattern()

    def process(self):
        if self._images is not None:
            # Increment repeat counter
            self.repeat_counter = int(self.repeat_counter) + 1

            # Only advance to next digit when repeat_count is reached
            if int(self.repeat_counter) >= int(self.repeat_count):
                filter_val = int(self.digit_filter)
                if filter_val == -1:
                    # Current behavior: sequential through all images
                    self.idx = (int(self.idx) + 1) % len(self._images)
                else:
                    # Pick random image from the specified digit class
                    indices = self._class_indices.get(filter_val, [])
                    if len(indices) > 0:
                        self.idx = np.random.choice(indices)
                self.repeat_counter = 0

        self._update_pattern()

    def _update_pattern(self):
        if self._images is not None:
            self.output_pattern = self._images[self.idx]
            self.pattern = self._images[self.idx]
            self.output_digit = int(self._labels[self.idx])
            self.digit = str(self._labels[self.idx])
