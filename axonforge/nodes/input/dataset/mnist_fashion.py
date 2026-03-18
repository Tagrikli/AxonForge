import numpy as np

from ....core.descriptors.actions import Action
from ....core.descriptors.displays import Heatmap, Text
from ....core.descriptors.fields import Integer, Range
from ....core.descriptors.ports import OutputPort
from ....core.descriptors.state import State
from ....core.node import Node, background_init
from ....nodes.utilities.dataset_cache import load_dataset_with_python_mnist


LABEL_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


class InputFashionMNIST(Node):
    """Cycle through Fashion-MNIST images."""

    output_pattern = OutputPort("Pattern", np.ndarray)
    output_category = OutputPort("Category", int)
    pattern = Heatmap("Pattern", colormap="grayscale")
    category = Text("Category", default="0")

    category_filter = Range("Category Filter", default=-1, min_val=-1, max_val=9, step=1)
    repeat = Integer("Repeat", default=1)

    prev_button = Action("◀", lambda self, params=None: self.prev_item(params))
    next_button = Action("▶", lambda self, params=None: self.next_item(params))

    idx = State(default=0)
    repeat_counter = State(default=0)

    @background_init
    def init(self):
        self._images, self._labels = load_dataset_with_python_mnist("fashion_mnist")
        if self._images is None or self._labels is None:
            return

        perm = np.random.permutation(len(self._images))
        self._images = self._images[perm]
        self._labels = self._labels[perm]

        self._class_indices = {}
        for c in range(10):
            self._class_indices[c] = np.where(self._labels == c)[0]

        self._update_output()

    def process(self):
        if self._images is None:
            return

        rep = int(self.repeat)
        if rep <= 0:
            self._update_output()
            return

        self.repeat_counter = int(self.repeat_counter) + 1
        if int(self.repeat_counter) >= rep:
            self.repeat_counter = 0
            self._advance(1)

        self._update_output()

    def next_item(self, params=None):
        if self._images is not None:
            self._advance(1)
            self._update_output()

    def prev_item(self, params=None):
        if self._images is not None:
            self._advance(-1)
            self._update_output()

    def _advance(self, direction):
        filt = int(self.category_filter)
        if filt == -1:
            self.idx = (int(self.idx) + direction) % len(self._images)
        else:
            indices = self._class_indices.get(filt, [])
            if len(indices) == 0:
                return
            pos = np.searchsorted(indices, int(self.idx))
            pos = (pos + direction) % len(indices)
            self.idx = int(indices[pos])

    def _update_output(self):
        if self._images is None:
            return
        i = int(self.idx)
        label = int(self._labels[i])
        self.output_pattern = self._images[i]
        self.pattern = self._images[i]
        self.output_category = label
        self.category = LABEL_NAMES[label] if label < len(LABEL_NAMES) else str(label)
