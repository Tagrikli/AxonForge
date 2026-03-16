import numpy as np

from ....core.node import Node, background_init
from ....core.descriptors.ports import InputPort, OutputPort
from ....core.descriptors.displays import Heatmap, Text
from ....core.descriptors.state import State
from ....core.descriptors import branch
from ....nodes.utilities.dataset_cache import load_dataset_with_python_mnist


class InputFashionMNIST(Node):
    """Cycle through Fashion-MNIST images."""

    output_pattern = OutputPort("Pattern", np.ndarray)
    output_category = OutputPort("Category", int)
    pattern = Heatmap("Pattern", colormap="grayscale")
    info = Text("Info", default="Fashion: 0")
    
    # State: current image index
    idx = State(default=0)
    
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

    @background_init
    def init(self):
        self._images, self._labels = load_dataset_with_python_mnist("fashion_mnist")
        self._update_pattern()

    def process(self):
        if self._images is not None:
            self.idx = (self.idx + 1) % len(self._images)
        self._update_pattern()

    def _update_pattern(self):
        if self._images is not None:
            self.output_pattern = self._images[self.idx]
            self.pattern = self._images[self.idx]
            l = int(self._labels[self.idx])
            self.output_category = l
            self.info = (
                f"Fashion: {self.LABEL_NAMES[l] if l < len(self.LABEL_NAMES) else l}"
            )
