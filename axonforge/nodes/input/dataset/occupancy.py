"""Occupancy dataset input node for MiniCortex."""

import numpy as np
from pathlib import Path
from PIL import Image

import random
from ....core.node import Node, background_init
from ....core.descriptors.ports import OutputPort
from ....core.descriptors.properties import Enum, Bool
from ....core.descriptors.displays import Vector2D
from ....core.descriptors.store import Store


class OccupancyDataset(Node):
    """Cycle through occupancy images.
    
    Displays images from data/classified/occupied and data/classified/not_occupied.
    """
    
    category = Enum("Category", ["occupied", "not_occupied", "mixed"], default="occupied", on_change="_on_category_change")
    random_order = Bool("Random", default=False)
    
    output_image = OutputPort("Image", np.ndarray)
    display = Vector2D("Display", color_mode="grayscale")
    
    idx = Store(default=0)
    
    def _on_category_change(self, new_val, old_val):
        self.idx = 0
        self._update_file_list()

    @background_init
    def init(self):
        self._update_file_list()
        if self._files:
            self._load_and_process(int(self.idx))

    def _update_file_list(self):
        # Go up from axonforge/nodes/input/dataset/ to project root
        base_path = Path(__file__).parent.parent.parent.parent.parent / "data" / "classified"
        occupied_path = base_path / "occupied"
        not_occupied_path = base_path / "not_occupied"
        
        self._files = []
        if self.category == "occupied" or self.category == "mixed":
            if occupied_path.exists():
                self._files.extend(sorted([str(p) for p in occupied_path.glob("*.jpeg")]))
        
        if self.category == "not_occupied" or self.category == "mixed":
            if not_occupied_path.exists():
                self._files.extend(sorted([str(p) for p in not_occupied_path.glob("*.jpeg")]))
        
        if self.category == "mixed":
            self._files.sort()

    def process(self):
        if not hasattr(self, "_files") or not self._files:
            self._update_file_list()
            
        if self._files:
            if self.random_order:
                current_idx = random.randint(0, len(self._files) - 1)
            else:
                # Ensure idx is within bounds
                current_idx = int(self.idx)
                if current_idx >= len(self._files):
                    current_idx = 0
            
            self._load_and_process(current_idx)
            
            if not self.random_order:
                # Advance to next image
                self.idx = (current_idx + 1) % len(self._files)
            else:
                # Keep track of last index even in random mode
                self.idx = current_idx

    def _load_and_process(self, index: int):
        if not self._files:
            return
            
        path = self._files[index % len(self._files)]
        try:
            # Read image, convert to grayscale
            img = Image.open(path).convert('L')
            # Convert to numpy array and normalize to 0-1
            img_np = np.array(img).astype(np.float32) / 255.0
            
            self.output_image = img_np
            self.display = img_np
        except Exception as e:
            print(f"Error loading image {path}: {e}")
