"""Generator input nodes for MiniCortex - synthetic pattern generators."""

import numpy as np

from ....core.node import Node
from ....core.descriptors.ports import InputPort, OutputPort
from ....core.descriptors.fields import Range, Integer, Bool, Enum
from ....core.descriptors.displays import Heatmap, Text
from ....core.descriptors.actions import Action
from ....core.descriptors.state import State
from ....core.descriptors import branch


class InputMovingShape(Node):
    """Generate a moving shape (square or circle) on a 2D grid."""

    output_pattern = OutputPort("Pattern", np.ndarray)
    pattern = Heatmap("Pattern", colormap="grayscale")
    info = Text("Info", default="Position: (0, 0)")

    # Properties
    grid_size = Integer("Grid Size", default=64)
    shape_type = Enum("Shape", ["Square", "Circle"], default="Square")
    shape_size = Integer("Shape Size", default=8)
    speed = Range("Speed", default=0.1, min_val=0.01, max_val=1.0, scale="log")
    interpolation = Enum("Interpolation", ["Linear", "Ease In", "Ease Out", "Ease In-Out"], default="Linear")
    
    # Checkboxes for modes
    auto_move = Bool("Auto Move", default=False)

    # Buttons
    prev_pos = Action("Prev", lambda self, params=None: self._on_prev(params))
    next_pos = Action("Next", lambda self, params=None: self._on_next(params))

    # State
    pos_x = State(default=0.0)
    pos_y = State(default=0.0)
    target_x = State(default=0.0)
    target_y = State(default=0.0)

    def init(self):
        self._generate_new_target()
        self._update_pattern()

    def process(self):
        if self.auto_move:
            self._move_towards_target()
        self._update_pattern()

    def _move_towards_target(self):
        """Move position towards target using interpolation."""
        speed = float(self.speed)
        current_x = float(self.pos_x)
        current_y = float(self.pos_y)
        target_x = float(self.target_x)
        target_y = float(self.target_y)
        
        # Calculate distance to target
        dx = target_x - current_x
        dy = target_y - current_y
        dist = np.sqrt(dx * dx + dy * dy)
        
        # Check if we've reached the target
        if dist < speed:
            self.pos_x = target_x
            self.pos_y = target_y
            # Generate new random target
            self._generate_new_target()
        else:
            # Move towards target
            self.pos_x = current_x + (dx / dist) * speed
            self.pos_y = current_y + (dy / dist) * speed

    def _generate_new_target(self):
        """Generate a new random target position."""
        grid_size = int(self.grid_size)
        shape_size = int(self.shape_size)
        margin = shape_size // 2 + 1
        self.target_x = np.random.uniform(margin, grid_size - margin - 1)
        self.target_y = np.random.uniform(margin, grid_size - margin - 1)

    def _apply_interpolation(self, t):
        """Apply interpolation function based on selected mode."""
        interp_mode = self.interpolation
        if interp_mode == "Linear":
            return t
        elif interp_mode == "Ease In":
            return t * t
        elif interp_mode == "Ease Out":
            return 1 - (1 - t) * (1 - t)
        elif interp_mode == "Ease In-Out":
            if t < 0.5:
                return 2 * t * t
            else:
                return 1 - 2 * (1 - t) * (1 - t)
        return t

    def _update_pattern(self):
        grid_size = int(self.grid_size)
        shape_size = int(self.shape_size)
        pos_x = float(self.pos_x)
        pos_y = float(self.pos_y)

        # Create coordinate grids
        y_coords, x_coords = np.ogrid[:grid_size, :grid_size]
        
        if self.shape_type == "Square":
            # Calculate distance from edges for anti-aliasing
            half_size = shape_size / 2.0
            dx = np.abs(x_coords - pos_x)
            dy = np.abs(y_coords - pos_y)
            
            # Anti-aliased square
            dist_x = half_size - dx
            dist_y = half_size - dy
            pattern = np.minimum(np.clip(dist_x, 0.0, 1.0), np.clip(dist_y, 0.0, 1.0))
        else:  # Circle
            # Calculate distance from center
            dx = x_coords - pos_x
            dy = y_coords - pos_y
            dist = np.sqrt(dx * dx + dy * dy)
            
            # Anti-aliased circle
            radius = shape_size / 2.0
            pattern = np.clip(radius - dist, 0.0, 1.0)

        self.output_pattern = pattern.astype(np.float32)
        self.pattern = pattern.astype(np.float32)
        
        # Update info
        px = int(pos_x)
        py = int(pos_y)
        if self.auto_move:
            tx = int(float(self.target_x))
            ty = int(float(self.target_y))
            self.info = f"Pos: ({px}, {py}) → ({tx}, {ty})"
        else:
            self.info = f"Pos: ({px}, {py})"

    def _on_prev(self, params):
        """Move to previous position (step back)."""
        step = 5
        self.pos_x = max(0, float(self.pos_x) - step)
        self._update_pattern()
        return {"status": "ok"}

    def _on_next(self, params):
        """Move to next position (step forward)."""
        step = 5
        grid_size = int(self.grid_size)
        self.pos_x = min(grid_size - 1, float(self.pos_x) + step)
        self._update_pattern()
        return {"status": "ok"}

