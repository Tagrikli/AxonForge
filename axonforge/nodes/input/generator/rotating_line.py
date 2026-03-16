"""Generator input nodes for MiniCortex - synthetic pattern generators."""

import numpy as np

from ....core.node import Node
from ....core.descriptors.ports import InputPort, OutputPort
from ....core.descriptors.fields import Range, Integer, Bool, Enum
from ....core.descriptors.displays import Heatmap, Text
from ....core.descriptors.actions import Action
from ....core.descriptors.state import State
from ....core.descriptors import branch


class InputRotatingLine(Node):
    """Generate a rotating line pattern on a square grid."""

    output_pattern = OutputPort("Pattern", np.ndarray)
    output_angle = OutputPort("Angle", float)
    pattern = Heatmap("Pattern", colormap="grayscale")
    info = Text("Info", default="Angle: 0°")

    # Properties
    size = Integer("Size", default=64)
    thickness = Integer("Thickness", default=2)
    rotation_speed = Range("Rotation Speed", default=0.1, min_val=0, max_val=1.0, scale="linear")
    interpolation = Enum("Interpolation", ["Linear", "Ease In", "Ease Out", "Ease In-Out"], default="Linear")
    
    # Checkboxes for modes
    auto_rotate = Bool("Auto Rotate", default=False)
    random_mode = Bool("Random Mode", default=False)

    # Buttons
    prev_pattern = Action("Prev", lambda self, params=None: self._on_prev(params))
    next_pattern = Action("Next", lambda self, params=None: self._on_next(params))

    # State
    angle = State(default=0.0)
    target_angle = State(default=0.0)
    interp_progress = State(default=0.0)
    start_angle = State(default=0.0)

    def init(self):
        self._generate_new_target()
        self._update_pattern()

    def process(self):
        if self.random_mode:
            # Move towards target angle using shortest path
            self._move_towards_target()
        elif self.auto_rotate:
            # Continuous rotation
            speed = float(self.rotation_speed)
            self.angle = (self.angle + speed) % (2 * np.pi)
        
        self._update_pattern()

    def _move_towards_target(self):
        """Move angle towards target using shortest path with interpolation."""
        speed = float(self.rotation_speed)
        current = float(self.angle)
        target = float(self.target_angle)
        
        # Calculate shortest path direction
        diff = target - current
        # Normalize to [-pi, pi]
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        
        # Check if we've reached the target
        if abs(diff) < speed:
            self.angle = target
            # Generate new random target
            self._generate_new_target()
        else:
            # Move in the shortest direction
            direction = 1 if diff > 0 else -1
            self.angle = (current + direction * speed) % (2 * np.pi)

    def _generate_new_target(self):
        """Generate a new random target angle."""
        self.target_angle = np.random.uniform(0, 2 * np.pi)

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
        size = int(self.size)
        thickness = int(self.thickness)
        angle = float(self.angle)

        # Create coordinate grids centered at the middle
        y_coords, x_coords = np.ogrid[:size, :size]
        cx, cy = size / 2.0, size / 2.0
        rx = x_coords - cx
        ry = y_coords - cy

        # Calculate perpendicular distance to the line through center at given angle
        # Line direction: (cos(angle), sin(angle))
        # Perpendicular distance: |rx * sin(angle) - ry * cos(angle)|
        line_dx = np.cos(angle)
        line_dy = np.sin(angle)
        perp_dist = np.abs(rx * line_dy - ry * line_dx)

        # Create line pattern with anti-aliasing at edges
        half_thickness = thickness / 2.0
        pattern = np.clip(half_thickness - perp_dist, 0.0, 1.0)

        self.output_pattern = pattern.astype(np.float32)
        self.pattern = pattern.astype(np.float32)
        
        # Output angle in degrees (0-360)
        angle_deg = np.degrees(angle) % 360.0
        self.output_angle = float(angle_deg)
        
        angle_deg = int(np.degrees(angle)) % 360
        if self.random_mode:
            target_deg = int(np.degrees(float(self.target_angle))) % 360
            self.info = f"Angle: {angle_deg}° → {target_deg}°"
        elif self.auto_rotate:
            self.info = f"Angle: {angle_deg}° | Auto"
        else:
            self.info = f"Angle: {angle_deg}°"

    def _on_prev(self, params):
        """Step angle backwards by 22.5 degrees."""
        step = np.pi / 8  # 22.5 degrees
        self.angle = (self.angle - step) % (2 * np.pi)
        self._update_pattern()
        return {"status": "ok"}

    def _on_next(self, params):
        """Step angle forwards by 22.5 degrees."""
        step = np.pi / 8  # 22.5 degrees
        self.angle = (self.angle + step) % (2 * np.pi)
        self._update_pattern()
        return {"status": "ok"}
