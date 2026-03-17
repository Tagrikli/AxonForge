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
    render_mode = Enum("Render", ["Anti-Aliased", "Pixel Perfect"], default="Anti-Aliased")
    
    # Motion controls
    animate = Bool("Animate", default=False)
    target_mode = Enum("Mode", ["Linear Mode", "Random Mode", "Ping Pong"], default="Random Mode")

    # Buttons
    prev_pattern = Action("Prev", lambda self, params=None: self._on_prev(params))
    next_pattern = Action("Next", lambda self, params=None: self._on_next(params))

    # State
    angle = State(default=0.0)
    target_angle = State(default=0.0)
    interp_progress = State(default=0.0)
    start_angle = State(default=0.0)
    motion_delta = State(default=0.0)
    ping_direction = State(default=1)

    def init(self):
        self._last_motion_mode = None
        self._reset_target_motion()
        self._update_pattern()

    def process(self):
        motion_mode = (bool(self.animate), str(self.target_mode))
        if motion_mode != self._last_motion_mode:
            self._reset_target_motion()
            self._last_motion_mode = motion_mode

        if self.animate:
            if str(self.target_mode) == "Linear Mode":
                speed = float(self.rotation_speed)
                self.angle = (self.angle + speed) % (2 * np.pi)
            else:
                self._move_towards_target()
        
        self._update_pattern()

    def _move_towards_target(self):
        """Move angle towards the prepared target using the selected easing curve."""
        speed = float(self.rotation_speed)
        if speed <= 0.0:
            return

        progress = min(1.0, float(self.interp_progress) + speed)
        self.interp_progress = progress
        eased = self._apply_interpolation(progress)
        self.angle = (float(self.start_angle) + float(self.motion_delta) * eased) % (2 * np.pi)

        if progress >= 1.0 - 1e-9:
            self.angle = float(self.target_angle)
            self._prepare_next_target()

    def _normalize_delta(self, delta):
        while delta > np.pi:
            delta -= 2 * np.pi
        while delta < -np.pi:
            delta += 2 * np.pi
        return delta

    def _prepare_next_target(self):
        start = float(self.angle)
        self.start_angle = start
        self.interp_progress = 0.0

        mode = str(self.target_mode)
        if mode == "Ping Pong":
            direction = 1 if int(self.ping_direction) >= 0 else -1
            delta = direction * np.pi
            self.ping_direction = -direction
        elif mode == "Random Mode":
            raw_target = float(np.random.uniform(0.0, 2.0 * np.pi))
            delta = self._normalize_delta(raw_target - start)
        else:
            delta = 0.0

        self.motion_delta = delta
        self.target_angle = (start + delta) % (2 * np.pi)

    def _reset_target_motion(self):
        self.start_angle = float(self.angle)
        self.target_angle = float(self.angle)
        self.interp_progress = 0.0
        self.motion_delta = 0.0
        self.ping_direction = 1
        if str(self.target_mode) != "Linear Mode":
            self._prepare_next_target()

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

        half_thickness = thickness / 2.0
        if str(self.render_mode) == "Pixel Perfect":
            pattern = (perp_dist <= half_thickness).astype(np.float32)
        else:
            pattern = np.clip(half_thickness - perp_dist, 0.0, 1.0)

        self.output_pattern = pattern.astype(np.float32)
        self.pattern = pattern.astype(np.float32)
        
        # Output angle in degrees (0-360)
        angle_deg = np.degrees(angle) % 360.0
        self.output_angle = float(angle_deg)
        
        angle_deg = int(np.degrees(angle)) % 360
        mode = str(self.target_mode)
        if mode == "Linear Mode":
            status = "Animate" if self.animate else "Paused"
            self.info = f"Angle: {angle_deg}° | Linear Mode | {status}"
        else:
            target_deg = int(np.degrees(float(self.target_angle))) % 360
            status = "Animate" if self.animate else "Paused"
            self.info = f"Angle: {angle_deg}° → {target_deg}° | {mode} | {status}"

    def _on_prev(self, params):
        """Step angle backwards by 22.5 degrees."""
        step = np.pi / 8  # 22.5 degrees
        self.angle = (self.angle - step) % (2 * np.pi)
        self._reset_target_motion()
        self._update_pattern()
        return {"status": "ok"}

    def _on_next(self, params):
        """Step angle forwards by 22.5 degrees."""
        step = np.pi / 8  # 22.5 degrees
        self.angle = (self.angle + step) % (2 * np.pi)
        self._reset_target_motion()
        self._update_pattern()
        return {"status": "ok"}
