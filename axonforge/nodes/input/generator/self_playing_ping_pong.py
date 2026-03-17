"""Self-playing ping pong generator input node."""

import math

import numpy as np

from ....core.node import Node
from ....core.descriptors.ports import OutputPort
from ....core.descriptors.fields import Integer, Bool, Range
from ....core.descriptors.displays import Heatmap, Text
from ....core.descriptors.actions import Action
from ....core.descriptors.state import State

_MOTION_PHASE_DEN = 1024


class InputSelfPlayingPingPong(Node):
    """Generate a deterministic self-playing ping pong animation."""

    output_frame = OutputPort("Frame", np.ndarray)

    preview = Heatmap("Preview", colormap="grayscale")
    info = Text("Info", default="")

    size = Integer("Dimension", default=64)
    paddle_height = Integer("Paddle Height", default=6)
    paddle_width = Integer("Paddle Width", default=1)
    ball_size = Integer("Ball Size", default=1)
    speed = Integer("Speed (steps/frame)", default=1)
    no_paddle = Bool("No Paddle", default=False)
    angle = Range("Angle", default=31.0, min_val=5.0, max_val=85.0, step=1.0, scale="linear")

    reset_btn = Action("Reset", lambda self, params=None: self._on_reset(params))

    ball_x = State(default=0)
    ball_y = State(default=0)
    ball_dx = State(default=1)
    ball_dy = State(default=1)
    phase_x = State(default=0)
    phase_y = State(default=0)
    left_y = State(default=0)
    right_y = State(default=0)
    step_count = State(default=0)

    def init(self):
        self._geometry_key = None
        self._sync_preview_config()
        self._reset_simulation()
        self._update_output()

    def process(self):
        self._ensure_geometry()

        for _ in range(max(1, int(self.speed))):
            geom = self._board_geometry()
            if not geom["wall_mode"]:
                left_target = self._ball_tracking_target(geom)
                right_target = self._ball_tracking_target(geom)
                paddle_step = max(1, abs(int(self.ball_dy)))
                self.left_y = self._move_toward(int(self.left_y), left_target, paddle_step)
                self.right_y = self._move_toward(int(self.right_y), right_target, paddle_step)
            self._advance_ball()
            self.step_count = int(self.step_count) + 1

        self._update_output()

    def _sync_preview_config(self):
        preview_descriptor = type(self).__dict__["preview"]
        preview_descriptor.change_scale(self, "manual", 0.0, 1.0)

    def _geometry_signature(self):
        return (
            max(4, int(self.size)),
            max(1, int(self.paddle_height)),
            max(1, int(self.paddle_width)),
            max(1, int(self.ball_size)),
            bool(self.no_paddle),
        )

    def _ensure_geometry(self):
        signature = self._geometry_signature()
        if signature != self._geometry_key:
            self._reset_simulation()

    def _board_geometry(self):
        size = max(4, int(self.size))
        cells = size

        paddle_w = min(max(1, int(self.paddle_width)), max(1, cells // 6))
        ball = min(max(1, int(self.ball_size)), max(1, cells // 5))
        paddle_h = min(max(2, int(self.paddle_height)), cells)
        wall_mode = bool(self.no_paddle)
        left_limit = 0 if wall_mode else paddle_w
        right_limit = cells - ball if wall_mode else cells - paddle_w - ball
        y_limit = cells - ball

        if right_limit <= left_limit:
            right_limit = left_limit + 1

        return {
            "size_px": size,
            "rows": cells,
            "cols": cells,
            "paddle_w": paddle_w,
            "paddle_h": paddle_h,
            "ball": ball,
            "wall_mode": wall_mode,
            "left_limit": left_limit,
            "right_limit": right_limit,
            "y_limit": max(0, y_limit),
        }

    def _motion_rates(self):
        angle_rad = math.radians(float(self.angle))
        vx = abs(math.cos(angle_rad))
        vy = abs(math.sin(angle_rad))

        if vx >= vy:
            rate_x = _MOTION_PHASE_DEN
            rate_y = int(round(_MOTION_PHASE_DEN * vy / max(vx, 1e-12)))
        else:
            rate_y = _MOTION_PHASE_DEN
            rate_x = int(round(_MOTION_PHASE_DEN * vx / max(vy, 1e-12)))

        rate_x = max(0, min(_MOTION_PHASE_DEN, rate_x))
        rate_y = max(0, min(_MOTION_PHASE_DEN, rate_y))
        return rate_x, rate_y

    def _reset_simulation(self):
        geom = self._board_geometry()
        self._geometry_key = self._geometry_signature()

        travel_x = max(1, geom["right_limit"] - geom["left_limit"])
        start_x = geom["left_limit"] + max(1, travel_x // 3)
        start_y = min(geom["y_limit"], max(0, (geom["y_limit"] * 5) // 17))
        paddle_range = max(0, geom["rows"] - geom["paddle_h"])

        self.ball_x = start_x
        self.ball_y = start_y
        self.ball_dx = 1
        self.ball_dy = 1
        self.phase_x = 0
        self.phase_y = _MOTION_PHASE_DEN // 3
        self.left_y = min(paddle_range, max(0, paddle_range // 3))
        self.right_y = min(paddle_range, max(0, (2 * paddle_range) // 3))
        if not geom["wall_mode"]:
            self.left_y = self._ball_tracking_target(geom)
            self.right_y = self._ball_tracking_target(geom)
        self.step_count = 0

    def _move_toward(self, current, target, max_step):
        if current < target:
            return min(current + max_step, target)
        if current > target:
            return max(current - max_step, target)
        return current

    def _reflect_axis(self, pos, delta, lower, upper):
        next_pos = pos + delta
        next_delta = delta

        while next_pos < lower or next_pos > upper:
            if next_pos < lower:
                next_pos = lower + (lower - next_pos)
                next_delta = -next_delta
            elif next_pos > upper:
                next_pos = upper - (next_pos - upper)
                next_delta = -next_delta

        return next_pos, next_delta

    def _step_ball_state(self, x_pos, y_pos, dx, dy, phase_x, phase_y, geom):
        rate_x, rate_y = self._motion_rates()

        phase_x_next = phase_x + rate_x
        phase_y_next = phase_y + rate_y
        move_x = 0
        move_y = 0

        if phase_x_next >= _MOTION_PHASE_DEN:
            phase_x_next -= _MOTION_PHASE_DEN
            move_x = dx
        if phase_y_next >= _MOTION_PHASE_DEN:
            phase_y_next -= _MOTION_PHASE_DEN
            move_y = dy

        x_next = x_pos
        dx_next = dx
        if move_x != 0:
            x_next, dx_next = self._reflect_axis(x_pos, move_x, geom["left_limit"], geom["right_limit"])
        y_next = y_pos
        dy_next = dy
        if move_y != 0:
            y_next, dy_next = self._reflect_axis(y_pos, move_y, 0, geom["y_limit"])

        return x_next, y_next, dx_next, dy_next, phase_x_next, phase_y_next

    def _advance_ball(self):
        geom = self._board_geometry()
        x_next, y_next, dx_next, dy_next, phase_x_next, phase_y_next = self._step_ball_state(
            int(self.ball_x),
            int(self.ball_y),
            int(self.ball_dx),
            int(self.ball_dy),
            int(self.phase_x),
            int(self.phase_y),
            geom,
        )

        self.ball_x = x_next
        self.ball_y = y_next
        self.ball_dx = dx_next
        self.ball_dy = dy_next
        self.phase_x = phase_x_next
        self.phase_y = phase_y_next

    def _ball_tracking_target(self, geom):
        paddle_range = max(0, geom["rows"] - geom["paddle_h"])
        _, next_y, _, _, _, _ = self._step_ball_state(
            int(self.ball_x),
            int(self.ball_y),
            int(self.ball_dx),
            int(self.ball_dy),
            int(self.phase_x),
            int(self.phase_y),
            geom,
        )
        ball_center = next_y + geom["ball"] / 2.0
        target = int(round(ball_center - geom["paddle_h"] / 2.0))
        return min(paddle_range, max(0, target))

    def _cell_bounds(self, index, count, size_px):
        start = (index * size_px) // count
        end = ((index + 1) * size_px) // count
        return start, end

    def _render_rect(self, frame, x0, y0, width, height, value, geom):
        size_px = geom["size_px"]
        cols = geom["cols"]
        rows = geom["rows"]

        for row in range(y0, y0 + height):
            py0, py1 = self._cell_bounds(row, rows, size_px)
            for col in range(x0, x0 + width):
                px0, px1 = self._cell_bounds(col, cols, size_px)
                frame[py0:py1, px0:px1] = value

    def _render_frame(self):
        geom = self._board_geometry()
        frame = np.zeros((geom["size_px"], geom["size_px"]), dtype=np.float32)

        if not geom["wall_mode"]:
            self._render_rect(
                frame,
                0,
                int(self.left_y),
                geom["paddle_w"],
                geom["paddle_h"],
                0.65,
                geom,
            )
            self._render_rect(
                frame,
                geom["cols"] - geom["paddle_w"],
                int(self.right_y),
                geom["paddle_w"],
                geom["paddle_h"],
                0.65,
                geom,
            )
        self._render_rect(
            frame,
            int(self.ball_x),
            int(self.ball_y),
            geom["ball"],
            geom["ball"],
            1.0,
            geom,
        )

        return frame

    def _update_output(self):
        self._sync_preview_config()
        geom = self._board_geometry()
        frame = self._render_frame()
        angle_val = float(self.angle)

        self.output_frame = frame
        self.preview = frame
        mode = "Walls" if geom["wall_mode"] else "Paddles"
        self.info = (
            f"Ball ({int(self.ball_x)}, {int(self.ball_y)}) d=({int(self.ball_dx)}, {int(self.ball_dy)}) | "
            f"Mode {mode} | Left {int(self.left_y)} Right {int(self.right_y)} | "
            f"Angle {angle_val:.0f}°"
        )

    def _on_reset(self, params=None):
        self._reset_simulation()
        self._update_output()
        return {"status": "ok", "message": "Reset self-playing ping pong"}
