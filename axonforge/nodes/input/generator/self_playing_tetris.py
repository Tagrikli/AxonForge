"""Self-playing Tetris-like generator input node."""

import numpy as np

from ....core.node import Node
from ....core.descriptors.ports import OutputPort
from ....core.descriptors.fields import Integer, Enum
from ....core.descriptors.displays import Heatmap, Text
from ....core.descriptors.actions import Action
from ....core.descriptors.state import State


def _piece(name, kind, cells, value):
    xs = [x for x, _ in cells]
    ys = [y for _, y in cells]
    return {
        "name": name,
        "kind": kind,
        "cells": tuple(cells),
        "value": float(value),
        "width": max(xs) + 1,
        "height": max(ys) + 1,
    }


_KIND_VALUES = {
    "I": 0.20,
    "O": 0.32,
    "T": 0.44,
    "L": 0.56,
    "J": 0.68,
    "S": 0.80,
    "Z": 0.92,
}

_PIECES = [
    _piece("I-H", "I", ((0, 0), (1, 0), (2, 0), (3, 0)), _KIND_VALUES["I"]),
    _piece("I-V", "I", ((0, 0), (0, 1), (0, 2), (0, 3)), _KIND_VALUES["I"]),
    _piece("O", "O", ((0, 0), (1, 0), (0, 1), (1, 1)), _KIND_VALUES["O"]),
    _piece("T-Up", "T", ((0, 0), (1, 0), (2, 0), (1, 1)), _KIND_VALUES["T"]),
    _piece("T-Down", "T", ((1, 0), (0, 1), (1, 1), (2, 1)), _KIND_VALUES["T"]),
    _piece("T-Left", "T", ((1, 0), (0, 1), (1, 1), (1, 2)), _KIND_VALUES["T"]),
    _piece("T-Right", "T", ((0, 0), (0, 1), (1, 1), (0, 2)), _KIND_VALUES["T"]),
    _piece("L-Up", "L", ((0, 0), (0, 1), (0, 2), (1, 2)), _KIND_VALUES["L"]),
    _piece("L-Right", "L", ((0, 0), (1, 0), (2, 0), (0, 1)), _KIND_VALUES["L"]),
    _piece("L-Down", "L", ((0, 0), (1, 0), (1, 1), (1, 2)), _KIND_VALUES["L"]),
    _piece("L-Left", "L", ((2, 0), (0, 1), (1, 1), (2, 1)), _KIND_VALUES["L"]),
    _piece("J-Up", "J", ((1, 0), (1, 1), (1, 2), (0, 2)), _KIND_VALUES["J"]),
    _piece("J-Right", "J", ((0, 0), (0, 1), (1, 1), (2, 1)), _KIND_VALUES["J"]),
    _piece("J-Down", "J", ((0, 0), (1, 0), (0, 1), (0, 2)), _KIND_VALUES["J"]),
    _piece("J-Left", "J", ((0, 0), (1, 0), (2, 0), (2, 1)), _KIND_VALUES["J"]),
    _piece("S-H", "S", ((1, 0), (2, 0), (0, 1), (1, 1)), _KIND_VALUES["S"]),
    _piece("S-V", "S", ((0, 0), (0, 1), (1, 1), (1, 2)), _KIND_VALUES["S"]),
    _piece("Z-H", "Z", ((0, 0), (1, 0), (1, 1), (2, 1)), _KIND_VALUES["Z"]),
    _piece("Z-V", "Z", ((1, 0), (0, 1), (1, 1), (0, 2)), _KIND_VALUES["Z"]),
]
_KIND_ORDER = ("I", "O", "T", "L", "J", "S", "Z")
_PIECES_BY_KIND = {
    kind: tuple(piece for piece in _PIECES if piece["kind"] == kind)
    for kind in _KIND_ORDER
}

_LINE_REWARD = 9.0
_HEIGHT_PENALTY = 0.42
_HOLE_PENALTY = 0.85
_BUMPINESS_PENALTY = 0.24
_MAX_HEIGHT_PENALTY = 0.30
_CENTER_BIAS = 0.03


def _column_heights(board):
    rows, cols = board.shape
    heights = np.zeros(cols, dtype=np.int32)
    for col in range(cols):
        filled = np.flatnonzero(board[:, col] > 0.0)
        if filled.size > 0:
            heights[col] = rows - int(filled[0])
    return heights


def _count_holes(board):
    holes = 0
    rows, cols = board.shape
    for col in range(cols):
        seen_block = False
        for row in range(rows):
            filled = board[row, col] > 0.0
            if filled:
                seen_block = True
            elif seen_block:
                holes += 1
    return holes


def _can_place(board, piece, x_pos, y_pos):
    rows, cols = board.shape
    for dx, dy in piece["cells"]:
        x = x_pos + dx
        y = y_pos + dy
        if x < 0 or x >= cols or y < 0 or y >= rows:
            return False
        if board[y, x] > 0.0:
            return False
    return True


def _drop_row(board, piece, x_pos):
    if not _can_place(board, piece, x_pos, 0):
        return None

    y_pos = 0
    while _can_place(board, piece, x_pos, y_pos + 1):
        y_pos += 1
    return y_pos


def _stamp_piece(board, piece, x_pos, y_pos):
    for dx, dy in piece["cells"]:
        board[y_pos + dy, x_pos + dx] = piece["value"]


def _clear_lines(board):
    full_rows = np.all(board > 0.0, axis=1)
    cleared = int(np.count_nonzero(full_rows))
    if cleared == 0:
        return board, 0

    remaining = board[~full_rows]
    pad = np.zeros((cleared, board.shape[1]), dtype=np.float32)
    return np.vstack((pad, remaining)).astype(np.float32), cleared


def _evaluate_board(board, cleared_lines, landing_x, piece_width):
    heights = _column_heights(board)
    holes = _count_holes(board)
    bumpiness = int(np.sum(np.abs(np.diff(heights)))) if heights.size > 1 else 0
    max_height = int(np.max(heights)) if heights.size > 0 else 0
    aggregate_height = int(np.sum(heights))

    center = (board.shape[1] - 1) / 2.0
    piece_center = landing_x + (piece_width - 1) / 2.0
    center_penalty = abs(piece_center - center)

    return (
        cleared_lines * _LINE_REWARD
        - aggregate_height * _HEIGHT_PENALTY
        - holes * _HOLE_PENALTY
        - bumpiness * _BUMPINESS_PENALTY
        - max_height * _MAX_HEIGHT_PENALTY
        - center_penalty * _CENTER_BIAS
    )


class InputSelfPlayingTetris(Node):
    """Generate a self-playing Tetris-style falling-block animation."""

    output_frame = OutputPort("Frame", np.ndarray)

    preview = Heatmap("Preview", colormap="grayscale")
    info = Text("Info", default="")

    mode = Enum("Mode", ["Self-Playing", "Single Drop"], default="Self-Playing")
    size = Integer("Dimension", default=64)
    block_size = Integer("Block Size", default=6)
    speed = Integer("Speed (frames/step)", default=4)

    reset_btn = Action("Reset", lambda self, params=None: self._on_reset(params))

    frame_hold = State(default=0)
    lines_cleared = State(default=0)
    games_played = State(default=0)
    kind_index = State(default=0)
    drop_phase = State(default="fall")
    active_name = State(default="None")
    active_x = State(default=0)
    active_y = State(default=0)

    def init(self):
        self._board = None
        self._board_shape = None
        self._cell_px = 1
        self._active_piece = None
        self._mode_name = None
        self._sync_preview_config()
        self._reset_board(reset_games=False)
        self._update_output()

    def process(self):
        self._ensure_state_matches_fields()

        if str(self.mode) == "Single Drop":
            self._process_single_drop()
        else:
            self._process_self_playing()

        self._update_output()

    def _process_self_playing(self):
        if self._active_piece is None:
            self._spawn_piece()
            return

        hold = int(self.frame_hold)
        step_frames = max(1, int(self.speed))
        if hold + 1 >= step_frames:
            self.frame_hold = 0
            self._advance_piece()
        else:
            self.frame_hold = hold + 1

    def _process_single_drop(self):
        phase = str(self.drop_phase)

        if phase == "clear":
            self._active_piece = None
            self.active_name = "None"
            self.active_x = 0
            self.active_y = 0
            self.drop_phase = "spawn"
            self.frame_hold = 0
            return

        if phase == "spawn" or self._active_piece is None:
            self._spawn_random_piece()
            self.drop_phase = "fall"
            return

        hold = int(self.frame_hold)
        step_frames = max(1, int(self.speed))
        if hold + 1 < step_frames:
            self.frame_hold = hold + 1
            return

        self.frame_hold = 0
        next_y = int(self.active_y) + 1
        if _can_place(self._board, self._active_piece, int(self.active_x), next_y):
            self.active_y = next_y
            return

        self.drop_phase = "clear"

    def _board_geometry(self):
        size = max(8, int(self.size))
        requested = max(1, int(self.block_size))
        cells = max(4, int(round(size / requested)))
        cell_px = max(1, size // cells)
        return size, cells, cells, cell_px

    def _sync_preview_config(self):
        preview_descriptor = type(self).__dict__["preview"]
        preview_descriptor.change_scale(self, "manual", 0.0, 1.0)

    def _ensure_state_matches_fields(self):
        _, rows, cols, cell_px = self._board_geometry()
        shape = (rows, cols)
        mode_name = str(self.mode)
        if (
            self._board is None
            or self._board_shape != shape
            or self._cell_px != cell_px
            or self._mode_name != mode_name
        ):
            self._reset_board(reset_games=False)

    def _reset_board(self, reset_games):
        size, rows, cols, cell_px = self._board_geometry()
        self._board = np.zeros((rows, cols), dtype=np.float32)
        self._board_shape = (rows, cols)
        self._cell_px = cell_px
        self._mode_name = str(self.mode)
        self._active_piece = None
        self.active_name = "None"
        self.active_x = 0
        self.active_y = 0
        self.frame_hold = 0
        self.lines_cleared = 0
        self.kind_index = 0
        self.drop_phase = "fall"
        if reset_games:
            self.games_played = int(self.games_played) + 1

    def _advance_piece(self):
        if self._active_piece is None:
            self._spawn_piece()
            return

        next_y = int(self.active_y) + 1
        if _can_place(self._board, self._active_piece, int(self.active_x), next_y):
            self.active_y = next_y
            return

        _stamp_piece(self._board, self._active_piece, int(self.active_x), int(self.active_y))
        self._board, cleared = _clear_lines(self._board)
        self.lines_cleared = int(self.lines_cleared) + cleared
        self._active_piece = None
        self.active_name = "None"
        self.active_x = 0
        self.active_y = 0
        self._spawn_piece()

    def _spawn_piece(self):
        kind = _KIND_ORDER[int(self.kind_index) % len(_KIND_ORDER)]
        choice = self._best_spawn_choice(_PIECES_BY_KIND[kind])
        if choice is None:
            choice = self._best_spawn_choice(_PIECES)
        if choice is None:
            self._reset_board(reset_games=True)
            kind = _KIND_ORDER[int(self.kind_index) % len(_KIND_ORDER)]
            choice = self._best_spawn_choice(_PIECES_BY_KIND[kind])
            if choice is None:
                choice = self._best_spawn_choice(_PIECES)
            if choice is None:
                return

        self._active_piece = choice["piece"]
        self.active_name = self._active_piece["name"]
        self.active_x = int(choice["x"])
        self.active_y = 0
        self.frame_hold = 0
        self.kind_index = int(self.kind_index) + 1

    def _spawn_random_piece(self):
        cols = self._board.shape[1]
        piece = _PIECES[np.random.randint(0, len(_PIECES))]
        max_x = cols - piece["width"]
        if max_x < 0:
            self._active_piece = None
            self.active_name = "None"
            return

        self._active_piece = piece
        self.active_name = piece["name"]
        self.active_x = int(np.random.randint(0, max_x + 1))
        self.active_y = 0
        self.frame_hold = 0

    def _best_spawn_choice(self, pieces):
        board = self._board
        if board is None:
            return None

        best = None
        best_score = None
        cols = board.shape[1]

        for piece in pieces:
            max_x = cols - piece["width"]
            if max_x < 0:
                continue

            for x_pos in range(max_x + 1):
                landing_y = _drop_row(board, piece, x_pos)
                if landing_y is None:
                    continue

                trial = board.copy()
                _stamp_piece(trial, piece, x_pos, landing_y)
                trial, cleared = _clear_lines(trial)
                score = _evaluate_board(trial, cleared, x_pos, piece["width"])

                if best_score is None or score > best_score:
                    best_score = score
                    best = {
                        "piece": piece,
                        "x": x_pos,
                        "landing_y": landing_y,
                    }

        return best

    def _cell_bounds(self, index, count, size):
        start = (index * size) // count
        end = ((index + 1) * size) // count
        return start, end

    def _render_frame(self):
        size, rows, cols, cell_px = self._board_geometry()
        frame = np.zeros((size, size), dtype=np.float32)

        for row in range(rows):
            y0, y1 = self._cell_bounds(row, rows, size)
            for col in range(cols):
                value = float(self._board[row, col])
                if value <= 0.0:
                    continue
                x0, x1 = self._cell_bounds(col, cols, size)
                frame[y0:y1, x0:x1] = value

        if self._active_piece is not None:
            value = self._active_piece["value"]
            for dx, dy in self._active_piece["cells"]:
                row = int(self.active_y) + dy
                col = int(self.active_x) + dx
                y0, y1 = self._cell_bounds(row, rows, size)
                x0, x1 = self._cell_bounds(col, cols, size)
                frame[y0:y1, x0:x1] = value

        return frame

    def _update_output(self):
        self._sync_preview_config()
        frame = self._render_frame()
        self.output_frame = frame
        self.preview = frame

        _, rows, cols, cell_px = self._board_geometry()
        piece_text = f"{self.active_name}@({int(self.active_x)},{int(self.active_y)})"
        if str(self.mode) == "Single Drop":
            self.info = (
                f"Mode Single Drop | Board {rows}x{cols} | Cell {cell_px}px | "
                f"Piece {piece_text} | Phase {self.drop_phase}"
            )
        else:
            self.info = (
                f"Mode Self-Playing | Board {rows}x{cols} | Cell {cell_px}px | "
                f"Piece {piece_text} | Lines {int(self.lines_cleared)} | Games {int(self.games_played)}"
            )

    def _on_reset(self, params=None):
        self._reset_board(reset_games=False)
        self._update_output()
        return {"status": "ok", "message": "Reset self-playing Tetris board"}
