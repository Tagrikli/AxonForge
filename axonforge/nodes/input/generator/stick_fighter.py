"""Stick figure fighter animation input node."""

import numpy as np

from ....core.node import Node
from ....core.descriptors.ports import OutputPort
from ....core.descriptors.fields import Integer, Range, Enum
from ....core.descriptors.displays import Heatmap, Text
from ....core.descriptors.actions import Action
from ....core.descriptors.state import State


# ---------------------------------------------------------------------------
# Skeleton definition
# ---------------------------------------------------------------------------
# Angle convention:
#   0 = straight DOWN (screen‑y positive)
#   positive = counter‑clockwise
#   π/2 → RIGHT,  π → UP,  3π/2 → LEFT
#
# Elbow / knee angles are RELATIVE to their parent limb.
#   absolute_forearm = shoulder_angle + elbow_angle
#   absolute_shin    = hip_angle     + knee_angle
#
# Bone hierarchy (parent → child):
#   neck → head (fixed up)
#   neck → l_elbow (upper arm) → l_hand (forearm)
#   neck → r_elbow (upper arm) → r_hand (forearm)
#   neck → hip (torso, fixed down)
#   hip  → l_knee (upper leg)  → l_foot (lower leg)
#   hip  → r_knee (upper leg)  → r_foot (lower leg)
# ---------------------------------------------------------------------------

_ANGLE_KEYS = [
    "l_shoulder", "l_elbow",
    "r_shoulder", "r_elbow",
    "l_hip", "l_knee",
    "r_hip", "r_knee",
]

# ---------------------------------------------------------------------------
# 15 fighting poses  (order matches _ANGLE_KEYS)
# ---------------------------------------------------------------------------
# Pose 0 is the GUARD stance — the neutral position that every attack
# transitions through, so all moves start and end from the same base.
#
# Angle cheat‑sheet (from DOWN, CCW):
#   0.0  ↓    0.79 ↘    1.57 →    2.36 ↗
#   3.14 ↑    3.93 ↖    4.71 ←    5.50 ↙
# ---------------------------------------------------------------------------

_POSE_LIBRARY = [
    # 0  guard stance
    #    arms: upper‑arms angled outward‑down, forearms fold up (fists near chin)
    #    legs: slightly apart, straight
    (5.50, -2.00,  0.80,  2.00,  6.10, 0.00,  0.20, 0.00),

    # 1  left jab — left arm extends upper‑left
    (4.20,  0.00,  0.80,  2.00,  6.10, 0.00,  0.20, 0.00),

    # 2  right jab — right arm extends upper‑right
    (5.50, -2.00,  2.10,  0.00,  6.10, 0.00,  0.20, 0.00),

    # 3  left cross — full extension upper‑left, rear hand tucked
    (3.90,  0.00,  0.50,  2.50,  6.10, 0.10,  0.20, -0.10),

    # 4  right cross — full extension upper‑right, rear hand tucked
    (5.80, -2.50,  2.40,  0.00,  6.10, -0.10,  0.20, 0.10),

    # 5  left uppercut — arm swings up from below
    (0.30,  3.00,  0.80,  2.00,  6.10, 0.20,  0.20, 0.00),

    # 6  right uppercut
    (5.50, -2.00,  6.00,  3.30,  6.10, 0.00,  0.20, 0.20),

    # 7  left hook — arm sweeps from the left
    (3.93,  0.50,  0.80,  2.00,  6.10, 0.00,  0.20, 0.00),

    # 8  right hook — arm sweeps from the right
    (5.50, -2.00,  2.36, -0.50,  6.10, 0.00,  0.20, 0.00),

    # 9  right side kick — right leg extends right
    (5.20, -1.80,  1.00,  1.80,  6.10, 0.00,  1.57, 0.00),

    # 10 left side kick — left leg extends left
    (5.50, -2.00,  0.80,  2.00,  4.71, 0.00,  0.20, 0.00),

    # 11 right high kick — right leg upper‑right diagonal
    (5.20, -1.80,  0.50,  1.80,  6.10, 0.00,  2.36, 0.00),

    # 12 left high kick — left leg upper‑left diagonal
    (5.50, -2.00,  1.00,  1.80,  3.93, 0.00,  0.20, 0.00),

    # 13 right knee strike — knee raised, shin folds down
    (5.50, -1.50,  0.80,  1.50,  6.10, 0.00,  2.80, -2.80),

    # 14 left knee strike
    (5.50, -1.50,  0.80,  1.50,  3.50, -3.50,  0.20, 0.00),
]

_GUARD_INDEX = 0
_WALK_POSES = [
    (5.85, -1.40,  0.45,  1.40,  5.85, 0.35,  0.45, -0.35),
    (5.65, -1.40,  0.65,  1.40,  5.45, -0.35,  0.85, 0.35),
]
_WALK_STEP_FRAC = 0.32
_SCENE_EDGE_PADDING = 2.0

# Bone lengths as fraction of total figure height
_HEAD_RADIUS_FRAC = 0.08
_NECK_LEN_FRAC    = 0.05
_TORSO_LEN_FRAC   = 0.28
_UPPER_ARM_FRAC   = 0.18
_FOREARM_FRAC     = 0.16
_UPPER_LEG_FRAC   = 0.22
_LOWER_LEG_FRAC   = 0.20


def _pose_to_angles(pose_tuple):
    """Convert a pose tuple to a dict keyed by _ANGLE_KEYS."""
    return dict(zip(_ANGLE_KEYS, pose_tuple))


def _forward_kinematics(angles, fig_height, origin_x=0.0):
    """Compute joint (x, y) positions via forward kinematics.

    Returns dict mapping joint name → (x, y).
    Origin is at neck, y‑axis points DOWN (screen coords).
    """
    neck = (origin_x, 0.0)
    head = (origin_x, -(_NECK_LEN_FRAC + _HEAD_RADIUS_FRAC) * fig_height)

    torso_len = _TORSO_LEN_FRAC * fig_height
    hip = (origin_x, torso_len)

    ua = _UPPER_ARM_FRAC * fig_height
    fa = _FOREARM_FRAC * fig_height
    ul = _UPPER_LEG_FRAC * fig_height
    ll = _LOWER_LEG_FRAC * fig_height

    def _chain(origin, angle, length):
        x = origin[0] + length * np.sin(angle)
        y = origin[1] + length * np.cos(angle)
        return (x, y)

    # Arms
    l_elbow = _chain(neck, angles["l_shoulder"], ua)
    l_hand  = _chain(l_elbow, angles["l_shoulder"] + angles["l_elbow"], fa)

    r_elbow = _chain(neck, angles["r_shoulder"], ua)
    r_hand  = _chain(r_elbow, angles["r_shoulder"] + angles["r_elbow"], fa)

    # Legs
    l_knee = _chain(hip, angles["l_hip"], ul)
    l_foot = _chain(l_knee, angles["l_hip"] + angles["l_knee"], ll)

    r_knee = _chain(hip, angles["r_hip"], ul)
    r_foot = _chain(r_knee, angles["r_hip"] + angles["r_knee"], ll)

    return {
        "head": head, "neck": neck,
        "l_elbow": l_elbow, "l_hand": l_hand,
        "r_elbow": r_elbow, "r_hand": r_hand,
        "hip": hip,
        "l_knee": l_knee, "l_foot": l_foot,
        "r_knee": r_knee, "r_foot": r_foot,
    }


def _pose_x_bounds(pose_tuple, fig_height, origin_x=0.0):
    """Return the horizontal bounds of a pose, including head radius."""
    joints = _forward_kinematics(_pose_to_angles(pose_tuple), fig_height, origin_x=origin_x)
    xs = [x for x, _ in joints.values()]
    head_x, _ = joints["head"]
    head_r = _HEAD_RADIUS_FRAC * fig_height
    xs.extend((head_x - head_r, head_x + head_r))
    return (min(xs), max(xs))


def _motion_x_bounds(fig_height):
    """Return conservative x bounds covering both fighting and walking poses."""
    min_x = float("inf")
    max_x = float("-inf")
    for pose in _POSE_LIBRARY + _WALK_POSES:
        pose_min_x, pose_max_x = _pose_x_bounds(pose, fig_height)
        min_x = min(min_x, pose_min_x)
        max_x = max(max_x, pose_max_x)
    return (min_x, max_x)


# Bones to draw as line segments
_BONES = [
    ("neck",    "hip"),       # torso
    ("neck",    "l_elbow"),   # left upper arm
    ("l_elbow", "l_hand"),    # left forearm
    ("neck",    "r_elbow"),   # right upper arm
    ("r_elbow", "r_hand"),    # right forearm
    ("hip",     "l_knee"),    # left upper leg
    ("l_knee",  "l_foot"),    # left lower leg
    ("hip",     "r_knee"),    # right upper leg
    ("r_knee",  "r_foot"),    # right lower leg
]


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _draw_line_aa(canvas, x0, y0, x1, y1, thickness=1.0):
    """Draw an anti‑aliased line on a float32 canvas using distance field."""
    h, w = canvas.shape
    margin = thickness + 2
    min_x = max(0, int(min(x0, x1) - margin))
    max_x = min(w, int(max(x0, x1) + margin) + 1)
    min_y = max(0, int(min(y0, y1) - margin))
    max_y = min(h, int(max(y0, y1) + margin) + 1)

    if min_x >= max_x or min_y >= max_y:
        return

    ys, xs = np.mgrid[min_y:max_y, min_x:max_x]
    xs = xs.astype(np.float32)
    ys = ys.astype(np.float32)

    dx = x1 - x0
    dy = y1 - y0
    seg_len_sq = dx * dx + dy * dy

    if seg_len_sq < 1e-6:
        dist = np.sqrt((xs - x0) ** 2 + (ys - y0) ** 2)
    else:
        t = ((xs - x0) * dx + (ys - y0) * dy) / seg_len_sq
        t = np.clip(t, 0.0, 1.0)
        proj_x = x0 + t * dx
        proj_y = y0 + t * dy
        dist = np.sqrt((xs - proj_x) ** 2 + (ys - proj_y) ** 2)

    half_t = thickness / 2.0
    intensity = np.clip(half_t + 0.5 - dist, 0.0, 1.0)
    region = canvas[min_y:max_y, min_x:max_x]
    np.maximum(region, intensity, out=region)


def _draw_line_pixel(canvas, x0, y0, x1, y1):
    """Draw a pixel‑perfect (Bresenham) line on a float32 canvas."""
    h, w = canvas.shape
    ix0, iy0 = int(round(x0)), int(round(y0))
    ix1, iy1 = int(round(x1)), int(round(y1))

    dx = abs(ix1 - ix0)
    dy = abs(iy1 - iy0)
    sx = 1 if ix0 < ix1 else -1
    sy = 1 if iy0 < iy1 else -1
    err = dx - dy

    while True:
        if 0 <= ix0 < w and 0 <= iy0 < h:
            canvas[iy0, ix0] = 1.0
        if ix0 == ix1 and iy0 == iy1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            ix0 += sx
        if e2 < dx:
            err += dx
            iy0 += sy


def _draw_circle_aa(canvas, cx, cy, radius):
    """Draw a filled anti‑aliased circle."""
    h, w = canvas.shape
    margin = radius + 2
    min_x = max(0, int(cx - margin))
    max_x = min(w, int(cx + margin) + 1)
    min_y = max(0, int(cy - margin))
    max_y = min(h, int(cy + margin) + 1)

    if min_x >= max_x or min_y >= max_y:
        return

    ys, xs = np.mgrid[min_y:max_y, min_x:max_x]
    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    intensity = np.clip(radius + 0.5 - dist, 0.0, 1.0)
    region = canvas[min_y:max_y, min_x:max_x]
    np.maximum(region, intensity, out=region)


def _draw_circle_pixel(canvas, cx, cy, radius):
    """Draw a filled pixel‑perfect circle."""
    h, w = canvas.shape
    r = int(round(radius))
    icx, icy = int(round(cx)), int(round(cy))
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy <= r * r:
                px, py = icx + dx, icy + dy
                if 0 <= px < w and 0 <= py < h:
                    canvas[py, px] = 1.0


def _render_frame(joints, resolution, render_mode):
    """Render a single frame given joint positions.

    Returns np.ndarray of shape (resolution, resolution), float32, values in [0, 1].
    """
    canvas = np.zeros((resolution, resolution), dtype=np.float32)
    fig_height = resolution * 0.6
    offset_x = resolution / 2.0
    offset_y = resolution * 0.25

    aa = render_mode == "Anti-Aliased"
    line_thickness = max(1.0, resolution / 20.0) if aa else 1.0

    def _to_px(jname):
        x, y = joints[jname]
        return (x + offset_x, y + offset_y)

    for j1, j2 in _BONES:
        px1 = _to_px(j1)
        px2 = _to_px(j2)
        if aa:
            _draw_line_aa(canvas, px1[0], px1[1], px2[0], px2[1], line_thickness)
        else:
            _draw_line_pixel(canvas, px1[0], px1[1], px2[0], px2[1])

    hx, hy = _to_px("head")
    head_r = _HEAD_RADIUS_FRAC * fig_height
    if aa:
        _draw_circle_aa(canvas, hx, hy, head_r)
    else:
        _draw_circle_pixel(canvas, hx, hy, head_r)

    return canvas


# ---------------------------------------------------------------------------
# Interpolation helpers
# ---------------------------------------------------------------------------

def _lerp_poses(pose_a, pose_b, t):
    """Linearly interpolate between two pose tuples."""
    return tuple(a + (b - a) * t for a, b in zip(pose_a, pose_b))


def _smooth_step(t):
    """Hermite smooth‑step: 3t² − 2t³."""
    return t * t * (3.0 - 2.0 * t)


# ---------------------------------------------------------------------------
# Cache key encoding
# ---------------------------------------------------------------------------

def _encode_cache_key(num_poses, resolution, speed, pose_indices, walk_directions, render_mode, interp_mode):
    idx_str = ".".join(str(i) for i in pose_indices)
    walk_str = "".join("r" if d > 0 else "l" for d in walk_directions) or "none"
    rm = "aa" if render_mode == "Anti-Aliased" else "px"
    im = "sm" if interp_mode == "Smooth" else "ln"
    return f"p{num_poses}_r{resolution}_s{speed}_i{idx_str}_w{walk_str}_{rm}_{im}"


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class InputStickFighter(Node):
    """Generate a looping stick‑figure fighting animation.

    The animation interleaves every attack pose with the guard stance and
    animated walks, so later moves can happen at a new horizontal position.
    """

    # -- Ports ---------------------------------------------------------------
    output_frame = OutputPort("Frame", np.ndarray)

    # -- Displays ------------------------------------------------------------
    preview = Heatmap("Preview", colormap="grayscale")
    cache_status = Text("Cache", default="Live")
    info = Text("Info", default="")

    # -- Fields --------------------------------------------------------------
    num_poses   = Integer("Moves", default=3)
    resolution  = Integer("Resolution", default=28)
    speed       = Integer("Speed (frames/transition)", default=30)
    render_mode = Enum("Render", ["Anti-Aliased", "Pixel Perfect"], default="Anti-Aliased")
    interp_mode = Enum("Interpolation", ["Linear", "Smooth"], default="Smooth")

    # -- Actions -------------------------------------------------------------
    compile_btn   = Action("Compile", lambda self, params=None: self._on_compile(params))
    randomize_btn = Action("Randomize", lambda self, params=None: self._on_randomize(params))

    # -- State ---------------------------------------------------------------
    frame_idx    = State(default=0)
    pose_indices = State(default=None)   # list[int] — chosen attack poses (excl. guard)
    walk_directions = State(default=None)   # list[int] — -1 for left, +1 for right
    cache_key    = State(default="")

    def init(self):
        self._cache = None
        self._pick_poses()
        self._update_info()

    def process(self):
        n_poses = self._desired_pose_count()
        spd = max(1, int(self.speed))
        res = max(4, int(self.resolution))

        seq = self._build_sequence(res)
        total_frames = max(1, (len(seq) - 1) * spd)

        idx = int(self.frame_idx) % total_frames

        current_key = _encode_cache_key(
            n_poses, res, spd, self._get_pose_indices(), self._get_walk_directions(),
            str(self.render_mode), str(self.interp_mode),
        )

        if self._cache is not None and str(self.cache_key) == current_key:
            frame = self._cache[idx]
            self.cache_status = f"Cached [{current_key}]"
        else:
            frame = self._render_single(idx, seq, spd, res)
            if self._cache is not None:
                self.cache_status = "Stale — params changed"
            else:
                self.cache_status = "Live"

        self.output_frame = frame
        self.preview = frame
        self.frame_idx = (idx + 1) % total_frames
        self._update_info()

    # -- Internal ------------------------------------------------------------

    def _desired_pose_count(self):
        return max(1, min(int(self.num_poses), len(_POSE_LIBRARY) - 1))

    def _coerce_state_list(self, value):
        if value is None:
            return []
        if isinstance(value, str):
            return [int(x) for x in value.strip("[]").split(",") if x.strip()]
        return [int(x) for x in value]

    def _get_pose_indices(self):
        pi = self._coerce_state_list(self.pose_indices)
        if len(pi) != self._desired_pose_count():
            self._pick_poses()
            pi = self._coerce_state_list(self.pose_indices)
        return pi

    def _get_walk_directions(self):
        directions = self._coerce_state_list(self.walk_directions)
        expected = max(0, self._desired_pose_count() - 1)
        if len(directions) != expected:
            self._pick_poses()
            directions = self._coerce_state_list(self.walk_directions)
        return [1 if d >= 0 else -1 for d in directions]

    def _walk_limits(self, resolution):
        fig_height = resolution * 0.6
        min_pose_x, max_pose_x = _motion_x_bounds(fig_height)
        center_x = resolution / 2.0
        min_offset = _SCENE_EDGE_PADDING - center_x - min_pose_x
        max_offset = (resolution - 1.0 - _SCENE_EDGE_PADDING) - center_x - max_pose_x
        if min_offset > max_offset:
            return (0.0, 0.0)
        return (min_offset, max_offset)

    def _choose_walk_target(self, current_x, direction, travel, min_offset, max_offset):
        for step_dir in (direction, -direction):
            target_x = float(np.clip(current_x + step_dir * travel, min_offset, max_offset))
            if abs(target_x - current_x) > 1e-3:
                return target_x
        return current_x

    def _build_sequence(self, resolution):
        """Build keyframed poses with persistent x translation between fights."""
        attacks = self._get_pose_indices()
        walk_directions = self._get_walk_directions()
        guard_pose = _POSE_LIBRARY[_GUARD_INDEX]
        fig_height = resolution * 0.6
        min_offset, max_offset = self._walk_limits(resolution)
        travel = min(fig_height * _WALK_STEP_FRAC, max_offset - min_offset)

        current_x = 0.0
        seq = [(guard_pose, current_x)]
        for idx, aid in enumerate(attacks):
            attack_pose = _POSE_LIBRARY[aid]
            seq.append((attack_pose, current_x))
            seq.append((guard_pose, current_x))

            if idx >= len(attacks) - 1 or travel <= 1e-3:
                continue

            next_x = self._choose_walk_target(
                current_x,
                walk_directions[idx] if idx < len(walk_directions) else 1,
                travel,
                min_offset,
                max_offset,
            )
            if abs(next_x - current_x) <= 1e-3:
                continue

            walk_poses = _WALK_POSES if next_x > current_x else list(reversed(_WALK_POSES))
            span_x = next_x - current_x
            for walk_idx, walk_pose in enumerate(walk_poses, start=1):
                frac = walk_idx / (len(walk_poses) + 1)
                seq.append((walk_pose, current_x + span_x * frac))
            seq.append((guard_pose, next_x))
            current_x = next_x

        # Close the loop with an animated return so the sequence does not
        # snap back to the original position between cycles.
        if abs(current_x) > 1e-3:
            target_x = 0.0
            walk_poses = _WALK_POSES if target_x > current_x else list(reversed(_WALK_POSES))
            span_x = target_x - current_x
            for walk_idx, walk_pose in enumerate(walk_poses, start=1):
                frac = walk_idx / (len(walk_poses) + 1)
                seq.append((walk_pose, current_x + span_x * frac))
            seq.append((guard_pose, target_x))

        return seq

    def _pick_poses(self):
        """Randomly choose num_poses distinct attack poses (excluding guard)."""
        n = self._desired_pose_count()
        indices = np.random.choice(
            range(1, len(_POSE_LIBRARY)), size=n, replace=False,
        ).tolist()
        self.pose_indices = indices
        if n > 1:
            self.walk_directions = np.random.choice([-1, 1], size=n - 1).tolist()
        else:
            self.walk_directions = []

    def _render_single(self, frame_idx, seq, frames_per_transition, resolution):
        """Render one frame live."""
        seg_count = max(1, len(seq) - 1)
        seg = min(frame_idx // frames_per_transition, seg_count - 1)
        local_t = (frame_idx % frames_per_transition) / max(1, frames_per_transition)

        pose_a, origin_a = seq[seg]
        pose_b, origin_b = seq[seg + 1]

        if str(self.interp_mode) == "Smooth":
            t = _smooth_step(local_t)
        else:
            t = local_t

        blended = _lerp_poses(pose_a, pose_b, t)
        angles = _pose_to_angles(blended)
        origin_x = origin_a + (origin_b - origin_a) * t

        fig_height = resolution * 0.6
        joints = _forward_kinematics(angles, fig_height, origin_x=origin_x)

        return _render_frame(joints, resolution, str(self.render_mode))

    def _compile_all_frames(self):
        """Pre‑render every frame and store as 3D cache array."""
        spd = max(1, int(self.speed))
        res = max(4, int(self.resolution))
        seq = self._build_sequence(res)
        total = max(1, (len(seq) - 1) * spd)

        frames = np.empty((total, res, res), dtype=np.float32)
        for i in range(total):
            frames[i] = self._render_single(i, seq, spd, res)

        self._cache = frames
        self.cache_key = _encode_cache_key(
            self._desired_pose_count(), res, spd, self._get_pose_indices(), self._get_walk_directions(),
            str(self.render_mode), str(self.interp_mode),
        )

    def _update_info(self):
        ids = self._get_pose_indices()
        walk_dirs = "".join("R" if d > 0 else "L" for d in self._get_walk_directions()) or "-"
        idx = int(self.frame_idx)
        seq = self._build_sequence(max(4, int(self.resolution)))
        spd = max(1, int(self.speed))
        total = max(1, (len(seq) - 1) * spd)
        self.info = f"Frame {idx}/{total} | Moves: {ids} | Walk: {walk_dirs}"

    # -- Action callbacks ----------------------------------------------------

    def _on_compile(self, params=None):
        self._compile_all_frames()
        self.cache_status = f"Cached [{self.cache_key}]"
        return {"status": "ok", "message": f"Compiled {self._cache.shape[0]} frames"}

    def _on_randomize(self, params=None):
        self._pick_poses()
        self._cache = None
        self.cache_key = ""
        self.cache_status = "Live"
        self.frame_idx = 0
        self._update_info()
        return {
            "status": "ok",
            "message": f"New moves: {self._get_pose_indices()} | Walk: {self._get_walk_directions()}",
        }
