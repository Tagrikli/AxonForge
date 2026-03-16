"""Noise generator nodes for MiniCortex."""

import numpy as np
from typing import Optional

from axonforge.core.node import Node
from axonforge.core.descriptors.ports import InputPort, OutputPort
from axonforge.core.descriptors.fields import Range, Integer
from axonforge.core.descriptors.state import State
from axonforge.core.descriptors import branch


class NoiseGaussian(Node):
    """Add Gaussian (normal) noise to input array."""

    input_data = InputPort("Input", np.ndarray)
    output = OutputPort("Output", np.ndarray)
    
    # Properties
    mean = Range("Mean", 0.0, -1.0, 1.0, scale="linear")
    std = Range("Std Dev", 0.1, 0.001, 1.0, scale="log")
    seed = Integer("Seed", default=0)
    
    # State
    rng_state = State(default=None)

    def init(self):
        self._rng = np.random.default_rng(int(self.seed) if self.seed else None)

    def process(self):
        if self.input_data is None:
            return
        mean = float(self.mean)
        std = float(self.std)
        noise = self._rng.normal(mean, std, self.input_data.shape)
        self.output = (self.input_data + noise).astype(np.float32)


class NoiseUniform(Node):
    """Add uniform noise to input array."""

    input_data = InputPort("Input", np.ndarray)
    output = OutputPort("Output", np.ndarray)
    
    # Properties
    low = Range("Low", -0.1, -1.0, 1.0, scale="linear")
    high = Range("High", 0.1, 0.0, 2.0, scale="linear")
    seed = Integer("Seed", default=0)
    
    # State
    rng_state = State(default=None)

    def init(self):
        self._rng = np.random.default_rng(int(self.seed) if self.seed else None)

    def process(self):
        if self.input_data is None:
            return
        low = float(self.low)
        high = float(self.high)
        noise = self._rng.uniform(low, high, self.input_data.shape)
        self.output = (self.input_data + noise).astype(np.float32)


class NoisePerlin(Node):
    """Add Perlin-like noise to input array."""

    input_data = InputPort("Input", np.ndarray)
    output = OutputPort("Output", np.ndarray)
    
    # Properties
    scale = Integer("Scale", default=4)
    octaves = Integer("Octaves", default=4)
    persistence = Range("Persistence", 0.5, 0.1, 0.9, scale="linear")
    intensity = Range("Intensity", 0.5, 0.0, 1.0, scale="linear")
    seed = Integer("Seed", default=0)
    
    # State
    rng_state = State(default=None)

    def init(self):
        self._rng = np.random.default_rng(int(self.seed) if self.seed else None)

    def process(self):
        if self.input_data is None:
            return
            
        shape = self.input_data.shape
        if len(shape) != 2:
            self.output = self.input_data
            return
            
        size_y, size_x = shape
        scale = max(1, int(self.scale))
        octaves = int(self.octaves)
        persistence = float(self.persistence)
        intensity = float(self.intensity)
        
        # Generate base gradients
        def generate_gradients(grid_size):
            angles = self._rng.uniform(0, 2 * np.pi, (grid_size, grid_size))
            return np.cos(angles), np.sin(angles)
        
        def fade(t):
            return t * t * t * (t * (t * 6 - 15) + 10)
        
        def lerp(a, b, t):
            return a + t * (b - a)
        
        def gradient(gx, gy, x, y, grid_sz):
            # Get grid cell coordinates
            x0, y0 = int(x), int(y)
            x1, y1 = x0 + 1, y0 + 1
            
            # Wrap coordinates
            x0, y0 = x0 % grid_sz, y0 % grid_sz
            x1, y1 = x1 % grid_sz, y1 % grid_sz
            
            # Relative position within cell
            sx, sy = x - int(x), y - int(y)
            
            # Fade curves
            u, v = fade(sx), fade(sy)
            
            # Gradients at corners
            g00 = gx[y0, x0], gy[y0, x0]
            g01 = gx[y1, x0], gy[y1, x0]
            g10 = gx[y0, x1], gy[y0, x1]
            g11 = gx[y1, x1], gy[y1, x1]
            
            # Dot products
            n00 = g00[0] * sx + g00[1] * sy
            n01 = g01[0] * sx + g01[1] * (sy - 1)
            n10 = g10[0] * (sx - 1) + g10[1] * sy
            n11 = g11[0] * (sx - 1) + g11[1] * (sy - 1)
            
            # Interpolate
            nx0 = lerp(n00, n10, u)
            nx1 = lerp(n01, n11, u)
            return lerp(nx0, nx1, v)
        
        # Accumulate octaves
        noise = np.zeros(shape, dtype=np.float32)
        amplitude = 1.0
        max_amplitude = 0.0
        freq = scale
        
        for _ in range(octaves):
            gx, gy = generate_gradients(freq + 1)
            
            for i in range(size_y):
                for j in range(size_x):
                    x = j * freq / size_x
                    y = i * freq / size_y
                    noise[i, j] += amplitude * gradient(gx, gy, x, y, freq + 1)
            
            max_amplitude += amplitude
            amplitude *= persistence
            freq *= 2
        
        # Normalize to [0, 1]
        noise = (noise / max_amplitude + 1) / 2
        
        # Apply intensity and add to input
        self.output = (self.input_data + noise * intensity).astype(np.float32)


class NoiseWorley(Node):
    """Add Worley (cellular) noise to input array."""

    input_data = InputPort("Input", np.ndarray)
    output = OutputPort("Output", np.ndarray)
    
    # Properties
    points = Integer("Points", default=8)
    intensity = Range("Intensity", 0.5, 0.0, 1.0, scale="linear")
    seed = Integer("Seed", default=0)
    
    # State
    rng_state = State(default=None)

    def init(self):
        self._rng = np.random.default_rng(int(self.seed) if self.seed else None)

    def process(self):
        if self.input_data is None:
            return
            
        shape = self.input_data.shape
        if len(shape) != 2:
            self.output = self.input_data
            return
            
        size_y, size_x = shape
        num_points = int(self.points)
        intensity = float(self.intensity)
        
        # Generate random points
        px = self._rng.uniform(0, 1, num_points)
        py = self._rng.uniform(0, 1, num_points)
        
        # Create coordinate grid
        y_coords, x_coords = np.mgrid[0:size_y, 0:size_x] / max(size_x, size_y)
        
        # Calculate distance to nearest point
        noise = np.ones(shape, dtype=np.float32) * 2
        
        for i in range(num_points):
            dist = np.sqrt((x_coords - px[i])**2 + (y_coords - py[i])**2)
            noise = np.minimum(noise, dist)
        
        # Normalize
        if noise.max() > 0:
            noise = noise / noise.max()
        
        # Apply intensity and add to input
        self.output = (self.input_data + noise * intensity).astype(np.float32)


class NoiseBlue(Node):
    """Add blue noise to input array."""

    input_data = InputPort("Input", np.ndarray)
    output = OutputPort("Output", np.ndarray)
    
    # Properties
    radius = Range("Min Radius", 0.1, 0.02, 0.5, scale="linear")
    intensity = Range("Intensity", 0.5, 0.0, 1.0, scale="linear")
    seed = Integer("Seed", default=0)
    
    # State
    rng_state = State(default=None)

    def init(self):
        self._rng = np.random.default_rng(int(self.seed) if self.seed else None)

    def process(self):
        if self.input_data is None:
            return
            
        shape = self.input_data.shape
        if len(shape) != 2:
            self.output = self.input_data
            return
            
        size_y, size_x = shape
        min_radius = float(self.radius)
        intensity = float(self.intensity)
        
        # Create empty noise grid
        noise = np.zeros(shape, dtype=np.float32)
        
        # Generate points using dart-throwing with minimum distance
        points = []
        attempts = size_y * size_x // 2
        
        for _ in range(attempts):
            x = self._rng.uniform(0, 1)
            y = self._rng.uniform(0, 1)
            
            # Check minimum distance to existing points
            valid = True
            for px, py in points:
                dist = np.sqrt((x - px)**2 + (y - py)**2)
                if dist < min_radius:
                    valid = False
                    break
            
            if valid:
                points.append((x, y))
        
        # Render points with anti-aliasing
        for px, py in points:
            ix = int(px * size_x)
            iy = int(py * size_y)
            
            # Draw a small dot
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    nx, ny = ix + dx, iy + dy
                    if 0 <= nx < size_x and 0 <= ny < size_y:
                        dist = np.sqrt(dx*dx + dy*dy)
                        noise[ny, nx] = max(noise[ny, nx], 1.0 - dist * 0.5)
        
        # Apply intensity and add to input
        self.output = (self.input_data + noise * intensity).astype(np.float32)


class NoiseSaltPepper(Node):
    """Add salt and pepper noise to input array."""

    input_data = InputPort("Input", np.ndarray)
    output = OutputPort("Output", np.ndarray)
    
    # Properties
    density = Range("Density", 0.05, 0.001, 0.5, scale="log")
    salt_ratio = Range("Salt Ratio", 0.5, 0.0, 1.0, scale="linear")
    seed = Integer("Seed", default=0)
    
    # State
    rng_state = State(default=None)

    def init(self):
        self._rng = np.random.default_rng(int(self.seed) if self.seed else None)

    def process(self):
        if self.input_data is None:
            return
            
        shape = self.input_data.shape
        density = float(self.density)
        salt_ratio = float(self.salt_ratio)
        
        # Copy input
        result = self.input_data.copy()
        
        # Generate random mask for noise positions
        mask = self._rng.uniform(0, 1, shape) < density
        
        # Determine salt vs pepper
        salt_mask = self._rng.uniform(0, 1, shape) < salt_ratio
        
        # Apply salt (white)
        result[mask & salt_mask] = 1.0
        
        # Apply pepper (black)
        result[mask & ~salt_mask] = 0.0
        
        self.output = result.astype(np.float32)
