"""Information Theory nodes for MiniCortex."""

import numpy as np

from axonforge.core.node import Node
from axonforge.core.descriptors.ports import InputPort, OutputPort
from axonforge.core.descriptors.displays import Text, Numeric
from axonforge.core.descriptors import branch


class Entropy(Node):
    """Compute Shannon entropy of the input.
    
    Entropy = -sum(p * log2(p))
    Measures the uncertainty/average information content.
    """

    input_data = InputPort("Input", np.ndarray)
    entropy = OutputPort("Entropy", float)
    
    display = Numeric("Entropy", format=".4f")
    info = Text("Info", default="No input")

    def process(self):
        if self.input_data is None:
            return

        # Flatten and normalize to get probability distribution
        flat = self.input_data.flatten().astype(np.float64)
        
        # Normalize to create probability distribution
        total = np.sum(flat)
        if total == 0:
            self.entropy = 0.0
            self.display = 0.0
            self.info = "Zero input"
            return
        
        p = flat / total
        
        # Filter out zero probabilities to avoid log(0)
        p = p[p > 0]
        
        # Compute entropy
        h = -np.sum(p * np.log2(p + 1e-12))
        
        self.entropy = float(h)
        self.display = float(h)
        self.info = f"Entropy: {h:.4f} bits"


class KLDivergence(Node):
    """Compute KL divergence from distribution Q to P.
    
    KL(P || Q) = sum(P * log(P / Q))
    Measures information gain when using Q instead of P.
    """

    input_p = InputPort("P (target)", np.ndarray)
    input_q = InputPort("Q (approx)", np.ndarray)
    kl_div = OutputPort("KL Divergence", float)
    
    display = Numeric("KL Divergence", format=".4f")
    info = Text("Info", default="No input")

    def process(self):
        if self.input_p is None or self.input_q is None:
            self.info = "Waiting for inputs"
            return

        p = self.input_p.flatten().astype(np.float64)
        q = self.input_q.flatten().astype(np.float64)

        if p.shape != q.shape:
            self.info = "Shape mismatch"
            return

        # Normalize to probability distributions
        p = p / (np.sum(p) + 1e-12)
        q = q / (np.sum(q) + 1e-12)

        # Filter out zero probabilities
        mask = (p > 0) & (q > 0)
        p = p[mask]
        q = q[mask]

        # Compute KL divergence
        kl = np.sum(p * np.log(p / (q + 1e-12) + 1e-12))
        
        self.kl_div = float(kl)
        self.display = float(kl)
        self.info = f"KL(P||Q): {kl:.4f}"


class MutualInformation(Node):
    """Compute mutual information between two variables.
    
    I(X; Y) = H(X) + H(Y) - H(X, Y)
    Measures the dependency between two variables.
    """

    input_x = InputPort("X", np.ndarray)
    input_y = InputPort("Y", np.ndarray)
    mi = OutputPort("Mutual Information", float)
    
    display = Numeric("MI", format=".4f")
    info = Text("Info", default="No input")

    def process(self):
        if self.input_x is None or self.input_y is None:
            self.info = "Waiting for inputs"
            return

        x = self.input_x.flatten().astype(np.float64)
        y = self.input_y.flatten().astype(np.float64)

        if len(x) != len(y):
            self.info = "Size mismatch"
            return

        # Discretize to compute joint distribution
        bins = int(np.sqrt(min(len(x), 1000)))
        
        # Compute histograms
        xy, _, _ = np.histogram2d(x, y, bins=bins)
        x_hist, _ = np.histogram(x, bins=bins)
        y_hist, _ = np.histogram(y, bins=bins)

        # Normalize to probability distributions
        pxy = xy.flatten() / (xy.sum() + 1e-12)
        px = x_hist / (x_hist.sum() + 1e-12)
        py = y_hist / (y_hist.sum() + 1e-12)

        # Compute mutual information
        px_py = np.outer(px, py).flatten()
        mask = (pxy > 0) & (px_py > 0)
        
        mi = np.sum(pxy[mask] * np.log2(pxy[mask] / (px_py[mask] + 1e-12) + 1e-12))
        
        self.mi = float(mi)
        self.display = float(mi)
        self.info = f"Mutual Information: {mi:.4f}"


class CrossEntropy(Node):
    """Compute cross entropy between distributions P and Q.
    
    H(P, Q) = -sum(P * log(Q))
    """

    input_p = InputPort("P (target)", np.ndarray)
    input_q = InputPort("Q (predicted)", np.ndarray)
    cross_entropy = OutputPort("Cross Entropy", float)
    
    display = Numeric("Cross Entropy", format=".4f")
    info = Text("Info", default="No input")

    def process(self):
        if self.input_p is None or self.input_q is None:
            self.info = "Waiting for inputs"
            return

        p = self.input_p.flatten().astype(np.float64)
        q = self.input_q.flatten().astype(np.float64)

        if p.shape != q.shape:
            self.info = "Shape mismatch"
            return

        # Normalize to probability distributions
        p = p / (np.sum(p) + 1e-12)
        q = q / (np.sum(q) + 1e-12)

        # Filter out zero probabilities
        mask = (p > 0) & (q > 0)
        p = p[mask]
        q = q[mask]

        # Compute cross entropy
        ce = -np.sum(p * np.log(q + 1e-12))
        
        self.cross_entropy = float(ce)
        self.display = float(ce)
        self.info = f"Cross Entropy: {ce:.4f}"


class Perplexity(Node):
    """Compute perplexity of a probability distribution.
    
    Perplexity = 2^H(P) where H is entropy
    Measures how well a distribution predicts samples.
    Lower perplexity = better predictions.
    """

    input_data = InputPort("Input", np.ndarray)
    perplexity = OutputPort("Perplexity", float)
    
    display = Numeric("Perplexity", format=".4f")
    info = Text("Info", default="No input")

    def process(self):
        if self.input_data is None:
            return

        flat = self.input_data.flatten().astype(np.float64)
        
        # Normalize to probability distribution
        total = np.sum(flat)
        if total == 0:
            self.perplexity = 1.0
            self.display = 1.0
            self.info = "Zero input"
            return
        
        p = flat / total
        p = p[p > 0]
        
        # Compute entropy
        h = -np.sum(p * np.log2(p + 1e-12))
        
        # Compute perplexity
        perp = 2 ** h
        
        self.perplexity = float(perp)
        self.display = float(perp)
        self.info = f"Perplexity: {perp:.4f}"


class Uniformity(Node):
    """Compute uniformity (1 - normalized entropy).
    
    0 = highly non-uniform (concentrated)
    1 = perfectly uniform
    """

    input_data = InputPort("Input", np.ndarray)
    uniformity = OutputPort("Uniformity", float)
    
    display = Numeric("Uniformity", format=".4f")
    info = Text("Info", default="No input")

    def process(self):
        if self.input_data is None:
            return

        flat = self.input_data.flatten().astype(np.float64)
        
        # Normalize to L2
        norm = np.linalg.norm(flat)
        if norm > 0:
            v = flat / norm
        else:
            self.uniformity = 0.0
            self.display = 0.0
            self.info = "Zero vector"
            return
        
        # Compute probability distribution
        p = v ** 2
        
        # Compute entropy
        entropy = -np.sum(p * np.log(p + 1e-9))
        
        # Max entropy for this size
        max_entropy = np.log(len(v))
        
        # Compute non-uniformity and uniformity
        non_uniformity = 1.0 - entropy / max_entropy if max_entropy > 0 else 0.0
        uniformity = 1.0 - non_uniformity
        
        self.uniformity = float(uniformity)
        self.display = float(uniformity)
        self.info = f"Uniformity: {uniformity:.4f}"
