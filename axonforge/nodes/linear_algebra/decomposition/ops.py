"""Matrix decomposition operations for MiniCortex."""

import numpy as np

from axonforge.core.node import Node
from axonforge.core.descriptors.ports import InputPort, OutputPort
from axonforge.core.descriptors.displays import Text, Plot
from axonforge.core.descriptors import branch


class SVD(Node):
    """Singular Value Decomposition (SVD).
    
    Computes: A = U * S * Vt
    Outputs: U, singular values S, and V transpose.
    """

    input_data = InputPort("Input", np.ndarray)
    output_u = OutputPort("U", np.ndarray)
    output_s = OutputPort("S (singular values)", np.ndarray)
    output_vt = OutputPort("V^T", np.ndarray)
    
    singular_values = Plot(
        "Singular Values",
        style="bar",
        color_positive="#d6e6ff",
        color_negative="#ff8a80",
    )
    info = Text("Info", default="No input")

    def process(self):
        if self.input_data is None:
            return

        try:
            U, s, Vt = np.linalg.svd(self.input_data, full_matrices=False)
            
            self.output_u = U.astype(np.float32)
            self.output_s = s.astype(np.float32)
            self.output_vt = Vt.astype(np.float32)
            self.singular_values = s.astype(np.float32)
            self.info = f"U: {U.shape}, S: {s.shape}, Vt: {Vt.shape}"
        except Exception as e:
            self.info = f"Error: {e}"


class EigenDecomposition(Node):
    """Eigenvalue decomposition for square symmetric matrices.
    
    Computes: A = V * D * V^-1
    Outputs: eigenvalues D (diagonal) and eigenvectors V.
    """

    input_data = InputPort("Input", np.ndarray)
    output_eigenvalues = OutputPort("Eigenvalues", np.ndarray)
    output_eigenvectors = OutputPort("Eigenvectors", np.ndarray)
    
    eigenvalues_display = Plot(
        "Eigenvalues",
        style="bar",
        color_positive="#d6e6ff",
        color_negative="#ff8a80",
    )
    info = Text("Info", default="No input")

    def process(self):
        if self.input_data is None:
            return

        try:
            eigenvalues, eigenvectors = np.linalg.eig(self.input_data)
            
            # For real symmetric matrices, eigenvalues should be real
            if np.iscomplex(eigenvalues).any():
                eigenvalues = eigenvalues.real
            
            self.output_eigenvalues = eigenvalues.astype(np.float32)
            self.output_eigenvectors = eigenvectors.astype(np.float32)
            self.eigenvalues_display = eigenvalues.astype(np.float32)
            self.info = f"Eigenvalues: {eigenvalues.shape}, Eigenvectors: {eigenvectors.shape}"
        except Exception as e:
            self.info = f"Error: {e}"


class QRDecomposition(Node):
    """QR decomposition.
    
    Computes: A = Q * R
    Where Q is orthonormal and R is upper triangular.
    """

    input_data = InputPort("Input", np.ndarray)
    output_q = OutputPort("Q", np.ndarray)
    output_r = OutputPort("R", np.ndarray)
    
    info = Text("Info", default="No input")

    def process(self):
        if self.input_data is None:
            return

        try:
            Q, R = np.linalg.qr(self.input_data)
            
            self.output_q = Q.astype(np.float32)
            self.output_r = R.astype(np.float32)
            self.info = f"Q: {Q.shape}, R: {R.shape}"
        except Exception as e:
            self.info = f"Error: {e}"
