"""Learning nodes for MiniCortex hypercolumn."""

import numpy as np

from axonforge.core.node import Node
from axonforge.core.descriptors.ports import InputPort, OutputPort
from axonforge.core.descriptors.actions import Action
from axonforge.core.descriptors import branch


@branch("Hypercolumn/Learning")
class GeodesicHebbian(Node):
    inputs = InputPort("Input", np.ndarray)
    activations = InputPort("Activations", np.ndarray)
    weights = InputPort("Weights", np.ndarray)
    alpha = InputPort("Alpha", float)

    output = OutputPort("Weights", np.ndarray)

    def process(self):
        if self.inputs is None or self.activations is None or self.weights is None:
            return

        eps = 1e-8
        w = self.weights
        x = self.inputs.ravel()
        s = self.activations.ravel()

        if self.alpha is None:
            return

        x_norm = x / (np.linalg.norm(x) + eps)

        cos_theta = np.clip(w @ x_norm, -1.0, 1.0)
        theta = np.arccos(cos_theta)

        frac = np.clip(self.alpha * s, 0.0, 1.0)
        delta = frac * theta

        sin_theta = np.sin(theta) + eps

        w_new = (
            (np.sin(theta - delta) / sin_theta)[:, None] * w
            + (np.sin(delta) / sin_theta)[:, None] * x_norm[None, :]
        )

        w_new = w_new / (np.linalg.norm(w_new, axis=1, keepdims=True) + eps)
        self.output = w_new


@branch("Hypercolumn/Learning")
class GeodesicHebbianResidual(Node):
    inputs = InputPort("Input", np.ndarray)
    activations = InputPort("Activations", np.ndarray)
    weights = InputPort("Weights", np.ndarray)
    alpha = InputPort("Alpha", float)

    output = OutputPort("Weights", np.ndarray)

    def process(self):
        eps = 1e-8
        w = np.asarray(self.weights, dtype=np.float32)
        x = np.asarray(self.inputs, dtype=np.float32).ravel()
        s = np.asarray(self.activations, dtype=np.float32).ravel()

        if self.alpha is None:
            return

        # keep templates unit-normalized
        w = w / (np.linalg.norm(w, axis=1, keepdims=True) + eps)

        # raw input norm
        x_norm_factor = np.linalg.norm(x) + eps

        # reconstruction from inhibited activations
        # s lives in normalized-input activation space, so decode there first
        recon_hat = w.T @ s                      # (D,)
        recon = recon_hat * x_norm_factor        # back to raw-input space

        # residual in input space
        residual = x - recon
        res_norm = np.linalg.norm(residual)

        if res_norm < eps:
            self.output = w
            return

        # residual direction for learning
        r_hat = residual / res_norm

        # angle from each template to residual direction
        cos_theta = np.clip(w @ r_hat, -1.0, 1.0)
        theta = np.arccos(cos_theta)

        # learning gate is NOT the inhibited activation anymore
        # templates learn based on how aligned they are with the residual
        g = np.maximum(w @ r_hat, 0.0)

        frac = np.clip(self.alpha * g, 0.0, 1.0)
        delta = frac * theta

        sin_theta = np.sin(theta)
        sin_theta = np.where(sin_theta < eps, eps, sin_theta)

        w_new = (
            (np.sin(theta - delta) / sin_theta)[:, None] * w
            + (np.sin(delta) / sin_theta)[:, None] * r_hat[None, :]
        )

        w_new = w_new / (np.linalg.norm(w_new, axis=1, keepdims=True) + eps)
        self.output = w_new


def geodesic_tangent(w_from: np.ndarray, w_to: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Return tangent vector at w_from pointing toward w_to on unit sphere.
    
    The tangent is the component of (w_to - w_from) orthogonal to w_from,
    normalized to unit length.
    
    Args:
        w_from: Unit vector on sphere (starting point)
        w_to: Unit vector on sphere (target point)
        eps: Small constant for numerical stability
        
    Returns:
        Unit tangent vector at w_from pointing toward w_to
    """
    # Project w_to onto tangent space at w_from
    cos_theta = np.dot(w_from, w_to)
    
    # Tangent = component of (w_to - w_from) orthogonal to w_from
    tangent = w_to - cos_theta * w_from
    
    # Normalize
    norm = np.linalg.norm(tangent)
    if norm < eps:
        return np.zeros_like(w_from)
    return tangent / norm


def rodrigues_rotation(w: np.ndarray, axis: np.ndarray, angle: float, eps: float = 1e-8) -> np.ndarray:
    """
    Rotate vector w around axis by angle using Rodrigues' rotation formula.
    
    Assumes w and axis are already normalized unit vectors, and axis is
    orthogonal to w (lies in tangent space at w).
    
    Args:
        w: Unit vector to rotate
        axis: Unit rotation axis (should be orthogonal to w)
        angle: Rotation angle in radians
        eps: Small constant for numerical stability
        
    Returns:
        Rotated unit vector
    """
    # Rodrigues' formula: w_rot = w * cos(angle) + (axis x w) * sin(angle) + axis * (axis·w) * (1 - cos(angle))
    # Since axis is orthogonal to w, axis·w = 0, so the last term vanishes
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    
    # For 1D arrays
    if w.ndim == 1:
        w_rot = w * cos_a + axis * sin_a
    else:
        # For batched 2D arrays
        w_rot = w * cos_a[:, None] + axis * sin_a[:, None]
    
    return w_rot


@branch("Hypercolumn/Learning")
class GeodesicPushPull(Node):
    """
    Force-directed competitive learning on the unit hypersphere.
    
    Combines data-driven attraction with constant repulsion between templates
    to achieve manifold coverage. Templates tile the data manifold like Voronoi
    cells, producing sparse, selective activations.
    
    Input Ports:
        inputs: Input pattern (will be normalized internally)
        weights: Weight matrix (n_templates, D), should be unit vectors
        alpha: Attraction strength - scales how much templates are pulled toward input (default: 1.0)
        beta: Repulsion strength - scales how much templates push each other apart (default: 1.0)
        step_fraction: Learning speed - fraction of angular gap to close per step (default: 0.1)
        
    Output Ports:
        weights: Updated weight matrix (n_templates, D)
        
    Parameter Semantics:
        - alpha (attraction_scale): Controls how strongly templates are attracted to the input.
          Higher values pull templates more aggressively toward data. Default 1.0 gives full
          attraction based on similarity-weighted forces.
          
        - beta (repulsion_scale): Controls how strongly templates repel each other.
          Higher values create more spread-out coverage. Must be balanced against alpha
          to prevent repulsion from dominating. Default 1.0 provides balanced spacing.
          
        - step_fraction: Controls learning speed by determining what fraction of the angular
          gap to close toward the target orientation. Independent of force magnitudes.
          Lower = slower, more stable learning. Higher = faster convergence but risk of jitter.
          Default 0.1 means close 10% of the remaining distance per step.
          
    Tuning Guidance:
        Start with defaults (alpha=1.0, beta=1.0, step_fraction=0.1) and adjust:
        - If templates collapse together: increase beta or decrease alpha
        - If templates don't cover data well: decrease beta or increase step_fraction
        - If learning is unstable/jittery: decrease step_fraction
        - If learning is too slow: increase step_fraction (carefully)
        - Typical ratio: alpha/beta ≈ 1 for balanced attraction/repulsion
    """
    inputs = InputPort("Input", np.ndarray)
    weights = InputPort("Weights", np.ndarray)
    alpha = InputPort("Alpha", float)
    beta = InputPort("Beta", float)
    step_fraction = InputPort("Step Fraction", float)

    output = OutputPort("Weights", np.ndarray)

    def process(self):
        eps = 1e-8
        
        # Get inputs with defaults
        if self.inputs is None or self.weights is None:
            return
            
        alpha = self.alpha if self.alpha is not None else 1.0
        beta = self.beta if self.beta is not None else 1.0
        step_fraction = self.step_fraction if self.step_fraction is not None else 0.1
        
        w = np.asarray(self.weights, dtype=np.float64)
        x = np.asarray(self.inputs, dtype=np.float64).ravel()
        
        n_templates, dim = w.shape
        
        # Normalize input
        x_norm = np.linalg.norm(x)
        if x_norm < eps:
            self.output = w
            return
        x_normalized = x / x_norm
        
        # Ensure weights are unit vectors
        w_norms = np.linalg.norm(w, axis=1, keepdims=True)
        w = w / (w_norms + eps)
        
        # Compute attraction tangents for all templates
        # Attraction weight: a_i = max(s_i, 0) where s_i = w_i · x_normalized
        similarities = w @ x_normalized  # (n_templates,)
        attraction_weights = np.maximum(similarities, 0.0)  # (n_templates,)
        
        # Compute attraction tangent for each template
        # τ_i^attr = a_i * geodesic_tangent(w_i → x_normalized)
        attraction_tangents = np.zeros_like(w)  # (n_templates, dim)
        for i in range(n_templates):
            if attraction_weights[i] > eps:
                tangent = geodesic_tangent(w[i], x_normalized, eps)
                attraction_tangents[i] = attraction_weights[i] * tangent
        
        # Compute repulsion tangents (pairwise between all templates)
        # τ_{i,j}^rep = cosine_ij * geodesic_tangent(w_i → w_j)
        # τ_i^rep = Σ_{j≠i, cosine_ij > 0} τ_{i,j}^rep
        repulsion_tangents = np.zeros_like(w)  # (n_templates, dim)
        
        for i in range(n_templates):
            for j in range(n_templates):
                if i != j:
                    cosine_ij = np.dot(w[i], w[j])
                    if cosine_ij > 0:  # only repel from same-hemisphere templates
                        weight = cosine_ij  # [0, 1], stronger for more similar
                        tangent = geodesic_tangent(w[i], w[j], eps)
                        repulsion_tangents[i] += weight * tangent
        
        # Apply fraction-of-gap semantics: compute target orientation and rotate toward it
        w_new = np.zeros_like(w)
        
        for i in range(n_templates):
            # Compute target orientation from forces
            target_vec = w[i] + alpha * attraction_tangents[i] - beta * repulsion_tangents[i]
            target_norm = np.linalg.norm(target_vec)
            
            if target_norm < eps:
                # No movement - target is at origin (shouldn't happen with unit vectors)
                w_new[i] = w[i]
                continue
            
            # Normalize target to unit sphere
            target_vec = target_vec / target_norm
            
            # Compute angular distance to target
            cos_angle = np.clip(np.dot(w[i], target_vec), -1.0, 1.0)
            angle_to_target = np.arccos(cos_angle)
            
            if angle_to_target < eps:
                # Already at target
                w_new[i] = w[i]
                continue
            
            # Rotate step_fraction of the way toward target
            actual_rotation = step_fraction * angle_to_target
            
            # Get rotation axis (geodesic tangent from current to target)
            rotation_axis = geodesic_tangent(w[i], target_vec, eps)
            
            # Apply rotation
            w_new[i] = rodrigues_rotation(w[i], rotation_axis, actual_rotation, eps)
        
                # Renormalize all templates
        w_new_norms = np.linalg.norm(w_new, axis=1, keepdims=True)
        w_new = w_new / (w_new_norms + eps)
        
        self.output = w_new


@branch("Hypercolumn/Learning")
class GeodesicFunctionalPushPull(Node):
    """
    Functional competitive learning on the unit hypersphere.
    
    Uses inhibited activations and inhibition signals to drive learning,
    moving from structural (weight-space) repulsion to functional 
    (activation-space) repulsion.
    
    Input Ports:
        inputs: Raw input pattern x (np.ndarray)
        activations: Inhibited activations s'_i (np.ndarray)
        inhibition: Inhibition signal inh_i (np.ndarray)
        weights: Current weight matrix W (np.ndarray)
        alpha: Attraction scale (default: 1.0)
        beta: Functional push scale (default: 1.0)
        step_fraction: Learning speed (default: 0.1)
        
    Output Ports:
        output: Updated weight matrix (np.ndarray)
    """
    inputs = InputPort("Input", np.ndarray)
    activations = InputPort("Activations", np.ndarray)
    inhibition = InputPort("Inhibition", np.ndarray)
    weights = InputPort("Weights", np.ndarray)
    alpha = InputPort("Alpha", float)
    beta = InputPort("Beta", float)
    step_fraction = InputPort("Step Fraction", float)

    output = OutputPort("Weights", np.ndarray)

    def process(self):
        eps = 1e-8
        
        if self.inputs is None or self.activations is None or self.weights is None:
            return
            
        alpha = self.alpha if self.alpha is not None else 1.0
        beta = self.beta if self.beta is not None else 1.0
        step_fraction = self.step_fraction if self.step_fraction is not None else 0.1
        
        # Use float64 for internal calculations
        w = np.asarray(self.weights, dtype=np.float64)
        x = np.asarray(self.inputs, dtype=np.float64).ravel()
        s_prime = np.asarray(self.activations, dtype=np.float64).ravel()
        
        # Handle optional inhibition
        if self.inhibition is not None:
            inh = np.asarray(self.inhibition, dtype=np.float64).ravel()
        else:
            inh = np.zeros_like(s_prime)
            
        n_templates, dim = w.shape
        
        # 1. Normalize input x and weights w_i to unit length
        x_norm = np.linalg.norm(x)
        if x_norm < eps:
            self.output = w
            return
        x_hat = x / x_norm
        
        w_norms = np.linalg.norm(w, axis=1, keepdims=True)
        w = w / (w_norms + eps)
        
        w_new = np.zeros_like(w)
        
        for i in range(n_templates):
            # 2. Compute the geodesic tangent tau_hat_i from w_i toward the normalized input x_hat
            tau_hat_i = geodesic_tangent(w[i], x_hat, eps)
            
            # 3. Compute the net force magnitude: F_i = alpha * s'_i - beta * inh_i
            f_i = alpha * s_prime[i] - beta * inh[i]
            
            # 4. The net tangent vector is tau_net = F_i * tau_hat_i
            tau_net = f_i * tau_hat_i
            
            # 5. Compute the target orientation: v_target = normalize(w_i + tau_net)
            v_target = w[i] + tau_net
            v_target_norm = np.linalg.norm(v_target)
            
            if v_target_norm < eps:
                w_new[i] = w[i]
                continue
                
            v_target = v_target / v_target_norm
            
            # 6. Compute the angular distance theta between w_i and v_target
            cos_theta = np.clip(np.dot(w[i], v_target), -1.0, 1.0)
            theta = np.arccos(cos_theta)
            
            if theta < eps:
                w_new[i] = w[i]
                continue
                
            # 7. Rotate w_i by gamma * theta toward v_target using the rodrigues_rotation helper
            # gamma is step_fraction
            actual_rotation = step_fraction * theta
            
            # Rotation axis is the geodesic tangent from current to target
            rotation_axis = geodesic_tangent(w[i], v_target, eps)
            
            w_new[i] = rodrigues_rotation(w[i], rotation_axis, actual_rotation, eps)
            
        # 8. Renormalize the final weights
        w_new_norms = np.linalg.norm(w_new, axis=1, keepdims=True)
        w_new = w_new / (w_new_norms + eps)
        
        self.output = w_new


@branch("Hypercolumn/Learning")
class GeodesicSoftmaxPushPull(Node):
    """
    Soft competitive learning on the unit hypersphere using Softmax probabilities.
    
    Uses Softmax probabilities to drive both attraction and repulsion,
    ensuring manifold coverage without the need for internal state or additive biases.
    
    Input Ports:
        inputs: Raw input pattern x (np.ndarray)
        activations: Softmax probabilities p_i (np.ndarray)
        weights: Current weight matrix W (np.ndarray)
        alpha: Attraction scale (default: 1.0)
        beta: Functional push scale (default: 1.0)
        step_fraction: Learning speed (default: 0.1)
        
    Output Ports:
        output: Updated weight matrix (np.ndarray)
    """
    inputs = InputPort("Input", np.ndarray)
    activations = InputPort("Activations", np.ndarray)
    weights = InputPort("Weights", np.ndarray)
    alpha = InputPort("Alpha", float)
    beta = InputPort("Beta", float)
    step_fraction = InputPort("Step Fraction", float)

    output = OutputPort("Weights", np.ndarray)

    def process(self):
        eps = 1e-8
        
        if self.inputs is None or self.activations is None or self.weights is None:
            return
            
        alpha = self.alpha if self.alpha is not None else 1.0
        beta = self.beta if self.beta is not None else 1.0
        step_fraction = self.step_fraction if self.step_fraction is not None else 0.1
        
        # Use float64 for internal calculations
        w = np.asarray(self.weights, dtype=np.float64)
        x = np.asarray(self.inputs, dtype=np.float64).ravel()
        p = np.asarray(self.activations, dtype=np.float64).ravel()
        
        n_templates, dim = w.shape
        
        # 1. Normalize input x and weights w_i to unit length
        x_norm = np.linalg.norm(x)
        if x_norm < eps:
            self.output = w
            return
        x_hat = x / x_norm
        
        w_norms = np.linalg.norm(w, axis=1, keepdims=True)
        w = w / (w_norms + eps)
        
        w_new = np.zeros_like(w)
        
        for i in range(n_templates):
            # 2. Compute the geodesic tangent tau_hat_i from w_i toward the normalized input x_hat
            tau_hat_i = geodesic_tangent(w[i], x_hat, eps)
            
            # 3. Compute the net force magnitude: F_i = alpha * p_i - beta * (1 - p_i)
            f_i = alpha * p[i] - beta * (1.0 - p[i])
            
            # 4. The net tangent vector is tau_net = F_i * tau_hat_i
            tau_net = f_i * tau_hat_i
            
            # 5. Compute the target orientation: v_target = normalize(w_i + tau_net)
            v_target = w[i] + tau_net
            v_target_norm = np.linalg.norm(v_target)
            
            if v_target_norm < eps:
                w_new[i] = w[i]
                continue
                
            v_target = v_target / v_target_norm
            
            # 6. Compute the angular distance theta between w_i and v_target
            cos_theta = np.clip(np.dot(w[i], v_target), -1.0, 1.0)
            theta = np.arccos(cos_theta)
            
            if theta < eps:
                w_new[i] = w[i]
                continue
                
            # 7. Rotate w_i by gamma * theta toward v_target using the rodrigues_rotation helper
            # gamma is step_fraction
            actual_rotation = step_fraction * theta
            
            # Rotation axis is the geodesic tangent from current to target
            rotation_axis = geodesic_tangent(w[i], v_target, eps)
            
            w_new[i] = rodrigues_rotation(w[i], rotation_axis, actual_rotation, eps)
            
        # 8. Renormalize the final weights
        w_new_norms = np.linalg.norm(w_new, axis=1, keepdims=True)
        w_new = w_new / (w_new_norms + eps)
        
        self.output = w_new

