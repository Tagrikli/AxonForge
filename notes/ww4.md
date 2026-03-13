# WW4: GeodesicPushPull - Force-Directed Competitive Learning

## Overview

GeodesicPushPull implements force-directed competitive learning on the unit hypersphere, combining data-driven attraction with constant repulsion between templates. This creates manifold coverage where templates tile the data region like Voronoi cells.

## Mathematical Formulation

For N templates {w_i} on unit sphere S^{D-1} and input x:

### 1. Attraction Force (Data-Driven)

For each template w_i:
- Compute cosine similarity: `s_i = w_i · x_normalized`
- Attraction weight: `a_i = max(s_i, 0)`  (only attract if somewhat aligned)
- Attraction tangent: `τ_i^attr = a_i * geodesic_tangent(w_i → x_normalized)`

### 2. Repulsion Force (Cosine-Weighted)

For each template w_i:
- For each other template w_j (j ≠ i):
  - Compute cosine similarity: `cosine_ij = w_i · w_j`
  - Repulsion weight: `r_ij = max(cosine_ij, 0)` (only repel if in same hemisphere)
  - Repulsion tangent: `τ_{i,j}^rep = r_ij * geodesic_tangent(w_i → w_j)`
- Total repulsion: `τ_i^rep = Σ_{j≠i} τ_{i,j}^rep`

**Why this works**:
- Both attraction and repulsion now use a unified [0, 1] scale based on cosine similarity.
- Templates only repel if they are in the same hemisphere (positive dot product).
- Orthogonal or opposite templates do not compete, allowing antipodal features to coexist without interference.

### 3. Net Update (Three-Parameter Control)

For each template w_i:

1. **Compute target orientation** with scaled forces:
   ```
   target_i = w_i + α * τ_i^attr - β * τ_i^rep
   target_i = target_i / ||target_i||  (normalize to unit sphere)
   ```

2. **Compute angular distance** to target:
   ```
   cos_angle = clip(w_i · target_i, -1.0, 1.0)
   angle_to_target = arccos(cos_angle)
   ```

3. **Rotate step_fraction toward target**:
   ```
   actual_rotation = γ * angle_to_target
   rotation_axis = geodesic_tangent(w_i, target_i)
   w_i^new = rotate(w_i, rotation_axis, actual_rotation)
   ```

4. **Renormalize**: `w_i^new = w_i^new / ||w_i^new||`

**Key insight**: The three parameters are now semantically separated and operate on the same scale:
- `α` (alpha) = attraction strength (how much templates are pulled toward input)
- `β` (beta) = repulsion strength (how much templates push each other apart)
- `γ` (gamma/step_fraction) = learning speed (what fraction of angular gap to close)

This separation allows independent control over "how attracted", "how repelled", and "how fast to learn".

## Geodesic Helpers

### geodesic_tangent(from, to)

```python
def geodesic_tangent(w_from, w_to):
    """Return tangent vector at w_from pointing toward w_to on unit sphere."""
    # Project w_to onto tangent space at w_from
    cos_theta = np.dot(w_from, w_to)
    # Tangent = component of (w_to - w_from) orthogonal to w_from
    tangent = w_to - cos_theta * w_from
    # Normalize
    norm = np.linalg.norm(tangent)
    if norm < 1e-8:
        return np.zeros_like(w_from)
    return tangent / norm
```

### rotate(w, axis, angle)

Uses Rodrigues' rotation formula: `w_rot = w * cos(θ) + axis * sin(θ)` (simplified since axis is orthogonal to w).

## Design Choices and Alternatives Considered

### Attraction: Data-Driven (Weighted by Cosine Similarity)

**Chosen approach**: Templates are attracted to the input proportionally to their current alignment.

- **Rationale**: Templates naturally concentrate on data-dense regions. Highly aligned templates move more toward the input, while poorly aligned templates are barely affected. This prevents dead templates from being pulled into irrelevant directions.
- **Alternative considered**: Equal attraction (all templates attracted equally regardless of alignment)
  - Would lead to uniform sphere coverage regardless of data distribution
  - Loses the ability to allocate more templates to high-variance regions

### Repulsion: Cosine-Weighted

**Chosen approach**: Each template repels from other templates in the same hemisphere, weighted by their cosine similarity.

- **Rationale**: Unified scale with attraction. Prevents templates from collapsing while allowing independent features (orthogonal/opposite) to coexist.
- **Biological Intuition**: Templates only compete if they could activate for the same inputs. If two templates are orthogonal, they represent distinct features and don't need to push each other away.
- **Alternative considered**: Constant repulsion (previous implementation)
  - Created mismatched scales between attraction and repulsion.
  - Forced antipodal templates to repel even when they didn't overlap in input space.

### Update Semantics: Fraction-of-Gap with Separate Parameter Control

**Selected approach**: Fraction-of-gap semantics with three independent parameters
- `alpha`: Controls attraction strength (scales attraction tangent contribution)
- `beta`: Controls repulsion strength (scales repulsion tangent contribution)
- `step_fraction` (gamma): Controls learning speed (fraction of angular gap to close)
- Step size is inherently bounded: max rotation = γ * π (when target is opposite)
- Self-limiting: prevents overshooting and reduces jitter
- More stable convergence behavior
- Parameters are semantically separated: attraction/repulsion balance vs learning speed

**Alternative considered**: Force-magnitude semantics (previous implementation)
- Net force magnitude determined rotation angle: `θ = ||α * τ_attr - β * τ_rep||`
- Problem: Force magnitude can be unbounded, leading to overshooting and oscillation
- Required explicit clamping to π/2 to prevent instability
- Jittery behavior when forces were large
- Single parameter (alpha) controlled both force magnitude AND learning speed

**Why the current approach is better**:
1. **Predictable step sizes**: Maximum rotation is known a priori (γ * π)
2. **No overshooting**: Step size scales with remaining distance, not force magnitude
3. **Reduced jitter**: Bounded rotations prevent oscillation around equilibrium
4. **Independent control**: Can adjust coverage (alpha/beta) and speed (step_fraction) separately
5. **Clearer semantics**: Each parameter has a single, well-defined purpose:
   - alpha = "how attracted"
   - beta = "how repelled"
   - step_fraction = "how fast to learn"

**alpha (attraction_scale)**: 1.0 default
- Controls **how strongly templates are attracted** to the input
- Scales the attraction tangent contribution: `α * τ_i^attr`
- Higher values pull templates more aggressively toward data
- Default 1.0 gives full attraction based on similarity-weighted forces
- Balance against beta to control attraction/repulsion trade-off

**beta (repulsion_scale)**: 1.0 default  
- Controls **how strongly templates repel** each other
- Scales the repulsion tangent contribution: `β * τ_i^rep`
- Now on the same scale as alpha [0, 1]
- Default 1.0 provides balanced spacing
- Ratio α/β ≈ 1 provides good balance for typical use cases

**step_fraction (gamma)**: 0.1 default
- Controls **learning speed** - what fraction of angular gap to close per step
- Self-limiting: maximum rotation per step is γ * π (when target is opposite)
- Independent of force magnitudes (alpha/beta)
- Prevents overshooting and reduces jitter compared to force-magnitude semantics
- Must be small enough to prevent oscillation, large enough to make progress
- Default 0.1 means close 10% of the remaining distance per step

### Self-Limiting Step Size (No Clamping Required)

The fraction-of-gap semantics provides natural step size limitation:
- Maximum rotation per step is γ * π (when target is 180° away)
- With γ=0.1, maximum rotation is ~0.31 radians (~18°)
- No explicit clamping needed - the step size is inherently bounded by step_fraction
- **Why this works**: Step_fraction scales the angular distance, not a force magnitude, preventing the unbounded rotations that caused jitter in the previous force-magnitude approach
- **Independent control**: Can adjust learning speed (step_fraction) without changing attraction/repulsion balance (alpha/beta)

**Edge case handling**: When w_i and target are opposite (cos_angle = -1), the geodesic_tangent() function still produces a valid rotation axis because the tangent component `w_to - cos_theta*w_from = w_to + w_from` is perpendicular to w_from (since w_from·(w_to+w_from) = -1 + 1 = 0).

## Parameter Tuning Guide

### Starting Point

Use the defaults as a baseline:
- **alpha = 1.0** (full attraction strength)
- **beta = 1.0** (full repulsion strength)
- **step_fraction = 0.1** (10% of gap per step)

### Troubleshooting Common Issues

**Templates collapse together (insufficient coverage)**
- Increase beta (more repulsion): try 1.5, 2.0
- Or decrease alpha (less attraction): try 0.8, 0.5

**Templates don't cover data well (poor manifold tiling)**
- Decrease beta (less repulsion): try 0.5, 0.2
- Or increase step_fraction (faster learning): try 0.15, 0.2

**Learning is unstable / templates jitter**
- Decrease step_fraction (slower learning): try 0.05, 0.02
- This is the primary control for stability

**Learning is too slow**
- Increase step_fraction: try 0.15, 0.2
- Increase cautiously - watch for jitter

**Some templates are "dead" (no activation)**
- Increase beta (more repulsion forces spread)
- Or increase step_fraction (faster response to forces)

### Parameter Interaction Guidelines

| Goal | Adjust | Direction |
|------|--------|-----------|
| More coverage | alpha/beta ratio | Decrease (more repulsion) |
| Tighter clustering | alpha/beta ratio | Increase (more attraction) |
| Faster learning | step_fraction | Increase |
| More stability | step_fraction | Decrease |
| Better stability without coverage loss | step_fraction ↓, beta ↑ | Combined |

### General Principles

1. **Tune step_fraction first**: Get stable learning, even if slow
2. **Then adjust alpha/beta balance**: Find good coverage vs clustering trade-off
3. **Typical alpha/beta ratio**: 1:1 works well for many cases
4. **Step_fraction rarely needs to exceed 0.3**: Beyond this, jitter becomes likely
5. **Beta can be comparable to alpha**: Since they use the same cosine-based weighting scale

## Expected Behavior

### Manifold Coverage

Templates tile the data manifold like Voronoi cells:
- Each template "owns" a territory where it is the closest template
- Territory boundaries form natural decision surfaces
- More templates allocated to high-density regions of the data distribution

### Sparse, Selective Activations

- For any input, only nearby templates activate strongly
- Far templates have near-zero activation
- Creates sparse representation useful for downstream tasks
- Natural discretization of continuous input space

### Stability Properties

- No dead templates: repulsion ensures all templates remain in the active region
- No runaway winners: repulsion prevents any single template from dominating
- Dynamic equilibrium: templates continuously adjust but maintain coverage

### Contrast with Hierarchical Decomposition

| Feature | GeodesicPushPull | Hierarchical Decomposition |
|---------|------------------|---------------------------|
| Representation | Distributed, competitive | Sequential, residual-based |
| Template roles | Equal peers | Primary → Secondary → ... |
| Dead units | Prevented by repulsion | Common problem |
| Coverage | Explicit manifold tiling | Implicit through residuals |
| Biological mapping | Cortical columns, competitive inhibition | Feedforward hierarchy |

## Connection to Biological Plausibility

### Local Computations

- Each template only needs: input similarity + pairwise template similarities
- No global optimization or backpropagation required
- O(N²) repulsion is acceptable for N~25-100 (biological column sizes)

### Competitive Dynamics

- Mirrors lateral inhibition in cortex
- Templates compete for "territory" on the input manifold
- Similar to self-organizing maps but on the hypersphere with geodesic updates

### Homeostatic Mechanism

- Repulsion acts as a homeostatic force preventing any region from being over- or under-represented
- Analogous to activity-dependent plasticity and synaptic normalization

## Connection to Task Efficiency

### Unsupervised Feature Learning

- No labels required
- Learns data manifold structure automatically
- Produces useful representations for downstream supervised tasks

### Sparse Coding Benefits

- Sparse activations enable:
  - Better generalization (fewer active units = simpler model)
  - Easier interpretation (each input represented by few templates)
  - Energy efficiency (only compute what you need)
  - Better linear separability in downstream layers

### Adaptability

- Online learning: can adapt to changing data distributions
- Incremental: new templates can be added without retraining
- Robust: graceful degradation if some templates fail

## Implementation Notes

- Vectorized operations where possible for efficiency
- Numerical stability: epsilon handling for zero tangents
- O(N²) repulsion acceptable for small N (~25-100)
- For larger N, consider caching pairwise tangents or using approximate methods (e.g., only repel from K nearest neighbors)
- Three-parameter API: `alpha` (attraction), `beta` (repulsion), `step_fraction` (learning speed)
- Default values: alpha=1.0, beta=0.1, step_fraction=0.1
- Parameters are independent: can tune coverage (alpha/beta) and speed (step_fraction) separately

---
### Final Research Findings & Reflections

**1. The Stability of Instant Adaptation ($\gamma = 1.0$)**
Contrary to typical gradient-based systems, setting the learning rate (step_fraction) to 1.0 proved stable. This is due to the self-correcting nature of the geodesic forces: as a template approaches its target, the tangential force naturally vanishes, preventing overshoot. This allows for 'instant adaptation' to the current input manifold.

**2. The Orthogonality Ceiling & Hemisphere Gating**
The repulsion force is gated by a positive cosine similarity (`if cosine_ij > 0`). This creates an 'Orthogonal Equilibrium' where templates stop pushing each other once they are 90° apart. In high-dimensional spaces (784D), this allows for a diverse set of 25 templates to settle into a stable, mutually indifferent configuration that perfectly tiles the manifold.

**3. Escaping the 'Reconstruction Trap'**
In high dimensions, random templates can often provide a low reconstruction error simply due to the vastness of the space. However, `GeodesicPushPull` avoids this trap by driving learning through **Geometry** rather than **Error**. It forces templates to align with the data manifold, resulting in meaningful feature discovery (recognizable strokes/edges) rather than just mathematical noise fitting.

**4. Generative Capacity as a Side Effect**
High-fidelity reconstruction was achieved without explicit optimization for it. This proves that **Competitive Repulsion is a sufficient proxy for Error Minimization** in unsupervised learning, providing a biologically plausible path to generative representation.

**5. Repulsion Insufficiency & The High-Dimensional Trap**
In high-dimensional spaces (like 784D), becoming orthogonal is 'too easy'. Two templates can be mathematically orthogonal by changing only a few pixel values while still representing the same semantic object (e.g., the digit '2'). Because the repulsion force is gated by positive cosine similarity (`if cosine_ij > 0`), it completely shuts off once orthogonality is reached. This means increasing the repulsion strength ($\beta$) is ineffective once templates are 90° apart, as $\beta \times 0 = 0$. This 'Cheap Orthogonality' can lead to semantic redundancy where multiple templates represent the same feature but stop repelling each other because they've found a high-dimensional 'shadow' to hide in.

---
