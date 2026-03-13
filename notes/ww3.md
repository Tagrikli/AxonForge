# WW3: Residual-Based Learning with Inhibition in Input Space

## What Was Tried

Reformulated learning to use residual-based updates:
- Templates activate based on input similarity (using lateral inhibition).
- Reconstruct input from weighted template sum: $\hat{x} = (\sum s_i w_i) \cdot \|x\|$.
- Compute residual $r = x - \hat{x}$.
- Templates learn from the residual direction $\hat{r}$ instead of the raw input.
- Each template's learning is gated by its alignment with the residual: $g_i = \max(w_i \cdot \hat{r}, 0)$.

## What Problem Was Observed

The population bifurcated into two stable but dysfunctional groups. Instead of a "winner-take-all" diversification, the templates split:

1. **The Overshooters (Winners):** Approximately half the templates became "winners" for the data. However, because multiple templates were active simultaneously ($\sum s_i > 1$), the reconstruction $\hat{x}$ overshot the actual input $x$.
2. **The Anti-Templates (Pushed Back):** Because $\hat{x}$ overshot $x$, the residual $r = x - \hat{x}$ flipped direction, pointing exactly opposite to the data. The other half of the templates, seeing this flipped residual, were "pushed back" to the completely opposite pattern (the negative hemisphere).

These "pushed back" templates were not "dead" or "trapped"; they were actively tracking the "anti-data" because the residual-based gating $g_i = \max(w_i \cdot \hat{r}, 0)$ made them sensitive to the flipped error signal. This created a parasitic attractor in the opposite hemisphere.

## Key Insight That Led to Next Iteration

Residual-based learning is extremely sensitive to the global scale of activations. Without strict normalization of the reconstruction or a hard winner-take-all mechanism, the residual can easily flip sign, creating parasitic attractors.

The fundamental issue: Making templates follow the residual alone is insufficient when the residual itself can become misleading due to population-level overshoot. The solution requires a more robust way to ensure manifold coverage, leading to the force-directed competitive learning (attraction/repulsion) in WW4.
