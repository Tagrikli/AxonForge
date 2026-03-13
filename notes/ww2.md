# WW2: Rotational Update with Winner-Take-All

## What Was Tried

Implemented the first learning loop using **Geodesic Hebbian Learning** on the unit hypersphere. Templates rotate toward the input pattern $\hat{x}$ along the shortest path (geodesic).

- **Activation**: Hyperpolarization with high $\beta$ to enforce Winner-Take-All (WTA).
- **Update Rule**: Each template $w_i$ rotates toward $\hat{x}$ by an angle $\delta_i$:
  $$\delta_i = \alpha \cdot s_i \cdot \theta_i$$
  where $\theta_i = \arccos(w_i \cdot \hat{x})$ is the angular distance and $s_i$ is the activation.
- **Implementation**: Uses spherical linear interpolation (Slerp):
  $$w_i(t+1) = \frac{\sin(\theta_i - \delta_i)}{\sin \theta_i} w_i(t) + \frac{\sin \delta_i}{\sin \theta_i} \hat{x}$$

## What Problem Was Observed

**Winner-Take-All Collapse (Rich-get-richer)**:
Because the activation $s_i$ is competitive, the template that is initially closest to the input pattern wins the competition ($s_{winner} \approx 1$) and inhibits all others ($s_{others} \approx 0$). 

This creates a runaway positive feedback loop:
1. The winner learns and becomes even more similar to the data.
2. The losers never learn because their $s_i$ is zero, so they never move.
3. Eventually, a single "super-template" learns the most frequent feature in the dataset, while all other templates remain in their initial random state (dead neurons).

## Key Insight That Led to Next Iteration

Standard Hebbian learning with WTA is too greedy. To utilize the entire population, we need a way for templates to "share" the input. 

Instead of templates competing to *respond* to the input, they should compete to **explain** the input. Once a template accounts for a part of the signal, that part should be removed, leaving a **residual** for other templates to learn from. This leads to the Residual-Based Learning in WW3.
