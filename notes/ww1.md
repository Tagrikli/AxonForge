# WW1: Activation Function Experiments

## What Was Tried

Started by experimenting with different activation functions for template matching on the unit hypersphere. The goal was to find a mechanism that produces selective, sparse responses to input patterns $\hat{x}$.

Three primary inhibition models were compared:

1. **Hyperpolarization (Subtractive)**:
   $$s_i = \max(w_i \cdot \hat{x} - \beta (A - a_i), 0)$$
   where $a_i = \max(w_i \cdot \hat{x}, 0)$ and $A = \sum a_j$. This implements lateral inhibition where active neurons subtract from their neighbors' drive.

2. **Shunting (Divisive)**:
   $$s_i = \frac{w_i \cdot \hat{x}}{1 + \beta A}$$
   This implements divisive normalization, scaling the gain of the entire population based on total activity. It preserves relative differences but compresses the dynamic range.

3. **Softmax (Exponential)**:
   $$s_i = \frac{\exp(w_i \cdot \hat{x} / T)}{\sum_j \exp(w_j \cdot \hat{x} / T)}$$
   A standard winner-take-all approximation that forces the population onto a probability simplex.

## What Problem Was Observed

- **Hyperpolarization** provided the best control over sparsity. By tuning $\beta$, one could achieve a "hard" winner-take-all or a "soft" k-winners-take-all regime.
- **Softmax** was found to be "intensity blind"—it discards the absolute magnitude of the match ($w_i \cdot \hat{x}$) in favor of relative probability, which is undesirable for detecting "none of the above" scenarios.
- **Shunting** acted like a global gain control but failed to produce the sharp competition needed for feature separation. It kept too many "weak" templates active.

## Key Insight That Led to Next Iteration

The activation function alone is just a filter. To develop a representation, we need **Learning Dynamics**. The next step is to allow templates to move toward the inputs they respond to, creating a feedback loop between activation and weight updates.
