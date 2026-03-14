# GeodesicIMDVv2 --- Unified LOO-Inhibited IMDV

## Vision

GeodesicIMDVv2 models a cortical hypercolumn where **a single mechanism---the Leave-One-Out (LOO) shadow---drives both learning and activation sparsity**. In real neocortex, basket cell interneurons create lateral inhibition at the dendrite level: a minicolumn's dendrites are suppressed for input features already "claimed" by other minicolumns. This dendritic suppression has two effects:

1. **Learning (plasticity):** Only non-inhibited synapses undergo plasticity, forcing each minicolumn to specialize in unclaimed features.
2. **Activation (soma output):** The soma fires only if enough non-inhibited dendrites are active---if most of its preferred input is already explained by others, it stays silent.

The LOO shadow produces both effects from one computation. There is no separate inhibition parameter, no k-WTA, no beta---the inhibition topology emerges from the learned weight geometry itself.

---

## Mathematical Formulation

### Setup

- **Weight bank:** $W \in \mathbb{R}^{N \times D}$, where $N$ is the number of templates (minicolumns) and $D$ is the input dimensionality. Each row $\mathbf{w}_i$ is a unit vector on the hypersphere $S^{D-1}$: $\|\mathbf{w}_i\| = 1$.
- **Input:** $\mathbf{x} \in \mathbb{R}^D$, normalized to unit length: $\hat{\mathbf{x}} = \mathbf{x} / \|\mathbf{x}\|$.

### Step 1: Raw Similarities and Activations

$$s_i = \mathbf{w}_i \cdot \hat{\mathbf{x}}$$

$$a_{\text{raw},i} = \max(s_i, 0)$$

$$A = \sum_{j=1}^{N} a_{\text{raw},j}$$

These are standard cosine similarities, rectified. On positive-orthant data (like pixel images), most templates will have positive $s_i$, so $a_{\text{raw}}$ is dense (~89% active for MNIST).

### Step 2: Global Reconstruction

$$\mathbf{x}_{\text{recon}} = \sum_{j=1}^{N} a_{\text{raw},j} \cdot \mathbf{w}_j = \mathbf{a}_{\text{raw}}^\top W$$

This is the population's collective "explanation" of the input.

### Step 3: LOO Shadow (Dendrite-Level Inhibition)

For each template $i$, remove its contribution:

$$\mathbf{x}_{\text{shadow},i} = \mathbf{x}_{\text{recon}} - a_{\text{raw},i} \cdot \mathbf{w}_i = \sum_{j \neq i} a_{\text{raw},j} \cdot \mathbf{w}_j$$

Normalize to the unit sphere:

$$\hat{\mathbf{x}}_{\text{shadow},i} = \frac{\mathbf{x}_{\text{shadow},i}}{\|\mathbf{x}_{\text{shadow},i}\| + \epsilon}$$

**Biological interpretation:** $\mathbf{x}_{\text{shadow},i}$ represents the collective explanation with template $i$ removed---both its direction and strength. The unnormalized shadow drives **activation inhibition** (Step 4), while the normalized shadow $\hat{\mathbf{x}}_{\text{shadow},i}$ provides the **push direction** for learning (see Learning Rule below).

### Step 4: Shadow Overlap (Dendrite-to-Soma Projection)

$$\text{shadow\_overlap}_i = \mathbf{w}_i \cdot \mathbf{x}_{\text{shadow},i}$$

Note: this uses the **unnormalized** shadow vector, not $\hat{\mathbf{x}}_{\text{shadow},i}$. The inhibition therefore scales with the **magnitude** of the collective explanation, not just its direction. A population that barely explains the input inhibits weakly; a population that strongly explains it inhibits strongly.

Expanding:

$$\text{shadow\_overlap}_i = \mathbf{w}_i \cdot \sum_{j \neq i} a_{\text{raw},j} \cdot \mathbf{w}_j = \sum_{j \neq i} a_{\text{raw},j} \cdot (\mathbf{w}_i \cdot \mathbf{w}_j)$$

This is unbounded above (it grows with population size and activation strength):

- **Large positive**: Template $i$'s receptive field is heavily covered by strongly active, similar templates. Strong inhibition.
- **Near zero**: Either few templates are active, or those that are active are orthogonal to $i$. Weak inhibition.
- **Negative**: Template $i$ points away from the collective. Rare in practice.

### Step 5: LOO-Inhibited Activation (Soma Output)

$$a_i = \max(s_i - \text{shadow\_overlap}_i, \, 0) \cdot \|\mathbf{x}\|$$

The activation is the raw cosine similarity minus the strength-weighted redundancy with the collective, scaled by input magnitude. Templates that are redundant with a strong collective get suppressed; templates that capture unique structure---or that dominate a weak collective---survive.

This is the **unified output**---no separate inhibition mechanism.

---

## Learning Rule

The LOO shadow drives both the activation inhibition (above) and the weight update (below).

### Pull Force: Toward the Input

The geodesic tangent vector pointing from $\mathbf{w}_i$ toward $\hat{\mathbf{x}}$ on the unit sphere:

$$\mathbf{v}_{\text{pull}} = \hat{\mathbf{x}} - (\mathbf{w}_i \cdot \hat{\mathbf{x}}) \cdot \mathbf{w}_i$$

$$\hat{\boldsymbol{\tau}}_{\text{pull},i} = \frac{\mathbf{v}_{\text{pull}}}{\|\mathbf{v}_{\text{pull}}\|}$$

**Biological interpretation:** This is the excitatory drive---the input "wants" the template to move toward it.

### Push Force: Away from the LOO Shadow

The geodesic tangent vector pointing from $\mathbf{w}_i$ toward $\hat{\mathbf{x}}_{\text{shadow},i}$:

$$\mathbf{v}_{\text{push}} = \hat{\mathbf{x}}_{\text{shadow},i} - (\mathbf{w}_i \cdot \hat{\mathbf{x}}_{\text{shadow},i}) \cdot \mathbf{w}_i$$

$$\hat{\boldsymbol{\tau}}_{\text{push},i} = \frac{\mathbf{v}_{\text{push}}}{\|\mathbf{v}_{\text{push}}\|}$$

**Biological interpretation:** The inhibited dendrites push the template *away from* what others already cover. This forces specialization.

### Inhibition Scaling

$$\text{inh}_i = \frac{A - a_{\text{raw},i}}{A + \epsilon}$$

This normalizes the inhibition to $[0, 1]$ and is dimension-invariant. Templates with high raw activation (relative to the population) receive less push, preserving their current specialization.

### Net Tangent

For **active** templates ($a_{\text{raw},i} > 0$):

$$\boldsymbol{\tau}_{\text{net},i} = (1 - \beta_{\text{bal}}) \cdot \hat{\boldsymbol{\tau}}_{\text{pull},i} - (\beta_{\text{bal}} \cdot \text{inh}_i) \cdot \hat{\boldsymbol{\tau}}_{\text{push},i}$$

For **inactive** templates ($a_{\text{raw},i} \leq 0$):

$$\boldsymbol{\tau}_{\text{net},i} = (1 - \beta_{\text{bal}}) \cdot \hat{\boldsymbol{\tau}}_{\text{pull},i}$$

Inactive templates receive only pull (toward the input). For templates with zero pull norm and zero activation, a random tangent vector is injected so they can explore.

The **balance** parameter $\beta_{\text{bal}} \in [0, 1]$ controls the excitatory/inhibitory tradeoff:
- $\beta_{\text{bal}} = 0$: Pure pull. All templates converge to the data centroid. No specialization.
- $\beta_{\text{bal}} = 1$: Pure push. Templates flee from each other. No learning from input.
- $\beta_{\text{bal}} = 0.5$: Equal pull and push. Good starting point.

### Geodesic Rotation (Rodrigues' Formula)

Templates are updated by rotating them on the hypersphere along the net tangent direction:

$$\theta_i = \eta \cdot \|\boldsymbol{\tau}_{\text{net},i}\|$$

$$\hat{\mathbf{r}}_i = \frac{\boldsymbol{\tau}_{\text{net},i}}{\|\boldsymbol{\tau}_{\text{net},i}\|}$$

$$\mathbf{w}_i^{\text{new}} = \mathbf{w}_i \cos(\theta_i) + \hat{\mathbf{r}}_i \sin(\theta_i)$$

$$\mathbf{w}_i^{\text{new}} \leftarrow \frac{\mathbf{w}_i^{\text{new}}}{\|\mathbf{w}_i^{\text{new}}\|}$$

Where $\eta$ is the **step fraction** (learning rate). The rotation preserves unit norm by construction (renormalization is a safety measure against floating-point drift).

---

## Relationship to Hyperpolarization

The classical subtractive lateral inhibition (implemented in the `Hyperpolarization` node) is:

$$a'_i = \max\!\Big(s_i - \beta \sum_{j \neq i} a_j, \, 0\Big)$$

Every other template inhibits you equally, weighted only by activation strength.

The LOO shadow inhibition is:

$$a'_i = \max(s_i - \mathbf{w}_i \cdot \mathbf{x}_{\text{shadow},i}, \, 0)$$

Expanding the unnormalized shadow overlap:

$$\mathbf{w}_i \cdot \mathbf{x}_{\text{shadow},i} = \sum_{j \neq i} a_j \cdot (\mathbf{w}_i \cdot \mathbf{w}_j)$$

The inhibition from template $j$ onto template $i$ is $a_j \cdot (\mathbf{w}_i \cdot \mathbf{w}_j)$---the product of $j$'s activation and the **receptive field overlap** between $i$ and $j$. No normalization by shadow magnitude. This is a structured, template-specific generalization of hyperpolarization:

| Property | Hyperpolarization | LOO Shadow Inhibition |
|----------|------------------|-----------------------|
| Inhibition weight from $j$ to $i$ | $\beta \cdot a_j$ (uniform) | $a_j \cdot (\mathbf{w}_i \cdot \mathbf{w}_j)$ |
| Orthogonal templates | Still inhibit each other | Zero mutual inhibition |
| Similar templates | Same inhibition as dissimilar | Strong mutual inhibition |
| Scales with population activity | Yes (via $\sum a_j$) | Yes (via $\sum a_j \cdot \text{overlap}$) |
| Free parameters | $\beta$ (hand-tuned) | None (emerges from weights) |
| Biological analog | Generic basket cell | Receptive-field-specific inhibition |

**Hyperpolarization is the special case where all weight vectors overlap equally** ($\mathbf{w}_i \cdot \mathbf{w}_j = \text{const}$ for all $i \neq j$). In that case, the LOO overlap reduces to $\text{const} \cdot \sum_{j \neq i} a_j$, recovering the classical form with $\beta = \text{const}$.

---

## How Sparsity Emerges

Sparsity emerges naturally:

1. **As templates specialize** (via push forces), their weight vectors become more dissimilar.
2. **Dissimilar templates have low mutual overlap** ($\mathbf{w}_i \cdot \mathbf{w}_j \approx 0$), so they don't contribute to each other's shadow overlap.
3. **But templates that respond to the same input features** have high mutual overlap, so they strongly inhibit each other's output.
4. **Only the best match survives** among a group of similar templates, while templates detecting different features can coexist.

The sparsity level is not a parameter---it's a consequence of the weight geometry. Early in training (random weights, high mutual overlap), activations are dense. As learning progresses and templates spread apart, activations become sparser.

This is the same dynamic as in cortex: sparsity increases with learning as receptive fields become more selective.

---

## Properties of the Algorithm

### Locality

Every computation for template $i$ depends only on:
- The input $\hat{\mathbf{x}}$ (available to all templates via feedforward connections)
- Template $i$'s own weights $\mathbf{w}_i$
- The population reconstruction $\mathbf{x}_{\text{recon}}$ (a single shared vector, computable by a pool of interneurons)

The LOO shadow $\mathbf{x}_{\text{shadow},i} = \mathbf{x}_{\text{recon}} - a_i \cdot \mathbf{w}_i$ only requires the global reconstruction minus self---a local subtraction. No template needs to know any other individual template's weights or activations.

### Online Learning

Each sample is processed independently. Weights are updated immediately after each sample (no batching, no gradient accumulation). This matches the biological constraint that synaptic plasticity operates on a per-stimulus timescale.

### Manifold Preservation

All operations respect the unit hypersphere geometry:
- Tangent vectors are projected to be perpendicular to $\mathbf{w}_i$ (tangent to the sphere)
- Weight updates use Rodrigues' rotation (geodesic transport)
- Renormalization ensures numerical stability

Templates never leave the sphere. Their magnitude is always 1. The input magnitude $\|\mathbf{x}\|$ is factored out and multiplied back into the output activations.

### No Dead Templates

Inactive templates (those in the negative hemisphere relative to the input) receive a pull force toward the input. Combined with random tangent injection for templates with zero pull norm, this ensures all templates eventually find a region of input space to specialize in.

### Feedback Modulation

An optional feedback port allows top-down signals to modulate the pull force per-template. This enables hierarchical plasticity: higher layers can gate which templates learn from the current input. When feedback is absent, all templates receive uniform pull scaling.

---

## Parameters

| Parameter | Default | Role |
|-----------|---------|------|
| `amount` | 25 | Number of templates (minicolumns) |
| `dim` | 784 | Dimensionality of each template |
| `balance` | 0.5 | E/I ratio. 0 = pure pull, 1 = pure push |
| `step_fraction` | 0.1 | Learning rate (geodesic rotation fraction) |
| `is_learning` | True | Enable/disable weight updates |

Note: **there is no sparsity parameter**. Sparsity is an emergent property of the LOO shadow inhibition and the learned weight geometry.

---

## Outputs

| Port | Content |
|------|---------|
| `weights` | Current weight matrix $W$ ($N \times D$) |
| `activations` | LOO-inhibited activations $\mathbf{a} \cdot \|\mathbf{x}\|$ |
| `raw_activations` | Raw cosine activations $\mathbf{a}_{\text{raw}} \cdot \|\mathbf{x}\|$ |

Both activation outputs are scaled by input magnitude to preserve signal strength in downstream processing.

---

## Biological Correspondence

| Algorithm Component | Cortical Analog |
|---------------------|-----------------|
| Template $\mathbf{w}_i$ | Minicolumn's synaptic weight profile |
| Raw similarity $s_i$ | Total dendritic excitation |
| LOO shadow $\mathbf{x}_{\text{shadow},i}$ | Inhibitory signal from basket cell network (strength + direction) |
| Shadow overlap $\mathbf{w}_i \cdot \mathbf{x}_{\text{shadow},i}$ | Strength-weighted dendritic shunting/inhibition |
| LOO-inhibited activation $a_i$ | Soma firing rate after lateral inhibition |
| Pull force | Hebbian LTP (long-term potentiation) |
| Push force | Heterosynaptic LTD driven by inhibitory context |
| Balance parameter | Neuromodulatory E/I tone (e.g., acetylcholine levels) |
| Step fraction | Plasticity rate (NMDA receptor efficacy) |
| Random tangent for inactive | Baseline synaptic noise / spontaneous fluctuations |
| Feedback modulation | Top-down attentional gating of plasticity |

---

## Pseudocode

```
function PROCESS(x, W, balance, step_fraction, is_learning, feedback):
    # ── Setup ──────────────────────────────────────────────────────
    N, D = shape(W)                      # N templates, D dimensions
    x_norm = ||x||
    if x_norm ≈ 0: return                # skip near-zero inputs
    x_hat = x / x_norm                   # unit input direction

    # Ensure all templates live on the unit sphere
    for i in 1..N:
        W[i] = W[i] / ||W[i]||

    # ── Step 1: Raw similarities & activations ─────────────────────
    # Each template's cosine similarity with the input, rectified.
    for i in 1..N:
        s[i] = W[i] · x_hat              # cosine similarity (input is unit)
        a[i] = max(s[i], 0)              # rectified: only positive responses
    A = sum(a)                            # total population activity

    # ── Step 2: Global reconstruction ──────────────────────────────
    # The population's collective "explanation" of the input.
    # Each template contributes its weight vector scaled by its activation.
    x_recon = Σ_j  a[j] * W[j]           # shared vector (one per population)

    # ── Step 3: LOO shadow (per-template) ──────────────────────────
    # For each template, remove its own contribution from the reconstruction.
    # This is what "everyone else" collectively explains.
    for i in 1..N:
        x_shadow[i] = x_recon - a[i] * W[i]

        # Normalized shadow — used only for the push DIRECTION in learning.
        x_shadow_hat[i] = x_shadow[i] / ||x_shadow[i]||

    # ── Step 4: Shadow overlap (activation inhibition) ─────────────
    # How much of template i's preferred direction is already explained
    # by others, weighted by the STRENGTH of their collective response.
    # Uses the UNNORMALIZED shadow — a weak collective inhibits weakly.
    for i in 1..N:
        shadow_overlap[i] = W[i] · x_shadow[i]
        # Equivalently: Σ_{j≠i} a[j] * (W[i] · W[j])

    # ── Step 5: LOO-inhibited activations (output) ─────────────────
    # Raw similarity minus strength-weighted redundancy with the collective.
    for i in 1..N:
        a_inhibited[i] = max(s[i] - shadow_overlap[i], 0) * x_norm

    # Also output raw activations for comparison / downstream use.
    for i in 1..N:
        a_raw[i] = a[i] * x_norm

    # ════════════════════════════════════════════════════════════════
    # LEARNING (weight update) — only if is_learning is True
    # ════════════════════════════════════════════════════════════════
    if not is_learning: return W, a_inhibited, a_raw

    # ── Step 6: Inhibition scaling ─────────────────────────────────
    # Fraction of total activity coming from others (not self).
    # High a[i] relative to A → low inh → less push → preserve specialization.
    for i in 1..N:
        inh[i] = (A - a[i]) / (A + eps)

    # ── Step 7: Geodesic tangent vectors ───────────────────────────
    for i in 1..N:

        # Pull tangent: direction on the sphere from W[i] toward x_hat.
        # This is the excitatory/Hebbian drive.
        v_pull = x_hat - (W[i] · x_hat) * W[i]        # project out radial component
        if ||v_pull|| ≈ 0:
            if a[i] ≤ 0:
                # Dead template: inject random tangent so it can explore.
                v_pull = random_tangent_at(W[i])
            else:
                tau_pull_hat[i] = 0                     # already at the input
        tau_pull_hat[i] = v_pull / ||v_pull||

        # Push tangent: direction on the sphere from W[i] toward x_shadow_hat[i].
        # This is the inhibitory/specialization drive.
        # Uses NORMALIZED shadow (we need direction, not magnitude).
        v_push = x_shadow_hat[i] - (W[i] · x_shadow_hat[i]) * W[i]
        if ||v_push|| ≈ 0:
            tau_push_hat[i] = 0
        else:
            tau_push_hat[i] = v_push / ||v_push||

    # ── Step 8: Net tangent ────────────────────────────────────────
    # Combine pull (toward input) and push (away from collective).
    # Balance controls the E/I tradeoff. Feedback optionally gates pull.
    for i in 1..N:
        pull_scale = (1 - balance)
        if feedback is not None:
            pull_scale = pull_scale * feedback[i]       # top-down gating

        if a[i] > 0:
            # Active: pulled toward input, pushed away from collective.
            tau_net[i] = pull_scale * tau_pull_hat[i]
                       - (balance * inh[i]) * tau_push_hat[i]
        else:
            # Inactive: only pulled toward input (or random exploration).
            tau_net[i] = pull_scale * tau_pull_hat[i]

    # ── Step 9: Geodesic rotation (Rodrigues' formula) ─────────────
    # Rotate each template along its net tangent on the hypersphere.
    for i in 1..N:
        theta = step_fraction * ||tau_net[i]||          # rotation angle
        r_hat = tau_net[i] / ||tau_net[i]||             # rotation axis (tangent dir)

        W[i] = W[i] * cos(theta) + r_hat * sin(theta)  # Rodrigues' rotation
        W[i] = W[i] / ||W[i]||                          # safety renormalize

    return W, a_inhibited, a_raw
```

---

### Some Notes:

Anomaly Detection: Because templates adapt to the "normal" manifold and repel each other to cover it efficiently, an input that yields very low raw similarities across all templates (or requires a massive pull force) can be immediately flagged as an anomaly.

2. The inhibition scaling term inhi=(A−araw,i)/(A+ϵ)\text{inh}_i = (A - a_{\text{raw},i}) / (A + \epsilon)
inhi​=(A−araw,i​)/(A+ϵ) is a heuristic.
This term ensures that a template with a large share of the total activation gets pushed less. The motivation is reasonable (don't destabilize well-specialized templates), but the functional form is somewhat arbitrary. Why this fraction and not, say, a softmax-weighted version, or a rank-based scaling? There's no theoretical derivation — it's a design choice that could meaningfully affect convergence behavior.

3. Scale sensitivity and population size.
The shadow overlap ∑j≠iaj⋅(wi⋅wj)\sum_{j \neq i} a_j \cdot (\mathbf{w}_i \cdot \mathbf{w}_j)
∑j=i​aj​⋅(wi​⋅wj​) grows with NN
N (number of templates). For a large population, even moderate pairwise overlaps accumulate into strong inhibition, which could suppress *all* templates below the ReLU threshold. The algorithm may become overly sparse or even degenerate (all activations zero) for large NN
N and high-dimensional inputs. The document doesn't discuss how the algorithm behaves as NN
N scales — this is a critical gap.


Because it doesn't calculate a global x−xrecon​ error gradient, the algorithm doesn't suffer from the vanishing gradient problem