# WW6: Inhibition-Mediated Data Vacuum (IMDV) - Gated Functional Learning

### The Problem
Softmax-based learning (WW5) handles the 'Dictator Problem' but disrupts activation intensity and forces the system into a simplex space.

### The Solution
A 'Gated Pure Functional' model where learning forces are directly tied to the competition results, but only for templates that are 'active' (have a positive match with the input).

- **Activity Gating**: A template only receives a learning signal if its raw similarity $s_i = w_i \cdot \hat{x}$ is greater than zero. This prevents silent templates from being pushed into the 'dead hemisphere'.
- **Winner's Pull**: Active templates that survive inhibition ($s'_i > 0$) are pulled toward the input.
- **Loser's Push**: Active templates that are suppressed by the population ($\text{inh}_i > 0$) are pushed away from the input.

### Force Formulation (V2)
$F_i = \alpha \cdot (s'_i - \text{inh}_i)$ if $s_i > 0$, else $0$.

### V5: Selective Inhibition
To prevent templates from being driven into the 'dead hemisphere', we gate the inhibition-based push.

- **Force Formulation (V5)**: 
    - If $s_i > 0$ (Active): $F_i = \alpha_{pull} - \alpha_{push} \cdot \text{inh}_i$
    - If $s_i \le 0$ (Inactive): $F_i = \alpha_{pull}$

- **Dynamics**:
    - **Active Templates**: Participate in the competition and can be pulled (winners) or pushed (losers).
    - **Inactive Templates**: Ignore the push and are always pulled toward the data manifold, preventing stagnation and permanent eviction.

### V6: Self-Contained Competition
To reduce graph complexity and decouple learning from activation dynamics, the competition is now calculated internally.

- **Internal Logic**:
    - $s_i = w_i \cdot \hat{x}$ (Cosine Similarity)
    - $a_i = \max(s_i, 0)$ (Rectified Activity)
    - $A = \sum a_j$ (Total Population Drive)
    - $\text{inh}_i = A - a_i$ (Inhibition from others)

- **Force Formulation (V6)**: 
    - If $s_i > 0$ (Active): $F_i = \alpha_{pull} - \alpha_{push} \cdot \text{inh}_i$
    - If $s_i \le 0$ (Inactive): $F_i = \alpha_{pull}$

### V7: Reconstruction-Mediated Repulsion
Pushing away from the raw input $x$ herds templates together toward the South Pole. To separate them, we now push away from the population's collective reconstruction.

- **Mechanism**:
    - **Reconstruction**: $x_{recon} = \sum a_j w_j$ (The weighted sum of all active templates).
    - **Pull Direction**: Toward the raw input $\hat{x}$.
    - **Push Direction**: Away from the normalized reconstruction $\hat{x}_{recon}$.

- **Force Formulation (V7)**: 
    - $\tau_{pull, i} = \alpha_{pull} \cdot \text{geodesic\_tangent}(w_i \to \hat{x})$
    - $\tau_{push, i} = (\alpha_{push} \cdot \text{inh}_i) \cdot \text{geodesic\_tangent}(w_i \to \hat{x}_{recon})$
    - If $s_i > 0$: $\tau_{net, i} = \tau_{pull, i} - \tau_{push, i}$
    - If $s_i \le 0$: $\tau_{net, i} = \tau_{pull, i}$

### V8: Inhibited Reconstruction - Breaking the Resolution Trade-off
- **The Problem**: Linear competition creates a hard trade-off: high repulsion (Beta) leads to functional difference but low resolution, while low repulsion leads to high resolution but redundancy.
- **The Solution**: Use the **Inhibited Activations** ($s'_i$) from the `Hyperpolarization` node to drive the learning.
- **Mechanism**:
    - **Sharper Shadow**: $x_{recon} = \sum s'_j w_j$. The reconstruction is now dominated by the winners, creating a high-resolution "Shadow" to push away from.
    - **Targeted Eviction**: $\text{inh}_{final, i} = \sum_{j \neq i} s'_j$. Losers are pushed away based on the **Winner's success**, not the population's average.
- **Result**: Templates can cluster to capture fine details (High Resolution) while remaining functionally distinct because only one fires at a time (High Sparsity).

### V9: Unified Competition - Synchronizing Activation and Learning
- **The Problem**: Redundant competition logic and separate 'Push' knobs create unnecessary complexity and tuning instability.
- **The Solution**: Unify the 'Brain' (Activation) and the 'Muscle' (Learning). The learning node now directly uses the `activations` ($s'_i$) and `inhibition` ($\text{inh}_i$) signals from the `Hyperpolarization` node.
- **Mechanism**:
    - **Pull**: Driven by the inhibited activation ($s'_i$) plus a constant vacuum floor ($\epsilon$).
    - **Push**: Driven by the inhibition signal ($\text{inh}_i$) received from the population.
    - **Direction**: Pull toward the raw input $\hat{x}$, push away from the inhibited reconstruction $\hat{x}_{recon}$.
- **Result**: A single 'Sparsity' knob (`beta` in the activation node) now controls both the firing patterns and the functional repulsion strength, ensuring perfect consistency.

- **V10: Self-Contained LOO Reconstruction - Breaking the Herding Effect**:
    - **The Problem**: Pushing away from a global reconstruction herds templates together, causing them to collapse into a single point.
    - **The Solution**: Each template now pushes away from the reconstruction of its **competitors only** (Leave-One-Out).
    - **Mechanism**:
        - **Internal Competition**: $s_i = w_i \cdot \hat{x}$, $a_i = \max(s_i, 0)$, $A = \sum a_j$.
        - **LOO Shadow**: $x_{shadow, i} = (\sum a_j w_j) - a_i w_i$. This is the 'Shadow' cast by everyone except template $.
        - **Force Formulation**: 
            - **Pull**: Toward raw input $\hat{x}$ with force $\alpha$.
            - **Push**: Away from normalized shadow $\hat{x}_{shadow, i}$ with force $\alpha \cdot (A - a_i)$.
    - **Result**: Templates push each other **apart** laterally, achieving true functional separation and high-resolution tiling without herding.

- **V11: Balanced LOO Reconstruction - Manual E/I Control**:
    - **The Problem**: In high dimensions, the inhibition signal is often too weak to challenge the pull force, causing templates to herd and collapse even with LOO shadows.
    - **The Solution**: Re-introduce separate `alpha_pull` and `alpha_push` knobs. This allows the researcher to amplify the repulsion gain (e.g., setting `alpha_push` to 5x or 10x the value of `alpha_pull`) to enforce functional separation.
    - **Force Formulation (V11)**: 
        - $\tau_{net, i} = \alpha_{pull} \cdot \hat{\tau}_{pull, i} - (\alpha_{push} \cdot \text{inh}_i) \cdot \hat{\tau}_{push, i}$
    - **Result**: Precise control over the 'Social Distancing' between templates, allowing for high-resolution tiling without collapse.

### Final Report: Success of the Balanced LOO Data Vacuum (V11)
- **Status**: Successful.
- **Observations**: The V11 algorithm has achieved stable, high-resolution manifold tiling. Templates are now functionally diverse and well-separated, effectively breaking the herding effect and template collapse issues identified in earlier versions.
- **Key Factors**:
    - **LOO Reconstruction**: Provided the necessary lateral repulsion to keep templates from merging.
    - **Manual E/I Control**: Allowed for precise balancing of the attraction/repulsion ratio.
    - **Self-Contained Logic**: Simplified the architecture and made the learning dynamics robust to external activation changes.
- **Conclusion**: The `MiniCortex` now has a robust foundation for self-organizing representation learning on high-dimensional manifolds.

### V12: Feedback as Plasticity Modulation, Not Activation Scaling

- **The Problem**: Feedback (e.g., from `IntegerFeedbackEncoder`) was applied as a multiplier to the raw activations ($a_i = a_i \cdot f_i$) before they were used for reconstruction, LOO shadows, inhibition, and force gating. This corrupted the network's internal model of what it was actually seeing:
    - A template with high real similarity to the input but low feedback would have its activation dampened, reducing its contribution to the reconstruction.
    - Other templates' LOO shadows would no longer reflect the true competitive landscape — they wouldn't "see" the dampened template properly and wouldn't get pushed away from its region.
    - The inhibition signal ($A - a_i$) was also deflated, weakening the push force across the board.
    - **Result**: Templates assigned to one class could drift into the region of another class because the push mechanism couldn't accurately detect and repel overlapping representations. Wrong templates specializing on wrong digits.

- **The Solution**: Keep activations raw (unmodified by feedback) for reconstruction, shadows, inhibition, and gating. Apply feedback as a scalar on the **pull tangent** instead:
    - $\tau_{net, i} = (\alpha_{pull} \cdot f_i) \cdot \hat{\tau}_{pull, i} - (\alpha_{push} \cdot \text{inh}_i) \cdot \hat{\tau}_{push, i}$

- **Rationale**: This cleanly separates two concerns:
    - **Perception** (what the network sees): Determined by raw activations. Reconstruction and push forces operate on the true competitive landscape.
    - **Plasticity** (who learns from this input): Modulated by top-down feedback. Templates assigned to the current class feel stronger pull; others feel weaker pull. But the push remains honest — every template is properly repelled from competitors regardless of feedback.

- **Biological Analogy**: In cortex, top-down attention modulates synaptic plasticity (learning rate), not the postsynaptic firing rate itself. The neuron still fires according to its receptive field; attention determines how much that firing drives weight changes.

### V13: Dimension-Invariant Inhibition & Unified Balance Knob

- **The Problem**: The raw inhibition signal $\text{inh}_i = A - a_i$ scales with $n/\sqrt{d}$ (number of templates over square root of dimensionality). This means `alpha_push` values that work for one $(n, d)$ setting don't transfer to another. Additionally, having separate `alpha_pull` and `alpha_push` creates correlated knobs — both affect learning speed, making tuning a 2D search when it should be 1D.

- **The Solution**: Two changes:
    1. **Normalized inhibition**: $\text{inh}_i = (A - a_i) / (A + \epsilon)$. Now $\text{inh}_i \in [0, 1]$ regardless of $n$ or $d$. Winner gets $\text{inh} \approx 0$, losers get $\text{inh} \approx 1$.
    2. **Single balance knob**: Replace `alpha_pull` and `alpha_push` with a single `balance` $\in [0, 1]$ that allocates a total force budget between pull and push:
        - $\text{pull\_weight} = 1 - \text{balance}$
        - $\text{push\_weight} = \text{balance}$

- **Force Formulation (V13)**:
    - $\tau_{net, i} = (1 - \text{balance}) \cdot \hat{\tau}_{pull, i} - (\text{balance} \cdot \text{inh}_i) \cdot \hat{\tau}_{push, i}$
    - With feedback: $(1 - \text{balance}) \cdot f_i \cdot \hat{\tau}_{pull, i} - (\text{balance} \cdot \text{inh}_i) \cdot \hat{\tau}_{push, i}$

- **Parameters** (two orthogonal knobs):
    - `balance` $\in [0, 1]$: E/I character. 0 = pure pull (collapse), 1 = pure push (dispersion). Controls the *type* of learning.
    - `step_fraction` $\in [0, 1]$: Learning speed. Controls *how much* rotation per step.

- **Self-Contained Node**: The `GeodesicIMDVv1` node now manages its own weights internally (initialized as random unit vectors, updated in-place each step). No external `Weights` node needed. Displays weight mosaic (bwr) and activation grid (grayscale) directly.

- **Biological Interpretation**: The normalized inhibition corresponds to *fractional population activity* — "what fraction of the total response comes from other neurons." This is what a biological interneuron naturally computes: it pools activity and broadcasts a normalized signal, not a raw sum. The balance knob corresponds to the E/I ratio set by neuromodulatory tone (e.g., dopamine/GABA balance).

- **What "Best Balance" Means**: The optimal balance is where templates tile the data manifold — not the full sphere. Measurable proxies:
    - **Marginal usage entropy**: over many inputs, each template should "win" roughly equally often.
    - **Low per-input co-activation**: few templates active per input (sparse coding).
    - **Low reconstruction error**: every input is well-approximated by its nearest template.
    - The balance achieving this depends on data geometry but is stable across ambient dimensions thanks to normalized inhibition.
