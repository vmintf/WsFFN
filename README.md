# WsFFN: World-synergistic Feed-Forward Network for LMs

WsFFN is a drop-in replacement for the standard SwiGLU feed-forward network (FFN) used in modern Transformer language models. It introduces two ideas:

- **Parallelized z-head projection** that yields Soft-MoE-like specialization while keeping the compute profile of a dense FFN.
- **An auxiliary latent space** with regularization and contrastive structure to encourage world-model-like representations.

This repository provides a compact PyTorch implementation intended to be embedded inside Transformer blocks.

---

## Key Ideas

- **SwiGLU FFN core**: WsFFN keeps the strong empirical performance of SwiGLU.
- **Head-wise partition** of the FFN hidden dimension $d_{\text{ffn}}$ into $n_{\text{head}}$ groups ("heads"). Each head operates on $z_{\text{dim-head}} = d_{\text{ffn}} / n_{\text{head}}$ channels.
- **Grouped, block-diagonal z-projection** that is computed in a single dense matmul for efficiency, yet preserves per-head independence.
- **Auxiliary training losses** on the z-head outputs that are fully batched and parallel, adding negligible overhead.

---

## Architecture

Given input $\mathbf{X} \in \mathbb{R}^{B \times L \times d_{\text{model}}}$:

### 1) SwiGLU Expansion

```math
\mathbf{h}_1 = \mathbf{X} \mathbf{W}_1, \quad \mathbf{h}_2 = \mathbf{X} \mathbf{W}_3
```

where $\mathbf{W}_1, \mathbf{W}_3 \in \mathbb{R}^{d_\text{model} \times d_\text{ffn}}$.

```math
\mathbf{g} = \text{SiLU}(\mathbf{h}_2) \odot \mathbf{h}_1
```

### 2) Output Projection

```math
\mathbf{Y} = \mathbf{g} \mathbf{W}_2
```

where $\mathbf{W}_2 \in \mathbb{R}^{d_\text{ffn} \times d_\text{model}}$.

This recovers the standard SwiGLU FFN.

### 3) Latent z-heads (Parallel)

We split $\mathbf{h}_1$ into heads along the hidden dimension:

```math
\mathbf{h}_1 \in \mathbb{R}^{B \times L \times d_{\text{ffn}}} \rightarrow \mathbf{H}_1 \in \mathbb{R}^{B \times L \times n_{\text{head}} \times z_{\text{dim}}}
```

where $z_{\text{dim}} = d_{\text{ffn}} / n_{\text{head}}$.

We apply a grouped linear projection $\mathbf{Z}: \mathbb{R}^{d_{\text{ffn}}} \rightarrow \mathbb{R}^{d_{\text{ffn}}}$ implemented as a single linear layer with weight initialized block-diagonally, so each head is independent while the operation is computed in one dense GEMM:

```math
\mathbf{Z}_{\text{flat}} = \mathbf{H}_{1,\text{flat}} \cdot \mathbf{W}_z
```

where $\mathbf{W}_z \in \mathbb{R}^{d_\text{ffn} \times d_\text{ffn}}$ is block-diagonal across heads.

We then reshape back to per-head tensors to obtain per-head latent vectors:

```math
\mathbf{Z} \in \mathbb{R}^{B \times L \times n_{\text{head}} \times z_{\text{dim}}}
```

---

## Losses (Auxiliary)

Let $\mathbf{z}$ denote the per-token latent vectors after z-head projection.

### 1) L2 Regularization on z (per-head, per-token)

```math
\mathcal{L}_Z = \lambda_z \cdot \mathbb{E}[\|\mathbf{z}\|_2^2]
```

**Implementation detail**: We flatten $\mathbf{Z}$ to $\mathbf{Z}_\text{flat} \in \mathbb{R}^{(B \cdot L \cdot n_\text{head}) \times z_\text{dim}}$ and compute:

```math
\mathcal{L}_Z = \lambda_z \cdot \frac{1}{N} \sum_{i=1}^{N} \|\mathbf{Z}_{\text{flat}}[i]\|_2^2
```

### 2) Contrastive InfoNCE on Contextual z

We average $\mathbf{Z}$ over the sequence dimension to obtain a per-example per-head context vector:

```math
\mathbf{z}_{\text{ctx}}[b, h] = \frac{1}{L} \sum_{t=1}^{L} \mathbf{Z}[b, t, h]
```

Flattening over batch and head yields $\mathbf{Z}_\text{ctx-flat} \in \mathbb{R}^{(B \cdot n_\text{head}) \times z_\text{dim}}$. 

We compute cosine similarities $s_{ij} = \cos(\mathbf{z}_i, \mathbf{z}_j)$ and the InfoNCE objective with temperature $\tau$:

```math
\mathcal{L}_C = \lambda_c \cdot \mathbb{E}_i \left[ -\log \frac{\exp(s_{ii}/\tau)}{\sum_j \exp(s_{ij}/\tau)} \right]
```

In practice, $s_{ii}$ is the self-similarity of each $\mathbf{z}_i$, so the loss encourages $\mathbf{z}_i$ to be distinct against the batch. This acts as a soft specialization pressure akin to Soft-MoE without routing.

### Total Auxiliary Loss

```math
\mathcal{L}_{\text{aux}} = \mathcal{L}_Z + \mathcal{L}_C
```

These losses are designed to be fully batched and parallelized, adding minimal per-step latency.

---

## Why Block-Diagonal z-Projection?

- **Efficiency**: A single dense `Linear(d_ffn→d_ffn)` achieves high utilization on modern accelerators.
- **Independence**: Initializing with block-diagonal weights assigns disjoint channel groups to different heads, preserving head-wise specialization pressure.
- **Flexibility**: Heads can learn to share information by drifting from the initial block structure if beneficial, but start with clear separation.

---

## PyTorch Module

**Class**: `wsFFN`  
**Config**: `d_model`, `d_ffn`, `n_head`, `λ_z`, `λ_c`, `λ_logits_z` (reserved), `use_aux_loss` toggles losses at call time.

**Important**: $d_{\text{ffn}}$ must be divisible by $n_{\text{head}}$; $z_{\text{dim-head}} = d_{\text{ffn}} / n_{\text{head}}$.

### Forward Signature

```python
Y, aux = wsffn(X, is_training=True)
```

- `X`: shape `[B, L, d_model]`
- `Y`: shape `[B, L, d_model]`
- `aux`: scalar tensor (auxiliary loss) when `is_training=True`; otherwise `None`.

---

## Example Usage

```python
import torch
from src import wsFFN, Config

cfg = Config(d_model=1024, d_ffn=4096, n_head=8, lambda_z=1e-5, lambda_c=5e-3)
ffn = wsFFN(cfg)

x = torch.randn(2, 128, 1024)  # [B, L, d_model]
y, aux = ffn(x, is_training=True)
loss = main_loss + aux  # add auxiliary loss during pretraining
```

During finetuning, you may choose to disable the auxiliary loss at call time:

```python
y, _ = ffn(x, is_training=False)
```

Or toggle it via config:

```python
cfg_finetune = cfg.for_finetuning()  # sets use_aux_loss=False
ffn_finetune = wsFFN(cfg_finetune)
```

---

## Integration into a Transformer Block

Replace the standard FFN with wsFFN inside your Transformer layer, keeping the same residual structure. For example:

- **Pre-LN**: `x = x + Dropout(wsFFN(LayerNorm(x)))`
- **Post-LN**: `x = LayerNorm(x + Dropout(wsFFN(x)))`

The module is purely feed-forward; there is no change to attention modules.

---

## Training Recipe

- **Pretraining**: Enable auxiliary losses (`is_training=True`) and include `aux` in the total loss with weights $\lambda_z$, $\lambda_c$.
- **Finetuning**: Optionally disable auxiliary losses (`is_training=False`) to reduce overhead.
- **Typical hyperparameters**: $\lambda_z \approx 10^{-5}$, $\lambda_c \approx 5 \times 10^{-3}$, $\tau = 0.07$.
- $d_{\text{ffn}}$ divisible by $n_{\text{head}}$; common setting: $d_{\text{ffn}} = 4 \times d_{\text{model}}$, $n_{\text{head}} \in \{4, 8, 16\}$.

---

## Rationale and Relation to Soft MoE

Soft MoE encourages specialization without discrete routing by creating pressure for different subspaces to model different features. WsFFN achieves a similar effect by:

- Structuring the FFN hidden space into heads.
- Applying per-head projections and contrastive pressure across batch×head elements.
- Keeping the compute dense and fully parallel, preserving throughput and hardware efficiency.

---

## Limitations and Notes

- The InfoNCE variant here uses self-similarity as the positive. It is simple and fully parallel, but you may experiment with other positives (e.g., augmentations, multi-view encoders) for stronger semantic structure.
- $\lambda_\text{logits-z}$ is included in the config for future extensions where a $\mathbf{z} \rightarrow \text{logits}$ head is added; it is unused in the current implementation.
- The z-head projection is initialized block-diagonally; training may alter this structure.

---

## Installation

Requires PyTorch.

```bash
pip install torch
```

Then import the module from this repository.

---

## Citation

If you use WsFFN in your research, please cite:

```bibtex
@misc{wsffn2025,
  title={WsFFN: World-Structured Feed-Forward Network for Language Models},
  author={vmintf},
  year={2025},
  url={https://github.com/[vmintf]/wsffn}
}
```

**Key references:**
- **Soft MoE**: Puigcerver, J., et al. (2023). "From Sparse to Soft Mixtures of Experts." arXiv:2308.00951.
- **SwiGLU**: Shazeer, N. (2020). "GLU Variants Improve Transformer." arXiv:2002.05202.
- **Contrastive Learning**: Oord, A., et al. (2018). "Representation Learning with Contrastive Predictive Coding." arXiv:1807.03748.

---

## License

MIT
