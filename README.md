[![PyPI version](https://badge.fury.io/py/WsFFN.svg)](https://pypi.org/project/WsFFN/)
[![Python](https://img.shields.io/pypi/pyversions/WsFFN)](https://pypi.org/project/WsFFN/)
[![Downloads](https://static.pepy.tech/badge/wsffn)](https://pepy.tech/project/wsffn)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Note:  
> The project is released on PyPI. You can install it with `pip install wsffn --pre`.  
> Warning: This is an alpha version. Please report any issues you encounter.  
> Feedback and contributions welcome!

# WsFFN: World-Structured Feed-Forward Network for LMs

WsFFN is a drop-in replacement for the standard SwiGLU feed-forward network (FFN) used in modern Transformer language models. It introduces two ideas:

- Parallelized z-head projection that yields Balanced Multi-head while keeping the compute profile of a dense FFN.
- An auxiliary latent space with regularization and diversity structure to encourage world-model-like representations.

This repository provides a compact PyTorch implementation intended to be embedded inside Transformer blocks.

---

## Key Ideas

- SwiGLU FFN core: WsFFN keeps the strong empirical performance of SwiGLU.
- Head-wise partition of the FFN hidden dimension d_ffn into n_head groups ("heads"). Each head operates on z_dim_head = d_ffn / n_head channels.
- Grouped, block-diagonal z-projection implemented efficiently as a single grouped 1x1 Conv (Conv1d with groups = n_head), which is mathematically equivalent to per-head linear projections but faster.
- Auxiliary training losses on the z-head outputs that are fully batched and parallel, adding negligible overhead.

---

# Acknowledgements
The core idea and architectural design of WsFFN originated from my initial insight into forming and regularizing a point-wise latent space within the FFN. Through a subsequent brainstorming process, this idea was refined and materialized into its current form, which aims to achieve Balanced Multi-head functionality using dense computation. The final PyTorch code implementation was executed with high fidelity and efficiency with the assistance of Claude Sonnet and Gemini.

---

## Architecture

Given input X in R^{B x L x d_model}:

### 1) SwiGLU Expansion

h1 = X W1,  h3 = X W3,  g = SiLU(h3) ⊙ h1

### 2) Output Projection

y = g W2,  where W2 ∈ R^{d_ffn x d_model}

This recovers the standard SwiGLU FFN.

### 3) Latent z-heads (Parallel)

We split h1 into heads along the hidden dimension:

h1 ∈ R^{B x L x d_ffn} → H1 ∈ R^{B x L x n_head x z_dim_head}

where z_dim_head = d_ffn / n_head.

We apply a grouped 1x1 convolution (Conv1d) with groups = n_head, which is equivalent to a block-diagonal linear map across heads but computed as a single dense kernel for efficiency:

Z_projected = Conv1d(groups=n_head, kernel=1)(h1^T)

Then transpose/reshape back to obtain per-head latent vectors:

Z ∈ R^{B x L x n_head x z_dim_head}

---

## Losses (Auxiliary)

Let z denote the per-token latent vectors after z-head projection.

### 1) L2 Regularization on z (per-head, per-token)

L_Z = lambda_z * E[ ||z||_2^2 ]

Implementation detail: z is flattened across batch, length, and heads, then mean-squared magnitude is computed.

### 2) Orthogonality/Diversity Loss on Contextual Heads

We average Z over the sequence dimension to obtain per-example, per-head context vectors:

z_ctx[b, h] = (1/L) * sum_{t=1..L} Z[b, t, h]

Stack into z_ctx ∈ R^{B x n_head x z_dim_head}. Normalize along the last dimension and compute the Gram matrix per batch (cosine similarities between heads). The target is an identity matrix to encourage head-wise disentanglement:

L_D = lambda_d * MSE(Gram(z_ctx_norm), I)

### Total Auxiliary Loss

L_aux = L_Z + L_D

These losses are fully batched and parallelized, adding minimal per-step latency.

---

## Why Grouped 1x1 Conv for z-Projection?

- Efficiency: A single grouped Conv1d(d_ffn→d_ffn, kernel=1, groups=n_head) achieves high utilization on modern accelerators.
- Independence: Grouping enforces per-head independence (block-diagonal structure) without custom kernels.
- Flexibility: Heads can still adapt weights within their partitions while starting from a clear separation.

---

## PyTorch Module

Class: wsFFN  
Config: d_model, d_ffn, n_head, lambda_z, lambda_d, use_aux_loss

Important: d_ffn must be divisible by n_head; z_dim_head = d_ffn / n_head.

### Forward Signature (minimal usage)

```python
import torch
from WsFFN import wsFFN, Config

B, L = 2, 128
cfg = Config(d_model=1024, d_ffn=4096, n_head=8, lambda_z=1e-4, lambda_d=0.1, use_aux_loss=True)
module = wsFFN(cfg)

x = torch.randn(B, L, cfg.d_model)
module.train()
y, aux_loss = module(x)  # y: [B, L, d_model], aux_loss: scalar (L_Z + L_D)

module.eval()
y_infer, aux_none = module(x)  # aux_none is None when not training or use_aux_loss=False
```

---

## Integration into a Transformer Block

Replace the standard FFN with wsFFN inside your Transformer layer, keeping the same residual structure. For example:

- Pre-LN: x = x + Dropout(wsFFN(LayerNorm(x)))
- Post-LN: x = LayerNorm(x + Dropout(wsFFN(x)))

If you stack multiple layers each returning (hidden, aux_loss), accumulate the auxiliary losses during training:

```python
aux_total = 0.0
for layer in layers:
    h, aux = layer.ffn(h)
    if aux is not None:
        aux_total = aux_total + aux
```

The module is purely feed-forward; there is no change to attention modules.

---

## Training Recipe

- Pretraining: Enable auxiliary losses (model.train()) and add L_aux with weights lambda_z and lambda_d to your main objective.
- Finetuning: Optionally disable auxiliary losses (use config.for_finetuning() or set use_aux_loss=False) to reduce overhead.
- Typical hyperparameters: lambda_z ≈ 1e-4, lambda_d ≈ 0.1.
- d_ffn divisible by n_head; common setting: d_ffn = 4 × d_model, n_head ∈ {4, 8, 16}.

---

## Rationale and Relation to Soft MoE

Soft MoE encourages specialization without discrete routing by creating pressure for different subspaces to model different features. WsFFN achieves a similar effect by:

- Structuring the FFN hidden space into heads.
- Applying per-head projections and diversity pressure across head context vectors via an orthogonality loss.
- Keeping the compute dense and fully parallel, preserving throughput and hardware efficiency.

---

## Limitations and Notes

- The orthogonality loss pushes heads toward decorrelation; you may tune lambda_d to balance specialization and stability. Extremely large values can slow or destabilize training.
- The previous InfoNCE contrastive objective has been replaced by the orthogonality/diversity loss for simplicity and efficiency.
- The z-head projection is implemented with grouped Conv1d (kernel=1), which is equivalent to a block-diagonal linear map but typically faster.

---

## Installation

Requires PyTorch.

```bash
pip install torch
```

Then import the module from this repository or install from PyPI:

```bash
pip install wsffn --pre
```

---

## Citation

If you use WsFFN in your research, please cite:

```bibtex
@misc{wsffn2025,
  title={WsFFN: World-Structured Feed-Forward Network for Language Models},
  author={vmintf},
  year={2025},
  url={https://github.com/vmintf/WsFFN}
}
```

Key references:
- Soft MoE: Puigcerver, J., et al. (2023). "From Sparse to Soft Mixtures of Experts." arXiv:2308.00951.
- SwiGLU: Shazeer, N. (2020). "GLU Variants Improve Transformer." arXiv:2002.05202.
- Orthogonality Regularization: Bjorck, J., et al. (2018). "Understanding Batch Normalization"; and related works exploring orthogonality/whitening losses for representation disentanglement.

---

## License

MIT
