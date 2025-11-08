# WsFFN: World-synergistic Feed-Forward Network for LMs

WsFFN is a drop-in replacement for the standard SwiGLU feed-forward network (FFN) used in modern Transformer language models. It introduces two ideas:

- **Parallelized z-head projection** that yields Soft-MoE-like specialization while keeping the compute profile of a dense FFN.
- **An auxiliary latent space** with regularization and contrastive structure to encourage world-model-like representations.

This repository provides a compact PyTorch implementation intended to be embedded inside Transformer blocks.

---

## Key Ideas

- **SwiGLU FFN core**: WsFFN keeps the strong empirical performance of SwiGLU.
- **Head-wise partition** of the FFN hidden dimension $d_{\text{ffn}}$ into $n_{\text{head}}$ groups ("heads"). Each head operates on $$z_{\text{dim-head}} = d_{\text{ffn}} / n_{\text{head}}$$ channels.
- **Grouped, block-diagonal z-projection** that is computed in a single dense matmul for efficiency, yet preserves per-head independence.
- **Auxiliary training losses** on the z-head outputs that are fully batched and parallel, adding negligible overhead.

---

# Acknowledgements
The core idea and architectural design of WsFFN originated from my initial insight into forming and regularizing a point-wise latent space within the FFN. Through a subsequent brainstorming process, this idea was refined and materialized into its current form, which aims to achieve Soft MoE-like functionality using dense computation. The final PyTorch code implementation was executed with high fidelity and efficiency with the assistance of Claude Sonnet and Gemini.

---

## Architecture

Given input $\mathbf{X} \in \mathbb{R}^{B \times L \times d_{\text{model}}}$:

### 1) SwiGLU Expansion

```math
\mathbf{h}_1 = \mathbf{X} \mathbf{W}_1, \quad \mathbf{h}_2 = \mathbf{X} \mathbf{W}_3
```

where $\mathbf{W}_1, \mathbf{W}_3 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ffn}}}$.

```math
\mathbf{g} = \text{SiLU}(\mathbf{h}_2) \odot \mathbf{h}_1
```

### 2) Output Projection

```math
\mathbf{Y} = \mathbf{g} \mathbf{W}_2
```

where $\mathbf{W}_2 \in \mathbb{R}^{d_{\text{ffn}} \times d_{\text{model}}}$.

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

where $\mathbf{W}_z \in \mathbb{R}^{d_{\text{ffn}} \times d_{\text{ffn}}}$ is block-diagonal across heads.

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

**Implementation detail**: We flatten $\mathbf{Z}$ to $\mathbf{Z}_{\text{flat}} \in \mathbb{R}^{(B \cdot L \cdot n_{\text{head}}) \times z_{\text{dim}}}$ and compute:

```math
\mathcal{L}_Z = \lambda_z \cdot \frac{1}{N} \sum_{i=1}^{N} \|\mathbf{Z}_{\text{flat}}[i]\|_2^2
```

### 2) Contrastive InfoNCE on Contextual z

We average $\mathbf{Z}$ over the sequence dimension to obtain a per-example per-head context vector:

```math
\mathbf{z}_{\text{ctx}}[b, h] = \frac{1}{L} \sum_{t=1}^{L} \mathbf{Z}[b, t, h]
```

Flattening over batch and head yields $\mathbf{Z}_{\text{ctx\_flat}} \in \mathbb{R}^{(B \cdot n_{\text{head}}) \times z_{\text{dim}}}$. 

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

**Important**: $d_{\text{ffn}}$ must be divisible by $n_{\text{head}}$; $$z_{\text{dimhead}} = d_{\text{ffn}} / n_{\text{head}}$$.

### Forward Signature
```python
import torch
import torch.nn as nn
from typing import Optional, Any, Tuple
from WsFFN import wsFFN, Config # Assuming wsFFN, Config are available

# --- CONCEPTUAL CLASS: Transformer Block (Wraps wsFFN) ---
# A simplified layer that integrates wsFFN and passes through the training status.
class TransformerLayer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        # Simplified: just the wsFFN part for demonstration
        self.ffn = wsFFN(config)
        self.attn = nn.Identity() 

    # Note: This forward method must match the call in the full model: h, _, aux_loss = layer(h, use_cache=False)
    def forward(self, x: torch.Tensor, use_cache: bool = False) -> Tuple[torch.Tensor, Any, torch.Tensor or None]:
        # Assume self.training is correctly set (via model.train() or passed down from full model)
        output, aux_loss = self.ffn(x) 
        # The full layer would also include Attention and Normalization, but we return a dummy cache slot
        return output, None, aux_loss 

# --- CONCEPTUAL CLASS: Full Model (The one you designed) ---
class Model(nn.Module):
    def __init__(self, config: Config, num_layers: int = 4, vocab_size: int = 32000):
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(vocab_size, config.d_model)
        # Create a stack of TransformerLayers, each containing a wsFFN
        self.layers = nn.ModuleList([TransformerLayer(config) for _ in range(num_layers)])
        self.norm_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, vocab_size, bias=False)

    # Simplified mock method for main loss calculation
    def compute_main_loss(self, logits, labels, objective_id):
        # In a real model, this would be Cross-Entropy or similar
        return torch.randn(1) * 10 # Mock loss

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None,
                objective_id: Optional[torch.Tensor] = None) -> Any:
        
        # Custom training check based on inputs being provided
        is_training = labels is not None and objective_id is not None

        # 1. Embeddings
        h = self.token_embeddings(input_ids)
        
        total_wsffn_aux_loss = torch.tensor(0.0, device=h.device)

        # 2. Iterate and Accumulate wsFFN Loss
        for layer in self.layers:
            # kv_cache is ignored for training loop
            # aux_loss contains L_Z + L_C from the wsFFN in that layer
            h, _, aux_loss = layer(h, use_cache=False) 
            
            if is_training and aux_loss is not None:
                total_wsffn_aux_loss += aux_loss

        # 3. Final Projection
        h = self.norm_f(h)
        logits = self.lm_head(h)

        if is_training:
            # Calculate Main Loss (e.g., cross-entropy)
            main_loss = self.compute_main_loss(logits, labels, objective_id)

            # Calculate Logits Z-Loss (The third component)
            logits_for_z = logits.detach()
            logits_z_loss = self.config.lambda_logits_z * torch.logsumexp(logits_for_z, dim=-1).pow(2).mean()

            # FINAL TOTAL LOSS = Main Loss + wsFFN Aux Loss (L_Z+L_C) + Logits Z-Loss
            total_loss = main_loss + total_wsffn_aux_loss + logits_z_loss

            # Prepare loss dictionary for logging/return
            loss_dict = {
                'total_loss': total_loss.item(),
                'main_loss': main_loss.item(),
                'wsffn_aux_loss': total_wsffn_aux_loss.item(),
                'z_loss': logits_z_loss.item(), 
            }
            
            # Simplified return, ignoring the NaN/Inf handling for clarity
            return total_loss, logits, loss_dict

        return logits


# --- USAGE DEMONSTRATION ---

# Configuration
VOCAB_SIZE = 32000
B, L = 2, 128
cfg = Config(d_model=1024, d_ffn=4096, n_head=8, 
             lambda_z=1e-5, lambda_c=5e-3, lambda_logits_z=1e-4, use_aux_loss=True)

# Instantiate the full model
full_model = Model(cfg)

# 1. TRAINING MODE (All Losses Calculated)

# Note: The model's loss calculation relies on the presence of labels/objective_id
# We must ensure model.train() is called for the internal wsFFN layers
full_model.train()

# Prepare dummy inputs for training
input_ids_train = torch.randint(0, VOCAB_SIZE, (B, L))
labels_train = torch.randint(0, VOCAB_SIZE, (B, L)) # Required to signal training
objective_id_train = torch.ones(B, L) # Required to signal training

print("\n--- Training (Auxiliary Losses Enabled) ---")
total_loss, logits, loss_dict = full_model(
    input_ids=input_ids_train, 
    labels=labels_train, 
    objective_id=objective_id_train
)

print(f"Total Loss (L_Main + L_wsFFN + L_LogitsZ): {total_loss.item():.4f}")
print(f"Loss Components: {loss_dict}")

# 2. INFERENCE MODE (Only Logits Returned)

# Note: While full_model.eval() is good practice, loss is avoided by omitting inputs
full_model.eval()

# Prepare dummy inputs for inference (omit labels/objective_id)
input_ids_eval = torch.randint(0, VOCAB_SIZE, (B, L))

print("\n--- Inference (Loss Calculation Skipped) ---")
with torch.no_grad():
    logits_eval = full_model(input_ids=input_ids_eval)

print(f"Output: Logits tensor of shape {logits_eval.shape}")
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
- $\lambda_{\text{logits\_z}}$ is included in the config for future extensions where a $\mathbf{z} \rightarrow \text{logits}$ head is added; it is unused in the current implementation.
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
  url={https://github.com/vmintf/WsFFN}
}
```

**Key references:**
- **Soft MoE**: Puigcerver, J., et al. (2023). "From Sparse to Soft Mixtures of Experts." arXiv:2308.00951.
- **SwiGLU**: Shazeer, N. (2020). "GLU Variants Improve Transformer." arXiv:2002.05202.
- **Contrastive Learning**: Oord, A., et al. (2018). "Representation Learning with Contrastive Predictive Coding." arXiv:1807.03748.

---

## License

MIT