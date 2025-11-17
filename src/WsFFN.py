# src/WsFFN.py
import math
from dataclasses import dataclass, replace
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Config:
    """Configuration for WsFFN module.

    Args:
        d_model: Model dimension
        d_ffn: FFN hidden dimension (must be divisible by n_head)
        n_head: Number of heads for latent space partitioning
        lambda_z: Weight for L2 regularization loss on z vectors
        lambda_c: Weight for contrastive loss
        lambda_logits_z: Weight for logits z-loss (reserved for future use)
        temperature: Temperature for contrastive loss scaling
        use_aux_loss: Whether to compute auxiliary losses during training
    """

    d_model: int
    d_ffn: int
    n_head: int
    lambda_z: float = 1e-5
    lambda_c: float = 0.005
    lambda_logits_z: float = 1e-4
    temperature: float = 0.07
    use_aux_loss: bool = True

    def for_finetuning(self):
        """Return a new config with auxiliary losses disabled."""
        return replace(self, use_aux_loss=False)

    def for_pretraining(self):
        """Return a new config with auxiliary losses enabled."""
        return replace(self, use_aux_loss=True)


def z_regularization_loss(z_vector: torch.Tensor, lambda_z: float) -> torch.Tensor:
    """Compute L2 regularization loss for wsFFN latent vectors.

    Args:
        z_vector: Latent vectors of shape [B*L*num_heads, z_dim_head]
        lambda_z: Regularization weight

    Returns:
        Scalar loss tensor
    """
    return lambda_z * torch.mean(z_vector.pow(2))


def contrastive_loss_wsffn_batched(
    z_context_flat: torch.Tensor, lambda_c: float, num_heads: int, temperature: float
) -> torch.Tensor:
    """Compute InfoNCE contrastive loss for wsFFN latent vectors.

    The loss encourages diversity across heads by treating each batch×head element
    as a separate sample and maximizing self-similarity relative to other samples.

    Args:
        z_context_flat: Context vectors of shape [B * num_heads, z_dim_head]
        lambda_c: Contrastive loss weight
        num_heads: Number of heads (kept for API compatibility)
        temperature: Temperature scaling factor for similarity scores

    Returns:
        Scalar loss tensor

    Note:
        Uses self-similarity as positive pairs. Negatives are from other heads
        in the batch, which is an intentional design choice for cross-head diversity.
    """
    N = z_context_flat.size(0)  # N = B * num_heads
    if N < 2:
        return torch.tensor(0.0, device=z_context_flat.device)

    # Calculate pairwise cosine similarity matrix: [N, N]
    # where N = batch_size * num_heads
    similarity_matrix = F.cosine_similarity(
        z_context_flat.unsqueeze(1), z_context_flat.unsqueeze(0), dim=2
    )

    # Positive scores: self-similarity (diagonal elements)
    positive_scores = similarity_matrix.diag() / temperature  # [N]

    # Denominator: logsumexp of all pairwise similarities
    all_scores_logsumexp = torch.logsumexp(similarity_matrix / temperature, dim=1)  # [N]

    # InfoNCE loss: -log(exp(pos) / sum(exp(all))) = -(pos - logsumexp(all))
    loss = -(positive_scores - all_scores_logsumexp)  # [N]

    return lambda_c * torch.mean(loss)


class wsFFN(nn.Module):
    """World-Structured Feed-Forward Network with multi-head latent space.

    A drop-in replacement for standard SwiGLU FFN that introduces:
    - Parallelized z-head projection for balanced multi-head structure
    - Auxiliary latent space with regularization and contrastive losses

    Args:
        config: Configuration object containing all hyperparameters

    Raises:
        ValueError: If d_ffn is not divisible by n_head

    Attributes:
        d_ffn: FFN hidden dimension
        num_heads: Number of heads for latent space partitioning
        z_dim_head: Dimension per head (d_ffn // num_heads)
        w1, w2, w3: SwiGLU projection layers
        z_projection: Block-diagonal projection for z-heads
    """

    def __init__(self, config: Config):
        super().__init__()
        d_model = config.d_model
        d_ffn = config.d_ffn
        num_heads = config.n_head

        if d_ffn % num_heads != 0:
            raise ValueError("d_ffn must be divisible by num_heads")

        self.d_ffn = d_ffn
        self.num_heads = num_heads
        self.z_dim_head = d_ffn // num_heads
        self.lambda_z = config.lambda_z
        self.lambda_c = config.lambda_c
        self.temperature = config.temperature

        # SwiGLU Layers (w1, w2, w3)
        self.w1 = nn.Linear(d_model, d_ffn, bias=False)
        self.w2 = nn.Linear(d_ffn, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ffn, bias=False)
        self.silu = nn.SiLU()

        # Z-Head Projection (W_z) - Initialized as Block Diagonal Matrix
        self.z_projection = nn.Linear(d_ffn, d_ffn, bias=False)

        # Store config value for forward pass logic
        self.use_aux_loss = config.use_aux_loss

        # Initialization: Block Diagonal Matrix (each head is independent)
        with torch.no_grad():
            # Zero out the entire weight matrix
            self.z_projection.weight.zero_()
            # Initialize only the blocks for each head using Kaiming
            for i in range(num_heads):
                start_idx = i * self.z_dim_head
                end_idx = (i + 1) * self.z_dim_head
                # Kaiming Initialization
                nn.init.kaiming_uniform_(
                    self.z_projection.weight[start_idx:end_idx, start_idx:end_idx], a=math.sqrt(5)
                )

    def _split_into_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape flat FFN dimension into separate heads.

        Args:
            x: Tensor of shape [B, L, d_ffn]

        Returns:
            Tensor of shape [B, L, num_heads, z_dim_head]
        """
        new_shape = x.size()[:-1] + (self.num_heads, self.z_dim_head)
        return x.view(new_shape)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Merge separate heads back into flat FFN dimension.

        Args:
            x: Tensor of shape [B, L, num_heads, z_dim_head]

        Returns:
            Tensor of shape [B, L, d_ffn]
        """
        return x.contiguous().view(x.size(0), x.size(1), self.d_ffn)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through wsFFN.

        Args:
            x: Input tensor of shape [B, L, d_model]

        Returns:
            Tuple of:
                - output: Transformed tensor of shape [B, L, d_model]
                - aux_loss: Auxiliary loss (L_Z + L_C) if training, else None
        """
        B, L, D_MODEL = x.shape

        # 1. SwiGLU internal expansion
        h_1_flat = self.w1(x)  # [B, L, d_ffn]
        h_2_flat = self.w3(x)  # [B, L, d_ffn]

        # 2. MHA style transformation
        h_1_heads = self._split_into_heads(h_1_flat)  # [B, L, num_heads, z_dim_head]
        h_2_heads = self._split_into_heads(h_2_flat)  # [B, L, num_heads, z_dim_head]

        # 3. SwiGLU operation
        # (W_1(x) ⊙ SiLU(W_3(x)))
        gate_output_heads = self.silu(h_2_heads) * h_1_heads

        # 4. Merge heads and final projection
        gate_output_flat = self._merge_heads(gate_output_heads)
        output = self.w2(gate_output_flat)  # [B, L, d_model]

        # 5. Calculate auxiliary loss during training
        aux_loss = None

        # Loss is calculated only if self.training is True AND use_aux_loss is True
        if self.training and self.use_aux_loss:
            z_heads_base = h_1_heads  # [B, L, num_heads, z_dim_head]

            # Z-Head Projection: True Parallel Processing (W_z)
            z_heads_base_flat = self._merge_heads(z_heads_base)  # [B, L, d_ffn]
            # Project Z vector using the Block Diagonal Matrix W_z
            z_heads_projected_flat = self.z_projection(z_heads_base_flat)  # [B, L, d_ffn]
            z_heads_projected = self._split_into_heads(
                z_heads_projected_flat
            )  # [B, L, num_heads, z_dim_head]

            # Calculate L_Z (Z-Regularization Loss)
            z_flat_all = z_heads_projected.view(-1, self.z_dim_head)  # [B*L*num_heads, z_dim_head]
            l_z_total = z_regularization_loss(z_flat_all, self.lambda_z)

            # Calculate L_C (Contrastive Loss)
            # Average over sequence length L to create context vector z_context
            z_context_all = torch.mean(z_heads_projected, dim=1)  # [B, num_heads, z_dim_head]
            z_context_flat = z_context_all.view(-1, self.z_dim_head)  # [B*num_heads, z_dim_head]

            l_c_total = contrastive_loss_wsffn_batched(
                z_context_flat, self.lambda_c, self.num_heads, self.temperature
            )

            # Sum the total auxiliary loss
            aux_loss = l_z_total + l_c_total

        return output, aux_loss
