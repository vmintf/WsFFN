# src/WsFFN.py
import math
from dataclasses import dataclass, replace
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Config:
    """
    Configuration for WsFFN module.

    Args:
        d_model: Model dimension
        d_ffn: FFN hidden dimension (must be divisible by n_head)
        n_head: Number of heads for latent space partitioning
        lambda_z: Weight for L2 regularization loss on z vectors
        lambda_d: Weight for orthogonality (diversity) loss
        use_aux_loss: Whether to compute auxiliary losses during training
    """

    d_model: int
    d_ffn: int
    n_head: int
    lambda_z: float = 1e-4
    lambda_d: float = 0.1
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
        z_vector: Latent vectors of shape [..., z_dim_head]
        lambda_z: Regularization weight

    Returns:
        Scalar loss tensor
    """
    return lambda_z * torch.mean(z_vector.pow(2))


def orthogonality_diversity_loss(z_heads: torch.Tensor, lambda_d: float) -> torch.Tensor:
    """Compute Orthogonality Loss to encourage head specialization.

    This loss minimizes the cosine similarity between different latent heads
    within the same sample, encouraging them to capture distinct features.
    The target is for the Gram matrix of heads to resemble an Identity matrix.

    Args:
        z_heads: Context vectors of shape [Batch, num_heads, head_dim]
                 (usually averaged over the sequence length)
        lambda_d: Diversity loss weight

    Returns:
        Scalar loss tensor
    """
    # 1. Normalize vectors along the last dimension for cosine similarity
    # Shape: [B, H, D]
    z_norm = F.normalize(z_heads, p=2, dim=-1)

    # 2. Compute pairwise cosine similarity matrix (Gram Matrix)
    # [B, H, D] @ [B, D, H] -> [B, H, H]
    # (i, j) element represents the similarity between head i and head j
    gram_matrix = torch.bmm(z_norm, z_norm.transpose(1, 2))

    # 3. Create Target Identity Matrix
    # We want diagonal elements to be 1 (self-similarity) and off-diagonals to be 0 (orthogonality)
    num_heads = z_heads.size(1)
    identity = torch.eye(num_heads, device=z_heads.device).unsqueeze(0)  # [1, H, H]

    # 4. Compute Mean Squared Error against Identity Matrix
    # This penalizes non-zero off-diagonal elements
    loss = torch.mean((gram_matrix - identity) ** 2)

    return lambda_d * loss


class wsFFN(nn.Module):
    """World-Structured Feed-Forward Network with multi-head latent space.

    A drop-in replacement for standard SwiGLU FFN that introduces:
    - Optimized Grouped Conv1d for block-diagonal latent projection
    - Orthogonality loss to enforce feature disentanglement across heads

    Args:
        config: Configuration object containing all hyperparameters

    Raises:
        ValueError: If d_ffn is not divisible by n_head

    Attributes:
        z_projection: Grouped Conv1d layer acting as block-diagonal linear projections
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        if config.d_ffn % config.n_head != 0:
            raise ValueError("d_ffn must be divisible by n_head")

        self.d_ffn = config.d_ffn
        self.n_head = config.n_head
        self.z_dim_head = config.d_ffn // config.n_head

        # SwiGLU Layers (w1, w2, w3)
        self.w1 = nn.Linear(config.d_model, config.d_ffn, bias=False)
        self.w2 = nn.Linear(config.d_ffn, config.d_model, bias=False)
        self.w3 = nn.Linear(config.d_model, config.d_ffn, bias=False)
        self.silu = nn.SiLU()

        # Optimized Z-Head Projection (W_z)
        # Using Conv1d with groups=n_head is mathematically equivalent to a
        # block-diagonal linear layer but significantly more efficient.
        # kernel_size=1 ensures it acts as a point-wise projection (like Linear).
        self.z_projection = nn.Conv1d(
            in_channels=config.d_ffn,
            out_channels=config.d_ffn,
            kernel_size=1,
            groups=config.n_head,  # Key: Independent computation per head
            bias=False,
        )

        # Initialize z_projection
        # Since structure is handled by 'groups', we can simply use Kaiming init
        nn.init.kaiming_uniform_(self.z_projection.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through wsFFN.

        Args:
            x: Input tensor of shape [B, L, d_model]

        Returns:
            Tuple of:
                - output: Transformed tensor of shape [B, L, d_model]
                - aux_loss: Scalar auxiliary loss (L_Z + L_D) if training, else None
        """
        # 1. Standard SwiGLU Operation
        h1 = self.w1(x)  # [B, L, d_ffn]
        h3 = self.w3(x)  # [B, L, d_ffn]

        # Gate logic: (SiLU(W_3(x)) * W_1(x))
        gate_output = self.silu(h3) * h1
        output = self.w2(gate_output)  # [B, L, d_model]

        # 2. Calculate Auxiliary Loss (Training Only)
        aux_loss = None

        if self.training and self.config.use_aux_loss:
            # Prepare input for Conv1d: [B, L, D] -> [B, D, L]
            # Conv1d expects channels (D) as the second dimension.
            z_input = h1.transpose(1, 2)

            # Project Z vector using optimized Grouped Conv1d
            # Output shape: [B, D, L]
            z_projected = self.z_projection(z_input)

            # Reshape for loss calculation:
            # 1. Transpose back: [B, D, L] -> [B, L, D]
            # 2. View as heads: [B, L, n_head, z_dim_head]
            z_heads = z_projected.transpose(1, 2).view(
                x.size(0), x.size(1), self.n_head, self.z_dim_head
            )

            # Calculate L_Z (Regularization Loss)
            # Flatten all dimensions except feature dim
            l_z = z_regularization_loss(z_heads, self.config.lambda_z)

            # Calculate L_D (Orthogonality/Diversity Loss)
            # Average over sequence length L to create a context vector per head
            # [B, L, H, D_h] -> [B, H, D_h]
            z_context = z_heads.mean(dim=1)

            l_d = orthogonality_diversity_loss(z_context, self.config.lambda_d)

            # Sum total auxiliary loss
            aux_loss = l_z + l_d

        return output, aux_loss