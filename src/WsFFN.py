# src/WsFFN.py
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass, replace

import math
from typing import Tuple


@dataclass(frozen=True)
class Config:
    d_model: int
    d_ffn: int
    n_head: int
    lambda_z: float = 1e-5
    lambda_c: float = 0.005
    lambda_logits_z: float = 1e-4
    use_aux_loss: bool = True

    def for_finetuning(self):
        return replace(self, use_aux_loss=False)

    def for_pretraining(self):
        return replace(self, use_aux_loss=True)


def z_regularization_loss(z_vector: torch.Tensor, lambda_z: float) -> torch.Tensor:
    """L_Z: L2 norm regularization loss for wsFFN latent vectors (batch operation possible)"""
    # z_vector: [B*L*num_heads, z_dim_head]
    return lambda_z * torch.mean(torch.sum(z_vector ** 2, dim=-1))


def contrastive_loss_wsffn_batched(z_context_flat: torch.Tensor, lambda_c: float,
                                   num_heads: int, batch_size: int,
                                   temperature: float = 0.07) -> torch.Tensor:
    """
    L_C: wsFFN Latent Vector of Contrastive Loss (InfoNCE) - Merging all the heads into one batch
    z_context_flat: [B * num_heads, z_dim_head]
    """
    N = z_context_flat.size(0)  # N = B * num_heads
    if N < 2:
        return torch.tensor(0.0, device=z_context_flat.device)

    # L_C is calculated by comparing all wsFFN head vectors within the batch.

    # 1. Calculate Similarity Matrix: [N, N]
    # N=B*H, B=batch_size, H=num_heads
    # Cosine Similarity: [N, D] -> compare [N, 1, D] and [1, N, D] -> [N, N]
    similarity_matrix = F.cosine_similarity(
        z_context_flat.unsqueeze(1),
        z_context_flat.unsqueeze(0),
        dim=2
    )

    # 2. Set Positive Pair (Similarity with itself)
    positive_scores = similarity_matrix.diag() / temperature  # [N]

    # 3. LogSumExp (Denominator)
    # Denominator: Similarity with all pairs
    all_scores_logsumexp = torch.logsumexp(similarity_matrix / temperature, dim=1)  # [N]

    # 4. Calculate InfoNCE Loss
    # Loss = - log( exp(pos) / sum(exp(all)) ) = - (pos - logsumexp(all))
    loss = - (positive_scores - all_scores_logsumexp)  # [N]

    # Return the average loss over all N elements
    return lambda_c * torch.mean(loss)


class wsFFN(nn.Module):
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
                    self.z_projection.weight[start_idx:end_idx, start_idx:end_idx],
                    a=math.sqrt(5)
                )

    def _split_into_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B, L, D_FFN] -> [B, L, num_heads, z_dim_head]
        new_shape = x.size()[:-1] + (self.num_heads, self.z_dim_head)
        return x.view(new_shape)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B, L, num_heads, z_dim_head] -> [B, L, D_FFN]
        return x.contiguous().view(x.size(0), x.size(1), self.d_ffn)

    def forward(self, x: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor or None]:
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
        aux_loss = None  # Initialize to None to return None if loss is not calculated

        # Loss is calculated only if self.training is True AND use_aux_loss is True
        if self.training and self.use_aux_loss:
            z_heads_base = h_1_heads  # [B, L, num_heads, z_dim_head]

            # --- Z-Head Projection: True Parallel Processing (W_z) ---
            z_heads_base_flat = self._merge_heads(z_heads_base)  # [B, L, d_ffn]
            # Project Z vector using the Block Diagonal Matrix W_z
            z_heads_projected_flat = self.z_projection(z_heads_base_flat)  # [B, L, d_ffn]
            z_heads_projected = self._split_into_heads(z_heads_projected_flat)  # [B, L, num_heads, z_dim_head]

            # --- Calculate L_Z (Z-Regularization Loss) ---
            z_flat_all = z_heads_projected.view(-1, self.z_dim_head)  # [B*L*num_heads, z_dim_head]
            l_z_total = z_regularization_loss(z_flat_all, self.lambda_z)

            # --- Calculate L_C (Contrastive Loss) ---
            # Average over sequence length L to create context vector z_context
            z_context_all = torch.mean(z_heads_projected, dim=1)  # [B, num_heads, z_dim_head]
            z_context_flat = z_context_all.view(-1, self.z_dim_head)  # [B*num_heads, z_dim_head]

            l_c_total = contrastive_loss_wsffn_batched(
                z_context_flat, self.lambda_c, self.num_heads, B
            )

            # Sum the total auxiliary loss
            aux_loss = l_z_total + l_c_total

        return output, aux_loss