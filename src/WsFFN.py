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
    """L_Z: wsFFN 잠재 벡터의 L2 놈 규제 손실 (배치 연산 가능)"""
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

    # 1. 유사도 행렬 계산: [N, N]
    # N=B*H, B=batch_size, H=num_heads
    # z_context_flat을 [B, H, D]로 재구성하여 배치 간 분리를 명확히 할 수도 있으나,
    # 기존 논문 구현은 일반적으로 모든 N개의 벡터를 하나의 배치로 봅니다.

    # 코사인 유사도: [N, D] -> [N, 1, D]와 [1, N, D]를 비교 -> [N, N]
    similarity_matrix = F.cosine_similarity(
        z_context_flat.unsqueeze(1),
        z_context_flat.unsqueeze(0),
        dim=2
    )

    # 2. Positive 쌍 설정 (자기 자신과의 유사도)
    positive_scores = similarity_matrix.diag() / temperature  # [N]

    # 3. 로그 합산 지수 (LogSumExp)
    # 분모: 모든 쌍과의 유사도
    all_scores_logsumexp = torch.logsumexp(similarity_matrix / temperature, dim=1)  # [N]

    # 4. InfoNCE Loss 계산
    # Loss = - log( exp(pos) / sum(exp(all)) ) = - (pos - logsumexp(all))
    loss = - (positive_scores - all_scores_logsumexp)  # [N]

    # 모든 N개 요소의 평균 손실을 반환
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

        # Z-Head용 투영 - 진짜 병렬 처리 가능하도록 변경
        # 방법 1: Grouped Linear (추천)
        # d_ffn 전체를 한 번에 처리하되, 각 헤드는 독립적으로
        self.z_projection = nn.Linear(d_ffn, d_ffn, bias=False)

        # 초기화: 블록 대각 행렬로 (각 헤드가 독립적)
        with torch.no_grad():
            # 전체를 0으로
            self.z_projection.weight.zero_()
            # 각 헤드의 블록만 초기화
            for i in range(num_heads):
                start_idx = i * self.z_dim_head
                end_idx = (i + 1) * self.z_dim_head
                # Kaiming 초기화
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

    def forward(self, x: torch.Tensor, is_training: bool = True) -> Tuple[torch.Tensor, torch.Tensor or None]:
        B, L, D_MODEL = x.shape

        # 1. SwiGLU 내부 확장
        h_1_flat = self.w1(x)  # [B, L, d_ffn]
        h_2_flat = self.w3(x)  # [B, L, d_ffn]

        # 2. MHA 스타일 변환
        h_1_heads = self._split_into_heads(h_1_flat)  # [B, L, num_heads, z_dim_head]
        h_2_heads = self._split_into_heads(h_2_flat)  # [B, L, num_heads, z_dim_head]

        # 3. SwiGLU 연산
        gate_output_heads = self.silu(h_2_heads) * h_1_heads

        # 4. 헤드 재통합 및 최종 투영
        gate_output_flat = self._merge_heads(gate_output_heads)
        output = self.w2(gate_output_flat)  # [B, L, d_model]

        # 5. 훈련 시 보조 손실 계산 (진짜 병렬화)
        aux_loss = torch.tensor(0.0, device=x.device)
        if is_training:
            z_heads_base = h_1_heads  # [B, L, num_heads, z_dim_head]

            # --- Z-Head 투영: 진짜 병렬 처리 ---
            z_heads_base_flat = self._merge_heads(z_heads_base)  # [B, L, d_ffn]
            z_heads_projected_flat = self.z_projection(z_heads_base_flat)  # [B, L, d_ffn] - 한 번에 처리!
            z_heads_projected = self._split_into_heads(z_heads_projected_flat)  # [B, L, num_heads, z_dim_head]

            # --- L_Z (Z-정규화 손실) 병렬 계산 ---
            z_flat_all = z_heads_projected.view(-1, self.z_dim_head)  # [B*L*num_heads, z_dim_head]
            l_z_total = z_regularization_loss(z_flat_all, self.lambda_z)
            aux_loss += l_z_total

            # --- L_C (Contrastive Loss) 병렬 계산 ---
            z_context_all = torch.mean(z_heads_projected, dim=1)  # [B, num_heads, z_dim_head]
            z_context_flat = z_context_all.view(-1, self.z_dim_head)  # [B*num_heads, z_dim_head]

            l_c_total = contrastive_loss_wsffn_batched(
                z_context_flat, self.lambda_c, self.num_heads, B
            )
            aux_loss += l_c_total

        return output, aux_loss if is_training else None