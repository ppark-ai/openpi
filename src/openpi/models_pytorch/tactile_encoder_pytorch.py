"""TacHand magnetic tactile encoder — PyTorch mirror of tactile_encoder.py.

Same 3 variants (linear / per_taxel / time_series), same shapes, same
public factory `build_tactile_encoder()`. Used by pi0_pytorch.py when
`config.use_tactile=True`.

The JAX reference lives at openpi.models.tactile_encoder. Keep both in
sync — checkpoints converted between frameworks rely on identical layer
order and naming.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn


TactileEncoderVariant = Literal["linear", "per_taxel", "time_series"]

N_TAXELS = 12
N_AXES = 3
TACTILE_DIM = N_TAXELS * N_AXES  # 36


class TactileLinearEncoder(nn.Module):
    """Option A: flatten (12, 3) → Linear → single token of size width."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.proj = nn.Linear(TACTILE_DIM, width)

    def forward(self, tactile: Tensor) -> Tensor:
        if tactile.ndim == 3:
            tactile = tactile.reshape(tactile.shape[0], -1)
        token = self.proj(tactile)
        return token[:, None, :]  # (B, 1, width)


class TactilePerTaxelEncoder(nn.Module):
    """Option B: per-taxel embedding + 1-layer self-attention → 12 tokens."""

    def __init__(self, width: int, n_heads: int = 4) -> None:
        super().__init__()
        self.width = width
        self.n_heads = n_heads
        self.taxel_proj = nn.Linear(N_AXES, width)
        self.pos_emb = nn.Parameter(torch.zeros(1, N_TAXELS, width))
        self.norm = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(
            embed_dim=width,
            num_heads=n_heads,
            batch_first=True,
        )

    def forward(self, tactile: Tensor) -> Tensor:
        if tactile.ndim == 2:
            tactile = tactile.reshape(tactile.shape[0], N_TAXELS, N_AXES)
        x = self.taxel_proj(tactile)            # (B, 12, width)
        x = x + self.pos_emb
        h = self.norm(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        return x + attn_out                     # (B, 12, width)


class TactileTimeSeriesEncoder(nn.Module):
    """Option C: temporal sequence (B, T, 12, 3) → T tokens of size width."""

    def __init__(self, width: int, max_seq_len: int = 100) -> None:
        super().__init__()
        self.width = width
        self.max_seq_len = max_seq_len
        self.frame_proj = nn.Linear(TACTILE_DIM, width)
        self.time_emb = nn.Parameter(torch.zeros(1, max_seq_len, width))

    def forward(self, tactile_seq: Tensor) -> Tensor:
        B, T = tactile_seq.shape[0], tactile_seq.shape[1]
        if T > self.max_seq_len:
            raise ValueError(f"sequence length {T} > max_seq_len {self.max_seq_len}")
        flat = tactile_seq.reshape(B, T, -1)    # (B, T, 36)
        x = self.frame_proj(flat)
        x = x + self.time_emb[:, :T, :]
        return x


def build_tactile_encoder(
    variant: TactileEncoderVariant,
    width: int,
    max_seq_len: int = 100,
) -> nn.Module:
    """Factory mirroring openpi.models.tactile_encoder.build_tactile_encoder.

    Args:
        variant: "linear" / "per_taxel" / "time_series".
        width: Output token width — match pi0 action_expert width.
        max_seq_len: Only used for time_series variant.
    """
    if variant == "linear":
        return TactileLinearEncoder(width=width)
    if variant == "per_taxel":
        return TactilePerTaxelEncoder(width=width)
    if variant == "time_series":
        return TactileTimeSeriesEncoder(width=width, max_seq_len=max_seq_len)
    raise ValueError(f"unknown variant: {variant!r}")
