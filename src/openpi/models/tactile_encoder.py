"""TacHand magnetic tactile encoder for pi0 fork (TacHand-VLA).

Three encoder variants for ablation:

  Option A — LinearProjection: simplest, single token from flattened (12,3).
  Option B — PerTaxelTransformer: 12 tokens (one per taxel), self-attention.
  Option C — TimeSeriesEncoder: temporal sequence (T, 12, 3) -> T tokens.

All variants output tokens in pi0's action_expert width so they can be
appended to the suffix sequence without shape changes.

Insertion point in pi0.py:
  embed_suffix() — alongside state_proj, add tactile_proj.

Usage:
    from openpi.models.tactile_encoder import TactileLinearEncoder

    enc = TactileLinearEncoder(width=action_expert_width, rngs=rngs)
    tactile_token = enc(obs.tactile)  # (B, 1, width)

Then in embed_suffix:
    tokens.append(tactile_token)
    input_mask.append(jnp.ones((B, 1), dtype=jnp.bool_))
    ar_mask += [True]
"""

from __future__ import annotations

from typing import Literal

import flax.nnx as nnx
import jax.numpy as jnp


TactileEncoderVariant = Literal["linear", "per_taxel", "time_series"]

# TacHand glove constants
N_TAXELS = 12
N_AXES = 3
TACTILE_DIM = N_TAXELS * N_AXES  # 36


class TactileLinearEncoder(nnx.Module):
    """Option A: flatten (12, 3) -> (36,) -> Linear -> single token of size width.

    Simplest possible encoder. Treats tactile as global state vector.
    Best baseline; minimal added parameters.
    """

    def __init__(self, width: int, rngs: nnx.Rngs) -> None:
        self.width = width
        self.proj = nnx.Linear(TACTILE_DIM, width, rngs=rngs)

    def __call__(self, tactile: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            tactile: (B, 12, 3) float32 magnetic readings.

        Returns:
            (B, 1, width) — single tactile token per batch element.
        """
        # Accept both (B, 12, 3) and (B, 36)
        if tactile.ndim == 3:
            tactile = tactile.reshape(tactile.shape[0], -1)
        token = self.proj(tactile)
        return token[:, None, :]  # (B, 1, width)


class TactilePerTaxelEncoder(nnx.Module):
    """Option B: per-taxel embedding then self-attention -> 12 tokens.

    Preserves per-finger / per-knuckle spatial structure. Lets the action
    expert attend to specific taxels (e.g., "thumb tip pressure").

    Output shape: (B, 12, width) — 12 tokens, one per taxel.
    """

    def __init__(
        self,
        width: int,
        n_heads: int = 4,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        assert rngs is not None
        self.width = width
        self.n_heads = n_heads
        # Per-taxel embedding: (3,) -> (width,)
        self.taxel_proj = nnx.Linear(N_AXES, width, rngs=rngs)
        # Learnable positional embedding for the 12 taxels
        self.pos_emb = nnx.Param(
            jnp.zeros((1, N_TAXELS, width), dtype=jnp.float32),
        )
        # Self-attention across taxels (1 layer)
        self.attn = nnx.MultiHeadAttention(
            num_heads=n_heads,
            in_features=width,
            qkv_features=width,
            out_features=width,
            rngs=rngs,
        )
        self.norm = nnx.LayerNorm(num_features=width, rngs=rngs)

    def __call__(self, tactile: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            tactile: (B, 12, 3).

        Returns:
            (B, 12, width).
        """
        if tactile.ndim == 2:
            tactile = tactile.reshape(tactile.shape[0], N_TAXELS, N_AXES)
        x = self.taxel_proj(tactile)            # (B, 12, width)
        x = x + self.pos_emb                    # broadcast positional
        x = x + self.attn(self.norm(x))         # residual self-attn
        return x


class TactileTimeSeriesEncoder(nnx.Module):
    """Option C: temporal sequence (T, 12, 3) -> T tokens.

    Captures grip dynamics: contact buildup, slip, hold quality.
    Most expressive but most expensive.

    Output shape: (B, T, width).
    """

    def __init__(
        self,
        width: int,
        max_seq_len: int = 100,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        assert rngs is not None
        self.width = width
        self.max_seq_len = max_seq_len
        # Per-frame embedding: flatten 12*3 then project
        self.frame_proj = nnx.Linear(TACTILE_DIM, width, rngs=rngs)
        # Temporal positional embedding
        self.time_emb = nnx.Param(
            jnp.zeros((1, max_seq_len, width), dtype=jnp.float32),
        )

    def __call__(self, tactile_seq: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            tactile_seq: (B, T, 12, 3).

        Returns:
            (B, T, width).
        """
        B, T = tactile_seq.shape[0], tactile_seq.shape[1]
        if T > self.max_seq_len:
            raise ValueError(f"sequence length {T} > max_seq_len {self.max_seq_len}")
        flat = tactile_seq.reshape(B, T, -1)    # (B, T, 36)
        x = self.frame_proj(flat)               # (B, T, width)
        x = x + self.time_emb[:, :T, :]
        return x


def build_tactile_encoder(
    variant: TactileEncoderVariant,
    width: int,
    rngs: nnx.Rngs,
    max_seq_len: int = 100,
) -> nnx.Module:
    """Factory for tactile encoders.

    Args:
        variant: "linear" / "per_taxel" / "time_series".
        width: Output token width (match pi0 action_expert width).
        rngs: NNX random keys.
        max_seq_len: Only used for time_series variant.

    Returns:
        nnx.Module that outputs tactile tokens.
    """
    if variant == "linear":
        return TactileLinearEncoder(width=width, rngs=rngs)
    elif variant == "per_taxel":
        return TactilePerTaxelEncoder(width=width, rngs=rngs)
    elif variant == "time_series":
        return TactileTimeSeriesEncoder(
            width=width, max_seq_len=max_seq_len, rngs=rngs,
        )
    else:
        raise ValueError(f"unknown variant: {variant!r}")
