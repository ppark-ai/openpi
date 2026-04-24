"""Tests for tactile encoder variants."""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models.tactile_encoder import (
    N_TAXELS,
    N_AXES,
    TACTILE_DIM,
    TactileLinearEncoder,
    TactilePerTaxelEncoder,
    TactileTimeSeriesEncoder,
    build_tactile_encoder,
)


def _make_rngs(seed: int = 0) -> nnx.Rngs:
    return nnx.Rngs(seed)


def test_linear_encoder_shape() -> None:
    enc = TactileLinearEncoder(width=128, rngs=_make_rngs())
    tactile = jnp.ones((4, N_TAXELS, N_AXES))
    out = enc(tactile)
    assert out.shape == (4, 1, 128)


def test_linear_encoder_accepts_flat_input() -> None:
    enc = TactileLinearEncoder(width=128, rngs=_make_rngs())
    tactile_flat = jnp.ones((4, TACTILE_DIM))
    out = enc(tactile_flat)
    assert out.shape == (4, 1, 128)


def test_per_taxel_encoder_shape() -> None:
    enc = TactilePerTaxelEncoder(width=128, n_heads=4, rngs=_make_rngs())
    tactile = jnp.ones((4, N_TAXELS, N_AXES))
    out = enc(tactile)
    assert out.shape == (4, N_TAXELS, 128)


def test_time_series_encoder_shape() -> None:
    enc = TactileTimeSeriesEncoder(width=128, max_seq_len=50, rngs=_make_rngs())
    tactile_seq = jnp.ones((2, 30, N_TAXELS, N_AXES))
    out = enc(tactile_seq)
    assert out.shape == (2, 30, 128)


def test_time_series_encoder_seq_too_long_raises() -> None:
    enc = TactileTimeSeriesEncoder(width=64, max_seq_len=10, rngs=_make_rngs())
    tactile_seq = jnp.ones((1, 20, N_TAXELS, N_AXES))
    try:
        enc(tactile_seq)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_factory_dispatches_correctly() -> None:
    rngs = _make_rngs()
    a = build_tactile_encoder("linear", width=64, rngs=rngs)
    b = build_tactile_encoder("per_taxel", width=64, rngs=rngs)
    c = build_tactile_encoder("time_series", width=64, rngs=rngs)
    assert isinstance(a, TactileLinearEncoder)
    assert isinstance(b, TactilePerTaxelEncoder)
    assert isinstance(c, TactileTimeSeriesEncoder)


def test_factory_unknown_raises() -> None:
    try:
        build_tactile_encoder("nonsense", width=64, rngs=_make_rngs())
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_linear_encoder_deterministic_eval() -> None:
    enc = TactileLinearEncoder(width=64, rngs=_make_rngs(42))
    tactile = jnp.array(np.random.RandomState(0).randn(2, N_TAXELS, N_AXES), dtype=jnp.float32)
    out1 = enc(tactile)
    out2 = enc(tactile)
    np.testing.assert_allclose(np.asarray(out1), np.asarray(out2), atol=1e-6)


def test_per_taxel_encoder_different_per_taxel() -> None:
    """Different taxel readings should produce different per-taxel outputs."""
    enc = TactilePerTaxelEncoder(width=32, n_heads=4, rngs=_make_rngs(0))
    tactile = jnp.zeros((1, N_TAXELS, N_AXES))
    tactile = tactile.at[0, 5, 1].set(1.0)  # spike on taxel 5 axis 1
    out = enc(tactile)
    # Token 5 should differ from token 0
    diff = float(jnp.linalg.norm(out[0, 5] - out[0, 0]))
    assert diff > 1e-3


def test_encoder_param_counts_reasonable() -> None:
    """Sanity check param counts are within expected order of magnitude."""
    # Linear: 36*128 + 128 = ~4.7K
    a = TactileLinearEncoder(width=128, rngs=_make_rngs())
    state_a = nnx.state(a, nnx.Param)
    n_a = sum(int(np.prod(p.shape)) for p in jax.tree_util.tree_leaves(state_a))
    assert 1_000 < n_a < 100_000

    # Per-taxel: bigger (attn + projections)
    b = TactilePerTaxelEncoder(width=128, n_heads=4, rngs=_make_rngs())
    state_b = nnx.state(b, nnx.Param)
    n_b = sum(int(np.prod(p.shape)) for p in jax.tree_util.tree_leaves(state_b))
    assert n_b > n_a  # at least bigger than linear


def test_constants_match_glove_spec() -> None:
    assert N_TAXELS == 12
    assert N_AXES == 3
    assert TACTILE_DIM == 36
