"""Tests for PyTorch tactile encoder variants — mirrors the JAX test file."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from openpi.models_pytorch.tactile_encoder_pytorch import (  # noqa: E402
    N_AXES,
    N_TAXELS,
    TACTILE_DIM,
    TactileLinearEncoder,
    TactilePerTaxelEncoder,
    TactileTimeSeriesEncoder,
    build_tactile_encoder,
)


def test_linear_encoder_shape() -> None:
    enc = TactileLinearEncoder(width=128)
    tactile = torch.ones(4, N_TAXELS, N_AXES)
    out = enc(tactile)
    assert out.shape == (4, 1, 128)


def test_linear_encoder_accepts_flat_input() -> None:
    enc = TactileLinearEncoder(width=128)
    tactile_flat = torch.ones(4, TACTILE_DIM)
    out = enc(tactile_flat)
    assert out.shape == (4, 1, 128)


def test_per_taxel_encoder_shape() -> None:
    enc = TactilePerTaxelEncoder(width=128, n_heads=4)
    tactile = torch.ones(4, N_TAXELS, N_AXES)
    out = enc(tactile)
    assert out.shape == (4, N_TAXELS, 128)


def test_time_series_encoder_shape() -> None:
    enc = TactileTimeSeriesEncoder(width=128, max_seq_len=50)
    tactile_seq = torch.ones(2, 30, N_TAXELS, N_AXES)
    out = enc(tactile_seq)
    assert out.shape == (2, 30, 128)


def test_time_series_encoder_seq_too_long_raises() -> None:
    enc = TactileTimeSeriesEncoder(width=64, max_seq_len=10)
    tactile_seq = torch.ones(1, 20, N_TAXELS, N_AXES)
    with pytest.raises(ValueError, match="sequence length"):
        enc(tactile_seq)


def test_factory_dispatches_correctly() -> None:
    assert isinstance(build_tactile_encoder("linear", width=64), TactileLinearEncoder)
    assert isinstance(build_tactile_encoder("per_taxel", width=64), TactilePerTaxelEncoder)
    assert isinstance(build_tactile_encoder("time_series", width=64), TactileTimeSeriesEncoder)


def test_factory_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown variant"):
        build_tactile_encoder("nonsense", width=64)


def test_linear_encoder_deterministic_eval() -> None:
    torch.manual_seed(42)
    enc = TactileLinearEncoder(width=64).eval()
    tactile = torch.from_numpy(np.random.RandomState(0).randn(2, N_TAXELS, N_AXES).astype(np.float32))
    with torch.no_grad():
        out1 = enc(tactile)
        out2 = enc(tactile)
    torch.testing.assert_close(out1, out2)


def test_per_taxel_encoder_different_per_taxel() -> None:
    """Spike on one taxel should produce a different token at that index."""
    torch.manual_seed(0)
    enc = TactilePerTaxelEncoder(width=32, n_heads=4).eval()
    tactile = torch.zeros(1, N_TAXELS, N_AXES)
    tactile[0, 5, 1] = 1.0
    with torch.no_grad():
        out = enc(tactile)
    diff = (out[0, 5] - out[0, 0]).norm().item()
    assert diff > 1e-3


def test_encoder_param_counts_reasonable() -> None:
    a = TactileLinearEncoder(width=128)
    n_a = sum(p.numel() for p in a.parameters())
    assert 1_000 < n_a < 100_000

    b = TactilePerTaxelEncoder(width=128, n_heads=4)
    n_b = sum(p.numel() for p in b.parameters())
    assert n_b > n_a


def test_grad_flows_through_encoder() -> None:
    """Sanity: backward through encoder yields non-zero grads on the proj weights."""
    torch.manual_seed(0)
    enc = TactileLinearEncoder(width=32)
    tactile = torch.randn(2, N_TAXELS, N_AXES)
    out = enc(tactile)
    out.sum().backward()
    assert enc.proj.weight.grad is not None
    assert enc.proj.weight.grad.abs().sum().item() > 0


def test_constants_match_glove_spec() -> None:
    assert N_TAXELS == 12
    assert N_AXES == 3
    assert TACTILE_DIM == 36
