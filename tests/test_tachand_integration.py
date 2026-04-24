"""End-to-end integration smoke tests for the TacHand-VLA patch.

Two layers:

  Smoke A — runs locally without JAX/transformers/paligemma. Verifies
  that the config field, Observation field, and tactile_encoder all
  cooperate. ~1s.

  Smoke B — needs the full openpi PyTorch stack (transformers_replace
  patches + paligemma weights + GPU). Constructs a real Pi0Pytorch with
  use_tactile=True and runs forward(). Skipped if env not ready.

Run:
    pytest tests/test_tachand_integration.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")


# ──────────────────────────────────────────────────────────────────────
# Smoke A: lightweight integration (always runnable)
# ──────────────────────────────────────────────────────────────────────


def test_pi0_config_accepts_tactile_flags() -> None:
    """Pi0Config should expose use_tactile and related fields."""
    pytest.importorskip("jax")
    from openpi.models.pi0_config import Pi0Config

    cfg = Pi0Config(
        action_dim=16,
        action_horizon=8,
        use_tactile=True,
        tactile_encoder_variant="linear",
    )
    assert cfg.use_tactile is True
    assert cfg.tactile_dim == 36
    assert cfg.tactile_n_tokens == 1
    assert cfg.tactile_encoder_variant == "linear"


def test_pi0_config_default_keeps_tactile_off() -> None:
    """Default config — backward compatibility."""
    pytest.importorskip("jax")
    from openpi.models.pi0_config import Pi0Config

    cfg = Pi0Config(action_dim=16, action_horizon=8)
    assert cfg.use_tactile is False


def test_pi0_config_inputs_spec_includes_tactile_when_enabled() -> None:
    pytest.importorskip("jax")
    from openpi.models.pi0_config import Pi0Config

    cfg = Pi0Config(action_dim=16, action_horizon=8, use_tactile=True)
    obs_spec, _ = cfg.inputs_spec(batch_size=2)
    assert obs_spec.tactile is not None
    assert obs_spec.tactile.shape == (2, 12, 3)


def test_pi0_config_inputs_spec_omits_tactile_when_disabled() -> None:
    pytest.importorskip("jax")
    from openpi.models.pi0_config import Pi0Config

    cfg = Pi0Config(action_dim=16, action_horizon=8, use_tactile=False)
    obs_spec, _ = cfg.inputs_spec(batch_size=2)
    assert obs_spec.tactile is None


def test_observation_dataclass_carries_tactile() -> None:
    """Observation must accept and round-trip a tactile array."""
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from openpi.models.model import Observation

    images = {"base_0_rgb": jnp.zeros((2, 224, 224, 3), dtype=jnp.float32)}
    image_masks = {"base_0_rgb": jnp.ones((2,), dtype=jnp.bool_)}
    tactile = jnp.ones((2, 12, 3), dtype=jnp.float32) * 0.5

    obs = Observation(
        images=images,
        image_masks=image_masks,
        state=jnp.zeros((2, 16), dtype=jnp.float32),
        tactile=tactile,
    )
    assert obs.tactile is not None
    assert obs.tactile.shape == (2, 12, 3)
    np.testing.assert_allclose(np.asarray(obs.tactile), 0.5)


def test_observation_from_dict_picks_up_tactile() -> None:
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from openpi.models.model import Observation

    data = {
        "image": {"base_0_rgb": jnp.zeros((1, 224, 224, 3), dtype=jnp.float32)},
        "image_mask": {"base_0_rgb": jnp.ones((1,), dtype=jnp.bool_)},
        "state": jnp.zeros((1, 16), dtype=jnp.float32),
        "tactile": jnp.ones((1, 12, 3), dtype=jnp.float32),
    }
    obs = Observation.from_dict(data)
    assert obs.tactile is not None
    assert obs.tactile.shape == (1, 12, 3)


def test_pytorch_encoder_factory_works_for_all_variants() -> None:
    """PyTorch encoder produces correct output shapes for every variant."""
    from openpi.models_pytorch.tactile_encoder_pytorch import build_tactile_encoder

    width = 64
    tactile = torch.randn(3, 12, 3)

    enc_linear = build_tactile_encoder("linear", width=width)
    out = enc_linear(tactile)
    assert out.shape == (3, 1, width)

    enc_per = build_tactile_encoder("per_taxel", width=width)
    out = enc_per(tactile)
    assert out.shape == (3, 12, width)

    enc_ts = build_tactile_encoder("time_series", width=width, max_seq_len=10)
    tactile_seq = torch.randn(3, 5, 12, 3)
    out = enc_ts(tactile_seq)
    assert out.shape == (3, 5, width)


def test_pytorch_encoder_grad_flows_back() -> None:
    """Backward through encoder produces non-zero grads on its weights."""
    from openpi.models_pytorch.tactile_encoder_pytorch import build_tactile_encoder

    enc = build_tactile_encoder("linear", width=32)
    tactile = torch.randn(2, 12, 3, requires_grad=False)
    out = enc(tactile)
    out.sum().backward()
    assert enc.proj.weight.grad is not None
    assert enc.proj.weight.grad.abs().sum().item() > 0


# ──────────────────────────────────────────────────────────────────────
# Smoke B: full Pi0Pytorch forward (skipped without env)
# ──────────────────────────────────────────────────────────────────────


def _full_pytorch_env_available() -> bool:
    try:
        import transformers  # noqa: F401
        from transformers.models.siglip import check  # noqa: F401

        return check.check_whether_transformers_replace_is_installed_correctly()
    except Exception:
        return False


@pytest.mark.skipif(
    not _full_pytorch_env_available(),
    reason="Full PyTorch openpi env (transformers_replace patches) not set up",
)
def test_pi0_pytorch_forward_with_tactile_returns_finite_loss() -> None:
    """Real Pi0Pytorch forward with use_tactile=True. Needs full env."""
    from openpi.models.pi0_config import Pi0Config
    from openpi.models.model import Observation
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    cfg = Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=16,
        action_horizon=8,
        use_tactile=True,
        tactile_encoder_variant="linear",
        pytorch_compile_mode=None,
    )
    model = PI0Pytorch(cfg)

    B = 2
    obs = Observation(
        images={"base_0_rgb": torch.zeros(B, 224, 224, 3)},
        image_masks={"base_0_rgb": torch.ones(B, dtype=torch.bool)},
        state=torch.zeros(B, cfg.action_dim),
        tactile=torch.randn(B, 12, 3),
    )
    actions = torch.zeros(B, cfg.action_horizon, cfg.action_dim)

    loss = model(obs, actions)
    assert torch.isfinite(loss).all()
