# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # TacHand-VLA Smoke Test (Colab)
#
# Verifies that the tactile-modality patches in `ppark-ai/openpi @ tachand-vla`
# work end-to-end on a real GPU runtime with JAX + PaliGemma.
#
# **Open in Colab**:
# https://colab.research.google.com/github/ppark-ai/openpi/blob/tachand-vla/examples/tachand/colab_smoke.ipynb
#
# **Runtime**: T4 (16 GB) is enough for the dummy paligemma_variant.
# A100 needed only if you swap in `gemma_2b`.
#
# **Convert this `.py` to a notebook locally** (one-time, before Colab upload):
# ```bash
# pip install jupytext
# jupytext --to ipynb examples/tachand/colab_smoke.py
# ```
# Or just open this `.py` directly in Colab — it understands the `# %%` cells.

# %% [markdown]
# ## 1. Clone fork + install
#
# Pulls `tachand-vla` branch which contains:
# - `tactile_encoder.py` (JAX) + `tactile_encoder_pytorch.py` (PyTorch)
# - PATCH_TACHAND v2 applied to `pi0.py` / `pi0_pytorch.py` / `pi0_config.py` / `model.py`
# - `tests/test_tachand_integration.py`

# %%
# !rm -rf /content/openpi
# !git clone --depth 1 -b tachand-vla https://github.com/ppark-ai/openpi.git /content/openpi
# %cd /content/openpi
# !git log --oneline -3

# %% [markdown]
# ### Install openpi (editable) with required extras

# %%
# !pip install -q uv
# !uv pip install --system -e ".[pytorch]"  # 'pytorch' extra pulls torch + transformers==4.53.2
# !uv pip install --system pynvml jupytext pytest

# %% [markdown]
# ### Apply transformers_replace patches
#
# openpi ships overrides for SigLIP / Gemma / PaliGemma to fix issues in the
# upstream transformers package. Without this patch, `Pi0Pytorch.__init__`
# raises a `ValueError`.

# %%
# !cp -r src/openpi/models_pytorch/transformers_replace/* \
#         /usr/local/lib/python3.10/dist-packages/transformers/
# !python -c "from transformers.models.siglip import check; print('replace ok:', check.check_whether_transformers_replace_is_installed_correctly())"

# %% [markdown]
# ## 2. Sanity check — JAX import + Pi0Config

# %%
import jax  # noqa: E402
print("JAX:", jax.__version__, "devices:", jax.devices())

from openpi.models.pi0_config import Pi0Config  # noqa: E402

cfg = Pi0Config(
    paligemma_variant="dummy",
    action_expert_variant="dummy",
    action_dim=16,
    action_horizon=8,
    use_tactile=True,
    tactile_encoder_variant="linear",
    pytorch_compile_mode=None,
)
print("use_tactile:", cfg.use_tactile)
print("tactile_n_tokens:", cfg.tactile_n_tokens)

obs_spec, act_spec = cfg.inputs_spec(batch_size=2)
print("obs.tactile spec:", obs_spec.tactile)
assert obs_spec.tactile.shape == (2, 12, 3)
assert act_spec.shape == (2, 8, 16)
print("OK — Pi0Config exposes tactile spec when use_tactile=True")

# %% [markdown]
# ## 3. Smoke A — JAX integration tests
#
# Runs the 6 JAX-only tests from `tests/test_tachand_integration.py`:
#
# - `Pi0Config` accepts/defaults tactile flags
# - `inputs_spec` includes/omits tactile correctly
# - `Observation` dataclass carries tactile
# - `Observation.from_dict` picks up tactile from dict

# %%
# !python -m pytest tests/test_tachand_integration.py -v --no-header -p no:cacheprovider \
#     -k "config or observation or default" 2>&1 | tail -25

# %% [markdown]
# Expected: 6 passed (the 2 PyTorch-encoder tests should also pass, the
# 1 Smoke-B test runs in section 5 below).

# %% [markdown]
# ## 4. Build a real Pi0 (JAX) and forward-pass with tactile

# %%
import jax.numpy as jnp  # noqa: E402

from openpi.models.model import Observation  # noqa: E402

rng = jax.random.PRNGKey(0)
model = cfg.create(rng)
print("Model parameter count (approx):",
      sum(int(jnp.prod(jnp.array(p.shape))) for p in jax.tree_util.tree_leaves(jax.tree_util.tree_map(lambda x: x, model))[:50]))

B = 2
images = {
    name: jnp.zeros((B, 224, 224, 3), dtype=jnp.float32)
    for name in ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
}
image_masks = {name: jnp.ones((B,), dtype=jnp.bool_) for name in images}
obs = Observation(
    images=images,
    image_masks=image_masks,
    state=jnp.zeros((B, cfg.action_dim), dtype=jnp.float32),
    tokenized_prompt=jnp.zeros((B, cfg.max_token_len), dtype=jnp.int32),
    tokenized_prompt_mask=jnp.zeros((B, cfg.max_token_len), dtype=jnp.bool_),
    tactile=jax.random.normal(rng, (B, 12, 3), dtype=jnp.float32),
)
actions = jnp.zeros((B, cfg.action_horizon, cfg.action_dim), dtype=jnp.float32)

loss = model.compute_loss(rng, obs, actions, train=False)
print("JAX loss shape:", loss.shape, "mean:", float(jnp.mean(loss)))
assert jnp.isfinite(loss).all(), "loss is NaN/Inf — patch is broken"
print("OK — JAX forward with tactile produces finite loss")

# %% [markdown]
# ## 5. Smoke B — PyTorch full forward
#
# This is the test that was skipped locally on Windows (no
# transformers_replace + no GPU). On Colab it should run.

# %%
import torch  # noqa: E402

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch  # noqa: E402

py_model = PI0Pytorch(cfg)
device = "cuda" if torch.cuda.is_available() else "cpu"
py_model = py_model.to(device)

py_obs = Observation(
    images={k: torch.zeros(B, 224, 224, 3, device=device) for k in images},
    image_masks={k: torch.ones(B, dtype=torch.bool, device=device) for k in images},
    state=torch.zeros(B, cfg.action_dim, device=device),
    tokenized_prompt=torch.zeros(B, cfg.max_token_len, dtype=torch.long, device=device),
    tokenized_prompt_mask=torch.zeros(B, cfg.max_token_len, dtype=torch.bool, device=device),
    tactile=torch.randn(B, 12, 3, device=device),
)
py_actions = torch.zeros(B, cfg.action_horizon, cfg.action_dim, device=device)

py_loss = py_model(py_obs, py_actions)
print("PyTorch loss shape:", py_loss.shape, "mean:", float(py_loss.mean()))
assert torch.isfinite(py_loss).all(), "PyTorch loss is NaN/Inf"
print("OK — PyTorch forward with tactile produces finite loss")

# %% [markdown]
# ## 6. Backward + grad check
#
# Confirms that gradients actually flow into the tactile encoder weights
# (not just into the action head).

# %%
py_loss.mean().backward()
tact_grad = py_model.tactile_encoder.proj.weight.grad
assert tact_grad is not None, "tactile_encoder.proj.weight has no grad"
print("tactile_encoder.proj.weight grad norm:", float(tact_grad.norm()))
assert float(tact_grad.norm()) > 0
print("OK — gradients flow back to tactile encoder")

# %% [markdown]
# ## Summary
#
# | Check | Status |
# |---|---|
# | JAX import + Pi0Config(use_tactile=True) | ✅ |
# | inputs_spec exposes tactile | ✅ |
# | Observation carries tactile | ✅ |
# | JAX `compute_loss` returns finite | ✅ |
# | PyTorch `forward` returns finite | ✅ |
# | Backward populates tactile encoder grads | ✅ |
#
# All 6 checks green ⇒ PATCH_TACHAND v2 is fully wired.
#
# Next: replace `paligemma_variant="dummy"` with `"gemma_2b"` to validate
# on the real backbone — needs ~12 GB VRAM (A100 / L4 recommended).
