# TacHand-VLA Patch Instructions

This file documents the minimal changes needed to add magnetic tactile
modality to pi0. Apply via `git apply` once JAX env is set up (Colab).

---

## Strategy

**Minimally invasive:** add tactile token to the suffix sequence
(alongside state token), do NOT modify the Observation dataclass.
Tactile is passed as part of state (concatenated 36-d) when `use_tactile=True`.

This is the smallest change that lets us test the architecture. After
validation we can add a proper `tactile` field to `Observation`.

---

## Changes (3 files)

### 1. `pi0_config.py` — add config flag

```diff
 @dataclasses.dataclass(frozen=True)
 class Pi0Config(_model.BaseModelConfig):
     ...
     pi05: bool = False
+    # TacHand: enable magnetic tactile modality
+    use_tactile: bool = False
+    tactile_dim: int = 36   # 12 taxels x 3 axes
+    tactile_encoder_variant: str = "linear"  # "linear" / "per_taxel" / "time_series"
```

### 2. `pi0.py` __init__ — instantiate tactile encoder

```diff
     else:
         self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
         self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
         self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
+
+    # TacHand-VLA: tactile encoder
+    if getattr(config, "use_tactile", False):
+        from openpi.models.tactile_encoder import build_tactile_encoder
+        self.tactile_encoder = build_tactile_encoder(
+            variant=config.tactile_encoder_variant,
+            width=action_expert_config.width,
+            rngs=rngs,
+        )
+        self.use_tactile = True
+    else:
+        self.use_tactile = False
```

### 3. `pi0.py` `embed_suffix()` — append tactile token

```diff
     if not self.pi05:
         # add a single state token
         state_token = self.state_proj(obs.state)[:, None, :]
         tokens.append(state_token)
         input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
         # image/language inputs do not attend to state or actions
         ar_mask += [True]
+
+    # TacHand-VLA: tactile token
+    if self.use_tactile and hasattr(obs, "tactile") and obs.tactile is not None:
+        tactile_token = self.tactile_encoder(obs.tactile)
+        # tactile_token shape: (B, n_tact_tokens, width)
+        n_tactile_tokens = tactile_token.shape[1]
+        tokens.append(tactile_token)
+        input_mask.append(
+            jnp.ones((obs.tactile.shape[0], n_tactile_tokens), dtype=jnp.bool_)
+        )
+        ar_mask += [True] * n_tactile_tokens
```

### 4. (Optional) Backward-compatible Observation field

If we want to expose `tactile` cleanly, add to `_model.Observation`:

```diff
 @struct.dataclass
 class Observation:
     images: dict[str, jnp.ndarray]
     image_masks: dict[str, jnp.ndarray]
     state: jnp.ndarray | None = None
     tokenized_prompt: jnp.ndarray | None = None
     tokenized_prompt_mask: jnp.ndarray | None = None
+    # TacHand-VLA optional modality
+    tactile: jnp.ndarray | None = None
```

This is optional — `obs.tactile` can be passed as a sidecar attribute
via a custom subclass during prototyping.

---

## Validation plan (Colab GPU)

```python
# 1. Build small Pi0 with use_tactile=True
config = Pi0Config(
    paligemma_variant="dummy",       # smallest variant for test
    action_dim=16,
    action_horizon=8,
    use_tactile=True,
    tactile_encoder_variant="linear",
)
model = config.create(rng)

# 2. Synthetic batch
obs = ...  # with images + state + tactile=(B, 12, 3)
actions = ...

# 3. Forward pass
logits = model.compute_loss(rng, obs, actions, ...)
assert jnp.isfinite(logits)

# 4. Gradient flow
grads = jax.grad(loss_fn)(model.params, ...)
# Check tactile_encoder grads are non-zero
```

---

## File status

- ✅ `tactile_encoder.py` — implemented (3 variants)
- ✅ `tactile_encoder_test.py` — 11 tests
- ⬜ `pi0_config.py` patch — documented here, apply when env ready
- ⬜ `pi0.py` patch — documented here, apply when env ready
- ⬜ End-to-end forward + backward test (needs JAX + GPU)

---

## Next steps after patch

1. Train tactile-only with frozen backbone (Stage 2): 100h glove data
2. Compare 3 encoder variants in ablation
3. Joint fine-tune (Stage 3) on 4 task demos
4. In-context learning interface
