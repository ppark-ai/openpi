# TacHand-VLA Patch Instructions (v2 — pi0.5 + PyTorch aware)

This file documents the minimal changes needed to add the magnetic tactile
modality to openpi. It targets `origin/main` (commit ~`650c5b0`), which
ships pi0.5 support in BOTH the JAX (`pi0.py`) and PyTorch
(`models_pytorch/pi0_pytorch.py`) paths.

> v1 of this doc only patched JAX/pi0. This v2 reflects the actual
> upstream state: pi0.5 (with adaRMSNorm + discrete-state input) is the
> current SOTA from PI, and openpi has dual-framework support.

---

## What changed since v1

| Item | v1 (Oct 2024 snapshot) | v2 (Apr 2026 main) |
|------|------------------------|---------------------|
| Base model | pi0 only | pi0 + pi0.5 (`pi05: bool = False`) |
| Framework | JAX/Flax | JAX/Flax **and** PyTorch |
| `pi0_config.py` | separate file | still separate on main; merged into pi0.py only on `kevin/pi05-support` |
| State injection | suffix (state token) | suffix for pi0; **discrete prompt** for pi0.5 |
| Time conditioning | concat MLP | adaRMSNorm for pi0.5 (`adarms_cond` returned) |

We will support **both** `pi05=False` and `pi05=True` from day one — same
config flag (`use_tactile`) toggles tactile injection in either mode.

---

## Tactile injection strategy (pi0 vs pi0.5)

In both modes we put the tactile token(s) in the **suffix**, alongside the
action expert tokens. Rationale:

- pi0.5's "state in prompt" rule is for proprioception (joint angles —
  semantic info the LLM should reason about). Tactile is a high-bandwidth
  servo signal closer in spirit to the action expert input.
- Keeping tactile in the suffix means a single code path; adaRMSNorm
  conditioning naturally covers it via `adarms_cond`.
- Future v3 may add an optional **prefix** path (tactile→language token)
  for in-context demos, but that is out of scope here.

Resulting suffix layout:

```
pi0  (pi05=False):  [ state ] [ tactile? ] [ action_time × H ]
pi0.5 (pi05=True):  [ tactile? ] [ action × H ]    ← state is in prompt
```

---

## Changes (4 files)

### 1. `src/openpi/models/pi0_config.py` — add config flags

```diff
 @dataclasses.dataclass(frozen=True)
 class Pi0Config(_model.BaseModelConfig):
     ...
     pi05: bool = False
     discrete_state_input: bool = None  # type: ignore
     pytorch_compile_mode: str | None = "max-autotune"
+
+    # ── TacHand-VLA: magnetic tactile modality ─────────────────────────
+    use_tactile: bool = False
+    tactile_dim: int = 36   # 12 taxels × 3 axes (Bx, By, Bz)
+    tactile_encoder_variant: str = "linear"  # "linear" | "per_taxel" | "time_series"
+    tactile_n_tokens: int = 1   # 1 for linear; 12 for per_taxel; T for time_series
```

Update `inputs_spec()` to expose tactile when enabled (optional — sidecar
is fine for prototyping):

```diff
         observation_spec = _model.Observation(
             ...
             tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
+            **(
+                {"tactile": jax.ShapeDtypeStruct([batch_size, 12, 3], jnp.float32)}
+                if self.use_tactile else {}
+            ),
         )
```
(Requires the v3 `Observation` field — see step 4.)

### 2. `src/openpi/models/pi0.py` — JAX __init__

```diff
         self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)
+
+        # TacHand-VLA: tactile encoder
+        if getattr(config, "use_tactile", False):
+            from openpi.models.tactile_encoder import build_tactile_encoder
+            self.tactile_encoder = build_tactile_encoder(
+                variant=config.tactile_encoder_variant,
+                width=action_expert_config.width,
+                rngs=rngs,
+            )
+            self.use_tactile = True
+            self.tactile_n_tokens = config.tactile_n_tokens
+        else:
+            self.use_tactile = False
+
         # This attribute gets automatically set by model.train() and model.eval().
         self.deterministic = True
```

### 3. `src/openpi/models/pi0.py` — JAX `embed_suffix()`

Insert **after** the `if not self.pi05:` state-token block, **before**
the action-expert block. Works in both pi0 and pi0.5 modes:

```diff
     if not self.pi05:
         # add a single state token
         state_token = self.state_proj(obs.state)[:, None, :]
         tokens.append(state_token)
         input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
         ar_mask += [True]
+
+    # ── TacHand-VLA: tactile token(s) ───────────────────────────────
+    if self.use_tactile and getattr(obs, "tactile", None) is not None:
+        tactile_tokens = self.tactile_encoder(obs.tactile)
+        # tactile_tokens shape: (B, n_tact, width)
+        n_tact = tactile_tokens.shape[1]
+        tokens.append(tactile_tokens)
+        input_mask.append(
+            jnp.ones((tactile_tokens.shape[0], n_tact), dtype=jnp.bool_)
+        )
+        # Tactile attends to image/language/state but action does not see it
+        # before its own block — so first tactile is ar=True, rest ar=False
+        ar_mask += [True] + [False] * (n_tact - 1)

     action_tokens = self.action_in_proj(noisy_actions)
```

### 4. `src/openpi/models_pytorch/pi0_pytorch.py` — PyTorch mirror

PyTorch `embed_suffix(self, state, noisy_actions, timestep)` does NOT
take `obs`. We extend the signature with an optional `tactile` arg and
have the caller (`forward()`) pass it through.

`__init__`:

```diff
         if self.pi05:
             self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
             self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
         else:
             self.state_proj = nn.Linear(config.action_dim, action_expert_config.width)
             self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
             self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
+
+        # TacHand-VLA: tactile encoder (PyTorch)
+        if getattr(config, "use_tactile", False):
+            from openpi.models_pytorch.tactile_encoder_pytorch import build_tactile_encoder
+            self.tactile_encoder = build_tactile_encoder(
+                variant=config.tactile_encoder_variant,
+                width=action_expert_config.width,
+            )
+            self.use_tactile = True
+        else:
+            self.use_tactile = False
```

`embed_suffix` signature + body:

```diff
-    def embed_suffix(self, state, noisy_actions, timestep):
+    def embed_suffix(self, state, noisy_actions, timestep, tactile=None):
         embs = []
         pad_masks = []
         att_masks = []

         if not self.pi05:
             ...
             embs.append(state_emb[:, None, :])
             pad_masks.append(state_mask)
             att_masks += [1]
+
+        # TacHand-VLA: tactile token(s)
+        if self.use_tactile and tactile is not None:
+            tactile_emb = self.tactile_encoder(tactile)   # (B, n_tact, width)
+            n_tact = tactile_emb.shape[1]
+            embs.append(tactile_emb)
+            pad_masks.append(torch.ones(tactile_emb.shape[0], n_tact, dtype=torch.bool, device=tactile_emb.device))
+            att_masks += [1] + [0] * (n_tact - 1)
```

`forward()` and `sample_actions()` need to accept `tactile` from
`observation` and pass it through. Search for `embed_suffix(` calls.

### 5. `src/openpi/models/model.py` — extend `Observation`

Optional but cleaner than sidecar passing:

```diff
 @struct.dataclass
 class Observation:
     images: dict[str, jnp.ndarray]
     image_masks: dict[str, jnp.ndarray]
     state: jnp.ndarray | None = None
     tokenized_prompt: jnp.ndarray | None = None
     tokenized_prompt_mask: jnp.ndarray | None = None
+    # TacHand-VLA optional tactile field — (B, 12, 3) float32 magnetic deltas
+    tactile: jnp.ndarray | None = None
```

PyTorch `_preprocess_observation` should also be updated to thread
`observation.tactile` through (no-op if None).

---

## Validation plan (Colab GPU or local PyTorch)

Two parallel paths — JAX is ground truth, PyTorch is what we'll use on
Windows for everyday work:

```python
# JAX path
from openpi.models.pi0_config import Pi0Config
config = Pi0Config(
    paligemma_variant="dummy",
    action_expert_variant="dummy",
    action_dim=16, action_horizon=8,
    pi05=False,                # also test pi05=True
    use_tactile=True,
    tactile_encoder_variant="linear",
)
model = config.create(rng)
obs = ...   # with tactile=(B, 12, 3)
loss = model.compute_loss(rng, obs, actions, train=True)
assert jnp.isfinite(loss).all()
```

```python
# PyTorch path (preferred on Windows)
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
config = Pi0Config(..., use_tactile=True)
model = PI0Pytorch(config).cuda()
loss = model(observation_with_tactile, actions)
loss.backward()
# Inspect model.tactile_encoder grads — must be non-zero
```

---

## File status

- ✅ `tactile_encoder.py` (JAX) — 3 variants implemented
- ✅ `tactile_encoder_test.py` — 11 tests (need JAX env to run)
- ⬜ `tactile_encoder_pytorch.py` — TODO, mirror of JAX version (~80 LOC)
- ⬜ `pi0_config.py` patch — documented here
- ⬜ `pi0.py` patch (JAX) — documented here
- ⬜ `pi0_pytorch.py` patch — documented here
- ⬜ `model.py` `Observation` field — documented here
- ⬜ End-to-end forward + backward test (needs JAX or torch + GPU)

---

## Compatibility with future PI releases

PI ships major upgrades roughly every 6 months (pi0 → pi0-FAST → pi0.5
→ pi0.6 in Jan 2026 with RECAP). Our patch is structured so that:

1. **All tactile logic lives in `embed_suffix`** + a single `__init__` block.
   New PI variants that change the prefix path (e.g. RECAP) leave our
   patch untouched.
2. **`use_tactile` defaults False** — every existing pi0 / pi0.5 / pi0.6
   config / checkpoint loads unchanged.
3. **`tactile_encoder` is a separate module** — easy to swap encoders
   without touching pi0.py.

When a new PI release lands, the rebase work is: re-apply this
~80 LOC patch to the new `pi0.py` / `pi0_pytorch.py`. No downstream code
changes.

---

## Next steps after patch

1. Stage 2 — tactile-only fine-tune (frozen backbone) on 100h glove data
2. Ablation — 3 encoder variants × {pi0, pi0.5} = 6 cells
3. Stage 3 — joint fine-tune on TacHand-4 task demos
4. In-context learning interface (RICL-style retrieval)
5. (Rebuttal) Re-base on pi0.6 / RECAP if released by submission deadline
