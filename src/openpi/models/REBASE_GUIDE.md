# Rebase Guide — TacHand-VLA on future Physical Intelligence releases

PI ships major upgrades roughly every 6 months. This doc tells future-you
how to land our ~80 LOC patch on the next pi0.X without re-deriving
everything from `git diff`.

---

## Release cadence (observed)

| Release | Date | Headline change |
|---------|------|------------------|
| π₀ | 2024-10 | Flow-matching action expert, baseline |
| π₀-FAST | 2025-01 | DCT-based autoregressive action tokens |
| π₀.5 | 2025-09 | adaRMSNorm + state-as-discrete-prompt |
| π₀.6 (RECAP) | 2026-01 | Frozen-backbone fine-tune, 85%+ ALOHA |
| π₀.7+ | TBD | — |

**Watch for new branches** on `Physical-Intelligence/openpi` like
`kevin/pi05-support` — that's the early-access pattern. Then features
are merged into `main` 1-3 months later.

---

## Files most likely to change

Ranked by historical churn (rebase risk):

| File | Risk | Why |
|------|------|-----|
| `pi0.py` (JAX) | HIGH | New conditioning heads, new attention patterns |
| `pi0_pytorch.py` | HIGH | Mirrors pi0.py + transformers integration breaks |
| `pi0_config.py` | MEDIUM | New flags every release — `pi06: bool`, etc. |
| `model.py` Observation | LOW | Append-only field additions are common |
| `gemma.py` / `siglip.py` | LOW | Backbones are stable |

Our patch lives in 4 files; the high-risk ones are the two `pi0` files.

---

## Step-by-step rebase procedure

### 1. Sync upstream

```bash
cd openpi-fork/openpi
git fetch upstream
git log upstream/main --oneline -20    # eyeball recent changes
git log upstream/main -- src/openpi/models/pi0.py | head -30
```

Look specifically for commits that touch `embed_suffix` / `Pi0Config` /
`embed_prefix`. Those are the load-bearing functions for our patch.

### 2. Detect breaking changes

Run this diagnostic:

```bash
git diff upstream/main..HEAD -- \
    src/openpi/models/pi0.py \
    src/openpi/models/pi0_config.py \
    src/openpi/models/model.py \
    src/openpi/models_pytorch/pi0_pytorch.py
```

Group hunks into:
- **Ours** (TacHand additions — usually marked with `# ── TacHand-VLA ──`)
- **Theirs** (new upstream code interleaved)

If the upstream changed `embed_suffix` signature or its body around the
`if not self.pi05:` block, our patch needs hand-merging. Otherwise a
3-way merge is enough.

### 3. Try the merge

```bash
git checkout -b tachand-vla-rebase-pi06
git merge upstream/main
# resolve conflicts in the 4 files
```

Conflict-resolution rules of thumb:
- Keep **all** TacHand `# ── TacHand-VLA ──` blocks intact
- For `embed_suffix`: ensure tactile-injection sits **between** any new
  pre-action conditioning blocks and the action tokens themselves
- For `Pi0Config`: append `use_tactile / tactile_dim / tactile_encoder_variant /
  tactile_n_tokens` after any new upstream fields, before `__post_init__`
- For `Observation`: append `tactile` field at the end (struct dataclass
  field order matters for some serializers)

### 4. Smoke-test locally

```bash
cd openpi-fork/openpi
PYTHONPATH=src python -m pytest \
    src/openpi/models_pytorch/tactile_encoder_pytorch_test.py \
    tests/test_tachand_integration.py \
    --no-header -p no:cacheprovider
```

**Pass criteria**: 14 passed / 7 skipped (same as before rebase).
JAX/full-PyTorch tests skip on Windows; that's fine.

If any of the 14 fails, the rebase broke something — do not ship.

### 5. Smoke-test on Colab

Open `examples/tachand/colab_smoke.py` in Colab. The 6-row summary table
at the bottom must show ✅ across all rows. Pay special attention to:
- "JAX `compute_loss` returns finite" — catches shape-mismatch in
  `embed_suffix` tactile insertion
- "Backward populates tactile encoder grads" — catches if a new
  upstream gradient stop somehow blocks tactile path

### 6. Update PATCH_TACHAND.md

If the rebase required non-trivial changes (new conditioning, signature
differences), bump PATCH version (v3, v4, ...) and document what
changed at the top:

```markdown
# TacHand-VLA Patch Instructions (v3 — pi0.6 RECAP aware)

## What changed since v2
- New `use_recap` flag in pi0.6; tactile injection unchanged
- ...
```

---

## Common rebase pitfalls (learned the hard way)

### A. New conditioning heads in `embed_suffix`

If upstream adds a new conditioning block (e.g., `if config.use_recap:`),
the suffix layout changes:

```
Before (v2):  [ state? ] [ tactile? ] [ action × H ]
After  (v3):  [ state? ] [ recap? ] [ tactile? ] [ action × H ]
```

Decide where tactile sits. Default rule: **tactile always immediately
before action block**, since downstream attention masks assume action
tokens are last.

### B. Signature changes in PyTorch `embed_suffix`

`pi0_pytorch.py` 's `embed_suffix(state, noisy_actions, timestep)` has
positional args. If upstream adds an arg in the middle, our `tactile=`
keyword arg might still work but caller sites break.

Always grep for ALL `embed_suffix(` callers when rebasing PyTorch:

```bash
grep -rn "embed_suffix(" src/openpi/models_pytorch/
```

In v2 there are 3 callers: `forward()`, `denoise_step()`, and the
`embed_suffix` definition itself. Future versions may add more (e.g.,
`compile()` paths, `vmap()` variants).

### C. `Observation` struct ordering

`flax.struct.dataclass` and `dataclasses.dataclass` both care about
field order for some operations (asdict, from_dict). Always append our
`tactile` field at the **end** of the class, after any new upstream
fields. Never insert it in the middle.

### D. Transformers replace patches drift

`models_pytorch/transformers_replace/` is updated when upstream bumps
the `transformers` version. If you see runtime errors about missing
attributes on `PaliGemmaModel`, re-run:

```bash
cp -r src/openpi/models_pytorch/transformers_replace/* \
      "$(python -c 'import transformers, os; print(os.path.dirname(transformers.__file__))')/"
```

(Already in `colab_smoke.py` setup cell.)

---

## Verification matrix

After any rebase, all of these must pass:

| Test | Where | What it catches |
|------|-------|------------------|
| `tactile_encoder_pytorch_test.py` (12 tests) | local | Encoder API drift |
| `test_tachand_integration.py` Smoke A (8 tests) | local | Config + Observation field |
| `test_tachand_integration.py` Smoke B (1 test) | Colab | Real PyTorch forward |
| `colab_smoke.py` JAX section | Colab | Real JAX forward |
| `colab_smoke.py` grad section | Colab | Backward path intact |
| Existing upstream `pi0_test.py` | Colab | We didn't break anything else |

Last test (upstream tests still pass) is the **single most important**
post-rebase check — confirms TacHand additions are truly opt-in.

---

## When NOT to rebase

If the new PI release introduces a fundamentally different interface
(e.g., dropping flow matching for diffusion, removing the prefix/suffix
split entirely), the patch model breaks down. In that case:

- Open an ADR (`docs/decisions/ADR-NNN-vla-rebase-strategy.md`)
- Decide between: (a) port tactile injection to the new architecture,
  (b) stay on pi0.5/.6 indefinitely, (c) fork the patch into a separate
  package that monkey-patches at runtime.

Default recommendation: stay one major version behind PI. Stable >
bleeding edge for a paper submission.
