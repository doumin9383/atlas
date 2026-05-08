# Local Frontier Routing Patch Plan

This branch adds opt-in hooks for training-free MoE active expert allocation experiments.
Default Atlas behavior must remain unchanged when the flags are absent.

## Implemented: Fixed Top-k Override

New `spark serve` flags:

```text
--moe-top-k-policy model-config|fixed
--moe-top-k-override <N>
```

Behavior:

- Default policy is `model-config`, preserving the checkpoint/config value.
- `--moe-top-k-override N` requires `--moe-top-k-policy fixed`.
- `N` must be in `1..=num_experts`.
- The override mutates `ModelConfig.num_experts_per_tok` before memory preflight, buffer sizing, model build, and MoE dispatch.
- Startup logs include default top-k, override top-k, expert count, and `norm_topk_prob`.

Initial experiment ladder:

- A3: `--moe-top-k-policy fixed --moe-top-k-override 3`
- A6: `--moe-top-k-policy fixed --moe-top-k-override 6`
- A9: `--moe-top-k-policy fixed --moe-top-k-override 9`
- A12: `--moe-top-k-policy fixed --moe-top-k-override 12`

## Validation Needed

- Host check: `cargo fmt --all -- --check`
- Host check: `ATLAS_SKIP_BUILD=1 cargo check -p spark-server`
- CUDA/image build on the development build host.
- `spark serve --help` shows both flags.
- DGX Spark smoke test with default config and with A6 override.

## Next Patch: Router Entropy Logging

Add logging-only flags:

```text
--moe-router-entropy-summary
--moe-router-logits-dump /results/router.jsonl
```

Constraints:

- No behavior change.
- Prefer summaries over full logits by default.
- Bound the token/layer volume to avoid large host sync overhead.
- Write JSONL records suitable for benchmark metadata joins.

