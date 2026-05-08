# Local Frontier Routing Patch Plan

This branch adds opt-in hooks for training-free MoE active expert allocation experiments.
Default Atlas behavior must remain unchanged when the flags are absent.

## Implemented: Fixed Top-k Override

New `spark serve` flags:

```text
--moe-top-k-policy model-config|fixed
--moe-top-k-override <N>
```

Environment override:

```text
ATLAS_MOE_TOP_K_OVERRIDE=<N>
```

Behavior:

- Default policy is `model-config`, preserving the checkpoint/config value.
- `--moe-top-k-override N` requires `--moe-top-k-policy fixed`.
- `N` must be in `1..=num_experts`.
- The override mutates `ModelConfig.num_experts_per_tok` before memory preflight, buffer sizing, model build, and MoE dispatch.
- Startup logs include default top-k, override top-k, expert count, and `norm_topk_prob`.
- Env override is treated as a fixed policy and is intended for experiment manifests.

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

## Implemented: Router Entropy / Selection Summary Logging

Logging-only environment flags:

```text
ATLAS_MOE_ROUTER_STATS=1
ATLAS_MOE_ROUTER_STATS_PATH=/results/router_stats.jsonl
ATLAS_MOE_ROUTER_STATS_MAX_TOKENS=4
```

Constraints:

- No behavior change when disabled.
- Summaries only; full logits dump is intentionally deferred.
- Token/layer volume is bounded by `ATLAS_MOE_ROUTER_STATS_MAX_TOKENS`.
- Stats are written after top-k selection from selected expert ids and selected gate weights.
- Decode stats are skipped during CUDA graph capture; prefill stats are the first useful target.

JSONL fields:

- `timestamp`
- `request_id` (`null` until request metadata is wired into `ForwardContext`)
- `layer`
- `token_index`
- `top_k`
- `selected_expert_ids`
- `selected_gate_scores`
- `entropy`
- `max_prob`
- `margin_top1_top2`
