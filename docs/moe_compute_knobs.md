# MoE Compute Knobs

Research-only compute knobs for Local Frontier Routing experiments.

Default behavior is unchanged unless an explicit feature flag is set.

## Implemented Atlas Knobs

### Global top-k override

Environment:

```bash
ATLAS_MOE_TOP_K_OVERRIDE=6
```

Equivalent CLI:

```bash
spark serve ... --moe-top-k-policy fixed --moe-top-k-override 6
```

If unset, Atlas uses the model config value. Invalid values fail at startup.

### MoE FFN block policy

Environment:

```bash
ATLAS_MOE_BLOCK_POLICY_PATH=/configs/moe_block_policy.yaml
```

If unset, the policy is not loaded and Qwen layer forward behavior stays on the original path.

Implemented for Qwen3.5/Qwen3.6-style `Qwen3AttentionLayer` where the FFN component is `FfnComponent::Moe`. The hook runs after attention/KV cache update and after the normal post-attention residual+RMSNorm step.

Supported:

- MoE FFN skip
- MoE FFN repeat
- `residual_scale`
- `renorm_between_repeats`
- startup validation
- per-layer one-time policy logs
- optional hidden norm sampling / non-finite detection

Not yet implemented:

- per-layer top-k from the block policy
- true fallback after NaN once hidden has already been mutated
- dense FFN or Gemma dual-FFN policy support

## External Recurrent Policy

Environment name reserved:

```bash
ATLAS_EXTERNAL_RECURRENT_POLICY_PATH=/configs/external_recurrent_policy.yaml
```

This is intentionally not implemented inside Atlas. External recurrent/refinement is implemented in the benchmark runner so Atlas remains a stateless OpenAI-compatible inference endpoint.

Runner behavior:

- R1: normal generation
- R2: validate then rewrite when `validator_failed` or always when configured
- R3/R4: loop supported by runner CLI, but early experiments should cap at R2

Required metadata is saved by the runner:

- `recurrent_depth`
- `refinement_mode`
- `num_actual_passes`
- `validator_before`
- `validator_after`
- `latency_passes`
- `tokens_per_pass`

## Safety Position

MoE block repeat is not a quality-improvement guarantee. It is an unsafe research knob for testing whether extra intra-layer MoE FFN compute can expose useful local operating points.

Initial constraints:

- `repeat <= safety.max_repeat`
- default `max_repeat = 2`
- `residual_scale > 0`
- `residual_scale <= 1.0`
- `skip=true` forces `repeat=0`
- `residual_scale != 1.0` is rejected for FP32-residual models until an FP32 scaled-add path exists

## Recommended Experiment Order

1. no feature flags
2. `ATLAS_MOE_TOP_K_OVERRIDE=6`
3. block policy with late-layer skip only
4. late-layer repeat with `repeat=2`, `residual_scale=0.5`
5. same repeat with `renorm_between_repeats=true`
6. combined global top-k override plus late repeat

Use only short smoke prompts before running benchmark datasets.

