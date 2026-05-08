# MoE Block Policy

Set:

```bash
ATLAS_MOE_BLOCK_POLICY_PATH=/configs/moe_block_policy.yaml
```

Unset means disabled.

## Example

```yaml
default:
  skip: false
  repeat: 1
  residual_scale: 1.0
  renorm_between_repeats: false

safety:
  max_repeat: 2
  fallback_on_nan: true
  log_hidden_norm: true

layer_groups:
  - name: late_repeat
    layers: [24,25,26,27]
    skip: false
    repeat: 2
    residual_scale: 0.5
    renorm_between_repeats: true

  - name: early_skip
    layers: [0,1,2,3]
    skip: true
    repeat: 0
```

## Semantics

Normal Qwen MoE layer flow:

```text
input RMSNorm
attention + KV cache update
hidden += attention_out, then RMSNorm -> normed2
moe_out = MoE(normed2)
hidden += moe_out
```

Skip:

```text
input RMSNorm
attention + KV cache update
hidden += attention_out, then RMSNorm -> normed2
do not run MoE FFN
do not add MoE residual
```

Repeat without renorm:

```text
hidden += residual_scale * MoE(normed2)
hidden += residual_scale * MoE(normed2)
```

Repeat with renorm:

```text
hidden += residual_scale * MoE(normed2_1)
normed2_2 = RMSNorm(hidden)
hidden += residual_scale * MoE(normed2_2)
```

## Validation

Startup fails if:

- `repeat > safety.max_repeat`
- `residual_scale <= 0`
- `residual_scale > 1.0`
- a layer index is outside model layer count
- `top_k` is set to anything other than `null`
- `residual_scale != 1.0` is used on an FP32-residual model

`skip=true` normalizes the effective repeat to `0`.

## Logging

Startup logs:

- policy path
- skip layers
- repeat layers
- `max_repeat`
- `fallback_on_nan`
- `log_hidden_norm`

Each affected layer logs once when policy is applied:

- layer index
- skip
- repeat
- residual scale
- renorm flag

When `log_hidden_norm=true`, Atlas samples up to 4096 BF16 hidden elements after the policy path and logs a debug norm. The same sampling path fails the request if a non-finite value is observed. This is detection, not true rollback.

## Scope

Initial implementation targets Qwen3.5/Qwen3.6 `Qwen3AttentionLayer` with `FfnComponent::Moe`.

The policy is intentionally not applied to:

- dense FFN layers
- Gemma dual-FFN layers
- standalone Nemotron MoE layers
- MTP head MoE

