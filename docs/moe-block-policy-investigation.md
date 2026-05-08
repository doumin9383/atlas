# MoE Block Skip/Repeat Policy Investigation

Status: investigation only. No runtime behavior is changed by this document.

## Question

Can Atlas expose MoE FFN block skip/repeat as an experimental policy knob, similar in spirit to the fixed MoE top-k override, without changing default inference behavior?

Target model family: Qwen3.5/Qwen3.6 35B-A3B NVFP4 first.

## Relevant Code Paths

- Transformer layer trait: `crates/spark-model/src/layer/transformer_layer.rs`
- Shared forward context: `crates/spark-model/src/layer.rs`
- Qwen attention layer struct: `crates/spark-model/src/layers/qwen3_attention/types.rs`
- Qwen attention decode: `crates/spark-model/src/layers/qwen3_attention/trait_impl/decode_inner.rs`
- Qwen attention prefill: `crates/spark-model/src/layers/qwen3_attention/trait_impl/prefill_inner.rs`
- FFN dispatch enum: `crates/spark-model/src/layers/mod.rs`
- MoE decode: `crates/spark-model/src/layers/moe/forward.rs`
- MoE prefill: `crates/spark-model/src/layers/moe/forward_prefill.rs`

## Findings

### 1. Transformer layer forward structure

`TransformerModel` iterates `self.layers` and calls `TransformerLayer::decode`, `prefill`, or `decode_multi_seq`. For Qwen3.5, each transformer layer is represented by `Qwen3AttentionLayer`, not by separate attention and MoE layer objects.

`Qwen3AttentionLayer` owns:

- `input_norm`
- attention weights and kernels
- `post_attn_norm`
- `ffn: FfnComponent`

For Qwen3.5 MoE models, `ffn` is `FfnComponent::Moe(MoeLayer)`.

### 2. Attention block and MoE FFN block are separated enough

In both decode and prefill, Atlas performs:

1. input RMSNorm and residual save
2. attention and KV cache update
3. attention residual add + post-attention RMSNorm into `normed2`
4. FFN/MoE execution
5. residual add of FFN/MoE output into `hidden`

The MoE FFN call is explicit:

- decode: `self.ffn.forward(normed2, ctx, stream)?`
- prefill: `self.ffn.forward_prefill(ctx.buffers.norm_output(), num_tokens, ctx, stream)?`, output read from `ctx.buffers.moe_output()`

This means an MoE-only hook can be placed after attention has already updated KV cache and before the FFN residual add.

### 3. MoE FFN skip feasibility

MoE-only skip is feasible for Qwen3.5.

For a selected layer, the safe skip semantics should be:

- still run input norm and attention
- still run `residual_add_rms_norm(hidden, attn_out, post_attn_norm, normed2, residual, ...)`
- do not call `self.ffn.forward*`
- do not add any FFN output into `hidden`
- continue to the next transformer layer

This produces:

```text
hidden_out = hidden_in + attention_out
```

with the usual post-attention normalization computed only as an intermediate. It does not affect KV cache writes because those happen in the attention step before the MoE branch.

Risk: this changes model semantics strongly and may destabilize generation. It should be gated behind an explicit policy file and smoke-tested only.

### 4. MoE FFN repeat feasibility

MoE-only repeat is feasible, but there are two possible semantics:

#### A. Same-input repeated FFN contribution

Run the same MoE FFN on the same `normed2` input multiple times, accumulating the FFN output into `hidden` each time:

```text
hidden = hidden + scale * MoE(normed2)
hidden = hidden + scale * MoE(normed2)
```

This is the smallest patch. It does not require re-running RMSNorm between repeats. For deterministic routing it will usually repeat the same experts and same output, so `repeat=2, residual_scale=0.5` may be almost identical to `repeat=1, residual_scale=1.0` except for BF16 rounding and any nondeterminism.

#### B. Recurrent-style re-normalized FFN repeat

After each FFN residual add, recompute `normed2 = RMSNorm(hidden, post_attn_norm)` and run MoE again:

```text
hidden = hidden + scale * MoE(normed2_1)
normed2_2 = RMSNorm(hidden, post_attn_norm)
hidden = hidden + scale * MoE(normed2_2)
```

This is a more meaningful internal refinement experiment because the second MoE pass sees an updated residual stream. It is also riskier and needs an extra norm call per repeat. For the first patch, this should be a separate mode such as `repeat_input: renorm` rather than hidden behind `repeat`.

Recommendation: start with skip plus same-input repeat to validate plumbing. Then add a `renorm_between_repeats` flag if smoke tests are stable.

### 5. RMSNorm and residual handling

The normal Qwen3.5 decode path uses:

```text
rms_norm_residual(hidden, input_norm) -> normed, residual
attention(normed) -> attn_out
residual_add_rms_norm(hidden, attn_out, post_attn_norm) -> normed2, residual
moe_out = ffn.forward(normed2)
residual_add(hidden, moe_out)
```

The prefill path follows the same structure batched over tokens, with `ffn.forward_prefill(...)` writing to `ctx.buffers.moe_output()`.

For `residual_scale != 1.0`, Atlas already has a wrapper for `bf16_scaled_add` in `crates/spark-model/src/layers/ops/activations.rs`, but `Qwen3AttentionLayer` currently does not hold a `scaled_add` kernel handle. A patch needs to add a `scaled_add_k` field loaded from `residual_add::bf16_scaled_add`.

If `config.use_fp32_residual()` is active, the current `residual_add_k` selection may use `f32_residual_add`. A BF16-only scaled add is not enough for all model types. Since the first target is Qwen3.5 35B-A3B NVFP4, confirm whether it uses BF16 or FP32 residual before enabling `residual_scale != 1.0`. If FP32 residual is active, add an FP32 scaled-add kernel or reject scaled policies at startup.

### 6. KV cache and position encoding impact

MoE skip/repeat after attention should not directly affect:

- KV cache allocation
- K/V writes
- RoPE/MRoPE position handling
- block tables
- high-speed-swap attention block movement

The KV cache is written during the attention step before the MoE FFN branch. Repeating MoE FFN does not write to KV cache.

Indirect effect: output hidden states change, so future layer inputs and future token logits change. That is the intended experiment.

### 7. Layer/group policy feasibility

Layer/group policy is feasible, but it should not mutate global `ModelConfig.num_experts_per_tok` per layer. The existing top-k override is global because all MoE kernels read `ctx.config.num_experts_per_tok`.

Per-layer top-k requires either:

1. passing an effective top-k into `MoeLayer::forward*`, or
2. adding per-layer policy to `ForwardContext` and teaching `MoeLayer` to resolve effective top-k from its `layer_idx`.

Because `MoeLayer` already has `layer_idx` for router stats in the Qwen3.5 loader, option 2 is practical.

## Proposed Minimal Patch

### Environment gate

```text
ATLAS_MOE_BLOCK_POLICY_PATH=/configs/moe_block_policy.yaml
```

No env var means no policy loaded and no behavior change.

### Policy shape

```yaml
default:
  top_k: null
  repeat: 1
  skip: false
  residual_scale: 1.0
  renorm_between_repeats: false
layer_groups:
  - name: late_repeat
    layers: [24,25,26,27]
    top_k: 6
    repeat: 2
    skip: false
    residual_scale: 0.5
    renorm_between_repeats: false
```

### Runtime structs

Add in `spark-model`:

```rust
pub struct MoeBlockPolicy {
    pub default: MoeBlockLayerPolicy,
    pub by_layer: Vec<MoeBlockLayerPolicy>,
}

pub struct MoeBlockLayerPolicy {
    pub top_k: Option<usize>,
    pub repeat: usize,
    pub skip: bool,
    pub residual_scale: f32,
    pub renorm_between_repeats: bool,
}
```

Add `moe_block_policy: Option<&MoeBlockPolicy>` to `ForwardContext`.

### Hook placement

Patch only `Qwen3AttentionLayer` initially:

- `decode_inner`
- `prefill_inner`
- later: `decode_multi_seq_inner` if small-batch serving uses it in the target workload

Pseudo-flow:

```rust
let policy = ctx.moe_block_policy.and_then(|p| p.for_layer(self.attn_layer_idx));

if policy.skip {
    log_once(layer, "skip");
    return Ok(());
}

for repeat_idx in 0..policy.repeat {
    if repeat_idx > 0 && policy.renorm_between_repeats {
        rms_norm(hidden, post_attn_norm, normed2)
    }
    let moe_out = self.ffn.forward(normed2, ctx, stream)?;
    if policy.residual_scale == 1.0 {
        residual_add(hidden, moe_out)
    } else {
        scaled_add(hidden, moe_out, policy.residual_scale)
    }
}
```

For prefill, use `forward_prefill(...)` and `ctx.buffers.moe_output()` for each repeat.

### Per-layer top-k

The global top-k override patch currently changes `ModelConfig.num_experts_per_tok` before model build. For layer policy, do not keep mutating global config during forward.

Recommended next patch:

- keep global top-k override as-is
- add `MoeLayer::effective_top_k(ctx)`:
  - if block policy has `top_k` for `self.layer_idx`, use it
  - else use `ctx.config.num_experts_per_tok`
- validate `top_k <= num_experts` at policy load time
- ensure scratch sizing uses the maximum top-k across the policy, not only the default top-k

Scratch sizing is the main safety issue. Several model paths allocate or offset scratch based on `config.num_experts_per_tok`, including prefill/decode metadata scratch sizing. If per-layer top-k can exceed the config default, startup must either:

- set `config.num_experts_per_tok` to the maximum policy top-k for buffer sizing while letting per-layer effective top-k choose smaller values, or
- add a separate `max_moe_top_k` config used only for buffers.

For the first smoke patch, prefer not combining per-layer top-k with block repeat. Test skip/repeat using default top-k first.

## Logging

At startup:

- policy path
- number of layer overrides
- max effective top-k
- whether scaled residual is allowed for the model residual dtype

Per layer, log once:

- layer
- top_k
- repeat
- skip
- residual_scale
- renorm_between_repeats

Per request/token logging should remain off by default; use aggregate logs first to avoid large I/O overhead.

## Minimal Experiment Order

Run short prompts only:

1. default policy absent
2. skip one late MoE FFN layer
3. skip a small late group, e.g. `[24,25,26,27]`
4. repeat one late MoE FFN layer with `repeat=2`, `residual_scale=1.0`
5. repeat same layer with `repeat=2`, `residual_scale=0.5`
6. only after that: `top_k=6 + repeat=2`

Use one model Pod and one request at a time. Do not enable this in the production namespace.

## Recommendation

Implementability: yes, for Qwen3.5/Qwen3.6 attention-backed MoE FFN blocks.

Risk level: high. Skip is straightforward; repeat is semantically experimental; per-layer top-k combined with repeat requires buffer sizing care.

First patch should include:

- policy file parser and startup validation
- skip/repeat/residual_scale for Qwen3.5 `decode_inner` and `prefill_inner`
- no per-layer top-k yet, unless buffer sizing is updated to max policy top-k
- feature flag default off
- smoke-test-only docs

