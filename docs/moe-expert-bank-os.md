# MoE Expert-Bank OS

This note captures a proposed extension to Atlas/MochiHop: treating a MoE model as a local substrate made from a stable trunk, swappable routed expert banks, and explicit routing profiles.

## Summary

Atlas is the inference engine. MochiHop is the runtime-elastic MoE control plane. Expert-Bank OS adds composition elasticity: the ability to register, map, graft, materialize, and eventually hot-switch expert banks and routing profiles.

```text
AtlasModel = Trunk + SharedCore + ExpertBank + RoutingProfile + RuntimePolicy
```

The goal is not to merge whole checkpoints blindly. The goal is to make MoE composition explicit and inspectable.

## Core decomposition

| Layer | Contents | Default policy | Purpose |
| --- | --- | --- | --- |
| Trunk / Core | embedding, attention, SSM/DeltaNet, norms, lm_head, vision tower | fixed | common body and cognition substrate |
| SharedCore | shared experts and other always-on common paths | fixed at first | common stable behavior across profiles |
| ExpertBank | routed experts from base, donor, or merged sources | swappable | skill, style, and domain material |
| RoutingProfile | top-k, masks, remaps, biases, fallback rules | selectable | behavior/personality/control policy |
| RuntimePolicy | residency, tiers, KV/attention budget, memory pressure response | dynamic | speed/memory/quality operating point |

Design note: shared experts are treated as SharedCore rather than removable ExpertBank entries. They affect every token and every profile, so keeping them outside the swappable routed bank preserves profile isolation.

## Relation to MochiHop

MochiHop currently aims at runtime elasticity: make a single MoE model behave like multiple smaller operating points by changing top-k, residency, hot/warm/cold tiers, attention budget, and KV policy.

Expert-Bank OS extends this into composition elasticity:

| MochiHop runtime elasticity | Expert-Bank OS composition elasticity |
| --- | --- |
| change how many experts are active | change which physical experts logical slots resolve to |
| dynamic MoE top-k | profile-level top-k plus mask/remap/bias |
| hot/warm/cold residency | source/domain/compatibility/residency metadata |
| weighted reranking | router score plus profile, memory, and domain terms |
| operating modes | full routing profiles / apparent model personas |
| elasticity benchmark | profile and graft frontier search |

In short: MochiHop controls how much of the MoE is active; Expert-Bank OS controls what the MoE is made of and how it is wired.

## Logical versus physical experts

Do not treat the router's expert index as a direct physical array index.

```text
router output: logical expert id
  -> RoutingProfile / ExpertIndexResolver
  -> physical expert ref
  -> dispatch / materialize / fallback
```

Example:

| Profile | layer 24 logical expert 193 resolves to |
| --- | --- |
| base | qwen3.6 layer 24 expert 193 |
| coder | qwen-coder-next layer 24 expert 37 |
| merged | 0.9 * qwen3.6 expert 193 + 0.1 * qwen-coder-next expert 37 |
| low_resource | nearest warm resident expert |
| emergency | base fallback, skip, or shared-only fallback |

This makes cross-model expert grafting and same-model memory-elastic fallback use the same abstraction.

## Required components

| Component | Responsibility |
| --- | --- |
| `MoE schema scanner` | discover layers, routed experts, shared experts, tensor names, shapes, dtypes |
| `CanonicalExpertSchema` | normalize model-specific state_dict keys into canonical refs |
| `CompatibilityScanner` | classify tensors as replaceable, mergeable, adapter-required, or incompatible |
| `ExpertBankRegistry` | manage base, donor, merged, resident, offloaded, and materialized experts |
| `ExpertIndexResolver` | map logical expert ids to physical expert refs using profile rules |
| `RoutingProfile` | define mask, remap, bias, top-k, rerank, fallback, and runtime policies |
| `ProfileMaterializer` | emit a normal checkpoint after applying a profile |
| `RuntimeProfileSwitch` | eventual hot-switching without writing a new checkpoint |
| `MoeTrace` / telemetry | record router logits, ranks, selected experts, score gaps, and usage |
| `ProfileEval` | compare quality, latency, memory, cold misses, and routing stability |

## Resolver types

Initial resolvers should be simple and explicit:

- `IdentityResolver`: `i -> i`
- `RangeRemapResolver`: map a logical range to a source bank range
- `OffsetResolver`: map into an appended expert bank
- `StaticTableResolver`: profile-owned per-layer mapping table
- `TierAwareResolver`: substitute cold experts with hot/warm alternatives
- `FallbackResolver`: return to base if mapping fails

Later resolvers can be data-driven:

- `WeightSimilarityResolver`
- `ActivationSimilarityResolver`
- `UsageAwareResolver`
- `DomainAwareResolver`
- `EvalOptimizedResolver`

## MVP: Qwen3.6 plus Qwen-Coder-Next

Target experiment:

| Field | Value |
| --- | --- |
| base trunk | Qwen3.6-35B-A3B |
| donor bank | Qwen-Coder-Next-80B-A3B routed experts |
| fixed | trunk, shared experts, router, embedding, lm_head, vision tower |
| mutable | routed experts only |
| first mode | checkpoint materialization, not runtime hot-swap |
| initial graft | late-layer partial expert merge |

Example profile:

```yaml
name: coder_late64
base_trunk: qwen3.6-35b-a3b

runtime:
  moe_top_k:
    min: 6
    max: 8
  residency: standard

expert_banks:
  base:
    source: qwen3.6-35b-a3b
  coder:
    source: qwen-coder-next-80b-a3b

shared_core:
  source: base
  mutable: false

routing:
  router_source: base
  router_bias:
    enabled: false

rules:
  - layers: [24, 39]
    logical_experts: [192, 255]
    resolver:
      type: range_remap
      source_bank: coder
      from: [192, 255]
      to: [0, 63]
    mode: merge
    alpha: 0.10
    fallback:
      source_bank: base
      resolver: identity

telemetry:
  trace_router_scores: true
  trace_expert_usage: true
  trace_quality_proxy: true
```

The first success criterion is not benchmark dominance. It is controlled survival: the materialized model loads, routes predictably, preserves baseline behavior enough to compare, and shows measurable changes on code-oriented prompts.

## Evaluation questions

- How much can routed experts be modified before general behavior collapses?
- Do late-layer expert grafts preserve base behavior better than early-layer grafts?
- Does thin linear merge outperform hard replacement?
- Can router telemetry predict which logical slots are safe to graft?
- Can donor experts improve a domain without touching trunk or shared core?
- Can profile-level remapping behave like multiple apparent models from one trunk?

## Roadmap connection

This document extends the existing MochiHop roadmap. Runtime elasticity comes first: telemetry, top-k separation, tiering, residency, and sweep benchmarks. Composition elasticity builds on that foundation: expert registries, canonical schemas, resolvers, profiles, materializers, runtime grafting, and eval-driven profile search.
