# Atlas MoE Roadmap

This roadmap groups the MochiHop runtime-elastic MoE work and the proposed Expert-Bank OS composition-elastic extension.

## Mental model

```text
Atlas
  = local inference engine

MochiHop
  = runtime-elastic MoE control plane

Expert-Bank OS
  = composition-elastic MoE control plane
```

MochiHop answers: how much of a MoE model should be active under current latency, memory, and quality pressure?

Expert-Bank OS answers: which physical experts should a logical router slot resolve to, and how can expert banks from base, donor, or merged sources be composed safely?

## Stage overview

| Stage | Area | Status | Purpose |
| ---: | --- | --- | --- |
| 0 | MoE top-k separation | existing issue / completed fix path | separate sampling top-k from MoE dispatch top-k |
| 1 | Router telemetry | issue #3 | observe router scores and expert usage |
| 2 | Runtime elasticity | issue #4 | dynamic top-k, tiering, residency, modes |
| 3 | Attention/KV pressure | issue #5 | adapt context, attention budget, KV cache policy |
| 4 | Auxiliary co-residency | issue #6 | shrink/recover LLM while hosting aux models |
| 5 | Elasticity benchmark | issue #7 | sweep operating points and produce frontier profiles |
| 6 | Expert bank registry | issue #32 | register base, donor, merged, resident, and offloaded experts |
| 7 | Canonical MoE schema | issue #33 | normalize model-specific expert structures and state_dict keys |
| 8 | Expert index resolver | issue #34 | map logical router ids to physical expert refs |
| 9 | Profile materializer | issue #35 | emit normal checkpoints from a profile for MVP testing |
| 10 | Runtime grafting | issue #36 | dispatch physical experts dynamically without writing checkpoints |
| 11 | Auto resolver/search | issue #37 | use telemetry, similarity, activation, and eval to build profiles |

## Runtime elasticity track

The existing MochiHop issues form the runtime track:

1. expose router telemetry and expert usage,
2. adjust MoE top-k and expert residency under pressure,
3. adapt attention and KV-cache budget,
4. shrink and recover for auxiliary model co-residency,
5. sweep the quality/latency/memory frontier.

This track should remain focused on making a single supported MoE model operate across many resource envelopes.

## Composition elasticity track

The new Expert-Bank OS track builds on the same telemetry and residency primitives, but generalizes them from same-model compression to cross-model composition.

Key additions:

- `ExpertBankRegistry`
- `CanonicalExpertSchema`
- `CompatibilityScanner`
- `ExpertIndexResolver`
- `RoutingProfile`
- `ExpertGraftProfile`
- `ProfileMaterializer`
- `RuntimeProfileSwitch`
- `ProfileEval`

The first implementation should prefer materialized checkpoints over runtime hot-swap. Runtime grafting can come after schema, resolver, and evaluation tools are stable.

## Shared expert policy

Shared experts and other always-on paths are treated as part of `SharedCore`, not as removable expert-bank entries. This keeps routed expert banks swappable without changing the common behavior of every profile.

```text
Trunk/Core = embedding + attention/SSM + norms + lm_head + vision + SharedCore
ExpertBank = routed experts only
RoutingProfile = how logical expert slots resolve and rerank
```

## MVP target

Initial target: Qwen3.6-35B-A3B as the base trunk and Qwen-Coder-Next-80B-A3B as a donor routed expert bank.

Constraints:

- keep Qwen3.6 trunk fixed,
- keep shared experts fixed,
- keep router fixed initially,
- touch only routed experts,
- materialize a Qwen3.6-compatible checkpoint,
- begin with late-layer partial grafts and small merge alphas.

Suggested first profile:

```text
layers: 24..39
logical expert slots: 192..255
donor experts: qwen-coder-next experts 0..63
mode: linear merge
alpha: 0.05, 0.10, 0.15
fallback: base identity
```

## Evaluation requirements

Every stage that changes routing or expert composition should produce:

- effective MoE top-k,
- selected expert histogram,
- score-gap distribution,
- hot/warm/cold transitions,
- cold miss / fallback rate,
- latency and throughput,
- memory footprint,
- quality proxy,
- prompt examples for regression review.

The same sweep harness used for MochiHop elasticity should eventually compare Expert-Bank OS profiles as well.

## Related design note

See [`docs/moe-expert-bank-os.md`](./moe-expert-bank-os.md) for the longer idea memo and the logical-versus-physical expert abstraction.
