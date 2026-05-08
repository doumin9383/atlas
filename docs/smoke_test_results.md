# Smoke Test Results

Date: 2026-05-08

Environment used for code validation:

- host: development machine, not DGX Spark runtime
- command: `ATLAS_SKIP_BUILD=1 CUDARC_CUDA_VERSION=12080 cargo check -p spark-server`
- result: passed

## Compile Checks

```text
cargo fmt --all
ATLAS_SKIP_BUILD=1 CUDARC_CUDA_VERSION=12080 cargo check -p spark-server
```

Result:

```text
Finished `dev` profile [unoptimized + debuginfo]
```

## Runtime Smoke Matrix

These require a GPU runtime image with the patched Atlas build. They have not been executed on DGX Spark from this dev host yet.

1. default policy, short prompt
2. top-k override only
3. skip late layers only
4. repeat late layers, `repeat=2`, `residual_scale=0.5`
5. repeat late layers with `renorm_between_repeats=true`
6. combined: `ATLAS_MOE_TOP_K_OVERRIDE=6` plus late repeat
7. invalid policy should fail at startup

## Example Commands

Default:

```bash
spark serve Sehyo/Qwen3.5-35B-A3B-NVFP4 --bind 0.0.0.0 --port 8888
```

Top-k:

```bash
ATLAS_MOE_TOP_K_OVERRIDE=6 \
spark serve Sehyo/Qwen3.5-35B-A3B-NVFP4 --bind 0.0.0.0 --port 8888
```

Block policy:

```bash
ATLAS_MOE_BLOCK_POLICY_PATH=/configs/moe_block_policy.yaml \
spark serve Sehyo/Qwen3.5-35B-A3B-NVFP4 --bind 0.0.0.0 --port 8888
```

Combined:

```bash
ATLAS_MOE_TOP_K_OVERRIDE=6 \
ATLAS_MOE_BLOCK_POLICY_PATH=/configs/moe_block_policy.yaml \
spark serve Sehyo/Qwen3.5-35B-A3B-NVFP4 --bind 0.0.0.0 --port 8888
```

## Expected Startup Failure Example

```yaml
safety:
  max_repeat: 2
layer_groups:
  - name: invalid
    layers: [24]
    repeat: 3
```

Expected result: startup error before serving requests.

