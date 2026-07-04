// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use super::{ModelWeightLoader, WeightFormat};
use crate::layer::TransformerLayer;
use crate::layers::{DenseFfnLayer, FfnComponent, Qwen3AttentionLayer, Qwen3SsmLayer};
use crate::tp_shard::{TpShardKind, load_qkvo_tp, shard_dense_bf16, shard_quantized_nvfp4};
use crate::weight_map::{
    AttentionWeights, DenseWeight, Fp8Weight, MtpWeights, Nvfp4Variant, SsmWeights, dense,
    dense_auto, dense_f32_safe, dense_keep_f32, dequant_nvfp4_to_bf16, detect_nvfp4_variant,
    gpu_concat_rows, interleave_ba, load_dense_ffn, load_fp8_block_scaled_as_fp8weight,
    load_kv_scales, load_mtp, quantize_to_nvfp4, quantized_auto,
};

mod loaders_b;

pub struct Qwen35DenseWeightLoader;

impl ModelWeightLoader for Qwen35DenseWeightLoader {
    fn supports_tp(&self) -> bool {
        true
    }

    fn load_layers(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        let layer_types = if config.layer_types.is_empty() {
            (0..config.num_hidden_layers)
                .map(|i| config.layer_type(i))
                .collect::<Vec<_>>()
        } else {
            config.layer_types.clone()
        };

        let mut layers: Vec<Box<dyn TransformerLayer>> =
            Vec::with_capacity(config.num_hidden_layers);
        let mut attn_idx = 0usize;

        let absmax_k = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
        let quantize_k = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
        let stream = gpu.default_stream();
        let h = config.hidden_size;

        let variant = detect_nvfp4_variant(store, config);
        let weight_format = WeightFormat::detect(store, config);
        tracing::info!(
            "Weight format: {:?}, NVFP4 variant: {:?}",
            weight_format,
            variant
        );

        let fp8_ssm_prefill = matches!(variant, Nvfp4Variant::Fp8Dequanted);
        let bf16_to_fp8_k = if fp8_ssm_prefill {
            tracing::info!(
                "SSM in_proj_qkv + out_proj via native FP8 prefill GEMM \
                 (BF16 act x FP8 weight via fp8_gemm_n128); NVFP4 kept as \
                 structural fallback for decode batch paths"
            );
            Some(gpu.kernel("w4a16", "bf16_to_fp8")?)
        } else {
            None
        };

        let mem_profile = std::env::var("ATLAS_MEM_PROFILE").is_ok();
        let log_free = |tag: &str| {
            if mem_profile && let Ok(free) = gpu.free_memory() {
                tracing::info!("MEM_PROFILE[{tag}]: {:.2} GB GPU-free", free as f64 / 1e9);
            }
        };
        log_free("dense-load-start");

        for (i, lt) in layer_types.iter().enumerate() {
            if i % 8 == 0 {
                log_free(&format!("layer-{i}"));
            }
            let lp = config.layer_prefix(i);
            let input_norm = dense(store, &format!("{lp}.input_layernorm.weight"))?;
            let post_attn_norm = dense(store, &format!("{lp}.post_attention_layernorm.weight"))?;

            let ffn_weights = load_dense_ffn(
                store, &lp, gpu, variant, absmax_k, quantize_k, stream, config,
            )?;
            let mut dffn = DenseFfnLayer::new(ffn_weights, gpu)?;
            dffn.finalize_q4k_load(gpu, h as u32, config.intermediate_size as u32, stream)?;
            dffn.finalize_nvfp4_mmq_load(gpu, h as u32, config.intermediate_size as u32, stream)?;
            let ffn = FfnComponent::Dense(dffn);

            match lt {
                LayerType::FullAttention => {
                    let p = format!("{lp}.self_attn");
                    let tp_rank = config.tp_rank;
                    let tp_size = config.tp_world_size.max(1);
                    let (attn, q_nvfp4, k_nvfp4, v_nvfp4) = match variant {
                        Nvfp4Variant::CompressedTensors => {
                            let group_size = 16usize;
                            let load_nvfp4 = |name: &str,
                                              full_n: usize,
                                              full_k: usize,
                                              kind: TpShardKind|
                             -> Result<crate::weight_map::QuantizedWeight> {
                                let src = quantized_auto(store, &format!("{p}.{name}"), gpu, variant)?;
                                if tp_size == 1 {
                                    return Ok(src);
                                }
                                let sharded = shard_quantized_nvfp4(
                                    &src, full_n, full_k, kind, tp_rank, tp_size, group_size, gpu,
                                )?;
                                gpu.free(src.weight)?;
                                gpu.free(src.weight_scale)?;
                                Ok(sharded)
                            };
                            let [q, k, v, o] = load_qkvo_tp(config, load_nvfp4)?;
                            let dummy = DenseWeight {
                                weight: spark_runtime::gpu::DevicePtr::NULL,
                            };
                            let (k_scale, v_scale) = load_kv_scales(store, &p, gpu);
                            let attn = AttentionWeights {
                                q_proj: dummy,
                                k_proj: dummy,
                                v_proj: dummy,
                                o_proj: o,
                                q_norm: dense(store, &format!("{p}.q_norm.weight"))?,
                                k_norm: dense(store, &format!("{p}.k_norm.weight"))?,
                                q_norm_full: None,
                                k_norm_full: None,
                                k_scale,
                                v_scale,
                            };
                            (attn, Some(q), Some(k), Some(v))
                        }
                        Nvfp4Variant::Standard
                        | Nvfp4Variant::Fp8Dequanted
                        | Nvfp4Variant::Bf16Raw => {
                            let load_bf16_then_nvfp4 = |name: &str,
                                                        full_n: usize,
                                                        full_k: usize,
                                                        kind: TpShardKind|
                             -> Result<(
                                DenseWeight,
                                crate::weight_map::QuantizedWeight,
                            )> {
                                let src = dense_auto(store, &format!("{p}.{name}.weight"), gpu)?;
                                let (sharded_ptr, local_n, local_k) = shard_dense_bf16(
                                    src.weight, full_n, full_k, kind, tp_rank, tp_size, gpu,
                                )?;
                                let sharded = DenseWeight {
                                    weight: sharded_ptr,
                                };
                                let q = quantize_to_nvfp4(
                                    &sharded, local_n, local_k, gpu, absmax_k, quantize_k, stream,
                                )?;
                                if sharded_ptr != src.weight {
                                    gpu.free(sharded_ptr)?;
                                }
                                Ok((src, q))
                            };
                            let [
                                (q_dense, q_nvfp4),
                                (k_dense, k_nvfp4),
                                (v_dense, v_nvfp4),
                                (o_dense, o_nvfp4),
                            ] = load_qkvo_tp(config, load_bf16_then_nvfp4)?;

                            let (k_scale, v_scale) = load_kv_scales(store, &p, gpu);

                            gpu.free(q_dense.weight)?;
                            gpu.free(k_dense.weight)?;
                            gpu.free(v_dense.weight)?;
                            gpu.free(o_dense.weight)?;

                            let attn = AttentionWeights {
                                q_proj: DenseWeight {
                                    weight: spark_runtime::gpu::DevicePtr::NULL,
                                },
                                k_proj: DenseWeight {
                                    weight: spark_runtime::gpu::DevicePtr::NULL,
                                },
                                v_proj: DenseWeight {
                                    weight: spark_runtime::gpu::DevicePtr::NULL,
                                },
                                o_proj: o_nvfp4,
                                q_norm: dense(store, &format!("{p}.q_norm.weight"))?,
                                k_norm: dense(store, &format!("{p}.k_norm.weight"))?,
                                q_norm_full: None,
                                k_norm_full: None,
                                k_scale,
                                v_scale,
                            };
                            (attn, Some(q_nvfp4), Some(k_nvfp4), Some(v_nvfp4))
                        }
                    };

                    let mut attn_layer = Qwen3AttentionLayer::new(
                        input_norm,
                        attn,
                        post_attn_norm,
                        ffn,
                        attn_idx,
                        q_nvfp4,
                        k_nvfp4,
                        v_nvfp4,
                        gpu,
                        layer_kv_dtypes[attn_idx],
                        config.fp8_kv_calibration_tokens,
                        config,
                    )?;
                    if let (Some(qw), Some(kw), Some(vw)) = (q_nvfp4, k_nvfp4, v_nvfp4) {
                        let nh = config.num_attention_heads;
                        let nkv = config.num_key_value_heads;
                        let hd = config.head_dim;
                        let hh = config.hidden_size;
                        let q_n = if config.attn_gated {
                            nh * hd * 2
                        } else {
                            nh * hd
                        };
                        let qt = qw.transpose_for_gemm(gpu, q_n, hh)?;
                        let kt = kw.transpose_for_gemm(gpu, nkv * hd, hh)?;
                        let vt = vw.transpose_for_gemm(gpu, nkv * hd, hh)?;
                        let ot = attn_layer
                            .attn
                            .o_proj
                            .transpose_for_gemm(gpu, hh, nh * hd)?;
                        attn_layer.set_prefill_weights(Some(qt), Some(kt), Some(vt), Some(ot));
                    }
                    layers.push(Box::new(attn_layer));
                    attn_idx += 1;
                }
                LayerType::LinearAttention => {
                    let nv = config.linear_num_value_heads;
                    let nk = config.linear_num_key_heads;
                    let qkv_rows = config.ssm_qkv_size();
                    let z_rows = config.ssm_z_size();
                    let value_dim = nv * config.linear_value_head_dim;
                    let la = format!("{lp}.linear_attn");

                    let load_ssm_proj =
                        |name: &str, rows: usize, cols: usize| -> Result<DenseWeight> {
                            if store.contains(&format!("{name}.weight_packed")) {
                                dequant_nvfp4_to_bf16(store, name, rows, cols, gpu)
                            } else {
                                dense_auto(store, &format!("{name}.weight"), gpu)
                            }
                        };
                    let qkv_dense = load_ssm_proj(&format!("{la}.in_proj_qkv"), qkv_rows, h)?;
                    let z_dense = load_ssm_proj(&format!("{la}.in_proj_z"), z_rows, h)?;
                    let out_proj_dense = load_ssm_proj(&format!("{la}.out_proj"), h, value_dim)?;

                    let in_proj_a = dense(store, &format!("{la}.in_proj_a.weight"))?;
                    let in_proj_b = dense(store, &format!("{la}.in_proj_b.weight"))?;
                    let conv1d = dense(store, &format!("{la}.conv1d.weight"))?;
                    let a_log = dense_keep_f32(store, &format!("{la}.A_log"), gpu)?;
                    let dt_bias = dense_keep_f32(store, &format!("{la}.dt_bias"), gpu)?;
                    let norm = dense_f32_safe(store, &format!("{la}.norm.weight"), gpu)?;

                    let qkvz_dense =
                        gpu_concat_rows(&qkv_dense, qkv_rows, &z_dense, z_rows, h, gpu)?;
                    gpu.free(qkv_dense.weight)?;
                    gpu.free(z_dense.weight)?;

                    let ba_dense = interleave_ba(&in_proj_a, &in_proj_b, nv, nk, h, gpu)?;

                    let qkvz_size = config.ssm_qkvz_size();
                    let qkvz_nvfp4 = quantize_to_nvfp4(
                        &qkvz_dense,
                        qkvz_size,
                        h,
                        gpu,
                        absmax_k,
                        quantize_k,
                        stream,
                    )?;
                    let qkvz_nvfp4_t = qkvz_nvfp4.transpose_for_gemm(gpu, qkvz_size, h)?;

                    let out_proj_nvfp4 = quantize_to_nvfp4(
                        &out_proj_dense,
                        h,
                        value_dim,
                        gpu,
                        absmax_k,
                        quantize_k,
                        stream,
                    )?;
                    let out_proj_nvfp4_t = out_proj_nvfp4.transpose_for_gemm(gpu, h, value_dim)?;

                    let (qkvz_fp8_prefill, out_proj_fp8_prefill) =
                        if let Some(b2f_k) = bf16_to_fp8_k {
                            let qkvz_total = (qkvz_size * h) as u32;
                            let qkvz_fp8 = gpu.alloc(qkvz_size * h)?;
                            crate::layers::ops::bf16_to_fp8(
                                gpu,
                                b2f_k,
                                qkvz_dense.weight,
                                qkvz_fp8,
                                qkvz_total,
                                stream,
                            )?;
                            let out_total = (h * value_dim) as u32;
                            let out_fp8 = gpu.alloc(h * value_dim)?;
                            crate::layers::ops::bf16_to_fp8(
                                gpu,
                                b2f_k,
                                out_proj_dense.weight,
                                out_fp8,
                                out_total,
                                stream,
                            )?;
                            gpu.synchronize(stream)?;
                            (Some(qkvz_fp8), Some(out_fp8))
                        } else {
                            (None, None)
                        };

                    gpu.free(qkvz_dense.weight)?;
                    gpu.free(out_proj_dense.weight)?;

                    let ssm = SsmWeights {
                        in_proj_qkvz: DenseWeight {
                            weight: spark_runtime::gpu::DevicePtr::NULL,
                        },
                        in_proj_ba: ba_dense,
                        conv1d,
                        a_log,
                        dt_bias,
                        norm,
                        out_proj: out_proj_nvfp4,
                    };

                    let mut layer = Qwen3SsmLayer::new_sequential(
                        input_norm,
                        ssm,
                        post_attn_norm,
                        ffn,
                        Some(qkvz_nvfp4),
                        Some(qkvz_nvfp4_t),
                        Some(out_proj_nvfp4_t),
                        config,
                        gpu,
                    )?;
                    layer.predequant_for_prefill(gpu, config, stream)?;
                    if qkvz_fp8_prefill.is_some() || out_proj_fp8_prefill.is_some() {
                        layer.set_fp8_prefill_only_weights(qkvz_fp8_prefill, out_proj_fp8_prefill);
                    }
                    layers.push(Box::new(layer));
                }
                LayerType::SlidingAttention => {
                    unreachable!("unexpected SlidingAttention in this loader")
                }
                LayerType::Moe => unreachable!("Qwen3.5 dense has no standalone MoE layers"),
            }

            if (i + 1) % 10 == 0 {
                tracing::info!("Loaded layers 0..{}", i + 1);
            }
        }

        tracing::info!(
            "Qwen3.5 dense weight loader: {} layers ({} attention, {} SSM, dense FFN)",
            layers.len(),
            attn_idx,
            layers.len() - attn_idx,
        );

        Ok(layers)
    }

    fn load_embedding(&self, store: &WeightStore, config: &ModelConfig) -> Result<DenseWeight> {
        loaders_b::load_embedding(store, config)
    }

    fn load_final_norm(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        loaders_b::load_final_norm(store, config)
    }

    fn load_lm_head(&self, store: &WeightStore, config: &ModelConfig) -> Result<DenseWeight> {
        loaders_b::load_lm_head(store, config)
    }

    fn load_mtp_weights(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<Option<MtpWeights>> {
        if !store.contains("mtp.fc.weight") {
            return Ok(None);
        }
        let variant = detect_nvfp4_variant(store, config);
        tracing::info!(
            "Loading dense MTP weights (variant={:?}, hidden={}, inter={})",
            variant,
            config.hidden_size,
            config.intermediate_size,
        );
        let mtp = load_mtp(store, config.num_experts, gpu, variant)?;
        if mtp.dense_ffn.is_some() {
            tracing::info!("Dense MTP head ready (FP8 e4m3 projections + dense gate/up/down MLP)");
        } else {
            tracing::info!(
                "MoE MTP head ready ({} experts) — dense loader sees MoE bundle",
                mtp.experts.len(),
            );
        }
        Ok(Some(mtp))
    }
}
