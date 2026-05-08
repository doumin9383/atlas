// SPDX-License-Identifier: AGPL-3.0-only

//! Experimental MoE FFN block policy application for Qwen attention layers.

use anyhow::{Result, bail};
use spark_runtime::gpu::DevicePtr;

use super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::moe_block_policy::{MoeBlockLayerPolicy, global_policy};

impl Qwen3AttentionLayer {
    pub(super) fn moe_block_policy(&self) -> Option<MoeBlockLayerPolicy> {
        if !self.ffn.is_moe() || self.moe_ffn.is_some() {
            return None;
        }
        global_policy().map(|policy| policy.for_layer(self.attn_layer_idx))
    }

    pub(super) fn apply_moe_block_decode(
        &self,
        hidden: DevicePtr,
        normed2: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let Some(policy) = self.moe_block_policy() else {
            let moe_out = self.ffn.forward(normed2, ctx, stream)?;
            return self.add_moe_residual(
                hidden,
                moe_out,
                ctx.config.hidden_size,
                1.0,
                ctx,
                stream,
            );
        };
        self.log_policy_once(policy);
        if policy.skip || policy.repeat == 0 {
            return Ok(());
        }
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        for repeat_idx in 0..policy.repeat {
            if repeat_idx > 0 && policy.renorm_between_repeats {
                ops::rms_norm(
                    ctx.gpu,
                    self.rms_norm_k,
                    hidden,
                    &self.post_attn_norm,
                    normed2,
                    1,
                    h as u32,
                    eps,
                    stream,
                )?;
            }
            let moe_out = self.ffn.forward(normed2, ctx, stream)?;
            self.add_moe_residual(hidden, moe_out, h, policy.residual_scale, ctx, stream)?;
        }
        self.check_hidden_if_requested(hidden, h, ctx, stream, "decode")
    }

    pub(super) fn apply_moe_block_prefill(
        &self,
        hidden: DevicePtr,
        normed2: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        let Some(policy) = self.moe_block_policy() else {
            self.ffn.forward_prefill(normed2, num_tokens, ctx, stream)?;
            return Ok(ctx.buffers.moe_output());
        };
        self.log_policy_once(policy);
        if policy.skip || policy.repeat == 0 {
            return Ok(DevicePtr::NULL);
        }
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let elems = num_tokens * h;
        let mut last_out = DevicePtr::NULL;
        for repeat_idx in 0..policy.repeat {
            if repeat_idx > 0 && policy.renorm_between_repeats {
                ops::rms_norm(
                    ctx.gpu,
                    self.rms_norm_k,
                    hidden,
                    &self.post_attn_norm,
                    normed2,
                    num_tokens as u32,
                    h as u32,
                    eps,
                    stream,
                )?;
            }
            self.ffn.forward_prefill(normed2, num_tokens, ctx, stream)?;
            let moe_out = ctx.buffers.moe_output();
            self.add_moe_residual(hidden, moe_out, elems, policy.residual_scale, ctx, stream)?;
            last_out = moe_out;
        }
        self.check_hidden_if_requested(hidden, elems, ctx, stream, "prefill")?;
        Ok(last_out)
    }

    fn add_moe_residual(
        &self,
        hidden: DevicePtr,
        moe_out: DevicePtr,
        num_elements: usize,
        residual_scale: f32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if residual_scale == 1.0 {
            return ops::residual_add(
                ctx.gpu,
                self.residual_add_k,
                hidden,
                moe_out,
                num_elements as u32,
                stream,
            );
        }
        if self.scaled_add_k.0 == 0 {
            bail!(
                "MoE block policy requested residual_scale={residual_scale}, but bf16_scaled_add kernel is unavailable"
            );
        }
        ops::scaled_add(
            ctx.gpu,
            self.scaled_add_k,
            hidden,
            moe_out,
            residual_scale,
            num_elements as u32,
            stream,
        )
    }

    fn log_policy_once(&self, policy: MoeBlockLayerPolicy) {
        static LOGGED: std::sync::OnceLock<std::sync::Mutex<std::collections::BTreeSet<usize>>> =
            std::sync::OnceLock::new();
        let logged =
            LOGGED.get_or_init(|| std::sync::Mutex::new(std::collections::BTreeSet::new()));
        if let Ok(mut logged) = logged.lock()
            && logged.insert(self.attn_layer_idx)
        {
            tracing::warn!(
                "MoE block policy applied: layer={}, skip={}, repeat={}, residual_scale={}, renorm_between_repeats={}",
                self.attn_layer_idx,
                policy.skip,
                policy.repeat,
                policy.residual_scale,
                policy.renorm_between_repeats
            );
        }
    }

    fn check_hidden_if_requested(
        &self,
        hidden: DevicePtr,
        num_elements: usize,
        ctx: &ForwardContext,
        stream: u64,
        phase: &str,
    ) -> Result<()> {
        let Some(policy) = global_policy() else {
            return Ok(());
        };
        if (!policy.safety.log_hidden_norm && !policy.safety.fallback_on_nan) || ctx.graph_capture {
            return Ok(());
        }
        ctx.gpu.synchronize(stream)?;
        let mut buf = vec![0u16; num_elements.min(4096)];
        let bytes =
            unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, buf.len() * 2) };
        ctx.gpu.copy_d2h(hidden, bytes)?;
        let mut sum = 0.0f32;
        let mut max_abs = 0.0f32;
        for raw in buf {
            let v = f32::from_bits((raw as u32) << 16);
            if !v.is_finite() {
                bail!(
                    "MoE block policy detected non-finite hidden value at layer {} phase {}",
                    self.attn_layer_idx,
                    phase
                );
            }
            sum += v * v;
            max_abs = max_abs.max(v.abs());
        }
        if policy.safety.log_hidden_norm {
            tracing::debug!(
                "MoE block policy hidden norm sample: layer={}, phase={}, sample_elems={}, l2={:.4}, max_abs={:.4}",
                self.attn_layer_idx,
                phase,
                num_elements.min(4096),
                sum.sqrt(),
                max_abs
            );
        }
        Ok(())
    }
}
