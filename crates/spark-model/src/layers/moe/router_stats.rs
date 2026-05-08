// SPDX-License-Identifier: AGPL-3.0-only

//! Opt-in MoE router summary logging.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

use parking_lot::Mutex;
use serde_json::json;
use spark_runtime::gpu::DevicePtr;

use super::*;

struct RouterStatsConfig {
    enabled: bool,
    path: String,
    max_tokens_per_call: usize,
}

static CONFIG: OnceLock<RouterStatsConfig> = OnceLock::new();
static FILE: OnceLock<Option<Mutex<File>>> = OnceLock::new();

fn config() -> &'static RouterStatsConfig {
    CONFIG.get_or_init(|| {
        let enabled = std::env::var("ATLAS_MOE_ROUTER_STATS")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let path = std::env::var("ATLAS_MOE_ROUTER_STATS_PATH")
            .unwrap_or_else(|_| "/tmp/atlas-router-stats.jsonl".to_string());
        let max_tokens_per_call = std::env::var("ATLAS_MOE_ROUTER_STATS_MAX_TOKENS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(4)
            .max(1);
        RouterStatsConfig {
            enabled,
            path,
            max_tokens_per_call,
        }
    })
}

fn stats_file(path: &str) -> Option<&'static Mutex<File>> {
    FILE.get_or_init(
        || match OpenOptions::new().create(true).append(true).open(path) {
            Ok(file) => Some(Mutex::new(file)),
            Err(err) => {
                tracing::warn!("ATLAS_MOE_ROUTER_STATS: failed to open {path}: {err}");
                None
            }
        },
    )
    .as_ref()
}

fn entropy(weights: &[f32]) -> f32 {
    let sum: f32 = weights.iter().copied().filter(|v| *v > 0.0).sum();
    if sum <= 0.0 {
        return 0.0;
    }
    weights
        .iter()
        .copied()
        .filter(|v| *v > 0.0)
        .map(|v| {
            let p = v / sum;
            -p * p.ln()
        })
        .sum()
}

impl MoeLayer {
    pub(super) fn maybe_log_router_stats(
        &self,
        indices_dev: DevicePtr,
        weights_dev: DevicePtr,
        num_tokens: u32,
        top_k: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) {
        let cfg = config();
        if !cfg.enabled || ctx.graph_capture || top_k == 0 || num_tokens == 0 {
            return;
        }
        let Some(file) = stats_file(&cfg.path) else {
            return;
        };

        if let Err(err) = ctx.gpu.synchronize(stream) {
            tracing::warn!("ATLAS_MOE_ROUTER_STATS: stream sync failed: {err:#}");
            return;
        }

        let token_count = (num_tokens as usize).min(cfg.max_tokens_per_call);
        let top_k_usize = top_k as usize;
        let mut idx_buf = vec![0u8; token_count * top_k_usize * 4];
        let mut wt_buf = vec![0u8; token_count * top_k_usize * 4];
        if let Err(err) = ctx.gpu.copy_d2h(indices_dev, &mut idx_buf) {
            tracing::warn!("ATLAS_MOE_ROUTER_STATS: expert id copy failed: {err:#}");
            return;
        }
        if let Err(err) = ctx.gpu.copy_d2h(weights_dev, &mut wt_buf) {
            tracing::warn!("ATLAS_MOE_ROUTER_STATS: gate score copy failed: {err:#}");
            return;
        }

        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        let mut out = file.lock();
        for token_index in 0..token_count {
            let offset = token_index * top_k_usize;
            let ids: Vec<u32> = (0..top_k_usize)
                .map(|i| {
                    let j = (offset + i) * 4;
                    u32::from_le_bytes([idx_buf[j], idx_buf[j + 1], idx_buf[j + 2], idx_buf[j + 3]])
                })
                .collect();
            let scores: Vec<f32> = (0..top_k_usize)
                .map(|i| {
                    let j = (offset + i) * 4;
                    f32::from_le_bytes([wt_buf[j], wt_buf[j + 1], wt_buf[j + 2], wt_buf[j + 3]])
                })
                .collect();
            let max_prob = scores.iter().copied().fold(0.0f32, f32::max);
            let second = scores
                .iter()
                .copied()
                .filter(|v| *v < max_prob)
                .fold(0.0f32, f32::max);
            let row = json!({
                "timestamp": timestamp_ms,
                "request_id": null,
                "layer": self.layer_idx,
                "token_index": token_index,
                "top_k": top_k,
                "selected_expert_ids": ids,
                "selected_gate_scores": scores,
                "entropy": entropy(&scores),
                "max_prob": max_prob,
                "margin_top1_top2": max_prob - second,
            });
            if let Err(err) = writeln!(out, "{row}") {
                tracing::warn!("ATLAS_MOE_ROUTER_STATS: write failed: {err}");
                return;
            }
        }
    }
}
