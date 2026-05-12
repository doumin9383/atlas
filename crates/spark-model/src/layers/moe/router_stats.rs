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
    candidate_enabled: bool,
    candidate_path: String,
    candidate_top_n: usize,
    candidate_mode: String,
    dispatch_summary_enabled: bool,
    dispatch_summary_path: String,
    dispatch_summary_sample_tokens: usize,
    dispatch_summary_every_n_layers: Option<usize>,
}

static CONFIG: OnceLock<RouterStatsConfig> = OnceLock::new();
static FILE: OnceLock<Option<Mutex<File>>> = OnceLock::new();
static CANDIDATE_FILE: OnceLock<Option<Mutex<File>>> = OnceLock::new();
static DISPATCH_SUMMARY_FILE: OnceLock<Option<Mutex<File>>> = OnceLock::new();

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
        let candidate_top_n = std::env::var("ATLAS_MOE_CANDIDATE_TOP_N")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(32);
        let candidate_path = std::env::var("ATLAS_MOE_CANDIDATE_LOG_PATH")
            .unwrap_or_else(|_| "/tmp/atlas-moe-candidates.jsonl".to_string());
        let candidate_mode =
            std::env::var("ATLAS_MOE_CANDIDATE_LOG_MODE").unwrap_or_else(|_| "summary".to_string());
        let dispatch_summary_path = std::env::var("ATLAS_MOE_DISPATCH_SUMMARY_LOG_PATH")
            .unwrap_or_else(|_| "/tmp/atlas-moe-dispatch-summary.jsonl".to_string());
        let dispatch_summary_sample_tokens = match std::env::var(
            "ATLAS_MOE_DISPATCH_SUMMARY_SAMPLE_TOKENS",
        )
        .unwrap_or_else(|_| "first_16".to_string())
        .as_str()
        {
            "first_64" => 64,
            "first_16" => 16,
            other => {
                tracing::warn!(
                    "ATLAS_MOE_DISPATCH_SUMMARY_SAMPLE_TOKENS={other:?} is invalid; using first_16"
                );
                16
            }
        };
        let dispatch_summary_every_n_layers =
            match std::env::var("ATLAS_MOE_DISPATCH_SUMMARY_SAMPLE_LAYERS")
                .unwrap_or_else(|_| "all".to_string())
                .as_str()
            {
                "all" => None,
                "every_4" => Some(4),
                other => {
                    tracing::warn!(
                        "ATLAS_MOE_DISPATCH_SUMMARY_SAMPLE_LAYERS={other:?} is invalid; using all"
                    );
                    None
                }
            };
        RouterStatsConfig {
            enabled,
            path,
            max_tokens_per_call,
            candidate_enabled: candidate_top_n > 0
                && std::env::var("ATLAS_MOE_CANDIDATE_LOG_PATH").is_ok(),
            candidate_path,
            candidate_top_n,
            candidate_mode,
            dispatch_summary_enabled: std::env::var("ATLAS_MOE_DISPATCH_SUMMARY_LOG_PATH").is_ok(),
            dispatch_summary_path,
            dispatch_summary_sample_tokens,
            dispatch_summary_every_n_layers,
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

fn candidate_file(path: &str) -> Option<&'static Mutex<File>> {
    CANDIDATE_FILE
        .get_or_init(
            || match OpenOptions::new().create(true).append(true).open(path) {
                Ok(file) => Some(Mutex::new(file)),
                Err(err) => {
                    tracing::warn!("ATLAS_MOE_CANDIDATE_LOG: failed to open {path}: {err}");
                    None
                }
            },
        )
        .as_ref()
}

fn dispatch_summary_file(path: &str) -> Option<&'static Mutex<File>> {
    DISPATCH_SUMMARY_FILE
        .get_or_init(
            || match OpenOptions::new().create(true).append(true).open(path) {
                Ok(file) => Some(Mutex::new(file)),
                Err(err) => {
                    tracing::warn!("ATLAS_MOE_DISPATCH_SUMMARY: failed to open {path}: {err}");
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

fn normalize_candidate_probs(scores: &[f32]) -> Vec<f32> {
    let positive_sum: f32 = scores.iter().copied().filter(|v| *v > 0.0).sum();
    if positive_sum > 0.0 && scores.iter().all(|v| *v >= 0.0) {
        return scores.iter().map(|v| (v / positive_sum).max(0.0)).collect();
    }
    let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scores.iter().map(|v| (*v - max_score).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum <= 0.0 || !sum.is_finite() {
        return vec![0.0; scores.len()];
    }
    exps.into_iter().map(|v| v / sum).collect()
}

fn env_opt(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.trim().is_empty())
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
        if (!cfg.enabled && !cfg.candidate_enabled && !cfg.dispatch_summary_enabled)
            || ctx.graph_capture
            || top_k == 0
            || num_tokens == 0
        {
            return;
        }

        if let Some(every) = cfg.dispatch_summary_every_n_layers
            && self.layer_idx % every != 0
            && !cfg.enabled
            && !cfg.candidate_enabled
        {
            return;
        }

        if let Err(err) = ctx.gpu.synchronize(stream) {
            tracing::warn!("ATLAS_MOE_ROUTER_STATS: stream sync failed: {err:#}");
            return;
        }

        let token_count = (num_tokens as usize).min(
            cfg.max_tokens_per_call
                .max(cfg.dispatch_summary_sample_tokens),
        );
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
        let mut stats_out = if cfg.enabled {
            stats_file(&cfg.path).map(|file| file.lock())
        } else {
            None
        };
        let mut candidate_out = if cfg.candidate_enabled {
            candidate_file(&cfg.candidate_path).map(|file| file.lock())
        } else {
            None
        };
        let mut dispatch_summary_out = if cfg.dispatch_summary_enabled
            && cfg
                .dispatch_summary_every_n_layers
                .map(|every| self.layer_idx % every == 0)
                .unwrap_or(true)
        {
            dispatch_summary_file(&cfg.dispatch_summary_path).map(|file| file.lock())
        } else {
            None
        };

        let summary_token_count = token_count.min(cfg.dispatch_summary_sample_tokens);
        let mut summary_unique_experts = std::collections::BTreeSet::new();
        let mut summary_entropy_sum = 0.0f32;
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
            if let Some(out) = stats_out.as_mut() {
                if let Err(err) = writeln!(out, "{row}") {
                    tracing::warn!("ATLAS_MOE_ROUTER_STATS: write failed: {err}");
                    return;
                }
            }
            if let Some(out) = candidate_out.as_mut() {
                let active_k = std::env::var("ATLAS_MOE_ACTIVE_K")
                    .ok()
                    .and_then(|v| v.trim().parse::<usize>().ok())
                    .unwrap_or(top_k_usize)
                    .clamp(1, top_k_usize);
                let candidate_n = cfg.candidate_top_n.max(active_k);
                let observed_top_n = candidate_n.min(top_k_usize);
                let candidate_scores = scores
                    .iter()
                    .take(observed_top_n)
                    .copied()
                    .collect::<Vec<_>>();
                let candidate_probs = normalize_candidate_probs(&candidate_scores);
                let selected_expert_ids = ids.iter().take(active_k).copied().collect::<Vec<_>>();
                let selected_weights = scores.iter().take(active_k).copied().collect::<Vec<_>>();
                let candidate_row = json!({
                    "schema_version": "atlas.local_frontier.candidate_topn.v1",
                    "timestamp": timestamp_ms,
                    "request_id": null,
                    "model_id": ctx.config.model_type,
                    "task_type": env_opt("ATLAS_TASK_TYPE").unwrap_or_else(|| "unknown".to_string()),
                    "policy_id": env_opt("ATLAS_LOCAL_FRONTIER_POLICY_ID"),
                    "layer_id": self.layer_idx,
                    "layer": self.layer_idx,
                    "token_index": token_index,
                    "candidate_N": candidate_n,
                    "candidate_n": candidate_n,
                    "active_k": active_k,
                    "top_n": candidate_n,
                    "observed_top_n": observed_top_n,
                    "candidate_log_mode": cfg.candidate_mode,
                    "expert_ids": ids.iter().take(observed_top_n).copied().collect::<Vec<_>>(),
                    "router_scores": candidate_scores,
                    "candidate_probs": candidate_probs,
                    "selected_expert_ids": selected_expert_ids,
                    "selected_weights": selected_weights,
                    "compute_policy_id": env_opt("ATLAS_COMPUTE_POLICY_ID")
                        .or_else(|| env_opt("ATLAS_LOCAL_FRONTIER_COMPUTE_POLICY_ID")),
                    "memory_mode": env_opt("ATLAS_MEMORY_MODE")
                        .or_else(|| env_opt("ATLAS_LOCAL_FRONTIER_MEMORY_MODE")),
                    "tier_map_version": env_opt("ATLAS_MOE_TIER_MAP_VERSION"),
                    "resident_set_version": env_opt("ATLAS_MOE_RESIDENT_SET_VERSION"),
                    "weighted_rerank_enabled": std::env::var("ATLAS_MOE_WEIGHTED_RERANK")
                        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                        .unwrap_or(false),
                    "weight_mode": std::env::var("ATLAS_MOE_RERANK_WEIGHT_MODE")
                        .unwrap_or_else(|_| "original_router_probs".to_string()),
                    "entropy": entropy(&scores.iter().take(observed_top_n).copied().collect::<Vec<_>>()),
                    "top1_top2_margin": max_prob - second,
                    "tail_mass_after_k": {
                        "3": if top_k_usize > 3 { scores.iter().skip(3).sum::<f32>() } else { 0.0 },
                        "6": if top_k_usize > 6 { scores.iter().skip(6).sum::<f32>() } else { 0.0 },
                        "10": if top_k_usize > 10 { scores.iter().skip(10).sum::<f32>() } else { 0.0 },
                    },
                });
                if let Err(err) = writeln!(out, "{candidate_row}") {
                    tracing::warn!("ATLAS_MOE_CANDIDATE_LOG: write failed: {err}");
                    return;
                }
            }
            if token_index < summary_token_count {
                summary_unique_experts.extend(ids.iter().copied());
                summary_entropy_sum += entropy(&scores);
            }
        }

        if let Some(out) = dispatch_summary_out.as_mut() {
            let entropy_avg = if summary_token_count > 0 {
                summary_entropy_sum / summary_token_count as f32
            } else {
                0.0
            };
            let row = json!({
                "timestamp": timestamp_ms,
                "request_id": null,
                "model": ctx.config.model_type,
                "layer": self.layer_idx,
                "sampled_tokens": summary_token_count,
                "effective_top_k": top_k,
                "selected_expert_count_min": top_k,
                "selected_expert_count_max": top_k,
                "selected_expert_count_avg": top_k as f32,
                "unique_experts_selected": summary_unique_experts.len(),
                "router_entropy_avg": entropy_avg,
            });
            if let Err(err) = writeln!(out, "{row}") {
                tracing::warn!("ATLAS_MOE_DISPATCH_SUMMARY: write failed: {err}");
            }
        }
    }
}
