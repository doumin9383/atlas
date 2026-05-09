// SPDX-License-Identifier: AGPL-3.0-only

//! CPU-side policy helpers for opt-in weighted MoE rerank.

#![allow(dead_code)]

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RerankWeightMode {
    OriginalRouterProbs,
    AdjustedSoftmax,
}

#[derive(Debug, Clone)]
pub struct WeightedRerankConfig {
    pub enabled: bool,
    pub tier_map_path: Option<String>,
    pub lambda_cost: f32,
    pub beta_hot: f32,
    pub candidate_top_n: usize,
    pub weight_mode: RerankWeightMode,
}

impl WeightedRerankConfig {
    pub fn from_env() -> anyhow::Result<Self> {
        let enabled = std::env::var("ATLAS_MOE_WEIGHTED_RERANK")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let weight_mode = match std::env::var("ATLAS_MOE_RERANK_WEIGHT_MODE")
            .unwrap_or_else(|_| "original_router_probs".to_string())
            .as_str()
        {
            "original_router_probs" => RerankWeightMode::OriginalRouterProbs,
            "adjusted_softmax" => RerankWeightMode::AdjustedSoftmax,
            other => anyhow::bail!(
                "Invalid ATLAS_MOE_RERANK_WEIGHT_MODE='{other}', expected original_router_probs|adjusted_softmax"
            ),
        };
        let cfg = Self {
            enabled,
            tier_map_path: std::env::var("ATLAS_MOE_EXPERT_TIER_MAP_PATH").ok(),
            lambda_cost: std::env::var("ATLAS_MOE_RERANK_LAMBDA_COST")
                .ok()
                .and_then(|v| v.parse::<f32>().ok())
                .unwrap_or(0.0),
            beta_hot: std::env::var("ATLAS_MOE_RERANK_BETA_HOT")
                .ok()
                .and_then(|v| v.parse::<f32>().ok())
                .unwrap_or(0.0),
            candidate_top_n: std::env::var("ATLAS_MOE_RERANK_CANDIDATE_TOP_N")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(0),
            weight_mode,
        };
        if cfg.enabled {
            tracing::info!(
                "ATLAS_MOE_WEIGHTED_RERANK enabled: lambda_cost={}, beta_hot={}, candidate_top_n={}, weight_mode={:?}, tier_map={:?}",
                cfg.lambda_cost,
                cfg.beta_hot,
                cfg.candidate_top_n,
                cfg.weight_mode,
                cfg.tier_map_path,
            );
        }
        Ok(cfg)
    }
}

pub fn adjusted_score(
    router_score: f32,
    tier_cost: f32,
    is_hot: bool,
    lambda_cost: f32,
    beta_hot: f32,
) -> f32 {
    router_score + beta_hot * if is_hot { 1.0 } else { 0.0 } - lambda_cost * tier_cost
}

pub fn weighted_topk(
    expert_ids: &[u32],
    router_scores: &[f32],
    tier_costs: &[f32],
    hot: &[bool],
    k: usize,
    lambda_cost: f32,
    beta_hot: f32,
) -> Vec<u32> {
    let mut rows: Vec<(f32, usize, u32)> = expert_ids
        .iter()
        .copied()
        .enumerate()
        .map(|(idx, expert_id)| {
            (
                adjusted_score(
                    router_scores[idx],
                    tier_costs.get(idx).copied().unwrap_or(0.0),
                    hot.get(idx).copied().unwrap_or(false),
                    lambda_cost,
                    beta_hot,
                ),
                idx,
                expert_id,
            )
        })
        .collect();
    rows.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });
    rows.into_iter()
        .take(k)
        .map(|(_, _, expert_id)| expert_id)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weighted_rerank_identity_mode() {
        let ids = [10, 11, 12, 13];
        let scores = [0.9, 0.8, 0.7, 0.6];
        let costs = [30.0, 1.0, 1.0, 1.0];
        let hot = [false, true, true, true];
        assert_eq!(
            weighted_topk(&ids, &scores, &costs, &hot, 2, 0.0, 0.0),
            vec![10, 11]
        );
    }

    #[test]
    fn weighted_rerank_cost_bias() {
        let ids = [10, 11, 12];
        let scores = [0.9, 0.89, 0.88];
        let costs = [30.0, 1.0, 1.0];
        let hot = [false, false, false];
        assert_eq!(
            weighted_topk(&ids, &scores, &costs, &hot, 1, 0.1, 0.0),
            vec![11]
        );
    }
}
