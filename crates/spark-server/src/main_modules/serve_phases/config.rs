// SPDX-License-Identifier: AGPL-3.0-only

//! Config / model-dir / vocab-cap helpers.

use std::path::Path;

use anyhow::{Context, Result};

use atlas_core::config::ModelConfig;

use crate::cli;

pub(crate) fn merge_sidecar_quant_config(model_dir: &Path, config: &mut ModelConfig) {
    if config.quantization_config.is_some() {
        return;
    }
    let hf_quant_path = model_dir.join("hf_quant_config.json");
    if !hf_quant_path.exists() {
        return;
    }
    match std::fs::read_to_string(&hf_quant_path) {
        Ok(raw_hq) => {
            let wrapped = format!(r#"{{"quantization_config":{raw_hq}}}"#);
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&wrapped) {
                config.quantization_config = atlas_core::config::parse_quantization_config(&v);
            }
        }
        Err(e) => tracing::warn!("Failed to read sibling hf_quant_config.json: {e}"),
    }
}

pub(crate) fn load_model_config(model_dir: &Path) -> Result<(ModelConfig, String)> {
    let config_path = model_dir.join("config.json");
    let params_path = model_dir.join("params.json");
    let config_json = if config_path.exists() {
        std::fs::read_to_string(&config_path)
            .with_context(|| format!("Failed to read {}", config_path.display()))?
    } else if params_path.exists() {
        std::fs::read_to_string(&params_path)
            .with_context(|| format!("Failed to read {}", params_path.display()))?
    } else {
        anyhow::bail!(
            "No config.json or params.json found in {}",
            model_dir.display()
        );
    };
    let config = if params_path.exists() && !config_path.exists() {
        atlas_core::config::parse_mistral_params(&config_json)
            .context("Failed to parse params.json (Mistral format)")?
    } else {
        atlas_core::config::parse_config(&config_json).context("Failed to parse config.json")?
    };
    Ok((config, config_json))
}

pub(crate) fn apply_moe_top_k_override(
    args: &cli::ServeArgs,
    config: &mut ModelConfig,
) -> Result<()> {
    let model_default_top_k = config.num_experts_per_tok;
    let env_override = match std::env::var("ATLAS_MOE_TOP_K_OVERRIDE") {
        Ok(raw) if !raw.trim().is_empty() => Some(
            raw.trim()
                .parse::<usize>()
                .with_context(|| format!("Invalid ATLAS_MOE_TOP_K_OVERRIDE value '{raw}'"))?,
        ),
        _ => None,
    };
    let override_top_k = args.moe_top_k_override.or(env_override);

    if args.moe_top_k_policy != "model-config" && args.moe_top_k_policy != "fixed" {
        anyhow::bail!(
            "Unknown --moe-top-k-policy '{}'. Supported: model-config, fixed",
            args.moe_top_k_policy,
        );
    }

    let Some(override_top_k) = override_top_k else {
        if args.moe_top_k_policy != "model-config" {
            anyhow::bail!(
                "--moe-top-k-policy={} requires --moe-top-k-override; use the default policy to preserve model config",
                args.moe_top_k_policy,
            );
        }
        log_moe_top_k_startup(model_default_top_k, None, config);
        return Ok(());
    };

    let env_forced = args.moe_top_k_override.is_none();
    if args.moe_top_k_policy != "fixed" && !env_forced {
        anyhow::bail!(
            "--moe-top-k-override requires --moe-top-k-policy fixed (got '{}')",
            args.moe_top_k_policy,
        );
    }
    if config.num_experts == 0 {
        anyhow::bail!("--moe-top-k-override was set, but model config has num_experts=0");
    }
    if override_top_k == 0 || override_top_k > config.num_experts {
        anyhow::bail!(
            "--moe-top-k-override must be in 1..={} for this model (got {})",
            config.num_experts,
            override_top_k,
        );
    }

    config.num_experts_per_tok = override_top_k;
    tracing::info!(
        "MoE top-k policy: fixed override active (source={}, default_top_k={}, override_top_k={}, num_experts={}, norm_topk_prob={})",
        if env_forced { "env" } else { "cli" },
        model_default_top_k,
        override_top_k,
        config.num_experts,
        config.norm_topk_prob,
    );
    log_moe_top_k_startup(model_default_top_k, Some(override_top_k), config);
    Ok(())
}

pub(crate) fn apply_local_frontier_resident_set(config: &mut ModelConfig) -> Result<()> {
    let raw = match std::env::var("ATLAS_MOE_RESIDENT_EXPERTS") {
        Ok(raw) if !raw.trim().is_empty() => raw,
        _ => {
            tracing::info!(
                "Local Frontier resident set: disabled (all {} experts resident)",
                config.num_experts
            );
            return Ok(());
        }
    };
    let resident = spark_runtime::weights::parse_expert_id_set(raw.trim())
        .with_context(|| "Invalid ATLAS_MOE_RESIDENT_EXPERTS")?;
    if resident.is_empty() {
        anyhow::bail!("ATLAS_MOE_RESIDENT_EXPERTS resolved to an empty resident set");
    }
    if let Some(max_id) = resident.iter().next_back()
        && *max_id >= config.num_experts
    {
        anyhow::bail!(
            "ATLAS_MOE_RESIDENT_EXPERTS contains expert {}, but model has experts 0..{}",
            max_id,
            config.num_experts.saturating_sub(1)
        );
    }

    let preview = resident
        .iter()
        .take(16)
        .map(|id| id.to_string())
        .collect::<Vec<_>>()
        .join(",");
    config.resident_expert_set = Some(resident);
    tracing::info!(
        "Local Frontier resident set: resident_experts={}, total_experts={}, resident_set_version={:?}, first_experts=[{}]",
        config
            .resident_expert_set
            .as_ref()
            .map(|s| s.len())
            .unwrap_or(0),
        config.num_experts,
        std::env::var("ATLAS_MOE_RESIDENT_SET_VERSION").ok(),
        preview,
    );
    Ok(())
}

fn log_moe_top_k_startup(
    model_default_top_k: usize,
    override_top_k: Option<usize>,
    config: &ModelConfig,
) {
    let candidate_top_n = std::env::var("ATLAS_MOE_CANDIDATE_TOP_N").ok();
    let weighted_rerank_enabled = std::env::var("ATLAS_MOE_WEIGHTED_RERANK")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    tracing::info!(
        "MoE startup top-k: model_default_top_k={}, ATLAS_MOE_TOP_K_OVERRIDE={:?}, effective_top_k={}, max_supported_top_k={}, scratch_buffer_top_k_capacity={}, candidate_top_n={:?}, weighted_rerank_enabled={}",
        model_default_top_k,
        std::env::var("ATLAS_MOE_TOP_K_OVERRIDE").ok(),
        config.num_experts_per_tok,
        config.num_experts,
        config.num_experts_per_tok,
        candidate_top_n,
        weighted_rerank_enabled,
    );
    if let Some(k) = override_top_k {
        tracing::info!("MoE startup effective_top_k={k}");
    }
}

pub(crate) fn resolve_model_dir(args: &cli::ServeArgs) -> Result<std::path::PathBuf> {
    use crate::model_resolver;
    if let Some(ref path) = args.model_from_path {
        model_resolver::resolve_model_dir(
            path.to_str().context("Invalid model path")?,
            args.cache_dir.as_deref(),
        )
    } else {
        let model_spec = args
            .model
            .as_deref()
            .context("Either MODEL or --model-from-path is required")?;
        model_resolver::resolve_model_dir(model_spec, args.cache_dir.as_deref())
    }
}

pub(crate) fn cap_vocab_size_to_tokenizer(model_dir: &Path, config: &mut ModelConfig) {
    let tok_path = model_dir.join("tokenizer.json");
    if tok_path.exists()
        && let Ok(tok) = tokenizers::Tokenizer::from_file(&tok_path)
    {
        let tok_vocab = tok.get_vocab_size(true);
        if tok_vocab > 0 && tok_vocab < config.vocab_size {
            tracing::info!(
                "Capping vocab_size from {} to {} (tokenizer)",
                config.vocab_size,
                tok_vocab,
            );
            config.vocab_size = tok_vocab;
        }
    }
}

pub(crate) fn apply_model_default_num_drafts(
    args: &mut cli::ServeArgs,
    ptx_set: &atlas_kernels::TargetPtxSet,
) {
    if ptx_set.behavior.default_num_drafts > 0 && args.num_drafts == 1 {
        let model_default = ptx_set.behavior.default_num_drafts as usize;
        if model_default != args.num_drafts {
            tracing::info!(
                "num_drafts: using MODEL.toml default_num_drafts={} (K={}) — pass --num-drafts to override",
                model_default,
                model_default + 1,
            );
            args.num_drafts = model_default;
        }
    }
}
