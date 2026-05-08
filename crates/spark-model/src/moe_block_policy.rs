// SPDX-License-Identifier: AGPL-3.0-only

//! Experimental MoE FFN block skip/repeat policy.
//!
//! Default behavior is disabled. The policy is loaded only when
//! `ATLAS_MOE_BLOCK_POLICY_PATH` is set, and validation is forced during model
//! construction so invalid experiments fail before serving traffic.

use std::collections::BTreeSet;
use std::sync::OnceLock;

use anyhow::{Context, Result, anyhow, bail};

#[derive(Clone, Copy, Debug)]
pub struct MoeBlockLayerPolicy {
    pub skip: bool,
    pub repeat: usize,
    pub residual_scale: f32,
    pub renorm_between_repeats: bool,
}

impl Default for MoeBlockLayerPolicy {
    fn default() -> Self {
        Self {
            skip: false,
            repeat: 1,
            residual_scale: 1.0,
            renorm_between_repeats: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MoeBlockSafety {
    pub max_repeat: usize,
    pub fallback_on_nan: bool,
    pub log_hidden_norm: bool,
}

impl Default for MoeBlockSafety {
    fn default() -> Self {
        Self {
            max_repeat: 2,
            fallback_on_nan: true,
            log_hidden_norm: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MoeBlockPolicy {
    path: String,
    default: MoeBlockLayerPolicy,
    by_layer: Vec<MoeBlockLayerPolicy>,
    pub safety: MoeBlockSafety,
}

impl MoeBlockPolicy {
    pub fn for_layer(&self, layer_idx: usize) -> MoeBlockLayerPolicy {
        self.by_layer
            .get(layer_idx)
            .copied()
            .unwrap_or(self.default)
    }

    pub fn path(&self) -> &str {
        &self.path
    }

    pub fn skip_layers(&self) -> Vec<usize> {
        self.by_layer
            .iter()
            .enumerate()
            .filter_map(|(idx, p)| p.skip.then_some(idx))
            .collect()
    }

    pub fn repeat_layers(&self) -> Vec<(usize, usize, f32, bool)> {
        self.by_layer
            .iter()
            .enumerate()
            .filter_map(|(idx, p)| {
                (p.repeat > 1).then_some((
                    idx,
                    p.repeat,
                    p.residual_scale,
                    p.renorm_between_repeats,
                ))
            })
            .collect()
    }
}

#[derive(Clone, Debug)]
struct LayerGroup {
    name: String,
    layers: Vec<usize>,
    policy: MoeBlockLayerPolicy,
}

#[derive(Default)]
struct ParsedPolicy {
    default: MoeBlockLayerPolicy,
    safety: MoeBlockSafety,
    groups: Vec<LayerGroup>,
}

static POLICY: OnceLock<Option<MoeBlockPolicy>> = OnceLock::new();

pub fn init_from_env(num_layers: usize, fp32_residual: bool) -> Result<()> {
    let loaded = load_from_env(num_layers, fp32_residual)?;
    if POLICY.set(loaded).is_err() {
        tracing::debug!("MoE block policy was already initialized");
    }
    Ok(())
}

pub fn global_policy() -> Option<&'static MoeBlockPolicy> {
    POLICY.get().and_then(|p| p.as_ref())
}

fn load_from_env(num_layers: usize, fp32_residual: bool) -> Result<Option<MoeBlockPolicy>> {
    let Ok(path) = std::env::var("ATLAS_MOE_BLOCK_POLICY_PATH") else {
        return Ok(None);
    };
    if path.trim().is_empty() {
        return Ok(None);
    }
    let content = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read ATLAS_MOE_BLOCK_POLICY_PATH={path}"))?;
    let parsed = parse_policy(&content)?;
    let mut by_layer = vec![parsed.default; num_layers];
    for group in parsed.groups {
        for layer in group.layers {
            if layer >= num_layers {
                bail!(
                    "MoE block policy group '{}' references layer {} but model has {} layers",
                    group.name,
                    layer,
                    num_layers
                );
            }
            by_layer[layer] = group.policy;
        }
    }
    for (layer_idx, policy) in by_layer.iter_mut().enumerate() {
        validate_layer_policy(layer_idx, policy, &parsed.safety, fp32_residual)?;
    }
    let policy = MoeBlockPolicy {
        path,
        default: parsed.default,
        by_layer,
        safety: parsed.safety,
    };
    tracing::warn!(
        "MoE block policy active: path={}, skip_layers={:?}, repeat_layers={:?}, max_repeat={}, fallback_on_nan={}, log_hidden_norm={}",
        policy.path(),
        policy.skip_layers(),
        policy.repeat_layers(),
        policy.safety.max_repeat,
        policy.safety.fallback_on_nan,
        policy.safety.log_hidden_norm
    );
    Ok(Some(policy))
}

fn validate_layer_policy(
    layer_idx: usize,
    policy: &mut MoeBlockLayerPolicy,
    safety: &MoeBlockSafety,
    fp32_residual: bool,
) -> Result<()> {
    if policy.skip {
        policy.repeat = 0;
    }
    if policy.repeat > safety.max_repeat {
        bail!(
            "MoE block policy layer {layer_idx}: repeat={} exceeds safety.max_repeat={}",
            policy.repeat,
            safety.max_repeat
        );
    }
    if !(policy.residual_scale > 0.0 && policy.residual_scale <= 1.0) {
        bail!(
            "MoE block policy layer {layer_idx}: residual_scale={} must be > 0 and <= 1.0",
            policy.residual_scale
        );
    }
    if fp32_residual && policy.residual_scale != 1.0 && !policy.skip {
        bail!(
            "MoE block policy layer {layer_idx}: residual_scale != 1.0 is currently BF16-residual only"
        );
    }
    Ok(())
}

fn parse_policy(content: &str) -> Result<ParsedPolicy> {
    let mut parsed = ParsedPolicy::default();
    let mut section = "";
    let mut current_group: Option<LayerGroup> = None;

    for raw in content.lines() {
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        match line {
            "default:" => {
                flush_group(&mut parsed, &mut current_group);
                section = "default";
                continue;
            }
            "safety:" => {
                flush_group(&mut parsed, &mut current_group);
                section = "safety";
                continue;
            }
            "layer_groups:" => {
                flush_group(&mut parsed, &mut current_group);
                section = "layer_groups";
                continue;
            }
            _ => {}
        }
        if section == "layer_groups" && line.starts_with("- ") {
            flush_group(&mut parsed, &mut current_group);
            current_group = Some(LayerGroup {
                name: String::new(),
                layers: Vec::new(),
                policy: parsed.default,
            });
            let rest = line.trim_start_matches("- ").trim();
            if !rest.is_empty() {
                apply_group_field(current_group.as_mut().unwrap(), rest)?;
            }
            continue;
        }
        match section {
            "default" => apply_layer_field(&mut parsed.default, line)?,
            "safety" => apply_safety_field(&mut parsed.safety, line)?,
            "layer_groups" => {
                let group = current_group
                    .as_mut()
                    .ok_or_else(|| anyhow!("layer_groups entry must start with '-'"))?;
                apply_group_field(group, line)?;
            }
            _ => bail!("unexpected MoE block policy line before section: {line}"),
        }
    }
    flush_group(&mut parsed, &mut current_group);
    Ok(parsed)
}

fn flush_group(parsed: &mut ParsedPolicy, group: &mut Option<LayerGroup>) {
    if let Some(group) = group.take() {
        parsed.groups.push(group);
    }
}

fn split_key_value(line: &str) -> Result<(&str, &str)> {
    let (key, value) = line
        .split_once(':')
        .ok_or_else(|| anyhow!("invalid policy line, expected key: value: {line}"))?;
    Ok((key.trim(), value.trim()))
}

fn apply_layer_field(policy: &mut MoeBlockLayerPolicy, line: &str) -> Result<()> {
    let (key, value) = split_key_value(line)?;
    match key {
        "skip" => policy.skip = parse_bool(value)?,
        "repeat" => policy.repeat = parse_usize(value)?,
        "residual_scale" => policy.residual_scale = parse_f32(value)?,
        "renorm_between_repeats" => policy.renorm_between_repeats = parse_bool(value)?,
        "top_k" => {
            // Per-layer top-k is intentionally deferred. Accept null so policy
            // files can share the future shape without changing behavior.
            if value != "null" {
                bail!("per-layer top_k is not implemented yet; use global ATLAS_MOE_TOP_K_OVERRIDE")
            }
        }
        other => bail!("unknown MoE block layer field: {other}"),
    }
    Ok(())
}

fn apply_safety_field(safety: &mut MoeBlockSafety, line: &str) -> Result<()> {
    let (key, value) = split_key_value(line)?;
    match key {
        "max_repeat" => safety.max_repeat = parse_usize(value)?,
        "fallback_on_nan" => safety.fallback_on_nan = parse_bool(value)?,
        "log_hidden_norm" => safety.log_hidden_norm = parse_bool(value)?,
        other => bail!("unknown MoE block safety field: {other}"),
    }
    Ok(())
}

fn apply_group_field(group: &mut LayerGroup, line: &str) -> Result<()> {
    let (key, value) = split_key_value(line)?;
    match key {
        "name" => group.name = value.trim_matches('"').trim_matches('\'').to_string(),
        "layers" => group.layers = parse_layers(value)?,
        _ => apply_layer_field(&mut group.policy, line)?,
    }
    Ok(())
}

fn parse_bool(value: &str) -> Result<bool> {
    match value {
        "true" | "True" | "TRUE" => Ok(true),
        "false" | "False" | "FALSE" => Ok(false),
        _ => bail!("invalid bool value: {value}"),
    }
}

fn parse_usize(value: &str) -> Result<usize> {
    value
        .parse::<usize>()
        .with_context(|| format!("invalid usize value: {value}"))
}

fn parse_f32(value: &str) -> Result<f32> {
    value
        .parse::<f32>()
        .with_context(|| format!("invalid f32 value: {value}"))
}

fn parse_layers(value: &str) -> Result<Vec<usize>> {
    let v = value.trim();
    if !v.starts_with('[') || !v.ends_with(']') {
        bail!("layers must use inline list syntax, e.g. [24,25,26,27]");
    }
    let mut layers = BTreeSet::new();
    let inner = &v[1..v.len() - 1];
    if inner.trim().is_empty() {
        return Ok(Vec::new());
    }
    for part in inner.split(',') {
        layers.insert(parse_usize(part.trim())?);
    }
    Ok(layers.into_iter().collect())
}
