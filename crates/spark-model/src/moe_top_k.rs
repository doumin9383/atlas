// SPDX-License-Identifier: AGPL-3.0-only

//! Per-request MoE expert top-k override via side-channel.
//!
//! Instead of threading `moe_top_k` through the entire pipeline
//! (API → scheduler → ForwardContext → MoE forward), which creates
//! merge conflicts on every upstream sync, this module uses a global
//! atomic that the API handler sets once per request and the MoE
//! forward functions read directly.
//!
//! Usage:
//!   // API handler entry point:
//!   moe_top_k::set(req.moe_top_k.unwrap_or(0));
//!
//!   // MoE forward function, instead of `config.num_experts_per_tok`:
//!   let top_k = moe_top_k::resolve_or(config.num_experts_per_tok as u32);

use std::sync::atomic::{AtomicU32, Ordering};

/// Global atomic for per-request MoE top-k override.
///
/// 0 = no override (use model config default).
/// Non-zero = override value, capped at `num_experts_per_tok` by the caller.
static MOE_TOP_K: AtomicU32 = AtomicU32::new(0);

/// Set the per-request MoE expert top-k override.
///
/// Called by the API handler at request entry. Pass `0` (or `None` mapped
/// to `0`) to use the model config default.
pub fn set(k: u32) {
    MOE_TOP_K.store(k, Ordering::Relaxed);
}

/// Resolve the effective top-k value for this request.
///
/// If an override was set (non-zero), returns it. Otherwise returns
/// `config_default` (typically `config.num_experts_per_tok`).
pub fn resolve_or(config_default: u32) -> u32 {
    let v = MOE_TOP_K.load(Ordering::Relaxed);
    if v == 0 { config_default } else { v }
}
