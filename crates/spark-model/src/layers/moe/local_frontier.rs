// SPDX-License-Identifier: AGPL-3.0-only

//! Local Frontier v1 routing helpers.
//!
//! These helpers keep candidate routing (`route_top_k`) separate from the
//! active compute fanout (`active_k`). They are deliberately host-mediated for
//! v1: slow but deterministic, easy to log, and adequate for residency bring-up.

use std::collections::BTreeSet;

use super::*;

impl MoeLayer {
    pub(super) fn local_frontier_active_k(&self, route_top_k: u32) -> u32 {
        let active = std::env::var("ATLAS_MOE_ACTIVE_K")
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
            .unwrap_or(route_top_k);
        active.clamp(1, route_top_k)
    }

    pub(super) fn maybe_compact_local_frontier_routes(
        &self,
        indices_dev: DevicePtr,
        weights_dev: DevicePtr,
        num_tokens: u32,
        route_top_k: u32,
        active_k: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        if active_k == route_top_k || num_tokens == 0 {
            return Ok(weights_dev);
        }
        if ctx.graph_capture {
            tracing::warn!(
                "Local Frontier active_k compaction disabled during graph capture; executing route_top_k={}",
                route_top_k
            );
            return Ok(weights_dev);
        }

        ctx.gpu.synchronize(stream)?;

        let route_count = num_tokens as usize * route_top_k as usize;
        let active_count = num_tokens as usize * active_k as usize;
        let mut idx_buf = vec![0u8; route_count * 4];
        let mut wt_buf = vec![0u8; route_count * 4];
        ctx.gpu.copy_d2h(indices_dev, &mut idx_buf)?;
        ctx.gpu.copy_d2h(weights_dev, &mut wt_buf)?;

        let resident = ctx.config.resident_expert_set.as_ref();
        let mut compact_idx = vec![0u8; active_count * 4];
        let mut compact_wt = vec![0u8; active_count * 4];
        let renorm = std::env::var("ATLAS_MOE_RENORM_ACTIVE_WEIGHTS")
            .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
            .unwrap_or(true);

        for token in 0..num_tokens as usize {
            let base = token * route_top_k as usize;
            let mut chosen = select_active_positions(
                &idx_buf,
                base,
                route_top_k as usize,
                active_k as usize,
                resident,
            );
            if chosen.len() < active_k as usize {
                chosen.truncate(active_k as usize);
            }

            let weight_sum = if renorm {
                chosen
                    .iter()
                    .map(|pos| read_f32(&wt_buf, base + *pos).max(0.0))
                    .sum::<f32>()
            } else {
                0.0
            };

            for (rank, pos) in chosen.iter().copied().enumerate() {
                let out = (token * active_k as usize + rank) * 4;
                let in_i = (base + pos) * 4;
                compact_idx[out..out + 4].copy_from_slice(&idx_buf[in_i..in_i + 4]);

                let mut weight = read_f32(&wt_buf, base + pos);
                if renorm && weight_sum > 0.0 {
                    weight /= weight_sum;
                }
                compact_wt[out..out + 4].copy_from_slice(&weight.to_le_bytes());
            }
        }

        let compact_weights_dev = indices_dev.offset(active_count * 4);
        ctx.gpu.copy_h2d(&compact_idx, indices_dev)?;
        ctx.gpu.copy_h2d(&compact_wt, compact_weights_dev)?;
        Ok(compact_weights_dev)
    }
}

fn select_active_positions(
    idx_buf: &[u8],
    base: usize,
    route_top_k: usize,
    active_k: usize,
    resident: Option<&BTreeSet<usize>>,
) -> Vec<usize> {
    let mut chosen = Vec::with_capacity(active_k);
    let mut seen = BTreeSet::new();
    if let Some(resident) = resident {
        for pos in 0..route_top_k {
            let expert = read_u32(idx_buf, base + pos) as usize;
            if resident.contains(&expert) && seen.insert(pos) {
                chosen.push(pos);
                if chosen.len() == active_k {
                    return chosen;
                }
            }
        }
    }
    for pos in 0..route_top_k {
        if seen.insert(pos) {
            chosen.push(pos);
            if chosen.len() == active_k {
                break;
            }
        }
    }
    chosen
}

fn read_u32(buf: &[u8], index: usize) -> u32 {
    let j = index * 4;
    u32::from_le_bytes([buf[j], buf[j + 1], buf[j + 2], buf[j + 3]])
}

fn read_f32(buf: &[u8], index: usize) -> f32 {
    let j = index * 4;
    f32::from_le_bytes([buf[j], buf[j + 1], buf[j + 2], buf[j + 3]])
}
