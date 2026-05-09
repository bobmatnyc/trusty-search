//! Reciprocal Rank Fusion (RRF) — parameter-free fusion of ranked result lists.
//!
//! Why: RRF is the standard, parameter-free way to combine heterogeneous ranked
//! lists (vector similarity + BM25). The constant `k=60` is the canonical Cormack
//! et al. value and works well across domains, so we never have to tune per-query.
//! What: Walk each lane, assign each doc a rank starting at 1, sum
//! `weight * 1/(k + rank)` across lanes. Docs absent from a lane contribute 0
//! from that lane (equivalent to rank = ∞).
//! Test: see the `tests` module — both-lane fusion, single-lane fallback, and
//! rank-not-score ordering.
use std::collections::HashMap;

/// Standard RRF constant from Cormack et al. (2009). Parameter-free in practice.
pub const RRF_K: f32 = 60.0;

/// Fuse HNSW (vector) and BM25 (lexical) result lists into a single ranked list.
///
/// - `hnsw_results`: `(chunk_id, score_or_distance)` — order is what matters; the
///   slice MUST already be sorted best-first (rank 1 = element 0). The score field
///   is ignored by RRF (rank-only fusion).
/// - `bm25_results`: same convention, sorted highest-BM25-first.
/// - `alpha`: vector lane weight (intent-derived).
/// - `beta`: BM25 lane weight (intent-derived).
/// - `k`: RRF damping constant; pass [`RRF_K`] (60.0) unless you have a reason.
/// - `top_k`: cap on returned results.
///
/// Returns `(chunk_id, fused_score)` sorted by fused score descending.
pub fn rrf_fuse(
    hnsw_results: &[(String, f32)],
    bm25_results: &[(String, f32)],
    alpha: f32,
    beta: f32,
    k: f32,
    top_k: usize,
) -> Vec<(String, f32)> {
    let mut accum: HashMap<String, f32> = HashMap::new();

    for (rank0, (id, _)) in hnsw_results.iter().enumerate() {
        let rank = (rank0 + 1) as f32;
        *accum.entry(id.clone()).or_insert(0.0) += alpha * (1.0 / (k + rank));
    }
    for (rank0, (id, _)) in bm25_results.iter().enumerate() {
        let rank = (rank0 + 1) as f32;
        *accum.entry(id.clone()).or_insert(0.0) += beta * (1.0 / (k + rank));
    }

    let mut fused: Vec<(String, f32)> = accum.into_iter().collect();
    // Sort by score desc; tie-break on id for determinism.
    fused.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    fused.truncate(top_k);
    fused
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(s: &str) -> String {
        s.to_string()
    }

    #[test]
    fn test_rrf_fusion_both_lanes() {
        // Doc "a" tops both lanes → should dominate.
        let hnsw = vec![(id("a"), 0.95), (id("b"), 0.80), (id("c"), 0.70)];
        let bm25 = vec![(id("a"), 12.0), (id("c"), 8.0), (id("d"), 4.0)];
        let fused = rrf_fuse(&hnsw, &bm25, 0.5, 0.5, RRF_K, 10);
        assert_eq!(fused[0].0, "a", "doc in both lanes at top must rank first");
        // "c" appears in both at rank 3 / 2, beats single-lane "b" / "d".
        let positions: Vec<&String> = fused.iter().map(|(i, _)| i).collect();
        let pos_c = positions.iter().position(|s| *s == "c").unwrap();
        let pos_b = positions.iter().position(|s| *s == "b").unwrap();
        let pos_d = positions.iter().position(|s| *s == "d").unwrap();
        assert!(pos_c < pos_b, "c (both lanes) should beat b (one lane)");
        assert!(pos_c < pos_d, "c (both lanes) should beat d (one lane)");
    }

    #[test]
    fn test_rrf_fusion_single_lane_bm25_only() {
        let hnsw: Vec<(String, f32)> = Vec::new();
        let bm25 = vec![(id("x"), 5.0), (id("y"), 3.0)];
        let fused = rrf_fuse(&hnsw, &bm25, 0.5, 0.5, RRF_K, 10);
        assert_eq!(fused.len(), 2);
        assert_eq!(fused[0].0, "x");
        assert_eq!(fused[1].0, "y");
    }

    #[test]
    fn test_rrf_fusion_single_lane_hnsw_only() {
        let hnsw = vec![(id("p"), 0.9), (id("q"), 0.5)];
        let bm25: Vec<(String, f32)> = Vec::new();
        let fused = rrf_fuse(&hnsw, &bm25, 0.7, 0.3, RRF_K, 10);
        assert_eq!(fused.len(), 2);
        assert_eq!(fused[0].0, "p");
    }

    #[test]
    fn test_rrf_top_k_truncates() {
        let hnsw = vec![(id("a"), 1.0), (id("b"), 0.9), (id("c"), 0.8)];
        let bm25: Vec<(String, f32)> = Vec::new();
        let fused = rrf_fuse(&hnsw, &bm25, 1.0, 0.0, RRF_K, 2);
        assert_eq!(fused.len(), 2);
    }

    #[test]
    fn test_rrf_uses_rank_not_score_magnitude() {
        // Even with hugely different raw scores, RRF only sees ranks.
        let hnsw = vec![(id("a"), 0.99), (id("b"), 0.98)];
        let bm25 = vec![(id("b"), 1000.0), (id("a"), 0.01)];
        let fused = rrf_fuse(&hnsw, &bm25, 0.5, 0.5, RRF_K, 10);
        // a: 1/(60+1) + 1/(60+2); b: 1/(60+2) + 1/(60+1) — symmetric → tie → id sort.
        assert!((fused[0].1 - fused[1].1).abs() < 1e-6);
    }
}
