//! Maximal Marginal Relevance (MMR) re-ranking for diversity.
//!
//! Why: RRF fusion produces a ranked list optimized for relevance, but adjacent
//! ranks are often near-duplicates (the same function chunked twice, sibling
//! impl blocks, etc.). MMR re-ranks by jointly maximising relevance and
//! diversity so the top-k carries less redundant information.
//!
//! What: greedy selection over `(chunk_id, score)` candidates. At each step,
//! picks the candidate maximising
//!     `λ * relevance - (1 - λ) * max_sim(candidate, already_selected)`.
//! `λ = 0.5` is the standard balance.
//!
//! Test: see `tests` below — three candidates A, B (≈A), C (≠A) verifies the
//! greedy ordering picks A → C → B, and a missing-embedding case falls back to
//! the input order without panicking.

use std::collections::HashMap;

/// Default λ for MMR (relevance vs diversity weight). 0.5 = balanced.
pub const DEFAULT_LAMBDA: f32 = 0.5;

/// Cosine similarity between two equal-length f32 vectors.
///
/// Returns 0.0 if either vector is zero-length or has zero norm — these are
/// degenerate inputs (an embedder bug or a freshly-zeroed buffer) and we'd
/// rather return a benign "no similarity" than NaN.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0_f32;
    let mut na = 0.0_f32;
    let mut nb = 0.0_f32;
    for i in 0..a.len() {
        let av = a[i];
        let bv = b[i];
        dot += av * bv;
        na += av * av;
        nb += bv * bv;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

/// MMR score for one candidate against the currently-selected set.
fn compute_mmr_score(
    id: &str,
    relevance: f32,
    selected: &[(String, f32)],
    embeddings: &HashMap<String, Vec<f32>>,
    lambda: f32,
) -> f32 {
    let max_sim = selected
        .iter()
        .filter_map(|(sel_id, _)| {
            let a = embeddings.get(id)?;
            let b = embeddings.get(sel_id)?;
            Some(cosine_similarity(a, b))
        })
        .fold(0.0_f32, f32::max);
    lambda * relevance - (1.0 - lambda) * max_sim
}

/// Re-rank `candidates` (already sorted by relevance desc) using MMR.
///
/// `embeddings` maps `chunk_id → embedding`. Candidates without an embedding
/// in the map are still selectable — they simply contribute zero similarity to
/// the diversity term, which makes MMR behave like pure relevance for them.
/// This is the "graceful fallback" path the spec asks for.
///
/// Returns at most `top_k` `(chunk_id, score)` pairs in greedy MMR order.
pub fn mmr_rerank(
    candidates: Vec<(String, f32)>,
    embeddings: &HashMap<String, Vec<f32>>,
    lambda: f32,
    top_k: usize,
) -> Vec<(String, f32)> {
    let mut selected: Vec<(String, f32)> = Vec::with_capacity(top_k.min(candidates.len()));
    let mut remaining = candidates;

    while selected.len() < top_k && !remaining.is_empty() {
        let best_idx = remaining
            .iter()
            .enumerate()
            .max_by(|(_, (id_a, score_a)), (_, (id_b, score_b))| {
                let mmr_a = compute_mmr_score(id_a, *score_a, &selected, embeddings, lambda);
                let mmr_b = compute_mmr_score(id_b, *score_b, &selected, embeddings, lambda);
                mmr_a
                    .partial_cmp(&mmr_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i);

        match best_idx {
            Some(idx) => {
                let item = remaining.remove(idx);
                selected.push(item);
            }
            None => break,
        }
    }
    selected
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pad(mut v: Vec<f32>, dim: usize) -> Vec<f32> {
        v.resize(dim, 0.0);
        v
    }

    #[test]
    fn test_mmr_picks_diverse_after_top() {
        // A and B point in nearly the same direction (similar); C is orthogonal.
        // After A is picked first (highest relevance), MMR should prefer C over
        // B because C is more diverse from A.
        let dim = 8;
        let mut embeddings: HashMap<String, Vec<f32>> = HashMap::new();
        embeddings.insert("A".to_string(), pad(vec![1.0, 0.0], dim));
        embeddings.insert("B".to_string(), pad(vec![1.0, 0.0], dim));
        embeddings.insert("C".to_string(), pad(vec![0.0, 1.0], dim));

        let cands = vec![
            ("A".to_string(), 1.0),
            ("B".to_string(), 0.9),
            ("C".to_string(), 0.8),
        ];
        let out = mmr_rerank(cands, &embeddings, 0.5, 3);
        let ids: Vec<&str> = out.iter().map(|(id, _)| id.as_str()).collect();
        assert_eq!(ids, vec!["A", "C", "B"], "MMR should pick A → C → B");
    }

    #[test]
    fn test_mmr_top_k_respected() {
        let mut embeddings: HashMap<String, Vec<f32>> = HashMap::new();
        embeddings.insert("A".to_string(), vec![1.0, 0.0]);
        embeddings.insert("B".to_string(), vec![0.0, 1.0]);
        let cands = vec![("A".to_string(), 1.0), ("B".to_string(), 0.5)];
        let out = mmr_rerank(cands, &embeddings, 0.5, 1);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, "A");
    }

    #[test]
    fn test_mmr_missing_embeddings_falls_back_gracefully() {
        // No embeddings supplied → diversity term is always 0 → MMR degenerates
        // to pure relevance order (multiplied by λ, but order preserved).
        let embeddings: HashMap<String, Vec<f32>> = HashMap::new();
        let cands = vec![
            ("A".to_string(), 1.0),
            ("B".to_string(), 0.9),
            ("C".to_string(), 0.8),
        ];
        let out = mmr_rerank(cands, &embeddings, 0.5, 3);
        let ids: Vec<&str> = out.iter().map(|(id, _)| id.as_str()).collect();
        assert_eq!(
            ids,
            vec!["A", "B", "C"],
            "missing embeddings → keep input order"
        );
    }

    #[test]
    fn test_cosine_similarity_basic() {
        assert!((cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
    }

    #[test]
    fn test_cosine_similarity_dim_mismatch() {
        assert_eq!(cosine_similarity(&[1.0], &[1.0, 0.0]), 0.0);
    }

    #[test]
    fn test_mmr_empty_candidates() {
        let embeddings: HashMap<String, Vec<f32>> = HashMap::new();
        let out = mmr_rerank(Vec::new(), &embeddings, 0.5, 5);
        assert!(out.is_empty());
    }
}
