//! Hybrid search pipeline for [`CodeIndexer`].
//!
//! Why: the search hot path (query classify → embed → HNSW + BM25 → RRF → KG
//! expand → MMR diversity → materialize) is the most-read code in this file
//! and benefits from living in a dedicated module away from ingest/persist.
//! What: holds `search`, the per-lane helpers (`embed_query`, `bm25_search`,
//! `vector_search`), the KG expansion logic (`kg_expand`, `edge_kinds_for_intent`),
//! and the post-fusion stages (`apply_mmr_rerank`, `apply_score_adjustments`,
//! `inject_entity_exact_match`, `expand_with_kg`, `materialize_search_results`).
//! Test: covered by every `test_search_*`, `test_kg_*`, and intent-routing test
//! in `indexer::tests`.

use std::collections::HashMap;

use anyhow::{Context, Result};

use crate::core::classifier::{QueryClassifier, QueryIntent};
use crate::core::entity::EdgeKind;
use crate::core::search::rrf::{rrf_fuse, RRF_K};

use super::{
    build_compact_snippet, compute_match_reason, file_type_score_multiplier, hash_query,
    raw_to_code_chunk, CodeChunk, CodeIndexer, SearchQuery, HNSW_OVERSAMPLE, KG_EXPAND_HOPS,
};

impl CodeIndexer {
    /// Retrieve a cached chunk embedding by `chunk_id`.
    ///
    /// Why: code-to-code similarity search (issue #31) needs the seed chunk's
    /// embedding to query the HNSW lane without re-embedding its source. We
    /// already populate `chunk_embeddings` on `add_chunk`, so this is an O(1)
    /// lookup. Returns `None` when the chunk doesn't exist or was indexed in
    /// BM25-only mode (no embedder wired).
    pub fn get_embedding(&self, chunk_id: &str) -> Option<Vec<f32>> {
        // `peek` doesn't promote the entry — we read through an `&RwLockReadGuard`
        // (immutable), and we don't want background reads to disturb LRU order
        // (only the write paths in `add_chunk` / batch commit promote on insert).
        self.chunk_embeddings
            .try_read()
            .ok()
            .and_then(|g| g.peek(chunk_id).cloned())
    }

    /// Resolve a query → embedding, using the LRU cache to skip repeats.
    pub(super) async fn embed_query(&self, query: &str) -> Result<Option<Vec<f32>>> {
        let Some(embedder) = self.embedder.clone() else {
            return Ok(None);
        };
        let key = hash_query(query);

        // Fast path: cache hit.
        if let Some(v) = self
            .query_cache
            .lock()
            .expect("query_cache mutex poisoned")
            .get(&key)
        {
            return Ok(Some(v.clone()));
        }

        let vec = embedder.embed(query).await.context("embed query")?;

        self.query_cache
            .lock()
            .expect("query_cache mutex poisoned")
            .put(key, vec.clone());

        Ok(Some(vec))
    }

    /// Run `query` against the hot, persistent BM25 index.
    ///
    /// Why: the previous implementation rebuilt the entire posting list on
    /// every search. On a 115k-chunk index that single line cost ~9.5s and
    /// caused all results to rank by BM25 alone (the HNSW lane completed
    /// fast but the latency budget was already gone). The index is now
    /// maintained incrementally by `add_chunk` / `index_files_batch` /
    /// `remove_*`, so the search hot path is just a read lock + posting walk.
    async fn bm25_search(&self, query: &str, want: usize) -> Result<Vec<(String, f32)>> {
        let bm25 = self.bm25.read().await;
        if bm25.is_empty() {
            return Ok(Vec::new());
        }
        Ok(bm25.score_query_all(query, want))
    }

    /// Run the HNSW lane. Returns `(chunk_id, distance)` style — we treat the
    /// `VectorStore`'s `score` as opaque since RRF only consumes rank.
    pub(super) async fn vector_search(
        &self,
        embedding: &[f32],
        want: usize,
    ) -> Result<Vec<(String, f32)>> {
        let Some(store) = &self.store else {
            return Ok(Vec::new());
        };
        let hits = store.search(embedding, want).await?;
        // VectorStore returns "higher = better" already (1 - cos_dist); we keep
        // that convention so callers can sort or display directly. RRF ignores
        // the magnitude.
        Ok(hits.into_iter().map(|h| (h.chunk_id, h.score)).collect())
    }

    /// Edge-kinds traversed for each query intent (issue #18).
    ///
    /// Each intent picks a small set of `EdgeKind`s most likely to surface
    /// adjacent code that's actually relevant to the question being asked.
    /// Score for each neighbour = `seed_score * edge_kind.score_multiplier()`.
    fn edge_kinds_for_intent(intent: QueryIntent) -> Vec<EdgeKind> {
        match intent {
            QueryIntent::Definition => {
                vec![EdgeKind::Implements, EdgeKind::Aliases, EdgeKind::UsesType]
            }
            QueryIntent::Usage => vec![
                EdgeKind::CallsFunction,
                EdgeKind::CalledByFunction,
                EdgeKind::TestedBy,
                EdgeKind::CoOccursInTest,
            ],
            QueryIntent::Conceptual => {
                vec![EdgeKind::ReferencesConcept, EdgeKind::Documents]
            }
            QueryIntent::BugDebt => vec![
                EdgeKind::RaisesError,
                EdgeKind::ErrorDescribes,
                EdgeKind::Configures,
            ],
            QueryIntent::Unknown => vec![EdgeKind::CallsFunction, EdgeKind::CalledByFunction],
        }
    }

    /// Intent-gated KG expansion (issue #18). For each seed
    /// `(chunk_id, score)`:
    /// 1. Look up the defining symbol of the seed chunk.
    /// 2. BFS its `EdgeKind`-filtered neighbourhood (intent-specific edges).
    /// 3. Score each neighbour as `seed_score * edge_kind.score_multiplier()`.
    ///
    /// Deduplicates: a chunk already in the seed set is never re-emitted; a
    /// chunk reachable through multiple seed/edge paths keeps its best score.
    async fn kg_expand(&self, seeds: &[(String, f32)], intent: QueryIntent) -> Vec<(String, f32)> {
        let graph = self.symbol_graph().await;
        if graph.node_count() == 0 || seeds.is_empty() {
            return Vec::new();
        }

        let edge_kinds = Self::edge_kinds_for_intent(intent);
        let seed_ids: std::collections::HashSet<&String> = seeds.iter().map(|(id, _)| id).collect();
        let mut best: HashMap<String, f32> = HashMap::new();

        for (seed_id, seed_score) in seeds {
            let Some(symbol) = graph.symbol_for_chunk(seed_id) else {
                continue;
            };
            for (_, neighbour_id, edge_kind) in
                graph.neighbors_by_edge(symbol, &edge_kinds, KG_EXPAND_HOPS)
            {
                if seed_ids.contains(&neighbour_id) {
                    continue;
                }
                let derived = seed_score * edge_kind.score_multiplier();
                best.entry(neighbour_id)
                    .and_modify(|s| {
                        if derived > *s {
                            *s = derived;
                        }
                    })
                    .or_insert(derived);
            }
        }

        let mut out: Vec<(String, f32)> = best.into_iter().collect();
        // Stable order: score desc, then id asc.
        out.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        out
    }

    /// Hybrid search: classify intent → route weights → HNSW + BM25 → RRF → KG.
    ///
    /// Steps:
    /// 1. Classify intent (regex-based, sub-ms) and pick `(alpha, beta, use_kg_first)`.
    /// 2. Embed the query (LRU-cached).
    /// 3. Run HNSW (`top_k * 4` candidates) and BM25 in parallel.
    /// 4. Fuse with RRF (`k=60`).
    /// 5. KG-expand (stub) when intent says so.
    /// 6. Materialise the top `top_k` chunk IDs into `CodeChunk`s with the
    ///    fused score and per-result `match_reason`.
    pub async fn search(&self, query: &SearchQuery) -> Result<Vec<CodeChunk>> {
        // Use the domain-aware classifier so per-index vocabulary from
        // `trusty-search.yaml` (`domain_terms:`) nudges otherwise-`Unknown`
        // queries to `Definition` intent. Falls back to plain `classify` when
        // `domain_terms` is empty (the common single-index case).
        let intent = QueryClassifier::classify_with_domain(&query.text, &self.domain_terms);
        let (alpha, beta, use_kg_first) = intent.weights();
        tracing::debug!(
            "search index={} query={:?} intent={:?} alpha={} beta={}",
            self.index_id,
            query.text,
            intent,
            alpha,
            beta
        );

        // 1) Embed (cache-first) — None when no embedder is wired.
        let embedding = self.embed_query(&query.text).await?;

        // 2) Run lanes (HNSW + BM25), then inject entity-exact-match if applicable.
        let want = query.top_k.saturating_mul(HNSW_OVERSAMPLE).max(query.top_k);
        let bm25_fut = self.bm25_search(&query.text, want);
        let hnsw_results = match &embedding {
            Some(v) => self.vector_search(v, want).await?,
            None => Vec::new(),
        };
        let mut bm25_results = bm25_fut.await?;
        self.inject_entity_exact_match(&intent, &query.text, beta, &mut bm25_results)
            .await;

        // 3) RRF fuse, then MMR diversity.
        let fused_raw = rrf_fuse(
            &hnsw_results,
            &bm25_results,
            alpha,
            beta,
            RRF_K,
            query.top_k,
        );
        let fused = self.apply_mmr_rerank(fused_raw, query.top_k).await;

        // 4) KG expand (conditional). Track which IDs came **only** from KG
        //    so the materialization step can label them "hybrid+kg".
        let (all, kg_ids) = self
            .expand_with_kg(fused, &intent, use_kg_first, query.expand_graph)
            .await;

        // 4a) Re-rank by score after KG expansion (issue #94): KG-expanded
        //     neighbours are appended after the fused list, so a naïve
        //     `take(top_k)` would silently discard them. Sort the merged
        //     `(id, score)` list so well-scored KG hits survive truncation
        //     and `match_reason: "hybrid+kg"` actually surfaces in results.
        // 4b) Apply a file-type multiplier for Definition intent (issue #92):
        //     when the user is looking for a symbol definition, prefer source
        //     files over docs/configs whose BM25 TF can spuriously rank them
        //     above the canonical .rs/.py/.go declaration.
        let all = self.apply_score_adjustments(all, &intent).await;

        // 5) Materialise the top-k IDs into `CodeChunk`s.
        let result = self
            .materialize_search_results(all, &hnsw_results, &bm25_results, &kg_ids, query)
            .await;
        Ok(result)
    }

    /// Re-rank merged direct+KG candidates and apply file-type weighting.
    ///
    /// Why: KG-expanded neighbours are appended after the RRF-fused list, so
    /// the naïve `take(top_k)` in `materialize_search_results` used to drop
    /// them (issue #94). At the same time, Definition-intent queries used to
    /// rank `.md` docs above source files because they had high BM25 TF for
    /// symbol names (issue #92). We solve both by adjusting every candidate's
    /// score in a single pass and re-sorting before truncation.
    /// What: for `Definition` intent, multiplies the score of each candidate
    /// by `0.5` if its file extension is in `DOC_EXTENSIONS`; for every other
    /// intent the multiplier is `1.0`. Then re-sorts by score descending,
    /// with id as a stable tie-breaker.
    /// Test: covered by `test_definition_demotes_markdown_below_source` and
    /// `test_kg_results_survive_top_k_truncation`.
    async fn apply_score_adjustments(
        &self,
        candidates: Vec<(String, f32)>,
        intent: &QueryIntent,
    ) -> Vec<(String, f32)> {
        let demote_docs = matches!(intent, QueryIntent::Definition);
        let chunks = self.chunks.read().await;
        let mut adjusted: Vec<(String, f32)> = candidates
            .into_iter()
            .map(|(id, score)| {
                let multiplier = if demote_docs {
                    chunks
                        .get(&id)
                        .map(|raw| file_type_score_multiplier(&raw.file))
                        .unwrap_or(1.0)
                } else {
                    1.0
                };
                (id, score * multiplier)
            })
            .collect();
        adjusted.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        adjusted
    }

    /// Issue #20: when intent is Definition or Unknown (a likely symbol
    /// lookup), inject the exact-name entity hit as the rank-1 BM25 result.
    ///
    /// Why: keeps the RRF lane seeing a strong signal even when the literal
    /// token didn't tokenize (e.g. underscore-heavy names). Lifting this out
    /// of `search` shrinks the latter's cyclomatic complexity.
    /// What: scoped to two intents; when an entity match is found, dedupes
    /// any prior occurrence and prepends a synthetic `(id, beta * 1.5)` pair.
    /// Test: covered by `test_entity_exact_match_struct_ranks_first`.
    async fn inject_entity_exact_match(
        &self,
        intent: &QueryIntent,
        query_text: &str,
        beta: f32,
        bm25_results: &mut Vec<(String, f32)>,
    ) {
        if !matches!(intent, QueryIntent::Definition | QueryIntent::Unknown) {
            return;
        }
        let Some(hit) = self.entity_exact_match(query_text).await else {
            return;
        };
        let injected_score = beta * 1.5;
        bm25_results.retain(|(id, _)| id != &hit);
        bm25_results.insert(0, (hit, injected_score));
    }

    /// MMR diversity pass (#28) over the RRF-fused candidate list.
    ///
    /// Why: re-ranks so adjacent near-duplicates don't crowd the top-k.
    /// λ=`DEFAULT_LAMBDA` (=0.5) balances relevance vs diversity.
    /// What: snapshots the embedding cache; if empty (BM25-only mode) falls
    /// back to the input order gracefully.
    /// Test: covered indirectly by every search integration test.
    async fn apply_mmr_rerank(
        &self,
        fused_raw: Vec<(String, f32)>,
        top_k: usize,
    ) -> Vec<(String, f32)> {
        // Snapshot only the candidate embeddings out of the LRU into a
        // transient `HashMap` for MMR. `peek` avoids promoting entries on
        // read (we only want the embed pipeline / batch commit to reorder
        // the LRU). Missing entries are handled gracefully by MMR — it
        // simply contributes zero diversity for that candidate.
        let emb_map = self.chunk_embeddings.read().await;
        if emb_map.is_empty() {
            return fused_raw;
        }
        let snapshot: HashMap<String, Vec<f32>> = fused_raw
            .iter()
            .filter_map(|(id, _)| emb_map.peek(id).map(|v| (id.clone(), v.clone())))
            .collect();
        drop(emb_map);
        crate::core::mmr::mmr_rerank(
            fused_raw,
            &snapshot,
            crate::core::mmr::DEFAULT_LAMBDA,
            top_k,
        )
    }

    /// KG expand the fused list when `use_kg_first` is on and the caller
    /// hasn't disabled `expand_graph`.
    ///
    /// Why: lifts the conditional and the "which-ids-came-only-from-KG"
    /// bookkeeping out of `search`.
    /// What: returns `(all_candidates, kg_only_ids)`. `all_candidates`
    /// starts as `fused` and is extended with KG-derived `(id, score)` pairs.
    /// Test: covered by `test_kg_expansion_marks_neighbours_with_hybrid_kg`
    /// and `test_kg_expansion_disabled_by_expand_graph_false`.
    async fn expand_with_kg(
        &self,
        fused: Vec<(String, f32)>,
        intent: &QueryIntent,
        use_kg_first: bool,
        expand_graph: bool,
    ) -> (Vec<(String, f32)>, std::collections::HashSet<String>) {
        let mut all = fused.clone();
        if !(use_kg_first && expand_graph) {
            return (all, std::collections::HashSet::new());
        }
        let expanded = self.kg_expand(&fused, intent.clone()).await;
        let kg_ids: std::collections::HashSet<String> =
            expanded.iter().map(|(id, _)| id.clone()).collect();
        all.extend(expanded);
        (all, kg_ids)
    }

    /// Materialize the top-k `(id, score)` pairs into `CodeChunk`s with the
    /// correct `match_reason` derived from the source lanes.
    ///
    /// Why: isolates the final per-result loop (lookup table joins, snippet
    /// construction, RawChunk → CodeChunk) so `search` stays focused on
    /// orchestration.
    /// What: builds lookup sets for HNSW and BM25 hit IDs, then for each of
    /// the top-k `(id, score)` pairs picks a `match_reason` and emits a
    /// `CodeChunk` via `raw_to_code_chunk`.
    /// Test: covered by every search integration test.
    async fn materialize_search_results(
        &self,
        all: Vec<(String, f32)>,
        hnsw_results: &[(String, f32)],
        bm25_results: &[(String, f32)],
        kg_ids: &std::collections::HashSet<String>,
        query: &SearchQuery,
    ) -> Vec<CodeChunk> {
        let in_hnsw: std::collections::HashSet<&String> =
            hnsw_results.iter().map(|(id, _)| id).collect();
        let in_bm25: std::collections::HashSet<&String> =
            bm25_results.iter().map(|(id, _)| id).collect();

        let chunks = self.chunks.read().await;
        let mut out = Vec::with_capacity(all.len().min(query.top_k));
        for (id, score) in all.into_iter().take(query.top_k) {
            let Some(raw) = chunks.get(&id) else {
                tracing::trace!("fused id {id} not in corpus — likely race; skipping");
                continue;
            };
            let match_reason = compute_match_reason(
                in_hnsw.contains(&id),
                in_bm25.contains(&id),
                kg_ids.contains(&id),
            );
            let snippet = if query.compact {
                Some(build_compact_snippet(&raw.content))
            } else {
                None
            };
            out.push(raw_to_code_chunk(raw, score, match_reason, snippet));
        }
        out
    }
}
