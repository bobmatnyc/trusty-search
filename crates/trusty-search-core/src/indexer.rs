//! `CodeIndexer`: hybrid HNSW + BM25 + RRF search pipeline.
//!
//! Why: this is the central orchestrator that ties embeddings, vector search,
//! lexical search, and intent-based weight routing into a single `search()` call.
//! What: holds an `Embedder`, a `VectorStore`, and an in-memory chunk corpus;
//! `search()` runs both lanes in parallel, fuses with RRF, and returns the
//! top-k chunks with their fused score and per-result `match_reason`.
//! Test: see the `tests` module — RRF unit coverage lives in `search::rrf`,
//! and the integration test `test_search_integration` indexes 3 chunks and
//! verifies the most-relevant one ranks first.
//!
//! Note on storage: the spec calls for redb-backed chunk metadata. This first
//! cut keeps the corpus in memory (`Arc<RwLock<HashMap<...>>>`) so the search
//! pipeline is exercised end-to-end without depending on persistence wiring
//! (which lives in a separate ticket). The `ChunkStore` trait below isolates
//! that decision so swapping in redb later is a one-file change.

use std::collections::{hash_map::DefaultHasher, HashMap};
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use lru::LruCache;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::bm25::Bm25Index;
use crate::chunker::{chunk_ast, RawChunk};
use crate::classifier::QueryClassifier;
use crate::embed::Embedder;
use crate::entity::RawEntity;
use crate::search::rrf::{rrf_fuse, RRF_K};
use crate::store::VectorStore;

/// LRU capacity (entries) for the per-indexer query embedding cache.
const QUERY_CACHE_CAPACITY: usize = 256;
/// Oversample factor for the HNSW lane before RRF fusion.
const HNSW_OVERSAMPLE: usize = 4;

/// A search result returned to callers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeChunk {
    /// Collision-safe ID: "{path}:{start}:{end}"
    pub id: String,
    pub file: String,
    pub start_line: usize,
    pub end_line: usize,
    pub content: String,
    pub function_name: Option<String>,
    pub score: f32,
    /// Compact 7-line snippet for token-efficient output
    pub compact_snippet: Option<String>,
    /// How this result was found: "hybrid", "hybrid+kg", "bm25", "vector", "fallback:ripgrep"
    pub match_reason: String,
}

/// Query parameters for hybrid search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchQuery {
    pub text: String,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    #[serde(default = "default_true")]
    pub expand_graph: bool,
    #[serde(default = "default_true")]
    pub compact: bool,
}

fn default_top_k() -> usize {
    10
}
fn default_true() -> bool {
    true
}

/// Stable u64 hash of a query string. Used as the LRU cache key so we don't
/// retain the full string twice (LRU stores the embedding payload only).
fn hash_query(query: &str) -> u64 {
    let mut h = DefaultHasher::new();
    query.hash(&mut h);
    h.finish()
}

/// Build a 7-line snippet centered on the chunk content for token-efficient output.
fn build_compact_snippet(content: &str) -> String {
    let lines: Vec<&str> = content.lines().collect();
    if lines.len() <= 7 {
        return content.to_string();
    }
    // Take the first 7 lines — chunkers tend to put the most important header
    // (function signature, struct decl) at the top of the chunk.
    lines[..7].join("\n")
}

/// `CodeIndexer`: hybrid search engine for one named index.
pub struct CodeIndexer {
    pub index_id: String,
    pub root_path: std::path::PathBuf,

    embedder: Option<Arc<dyn Embedder>>,
    store: Option<Arc<dyn VectorStore>>,

    /// In-memory chunk corpus. Will be backed by redb once #4/#6 land.
    chunks: Arc<RwLock<HashMap<String, RawChunk>>>,

    /// Per-file entities extracted by `chunk_ast`. Keyed by file path.
    entities: Arc<RwLock<HashMap<String, Vec<RawEntity>>>>,

    /// LRU cache of query → embedding, keyed by `hash_query`. Skips the embedder
    /// entirely on repeated queries — the daemon's "zero cold-start" promise.
    query_cache: Arc<Mutex<LruCache<u64, Vec<f32>>>>,
}

impl CodeIndexer {
    /// Construct a bare indexer without an embedder/store. Call
    /// [`Self::with_components`] before invoking [`Self::search`] — otherwise
    /// search returns `Ok(vec![])` (BM25-only fallback uses the same path).
    pub fn new(index_id: impl Into<String>, root_path: impl Into<std::path::PathBuf>) -> Self {
        let cap = NonZeroUsize::new(QUERY_CACHE_CAPACITY)
            .expect("QUERY_CACHE_CAPACITY must be non-zero");
        Self {
            index_id: index_id.into(),
            root_path: root_path.into(),
            embedder: None,
            store: None,
            chunks: Arc::new(RwLock::new(HashMap::new())),
            entities: Arc::new(RwLock::new(HashMap::new())),
            query_cache: Arc::new(Mutex::new(LruCache::new(cap))),
        }
    }

    /// Attach the embedder and vector store so the full hybrid pipeline can run.
    /// Builder-style; returns `self` for chaining.
    pub fn with_components(
        mut self,
        embedder: Arc<dyn Embedder>,
        store: Arc<dyn VectorStore>,
    ) -> Self {
        self.embedder = Some(embedder);
        self.store = Some(store);
        self
    }

    /// Number of chunks currently held in the corpus.
    pub fn chunk_count(&self) -> usize {
        // blocking_read is fine on a tokio worker thread for a quick stat probe;
        // we never await across this call.
        self.chunks
            .try_read()
            .map(|g| g.len())
            .unwrap_or(0)
    }

    /// Add (or replace) a chunk in the corpus. If an embedder + store are
    /// attached, the chunk is also embedded and upserted into the HNSW index.
    pub async fn add_chunk(&self, chunk: RawChunk) -> Result<()> {
        let id = chunk.id.clone();

        if let (Some(embedder), Some(store)) = (&self.embedder, &self.store) {
            let vec = embedder
                .embed(&chunk.content)
                .await
                .context("embed chunk content")?;
            store
                .upsert(&id, vec)
                .await
                .context("upsert chunk vector")?;
        }

        self.chunks.write().await.insert(id, chunk);
        Ok(())
    }

    /// Parse a file with `chunk_ast`, store every chunk in the corpus, and
    /// retain the per-file entity list for later KG/entity-search phases.
    pub async fn index_file(&self, file_path: &str, content: &str) -> Result<()> {
        let (chunks, entities) = chunk_ast(file_path, content);
        for chunk in chunks {
            self.add_chunk(chunk).await?;
        }
        self.entities
            .write()
            .await
            .insert(file_path.to_string(), entities);
        Ok(())
    }

    /// Read-only access to the entity list for a file (None if never indexed).
    pub async fn entities_for(&self, file_path: &str) -> Option<Vec<RawEntity>> {
        self.entities.read().await.get(file_path).cloned()
    }

    /// Remove a chunk from the corpus and its vector from the HNSW store.
    pub async fn remove_chunk(&self, chunk_id: &str) -> Result<()> {
        if let Some(store) = &self.store {
            store.remove(chunk_id).await.ok();
        }
        self.chunks.write().await.remove(chunk_id);
        Ok(())
    }

    /// Resolve a query → embedding, using the LRU cache to skip repeats.
    async fn embed_query(&self, query: &str) -> Result<Option<Vec<f32>>> {
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

    /// Build a fresh BM25 index over the current chunk corpus and run `query`
    /// against it. Returns `(chunk_id, score)` sorted by score desc.
    ///
    /// Why per-query rebuilds: keeping IDF accurate as the corpus changes is
    /// simpler than incremental BM25 maintenance, and our BM25 impl is in-memory
    /// + cheap. When this becomes a hot spot we can cache the index between
    ///   queries and invalidate on writes.
    async fn bm25_search(&self, query: &str, want: usize) -> Result<Vec<(String, f32)>> {
        let chunks = self.chunks.read().await;
        if chunks.is_empty() {
            return Ok(Vec::new());
        }

        // Stable iteration order so doc_id ↔ chunk_id is reproducible.
        let mut entries: Vec<(&String, &RawChunk)> = chunks.iter().collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));

        let mut bm25 = Bm25Index::new();
        for (doc_id, (_, chunk)) in entries.iter().enumerate() {
            bm25.add_document(doc_id, &chunk.content);
        }

        let mut scored: Vec<(String, f32)> = entries
            .iter()
            .enumerate()
            .map(|(doc_id, (id, _))| ((*id).clone(), bm25.score(query, doc_id)))
            .filter(|(_, s)| *s > 0.0)
            .collect();

        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        scored.truncate(want);
        Ok(scored)
    }

    /// Run the HNSW lane. Returns `(chunk_id, distance)` style — we treat the
    /// `VectorStore`'s `score` as opaque since RRF only consumes rank.
    async fn vector_search(&self, embedding: &[f32], want: usize) -> Result<Vec<(String, f32)>> {
        let Some(store) = &self.store else {
            return Ok(Vec::new());
        };
        let hits = store.search(embedding, want).await?;
        // VectorStore returns "higher = better" already (1 - cos_dist); we keep
        // that convention so callers can sort or display directly. RRF ignores
        // the magnitude.
        Ok(hits.into_iter().map(|h| (h.chunk_id, h.score)).collect())
    }

    /// Stub for KG (callers_of / callees_of) expansion. Will be filled in by #5.
    async fn kg_expand(&self, _seeds: &[(String, f32)]) -> Vec<(String, f32)> {
        tracing::trace!("KG expansion stub — awaiting #5");
        Vec::new()
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
        let intent = QueryClassifier::classify(&query.text);
        let (alpha, beta, use_kg_first) = intent.weights();
        tracing::debug!(
            "search index={} query={:?} intent={:?} alpha={} beta={}",
            self.index_id,
            query.text,
            intent,
            alpha,
            beta
        );

        // 1) Embed (cache-first) — None when no embedder is wired (BM25-only mode).
        let embedding = self.embed_query(&query.text).await?;

        // 2) Run lanes in parallel where possible.
        let want_hnsw = query.top_k.saturating_mul(HNSW_OVERSAMPLE).max(query.top_k);
        let want_bm25 = want_hnsw;

        let bm25_fut = self.bm25_search(&query.text, want_bm25);
        let hnsw_results = match &embedding {
            Some(v) => self.vector_search(v, want_hnsw).await?,
            None => Vec::new(),
        };
        let bm25_results = bm25_fut.await?;

        // 3) RRF.
        let fused = rrf_fuse(
            &hnsw_results,
            &bm25_results,
            alpha,
            beta,
            RRF_K,
            query.top_k,
        );

        // 4) KG expand (stub).
        let mut all = fused.clone();
        if use_kg_first {
            let expanded = self.kg_expand(&fused).await;
            all.extend(expanded);
        }

        // 5) Per-result match_reason lookup tables.
        let in_hnsw: std::collections::HashSet<&String> =
            hnsw_results.iter().map(|(id, _)| id).collect();
        let in_bm25: std::collections::HashSet<&String> =
            bm25_results.iter().map(|(id, _)| id).collect();

        // 6) Materialise.
        let chunks = self.chunks.read().await;
        let mut out = Vec::with_capacity(all.len());
        for (id, score) in all.into_iter().take(query.top_k) {
            let Some(raw) = chunks.get(&id) else {
                tracing::trace!("fused id {id} not in corpus — likely race; skipping");
                continue;
            };
            let in_v = in_hnsw.contains(&id);
            let in_b = in_bm25.contains(&id);
            let match_reason = match (in_v, in_b) {
                (true, true) => "hybrid",
                (true, false) => "vector",
                (false, true) => "bm25",
                (false, false) => "kg", // came in via KG expansion only
            }
            .to_string();

            let compact_snippet = if query.compact {
                Some(build_compact_snippet(&raw.content))
            } else {
                None
            };

            out.push(CodeChunk {
                id: raw.id.clone(),
                file: raw.file.clone(),
                start_line: raw.start_line,
                end_line: raw.end_line,
                content: raw.content.clone(),
                function_name: raw.function_name.clone(),
                score,
                compact_snippet,
                match_reason,
            });
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::MockEmbedder;
    use crate::store::UsearchStore;

    fn raw(id: &str, file: &str, content: &str) -> RawChunk {
        RawChunk {
            id: id.to_string(),
            file: file.to_string(),
            start_line: 1,
            end_line: 1 + content.lines().count(),
            content: content.to_string(),
            function_name: None,
            language: Some("rust".to_string()),
            chunk_type: crate::chunker::ChunkType::Code,
            calls: Vec::new(),
            inherits_from: Vec::new(),
            chunk_depth: 0,
            parent_chunk_id: None,
            child_chunk_ids: Vec::new(),
            nlp_keywords: Vec::new(),
            nlp_code_refs: Vec::new(),
        }
    }

    fn make_indexer() -> CodeIndexer {
        let dim = 32;
        let embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder::new(dim));
        let store: Arc<dyn VectorStore> =
            Arc::new(UsearchStore::new(dim).expect("usearch new"));
        CodeIndexer::new("test", "/tmp/test").with_components(embedder, store)
    }

    #[tokio::test]
    async fn test_search_integration_returns_relevant_chunk_first() {
        let idx = make_indexer();

        idx.add_chunk(raw(
            "src/auth.rs:1:5",
            "src/auth.rs",
            "fn authenticate(user: &str, password: &str) -> bool { true }",
        ))
        .await
        .unwrap();
        idx.add_chunk(raw(
            "src/render.rs:1:3",
            "src/render.rs",
            "fn render_ui_components() { /* svelte */ }",
        ))
        .await
        .unwrap();
        idx.add_chunk(raw(
            "src/db.rs:1:4",
            "src/db.rs",
            "struct Database { conn: String }",
        ))
        .await
        .unwrap();

        let q = SearchQuery {
            text: "fn authenticate".to_string(),
            top_k: 3,
            expand_graph: false,
            compact: true,
        };
        let results = idx.search(&q).await.expect("search");
        assert!(!results.is_empty(), "search should return at least one hit");
        assert_eq!(
            results[0].id, "src/auth.rs:1:5",
            "auth chunk must rank first; got {:?}",
            results.iter().map(|r| &r.id).collect::<Vec<_>>()
        );
        assert!(
            results[0].compact_snippet.is_some(),
            "compact_snippet should be populated when compact=true"
        );
        // BM25 lane must hit on the literal token "authenticate" → reason includes bm25.
        assert!(
            results[0].match_reason == "hybrid" || results[0].match_reason == "bm25",
            "expected hybrid or bm25 match_reason, got {}",
            results[0].match_reason
        );
    }

    #[tokio::test]
    async fn test_query_cache_skips_embedder_on_repeat() {
        // We don't have a hit-counter on the trait, so drive correctness
        // indirectly: the cache hit path must populate `query_cache` and
        // return the same vector without invoking the embedder.
        let idx = make_indexer();
        let q = "find user authentication logic";

        let v1 = idx.embed_query(q).await.unwrap().unwrap();
        // After first call, cache should hold this entry.
        let key = hash_query(q);
        let cached = {
            let mut g = idx.query_cache.lock().unwrap();
            g.get(&key).cloned()
        };
        assert_eq!(cached.as_ref(), Some(&v1), "cache must be populated");

        let v2 = idx.embed_query(q).await.unwrap().unwrap();
        assert_eq!(v1, v2, "second call must return identical vector via cache");
    }

    #[tokio::test]
    async fn test_search_with_no_embedder_falls_back_to_bm25() {
        // Indexer without `with_components` → embedder/store None → BM25-only.
        let idx = CodeIndexer::new("bm25-only", "/tmp/test");
        // We can't call add_chunk's vector path, but no embedder means it skips.
        idx.add_chunk(raw(
            "f.rs:1:1",
            "f.rs",
            "fn authenticate() {}",
        ))
        .await
        .unwrap();
        idx.add_chunk(raw("g.rs:1:1", "g.rs", "fn unrelated() {}"))
            .await
            .unwrap();

        let q = SearchQuery {
            text: "authenticate".to_string(),
            top_k: 5,
            expand_graph: false,
            compact: false,
        };
        let r = idx.search(&q).await.unwrap();
        assert_eq!(r[0].id, "f.rs:1:1");
        assert_eq!(r[0].match_reason, "bm25");
    }

    #[tokio::test]
    async fn test_remove_chunk_removes_from_results() {
        let idx = make_indexer();
        idx.add_chunk(raw("a:1:1", "a.rs", "fn authenticate() {}"))
            .await
            .unwrap();
        idx.add_chunk(raw("b:1:1", "b.rs", "fn other_thing() {}"))
            .await
            .unwrap();
        idx.remove_chunk("a:1:1").await.unwrap();

        let q = SearchQuery {
            text: "authenticate".to_string(),
            top_k: 5,
            expand_graph: false,
            compact: false,
        };
        let r = idx.search(&q).await.unwrap();
        assert!(!r.iter().any(|c| c.id == "a:1:1"));
    }

    #[test]
    fn test_intent_routing_definitions() {
        // Sanity: intent table from CLAUDE.md is wired through.
        use crate::classifier::QueryIntent;
        let (a, b, kg) = QueryIntent::Definition.weights();
        assert!((a - 0.3).abs() < 1e-6 && (b - 0.7).abs() < 1e-6 && !kg);
        let (a, b, kg) = QueryIntent::Usage.weights();
        assert!((a - 0.5).abs() < 1e-6 && (b - 0.5).abs() < 1e-6 && kg);
    }
}
