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
use crate::chunker::{chunk_ast, ChunkType, RawChunk};
use crate::classifier::{QueryClassifier, QueryIntent};
use crate::embed::Embedder;
use crate::entity::{EdgeKind, EntityType, RawEntity};
use crate::search::rrf::{rrf_fuse, RRF_K};
use crate::store::VectorStore;
use crate::symbol_graph::{ChunkTuple, SymbolGraph};

/// LRU capacity (entries) for the per-indexer query embedding cache.
const QUERY_CACHE_CAPACITY: usize = 256;
/// Oversample factor for the HNSW lane before RRF fusion.
const HNSW_OVERSAMPLE: usize = 4;
/// Legacy default score multiplier applied to chunks brought in via KG
/// expansion. Retained for backwards-compat documentation: the live pipeline
/// now uses [`EdgeKind::score_multiplier`] (issue #18) so each edge type
/// contributes its own weight. Tests still reference this constant when
/// validating the `CallsFunction` baseline.
#[allow(dead_code)]
const KG_EXPAND_SCORE_FACTOR: f32 = 0.7;
/// Default BFS depth for KG expansion (1 hop = direct callers/callees only).
const KG_EXPAND_HOPS: usize = 1;

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

    // Issue #29 — structural metadata propagated from RawChunk / entity extractor.
    /// Structural kind of this chunk (Function, Struct, Trait, …). Defaults to
    /// `Unknown` so older serialized payloads round-trip cleanly.
    #[serde(default)]
    pub chunk_type: ChunkType,
    /// Function/method names called within this chunk's body.
    #[serde(default)]
    pub calls: Vec<String>,
    /// Parent type names this chunk's type inherits from / implements.
    #[serde(default)]
    pub inherits_from: Vec<String>,
    /// Halstead-proxy complexity: unique operator + operand count over `content`.
    /// Zero when not computable.
    #[serde(default)]
    pub complexity_score: u32,
    /// Nesting depth of this chunk in the file's AST (0 = top-level).
    #[serde(default)]
    pub chunk_depth: u8,
}

/// Halstead-proxy complexity score: unique alphanumeric identifiers (operands)
/// plus unique single-char operator symbols. Cheap, no AST required.
fn compute_complexity(content: &str) -> u32 {
    use std::collections::HashSet;
    let mut operands: HashSet<&str> = HashSet::new();
    for tok in content.split(|c: char| !c.is_alphanumeric() && c != '_') {
        if !tok.is_empty() {
            operands.insert(tok);
        }
    }
    let mut operators: HashSet<char> = HashSet::new();
    for c in content.chars() {
        if matches!(
            c,
            '+' | '-' | '*' | '/' | '%' | '=' | '<' | '>' | '&' | '|' | '!' | '^' | '?' | ':'
        ) {
            operators.insert(c);
        }
    }
    (operands.len() + operators.len()) as u32
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

    /// Call graph derived from the chunk corpus. Rebuilt cheaply after each
    /// corpus mutation; reads via `Arc::clone` are lock-free.
    symbol_graph: Arc<RwLock<Arc<SymbolGraph>>>,
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
            symbol_graph: Arc::new(RwLock::new(Arc::new(SymbolGraph::new()))),
        }
    }

    /// Snapshot the current symbol graph. Cheap (`Arc::clone`); intended for
    /// read-only KG queries from concurrent search handlers.
    pub async fn symbol_graph(&self) -> Arc<SymbolGraph> {
        Arc::clone(&*self.symbol_graph.read().await)
    }

    /// Rebuild the symbol graph from the current corpus. Called after any
    /// mutation (`add_chunk`, `remove_chunk`, `index_file`). Rebuilding is
    /// O(N + E) over chunks/calls and the corpus is small + in-memory, so we
    /// favour simplicity over incremental maintenance.
    async fn rebuild_symbol_graph(&self) {
        let chunks = self.chunks.read().await;
        let tuples: Vec<ChunkTuple> = chunks
            .values()
            .map(|c| {
                (
                    c.id.clone(),
                    c.file.clone(),
                    c.function_name.clone(),
                    c.calls.clone(),
                    c.inherits_from.clone(),
                    c.chunk_type.clone(),
                )
            })
            .collect();
        drop(chunks);
        let new_graph = Arc::new(SymbolGraph::build_from_chunks(&tuples));
        *self.symbol_graph.write().await = new_graph;
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
        self.rebuild_symbol_graph().await;
        Ok(())
    }

    /// Parse a file with `chunk_ast`, store every chunk in the corpus, and
    /// retain the per-file entity list for later KG/entity-search phases.
    pub async fn index_file(&self, file_path: &str, content: &str) -> Result<()> {
        let (mut chunks, entities) = chunk_ast(file_path, content);

        // Issue #19: populate virtual_terms per chunk from entities whose source
        // line falls inside the chunk's [start_line, end_line] range. We dedupe
        // by entity text so heavy literal repeats don't dominate IDF.
        for chunk in chunks.iter_mut() {
            let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
            let mut terms: Vec<String> = Vec::new();
            for ent in &entities {
                if ent.line >= chunk.start_line
                    && ent.line <= chunk.end_line
                    && seen.insert(ent.text.as_str())
                {
                    terms.push(ent.text.clone());
                }
            }
            chunk.virtual_terms = terms;
        }

        for chunk in chunks {
            self.add_chunk(chunk).await?;
        }
        self.entities
            .write()
            .await
            .insert(file_path.to_string(), entities);
        // `add_chunk` already rebuilds, but we also rebuild once more here so a
        // partial failure mid-file doesn't leave a stale graph; this is cheap.
        self.rebuild_symbol_graph().await;
        Ok(())
    }

    /// Read-only access to the entity list for a file (None if never indexed).
    pub async fn entities_for(&self, file_path: &str) -> Option<Vec<RawEntity>> {
        self.entities.read().await.get(file_path).cloned()
    }

    /// Issue #20: exact-name entity lookup. Scans the in-memory entity index
    /// for an entry whose text matches `query` (case-insensitive, trimmed) and
    /// returns the chunk_id of a chunk in that entity's file whose source line
    /// range contains the entity. Returns the first match found — fine for
    /// rank-1 BM25 injection where we just need a strong anchor.
    ///
    /// Restricted to `NamedType` and `ModulePath` entities — these are the
    /// taxonomy members that behave like symbol names. Other entity types
    /// (string literals, annotations, error variants) are noisier and should
    /// not anchor an exact-match boost.
    async fn entity_exact_match(&self, query: &str) -> Option<String> {
        let needle = query.trim();
        if needle.is_empty() || needle.contains(' ') {
            // Multi-word queries are not symbol names; skip the exact-match path.
            return None;
        }
        let entities = self.entities.read().await;
        let chunks = self.chunks.read().await;
        for (file, ents) in entities.iter() {
            for ent in ents {
                if !matches!(
                    ent.entity_type,
                    EntityType::NamedType | EntityType::ModulePath
                ) {
                    continue;
                }
                if ent.text.eq_ignore_ascii_case(needle) {
                    // Find a chunk in `file` whose [start_line, end_line] contains ent.line.
                    if let Some(c) = chunks
                        .values()
                        .filter(|c| c.file == *file)
                        .find(|c| ent.line >= c.start_line && ent.line <= c.end_line)
                    {
                        return Some(c.id.clone());
                    }
                }
            }
        }
        None
    }

    /// Remove every chunk belonging to a file, plus its entity list.
    ///
    /// Why: `index-file` re-indexes a file in place, but file deletion (and
    /// `FileWatcher` rename/remove events) needs to drop all of a file's
    /// chunks at once. Returns the number of chunks removed.
    pub async fn remove_file(&self, file_path: &str) -> Result<usize> {
        let ids: Vec<String> = {
            let chunks = self.chunks.read().await;
            chunks
                .values()
                .filter(|c| c.file == file_path)
                .map(|c| c.id.clone())
                .collect()
        };
        let removed = ids.len();
        for id in &ids {
            if let Some(store) = &self.store {
                store.remove(id).await.ok();
            }
        }
        {
            let mut chunks = self.chunks.write().await;
            for id in &ids {
                chunks.remove(id);
            }
        }
        self.entities.write().await.remove(file_path);
        self.rebuild_symbol_graph().await;
        Ok(removed)
    }

    /// Remove a chunk from the corpus and its vector from the HNSW store.
    pub async fn remove_chunk(&self, chunk_id: &str) -> Result<()> {
        if let Some(store) = &self.store {
            store.remove(chunk_id).await.ok();
        }
        self.chunks.write().await.remove(chunk_id);
        self.rebuild_symbol_graph().await;
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
            // Issue #19: append entity-derived virtual_terms so symbolic queries
            // ("authenticate", "Arc", "ParseError") can match the chunk via BM25
            // even when those terms don't appear literally in the body.
            if chunk.virtual_terms.is_empty() {
                bm25.add_document(doc_id, &chunk.content);
            } else {
                let mut doc = String::with_capacity(
                    chunk.content.len()
                        + chunk.virtual_terms.iter().map(|t| t.len() + 1).sum::<usize>(),
                );
                doc.push_str(&chunk.content);
                for t in &chunk.virtual_terms {
                    doc.push(' ');
                    doc.push_str(t);
                }
                bm25.add_document(doc_id, &doc);
            }
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

    /// Edge-kinds traversed for each query intent (issue #18).
    ///
    /// Each intent picks a small set of `EdgeKind`s most likely to surface
    /// adjacent code that's actually relevant to the question being asked.
    /// Score for each neighbour = `seed_score * edge_kind.score_multiplier()`.
    fn edge_kinds_for_intent(intent: QueryIntent) -> Vec<EdgeKind> {
        match intent {
            QueryIntent::Definition => vec![
                EdgeKind::Implements,
                EdgeKind::Aliases,
                EdgeKind::UsesType,
            ],
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
            QueryIntent::Unknown => vec![
                EdgeKind::CallsFunction,
                EdgeKind::CalledByFunction,
            ],
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
    async fn kg_expand(
        &self,
        seeds: &[(String, f32)],
        intent: QueryIntent,
    ) -> Vec<(String, f32)> {
        let graph = self.symbol_graph().await;
        if graph.node_count() == 0 || seeds.is_empty() {
            return Vec::new();
        }

        let edge_kinds = Self::edge_kinds_for_intent(intent);
        let seed_ids: std::collections::HashSet<&String> =
            seeds.iter().map(|(id, _)| id).collect();
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
        let mut bm25_results = bm25_fut.await?;

        // Issue #20: when intent is Definition or Unknown (a likely symbol
        // lookup), check the entity index for an exact-name match and inject
        // it as the rank-1 BM25 hit so the RRF lane sees a strong signal even
        // if the literal token didn't tokenize (e.g. an underscore-heavy name).
        if matches!(intent, QueryIntent::Definition | QueryIntent::Unknown) {
            if let Some(hit) = self.entity_exact_match(&query.text).await {
                let injected_score = beta * 1.5;
                bm25_results.retain(|(id, _)| id != &hit);
                bm25_results.insert(0, (hit, injected_score));
            }
        }

        // 3) RRF.
        let fused = rrf_fuse(
            &hnsw_results,
            &bm25_results,
            alpha,
            beta,
            RRF_K,
            query.top_k,
        );

        // 4) KG expand. Only runs when intent routing requested it AND
        //    `expand_graph` wasn't disabled by the caller.
        let mut all = fused.clone();
        let kg_ids: std::collections::HashSet<String> = if use_kg_first && query.expand_graph {
            let expanded = self.kg_expand(&fused, intent.clone()).await;
            let ids: std::collections::HashSet<String> =
                expanded.iter().map(|(id, _)| id.clone()).collect();
            all.extend(expanded);
            ids
        } else {
            std::collections::HashSet::new()
        };

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
            let in_kg = kg_ids.contains(&id);
            // Per CLAUDE.md: KG-derived results carry "hybrid+kg". Direct hits
            // (BM25 and/or vector) take precedence — KG expansion deduplicates
            // against the seed set, so the "in_kg" arm only fires for chunks
            // whose sole path into the result set was the call graph.
            let match_reason = match (in_v, in_b, in_kg) {
                (true, true, _) => "hybrid",
                (true, false, _) => "vector",
                (false, true, _) => "bm25",
                (false, false, true) => "hybrid+kg",
                (false, false, false) => "fallback",
            }
            .to_string();

            let compact_snippet = if query.compact {
                Some(build_compact_snippet(&raw.content))
            } else {
                None
            };

            // chunk_depth on RawChunk is usize; clamp into u8 (deeply nested
            // ASTs beyond 255 are vanishingly rare and don't help routing).
            let chunk_depth: u8 = raw.chunk_depth.min(u8::MAX as usize) as u8;
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
                chunk_type: raw.chunk_type.clone(),
                calls: raw.calls.clone(),
                inherits_from: raw.inherits_from.clone(),
                complexity_score: compute_complexity(&raw.content),
                chunk_depth,
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
            virtual_terms: Vec::new(),
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

    #[tokio::test]
    async fn test_kg_expansion_marks_neighbours_with_hybrid_kg() {
        // Build a corpus where "login_handler" calls "authenticate".
        // Query for "authenticate" with Usage intent so KG expansion fires;
        // login_handler should appear via KG with match_reason "hybrid+kg".
        //
        // Use BM25-only mode (no embedder) so the vector lane can't pull
        // login_handler in as a near-neighbour and dilute the test signal.
        let idx = CodeIndexer::new("kg-test", "/tmp/test");
        // Caller's *body* deliberately omits the literal token "authenticate"
        // so BM25 / vector lanes won't surface it directly — its only path into
        // the result set is via KG expansion from the authenticate chunk.
        idx.add_chunk(RawChunk {
            id: "h:1".to_string(),
            file: "h.rs".to_string(),
            start_line: 1,
            end_line: 3,
            content: "fn login_handler() { /* dispatch to verifier */ }".to_string(),
            function_name: Some("login_handler".to_string()),
            language: Some("rust".to_string()),
            chunk_type: crate::chunker::ChunkType::Function,
            calls: vec!["authenticate".to_string()],
            inherits_from: Vec::new(),
            chunk_depth: 0,
            parent_chunk_id: None,
            child_chunk_ids: Vec::new(),
            nlp_keywords: Vec::new(),
            nlp_code_refs: Vec::new(),
            virtual_terms: Vec::new(),
        })
        .await
        .unwrap();
        idx.add_chunk(RawChunk {
            id: "a:1".to_string(),
            file: "a.rs".to_string(),
            start_line: 1,
            end_line: 1,
            content: "fn authenticate() {}".to_string(),
            function_name: Some("authenticate".to_string()),
            language: Some("rust".to_string()),
            chunk_type: crate::chunker::ChunkType::Function,
            calls: Vec::new(),
            inherits_from: Vec::new(),
            chunk_depth: 0,
            parent_chunk_id: None,
            child_chunk_ids: Vec::new(),
            nlp_keywords: Vec::new(),
            nlp_code_refs: Vec::new(),
            virtual_terms: Vec::new(),
        })
        .await
        .unwrap();

        // "callers of authenticate" → Usage intent → use_kg_first=true
        let q = SearchQuery {
            text: "callers of authenticate".to_string(),
            top_k: 10,
            expand_graph: true,
            compact: false,
        };
        let results = idx.search(&q).await.unwrap();
        let login = results
            .iter()
            .find(|c| c.id == "h:1")
            .expect("login_handler should surface via KG expansion");
        assert_eq!(
            login.match_reason, "hybrid+kg",
            "KG-expanded chunks must carry hybrid+kg marker, got {}",
            login.match_reason
        );

        // Verify the 0.7× score factor: login_handler's score should be
        // exactly 0.7 × the trigger chunk's RRF score (within fp tolerance),
        // unless it was also a direct hit (then RRF would have ranked it).
        let trigger = results
            .iter()
            .find(|c| c.id == "a:1")
            .expect("authenticate must appear directly");
        let expected = trigger.score * KG_EXPAND_SCORE_FACTOR;
        assert!(
            (login.score - expected).abs() < 1e-5,
            "expected KG score = 0.7 * {} = {}, got {}",
            trigger.score,
            expected,
            login.score
        );
    }

    #[tokio::test]
    async fn test_kg_expansion_disabled_by_expand_graph_false() {
        let idx = make_indexer();
        idx.add_chunk(RawChunk {
            id: "h:1".to_string(),
            file: "h.rs".to_string(),
            start_line: 1,
            end_line: 1,
            content: "fn caller() { target(); }".to_string(),
            function_name: Some("caller".to_string()),
            language: Some("rust".to_string()),
            chunk_type: crate::chunker::ChunkType::Function,
            calls: vec!["target".to_string()],
            inherits_from: Vec::new(),
            chunk_depth: 0,
            parent_chunk_id: None,
            child_chunk_ids: Vec::new(),
            nlp_keywords: Vec::new(),
            nlp_code_refs: Vec::new(),
            virtual_terms: Vec::new(),
        })
        .await
        .unwrap();
        idx.add_chunk(RawChunk {
            id: "t:1".to_string(),
            file: "t.rs".to_string(),
            start_line: 1,
            end_line: 1,
            content: "fn target() {}".to_string(),
            function_name: Some("target".to_string()),
            language: Some("rust".to_string()),
            chunk_type: crate::chunker::ChunkType::Function,
            calls: Vec::new(),
            inherits_from: Vec::new(),
            chunk_depth: 0,
            parent_chunk_id: None,
            child_chunk_ids: Vec::new(),
            nlp_keywords: Vec::new(),
            nlp_code_refs: Vec::new(),
            virtual_terms: Vec::new(),
        })
        .await
        .unwrap();

        let q = SearchQuery {
            text: "callers of target".to_string(),
            top_k: 10,
            expand_graph: false,
            compact: false,
        };
        let results = idx.search(&q).await.unwrap();
        assert!(
            !results.iter().any(|c| c.match_reason.contains("kg")),
            "expand_graph=false must suppress KG expansion, got {results:#?}"
        );
    }

    #[tokio::test]
    async fn test_symbol_graph_rebuilds_after_indexing() {
        let idx = make_indexer();
        assert_eq!(idx.symbol_graph().await.node_count(), 0);
        idx.index_file("a.rs", "fn alpha() { beta(); }\nfn beta() {}\n")
            .await
            .unwrap();
        let g = idx.symbol_graph().await;
        assert!(g.node_count() >= 2, "graph should hold alpha + beta");
        assert!(
            !g.callees_of("alpha", 1).is_empty(),
            "alpha should have a callee edge to beta"
        );
    }

    #[tokio::test]
    async fn test_entity_exact_match_finds_chunk() {
        // Issue #20: an exact-name entity hit should resolve to a chunk in the
        // entity's file whose line range contains the entity. We use a struct
        // declaration so the AST emits a NamedType that matches the query.
        let idx = make_indexer();
        idx.index_file("e.rs", "pub struct MyType { x: u32 }\nfn f() {}\n")
            .await
            .unwrap();
        let hit = idx.entity_exact_match("MyType").await;
        assert!(hit.is_some(), "expected entity_exact_match to find MyType");
        let hit_id = hit.unwrap();
        let chunks = idx.chunks.read().await;
        assert!(
            chunks.get(&hit_id).map(|c| c.file == "e.rs").unwrap_or(false),
            "matched chunk should live in e.rs",
        );
    }

    #[tokio::test]
    async fn test_entity_exact_match_struct_ranks_first() {
        // Issue #20: indexing a Rust snippet with `struct FooBar` and querying
        // "FooBar" must surface that chunk at rank 1 via the synthetic BM25
        // injection. We use BM25-only mode so the vector lane can't dilute
        // the signal with a near-neighbour.
        let idx = CodeIndexer::new("ent-rank-1", "/tmp/test");
        idx.index_file(
            "src/types.rs",
            "pub struct FooBar { pub x: u32 }\n\nfn unrelated() { let _ = 1; }\n",
        )
        .await
        .unwrap();
        idx.index_file("src/other.rs", "fn other_thing() {}\n")
            .await
            .unwrap();

        let q = SearchQuery {
            text: "FooBar".to_string(),
            top_k: 5,
            expand_graph: false,
            compact: false,
        };
        let results = idx.search(&q).await.expect("search");
        assert!(!results.is_empty(), "search must return at least one hit");
        assert_eq!(
            results[0].file, "src/types.rs",
            "FooBar's defining file must rank first; got {:?}",
            results.iter().map(|r| &r.file).collect::<Vec<_>>(),
        );
        assert!(
            results[0].content.contains("FooBar"),
            "rank-1 chunk must contain the FooBar definition; got {:?}",
            results[0].content,
        );
    }

    #[tokio::test]
    async fn test_entity_exact_match_skips_non_symbol_entities() {
        // Issue #20: only NamedType and ModulePath entities should anchor
        // exact-name boosts. A LiteralString like "this is a long literal"
        // appearing in a file must not be returned as an entity match.
        let idx = make_indexer();
        idx.index_file(
            "lit.rs",
            "fn f() { let _ = \"this is a long literal\"; }\n",
        )
        .await
        .unwrap();
        // Single-word literal subset that exists as a string token but is
        // neither a NamedType nor a ModulePath — must miss.
        assert!(
            idx.entity_exact_match("literal").await.is_none(),
            "non-symbol entity types must not satisfy entity_exact_match"
        );
    }

    #[tokio::test]
    async fn test_entity_exact_match_skips_multiword_query() {
        let idx = make_indexer();
        idx.index_file("e.rs", "use std::sync::Arc;\nfn f() {}\n")
            .await
            .unwrap();
        assert!(idx.entity_exact_match("Arc thing").await.is_none());
    }

    #[tokio::test]
    async fn test_virtual_terms_populated_from_entities() {
        // Issue #19: chunks should pick up entity text as virtual_terms so
        // BM25 matches symbolic queries that don't appear literally in the body.
        let idx = make_indexer();
        idx.index_file(
            "v.rs",
            "use std::sync::Arc;\nfn f() { let _x: Arc<String> = Arc::new(String::new()); }\n",
        )
        .await
        .unwrap();
        let chunks = idx.chunks.read().await;
        let f_chunk = chunks
            .values()
            .find(|c| c.function_name.as_deref() == Some("f"))
            .expect("f chunk");
        assert!(
            f_chunk.virtual_terms.iter().any(|t| t == "Arc"),
            "expected 'Arc' in virtual_terms, got {:?}",
            f_chunk.virtual_terms
        );
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
