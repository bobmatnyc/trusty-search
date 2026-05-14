//! `CodeIndexer`: hybrid HNSW + BM25 + RRF search pipeline.
//!
//! Why: this is the central orchestrator that ties embeddings, vector search,
//! lexical search, and intent-based weight routing into a single `search()` call.
//! What: holds an `Embedder`, a `VectorStore`, and an in-memory chunk corpus;
//! `search()` runs both lanes in parallel, fuses with RRF, and returns the
//! top-k chunks with their fused score and per-result `match_reason`.
//! Test: see the `tests` submodule — RRF unit coverage lives in `search::rrf`,
//! and the integration test `test_search_integration` indexes 3 chunks and
//! verifies the most-relevant one ranks first.
//!
//! Module layout (issue #96 — god-object split):
//!   * `mod.rs` (this file): types, free helpers, struct definition, constructors.
//!   * `ingest`: add/index/batch parse+embed/commit pipeline.
//!   * `persist`: snapshot/restore + background incremental persist.
//!   * `files`: remove + lookup + entity-exact-match helpers.
//!   * `search`: hybrid query pipeline (HNSW + BM25 + RRF + KG + MMR).
//!   * `tests`: every test in one place so private fields stay accessible.

use std::collections::{hash_map::DefaultHasher, HashMap};
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use lru::LruCache;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::core::bm25::Bm25Index;
use crate::core::chunker::{ChunkType, RawChunk};
use crate::core::embed::Embedder;
use crate::core::entity::RawEntity;
use crate::core::store::VectorStore;
use crate::core::symbol_graph::SymbolGraph;

mod files;
mod ingest;
mod persist;
mod search;

#[cfg(test)]
mod tests;

/// LRU capacity (entries) for the per-indexer query embedding cache.
const QUERY_CACHE_CAPACITY: usize = 256;
/// Oversample factor for the HNSW lane before RRF fusion.
pub(crate) const HNSW_OVERSAMPLE: usize = 4;
/// Default LRU capacity for the per-indexer chunk embedding cache.
///
/// Each entry is `dim × 4` bytes (384-dim f32 ≈ 1 536 B). 1 000 entries ≈
/// ~1.5 MB of RAM per index. Evicted entries are simply re-embedded on demand
/// (MMR rerank gracefully falls back when an embedding is missing). Lowered
/// from 10 000 → 1 000 (issue #79) after a daemon was observed at 43.9 GB RSS;
/// the cache was a meaningful contributor on multi-index hosts. Override
/// at runtime via `TRUSTY_EMBEDDING_CACHE`.
const DEFAULT_EMBEDDING_CACHE_CAP: usize = 1_000;

/// Read the embedding-cache LRU cap from the environment, with a sane default.
fn embedding_cache_cap() -> usize {
    std::env::var("TRUSTY_EMBEDDING_CACHE")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n: &usize| n > 0)
        .unwrap_or(DEFAULT_EMBEDDING_CACHE_CAP)
}

/// Default hard cap on chunks per index. Also used as the HNSW
/// `max_elements`-style sanity guard. 200 000 chunks × ~5 KB metadata ≈ 1.0 GB
/// of RAM-resident chunk corpus on a single index. Lowered from 500 000 →
/// 200 000 (issue #79) — the previous default permitted >2.5 GB / index just
/// for chunk metadata, on top of HNSW and BM25 structures. Operators with
/// large monorepos can still raise this via `TRUSTY_MAX_CHUNKS`.
const DEFAULT_MAX_CHUNKS_PER_INDEX: usize = 200_000;

/// Read the per-index chunk cap from the environment, with a sane default.
pub(crate) fn max_chunks_per_index() -> usize {
    std::env::var("TRUSTY_MAX_CHUNKS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n: &usize| n > 0)
        .unwrap_or(DEFAULT_MAX_CHUNKS_PER_INDEX)
}
/// Batch size for the fastembed ONNX call when bulk-indexing files.
///
/// 128 chunks per batch balances SIMD/tensor-setup amortisation against ONNX
/// session arena growth. ORT retains per-session activation buffers sized to
/// the largest batch it has seen; on large repos a 256-chunk batch combined
/// with a 512-file reindex batch caused the arena to grow into the tens of
/// GBs and trigger macOS Jetsam kills. 128 keeps the per-call tensor footprint
/// bounded while still being large enough to amortise ONNX kernel launch
/// overhead.
///
/// Override at runtime via `TRUSTY_MAX_BATCH_SIZE` (clamped to
/// `[EMBED_BATCH_MIN, EMBED_BATCH_MAX]`).
///
/// Default lowered from 512 → 128 (issue #79) — the ONNX activation arena
/// retains buffers sized to the largest batch it has seen, and on Apple
/// Silicon this triggered Jetsam kills on large repos. The authoritative
/// per-tier default is now computed by [`crate::core::MemoryPolicy`] and
/// written back into `TRUSTY_MAX_BATCH_SIZE` before this function is called,
/// so the constant here is only a safety net when the env is unset.
const DEFAULT_EMBED_BATCH_SIZE: usize = 32;
/// Floor for env-clamped batch size. Aligned with
/// `core::memory_policy::MIN_COMPUTED_BATCH_SIZE` (lowered from 32 → 8 after
/// the 94 GB reindex incident — the corrected 200 MB/slot ORT estimate makes
/// even 32 dangerous on the Medium tier).
const EMBED_BATCH_MIN: usize = 8;
/// Ceiling for env-clamped batch size. Aligned with the tier hard-cap
/// envelope in `core::memory_policy` (XLarge=64) PLUS headroom for the GPU
/// opt-out path (`TRUSTY_MAX_BATCH_SIZE_EXPLICIT=1` sets 512 on CUDA/CoreML).
/// Lowered from 2048 → 512 since no real workload above 512 has been
/// validated and the previous ceiling allowed catastrophic env-typo values.
const EMBED_BATCH_MAX: usize = 512;

/// Read the embedding batch size from `TRUSTY_MAX_BATCH_SIZE`, clamped to
/// `[EMBED_BATCH_MIN, EMBED_BATCH_MAX]`. Falls back to `DEFAULT_EMBED_BATCH_SIZE`
/// when unset or unparseable.
///
/// Why: large repos can exhaust process memory if batches grow unbounded. This
/// gives operators a runtime knob to dial batch size up (faster indexing on
/// memory-rich hosts) or down (safer on constrained hosts) without rebuilding.
/// What: parses env, clamps via `.clamp()`. Filter-then-clamp ensures both
/// missing and zero values fall through to the default.
/// Test: see `tests::test_embed_batch_size_env_clamp`.
pub(crate) fn embed_batch_size() -> usize {
    std::env::var("TRUSTY_MAX_BATCH_SIZE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .map(|n| n.clamp(EMBED_BATCH_MIN, EMBED_BATCH_MAX))
        .unwrap_or(DEFAULT_EMBED_BATCH_SIZE)
}
/// Legacy default score multiplier applied to chunks brought in via KG
/// expansion. Retained for backwards-compat documentation: the live pipeline
/// now uses [`EdgeKind::score_multiplier`] (issue #18) so each edge type
/// contributes its own weight. Tests still reference this constant when
/// validating the `CallsFunction` baseline.
#[allow(dead_code)]
pub(crate) const KG_EXPAND_SCORE_FACTOR: f32 = 0.7;
/// Default BFS depth for KG expansion (1 hop = direct callers/callees only).
pub(crate) const KG_EXPAND_HOPS: usize = 1;

/// A search result returned to callers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeChunk {
    /// Collision-safe ID: "{path}:{start}:{end}"
    pub id: String,
    pub file: String,
    #[serde(default)]
    pub language: Option<String>,
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
    /// Nesting depth of this chunk in the file's AST (0 = top-level).
    #[serde(default)]
    pub chunk_depth: u8,

    // Note: complexity metrics and git blame metadata are now owned by
    // trusty-analyzer (issue #71). Removing them here keeps `CodeChunk` lean
    // and avoids duplicating canonical computation.

    // Issue #10 — cross-project search fan-out: when a chunk is returned by
    // the global `POST /search` endpoint (or `search_all` MCP tool), this is
    // populated with the IndexId that produced it. `None` for per-index
    // search responses so older clients round-trip cleanly.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index_id: Option<String>,
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
pub(crate) fn hash_query(query: &str) -> u64 {
    let mut h = DefaultHasher::new();
    query.hash(&mut h);
    h.finish()
}

/// Build a 7-line snippet centered on the chunk content for token-efficient output.
pub(crate) fn build_compact_snippet(content: &str) -> String {
    let lines: Vec<&str> = content.lines().collect();
    if lines.len() <= 7 {
        return content.to_string();
    }
    // Take the first 7 lines — chunkers tend to put the most important header
    // (function signature, struct decl) at the top of the chunk.
    lines[..7].join("\n")
}

/// Materialize a `RawChunk` into a `CodeChunk` with the given score, match
/// reason, and optional compact snippet.
///
/// Why: four call sites (`similar_by_embedding`, `all_chunks`,
/// `enumerate_chunks`, the `search` materialization tail) used to inline the
/// same 18-field struct literal. Consolidating them removes ~60 lines of
/// duplication and the inevitable per-site drift when new fields are added.
/// What: clones every metadata field and derives `chunk_depth` (clamped to u8).
/// Test: covered indirectly by every search/materialization test in this file.
pub(crate) fn raw_to_code_chunk(
    raw: &RawChunk,
    score: f32,
    match_reason: &str,
    compact_snippet: Option<String>,
) -> CodeChunk {
    let chunk_depth: u8 = raw.chunk_depth.min(u8::MAX as usize) as u8;
    CodeChunk {
        id: raw.id.clone(),
        file: raw.file.clone(),
        language: raw.language.clone(),
        start_line: raw.start_line,
        end_line: raw.end_line,
        content: raw.content.clone(),
        function_name: raw.function_name.clone(),
        score,
        compact_snippet,
        match_reason: match_reason.to_string(),
        chunk_type: raw.chunk_type.clone(),
        calls: raw.calls.clone(),
        inherits_from: raw.inherits_from.clone(),
        chunk_depth,
        index_id: None,
    }
}

/// Populate `virtual_terms` on each chunk from entities whose source line
/// falls within the chunk's `[start_line, end_line]` range.
///
/// Why: two call sites (`index_file` and `parse_and_embed_files`) used the
/// same dedupe-by-entity-text loop. Extracting prevents drift.
/// What: for each chunk, walks `entities` once, inserting each entity's text
/// at most once into a fresh `virtual_terms` vector.
/// Test: covered by `test_virtual_terms_populated_from_entities`.
pub(crate) fn populate_virtual_terms(chunks: &mut [RawChunk], entities: &[RawEntity]) {
    for chunk in chunks.iter_mut() {
        let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
        let mut terms: Vec<String> = Vec::new();
        for ent in entities {
            if ent.line >= chunk.start_line
                && ent.line <= chunk.end_line
                && seen.insert(ent.text.as_str())
            {
                terms.push(ent.text.clone());
            }
        }
        chunk.virtual_terms = terms;
    }
}

/// Score multiplier applied to a chunk for Definition-intent queries (issue #92).
///
/// Why: Definition queries (e.g. "struct CodeChunk fields") should surface the
/// canonical source-file declaration, not the Markdown / TOML / YAML file that
/// happens to mention the symbol many times. We demote doc/config files by 50%
/// only for Definition intent; Conceptual queries still surface `.md` docs.
/// What: returns `0.5` when the path ends with a known doc/config extension,
/// `1.0` otherwise.
/// Test: covered by `test_file_type_multiplier_demotes_docs` and the
/// integration test `test_definition_demotes_markdown_below_source`.
pub(crate) fn file_type_score_multiplier(path: &str) -> f32 {
    const DOC_EXTENSIONS: &[&str] = &[".md", ".txt", ".toml", ".yaml", ".yml", ".json"];
    let lower = path.to_ascii_lowercase();
    if DOC_EXTENSIONS.iter().any(|ext| lower.ends_with(ext)) {
        0.5
    } else {
        1.0
    }
}

/// Map (`in_hnsw`, `in_bm25`, `in_kg`) booleans to a stable `match_reason`
/// label.
///
/// Why: lifted out of `search` to keep the materialization loop short and
/// to make the precedence rules unit-testable in isolation.
/// What: direct hits (HNSW and/or BM25) take precedence over KG-only paths.
/// Test: covered indirectly by `test_kg_expansion_marks_neighbours_with_hybrid_kg`.
pub(crate) fn compute_match_reason(in_v: bool, in_b: bool, in_kg: bool) -> &'static str {
    match (in_v, in_b, in_kg) {
        (true, true, _) => "hybrid",
        (true, false, _) => "vector",
        (false, true, _) => "bm25",
        (false, false, true) => "hybrid+kg",
        (false, false, false) => "fallback",
    }
}

/// On-disk shape of a chunk corpus snapshot (issue #85). Stored as JSON next
/// to the HNSW snapshot so the daemon can restore an index without re-parsing
/// the source tree.
///
/// Why: BM25 + the symbol graph are both derivable from the chunk corpus, so
/// persisting just the chunks (and the per-file entity lists) is enough to
/// warm-boot the whole search pipeline. We deliberately do NOT persist BM25
/// posting lists — rebuilding them from chunks at load time is O(N tokens)
/// and avoids a second on-disk schema to migrate.
/// What: versioned wrapper around `Vec<RawChunk>` plus the entities map.
/// Test: covered by `tests::test_save_chunks_roundtrip`.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct ChunkSnapshot {
    /// File-format version. Bump when changing the shape so older daemons
    /// fall through to the empty-corpus branch instead of producing garbage.
    pub(crate) version: u32,
    pub(crate) chunks: Vec<RawChunk>,
    pub(crate) entities: Vec<(String, Vec<RawEntity>)>,
}

/// Output of the parse+embed phase: chunks paired with their (optional)
/// embeddings plus the per-file entity lists, ready to be committed into the
/// indexer's shared state. Held without any write lock so it can be shipped
/// between async tasks freely.
#[derive(Default)]
pub struct ParsedBatch {
    pub chunks: Vec<RawChunk>,
    /// `embeddings[i]` is `Some(vec)` iff an embedder was wired during parse.
    /// Always the same length as `chunks`.
    pub embeddings: Vec<Option<Vec<f32>>>,
    pub entities_by_file: Vec<(String, Vec<RawEntity>)>,
    /// Wall-clock time spent in `parse_files_parallel` (tree-sitter chunking).
    pub parse_ms: u64,
    /// Wall-clock time spent in `embed_chunks_in_batches` (ONNX embedding).
    /// `0` when no embedder was wired (BM25-only mode).
    pub embed_ms: u64,
    /// Number of chunks for which `Some(embedding)` was produced. `0` means
    /// the embedder was unavailable and the index degraded to BM25-only mode.
    pub vector_count: usize,
}

/// Per-batch timings emitted by [`CodeIndexer::commit_parsed_batch`]. Captures
/// the cost of the commit-phase work (BM25 ingest, vector upsert, KG rebuild).
///
/// Why: surfaced to the reindex orchestrator so it can accumulate per-subsystem
/// totals across all batches and emit them in the SSE `complete` event. This
/// gives operators visibility into where indexing time was actually spent and
/// is the smoking-gun signal for the "embedder silently fell back to BM25"
/// failure mode (`vector_count == 0` while `chunks > 0`).
#[derive(Debug, Default, Clone, Copy)]
pub struct CommitTimings {
    /// Chunks added by this commit. May be 0 if the batch was empty.
    pub chunks: usize,
    /// Time spent under the BM25 write lock ingesting tokens for this batch.
    pub bm25_ms: u64,
    /// Time spent in the HNSW `upsert_batch` call (vectors only).
    pub vector_upsert_ms: u64,
    /// Time spent rebuilding the symbol graph at the end of this commit. `0`
    /// when `defer_graph_rebuild=true` (the reindex orchestrator path).
    pub kg_ms: u64,
}

/// `CodeIndexer`: hybrid search engine for one named index.
///
/// Fields are crate-visible so the submodule `impl` blocks (`ingest`, `persist`,
/// `files`, `search`) can mutate state without going through accessors. They
/// remain `pub(super)`-equivalent from outside `core::indexer`.
pub struct CodeIndexer {
    pub index_id: String,
    pub root_path: std::path::PathBuf,

    pub(super) embedder: Option<Arc<dyn Embedder>>,
    pub(super) store: Option<Arc<dyn VectorStore>>,

    /// In-memory chunk corpus. Will be backed by redb once #4/#6 land.
    pub(super) chunks: Arc<RwLock<HashMap<String, RawChunk>>>,

    /// Per-file entities extracted by `chunk_ast`. Keyed by file path.
    pub(super) entities: Arc<RwLock<HashMap<String, Vec<RawEntity>>>>,

    /// Cached chunk embeddings, keyed by `chunk_id`. Populated whenever an
    /// embedder is wired (`add_chunk` writes here). Used by the MMR diversity
    /// pass (#28) which needs vectors for already-ranked chunks without paying
    /// a re-embed or HNSW round-trip per candidate.
    ///
    /// Bounded by `embedding_cache_cap()` to keep the daemon from holding the
    /// entire corpus's embeddings in RAM (issue #75). Evicted entries are
    /// gracefully re-embedded on demand (MMR falls back to relevance-only when
    /// an entry is missing). Use `LruCache::put` / `peek` / `pop`.
    pub(super) chunk_embeddings: Arc<RwLock<LruCache<String, Vec<f32>>>>,

    /// Persistent BM25 index kept hot alongside the HNSW index. Mutated by
    /// `add_chunk` / `index_files_batch` / `remove_*` so the search hot path
    /// just acquires a read lock and runs `score_query_all` instead of
    /// rebuilding the entire posting list every query (was O(N) over all
    /// chunks; on a 115k-chunk index that dominated p50 latency by ~9s).
    pub(super) bm25: Arc<RwLock<Bm25Index>>,

    /// LRU cache of query → embedding, keyed by `hash_query`. Skips the embedder
    /// entirely on repeated queries — the daemon's "zero cold-start" promise.
    pub(super) query_cache: Arc<Mutex<LruCache<u64, Vec<f32>>>>,

    /// Call graph derived from the chunk corpus. Rebuilt cheaply after each
    /// corpus mutation; reads via `Arc::clone` are lock-free.
    pub(super) symbol_graph: Arc<RwLock<Arc<SymbolGraph>>>,

    /// Optional ONNX NER for `NaturalLanguagePhrase` extraction from doc
    /// comments (issue #23). Always present, but inert unless both the `ner`
    /// feature is compiled in and `~/.trusty-search/models/ner.onnx` exists.
    pub(super) ner: crate::core::ner::NerExtractor,

    /// Coalescing state for `spawn_incremental_persist` (memory-explosion fix).
    /// See [`PersistState`] for the full protocol description.
    pub(super) persist_state: Arc<PersistState>,

    /// Per-index domain vocabulary used by `QueryClassifier::classify_with_domain`
    /// at search time. Sourced from `trusty-search.yaml`'s `domain_terms:` field
    /// and forwarded by the daemon when constructing the indexer.
    ///
    /// Why: a query like "PMS integration" carries no syntactic signal the
    /// generic regex chain can match (no `fn`, `class`, `callers of`, …),
    /// so it falls into `Unknown` and gets generic weights. Per-index
    /// vocabulary lets the classifier nudge such queries to `Definition`
    /// intent, which routes them to the lexical-heavy weighting that finds
    /// the underlying symbol.
    /// What: a `Vec<String>` of case-insensitive substrings. Empty = standard
    /// classifier behaviour.
    /// Test: `tests::search_uses_domain_terms_when_provided`.
    pub(super) domain_terms: Vec<String>,
}

/// Coalescing state for `spawn_incremental_persist`.
///
/// Why: prior to this guard, every call to `commit_parsed_batch` spawned a
/// fire-and-forget tokio task that cloned the **entire** chunk corpus
/// (every `RawChunk.content` String) into a `Vec<RawChunk>` and serialized
/// it to JSON. On a 200k-chunk corpus that's ~400 MB of `Vec<RawChunk>`
/// plus another ~800 MB of serialized `Vec<u8>` per task. A reindex emits
/// one commit per 128 files, so a 76 800-file repo would stack ~600 of
/// these tasks. With no concurrency limit, RAM ballooned to 46–174 GB
/// before the OS killed the daemon (observed on ~/Duetto/cto and
/// ~/Duetto/repos/duetto). The `TRUSTY_MEMORY_LIMIT_MB` poller could not
/// catch it because the runaway allocator was a detached task ladder, not
/// the reindex loop itself.
///
/// What: `in_flight` guarantees only one persist task is alive at a time
/// for this index; `dirty` lets later commits coalesce — when the running
/// task completes it re-runs once if `dirty` was set during its snapshot,
/// guaranteeing the on-disk file converges to the latest in-memory state
/// without ever allocating more than ~1× the corpus footprint.
///
/// Test: `tests::test_persist_coalesces_concurrent_calls`.
#[derive(Debug, Default)]
pub(crate) struct PersistState {
    /// True while a persist task is actively snapshotting + writing.
    pub(crate) in_flight: AtomicBool,
    /// Set by every caller before checking `in_flight`. The active task clears
    /// this before snapshotting; if any caller re-sets it during the snapshot
    /// the task loops once more so the final on-disk file reflects the latest
    /// committed state.
    pub(crate) dirty: AtomicBool,
}

impl CodeIndexer {
    /// Construct a bare indexer without an embedder/store. Call
    /// [`Self::with_components`] before invoking [`Self::search`] — otherwise
    /// search returns `Ok(vec![])` (BM25-only fallback uses the same path).
    pub fn new(index_id: impl Into<String>, root_path: impl Into<std::path::PathBuf>) -> Self {
        let cap =
            NonZeroUsize::new(QUERY_CACHE_CAPACITY).expect("QUERY_CACHE_CAPACITY must be non-zero");
        let emb_cap = NonZeroUsize::new(embedding_cache_cap())
            .expect("embedding_cache_cap must be non-zero (env var filtered)");
        Self {
            index_id: index_id.into(),
            root_path: root_path.into(),
            embedder: None,
            store: None,
            chunks: Arc::new(RwLock::new(HashMap::new())),
            entities: Arc::new(RwLock::new(HashMap::new())),
            chunk_embeddings: Arc::new(RwLock::new(LruCache::new(emb_cap))),
            bm25: Arc::new(RwLock::new(Bm25Index::new())),
            query_cache: Arc::new(Mutex::new(LruCache::new(cap))),
            symbol_graph: Arc::new(RwLock::new(Arc::new(SymbolGraph::new()))),
            ner: crate::core::ner::NerExtractor::try_load(),
            persist_state: Arc::new(PersistState::default()),
            domain_terms: Vec::new(),
        }
    }

    /// Builder-style setter for the per-index domain vocabulary.
    ///
    /// Why: lets the daemon attach `trusty-search.yaml`'s `domain_terms:`
    /// without leaking the field into every constructor call site.
    /// What: stores the vector verbatim (case-insensitive matching happens
    /// inside `classify_with_domain`).
    /// Test: see `tests::search_uses_domain_terms_when_provided`.
    pub fn with_domain_terms(mut self, terms: Vec<String>) -> Self {
        self.domain_terms = terms;
        self
    }

    /// Replace the per-index domain vocabulary in place. Used by the daemon
    /// when restoring a persisted index — we already have an indexer via
    /// `build_indexer_with_persisted_state` and just need to attach the
    /// vocabulary alongside it.
    pub fn set_domain_terms(&mut self, terms: Vec<String>) {
        self.domain_terms = terms;
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
}
