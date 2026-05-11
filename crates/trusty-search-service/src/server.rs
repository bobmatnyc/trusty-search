//! HTTP daemon: axum router exposing the trusty-search REST API.
//!
//! Why: Single shared `SearchAppState` (wrapped in `Arc`) lets every handler
//! read from the `IndexRegistry` concurrently. `DashMap` shard-locks per index
//! so different indexes never contend, and `Arc<RwLock<CodeIndexer>>` allows
//! many simultaneous readers per index.
//!
//! What: Routes implement the API described in `CLAUDE.md`:
//! - `GET /health`
//! - `GET /indexes`                       list registered indexes
//! - `POST /indexes`                      register a new (empty) index
//! - `GET /indexes/:id/status`            chunk count + root path
//! - `POST /indexes/:id/search`           hybrid search
//! - `POST /indexes/:id/index-file`       add/update one file
//! - `POST /indexes/:id/remove-file`      drop a file's chunks
//! - `POST /indexes/:id/reindex`          fire-and-forget full reindex
//!
//! Test: `cargo test -p trusty-search-service` boots the router with an
//! in-process registry and exercises each endpoint.

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        Json,
    },
    routing::{delete, get, post},
    Router,
};
use dashmap::DashMap;
use futures::stream::{self, Stream, StreamExt};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::sync::Arc;
use std::time::Instant;
use tokio_stream::wrappers::BroadcastStream;
use trusty_search_core::{
    classifier::QueryClassifier,
    embed::Embedder,
    facts::{FactRecord, FactStore},
    indexer::{CodeIndexer, SearchQuery},
    registry::{IndexHandle, IndexId, IndexRegistry},
    store::{UsearchStore, VectorStore},
};

use crate::reindex::{spawn_reindex, ReindexProgress, ReindexStatus};

/// Shared state injected into every axum handler.
#[derive(Clone)]
pub struct SearchAppState {
    pub registry: IndexRegistry,
    /// Optional canonical facts store. `None` disables the `/facts` endpoints
    /// (they return 503 when unavailable) — useful for tests that don't need
    /// persistence.
    pub facts: Option<FactStore>,
    /// Per-index reindex progress (live counters + SSE replay buffer). Started
    /// by `POST /indexes/:id/reindex`, consumed by
    /// `GET /indexes/:id/reindex/stream`. Lazily populated.
    pub reindex_progress: Arc<DashMap<IndexId, Arc<ReindexProgress>>>,
    /// Process-wide embedder shared across every index so the (expensive)
    /// fastembed ONNX session is initialized once. `None` keeps the daemon
    /// in BM25-only mode — useful for tests that don't want to download the
    /// model. The vector dimensionality is read from the embedder.
    pub embedder: Option<Arc<dyn Embedder>>,
    /// Port the daemon ended up listening on. Injected into the served
    /// `index.html` as `window.__DAEMON_PORT__` so the SPA knows which host
    /// to call when opened directly. `None` falls back to 7878 in the UI.
    pub daemon_port: Option<u16>,
    /// Whether `OPENROUTER_API_KEY` is set when the daemon starts. Toggles
    /// the Chat panel in the SPA via `window.__OPENROUTER_ENABLED__`.
    pub openrouter_enabled: bool,
    /// Monotonic timestamp captured when the AppState was constructed.
    /// Used to compute `uptime_secs` in the `/health` response (issue #34).
    pub started_at: Instant,
}

impl SearchAppState {
    /// Convenience constructor for callers (`daemon`, tests) that want default
    /// reindex tracking without hand-rolling the `Arc<DashMap<…>>`. Defaults
    /// to BM25-only mode (no embedder); use [`Self::with_embedder`] to enable
    /// the vector lane.
    pub fn new(registry: IndexRegistry, facts: Option<FactStore>) -> Self {
        Self {
            registry,
            facts,
            reindex_progress: Arc::new(DashMap::new()),
            embedder: None,
            daemon_port: None,
            openrouter_enabled: std::env::var("OPENROUTER_API_KEY").is_ok(),
            started_at: Instant::now(),
        }
    }

    /// Builder-style: record the actual port the daemon bound. Used by
    /// the UI handler to inject `window.__DAEMON_PORT__`.
    pub fn with_daemon_port(mut self, port: u16) -> Self {
        self.daemon_port = Some(port);
        self
    }

    /// Builder-style: attach a shared embedder so newly registered indexes
    /// run the full hybrid pipeline. The embedder is shared across every
    /// index registered after this point.
    pub fn with_embedder(mut self, embedder: Arc<dyn Embedder>) -> Self {
        self.embedder = Some(embedder);
        self
    }
}

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
    version: &'static str,
    indexes: usize,
    uptime_secs: u64,
}

#[derive(Serialize)]
struct IndexListResponse {
    indexes: Vec<String>,
}

#[derive(Deserialize)]
pub struct CreateIndexRequest {
    pub id: String,
    pub root_path: std::path::PathBuf,
}

#[derive(Deserialize)]
pub struct IndexFileRequest {
    pub path: String,
    pub content: String,
}

#[derive(Deserialize)]
pub struct RemoveFileRequest {
    pub path: String,
}

/// Build the axum router with the shared state.
///
/// Wraps `state` in an `Arc` so every handler clones the pointer cheaply.
pub fn build_router(state: SearchAppState) -> Router {
    use crate::ui::{chat_handler, ui_asset_handler, ui_index_handler};
    let router = Router::new()
        .route("/health", get(health_handler))
        .route(
            "/indexes",
            get(list_indexes_handler).post(create_index_handler),
        )
        .route("/indexes/:id", delete(delete_index_handler))
        .route("/ui", get(ui_index_handler))
        .route("/ui/", get(ui_index_handler))
        .route("/ui/*path", get(ui_asset_handler))
        .route("/chat", post(chat_handler))
        .route("/search", post(global_search_handler))
        .route("/indexes/:id/search", post(search_handler))
        .route("/indexes/:id/search_similar", post(search_similar_handler))
        .route("/indexes/:id/status", get(index_status_handler))
        .route("/indexes/:id/index-file", post(index_file_handler))
        .route("/indexes/:id/remove-file", post(remove_file_handler))
        .route("/indexes/:id/reindex", post(reindex_handler))
        .route("/indexes/:id/reindex/stream", get(reindex_stream_handler))
        .route("/indexes/:id/chunks", get(get_index_chunks_handler))
        .route(
            "/indexes/:id/complexity_hotspots",
            get(complexity_hotspots_handler),
        )
        .route("/indexes/:id/smells", get(smells_handler))
        .route("/indexes/:id/quality", get(quality_handler))
        .route("/facts", get(list_facts_handler).post(upsert_fact_handler))
        .route("/facts/:id", delete(delete_fact_handler))
        .with_state(Arc::new(state));
    // Standard middleware stack (CORS, tracing, gzip) lives in trusty-common
    // so every trusty-* daemon ships with the same defaults.
    trusty_common::server::with_standard_middleware(router)
}

#[derive(Deserialize)]
pub struct FactQueryParams {
    pub subject: Option<String>,
    pub predicate: Option<String>,
    pub object: Option<String>,
}

/// Inbound payload for upserting a fact. `id` and `created_at` are derived
/// server-side; callers don't need to compute the hash.
#[derive(Deserialize)]
pub struct UpsertFactRequest {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub index_id: String,
    #[serde(default = "default_confidence")]
    pub confidence: f32,
    #[serde(default)]
    pub provenance: Vec<String>,
}

fn default_confidence() -> f32 {
    1.0
}

async fn list_facts_handler(
    State(state): State<Arc<SearchAppState>>,
    Query(params): Query<FactQueryParams>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let Some(store) = &state.facts else {
        return Err(StatusCode::SERVICE_UNAVAILABLE);
    };
    let hits = store
        .query(
            params.subject.as_deref(),
            params.predicate.as_deref(),
            params.object.as_deref(),
        )
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(serde_json::json!({
        "facts": hits,
        "count": hits.len(),
    })))
}

async fn upsert_fact_handler(
    State(state): State<Arc<SearchAppState>>,
    Json(req): Json<UpsertFactRequest>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let Some(store) = &state.facts else {
        return Err(StatusCode::SERVICE_UNAVAILABLE);
    };
    let mut fact = FactRecord::new(req.subject, req.predicate, req.object, req.index_id)
        .with_confidence(req.confidence);
    fact.provenance = req.provenance;
    let id = fact.id;
    store
        .upsert(fact)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(serde_json::json!({ "id": id, "upserted": true })))
}

async fn delete_fact_handler(
    State(state): State<Arc<SearchAppState>>,
    Path(id): Path<u64>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let Some(store) = &state.facts else {
        return Err(StatusCode::SERVICE_UNAVAILABLE);
    };
    let removed = store
        .delete(id)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(serde_json::json!({ "id": id, "removed": removed })))
}

async fn health_handler(State(state): State<Arc<SearchAppState>>) -> Json<HealthResponse> {
    // Why: open-mpm (and other external integrators) probe `/health` to detect
    // a running trusty-search daemon before spawning their own. Including
    // `indexes` count lets the caller verify the daemon is not only alive but
    // also has the expected registry populated (issue #34).
    // What: returns `{ status, version, indexes, uptime_secs }` where
    // `indexes` is the number of registered IndexHandles in the registry
    // and `uptime_secs` is wall-clock seconds since AppState construction.
    // Test: register N indexes, GET /health, assert `indexes == N` and
    // `uptime_secs >= 0`.
    Json(HealthResponse {
        status: "ok",
        version: env!("CARGO_PKG_VERSION"),
        indexes: state.registry.list().len(),
        uptime_secs: state.started_at.elapsed().as_secs(),
    })
}

async fn list_indexes_handler(State(state): State<Arc<SearchAppState>>) -> Json<IndexListResponse> {
    Json(IndexListResponse {
        indexes: state.registry.list().into_iter().map(|id| id.0).collect(),
    })
}

async fn create_index_handler(
    State(state): State<Arc<SearchAppState>>,
    Json(req): Json<CreateIndexRequest>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let id = IndexId::new(req.id.clone());
    if state.registry.get(&id).is_some() {
        return Ok(Json(serde_json::json!({
            "id": req.id,
            "created": false,
            "reason": "already exists",
        })));
    }
    // Bug A fix: when an embedder is attached to the shared state, wire the
    // newly created indexer with both an `Embedder` and a `VectorStore` so
    // the HNSW lane actually contributes results. Previously every index
    // was BM25-only because `with_components` was never called, which is
    // why the benchmark observed `match_reason: "bm25"` for 100% of hits.
    let mut indexer = CodeIndexer::new(req.id.clone(), req.root_path.clone());
    if let Some(embedder) = &state.embedder {
        let dim = embedder.dimension();
        match UsearchStore::new(dim) {
            Ok(store) => {
                let store: Arc<dyn VectorStore> = Arc::new(store);
                indexer = indexer.with_components(Arc::clone(embedder), store);
            }
            Err(e) => {
                tracing::error!(
                    "failed to allocate UsearchStore for index {}: {e} \
                     — index will run in BM25-only mode",
                    req.id
                );
            }
        }
    }
    let handle = IndexHandle {
        id: id.clone(),
        indexer: Arc::new(tokio::sync::RwLock::new(indexer)),
        root_path: req.root_path,
    };
    state.registry.register(handle);
    Ok(Json(serde_json::json!({ "id": req.id, "created": true })))
}

/// `DELETE /indexes/:id` — drop an index from the registry.
///
/// Why: The admin UI needs a way to evict mistakes / abandoned projects
/// without restarting the daemon. The on-disk redb store (if any) is left
/// alone — re-registering with the same id reuses it.
/// What: Calls `IndexRegistry::unregister`. Returns `{removed: bool}`.
/// Test: register → delete → list returns empty.
async fn delete_index_handler(
    State(state): State<Arc<SearchAppState>>,
    Path(id): Path<String>,
) -> Json<serde_json::Value> {
    let index_id = IndexId::new(id.clone());
    let removed = state.registry.unregister(&index_id);
    state.reindex_progress.remove(&index_id);
    Json(serde_json::json!({ "id": id, "removed": removed }))
}

async fn search_handler(
    State(state): State<Arc<SearchAppState>>,
    Path(id): Path<String>,
    Json(query): Json<SearchQuery>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let index_id = IndexId::new(id);
    let handle = state.registry.get(&index_id).ok_or(StatusCode::NOT_FOUND)?;
    let intent = QueryClassifier::classify(&query.text);
    let started = std::time::Instant::now();
    let indexer = handle.indexer.read().await;
    let results = indexer
        .search(&query)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let latency_ms = started.elapsed().as_millis() as u64;
    Ok(Json(serde_json::json!({
        "results": results,
        "intent": format!("{:?}", intent),
        "latency_ms": latency_ms,
    })))
}

/// Body for the global `POST /search` endpoint (issue #10 — cross-project
/// search fan-out).
///
/// Why: callers (LLM agents, the UI search bar) often don't know which
/// project an answer lives in. A single fan-out search across every
/// registered index, with results re-ranked via Reciprocal Rank Fusion, lets
/// them ask one question and get one merged answer.
#[derive(Deserialize)]
pub struct GlobalSearchRequest {
    pub query: String,
    #[serde(default = "default_global_top_k")]
    pub top_k: usize,
    /// When true, response chunks include the full `content` field. When
    /// false (default), the daemon still returns chunks with content — clients
    /// that want compact responses can read `compact_snippet`.
    #[serde(default)]
    pub full_content: bool,
}

fn default_global_top_k() -> usize {
    10
}

/// `POST /search` — fan-out hybrid search across every registered index.
///
/// Why: see [`GlobalSearchRequest`] doc. This is distinct from
/// `POST /indexes/:id/search`, which targets a single index.
/// What: runs per-index search concurrently, tags each result with its
/// `index_id`, then re-runs RRF (k=60) over the per-index ranked lists
/// (each index treated as an equally-weighted lane) and returns the top-k
/// merged results. Indexes that error during search are skipped (logged) so
/// one bad index doesn't take down the whole fan-out.
/// Test: `test_global_search_fans_out_and_merges` registers two indexes,
/// indexes a file into each, and asserts both contribute results tagged with
/// the right `index_id`.
async fn global_search_handler(
    State(state): State<Arc<SearchAppState>>,
    Json(req): Json<GlobalSearchRequest>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    use trusty_search_core::search::rrf::{rrf_fuse, RRF_K};

    let index_ids = state.registry.list();
    let total_indexes = index_ids.len();
    if index_ids.is_empty() {
        return Ok(Json(serde_json::json!({
            "results": Vec::<trusty_search_core::indexer::CodeChunk>::new(),
            "indexes_searched": Vec::<String>::new(),
            "total_indexes": 0_usize,
            "latency_ms": 0_u64,
            "intent": format!("{:?}", QueryClassifier::classify(&req.query)),
        })));
    }

    let started = std::time::Instant::now();
    let intent = QueryClassifier::classify(&req.query);

    // Build the same SearchQuery shape every per-index search uses. We
    // oversample per-index by passing the user's top_k unchanged: each lane
    // contributes up to top_k candidates, then RRF picks the best top_k
    // overall.
    let per_index_query = SearchQuery {
        text: req.query.clone(),
        top_k: req.top_k,
        expand_graph: true,
        compact: !req.full_content,
    };

    // Run all per-index searches concurrently. Any index that errors is
    // skipped with a log line so a single broken index doesn't 500 the
    // whole fan-out.
    let registry = state.registry.clone();
    let futures = index_ids.into_iter().map(|id| {
        let registry = registry.clone();
        let query = per_index_query.clone();
        async move {
            let handle = registry.get(&id)?;
            let indexer = handle.indexer.read().await;
            match indexer.search(&query).await {
                Ok(results) => Some((id, results)),
                Err(e) => {
                    tracing::warn!("global search: index {} errored: {e}", id);
                    None
                }
            }
        }
    });
    let per_index_results: Vec<(IndexId, Vec<trusty_search_core::indexer::CodeChunk>)> =
        futures::future::join_all(futures)
            .await
            .into_iter()
            .flatten()
            .collect();

    // Build a flat lookup table from "namespaced" chunk_id
    // ({index_id}::{chunk.id}) back to the tagged CodeChunk, plus per-index
    // ranked id lists for RRF. Namespacing is required because different
    // indexes can produce colliding chunk_ids (same relative file path in
    // two projects).
    let mut chunk_lookup: std::collections::HashMap<
        String,
        trusty_search_core::indexer::CodeChunk,
    > = std::collections::HashMap::new();
    let mut lanes: Vec<Vec<(String, f32)>> = Vec::with_capacity(per_index_results.len());
    let mut indexes_searched: Vec<String> = Vec::with_capacity(per_index_results.len());
    for (id, results) in per_index_results {
        indexes_searched.push(id.0.clone());
        let mut lane: Vec<(String, f32)> = Vec::with_capacity(results.len());
        for mut chunk in results {
            let namespaced = format!("{}::{}", id.0, chunk.id);
            // Tag the chunk with its origin index before storing it so the
            // returned CodeChunks know where they came from.
            chunk.index_id = Some(id.0.clone());
            lane.push((namespaced.clone(), chunk.score));
            chunk_lookup.insert(namespaced, chunk);
        }
        lanes.push(lane);
    }

    // RRF fuse across lanes. `rrf_fuse` takes exactly two lanes, so we fold
    // pairwise: start with empty + lane0, then merge each subsequent lane.
    // Each fold step uses alpha=1, beta=1 — every index lane contributes
    // equally. The output is sorted by fused score desc.
    let mut fused: Vec<(String, f32)> = Vec::new();
    let oversample = req.top_k.saturating_mul(4).max(req.top_k).max(10);
    for lane in lanes {
        fused = rrf_fuse(&fused, &lane, 1.0, 1.0, RRF_K, oversample);
    }
    fused.truncate(req.top_k);

    let results: Vec<trusty_search_core::indexer::CodeChunk> = fused
        .into_iter()
        .filter_map(|(id, fused_score)| {
            let mut chunk = chunk_lookup.remove(&id)?;
            chunk.score = fused_score;
            Some(chunk)
        })
        .collect();

    let latency_ms = started.elapsed().as_millis() as u64;
    Ok(Json(serde_json::json!({
        "results": results,
        "indexes_searched": indexes_searched,
        "total_indexes": total_indexes,
        "latency_ms": latency_ms,
        "intent": format!("{:?}", intent),
    })))
}

/// Body for `POST /indexes/:id/search_similar`.
///
/// Why: code-to-code similarity (issue #31). The caller knows the *file +
/// optional function name* of the chunk they want to find neighbours of, not
/// its synthetic chunk id.
#[derive(Deserialize)]
pub struct SearchSimilarRequest {
    pub file: String,
    #[serde(default)]
    pub function: Option<String>,
    #[serde(default = "default_similar_top_k")]
    pub top_k: usize,
}

fn default_similar_top_k() -> usize {
    10
}

async fn search_similar_handler(
    State(state): State<Arc<SearchAppState>>,
    Path(id): Path<String>,
    Json(req): Json<SearchSimilarRequest>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let index_id = IndexId::new(id);
    let handle = state.registry.get(&index_id).ok_or(StatusCode::NOT_FOUND)?;
    let started = std::time::Instant::now();
    let indexer = handle.indexer.read().await;
    let chunk_id = indexer
        .find_chunk_id(&req.file, req.function.as_deref())
        .await
        .ok_or(StatusCode::NOT_FOUND)?;
    let embedding = indexer
        .get_embedding(&chunk_id)
        .ok_or(StatusCode::NOT_FOUND)?;
    let results = indexer
        .similar_by_embedding(&embedding, req.top_k, Some(&chunk_id))
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let latency_ms = started.elapsed().as_millis() as u64;
    Ok(Json(serde_json::json!({
        "results": results,
        "seed_chunk_id": chunk_id,
        "latency_ms": latency_ms,
    })))
}

async fn index_status_handler(
    State(state): State<Arc<SearchAppState>>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let index_id = IndexId::new(id);
    let handle = state.registry.get(&index_id).ok_or(StatusCode::NOT_FOUND)?;
    let indexer = handle.indexer.read().await;
    Ok(Json(serde_json::json!({
        "index_id": index_id.0,
        "root_path": handle.root_path,
        "chunk_count": indexer.chunk_count(),
    })))
}

async fn index_file_handler(
    State(state): State<Arc<SearchAppState>>,
    Path(id): Path<String>,
    Json(req): Json<IndexFileRequest>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let index_id = IndexId::new(id);
    let handle = state.registry.get(&index_id).ok_or(StatusCode::NOT_FOUND)?;
    let indexer = handle.indexer.read().await;
    indexer
        .index_file(&req.path, &req.content)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(serde_json::json!({
        "index_id": index_id.0,
        "path": req.path,
        "indexed": true,
    })))
}

async fn remove_file_handler(
    State(state): State<Arc<SearchAppState>>,
    Path(id): Path<String>,
    Json(req): Json<RemoveFileRequest>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let index_id = IndexId::new(id);
    let handle = state.registry.get(&index_id).ok_or(StatusCode::NOT_FOUND)?;
    let indexer = handle.indexer.read().await;
    let removed = indexer
        .remove_file(&req.path)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(serde_json::json!({
        "index_id": index_id.0,
        "path": req.path,
        "removed_chunks": removed,
    })))
}

/// Query params for `GET /indexes/:id/complexity_hotspots`.
#[derive(Deserialize)]
pub struct HotspotsParams {
    #[serde(default = "default_hotspots_top_n")]
    pub top_n: usize,
}

fn default_hotspots_top_n() -> usize {
    20
}

/// Query params for `GET /indexes/:id/chunks` (issue #54).
#[derive(Deserialize)]
pub struct ChunksParams {
    #[serde(default)]
    pub offset: usize,
    #[serde(default = "default_chunks_limit")]
    pub limit: usize,
}

fn default_chunks_limit() -> usize {
    100
}

/// Hard ceiling on a single `chunks` page so a misconfigured client can't pull
/// the entire corpus into one response. Mirrored in the `list_chunks` MCP tool.
const MAX_CHUNKS_LIMIT: usize = 1_000;

/// `GET /indexes/:id/chunks?offset=&limit=` — paginated enumeration of an index.
///
/// Why: trusty-analyzer (sidecar daemon) and external tooling need to page
/// through every chunk in batches without loading the whole corpus at once.
/// Issue #54 introduces stable-order pagination on top of the existing bulk
/// export.
/// What: Returns
/// `{ index_id, total, offset, limit, chunks: [...] }`. `chunks` is the slice
/// `[offset .. offset+limit]` of the corpus sorted by `(file, start_line)`.
/// `limit` is clamped to `MAX_CHUNKS_LIMIT` (1000); the value echoed back in
/// the response is the post-clamp value so clients can detect the clamp.
/// Test: `test_get_index_chunks_paginates` registers an index, indexes a few
/// files, asserts page1 + page2 cover all chunks without overlap.
async fn get_index_chunks_handler(
    State(state): State<Arc<SearchAppState>>,
    Path(id): Path<String>,
    Query(params): Query<ChunksParams>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let index_id = IndexId::new(id);
    let handle = state.registry.get(&index_id).ok_or(StatusCode::NOT_FOUND)?;
    let limit = params.limit.min(MAX_CHUNKS_LIMIT);
    let indexer = handle.indexer.read().await;
    let (total, chunks) = indexer.enumerate_chunks(params.offset, limit).await;
    Ok(Json(serde_json::json!({
        "index_id": index_id.0,
        "total": total,
        "offset": params.offset,
        "limit": limit,
        "chunks": chunks,
    })))
}

async fn complexity_hotspots_handler(
    State(state): State<Arc<SearchAppState>>,
    Path(id): Path<String>,
    Query(params): Query<HotspotsParams>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let index_id = IndexId::new(id);
    let handle = state.registry.get(&index_id).ok_or(StatusCode::NOT_FOUND)?;
    let indexer = handle.indexer.read().await;
    let mut chunks = indexer.all_chunks().await;
    chunks.sort_by(|a, b| b.complexity.cyclomatic.cmp(&a.complexity.cyclomatic));
    chunks.truncate(params.top_n);
    Ok(Json(serde_json::json!({
        "index_id": index_id.0,
        "top_n": params.top_n,
        "hotspots": chunks,
    })))
}

async fn smells_handler(
    State(state): State<Arc<SearchAppState>>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let index_id = IndexId::new(id);
    let handle = state.registry.get(&index_id).ok_or(StatusCode::NOT_FOUND)?;
    let indexer = handle.indexer.read().await;
    let chunks: Vec<_> = indexer
        .all_chunks()
        .await
        .into_iter()
        .filter(|c| !c.complexity.smells.is_empty())
        .collect();
    Ok(Json(serde_json::json!({
        "index_id": index_id.0,
        "count": chunks.len(),
        "chunks": chunks,
    })))
}

async fn quality_handler(
    State(state): State<Arc<SearchAppState>>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    use trusty_search_core::complexity::ComplexityGrade;
    let index_id = IndexId::new(id);
    let handle = state.registry.get(&index_id).ok_or(StatusCode::NOT_FOUND)?;
    let indexer = handle.indexer.read().await;
    let chunks = indexer.all_chunks().await;
    let chunk_count = chunks.len();
    let (sum_cyclo, grade_a, smell_count) =
        chunks.iter().fold((0u64, 0usize, 0usize), |(s, a, sm), c| {
            let a_inc = if c.complexity.grade == ComplexityGrade::A {
                1
            } else {
                0
            };
            (
                s + c.complexity.cyclomatic as u64,
                a + a_inc,
                sm + c.complexity.smells.len(),
            )
        });
    let avg_cyclomatic = if chunk_count == 0 {
        0.0_f32
    } else {
        sum_cyclo as f32 / chunk_count as f32
    };
    let pct_grade_a = if chunk_count == 0 {
        0.0_f32
    } else {
        grade_a as f32 / chunk_count as f32
    };
    Ok(Json(serde_json::json!({
        "avg_cyclomatic": avg_cyclomatic,
        "pct_grade_a": pct_grade_a,
        "smell_count": smell_count,
        "chunk_count": chunk_count,
    })))
}

/// Optional body for `POST /indexes/:id/reindex`: lets the CLI override the
/// `root_path` stored on the handle (useful when registering + reindexing in
/// one CLI flow).
#[derive(Deserialize, Default)]
pub struct ReindexRequest {
    #[serde(default)]
    pub root_path: Option<std::path::PathBuf>,
    /// When `true`, the daemon clears the per-index content-hash cache before
    /// walking the tree, forcing every file to be re-embedded even if its
    /// content hasn't changed. Set by `trusty-search index --force`.
    #[serde(default)]
    pub force: Option<bool>,
}

async fn reindex_handler(
    State(state): State<Arc<SearchAppState>>,
    Path(id): Path<String>,
    body: Option<Json<ReindexRequest>>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let index_id = IndexId::new(id.clone());
    let mut handle = state.registry.get(&index_id).ok_or(StatusCode::NOT_FOUND)?;

    // If caller supplied a root_path and the stored handle doesn't have one
    // (or differs), re-register with the new path. We can't mutate the
    // existing Arc in place, but registering replaces the entry.
    let mut force = false;
    if let Some(Json(req)) = body {
        force = req.force.unwrap_or(false);
        if let Some(new_root) = req.root_path {
            if handle.root_path.as_os_str().is_empty() || handle.root_path != new_root {
                let indexer = Arc::clone(&handle.indexer);
                let new_handle = IndexHandle {
                    id: index_id.clone(),
                    indexer,
                    root_path: new_root,
                };
                handle = state.registry.register(new_handle);
            }
        }
    }

    // Replace any prior progress entry so SSE subscribers see fresh state.
    let progress = Arc::new(ReindexProgress::new());
    state
        .reindex_progress
        .insert(index_id.clone(), Arc::clone(&progress));

    spawn_reindex(handle, progress, force);

    Ok(Json(serde_json::json!({
        "index_id": index_id.0,
        "queued": true,
        "stream_url": format!("/indexes/{}/reindex/stream", index_id.0),
    })))
}

/// SSE stream of reindex progress events.
///
/// Replays any events already buffered (so a late subscriber still sees the
/// `start` event) and then streams live events from the broadcast channel
/// until the reindex completes.
async fn reindex_stream_handler(
    State(state): State<Arc<SearchAppState>>,
    Path(id): Path<String>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, StatusCode> {
    let index_id = IndexId::new(id);
    let progress = state
        .reindex_progress
        .get(&index_id)
        .map(|r| Arc::clone(r.value()))
        .ok_or(StatusCode::NOT_FOUND)?;

    // Snapshot the replay buffer first so we don't miss the `start` event,
    // then subscribe for live updates. New events that arrive between the
    // snapshot and subscription will appear in both — duplicates are harmless
    // for SSE consumers and rare in practice.
    let replay = progress.events.lock().await.clone();
    let rx = progress.sender.subscribe();
    let live = BroadcastStream::new(rx).filter_map(|r| async move { r.ok() });

    let initial_status = progress.status.load();
    let stream = stream::iter(replay)
        .chain(live)
        .map(|line| Ok(Event::default().data(line)));

    // If the reindex already finished before the subscriber connected, the
    // replay buffer contains the terminal `complete` event and the live
    // stream will idle forever. Trim to just the replay in that case.
    let stream: futures::future::Either<_, _> = if initial_status != ReindexStatus::Running {
        let replay_only = progress.events.lock().await.clone();
        futures::future::Either::Left(
            stream::iter(replay_only).map(|line| Ok(Event::default().data(line))),
        )
    } else {
        futures::future::Either::Right(stream)
    };

    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Why: `/health` is consumed by external probes (open-mpm,
    /// `ensure_daemon_running`) — the contract `{ status, version, indexes,
    /// uptime_secs }` must remain stable.
    /// What: Builds an AppState with N registered indexes and asserts the
    /// HealthResponse JSON shape and counts.
    /// Test: covers issue #34's acceptance (indexes counter + uptime_secs).
    #[tokio::test]
    async fn health_handler_reports_indexes_and_uptime() {
        use std::sync::Arc;
        use tokio::sync::RwLock;
        use trusty_search_core::{
            indexer::CodeIndexer,
            registry::{IndexHandle, IndexId, IndexRegistry},
        };

        let registry = IndexRegistry::new();
        let id = IndexId::new("health-test");
        registry.register(IndexHandle {
            id: id.clone(),
            indexer: Arc::new(RwLock::new(CodeIndexer::new(
                "health-test",
                "/tmp/health-test",
            ))),
            root_path: "/tmp/health-test".into(),
        });
        let state = Arc::new(SearchAppState::new(registry, None));
        let Json(resp) = health_handler(State(state)).await;
        assert_eq!(resp.status, "ok");
        assert_eq!(resp.version, env!("CARGO_PKG_VERSION"));
        assert_eq!(resp.indexes, 1);
        // uptime_secs is u64 — always >= 0 by type; just exercise the path.
        let _ = resp.uptime_secs;
    }

    /// Issue #10 — `POST /search` fan-out: with two registered indexes each
    /// holding a single file, the global search must return results tagged
    /// with the correct `index_id` and the response must list both indexes
    /// as searched. BM25-only path (no embedder) keeps the test hermetic.
    #[tokio::test]
    async fn global_search_fans_out_and_merges() {
        use std::sync::Arc;
        use tokio::sync::RwLock;
        use trusty_search_core::{
            indexer::CodeIndexer,
            registry::{IndexHandle, IndexId, IndexRegistry},
        };

        let registry = IndexRegistry::new();
        for name in ["proj-a", "proj-b"] {
            let id = IndexId::new(name);
            let indexer = CodeIndexer::new(name, format!("/tmp/{name}"));
            // Seed one file per index with content matching the query "alpha".
            indexer
                .index_file(
                    &format!("{name}/lib.rs"),
                    &format!("fn alpha_{name}() {{ println!(\"alpha hit\"); }}"),
                )
                .await
                .expect("index_file ok");
            registry.register(IndexHandle {
                id: id.clone(),
                indexer: Arc::new(RwLock::new(indexer)),
                root_path: format!("/tmp/{name}").into(),
            });
        }

        let state = Arc::new(SearchAppState::new(registry, None));
        let Json(value) = global_search_handler(
            State(state),
            Json(GlobalSearchRequest {
                query: "alpha".into(),
                top_k: 10,
                full_content: false,
            }),
        )
        .await
        .expect("handler ok");

        let total = value["total_indexes"].as_u64().expect("total_indexes");
        assert_eq!(total, 2, "both indexes counted");

        let searched: Vec<String> = value["indexes_searched"]
            .as_array()
            .expect("indexes_searched array")
            .iter()
            .filter_map(|v| v.as_str().map(str::to_owned))
            .collect();
        assert_eq!(searched.len(), 2);
        assert!(searched.contains(&"proj-a".to_string()));
        assert!(searched.contains(&"proj-b".to_string()));

        let results = value["results"].as_array().expect("results array");
        assert!(!results.is_empty(), "expected at least one hit");
        // Every result must carry an index_id tagged with one of the two
        // registered indexes.
        let mut from_a = false;
        let mut from_b = false;
        for r in results {
            let idx = r["index_id"]
                .as_str()
                .expect("each result must be tagged with index_id");
            assert!(
                idx == "proj-a" || idx == "proj-b",
                "unexpected index_id: {idx}"
            );
            from_a |= idx == "proj-a";
            from_b |= idx == "proj-b";
        }
        // Both indexes share the same query term "alpha", so RRF should
        // surface at least one hit from each.
        assert!(from_a, "expected a result tagged with proj-a");
        assert!(from_b, "expected a result tagged with proj-b");
    }

    /// Issue #10 — `POST /search` with no indexes registered must return an
    /// empty result set (not 500). This guards the empty-registry edge case
    /// the fan-out path checks before spawning per-index futures.
    #[tokio::test]
    async fn global_search_empty_registry_returns_empty_results() {
        use trusty_search_core::registry::IndexRegistry;
        let state = Arc::new(SearchAppState::new(IndexRegistry::new(), None));
        let Json(value) = global_search_handler(
            State(state),
            Json(GlobalSearchRequest {
                query: "anything".into(),
                top_k: 5,
                full_content: false,
            }),
        )
        .await
        .expect("handler ok");
        assert_eq!(value["total_indexes"].as_u64(), Some(0));
        assert!(value["results"].as_array().unwrap().is_empty());
        assert!(value["indexes_searched"].as_array().unwrap().is_empty());
    }
}
