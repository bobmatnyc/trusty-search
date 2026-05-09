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
    extract::{Path, State},
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use trusty_search_core::{
    classifier::QueryClassifier,
    indexer::{CodeIndexer, SearchQuery},
    registry::{IndexHandle, IndexId, IndexRegistry},
};

/// Shared state injected into every axum handler.
#[derive(Clone)]
pub struct SearchAppState {
    pub registry: IndexRegistry,
}

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
    version: &'static str,
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
    Router::new()
        .route("/health", get(health_handler))
        .route("/indexes", get(list_indexes_handler).post(create_index_handler))
        .route("/indexes/:id/search", post(search_handler))
        .route("/indexes/:id/status", get(index_status_handler))
        .route("/indexes/:id/index-file", post(index_file_handler))
        .route("/indexes/:id/remove-file", post(remove_file_handler))
        .route("/indexes/:id/reindex", post(reindex_handler))
        .with_state(Arc::new(state))
}

async fn health_handler() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        version: env!("CARGO_PKG_VERSION"),
    })
}

async fn list_indexes_handler(
    State(state): State<Arc<SearchAppState>>,
) -> Json<IndexListResponse> {
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
    let indexer = CodeIndexer::new(req.id.clone(), req.root_path.clone());
    let handle = IndexHandle {
        id: id.clone(),
        indexer: Arc::new(tokio::sync::RwLock::new(indexer)),
        root_path: req.root_path,
    };
    state.registry.register(handle);
    Ok(Json(serde_json::json!({ "id": req.id, "created": true })))
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

async fn reindex_handler(
    State(state): State<Arc<SearchAppState>>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let index_id = IndexId::new(id);
    let _handle = state.registry.get(&index_id).ok_or(StatusCode::NOT_FOUND)?;
    // Fire-and-forget: a real reindex would walk root_path and re-feed every
    // file. For now we acknowledge — issue #3 owns the walker.
    Ok(Json(serde_json::json!({
        "index_id": index_id.0,
        "queued": true,
    })))
}
