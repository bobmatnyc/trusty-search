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

use crate::core::{
    classifier::QueryClassifier,
    embed::Embedder,
    indexer::SearchQuery,
    registry::{IndexHandle, IndexId, IndexRegistry},
};
use axum::{
    body::Body,
    extract::{Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Json, Redirect, Response},
    routing::{delete, get, post},
    Router,
};
use dashmap::DashMap;
use futures::stream::{self, StreamExt};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;
use tokio::sync::{broadcast, watch, OnceCell, RwLock};
use tokio_stream::wrappers::BroadcastStream;
use trusty_common::{ChatProvider, LocalModelConfig};

use crate::service::reindex::{spawn_reindex_with_cleanup, ReindexProgress, ReindexStatus};

/// Live daemon events pushed to dashboard subscribers via the `/status/stream`
/// SSE feed.
///
/// Why: Mirrors the trusty-memory broadcast-channel pattern — a single tagged
/// enum fanned out to every connected browser tab so the UI updates without
/// per-tab polling.
/// What: Tagged-enum (snake_case) serialised as `{"type": "status_changed",
/// ...fields}`. Only `StatusChanged` exists today; new variants (e.g.
/// `IndexCreated`, `ReindexCompleted`) plug in here without touching the
/// handler.
/// Test: subscribe to `/status/stream`, wait > 2s, parse a `status_changed`
/// frame and assert the four fields are present.
#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DaemonEvent {
    StatusChanged {
        indexes: u64,
        total_chunks: u64,
        uptime_secs: u64,
        version: String,
    },
    /// Emitted by `POST /indexes` when a brand-new index is registered.
    ///
    /// Why: The dashboard's "Recent indexes" table is populated by a one-shot
    /// `GET /indexes` fan-out at mount time; without a push event a user
    /// running `trusty-search index <path>` would have to refresh the page to
    /// see the new index. Emitting a tagged event lets the SPA call
    /// `refreshIndexes()` immediately.
    /// What: `{"type":"index_registered","id":"<index-id>"}`.
    /// Test: subscribe to `/status/stream`, `POST /indexes`, assert an
    /// `index_registered` frame with the matching id arrives.
    IndexRegistered { id: String },
    /// Emitted by `DELETE /indexes/:id` when an index is actually evicted.
    ///
    /// Why: Same rationale as `IndexRegistered` — keep dashboards reactive
    /// without page refreshes.
    /// What: `{"type":"index_removed","id":"<index-id>"}`.
    /// Test: register → delete, subscribe before delete, assert an
    /// `index_removed` frame arrives.
    IndexRemoved { id: String },
}

/// Shared state injected into every axum handler.
#[derive(Clone)]
pub struct SearchAppState {
    pub registry: IndexRegistry,
    /// Per-index reindex progress (live counters + SSE replay buffer). Started
    /// by `POST /indexes/:id/reindex`, consumed by
    /// `GET /indexes/:id/reindex/stream`. Lazily populated.
    pub reindex_progress: Arc<DashMap<IndexId, Arc<ReindexProgress>>>,
    /// Issue #120: per-index timestamp of the most recent reindex that
    /// aborted at the memory limit. Used by `reindex_handler` to apply a
    /// cooldown (`TRUSTY_REINDEX_COOLDOWN_SECS`, default 300 s) before
    /// honouring another reindex request — re-running immediately would
    /// hit the same limit and produce a tight loop.
    ///
    /// Why: when a reindex aborts at the memory limit, some files have no
    /// content-hash entry yet, so a follow-up reindex sees them as "new"
    /// and re-processes them — hitting the limit again. The cooldown gives
    /// operators time to lower batch size / raise the limit before another
    /// attempt.
    /// What: written by `spawn_reindex_with_cleanup` when `mem_limit_hit`
    /// is true; read by `reindex_handler` before queuing.
    /// Test: covered by `reindex_handler_rejects_within_cooldown` in
    /// `src/service/server.rs#tests`.
    pub last_reindex_aborted_at: Arc<DashMap<IndexId, std::time::Instant>>,
    /// Process-wide embedder shared across every index so the (expensive)
    /// fastembed ONNX session is initialized once. `None` keeps the daemon
    /// in BM25-only mode — useful for tests that don't want to download the
    /// model. The vector dimensionality is read from the embedder.
    pub embedder: Option<Arc<dyn Embedder>>,
    /// Mutable embedder slot used by the deferred-init flow: the daemon binds
    /// its HTTP port immediately, then a background task loads the fastembed
    /// model and writes it here before flipping `embedder_ready` to `true`.
    ///
    /// Why: ONNX/CoreML model loading takes 15–30 s on first run, but the
    /// outer `Option<Arc<dyn Embedder>>` is captured by reference in many
    /// places. A separate `Arc<RwLock<…>>` lets the init task replace the
    /// value once without rewriting handler signatures.
    /// Test: start daemon; `/health` returns `embedder: "initializing"` for a
    /// few seconds, then flips to `"ready"`.
    pub embedder_slot: Arc<RwLock<Option<Arc<dyn Embedder>>>>,
    /// Watch channel signalling embedder readiness. Handlers that need the
    /// embedder (search, create_index in hybrid mode, index-file) check
    /// `*embedder_ready.borrow()` and return `503 Service Unavailable` until
    /// the value flips to `true`.
    ///
    /// Why: lets `trusty-search index` and `trusty-search start` connect to
    /// the daemon within ~1 s instead of waiting 15–30 s for the embedder to
    /// finish loading. Callers can poll `/health` (cheap) or just hit the
    /// real endpoint and retry on 503.
    /// Test: start daemon; `POST /indexes` immediately returns 503 with
    /// `{"error":"embedder initializing"}`; after a few seconds the same call
    /// succeeds.
    pub embedder_ready: watch::Receiver<bool>,
    /// Sender half of the readiness watch, held by the AppState so the
    /// background embedder-init task can flip readiness from `false` to
    /// `true` once `FastEmbedder::new()` completes.
    ///
    /// Why: kept inside the state (rather than handed off as a free variable)
    /// so test code constructing a fresh `SearchAppState` doesn't have to
    /// thread a sender through every helper. The Arc lets `start.rs` clone
    /// it into the background task.
    pub embedder_ready_tx: Arc<watch::Sender<bool>>,
    /// Last error message captured by the embedder background-init task, or
    /// `None` when init is still in flight or succeeded.
    ///
    /// Why (issue #121): on Intel Xeon AVX-512 hosts, `ort-2.0.0-rc.12`'s
    /// CPU session init can block forever — the daemon stays alive but every
    /// `POST /indexes` hangs (or returns "initializing" indefinitely). With
    /// no visible error, operators waste hours debugging. Surfacing the
    /// init-task error here lets `/health` report `embedder: "error"` with a
    /// human-readable message and lets `POST /indexes` fail fast with a 503
    /// instead of dangling forever.
    /// What: an `Arc<RwLock<Option<String>>>` set by `install_embedder_error`
    /// when `build_embedder()` returns `Err`, or when the init task times out.
    /// Test: `health_reports_embedder_error_when_init_fails` verifies the
    /// `/health` response includes `embedder: "error"` and an `embedder_error`
    /// string after the init task sets an error.
    pub embedder_error: Arc<RwLock<Option<String>>>,
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
    /// Local-model (Ollama / LM Studio / llama.cpp server) configuration loaded
    /// from `~/.trusty-search/config.toml`. Drives `auto_detect_local_provider`
    /// and the `/api/chat/providers` payload.
    pub local_model: LocalModelConfig,
    /// OpenRouter model id (loaded from config; default
    /// `anthropic/claude-haiku-4.5`). Used by the OpenRouter fallback provider.
    pub openrouter_model: String,
    /// OpenRouter API key resolved at startup. May be empty when the user
    /// only configured a local model; the chat handler returns 503 in that case.
    pub openrouter_api_key: String,
    /// Lazily-initialised active chat provider. Auto-detection happens on the
    /// first chat call and the result is cached for the daemon's lifetime.
    pub chat_provider: Arc<OnceCell<Option<Arc<dyn ChatProvider>>>>,
    /// Broadcast sender for live `DaemonEvent` pushes to SSE subscribers.
    ///
    /// Why: Lets the periodic status-ticker (and any future mutating handler)
    /// emit events that every connected dashboard receives instantly. Mirrors
    /// the trusty-memory pattern: cap of 128 buffers transient slow readers;
    /// if a receiver lags it gets `RecvError::Lagged` and we emit a `lag` frame.
    /// What: A `tokio::sync::broadcast::Sender<DaemonEvent>` wrapped in `Arc`
    /// so it's cheap to clone across the AppState.
    /// Test: `emit_propagates_to_subscriber` verifies a subscriber observes
    /// the emitted event.
    pub events: Arc<broadcast::Sender<DaemonEvent>>,
}

impl SearchAppState {
    /// Convenience constructor for callers (`daemon`, tests) that want default
    /// reindex tracking without hand-rolling the `Arc<DashMap<…>>`. Defaults
    /// to BM25-only mode (no embedder); use [`Self::with_embedder`] to enable
    /// the vector lane.
    pub fn new(registry: IndexRegistry) -> Self {
        let openrouter_api_key = std::env::var("OPENROUTER_API_KEY").unwrap_or_default();
        let (events_tx, _) = broadcast::channel::<DaemonEvent>(128);
        // Default-constructed state has no embedder and the readiness watch
        // stays at `false`. Tests exercising BM25-only paths use this default.
        // Production daemon boot overrides via `with_embedder_ready_channel`
        // so the background init task can flip readiness once the model loads.
        let (ready_tx, ready_rx) = watch::channel(false);
        Self {
            registry,
            reindex_progress: Arc::new(DashMap::new()),
            last_reindex_aborted_at: Arc::new(DashMap::new()),
            embedder: None,
            embedder_slot: Arc::new(RwLock::new(None)),
            embedder_ready: ready_rx,
            embedder_ready_tx: Arc::new(ready_tx),
            embedder_error: Arc::new(RwLock::new(None)),
            daemon_port: None,
            openrouter_enabled: !openrouter_api_key.is_empty(),
            started_at: Instant::now(),
            local_model: LocalModelConfig::default(),
            openrouter_model: "anthropic/claude-haiku-4.5".to_string(),
            openrouter_api_key,
            chat_provider: Arc::new(OnceCell::new()),
            events: Arc::new(events_tx),
        }
    }

    /// Send a `DaemonEvent` to all connected SSE subscribers.
    ///
    /// Why: Best-effort fan-out — `broadcast::Sender::send` only fails when
    /// there are no live receivers, which is fine (no listeners == no work).
    /// What: Drops the result, callers don't need to check anything.
    /// Test: `emit_propagates_to_subscriber` subscribes then emits and asserts
    /// the event arrives.
    pub fn emit(&self, event: DaemonEvent) {
        let _ = self.events.send(event);
    }

    /// Builder-style: install user-loaded `local_model` settings (e.g. from
    /// `~/.trusty-search/config.toml`). Replaces the default Ollama address.
    pub fn with_local_model(mut self, cfg: LocalModelConfig) -> Self {
        self.local_model = cfg;
        self
    }

    /// Builder-style: override the OpenRouter model id (defaults to
    /// `anthropic/claude-haiku-4.5`).
    pub fn with_openrouter_model(mut self, model: impl Into<String>) -> Self {
        self.openrouter_model = model.into();
        self
    }

    /// Builder-style: set the OpenRouter API key (loaded from config or env).
    pub fn with_openrouter_api_key(mut self, api_key: impl Into<String>) -> Self {
        let api_key_str = api_key.into();
        self.openrouter_enabled = !api_key_str.is_empty();
        self.openrouter_api_key = api_key_str;
        self
    }

    /// Resolve the active chat provider, auto-detecting on first call.
    ///
    /// Why: Provider selection depends on (a) filesystem-loaded config and (b)
    /// a network probe to a local Ollama / LM Studio instance, so it must be
    /// lazily initialised at runtime. Caching the choice in a `OnceCell` keeps
    /// it stable across concurrent chat requests without re-probing.
    /// What: On first use prefers an auto-detected local server when
    /// `local_model.enabled`, otherwise falls back to OpenRouter when an API
    /// key is configured. Returns `None` when neither is available so the
    /// caller can emit a 503.
    /// Test: Covered by `chat_provider_endpoint_returns_payload` in this crate.
    pub async fn chat_provider(&self) -> Option<Arc<dyn ChatProvider>> {
        self.chat_provider
            .get_or_init(|| async {
                if self.local_model.enabled {
                    if let Some(mut p) =
                        trusty_common::auto_detect_local_provider(&self.local_model.base_url).await
                    {
                        p.model = self.local_model.model.clone();
                        return Some(Arc::new(p) as Arc<dyn ChatProvider>);
                    }
                }
                if !self.openrouter_api_key.is_empty() {
                    return Some(Arc::new(trusty_common::OpenRouterProvider::new(
                        self.openrouter_api_key.clone(),
                        self.openrouter_model.clone(),
                    )) as Arc<dyn ChatProvider>);
                }
                None
            })
            .await
            .clone()
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
        self.embedder = Some(Arc::clone(&embedder));
        // Tests that wire a pre-built embedder expect the daemon to behave as
        // if init has already completed. Mirror that to the slot + watch so
        // handlers using the deferred-init path see a ready embedder too.
        if let Ok(mut slot) = self.embedder_slot.try_write() {
            *slot = Some(embedder);
        }
        let _ = self.embedder_ready_tx.send(true);
        self
    }

    /// Install the embedder produced by the background init task and flip the
    /// readiness watch to `true`.
    ///
    /// Why: the daemon starts serving HTTP before the embedder is loaded so
    /// readiness probes from `trusty-search start` / `trusty-search index`
    /// don't time out waiting for ONNX model load (15–30 s on first run).
    /// The init task calls this when `FastEmbedder::new()` completes; any
    /// in-flight handler observes the readiness flip via the watch channel.
    /// What: writes the embedder into `embedder_slot` and broadcasts `true`
    /// on `embedder_ready_tx` so all `*embedder_ready.borrow()` callers
    /// transition out of the "initializing" branch.
    /// Test: spawn a task that calls this after 1 s; assert `embedder_ready`
    /// flips and subsequent `POST /indexes` calls succeed.
    pub async fn install_embedder(&self, embedder: Arc<dyn Embedder>) {
        let mut slot = self.embedder_slot.write().await;
        *slot = Some(embedder);
        drop(slot);
        // Clear any previously recorded init error — the embedder is now ready.
        {
            let mut err = self.embedder_error.write().await;
            *err = None;
        }
        let _ = self.embedder_ready_tx.send(true);
    }

    /// Record a fatal embedder-init error so `/health` can surface it and
    /// `POST /indexes` can fail fast with a useful message instead of hanging
    /// on "initializing" forever.
    ///
    /// Why (issue #121): the background init task may abort because (a) ORT
    /// session init returned `Err` or (b) the init-timeout fired. In either
    /// case the embedder slot stays empty AND `embedder_ready` stays `false`;
    /// previously this was indistinguishable from "still loading", so callers
    /// retried forever. Capturing the error lets handlers and `/health`
    /// distinguish "transient" from "broken".
    /// What: writes the error message into `embedder_error`. Does NOT flip
    /// `embedder_ready` — `is_embedder_ready()` still returns `false`, so
    /// hybrid-pipeline code paths keep returning 503 rather than producing a
    /// BM25-only index by accident.
    /// Test: `install_embedder_error_surfaces_in_health` verifies the message
    /// is visible via `/health`.
    pub async fn install_embedder_error(&self, message: impl Into<String>) {
        let msg = message.into();
        tracing::error!("embedder init failed: {msg}");
        let mut err = self.embedder_error.write().await;
        *err = Some(msg);
    }

    /// Snapshot the current embedder-init error, if any. `None` means the
    /// background init task is still running or completed successfully.
    pub fn current_embedder_error(&self) -> Option<String> {
        self.embedder_error.try_read().ok().and_then(|g| g.clone())
    }

    /// Snapshot the currently-installed embedder (post-init) or `None` when
    /// the daemon is still warming up. Handlers prefer this over
    /// `self.embedder` so the deferred-init flow works transparently.
    pub async fn current_embedder(&self) -> Option<Arc<dyn Embedder>> {
        let slot = self.embedder_slot.read().await;
        slot.clone()
    }

    /// Cheap, non-blocking readiness check. Returns `true` once the
    /// background embedder-init task has flipped the watch channel.
    pub fn is_embedder_ready(&self) -> bool {
        *self.embedder_ready.borrow()
    }
}

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
    version: &'static str,
    indexes: usize,
    uptime_secs: u64,
    /// Why: operators previously had no way to tell whether the daemon
    /// loaded the embedding model. Silent BM25-only fallback wasted hours
    /// of debugging on "17k files indexed in 12 seconds" symptoms. Now
    /// `/health` reports `"ready"` when an embedder is attached and
    /// `"unavailable"` when the daemon is running BM25-only.
    /// What: `"ready"` if `state.embedder.is_some()` else `"unavailable"`.
    /// Test: start daemon, GET /health, assert `embedder == "ready"`.
    embedder: &'static str,
    /// Human-readable error message captured from a failed embedder-init task,
    /// surfaced alongside `embedder == "error"` (issue #121). `None` when the
    /// embedder is healthy or still warming up. Omitted from JSON when absent.
    #[serde(skip_serializing_if = "Option::is_none")]
    embedder_error: Option<String>,
}

#[derive(Serialize)]
struct IndexListResponse {
    indexes: Vec<String>,
}

#[derive(Deserialize)]
pub struct CreateIndexRequest {
    pub id: String,
    pub root_path: std::path::PathBuf,
    /// Subtrees (relative to `root_path`) to restrict indexing to. Forwarded
    /// from `trusty-search.yaml`'s `paths:` field by `trusty-search index`.
    /// Empty / missing = walk the entire `root_path`.
    #[serde(default)]
    pub include_paths: Option<Vec<String>>,
    /// Glob patterns to exclude on top of the built-in ignores.
    #[serde(default)]
    pub exclude_globs: Option<Vec<String>>,
    /// Extension allow-list (e.g. `["rs", "py"]`, without leading dot).
    #[serde(default)]
    pub extensions: Option<Vec<String>>,
    /// Domain vocabulary for the per-index intent classifier.
    #[serde(default)]
    pub domain_terms: Option<Vec<String>>,
    /// Glob patterns (issue #111) matched against the immediate subdirectory
    /// name under `root_path`. When non-empty, only files inside subdirectories
    /// whose basename matches at least one pattern are indexed. Supports `*`
    /// wildcards (no `**`). Distinct from `include_paths` (absolute subtrees
    /// from `trusty-search.yaml`) — `path_filter` is the API-level glob filter
    /// intended for filtering polyrepo monorepos by repo-name pattern.
    #[serde(default)]
    pub path_filter: Option<Vec<String>>,
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
    use crate::service::ui::{
        chat_handler, list_chat_providers, ui_asset_handler, ui_index_handler,
    };
    // Why: Vite builds the UI bundle with `base: './'` so `index.html` references
    // assets via relative paths (e.g. `./assets/index-XXX.js`). When the browser
    // loads the page at `/ui` (no trailing slash) it resolves those relative
    // URLs against `/`, requesting `/assets/...` which 404s. Redirecting
    // `/ui` → `/ui/` forces the browser to use `/ui/` as the base so asset
    // requests land on `/ui/assets/...` and hit `ui_asset_handler`. The root
    // `/` redirect makes the daemon's landing page friendly (mirrors the
    // `.fallback(static_handler)` shape trusty-memory uses to serve its SPA
    // at `/`).
    let state_arc = Arc::new(state);
    spawn_status_ticker(Arc::clone(&state_arc));
    let router = Router::new()
        .route("/", get(|| async { Redirect::permanent("/ui/") }))
        .route("/health", get(health_handler))
        .route("/status/stream", get(status_stream_handler))
        .route(
            "/indexes",
            get(list_indexes_handler).post(create_index_handler),
        )
        .route("/indexes/{id}", delete(delete_index_handler))
        .route("/ui", get(|| async { Redirect::permanent("/ui/") }))
        .route("/ui/", get(ui_index_handler))
        .route("/ui/{*path}", get(ui_asset_handler))
        .route("/chat", post(chat_handler))
        .route("/api/chat/providers", get(list_chat_providers))
        .route("/search", post(global_search_handler))
        .route("/indexes/{id}/search", post(search_handler))
        .route("/indexes/{id}/search_similar", post(search_similar_handler))
        .route("/indexes/{id}/status", get(index_status_handler))
        .route("/indexes/{id}/index-file", post(index_file_handler))
        .route("/indexes/{id}/remove-file", post(remove_file_handler))
        .route("/indexes/{id}/reindex", post(reindex_handler))
        .route("/indexes/{id}/reindex/stream", get(reindex_stream_handler))
        .route("/indexes/{id}/chunks", get(get_index_chunks_handler))
        .with_state(Arc::clone(&state_arc));
    // Standard middleware stack (CORS, tracing, gzip) lives in trusty-common
    // so every trusty-* daemon ships with the same defaults.
    trusty_common::server::with_standard_middleware(router)
}

/// Spawn a background ticker that emits `StatusChanged` every 2 seconds.
///
/// Why: trusty-memory's pattern is push-driven via mutating handlers, but
/// trusty-search's headline stats (chunk count) change continuously during
/// reindex without a discrete event. A 2s ticker keeps the dashboard's
/// stat cards live (same cadence as the previous poll-based implementation)
/// while still routing through the broadcast channel so the SSE handler
/// stays purely subscription-driven.
/// What: Spawns a detached tokio task holding a `Weak<SearchAppState>` so
/// the ticker terminates automatically when the daemon shuts down (drops the
/// last `Arc`). Each tick recomputes counts and emits one event.
/// Test: subscribe to `/status/stream`, wait > 2s, observe a `status_changed`
/// frame.
fn spawn_status_ticker(state: Arc<SearchAppState>) {
    let weak = Arc::downgrade(&state);
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(2));
        // Skip the immediate first tick — subscribers get an explicit
        // `connected` frame, and a snapshot follows on the next tick.
        interval.tick().await;
        loop {
            interval.tick().await;
            let Some(state) = weak.upgrade() else {
                break;
            };
            let (indexes, total_chunks) = collect_status_counts(&state).await;
            state.emit(DaemonEvent::StatusChanged {
                indexes: indexes as u64,
                total_chunks: total_chunks as u64,
                uptime_secs: state.started_at.elapsed().as_secs(),
                version: env!("CARGO_PKG_VERSION").to_string(),
            });
        }
    });
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
    let embedder_error = state.current_embedder_error();
    let embedder_status = if state.is_embedder_ready() {
        "ready"
    } else if state.embedder.is_some()
        || state
            .embedder_slot
            .try_read()
            .map(|g| g.is_some())
            .unwrap_or(false)
    {
        // Slot populated but readiness flag not yet flipped — treat as ready.
        "ready"
    } else if embedder_error.is_some() {
        // Init task failed or timed out (issue #121). Callers must not retry
        // forever — report a terminal error state so operators can intervene.
        "error"
    } else {
        // Daemon is up but embedder still loading. Callers should retry
        // mutating endpoints; `/health` itself always returns 200 so
        // `trusty-search start`'s readiness probe succeeds quickly.
        "initializing"
    };
    Json(HealthResponse {
        status: "ok",
        version: env!("CARGO_PKG_VERSION"),
        indexes: state.registry.list().len(),
        uptime_secs: state.started_at.elapsed().as_secs(),
        embedder: embedder_status,
        embedder_error,
    })
}

/// Snapshot used by both `/health` (one-shot) and `/status/stream` (SSE tick).
///
/// Why: The dashboard needs live counts of registered indexes + total chunks
/// across the whole daemon. Computing this requires acquiring a read-lock on
/// every indexer, so the work is centralised here to keep the SSE loop tidy.
/// What: Returns `(indexes_count, total_chunks)` summed across the registry.
/// Test: Register two indexes seeded with one file each; the helper returns
/// `(2, chunks_in_file_a + chunks_in_file_b)`.
async fn collect_status_counts(state: &SearchAppState) -> (usize, usize) {
    let ids = state.registry.list();
    let indexes_count = ids.len();
    let mut total_chunks: usize = 0;
    for id in ids {
        if let Some(handle) = state.registry.get(&id) {
            let indexer = handle.indexer.read().await;
            total_chunks = total_chunks.saturating_add(indexer.chunk_count());
        }
    }
    (indexes_count, total_chunks)
}

/// `GET /status/stream` — Server-Sent Events stream of live daemon stats.
///
/// Why: The admin dashboard's headline stat cards (Indexes, Documents,
/// Uptime, Version) should update without a manual refresh. Mirrors the
/// trusty-memory `/sse` pattern — subscribers receive `DaemonEvent` frames
/// pushed via the shared `broadcast::Sender` on `SearchAppState`.
/// What: Subscribes to `state.events`, emits an initial `{"type":"connected"}`
/// frame, then forwards every `DaemonEvent` as `data: <json>\n\n`. Lagged
/// subscribers receive a `{"type":"lag","skipped":N}` frame. The 2s status
/// cadence is supplied by the background ticker spawned in `build_router`.
/// Test: `curl -N http://127.0.0.1:7878/status/stream` shows a `connected`
/// frame immediately and a `status_changed` frame every ~2s.
async fn status_stream_handler(State(state): State<Arc<SearchAppState>>) -> impl IntoResponse {
    let rx = state.events.subscribe();
    let initial = stream::once(async {
        Ok::<axum::body::Bytes, std::io::Error>(axum::body::Bytes::from(
            "data: {\"type\":\"connected\"}\n\n",
        ))
    });
    let events = BroadcastStream::new(rx).map(|res| {
        let frame = match res {
            Ok(event) => match serde_json::to_string(&event) {
                Ok(json) => format!("data: {json}\n\n"),
                Err(e) => format!("data: {{\"type\":\"error\",\"message\":\"{e}\"}}\n\n"),
            },
            Err(tokio_stream::wrappers::errors::BroadcastStreamRecvError::Lagged(n)) => {
                format!("data: {{\"type\":\"lag\",\"skipped\":{n}}}\n\n")
            }
        };
        Ok::<axum::body::Bytes, std::io::Error>(axum::body::Bytes::from(frame))
    });
    let stream = initial.chain(events);

    Response::builder()
        .header("Content-Type", "text/event-stream")
        .header("Cache-Control", "no-cache")
        .header("X-Accel-Buffering", "no")
        .body(Body::from_stream(stream))
        .expect("valid SSE response")
}

async fn list_indexes_handler(State(state): State<Arc<SearchAppState>>) -> Json<IndexListResponse> {
    Json(IndexListResponse {
        indexes: state.registry.list().into_iter().map(|id| id.0).collect(),
    })
}

async fn create_index_handler(
    State(state): State<Arc<SearchAppState>>,
    Json(req): Json<CreateIndexRequest>,
) -> Response {
    let id = IndexId::new(req.id.clone());
    if state.registry.get(&id).is_some() {
        return Json(serde_json::json!({
            "id": req.id,
            "created": false,
            "reason": "already exists",
        }))
        .into_response();
    }
    // Why (issue: 10s readiness timeout): the embedder may still be loading
    // when the daemon accepts its first request. Reject hybrid-index creation
    // with `503 Service Unavailable` so the caller (`trusty-search index`)
    // retries instead of producing a BM25-only index that will quietly miss
    // the vector lane forever.
    let Some(embedder) = state.current_embedder().await else {
        // Issue #121: distinguish "still warming up" from "init failed
        // permanently". When the background task has recorded an error,
        // surface it in the 503 so callers stop polling and operators see
        // a useful message in logs / dashboards.
        if let Some(err) = state.current_embedder_error() {
            return embedder_error_response(&err);
        }
        return embedder_initializing_response();
    };
    // Bug A fix: when an embedder is attached to the shared state, wire the
    // newly created indexer with both an `Embedder` and a `VectorStore` so
    // the HNSW lane actually contributes results. Previously every index
    // was BM25-only because `with_components` was never called, which is
    // why the benchmark observed `match_reason: "bm25"` for 100% of hits.
    //
    // Issue #85: if a previously-saved HNSW snapshot + chunks file exist for
    // this id, restore them so the daemon warm-boots without re-indexing.
    let mut indexer = crate::service::persistence_loader::build_indexer_with_persisted_state(
        &req.id,
        req.root_path.clone(),
        &embedder,
    )
    .await;

    // Resolve repo-config filters (issue: trusty-search.yaml wiring). The
    // CLI sends `paths:` as relative strings; resolve them against `root_path`
    // here so the registry handle carries absolute subtrees ready for the
    // reindex walker. `domain_terms` is attached to the indexer so its
    // `classify_with_domain` lookup runs on every search without needing to
    // reach back into the handle.
    let include_paths: Vec<std::path::PathBuf> = req
        .include_paths
        .clone()
        .unwrap_or_default()
        .into_iter()
        .filter(|p| !p.trim().is_empty() && p.trim() != ".")
        .map(|p| req.root_path.join(p.trim()))
        .collect();
    let exclude_globs: Vec<String> = req.exclude_globs.clone().unwrap_or_default();
    let extensions: Vec<String> = req
        .extensions
        .clone()
        .unwrap_or_default()
        .into_iter()
        .map(|e| e.trim_start_matches('.').to_string())
        .filter(|e| !e.is_empty())
        .collect();
    let domain_terms: Vec<String> = req.domain_terms.clone().unwrap_or_default();
    let path_filter: Vec<String> = req
        .path_filter
        .clone()
        .unwrap_or_default()
        .into_iter()
        .filter(|p| !p.trim().is_empty())
        .collect();
    indexer.set_domain_terms(domain_terms.clone());

    // Persist the registration so a daemon restart can re-register
    // automatically. Best-effort: a write failure is logged but doesn't fail
    // the request — the in-memory registry still has the index.
    if let Err(e) = crate::service::persistence::upsert_index_registry_entry(
        crate::service::persistence::PersistedIndex {
            id: req.id.clone(),
            root_path: req.root_path.clone(),
            include_paths: req.include_paths.clone().unwrap_or_default(),
            exclude_globs: exclude_globs.clone(),
            extensions: extensions.clone(),
            domain_terms: domain_terms.clone(),
            path_filter: path_filter.clone(),
        },
    ) {
        tracing::warn!("could not persist index registry for {}: {e}", req.id);
    }

    let handle = IndexHandle {
        id: id.clone(),
        indexer: Arc::new(tokio::sync::RwLock::new(indexer)),
        root_path: req.root_path,
        include_paths,
        exclude_globs,
        extensions,
        domain_terms,
        path_filter,
        context_embedding: Arc::new(tokio::sync::RwLock::new(None)),
        context_summary: Arc::new(tokio::sync::RwLock::new(None)),
    };
    state.registry.register(handle);
    // Push event so connected dashboards refresh their index list without a
    // page reload (mirrors the trusty-memory `palace_created` pattern).
    state.emit(DaemonEvent::IndexRegistered { id: req.id.clone() });
    Json(serde_json::json!({ "id": req.id, "created": true })).into_response()
}

/// Build a `503 Service Unavailable` response for handlers that require the
/// embedder before the background init task has finished.
///
/// Why: callers (CLI, MCP, integrators) need to distinguish "transient — try
/// again in a few seconds" from real failures. A standard 503 with a typed
/// JSON body lets `trusty-search index` retry, while exposing a clear
/// `embedder initializing` reason for human operators reading logs.
/// What: returns `(503, {"error": "embedder initializing, retry in a few seconds"})`.
/// Test: hit `POST /indexes` immediately after daemon boot; assert 503 and
/// JSON body shape.
fn embedder_initializing_response() -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(serde_json::json!({
            "error": "embedder initializing, retry in a few seconds"
        })),
    )
        .into_response()
}

/// Build a `503 Service Unavailable` response when the embedder background
/// init task has recorded a permanent failure (issue #121).
///
/// Why: previously a hung/failed init left the daemon stuck in
/// `"initializing"` forever, so retry loops in `trusty-search index` and
/// downstream clients spun indefinitely. Returning a typed error body with
/// the recorded message lets callers fail fast and surfaces the root cause
/// (e.g. "init timed out after 60s") in logs and CLI output.
/// What: returns `(503, {"error": "embedder init failed: <message>"})`.
/// Test: `create_index_returns_503_with_error_when_embedder_failed`.
fn embedder_error_response(message: &str) -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(serde_json::json!({
            "error": format!("embedder init failed: {message}"),
        })),
    )
        .into_response()
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
    if removed {
        // Issue #85: drop the on-disk footprint so the index doesn't come
        // back on the next daemon restart. Best-effort — log on failure.
        if let Err(e) = crate::service::persistence::remove_index_registry_entry(&id) {
            tracing::warn!("could not remove '{id}' from indexes.toml: {e}");
        }
        if let Err(e) = crate::service::persistence::remove_index_data_dir(&id) {
            tracing::warn!("could not remove on-disk data for '{id}': {e}");
        }
        // Push event so connected dashboards drop the row without refresh.
        state.emit(DaemonEvent::IndexRemoved { id: id.clone() });
    }
    Json(serde_json::json!({ "id": id, "removed": removed }))
}

async fn search_handler(
    State(state): State<Arc<SearchAppState>>,
    Path(id): Path<String>,
    Json(query): Json<SearchQuery>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let index_id = IndexId::new(id);
    let handle = state.registry.get(&index_id).ok_or(StatusCode::NOT_FOUND)?;
    // Use the same domain-aware classifier as `CodeIndexer::search` so the
    // intent reported back to the caller matches what was used for routing.
    let intent = QueryClassifier::classify_with_domain(&query.text, &handle.domain_terms);
    let started = std::time::Instant::now();
    let indexer = handle.indexer.read().await;
    let results = indexer
        .search(&query)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let latency_ms = started.elapsed().as_millis() as u64;
    tracing::info!(
        index_id = %index_id,
        intent = %format!("{intent:?}"),
        latency_ms = latency_ms,
        results = results.len(),
        query = %&query.text[..query.text.len().min(80)],
        "search"
    );
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
    /// Optional allow-list of index ids to fan out to (issue #110). When
    /// present, only the named indexes are searched; unknown ids are
    /// silently skipped (logged at debug). When absent / empty, the daemon
    /// fans out to every registered index (legacy behaviour).
    #[serde(default)]
    pub indexes: Option<Vec<String>>,

    /// Fan-out routing strategy (issue #112). Controls how the daemon
    /// weights or filters the per-index lanes by cosine similarity between
    /// the query embedding and each index's stored `context_embedding`.
    ///
    /// - `"all"` (default): every index is searched; each index's RRF lane
    ///   is multiplied by its cosine similarity weight (indexes with no
    ///   context embedding use the neutral 1.0).
    /// - `"top_n"`: only the top-N indexes (by cosine similarity) are
    ///   searched; `routing_n` controls N (default 3).
    /// - `"threshold"`: indexes with cosine similarity below
    ///   `routing_threshold` (default 0.3) are skipped.
    #[serde(default)]
    pub routing: Option<String>,
    /// Number of indexes to keep for `routing = "top_n"`. Default 3.
    #[serde(default)]
    pub routing_n: Option<usize>,
    /// Cosine-similarity cutoff for `routing = "threshold"`. Default 0.3.
    #[serde(default)]
    pub routing_threshold: Option<f32>,
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
    use crate::core::search::rrf::{rrf_fuse, RRF_K};

    let all_ids = state.registry.list();
    // Issue #110: when caller supplies `indexes`, restrict fan-out to that
    // set. Unknown ids are dropped here (the per-index branch below would
    // emit a 404; we'd rather silently skip so a stale caller doesn't
    // poison an otherwise-good fan-out).
    let index_ids: Vec<IndexId> = if let Some(requested) = req.indexes.as_ref() {
        let allow: std::collections::HashSet<&str> = requested.iter().map(|s| s.as_str()).collect();
        all_ids
            .into_iter()
            .filter(|id| allow.contains(id.0.as_str()))
            .collect()
    } else {
        all_ids
    };
    let total_indexes = index_ids.len();
    if index_ids.is_empty() {
        return Ok(Json(serde_json::json!({
            "results": Vec::<crate::core::indexer::CodeChunk>::new(),
            "indexes_searched": Vec::<String>::new(),
            "total_indexes": 0_usize,
            "latency_ms": 0_u64,
            "intent": format!("{:?}", QueryClassifier::classify(&req.query)),
        })));
    }

    let started = std::time::Instant::now();
    let intent = QueryClassifier::classify(&req.query);

    // Issue #112: compute per-index context weights, then apply the routing
    // strategy to decide which indexes participate in the fan-out.
    let routing_mode = RoutingMode::from_request(&req);
    let weights = compute_context_weights(&state.registry, &index_ids, &req.query).await;
    let (active_ids, weight_map) = routing_mode.apply(&index_ids, &weights);
    let routing_label = routing_mode.label().to_string();
    let routing_decisions: Vec<serde_json::Value> = index_ids
        .iter()
        .map(|id| {
            let w = weights.get(id).copied().unwrap_or(1.0);
            let included = weight_map.contains_key(id);
            serde_json::json!({
                "index_id": id.0,
                "cosine_similarity": w,
                "included": included,
            })
        })
        .collect();

    // Build the same SearchQuery shape every per-index search uses. We
    // oversample per-index by passing the user's top_k unchanged: each lane
    // contributes up to top_k candidates, then RRF picks the best top_k
    // overall.
    let per_index_query = SearchQuery {
        text: req.query.clone(),
        top_k: req.top_k,
        expand_graph: true,
        compact: !req.full_content,
        branch_files: None,
        branch_boost: SearchQuery::default_branch_boost(),
        branch: None,
    };

    // Run all per-index searches concurrently. Any index that errors is
    // skipped with a log line so a single broken index doesn't 500 the
    // whole fan-out.
    let registry = state.registry.clone();
    let futures = active_ids.into_iter().map(|id| {
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
    let per_index_results: Vec<(IndexId, Vec<crate::core::indexer::CodeChunk>)> =
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
    let mut chunk_lookup: std::collections::HashMap<String, crate::core::indexer::CodeChunk> =
        std::collections::HashMap::new();
    let mut lanes: Vec<Vec<(String, f32)>> = Vec::with_capacity(per_index_results.len());
    let mut indexes_searched: Vec<String> = Vec::with_capacity(per_index_results.len());
    for (id, results) in per_index_results {
        indexes_searched.push(id.0.clone());
        // Issue #112: in `"all"` mode, multiply each lane's scores by the
        // index's cosine-similarity weight; in `"top_n"` / `"threshold"`
        // modes the weight is always 1.0 (selection has already happened).
        let weight = weight_map.get(&id).copied().unwrap_or(1.0);
        let mut lane: Vec<(String, f32)> = Vec::with_capacity(results.len());
        for mut chunk in results {
            let namespaced = format!("{}::{}", id.0, chunk.id);
            // Tag the chunk with its origin index before storing it so the
            // returned CodeChunks know where they came from.
            chunk.index_id = Some(id.0.clone());
            let weighted_score = chunk.score * weight;
            lane.push((namespaced.clone(), weighted_score));
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

    let results: Vec<crate::core::indexer::CodeChunk> = fused
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
        "routing": routing_label,
        "routing_decisions": routing_decisions,
    })))
}

/// Fan-out routing strategy resolved from a `GlobalSearchRequest` (issue #112).
///
/// Why: keeps the per-strategy logic (weight every lane, take top-N, filter
/// below threshold) in one place so the global-search handler stays small.
#[derive(Debug, Clone, Copy)]
enum RoutingMode {
    /// Search every index; multiply each lane's RRF scores by the index's
    /// context cosine similarity (indexes with no context use 1.0).
    All,
    /// Search only the top-N indexes by cosine similarity. Weights are not
    /// applied to lane scores (selection already encodes relevance).
    TopN(usize),
    /// Search only indexes whose cosine similarity ≥ threshold. Weights are
    /// not applied to lane scores.
    Threshold(f32),
}

impl RoutingMode {
    const DEFAULT_TOP_N: usize = 3;
    const DEFAULT_THRESHOLD: f32 = 0.3;

    fn from_request(req: &GlobalSearchRequest) -> Self {
        match req.routing.as_deref() {
            Some("top_n") => Self::TopN(req.routing_n.unwrap_or(Self::DEFAULT_TOP_N).max(1)),
            Some("threshold") => {
                Self::Threshold(req.routing_threshold.unwrap_or(Self::DEFAULT_THRESHOLD))
            }
            // "all" or anything else (or absent) defaults to All.
            _ => Self::All,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::TopN(_) => "top_n",
            Self::Threshold(_) => "threshold",
        }
    }

    /// Filter `index_ids` according to the strategy and return the active
    /// id list plus the per-id weight map the lane builder will consult.
    ///
    /// - `All`: every id is active; weight = its cosine similarity.
    /// - `TopN`: top N by cosine similarity; weight = 1.0 for selected ids.
    /// - `Threshold`: cosine ≥ threshold; weight = 1.0 for selected ids.
    fn apply(
        self,
        index_ids: &[IndexId],
        weights: &std::collections::HashMap<IndexId, f32>,
    ) -> (Vec<IndexId>, std::collections::HashMap<IndexId, f32>) {
        match self {
            Self::All => {
                let active: Vec<IndexId> = index_ids.to_vec();
                let map: std::collections::HashMap<IndexId, f32> = index_ids
                    .iter()
                    .map(|id| (id.clone(), weights.get(id).copied().unwrap_or(1.0)))
                    .collect();
                (active, map)
            }
            Self::TopN(n) => {
                let mut ranked: Vec<(&IndexId, f32)> = index_ids
                    .iter()
                    .map(|id| (id, weights.get(id).copied().unwrap_or(1.0)))
                    .collect();
                ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let active: Vec<IndexId> =
                    ranked.iter().take(n).map(|(id, _)| (*id).clone()).collect();
                let map: std::collections::HashMap<IndexId, f32> =
                    active.iter().map(|id| (id.clone(), 1.0)).collect();
                (active, map)
            }
            Self::Threshold(t) => {
                let active: Vec<IndexId> = index_ids
                    .iter()
                    .filter(|id| weights.get(id).copied().unwrap_or(1.0) >= t)
                    .cloned()
                    .collect();
                let map: std::collections::HashMap<IndexId, f32> =
                    active.iter().map(|id| (id.clone(), 1.0)).collect();
                (active, map)
            }
        }
    }
}

/// Embed the query once and compute cosine similarity against every index's
/// stored `context_embedding` (issue #112).
///
/// Why: the fan-out router needs a single relevance score per index. Indexes
/// without a context embedding (no recognised metadata, embedder unavailable
/// during last reindex) default to a neutral 1.0 so they participate
/// normally — the absence of a fingerprint is not a relevance signal.
/// What: returns a `HashMap<IndexId, f32>` where every id in `index_ids` has
/// an entry; the value is either `cosine_similarity(query, context)` or
/// `1.0` for indexes with no context. Failures embedding the query (e.g.
/// embedder not wired) also fall back to 1.0 across the board so the global
/// search keeps working as a plain fan-out.
async fn compute_context_weights(
    registry: &crate::core::registry::IndexRegistry,
    index_ids: &[IndexId],
    query: &str,
) -> std::collections::HashMap<IndexId, f32> {
    use crate::core::mmr::cosine_similarity;

    // Try to obtain a query embedding from any index that has an embedder
    // wired. Every index in the registry shares the same machine-wide
    // FastEmbedder, so the first successful embed is reused for all.
    let mut query_embedding: Option<Vec<f32>> = None;
    for id in index_ids {
        let Some(handle) = registry.get(id) else {
            continue;
        };
        let indexer = handle.indexer.read().await;
        match indexer.embed_text(query).await {
            Ok(Some(vec)) => {
                query_embedding = Some(vec);
                break;
            }
            Ok(None) => continue,
            Err(e) => {
                tracing::debug!("context_routing: embed_text failed on {}: {e}", id.0);
                continue;
            }
        }
    }

    let mut out = std::collections::HashMap::with_capacity(index_ids.len());
    let Some(q) = query_embedding else {
        // Couldn't embed at all — fall back to neutral weights everywhere.
        for id in index_ids {
            out.insert(id.clone(), 1.0);
        }
        return out;
    };

    for id in index_ids {
        let Some(handle) = registry.get(id) else {
            out.insert(id.clone(), 1.0);
            continue;
        };
        let ctx_guard = handle.context_embedding.read().await;
        let weight = match ctx_guard.as_ref() {
            Some(ctx) if ctx.len() == q.len() => cosine_similarity(&q, ctx).max(0.0),
            _ => 1.0,
        };
        out.insert(id.clone(), weight);
    }
    out
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
    // Issue #111: surface `path_filter` so callers can see which glob filter
    // (if any) is active for the index. Returns `null` when no filter is set.
    let path_filter = if handle.path_filter.is_empty() {
        serde_json::Value::Null
    } else {
        serde_json::Value::Array(
            handle
                .path_filter
                .iter()
                .map(|s| serde_json::Value::String(s.clone()))
                .collect(),
        )
    };
    // Issue #112: surface whether a context embedding has been computed
    // for this index, plus the truncated human-readable summary that
    // produced it. Helps operators verify metadata scraping found a
    // recognised file.
    let has_context_embedding = handle.context_embedding.read().await.is_some();
    let context_summary = handle
        .context_summary
        .read()
        .await
        .clone()
        .map(serde_json::Value::String)
        .unwrap_or(serde_json::Value::Null);
    Ok(Json(serde_json::json!({
        "index_id": index_id.0,
        "root_path": handle.root_path,
        "chunk_count": indexer.chunk_count(),
        "path_filter": path_filter,
        "has_context_embedding": has_context_embedding,
        "context_summary": context_summary,
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
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let index_id = IndexId::new(id.clone());
    let mut handle = state.registry.get(&index_id).ok_or((
        StatusCode::NOT_FOUND,
        Json(serde_json::json!({
            "error": format!("unknown index: {}", index_id.0),
        })),
    ))?;

    // Issue #120: cooldown guard. If the most recent reindex for this index
    // aborted at the memory limit, refuse to queue another one for
    // `TRUSTY_REINDEX_COOLDOWN_SECS` (default 300 s). Re-running immediately
    // would just hit the limit again because the un-processed files have no
    // content-hash entries yet, producing an infinite reindex loop. Operators
    // can lower batch size / raise the memory limit and try again after the
    // cooldown elapses.
    if let Some(aborted_at) = state.last_reindex_aborted_at.get(&index_id) {
        let elapsed = aborted_at.elapsed();
        let cooldown = std::time::Duration::from_secs(
            std::env::var("TRUSTY_REINDEX_COOLDOWN_SECS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(300),
        );
        if elapsed < cooldown {
            let remaining_secs = (cooldown - elapsed).as_secs();
            tracing::warn!(
                "reindex_handler: refusing reindex for index {} — last run \
                 aborted at memory limit {}s ago, cooldown {}s remaining",
                index_id.0,
                elapsed.as_secs(),
                remaining_secs,
            );
            return Err((
                StatusCode::TOO_MANY_REQUESTS,
                Json(serde_json::json!({
                    "error": "reindex cooldown active after memory-limit abort",
                    "index_id": index_id.0,
                    "retry_after_secs": remaining_secs,
                    "cooldown_secs": cooldown.as_secs(),
                    "hint": "lower TRUSTY_MAX_BATCH_SIZE or raise TRUSTY_MEMORY_LIMIT_MB before retrying",
                })),
            ));
        }
        // Cooldown elapsed — drop the stale entry so the next abort (if any)
        // starts a fresh window. Done outside the `get()` guard to avoid
        // holding a DashMap shard lock across the removal.
        drop(aborted_at);
        state.last_reindex_aborted_at.remove(&index_id);
    }

    // If caller supplied a root_path and the stored handle doesn't have one
    // (or differs), re-register with the new path. We can't mutate the
    // existing Arc in place, but registering replaces the entry.
    let mut force = false;
    if let Some(Json(req)) = body {
        force = req.force.unwrap_or(false);
        if let Some(new_root) = req.root_path {
            if handle.root_path.as_os_str().is_empty() || handle.root_path != new_root {
                let indexer = Arc::clone(&handle.indexer);
                // Preserve the filter set / domain vocabulary recorded on the
                // existing handle — only the root_path is being overridden.
                let new_handle = IndexHandle {
                    id: index_id.clone(),
                    indexer,
                    root_path: new_root,
                    include_paths: handle.include_paths.clone(),
                    exclude_globs: handle.exclude_globs.clone(),
                    extensions: handle.extensions.clone(),
                    domain_terms: handle.domain_terms.clone(),
                    path_filter: handle.path_filter.clone(),
                    // Preserve the previously inferred context (if any). A
                    // fresh reindex will overwrite this with the metadata
                    // scraped from the new root.
                    context_embedding: Arc::clone(&handle.context_embedding),
                    context_summary: Arc::clone(&handle.context_summary),
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

    spawn_reindex_with_cleanup(
        handle,
        progress,
        force,
        Some(Arc::clone(&state.reindex_progress)),
        Some(Arc::clone(&state.last_reindex_aborted_at)),
    );

    Ok(Json(serde_json::json!({
        "index_id": index_id.0,
        "queued": true,
        "stream_url": format!("/indexes/{}/reindex/stream", index_id.0),
    })))
}

/// SSE stream of reindex progress events.
///
/// Mirrors the `/status/stream` SSE pattern (manual `Response::builder()`
/// with `text/event-stream` + `no-cache` + `X-Accel-Buffering: no`).
/// Replays any events already buffered (so a late subscriber still sees the
/// `start` event) and then streams live events from the broadcast channel
/// until the reindex completes. Lagged subscribers receive a
/// `{"type":"lag","skipped":N}` frame.
async fn reindex_stream_handler(
    State(state): State<Arc<SearchAppState>>,
    Path(id): Path<String>,
) -> Result<Response, StatusCode> {
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
    let initial_status = progress.status.load();
    let rx = progress.sender.subscribe();

    fn frame(line: String) -> Result<axum::body::Bytes, std::io::Error> {
        Ok(axum::body::Bytes::from(format!("data: {line}\n\n")))
    }

    let replay_stream = stream::iter(replay).map(frame);

    // If the reindex already finished before the subscriber connected, the
    // replay buffer contains the terminal `complete` event and the live
    // stream would idle forever. Return the replay only in that case.
    let body = if initial_status != ReindexStatus::Running {
        Body::from_stream(replay_stream)
    } else {
        let live = BroadcastStream::new(rx).map(|res| match res {
            Ok(line) => frame(line),
            Err(tokio_stream::wrappers::errors::BroadcastStreamRecvError::Lagged(n)) => Ok(
                axum::body::Bytes::from(format!("data: {{\"type\":\"lag\",\"skipped\":{n}}}\n\n")),
            ),
        });
        Body::from_stream(replay_stream.chain(live))
    };

    Ok(Response::builder()
        .header("Content-Type", "text/event-stream")
        .header("Cache-Control", "no-cache")
        .header("X-Accel-Buffering", "no")
        .body(body)
        .expect("valid SSE response"))
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
        use crate::core::{
            indexer::CodeIndexer,
            registry::{IndexHandle, IndexId, IndexRegistry},
        };
        use std::sync::Arc;
        use tokio::sync::RwLock;

        let registry = IndexRegistry::new();
        let id = IndexId::new("health-test");
        registry.register(IndexHandle::bare(
            id.clone(),
            Arc::new(RwLock::new(CodeIndexer::new(
                "health-test",
                "/tmp/health-test",
            ))),
            "/tmp/health-test".into(),
        ));
        let state = Arc::new(SearchAppState::new(registry));
        let Json(resp) = health_handler(State(state)).await;
        assert_eq!(resp.status, "ok");
        assert_eq!(resp.version, env!("CARGO_PKG_VERSION"));
        assert_eq!(resp.indexes, 1);
        // uptime_secs is u64 — always >= 0 by type; just exercise the path.
        let _ = resp.uptime_secs;
        // No embedder attached in this test. With the deferred-init flow,
        // a fresh `SearchAppState::new()` reports "initializing" (the
        // background task hasn't installed an embedder yet) rather than
        // "unavailable". "unavailable" is reserved for the post-failure
        // case where the init task explicitly errored.
        assert_eq!(resp.embedder, "initializing");
    }

    /// Issue #10 — `POST /search` fan-out: with two registered indexes each
    /// holding a single file, the global search must return results tagged
    /// with the correct `index_id` and the response must list both indexes
    /// as searched. BM25-only path (no embedder) keeps the test hermetic.
    #[tokio::test]
    async fn global_search_fans_out_and_merges() {
        use crate::core::{
            indexer::CodeIndexer,
            registry::{IndexHandle, IndexId, IndexRegistry},
        };
        use std::sync::Arc;
        use tokio::sync::RwLock;

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
            registry.register(IndexHandle::bare(
                id.clone(),
                Arc::new(RwLock::new(indexer)),
                format!("/tmp/{name}").into(),
            ));
        }

        let state = Arc::new(SearchAppState::new(registry));
        let Json(value) = global_search_handler(
            State(state),
            Json(GlobalSearchRequest {
                query: "alpha".into(),
                top_k: 10,
                full_content: false,
                indexes: None,
                routing: None,
                routing_n: None,
                routing_threshold: None,
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
        use crate::core::registry::IndexRegistry;
        let state = Arc::new(SearchAppState::new(IndexRegistry::new()));
        let Json(value) = global_search_handler(
            State(state),
            Json(GlobalSearchRequest {
                query: "anything".into(),
                top_k: 5,
                full_content: false,
                indexes: None,
                routing: None,
                routing_n: None,
                routing_threshold: None,
            }),
        )
        .await
        .expect("handler ok");
        assert_eq!(value["total_indexes"].as_u64(), Some(0));
        assert!(value["results"].as_array().unwrap().is_empty());
        assert!(value["indexes_searched"].as_array().unwrap().is_empty());
    }

    /// Issue #110 — `POST /search` with explicit `indexes: [...]` must only
    /// fan out to the named indexes; results from indexes outside the
    /// allow-list must not appear, even when they match the query.
    #[tokio::test]
    async fn global_search_restricts_to_named_indexes() {
        use crate::core::{
            indexer::CodeIndexer,
            registry::{IndexHandle, IndexId, IndexRegistry},
        };
        use std::sync::Arc;
        use tokio::sync::RwLock;

        let registry = IndexRegistry::new();
        for name in ["proj-a", "proj-b", "proj-c"] {
            let id = IndexId::new(name);
            let indexer = CodeIndexer::new(name, format!("/tmp/{name}"));
            indexer
                .index_file(
                    &format!("{name}/lib.rs"),
                    &format!("fn alpha_{name}() {{ println!(\"alpha hit\"); }}"),
                )
                .await
                .expect("index_file ok");
            registry.register(IndexHandle::bare(
                id.clone(),
                Arc::new(RwLock::new(indexer)),
                format!("/tmp/{name}").into(),
            ));
        }
        let state = Arc::new(SearchAppState::new(registry));
        let Json(value) = global_search_handler(
            State(state),
            Json(GlobalSearchRequest {
                query: "alpha".into(),
                top_k: 10,
                full_content: false,
                indexes: Some(vec!["proj-a".into(), "proj-c".into()]),
                routing: None,
                routing_n: None,
                routing_threshold: None,
            }),
        )
        .await
        .expect("handler ok");

        // total_indexes reflects the *filtered* set we actually fanned out to.
        assert_eq!(value["total_indexes"].as_u64(), Some(2));

        let searched: std::collections::HashSet<String> = value["indexes_searched"]
            .as_array()
            .expect("array")
            .iter()
            .filter_map(|v| v.as_str().map(str::to_owned))
            .collect();
        assert!(searched.contains("proj-a"));
        assert!(searched.contains("proj-c"));
        assert!(!searched.contains("proj-b"), "proj-b must be excluded");

        for r in value["results"].as_array().unwrap() {
            let idx = r["index_id"].as_str().unwrap();
            assert_ne!(idx, "proj-b", "no result may come from excluded index");
        }
    }

    /// Issue #112: `RoutingMode::All` keeps every index and surfaces the
    /// cosine-similarity weight verbatim. Indexes without a weight entry
    /// fall back to 1.0.
    #[test]
    fn routing_mode_all_preserves_every_index_with_weights() {
        let ids = vec![IndexId::new("a"), IndexId::new("b"), IndexId::new("c")];
        let weights: std::collections::HashMap<IndexId, f32> = [
            (IndexId::new("a"), 0.9_f32),
            (IndexId::new("b"), 0.2),
            // "c" deliberately absent → falls back to 1.0
        ]
        .into_iter()
        .collect();

        let (active, map) = RoutingMode::All.apply(&ids, &weights);
        assert_eq!(active.len(), 3, "all routing keeps every index");
        assert!((map.get(&IndexId::new("a")).copied().unwrap() - 0.9).abs() < 1e-6);
        assert!((map.get(&IndexId::new("b")).copied().unwrap() - 0.2).abs() < 1e-6);
        assert!((map.get(&IndexId::new("c")).copied().unwrap() - 1.0).abs() < 1e-6);
    }

    /// Issue #112: `RoutingMode::TopN` keeps only the N highest-similarity
    /// indexes (ranked desc) and zeroes weights to 1.0 — selection has
    /// already absorbed relevance.
    #[test]
    fn routing_mode_top_n_keeps_only_highest_similarity() {
        let ids = vec![IndexId::new("low"), IndexId::new("hi"), IndexId::new("mid")];
        let weights: std::collections::HashMap<IndexId, f32> = [
            (IndexId::new("low"), 0.1_f32),
            (IndexId::new("hi"), 0.95),
            (IndexId::new("mid"), 0.5),
        ]
        .into_iter()
        .collect();

        let (active, map) = RoutingMode::TopN(2).apply(&ids, &weights);
        assert_eq!(active.len(), 2);
        let active_set: std::collections::HashSet<&str> =
            active.iter().map(|id| id.0.as_str()).collect();
        assert!(active_set.contains("hi"));
        assert!(active_set.contains("mid"));
        assert!(!active_set.contains("low"));
        // Selected entries normalised to weight 1.0.
        assert!((map.get(&IndexId::new("hi")).copied().unwrap() - 1.0).abs() < 1e-6);
        assert!((map.get(&IndexId::new("mid")).copied().unwrap() - 1.0).abs() < 1e-6);
        assert!(!map.contains_key(&IndexId::new("low")));
    }

    /// Issue #112: `RoutingMode::Threshold` drops anything strictly below
    /// the threshold and keeps entries at/above it.
    #[test]
    fn routing_mode_threshold_drops_below_cutoff() {
        let ids = vec![IndexId::new("a"), IndexId::new("b"), IndexId::new("c")];
        let weights: std::collections::HashMap<IndexId, f32> = [
            (IndexId::new("a"), 0.1_f32),
            (IndexId::new("b"), 0.5),
            (IndexId::new("c"), 0.8),
        ]
        .into_iter()
        .collect();

        let (active, map) = RoutingMode::Threshold(0.4).apply(&ids, &weights);
        let active_set: std::collections::HashSet<&str> =
            active.iter().map(|id| id.0.as_str()).collect();
        assert!(!active_set.contains("a"), "0.1 < 0.4 must drop");
        assert!(active_set.contains("b"), "0.5 >= 0.4 must keep");
        assert!(active_set.contains("c"));
        assert!(!map.contains_key(&IndexId::new("a")));
    }

    /// Indexes missing a weight entry default to neutral 1.0, so threshold
    /// routing must not silently drop them — otherwise "no metadata"
    /// becomes "no relevance" by accident.
    #[test]
    fn routing_threshold_keeps_neutral_indexes() {
        let ids = vec![IndexId::new("known"), IndexId::new("missing")];
        let weights: std::collections::HashMap<IndexId, f32> =
            [(IndexId::new("known"), 0.05_f32)].into_iter().collect();

        let (active, _map) = RoutingMode::Threshold(0.5).apply(&ids, &weights);
        let active_set: std::collections::HashSet<&str> =
            active.iter().map(|id| id.0.as_str()).collect();
        assert!(!active_set.contains("known"), "0.05 < 0.5 dropped");
        // Missing entries default to 1.0 → kept.
        assert!(
            active_set.contains("missing"),
            "indexes without a context embedding must use neutral 1.0 weight"
        );
    }

    /// Verify request → routing-mode resolution: missing or unknown values
    /// fall back to `All`; explicit values pick the right strategy and
    /// honour their `n` / `threshold` knobs.
    #[test]
    fn routing_mode_from_request_resolves_strategy() {
        let base =
            |routing: Option<&str>, n: Option<usize>, t: Option<f32>| -> GlobalSearchRequest {
                GlobalSearchRequest {
                    query: "x".into(),
                    top_k: 1,
                    full_content: false,
                    indexes: None,
                    routing: routing.map(|s| s.to_string()),
                    routing_n: n,
                    routing_threshold: t,
                }
            };
        assert!(matches!(
            RoutingMode::from_request(&base(None, None, None)),
            RoutingMode::All
        ));
        assert!(matches!(
            RoutingMode::from_request(&base(Some("garbage"), None, None)),
            RoutingMode::All
        ));
        match RoutingMode::from_request(&base(Some("top_n"), Some(5), None)) {
            RoutingMode::TopN(n) => assert_eq!(n, 5),
            _ => panic!("expected TopN"),
        }
        match RoutingMode::from_request(&base(Some("top_n"), None, None)) {
            RoutingMode::TopN(n) => assert_eq!(n, RoutingMode::DEFAULT_TOP_N),
            _ => panic!("expected TopN default"),
        }
        match RoutingMode::from_request(&base(Some("threshold"), None, Some(0.7))) {
            RoutingMode::Threshold(t) => assert!((t - 0.7).abs() < 1e-6),
            _ => panic!("expected Threshold"),
        }
    }

    /// Issue #121: after `install_embedder_error` records a hang/timeout,
    /// `/health` must report `embedder: "error"` plus a human-readable
    /// `embedder_error` field so operators don't waste hours debugging a
    /// daemon stuck in `"initializing"`.
    #[tokio::test]
    async fn install_embedder_error_surfaces_in_health() {
        use crate::core::registry::IndexRegistry;

        let state = SearchAppState::new(IndexRegistry::new());
        state
            .install_embedder_error("init timed out after 60s")
            .await;
        let state_arc = Arc::new(state);
        let Json(resp) = health_handler(State(state_arc)).await;
        assert_eq!(resp.embedder, "error");
        assert_eq!(
            resp.embedder_error.as_deref(),
            Some("init timed out after 60s"),
        );
    }

    /// Issue #121: when the embedder init task recorded a permanent error,
    /// `POST /indexes` must return a 503 carrying the error message rather
    /// than the generic "initializing" reason. Callers (CLI, MCP) rely on
    /// the message to surface the underlying cause to operators.
    #[tokio::test]
    async fn create_index_returns_503_with_error_when_embedder_failed() {
        use crate::core::registry::IndexRegistry;
        use axum::body::to_bytes;

        let state = SearchAppState::new(IndexRegistry::new());
        state
            .install_embedder_error("init timed out after 60s")
            .await;
        let state_arc = Arc::new(state);
        let resp = create_index_handler(
            State(state_arc),
            Json(CreateIndexRequest {
                id: "demo".to_string(),
                root_path: std::path::PathBuf::from("/tmp/demo"),
                include_paths: None,
                exclude_globs: None,
                extensions: None,
                domain_terms: None,
                path_filter: None,
            }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body_bytes = to_bytes(resp.into_body(), 64 * 1024)
            .await
            .expect("read body");
        let body: serde_json::Value = serde_json::from_slice(&body_bytes).expect("valid json");
        let err_str = body
            .get("error")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        assert!(
            err_str.contains("embedder init failed"),
            "expected error message to mention init failure, got: {err_str}",
        );
        assert!(
            err_str.contains("init timed out after 60s"),
            "expected recorded timeout message to be surfaced, got: {err_str}",
        );
    }

    /// Issue #121: after the embedder is installed successfully, a previously
    /// recorded error must be cleared so `/health` reports `"ready"` and not
    /// `"error"` (e.g. if a retry succeeded after a transient failure).
    #[tokio::test]
    async fn install_embedder_clears_previous_error() {
        use crate::core::embed::MockEmbedder;
        use crate::core::registry::IndexRegistry;

        let state = SearchAppState::new(IndexRegistry::new());
        state.install_embedder_error("transient hang").await;
        // Verify the error is recorded.
        assert!(state.current_embedder_error().is_some());

        // Install a healthy embedder — the error must clear.
        let embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder::new(8));
        state.install_embedder(embedder).await;
        assert!(state.current_embedder_error().is_none());
        assert!(state.is_embedder_ready());

        let state_arc = Arc::new(state);
        let Json(resp) = health_handler(State(state_arc)).await;
        assert_eq!(resp.embedder, "ready");
        assert!(resp.embedder_error.is_none());
    }

    /// Issue #120: when the previous reindex for an index aborted at the
    /// memory limit, a follow-up `POST /indexes/:id/reindex` request must be
    /// refused with `429 Too Many Requests` for the duration of the cooldown.
    ///
    /// Why: without the guard, an external caller (CLI watchdog, open-mpm)
    /// that retries on abort would loop: each retry re-processes files that
    /// had no content-hash entry yet, pushes RSS over the limit again, and
    /// aborts again.
    /// What: stages an index, records a memory-abort timestamp, calls
    /// `reindex_handler` and asserts the 429 + JSON body shape. Then resets
    /// the cooldown env to 0 s, removes the entry, and verifies the next
    /// call queues successfully.
    /// Test: this test.
    #[tokio::test]
    async fn reindex_handler_rejects_within_cooldown() {
        use crate::core::{
            indexer::CodeIndexer,
            registry::{IndexHandle, IndexId, IndexRegistry},
        };
        use std::sync::Arc;
        use tokio::sync::RwLock;

        let registry = IndexRegistry::new();
        let id = IndexId::new("cooldown-test");
        let tmp = tempfile::tempdir().expect("tempdir");
        registry.register(IndexHandle::bare(
            id.clone(),
            Arc::new(RwLock::new(CodeIndexer::new("cooldown-test", tmp.path()))),
            tmp.path().to_path_buf(),
        ));
        let state = Arc::new(SearchAppState::new(registry));

        // Simulate a prior memory abort by writing a fresh timestamp.
        state
            .last_reindex_aborted_at
            .insert(id.clone(), std::time::Instant::now());

        // Default cooldown is 300 s — handler must refuse with 429.
        let result = reindex_handler(
            State(Arc::clone(&state)),
            axum::extract::Path("cooldown-test".to_string()),
            None,
        )
        .await;
        let err = result.expect_err("expected 429 inside cooldown window");
        assert_eq!(err.0, StatusCode::TOO_MANY_REQUESTS);
        let body = err.1 .0;
        assert!(body.get("retry_after_secs").is_some());
        assert!(body.get("hint").is_some());
        assert_eq!(body["index_id"], "cooldown-test");

        // Drop the abort entry and verify the next call queues successfully.
        state.last_reindex_aborted_at.remove(&id);
        let ok = reindex_handler(
            State(Arc::clone(&state)),
            axum::extract::Path("cooldown-test".to_string()),
            None,
        )
        .await
        .expect("queued");
        assert_eq!(ok.0["queued"], serde_json::Value::Bool(true));
    }

    /// Issue #120: the `AbortedMemory` variant must serialize to the
    /// kebab-case-but-lowercase form (`"abortedmemory"`) consistent with the
    /// existing `Complete`/`Failed`/`Running` variants. External callers
    /// parse the status string off the SSE stream, so the wire format is
    /// load-bearing.
    /// Test: this test.
    #[tokio::test]
    async fn reindex_status_aborted_memory_serializes_lowercase() {
        let status = crate::service::reindex::ReindexStatus::AbortedMemory;
        let json = serde_json::to_string(&status).expect("serialize");
        assert_eq!(json, "\"abortedmemory\"");
    }
}
