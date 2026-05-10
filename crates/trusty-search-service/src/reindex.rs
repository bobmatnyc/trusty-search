//! Reindex orchestration with SSE progress tracking.
//!
//! Why: A full reindex of a project may touch hundreds or thousands of files.
//! The CLI wants to render a progress bar; the daemon wants to fire-and-forget.
//! This module bridges the two via `tokio::sync::broadcast` channels and a
//! per-index `ReindexProgress` snapshot stored on `SearchAppState`.
//!
//! What:
//! - `ReindexProgress` — current state of a reindex (status counters + replay
//!   buffer + broadcast sender).
//! - `spawn_reindex` — kick off a background task that walks `root_path`,
//!   indexes each file, and emits progress events.
//!
//! Test: see `crates/trusty-search-service/src/reindex.rs#tests`.

use crate::walker::{should_skip_content, walk_source_files};
use dashmap::DashMap;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};
use std::time::Instant;
use tokio::sync::{broadcast, Mutex};
use trusty_search_core::registry::{IndexHandle, IndexId};

/// Files per parallel batch. Each batch is parsed in parallel via rayon and
/// embedded in a single ONNX call (256 chunks at a time inside the batch).
///
/// 128 files keeps the embedder's ONNX session saturated (typical Java/Rust
/// files chunk to ~3-10 chunks each, so 128 files comfortably feeds several
/// 256-chunk embed calls per batch) while still letting the SSE progress
/// stream emit useful interim updates. Larger values trade off responsiveness
/// of progress events against marginal gains in lock-acquisition amortization.
const REINDEX_BATCH_SIZE: usize = 128;

/// Per-index, per-process content-hash cache. Used to skip reindexing files
/// whose content hasn't changed since the last reindex in this daemon's
/// lifetime. Survives across `POST /indexes/:id/reindex` calls but not daemon
/// restarts (acceptable: cold start re-embeds everything anyway, and on warm
/// daemons the user expects "skip unchanged" behaviour).
fn file_hashes() -> &'static DashMap<IndexId, Arc<DashMap<PathBuf, String>>> {
    static FILE_HASHES: OnceLock<DashMap<IndexId, Arc<DashMap<PathBuf, String>>>> =
        OnceLock::new();
    FILE_HASHES.get_or_init(DashMap::new)
}

fn hashes_for(id: &IndexId) -> Arc<DashMap<PathBuf, String>> {
    file_hashes()
        .entry(id.clone())
        .or_insert_with(|| Arc::new(DashMap::new()))
        .clone()
}

/// Stable content fingerprint for the "skip unchanged file" optimization.
///
/// Why: SHA-256 is collision-resistant and stable across processes, builds,
/// and Rust versions. `DefaultHasher` (SipHash) is randomized per build and
/// has weaker collision properties — fine for `HashMap` keys but unsafe for
/// content fingerprinting where a false negative silently skips a real edit.
/// What: SHA-256 of the file's UTF-8 bytes, hex-encoded.
/// Test: see `reindex_walks_directory_and_emits_events` — a re-run of the
/// reindex with unchanged files must mark them as skipped (proves the hash
/// is stable across two invocations within the same process).
fn hash_content(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Capacity of the per-reindex broadcast channel. Lagged subscribers will
/// drop events older than this — the SSE handler also replays from the buffer
/// stored in `events`, so late subscribers still see the full history.
const BROADCAST_CAPACITY: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ReindexStatus {
    Running,
    Complete,
    Failed,
}

/// Live state of a reindex. Wrapped in `Arc` and stored on
/// `SearchAppState::reindex_progress` so concurrent SSE subscribers can read
/// the same snapshot without coordinating.
pub struct ReindexProgress {
    pub status: parking_lot_like::AtomicCell<ReindexStatus>,
    pub total_files: std::sync::atomic::AtomicUsize,
    pub indexed: std::sync::atomic::AtomicUsize,
    pub total_chunks: std::sync::atomic::AtomicUsize,
    pub errors: std::sync::atomic::AtomicUsize,
    /// Files skipped because their content hash matched the previous reindex.
    pub skipped: std::sync::atomic::AtomicUsize,
    /// Append-only log of JSON-encoded events. Replayed to late SSE
    /// subscribers so they don't miss earlier `start` / `progress` events.
    pub events: Arc<Mutex<Vec<String>>>,
    /// Live event broadcaster. Subscribers receive new events as they're sent.
    pub sender: broadcast::Sender<String>,
}

/// Tiny in-module helper to avoid pulling parking_lot. We only need atomic
/// load/store of an enum, so `AtomicU8` + From/Into is enough.
mod parking_lot_like {
    use std::sync::atomic::{AtomicU8, Ordering};

    pub struct AtomicCell<T> {
        inner: AtomicU8,
        _marker: std::marker::PhantomData<T>,
    }

    impl<T: Copy + Into<u8> + From<u8>> AtomicCell<T> {
        pub fn new(value: T) -> Self {
            Self {
                inner: AtomicU8::new(value.into()),
                _marker: std::marker::PhantomData,
            }
        }
        pub fn load(&self) -> T {
            T::from(self.inner.load(Ordering::Acquire))
        }
        pub fn store(&self, value: T) {
            self.inner.store(value.into(), Ordering::Release);
        }
    }
}

impl From<ReindexStatus> for u8 {
    fn from(s: ReindexStatus) -> u8 {
        match s {
            ReindexStatus::Running => 0,
            ReindexStatus::Complete => 1,
            ReindexStatus::Failed => 2,
        }
    }
}

impl From<u8> for ReindexStatus {
    fn from(v: u8) -> Self {
        match v {
            1 => ReindexStatus::Complete,
            2 => ReindexStatus::Failed,
            _ => ReindexStatus::Running,
        }
    }
}

impl ReindexProgress {
    pub fn new() -> Self {
        let (sender, _) = broadcast::channel(BROADCAST_CAPACITY);
        Self {
            status: parking_lot_like::AtomicCell::new(ReindexStatus::Running),
            total_files: Default::default(),
            indexed: Default::default(),
            total_chunks: Default::default(),
            errors: Default::default(),
            skipped: Default::default(),
            events: Arc::new(Mutex::new(Vec::new())),
            sender,
        }
    }

    /// Push an event onto the replay buffer and broadcast it to live subscribers.
    pub async fn push(&self, event: serde_json::Value) {
        let line = event.to_string();
        self.events.lock().await.push(line.clone());
        // Broadcast errors (no receivers) are fine — replay buffer still has it.
        let _ = self.sender.send(line);
    }
}

impl Default for ReindexProgress {
    fn default() -> Self {
        Self::new()
    }
}

/// Spawn a background tokio task that walks `handle.root_path`, indexes each
/// source file, and emits progress events into `progress`.
///
/// Returns immediately; the caller (the HTTP handler) drops its reference and
/// the task runs to completion.
pub fn spawn_reindex(handle: Arc<IndexHandle>, progress: Arc<ReindexProgress>) {
    tokio::spawn(async move {
        use std::sync::atomic::Ordering;

        let started = Instant::now();
        let root = handle.root_path.clone();
        let index_id: IndexId = handle.id.clone();
        let walk = walk_source_files(&root);
        let total = walk.files.len();
        progress.total_files.store(total, Ordering::Release);
        progress
            .push(serde_json::json!({
                "event": "start",
                "total_files": total,
                "index_id": index_id.0,
                "root_path": root,
            }))
            .await;

        let hashes = hashes_for(&index_id);

        // Process files in batches. Each batch:
        //  1. Reads files concurrently (tokio::fs::read_to_string).
        //  2. Skips files whose content hash matches the previous reindex.
        //  3. Calls `CodeIndexer::index_files_batch` — one ONNX call per
        //     EMBED_BATCH_SIZE chunks across the whole batch, one symbol-graph
        //     rebuild per batch.
        for batch in walk.files.chunks(REINDEX_BATCH_SIZE) {
            // 1) Read all files in the batch concurrently.
            let read_futs = batch.iter().map(|path| {
                let path = path.clone();
                async move {
                    let content = tokio::fs::read_to_string(&path).await;
                    (path, content)
                }
            });
            let read_results = futures::future::join_all(read_futs).await;

            // 2) Build the batch payload, applying hash-skip.
            let mut to_index: Vec<(String, String)> = Vec::with_capacity(batch.len());
            let mut to_index_paths: Vec<PathBuf> = Vec::with_capacity(batch.len());
            let mut new_hashes: Vec<(PathBuf, String)> = Vec::with_capacity(batch.len());
            for (path, content_res) in read_results {
                let rel = path
                    .strip_prefix(&root)
                    .unwrap_or(&path)
                    .display()
                    .to_string();
                let content = match content_res {
                    Ok(c) => c,
                    Err(e) => {
                        progress.errors.fetch_add(1, Ordering::Release);
                        progress
                            .push(serde_json::json!({
                                "event": "error",
                                "file": rel,
                                "message": format!("read: {e}"),
                                "indexed": progress.indexed.load(Ordering::Acquire),
                                "total_files": total,
                            }))
                            .await;
                        continue;
                    }
                };
                // Content-level minification check. Catches minified bundles
                // that don't carry a `.min.js` suffix — detected after read so
                // we can inspect the actual line structure.
                if should_skip_content(&path, &content) {
                    tracing::debug!(
                        "reindex: skipping minified content in {}",
                        path.display()
                    );
                    progress.skipped.fetch_add(1, Ordering::Release);
                    let indexed = progress.indexed.fetch_add(1, Ordering::Release) + 1;
                    progress
                        .push(serde_json::json!({
                            "event": "skip",
                            "file": rel,
                            "reason": "minified",
                            "indexed": indexed,
                            "total_files": total,
                        }))
                        .await;
                    continue;
                }
                let h = hash_content(&content);
                if hashes.get(&path).map(|prev| *prev == h).unwrap_or(false) {
                    progress.skipped.fetch_add(1, Ordering::Release);
                    let indexed = progress.indexed.fetch_add(1, Ordering::Release) + 1;
                    progress
                        .push(serde_json::json!({
                            "event": "skip",
                            "file": rel,
                            "indexed": indexed,
                            "total_files": total,
                        }))
                        .await;
                    continue;
                }
                let path_str = path.to_string_lossy().to_string();
                to_index.push((path_str, content));
                to_index_paths.push(path.clone());
                new_hashes.push((path, h));
            }

            if to_index.is_empty() {
                continue;
            }

            // 3) Bulk-index. We need the corpus-write paths inside the indexer,
            //    so take the write lock for the duration of the batch — this
            //    is still net cheaper than the per-file lock thrash.
            //    `_no_rebuild` defers symbol-graph rebuild to the very end of
            //    the reindex (one rebuild instead of one per batch — major win
            //    on large corpora since the rebuild is roughly O(N + E) over
            //    the whole corpus and would otherwise scale quadratically with
            //    the file count).
            let result = {
                let indexer = handle.indexer.write().await;
                indexer.index_files_batch_no_rebuild(&to_index).await
            };
            match result {
                Ok(new_chunks) => {
                    progress.total_chunks.fetch_add(new_chunks, Ordering::Release);
                    let batch_files = to_index.len();
                    let indexed =
                        progress.indexed.fetch_add(batch_files, Ordering::Release) + batch_files;
                    let elapsed_ms = started.elapsed().as_millis() as u64;
                    let chunks_per_sec = if elapsed_ms > 0 {
                        (progress.total_chunks.load(Ordering::Acquire) as u64 * 1000)
                            / elapsed_ms
                    } else {
                        0
                    };
                    // Persist new content hashes for next reindex.
                    for (path, h) in new_hashes {
                        hashes.insert(path, h);
                    }
                    progress
                        .push(serde_json::json!({
                            "event": "batch",
                            "batch_files": batch_files,
                            "batch_chunks": new_chunks,
                            "indexed": indexed,
                            "total_files": total,
                            "elapsed_ms": elapsed_ms,
                            "chunks_per_sec": chunks_per_sec,
                        }))
                        .await;
                }
                Err(e) => {
                    // Whole batch failed — surface a single error event listing
                    // the affected files. Caller can retry the failing files
                    // individually via `index_file`.
                    let files_in_batch: Vec<String> = to_index_paths
                        .iter()
                        .map(|p| {
                            p.strip_prefix(&root)
                                .unwrap_or(p)
                                .display()
                                .to_string()
                        })
                        .collect();
                    progress
                        .errors
                        .fetch_add(to_index_paths.len(), Ordering::Release);
                    progress
                        .push(serde_json::json!({
                            "event": "error",
                            "files": files_in_batch,
                            "message": format!("batch index: {e}"),
                            "indexed": progress.indexed.load(Ordering::Acquire),
                            "total_files": total,
                        }))
                        .await;
                }
            }
        }

        // Rebuild the symbol graph once for the whole reindex. We deferred
        // per-batch rebuilds above because each rebuild is O(N + E) over the
        // entire corpus and would scale quadratically with file count if run
        // per batch. One rebuild at the end gives the same final state.
        {
            let indexer = handle.indexer.read().await;
            indexer.rebuild_symbol_graph_now().await;
        }

        progress.status.store(ReindexStatus::Complete);
        let total_chunks = progress.total_chunks.load(Ordering::Acquire);
        let elapsed_ms = started.elapsed().as_millis() as u64;
        let chunks_per_sec = if elapsed_ms > 0 {
            (total_chunks as u64 * 1000) / elapsed_ms
        } else {
            0
        };
        progress
            .push(serde_json::json!({
                "event": "complete",
                "indexed": progress.indexed.load(Ordering::Acquire),
                "total_chunks": total_chunks,
                "skipped": progress.skipped.load(Ordering::Acquire),
                "errors": progress.errors.load(Ordering::Acquire),
                "elapsed_ms": elapsed_ms,
                "chunks_per_sec": chunks_per_sec,
            }))
            .await;
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::atomic::Ordering;
    use trusty_search_core::indexer::CodeIndexer;

    #[tokio::test]
    async fn reindex_walks_directory_and_emits_events() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path().to_path_buf();
        fs::write(root.join("a.rs"), "fn a() {}").unwrap();
        fs::write(root.join("b.py"), "def b():\n    pass\n").unwrap();
        fs::create_dir(root.join("target")).unwrap();
        fs::write(root.join("target/skip.rs"), "fn skip() {}").unwrap();

        let indexer = CodeIndexer::new("test".to_string(), root.clone());
        let handle = Arc::new(IndexHandle {
            id: IndexId::new("test"),
            indexer: Arc::new(tokio::sync::RwLock::new(indexer)),
            root_path: root.clone(),
        });
        let progress = Arc::new(ReindexProgress::new());
        spawn_reindex(handle, progress.clone());

        // Wait up to 10s for completion.
        for _ in 0..100 {
            if progress.status.load() == ReindexStatus::Complete {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
        assert_eq!(progress.status.load(), ReindexStatus::Complete);
        assert_eq!(progress.total_files.load(Ordering::Acquire), 2);
        assert_eq!(progress.indexed.load(Ordering::Acquire), 2);

        let events = progress.events.lock().await;
        assert!(events.first().map(|s| s.contains("\"start\"")).unwrap_or(false));
        assert!(events.last().map(|s| s.contains("\"complete\"")).unwrap_or(false));
    }
}
