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

use crate::core::registry::{IndexHandle, IndexId};
use crate::service::walker::{should_skip_content, walk_source_files};
use crossbeam_utils::atomic::AtomicCell;
use dashmap::DashMap;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};
use std::time::Instant;
use tokio::sync::{broadcast, Mutex, Semaphore};

/// Machine-wide reindex serializer.
///
/// Why: The ONNX fastembed model allocates large working buffers per session
/// (~7–10 GB virtual mem per concurrent `embed()` call on a real codebase).
/// When multiple `POST /indexes/:id/reindex` requests arrive concurrently
/// (e.g. benchmark agents, parallel CLI runs), each `spawn_reindex` task
/// races into `parse_and_embed_files` and the daemon balloons to 28–46 GB,
/// triggering macOS Jetsam to kill the process.
///
/// What: A 1-permit semaphore. Waiting reindexes queue (the SSE stream is
/// already connected and will start emitting events once the permit is held).
/// The permit is released when the spawned task's async block returns.
fn reindex_semaphore() -> &'static Semaphore {
    static SEM: OnceLock<Semaphore> = OnceLock::new();
    SEM.get_or_init(|| Semaphore::new(1))
}

/// Files per parallel batch. Each batch is parsed in parallel via rayon and
/// embedded in ONNX batches (`EMBED_BATCH_SIZE` chunks at a time inside the
/// batch). The full `ParsedBatch` (chunk content + embeddings + entities for
/// every file in the batch) is held in memory until the commit phase finishes.
///
/// 128 files bounds peak memory during reindex. On a 595-file repo with ~8
/// chunks/file, a 512-file batch held ~4k chunks of source content plus their
/// 384-dim f32 embeddings plus ONNX intermediate activation tensors retained
/// across the embed loop — pushing RSS to 33–50 GB and triggering macOS Jetsam
/// kill. With 128 files per batch, the working set caps at ~1k chunks worth of
/// memory, and the ONNX session arena gets multiple opportunities to release
/// transient buffers between commits. SSE progress events fire per batch, so a
/// smaller batch size also gives more granular progress updates — the downside
/// is slightly more lock-acquisition overhead, which is negligible vs. the
/// per-batch parse+embed cost.
const REINDEX_BATCH_SIZE: usize = 128;

/// Per-index, per-process content-hash cache. Used to skip reindexing files
/// whose content hasn't changed since the last reindex in this daemon's
/// lifetime. Survives across `POST /indexes/:id/reindex` calls but not daemon
/// restarts (acceptable: cold start re-embeds everything anyway, and on warm
/// daemons the user expects "skip unchanged" behaviour).
fn file_hashes() -> &'static DashMap<IndexId, Arc<DashMap<PathBuf, String>>> {
    static FILE_HASHES: OnceLock<DashMap<IndexId, Arc<DashMap<PathBuf, String>>>> = OnceLock::new();
    FILE_HASHES.get_or_init(DashMap::new)
}

fn hashes_for(id: &IndexId) -> Arc<DashMap<PathBuf, String>> {
    file_hashes()
        .entry(id.clone())
        .or_insert_with(|| Arc::new(DashMap::new()))
        .clone()
}

/// Per-index ceiling on the content-hash cache (issue #75). Each entry holds
/// a `PathBuf` + 64-char hex SHA-256 string, so 200k entries ≈ ~30–60 MB.
/// When exceeded we drain ~10% of the entries (DashMap has no ordering, so
/// the eviction set is arbitrary — those files are simply re-hashed on the
/// next reindex, which is the safe, correct fallback).
const MAX_FILE_HASHES_PER_INDEX: usize = 200_000;

/// Drop ~10% of entries from `map` when above `MAX_FILE_HASHES_PER_INDEX`.
///
/// Why: prevents an unbounded growth in the per-daemon content-hash cache
/// when a project gets ever-larger or files are renamed many times. The
/// hash cache is a pure speed optimisation (skip re-embed for unchanged
/// files), so evicting entries is always safe — affected files just get
/// re-hashed and re-embedded on the next reindex.
/// What: collects an arbitrary subset of keys and removes them. DashMap has
/// no insertion-order metadata so we can't do "true" LRU; arbitrary eviction
/// is acceptable for a cache whose miss penalty is just extra work.
/// Test: covered indirectly by the reindex test (oversizing not exercised).
fn shrink_hashes_if_needed(map: &DashMap<PathBuf, String>) {
    let len = map.len();
    if len <= MAX_FILE_HASHES_PER_INDEX {
        return;
    }
    let target = MAX_FILE_HASHES_PER_INDEX * 9 / 10;
    let to_remove = len.saturating_sub(target);
    let keys: Vec<PathBuf> = map
        .iter()
        .take(to_remove)
        .map(|e| e.key().clone())
        .collect();
    for k in keys {
        map.remove(&k);
    }
    tracing::info!(
        "file-hash cache exceeded {} entries — dropped {} to bound memory",
        MAX_FILE_HASHES_PER_INDEX,
        to_remove
    );
}

/// Max replay events buffered on a `ReindexProgress`. A full reindex emits
/// ~100 events for a 14k-file repo (one per batch + start/complete), but
/// pathological cases (per-file errors) could otherwise grow the vector
/// without bound. Late SSE subscribers still see the most recent 500 events,
/// which is more than enough to replay context.
const MAX_REPLAY_EVENTS: usize = 500;

/// How long to keep a completed (`Complete` / `Failed`) `ReindexProgress`
/// on `SearchAppState::reindex_progress` before garbage-collecting it.
/// 60 s is enough for late SSE subscribers to attach and read the final
/// state but short enough that long-running daemons don't accumulate
/// thousands of stale progress entries.
const REINDEX_PROGRESS_TTL_SECS: u64 = 60;

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
    pub status: AtomicCell<ReindexStatus>,
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

impl ReindexProgress {
    pub fn new() -> Self {
        let (sender, _) = broadcast::channel(BROADCAST_CAPACITY);
        Self {
            status: AtomicCell::new(ReindexStatus::Running),
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
    /// Caps the replay buffer at `MAX_REPLAY_EVENTS` to bound memory under
    /// pathological reindexes (e.g. one error event per file).
    pub async fn push(&self, event: serde_json::Value) {
        let line = event.to_string();
        {
            let mut buf = self.events.lock().await;
            if buf.len() >= MAX_REPLAY_EVENTS {
                // Drop the oldest event. `remove(0)` is O(n) but n ≤ 500.
                buf.remove(0);
            }
            buf.push(line.clone());
        }
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
/// the task runs to completion. When `cleanup_map` is `Some`, the entry for
/// `handle.id` is removed from the map `REINDEX_PROGRESS_TTL_SECS` after
/// completion (issue #75: bounds long-running daemon memory by GC'ing stale
/// progress entries while still letting late SSE subscribers read the final
/// state for a short window).
pub fn spawn_reindex(handle: Arc<IndexHandle>, progress: Arc<ReindexProgress>, force: bool) {
    spawn_reindex_with_cleanup(handle, progress, force, None);
}

/// Variant of `spawn_reindex` that GC's the progress map after completion.
/// See `spawn_reindex` for the rationale.
pub fn spawn_reindex_with_cleanup(
    handle: Arc<IndexHandle>,
    progress: Arc<ReindexProgress>,
    force: bool,
    cleanup_map: Option<Arc<DashMap<IndexId, Arc<ReindexProgress>>>>,
) {
    let cleanup_id = handle.id.clone();
    tokio::spawn(async move {
        use std::sync::atomic::Ordering;

        // Serialize reindexes machine-wide to avoid stacking multiple
        // simultaneous ONNX embedder sessions (Jetsam kill at ~28GB on macOS).
        // Late arrivals queue here; their SSE stream is already attached and
        // will replay buffered events once the permit is acquired.
        let _permit = reindex_semaphore()
            .acquire()
            .await
            .expect("reindex semaphore is never closed");

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
                "force": force,
            }))
            .await;

        let hashes = hashes_for(&index_id);
        // `--force` wipes the per-index content-hash cache so every file is
        // re-parsed, re-embedded, and re-committed even if its bytes haven't
        // changed since the last reindex in this daemon's lifetime. Without
        // this, the hash-skip check below silently turns `--force` into a
        // no-op on a warm daemon.
        if force {
            hashes.clear();
        }

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
                    tracing::debug!("reindex: skipping minified content in {}", path.display());
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

            // 3) Bulk-index. We split the work into:
            //    (a) parse + embed — pure CPU/ONNX, NO write lock required.
            //        Concurrent searches and the next batch's I/O proceed
            //        unblocked while this runs.
            //    (b) commit — acquires the corpus/BM25/HNSW write locks for
            //        the minimum window needed to install the new chunks.
            //    The graph rebuild is deferred to the very end of the reindex
            //    (one rebuild instead of one per batch).
            let result: anyhow::Result<usize> = async {
                let parsed = {
                    let indexer = handle.indexer.read().await;
                    indexer.parse_and_embed_files(to_index.clone()).await?
                };
                let indexer = handle.indexer.write().await;
                indexer.commit_parsed_batch(parsed, true).await
            }
            .await;
            match result {
                Ok(new_chunks) => {
                    progress
                        .total_chunks
                        .fetch_add(new_chunks, Ordering::Release);
                    let batch_files = to_index.len();
                    let indexed =
                        progress.indexed.fetch_add(batch_files, Ordering::Release) + batch_files;
                    let elapsed_ms = started.elapsed().as_millis() as u64;
                    let chunks_per_sec = (progress.total_chunks.load(Ordering::Acquire) as u64
                        * 1000)
                        .checked_div(elapsed_ms)
                        .unwrap_or(0);
                    // Persist new content hashes for next reindex.
                    for (path, h) in new_hashes {
                        hashes.insert(path, h);
                    }
                    // Issue #75: cap per-index hash-cache size. This is a
                    // pure speed cache (skip-unchanged) so arbitrary
                    // eviction is always safe.
                    shrink_hashes_if_needed(&hashes);
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
                        .map(|p| p.strip_prefix(&root).unwrap_or(p).display().to_string())
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
        let chunks_per_sec = (total_chunks as u64 * 1000)
            .checked_div(elapsed_ms)
            .unwrap_or(0);
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

        // Issue #75: GC the progress entry after a short delay so the
        // `reindex_progress` map doesn't grow unboundedly on long-running
        // daemons. The delay lets late SSE subscribers read the final
        // `complete`/`error` event before the map drops its reference.
        if let Some(map) = cleanup_map {
            tokio::spawn(async move {
                tokio::time::sleep(std::time::Duration::from_secs(REINDEX_PROGRESS_TTL_SECS)).await;
                map.remove(&cleanup_id);
            });
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::indexer::CodeIndexer;
    use std::fs;
    use std::sync::atomic::Ordering;

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
        spawn_reindex(handle, progress.clone(), false);

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
        assert!(events
            .first()
            .map(|s| s.contains("\"start\""))
            .unwrap_or(false));
        assert!(events
            .last()
            .map(|s| s.contains("\"complete\""))
            .unwrap_or(false));
    }
}
