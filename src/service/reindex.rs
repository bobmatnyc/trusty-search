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

use crate::core::indexer::CommitTimings;
use crate::core::memguard::{current_rss_mb, memory_limit_mb};
use crate::core::registry::{IndexHandle, IndexId};
use crate::service::walker::{should_skip_content, walk_source_files};
use crossbeam_utils::atomic::AtomicCell;
use dashmap::DashMap;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering as AtomicOrdering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};
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

        // Determine which subtrees to walk. With no `include_paths` on the
        // handle (the common single-index case) we walk the entire root.
        // Otherwise we run one walk per configured subtree and concatenate —
        // this is how `trusty-search.yaml` slices a polyrepo into independent
        // indexes.
        let include_paths: Vec<PathBuf> = if handle.include_paths.is_empty() {
            vec![root.clone()]
        } else {
            handle.include_paths.clone()
        };
        let mut walked_files: Vec<PathBuf> = Vec::new();
        let mut total_skipped_dirs: usize = 0;
        for subtree in &include_paths {
            let w = walk_source_files(subtree);
            walked_files.extend(w.files);
            total_skipped_dirs = total_skipped_dirs.saturating_add(w.skipped_dirs);
        }

        // Apply repo-config filters. These are AND-composed on top of the
        // walker's built-in ignores (`SKIP_DIRS`, `should_skip_path`).
        //
        // 1. `exclude_globs`: drop any file whose path matches one of the
        //    user-supplied glob patterns.
        // 2. `extensions`: when non-empty, keep only files whose extension
        //    appears in the allow-list (caller writes them without the
        //    leading dot, e.g. `["rs", "py"]`).
        if !handle.exclude_globs.is_empty() {
            let excludes = handle.exclude_globs.clone();
            walked_files.retain(|p| !crate::core::repo_config::path_matches_any_glob(p, &excludes));
        }
        if !handle.extensions.is_empty() {
            let allowed = handle.extensions.clone();
            walked_files.retain(|p| {
                p.extension()
                    .and_then(|e| e.to_str())
                    .map(|e| allowed.iter().any(|x| x.eq_ignore_ascii_case(e)))
                    .unwrap_or(false)
            });
        }

        // De-duplicate when multiple `include_paths` overlap (e.g. `["."]` plus
        // `["src"]`). `walk_source_files` returns canonicalised paths inside
        // each subtree but doesn't dedupe across subtrees.
        walked_files.sort();
        walked_files.dedup();

        let walk = crate::service::walker::WalkResult {
            files: walked_files,
            skipped_dirs: total_skipped_dirs,
        };
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

        // Per-subsystem timing accumulators. Each phase (parse, embed, BM25,
        // vector upsert) is measured inside the indexer (see `ParsedBatch` /
        // `CommitTimings`) and summed across all batches here. KG is measured
        // separately at the end. Together with `vector_count`, this gives
        // operators per-subsystem visibility — and crucially, a non-zero
        // `embed_ms` with `vector_count == 0` is the smoking-gun signal for the
        // "embedder silently fell back to BM25" failure mode.
        let mut total_parse_ms: u64 = 0;
        let mut total_embed_ms: u64 = 0;
        let mut total_bm25_ms: u64 = 0;
        let mut total_vector_upsert_ms: u64 = 0;
        let mut total_vector_count: usize = 0;

        // Memory-protection state (issues #76, #82). `mem_limit` is `Some`
        // only when `TRUSTY_MEMORY_LIMIT_MB` is set. The previous design
        // sampled RSS every 10 batches *before* parse/embed/commit, which let
        // a single batch push RSS 4× over the configured limit before being
        // noticed (issue #82: 10 GB limit → 40 GB actual).
        //
        // The new design has three layers of protection:
        //   1. A background poller task (spawned below) samples RSS every
        //      `MEM_POLL_INTERVAL` and sets `mem_abort` the moment the limit
        //      is breached. This catches mid-batch spikes that batch-boundary
        //      checks miss.
        //   2. The main loop checks `mem_abort` on EVERY batch (not every 10)
        //      and also AFTER `commit_parsed_batch` returns, so the largest
        //      allocator (HNSW + redb commit) is bracketed by checks.
        //   3. The KG rebuild at the end also honours the abort flag.
        //
        // `peak_rss_mb` is updated by the poller via an atomic so the final
        // log line reflects the true peak, not just batch-boundary samples.
        let mem_limit = memory_limit_mb();
        let mem_abort = Arc::new(AtomicBool::new(false));
        let peak_rss_atomic = Arc::new(AtomicU64::new(current_rss_mb().unwrap_or(0)));
        let mut mem_limit_hit: bool = false;

        /// How often the background poller samples RSS. 1 s strikes a
        /// balance between catching mid-batch spikes and the cost of
        /// `sysinfo::refresh_processes_specifics` (~1–3 ms on macOS).
        const MEM_POLL_INTERVAL: Duration = Duration::from_secs(1);

        // Spawn the background poller. It runs until `poller_stop` flips,
        // updating `peak_rss_atomic` and tripping `mem_abort` whenever RSS
        // crosses `mem_limit`. When no limit is configured we still run the
        // poller so `peak_rss_mb` is accurate for the final log line — the
        // overhead is one sysinfo refresh per second.
        let poller_stop = Arc::new(AtomicBool::new(false));
        let poller_handle = {
            let mem_abort = mem_abort.clone();
            let peak_rss = peak_rss_atomic.clone();
            let stop = poller_stop.clone();
            let index_id_for_log = index_id.0.clone();
            tokio::spawn(async move {
                let mut ticker = tokio::time::interval(MEM_POLL_INTERVAL);
                // Drop the immediate first tick so we don't double-sample
                // with the synchronous `current_rss_mb()` already done above.
                ticker.tick().await;
                loop {
                    if stop.load(AtomicOrdering::Acquire) {
                        break;
                    }
                    if let Some(rss) = current_rss_mb() {
                        // Update peak monotonically.
                        let mut prev = peak_rss.load(AtomicOrdering::Acquire);
                        while rss > prev {
                            match peak_rss.compare_exchange_weak(
                                prev,
                                rss,
                                AtomicOrdering::AcqRel,
                                AtomicOrdering::Acquire,
                            ) {
                                Ok(_) => break,
                                Err(cur) => prev = cur,
                            }
                        }
                        if let Some(limit) = mem_limit {
                            if rss >= limit && !mem_abort.load(AtomicOrdering::Acquire) {
                                tracing::warn!(
                                    "reindex memory poller: rss={}MB >= limit={}MB \
                                     — tripping abort flag for index {}",
                                    rss,
                                    limit,
                                    index_id_for_log,
                                );
                                mem_abort.store(true, AtomicOrdering::Release);
                                // Keep polling so peak_rss continues to track
                                // until the main loop notices the flag.
                            }
                        }
                    }
                    ticker.tick().await;
                }
            })
        };

        // Process files in batches. Each batch:
        //  1. Reads files concurrently (tokio::fs::read_to_string).
        //  2. Skips files whose content hash matches the previous reindex.
        //  3. Calls `CodeIndexer::index_files_batch` — one ONNX call per
        //     EMBED_BATCH_SIZE chunks across the whole batch, one symbol-graph
        //     rebuild per batch.
        for batch in walk.files.chunks(REINDEX_BATCH_SIZE) {
            // Memory-protection check (issues #76, #82). Honour the abort
            // flag set by the background poller — checked on EVERY batch.
            // Skipping remaining batches preserves all chunks already
            // committed; a partial reindex is safer than an OOM-kill that
            // loses the daemon's in-memory state entirely.
            if mem_abort.load(AtomicOrdering::Acquire) {
                let rss = current_rss_mb().unwrap_or(0);
                tracing::warn!(
                    "reindex: memory limit hit before batch (rss={}MB, \
                     limit={:?}MB) — skipping remaining batches for index {}",
                    rss,
                    mem_limit,
                    index_id.0
                );
                mem_limit_hit = true;
                break;
            }

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
            // Each batch returns both the `ParsedBatch` timings (parse_ms,
            // embed_ms, vector_count) and the `CommitTimings` (bm25_ms,
            // vector_upsert_ms). We capture both so the orchestrator can
            // accumulate per-subsystem totals across the whole reindex.
            let result: anyhow::Result<(u64, u64, usize, CommitTimings)> = async {
                let parsed = {
                    let indexer = handle.indexer.read().await;
                    indexer.parse_and_embed_files(to_index.clone()).await?
                };
                let parse_ms = parsed.parse_ms;
                let embed_ms = parsed.embed_ms;
                let vector_count = parsed.vector_count;
                let indexer = handle.indexer.write().await;
                let commit = indexer.commit_parsed_batch(parsed, true).await?;
                Ok((parse_ms, embed_ms, vector_count, commit))
            }
            .await;
            match result {
                Ok((parse_ms, embed_ms, vector_count, commit)) => {
                    let new_chunks = commit.chunks;
                    total_parse_ms = total_parse_ms.saturating_add(parse_ms);
                    total_embed_ms = total_embed_ms.saturating_add(embed_ms);
                    total_bm25_ms = total_bm25_ms.saturating_add(commit.bm25_ms);
                    total_vector_upsert_ms =
                        total_vector_upsert_ms.saturating_add(commit.vector_upsert_ms);
                    total_vector_count = total_vector_count.saturating_add(vector_count);
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
                    // Post-commit memory check (issue #82). The commit phase
                    // (HNSW insert + redb write + BM25 update) is the single
                    // largest in-batch allocator. Sampling RSS *after* commit
                    // — in addition to the pre-batch abort-flag check — means
                    // a runaway batch can only push RSS one batch over the
                    // limit before being noticed, instead of accumulating
                    // across the previous `MEM_CHECK_EVERY_N_BATCHES` cadence.
                    if let Some(limit) = mem_limit {
                        if let Some(rss) = current_rss_mb() {
                            let prev_peak = peak_rss_atomic.load(AtomicOrdering::Acquire);
                            if rss > prev_peak {
                                peak_rss_atomic.store(rss, AtomicOrdering::Release);
                            }
                            if rss >= limit {
                                tracing::warn!(
                                    "reindex: memory limit hit after commit \
                                     (rss={}MB >= limit={}MB) — skipping \
                                     remaining batches for index {}",
                                    rss,
                                    limit,
                                    index_id.0
                                );
                                mem_abort.store(true, AtomicOrdering::Release);
                                mem_limit_hit = true;
                                break;
                            }
                        }
                    }
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
        //
        // Issue #90: previously this stage was skipped whenever
        // `mem_limit_hit` had been tripped at any point during batch
        // processing — even if memory pressure had since subsided. That left
        // every reindex on memory-constrained hosts (or hosts with a
        // transient embedding spike) with `symbol_count: 0, edge_count: 0`,
        // which silently disabled KG-expansion (`hybrid+kg` never appears in
        // search results) for the entire daemon lifetime until the next
        // mutation-driven rebuild. The persisted chunks DO carry
        // `function_name` and `calls`, so building the graph is cheap and
        // bounded by `TRUSTY_MAX_KG_NODES` (default 100k). It is independent
        // of the embedding pipeline that caused the spike, so we always run
        // it now.
        //
        // Issue #82 (original concern): petgraph adjacency lists scale with
        // edge count. For a 100k-node graph that's ~30 MB — negligible next
        // to the GB-scale embedding spike. The hard cap remains the
        // load-bearing safety net.
        let kg_start = Instant::now();
        let kg_skipped;
        let symbol_count;
        let edge_count;
        {
            let indexer = handle.indexer.read().await;
            indexer.rebuild_symbol_graph_now().await;
            let g = indexer.symbol_graph().await;
            symbol_count = g.node_count();
            edge_count = g.edge_count();
            kg_skipped = false;
        }
        if mem_limit_hit || mem_abort.load(AtomicOrdering::Acquire) {
            tracing::warn!(
                "reindex: memory limit was breached during batch processing for \
                 index {} (peak_rss={}MB, limit={:?}MB) — KG was still rebuilt \
                 (symbols={}, edges={}) because graph construction is bounded by \
                 TRUSTY_MAX_KG_NODES and independent of the embedding spike",
                index_id.0,
                peak_rss_atomic.load(AtomicOrdering::Acquire),
                mem_limit,
                symbol_count,
                edge_count,
            );
        }
        let kg_ms = kg_start.elapsed().as_millis() as u64;

        // Stop the background poller and collect the true peak it observed.
        poller_stop.store(true, AtomicOrdering::Release);
        // Best-effort: don't fail completion if the poller is wedged.
        let _ = poller_handle.await;

        progress.status.store(ReindexStatus::Complete);
        let total_chunks = progress.total_chunks.load(Ordering::Acquire);
        let elapsed_ms = started.elapsed().as_millis() as u64;
        let chunks_per_sec = (total_chunks as u64 * 1000)
            .checked_div(elapsed_ms)
            .unwrap_or(0);

        // Final synchronous RSS poll so the peak reflects post-KG memory
        // (the symbol graph rebuild may itself push RSS higher than any
        // background sample taken before it ran).
        if let Some(rss) = current_rss_mb() {
            let prev = peak_rss_atomic.load(AtomicOrdering::Acquire);
            if rss > prev {
                peak_rss_atomic.store(rss, AtomicOrdering::Release);
            }
        }
        let peak_rss_mb = peak_rss_atomic.load(AtomicOrdering::Acquire);
        let indexed_final = progress.indexed.load(Ordering::Acquire);
        tracing::info!(
            "reindex complete: index={} files={} chunks={} elapsed_ms={} \
             peak_rss_mb={} memory_limit_hit={}",
            index_id.0,
            indexed_final,
            total_chunks,
            elapsed_ms,
            peak_rss_mb,
            mem_limit_hit,
        );

        progress
            .push(serde_json::json!({
                "event": "complete",
                "indexed": indexed_final,
                "total_chunks": total_chunks,
                "skipped": progress.skipped.load(Ordering::Acquire),
                "errors": progress.errors.load(Ordering::Acquire),
                "elapsed_ms": elapsed_ms,
                "chunks_per_sec": chunks_per_sec,
                "peak_rss_mb": peak_rss_mb,
                "memory_limit_hit": mem_limit_hit,
                "kg_skipped": kg_skipped,
                "timings": {
                    "parse_ms": total_parse_ms,
                    "embed_ms": total_embed_ms,
                    "bm25_ms": total_bm25_ms,
                    "vector_upsert_ms": total_vector_upsert_ms,
                    "kg_ms": kg_ms,
                    "vector_count": total_vector_count,
                    "symbol_count": symbol_count,
                    "edge_count": edge_count,
                },
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

    /// Filter wiring: with `include_paths` set on the handle, the reindex
    /// must walk ONLY those subtrees. Files outside the configured slice
    /// must not appear in the corpus.
    ///
    /// Why: `trusty-search.yaml` declares `paths: [api/src]` to slice a
    /// polyrepo. Without this test, a regression that drops the
    /// `handle.include_paths` branch silently reverts to "walk everything",
    /// which is the bug the YAML config exists to avoid.
    /// What: stage a fixture with `api/keep.rs` and `ui/drop.rs`, register a
    /// handle whose `include_paths = [<root>/api]`, run the reindex, and
    /// assert only the api file was indexed.
    /// Test: this test.
    #[tokio::test]
    async fn reindex_honours_include_paths_filter() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path().to_path_buf();
        fs::create_dir_all(root.join("api")).unwrap();
        fs::create_dir_all(root.join("ui")).unwrap();
        fs::write(root.join("api/keep.rs"), "fn keep_me() {}\n").unwrap();
        fs::write(root.join("ui/drop.rs"), "fn drop_me() {}\n").unwrap();

        let indexer = CodeIndexer::new("filter-test", root.clone());
        let handle = Arc::new(IndexHandle {
            id: IndexId::new("filter-test"),
            indexer: Arc::new(tokio::sync::RwLock::new(indexer)),
            root_path: root.clone(),
            include_paths: vec![root.join("api")],
            exclude_globs: vec![],
            extensions: vec![],
            domain_terms: vec![],
        });
        let progress = Arc::new(ReindexProgress::new());
        spawn_reindex(handle.clone(), progress.clone(), false);

        // Wait up to 10s for completion.
        for _ in 0..100 {
            if progress.status.load() == ReindexStatus::Complete {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
        assert_eq!(progress.status.load(), ReindexStatus::Complete);
        assert_eq!(
            progress.total_files.load(Ordering::Acquire),
            1,
            "only api/keep.rs should be walked"
        );

        // And the corpus must contain `keep_me` but not `drop_me`.
        let idx = handle.indexer.read().await;
        let r = idx
            .search(&crate::core::indexer::SearchQuery {
                text: "keep_me".into(),
                top_k: 5,
                expand_graph: false,
                compact: false,
            })
            .await
            .unwrap();
        assert!(r.iter().any(|c| c.content.contains("keep_me")));
        let r2 = idx
            .search(&crate::core::indexer::SearchQuery {
                text: "drop_me".into(),
                top_k: 5,
                expand_graph: false,
                compact: false,
            })
            .await
            .unwrap();
        assert!(
            !r2.iter().any(|c| c.content.contains("drop_me")),
            "ui/drop.rs must not have been indexed"
        );
    }

    #[tokio::test]
    async fn reindex_walks_directory_and_emits_events() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path().to_path_buf();
        fs::write(root.join("a.rs"), "fn a() {}").unwrap();
        fs::write(root.join("b.py"), "def b():\n    pass\n").unwrap();
        fs::create_dir(root.join("target")).unwrap();
        fs::write(root.join("target/skip.rs"), "fn skip() {}").unwrap();

        let indexer = CodeIndexer::new("test".to_string(), root.clone());
        let handle = Arc::new(IndexHandle::bare(
            IndexId::new("test"),
            Arc::new(tokio::sync::RwLock::new(indexer)),
            root.clone(),
        ));
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
