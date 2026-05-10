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

use crate::walker::walk_source_files;
use serde::Serialize;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{broadcast, Mutex};
use trusty_search_core::registry::{IndexHandle, IndexId};

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
        let started = Instant::now();
        let root = handle.root_path.clone();
        let index_id: IndexId = handle.id.clone();
        let walk = walk_source_files(&root);
        let total = walk.files.len();
        progress
            .total_files
            .store(total, std::sync::atomic::Ordering::Release);
        progress
            .push(serde_json::json!({
                "event": "start",
                "total_files": total,
                "index_id": index_id.0,
                "root_path": root,
            }))
            .await;

        for path in walk.files {
            let rel = path
                .strip_prefix(&root)
                .unwrap_or(&path)
                .display()
                .to_string();
            let content = match tokio::fs::read_to_string(&path).await {
                Ok(c) => c,
                Err(e) => {
                    progress
                        .errors
                        .fetch_add(1, std::sync::atomic::Ordering::Release);
                    progress
                        .push(serde_json::json!({
                            "event": "error",
                            "file": rel,
                            "message": format!("read: {e}"),
                            "indexed": progress.indexed.load(std::sync::atomic::Ordering::Acquire),
                            "total_files": total,
                        }))
                        .await;
                    continue;
                }
            };

            let path_str = path.to_string_lossy().to_string();
            let before_chunks = {
                let indexer = handle.indexer.read().await;
                indexer.chunk_count()
            };
            let result = {
                let indexer = handle.indexer.read().await;
                indexer.index_file(&path_str, &content).await
            };
            match result {
                Ok(()) => {
                    let after_chunks = {
                        let indexer = handle.indexer.read().await;
                        indexer.chunk_count()
                    };
                    let new_chunks = after_chunks.saturating_sub(before_chunks);
                    progress
                        .total_chunks
                        .fetch_add(new_chunks, std::sync::atomic::Ordering::Release);
                    let indexed = progress
                        .indexed
                        .fetch_add(1, std::sync::atomic::Ordering::Release)
                        + 1;
                    progress
                        .push(serde_json::json!({
                            "event": "progress",
                            "file": rel,
                            "chunks": new_chunks,
                            "indexed": indexed,
                            "total_files": total,
                            "elapsed_ms": started.elapsed().as_millis() as u64,
                        }))
                        .await;
                }
                Err(e) => {
                    progress
                        .errors
                        .fetch_add(1, std::sync::atomic::Ordering::Release);
                    progress
                        .push(serde_json::json!({
                            "event": "error",
                            "file": rel,
                            "message": format!("index: {e}"),
                            "indexed": progress.indexed.load(std::sync::atomic::Ordering::Acquire),
                            "total_files": total,
                        }))
                        .await;
                }
            }
        }

        progress.status.store(ReindexStatus::Complete);
        progress
            .push(serde_json::json!({
                "event": "complete",
                "indexed": progress.indexed.load(std::sync::atomic::Ordering::Acquire),
                "total_chunks": progress.total_chunks.load(std::sync::atomic::Ordering::Acquire),
                "errors": progress.errors.load(std::sync::atomic::Ordering::Acquire),
                "elapsed_ms": started.elapsed().as_millis() as u64,
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
