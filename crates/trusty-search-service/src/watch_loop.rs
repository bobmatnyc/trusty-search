//! Glue between [`crate::watcher::FileWatcher`] and `CodeIndexer`.
//!
//! Why: The watcher emits raw filesystem events; the indexer wants
//! `index_file` / `remove_chunk` calls. This module bridges them and
//! maintains an [`IndexedFiles`] side-map so that file deletions can locate
//! the chunk IDs that need to come out of the HNSW + corpus.
//!
//! What: [`spawn_watch_loop`] starts the [`FileWatcher`] and a long-running
//! tokio task that consumes events. Returns a `WatcherTask` handle that owns
//! both the `FileWatcher` (so dropping it stops the OS watcher) and the
//! `JoinHandle` of the consumer task.
//!
//! Test: integration test below boots the loop on a temp dir, writes a file,
//! and asserts the indexer's `chunk_count()` increases.

use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use tokio::sync::mpsc;
use tokio::sync::RwLock;
use tokio::task::JoinHandle;
use trusty_search_core::chunker::chunk_ast;
use trusty_search_core::CodeIndexer;

use crate::indexed_files::IndexedFiles;
use crate::watcher::{FileWatcher, WatchEvent};

/// Handle for a running watch loop. Drop it to stop watching and join the
/// consumer task on the next `await` boundary.
pub struct WatcherTask {
    _watcher: FileWatcher,
    _join: JoinHandle<()>,
}

/// Start watching `root_path` and forward changes into `indexer`.
///
/// `indexed_files` is shared with anyone else who needs to know which chunks
/// belong to which path (e.g. an explicit `remove_file` HTTP handler).
pub fn spawn_watch_loop(
    root_path: &Path,
    indexer: Arc<RwLock<CodeIndexer>>,
    indexed_files: IndexedFiles,
) -> Result<WatcherTask> {
    let (tx, mut rx) = mpsc::unbounded_channel::<WatchEvent>();
    let watcher = FileWatcher::start(root_path.to_path_buf(), tx)?;

    let join = tokio::spawn(async move {
        while let Some(event) = rx.recv().await {
            match event {
                WatchEvent::Modified(path) => {
                    handle_modified(&path, &indexer, &indexed_files).await;
                }
                WatchEvent::Removed(path) => {
                    handle_removed(&path, &indexer, &indexed_files).await;
                }
            }
        }
    });

    Ok(WatcherTask {
        _watcher: watcher,
        _join: join,
    })
}

/// Re-chunk the file and merge it into the indexer. Stale chunks from a
/// previous version of the same file are removed first so we don't accumulate
/// dead entries on edit.
async fn handle_modified(
    path: &Path,
    indexer: &Arc<RwLock<CodeIndexer>>,
    indexed_files: &IndexedFiles,
) {
    // Skip directories — the watcher fires on parent mtime updates too.
    if path.is_dir() {
        return;
    }

    let content = match tokio::fs::read_to_string(path).await {
        Ok(s) => s,
        Err(err) => {
            tracing::debug!(?err, ?path, "skip unreadable file");
            return;
        }
    };

    // Drop any prior chunks for this file before re-indexing.
    if let Some(stale_ids) = indexed_files.take(path).await {
        let idx = indexer.read().await;
        for id in stale_ids {
            if let Err(err) = idx.remove_chunk(&id).await {
                tracing::warn!(?err, %id, "remove_chunk failed");
            }
        }
    }

    // Compute fresh chunk IDs and feed them to the indexer.
    let path_str = path.to_string_lossy().into_owned();
    let (chunks, _entities) = chunk_ast(&path_str, &content);
    let new_ids: Vec<String> = chunks.iter().map(|c| c.id.clone()).collect();

    let idx = indexer.read().await;
    if let Err(err) = idx.index_file(&path_str, &content).await {
        tracing::warn!(?err, ?path, "index_file failed");
        return;
    }
    drop(idx);

    indexed_files.record(path.to_path_buf(), new_ids).await;
}

/// Drop every chunk we previously recorded for `path` from the indexer.
async fn handle_removed(
    path: &Path,
    indexer: &Arc<RwLock<CodeIndexer>>,
    indexed_files: &IndexedFiles,
) {
    let Some(ids) = indexed_files.take(path).await else {
        return;
    };
    let idx = indexer.read().await;
    for id in ids {
        if let Err(err) = idx.remove_chunk(&id).await {
            tracing::warn!(?err, %id, "remove_chunk failed");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tokio::sync::RwLock;

    /// End-to-end: writing a `.rs` file inside a watched directory causes the
    /// indexer's chunk count to grow within ~2s.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn modified_file_triggers_indexing() {
        let dir = tempfile::tempdir().expect("tempdir");
        let indexer = Arc::new(RwLock::new(CodeIndexer::new("test", dir.path())));
        let tracker = IndexedFiles::new();

        let _task = spawn_watch_loop(dir.path(), Arc::clone(&indexer), tracker.clone())
            .expect("watch loop starts");

        // Allow the OS watcher to install.
        tokio::time::sleep(Duration::from_millis(150)).await;

        let file = dir.path().join("lib.rs");
        tokio::fs::write(&file, "fn alpha() {}\nfn beta() {}\n")
            .await
            .expect("write file");

        // Poll up to 2s for the indexer to pick the change up.
        let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
        loop {
            let count = indexer.read().await.chunk_count();
            if count > 0 {
                break;
            }
            if tokio::time::Instant::now() > deadline {
                panic!("chunk_count never grew above 0");
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        assert!(tracker.len().await >= 1, "expected at least one tracked file");
    }
}
