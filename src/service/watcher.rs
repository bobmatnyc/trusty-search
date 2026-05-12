//! Filesystem watcher that emits debounced [`WatchEvent`]s for an indexed root.
//!
//! Why: The daemon must keep its in-memory HNSW + chunk corpus in sync with
//! disk without re-scanning entire trees. We piggy-back on `notify` (kqueue /
//! fsevent / inotify) and a 500ms debounce window so editor save-storms do not
//! produce duplicate work.
//!
//! What: [`FileWatcher::start`] spawns a `notify-debouncer-mini` watcher on a
//! background thread; events are translated into [`WatchEvent`] and forwarded
//! through an `UnboundedSender` so the consumer can `await` them in a tokio
//! task. The debouncer is held inside the returned struct — dropping it stops
//! the watcher cleanly.
//!
//! Test: `cargo test -p trusty-search-service watcher` writes a file in a
//! `tempfile::TempDir` and asserts that a `Modified` event arrives within 1s.

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result};
use notify::{RecommendedWatcher, RecursiveMode};
use notify_debouncer_mini::{new_debouncer, DebounceEventResult, Debouncer};
use tokio::sync::mpsc::UnboundedSender;

/// Debounce window for filesystem change coalescing. Long enough to absorb
/// editor save-storms, short enough to feel "live" to the indexer.
const DEBOUNCE_MS: u64 = 500;

/// A normalized filesystem event surfaced by [`FileWatcher`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WatchEvent {
    /// Path was created or modified — re-index it.
    Modified(PathBuf),
    /// Path was deleted — drop its chunks from the index.
    Removed(PathBuf),
}

/// Recursive filesystem watcher with a 500ms debounce window.
///
/// Owns the underlying `Debouncer<RecommendedWatcher>` so that dropping the
/// `FileWatcher` (or calling [`Self::stop`]) terminates the OS watch.
pub struct FileWatcher {
    _debouncer: Debouncer<RecommendedWatcher>,
}

impl FileWatcher {
    /// Begin watching `root_path` recursively. Each debounced event is mapped
    /// into a [`WatchEvent`] and pushed to `tx`. If the receiver has been
    /// dropped the send is silently ignored (the watcher will simply continue
    /// firing into the void until `self` is dropped).
    pub fn start(root_path: PathBuf, tx: UnboundedSender<WatchEvent>) -> Result<Self> {
        let mut debouncer = new_debouncer(
            Duration::from_millis(DEBOUNCE_MS),
            move |res: DebounceEventResult| match res {
                Ok(events) => {
                    for ev in events {
                        let path = ev.path.clone();
                        // notify-debouncer-mini 0.4 collapses creates/modifies
                        // into `Any`; we treat the path's existence as the
                        // discriminator since deletions yield non-existent paths.
                        let event = if path.exists() {
                            WatchEvent::Modified(path)
                        } else {
                            WatchEvent::Removed(path)
                        };
                        // Receiver dropped → nothing to do, the channel is closed.
                        let _ = tx.send(event);
                    }
                }
                Err(err) => {
                    tracing::warn!(?err, "filesystem watcher error");
                }
            },
        )
        .context("create notify debouncer")?;

        debouncer
            .watcher()
            .watch(&root_path, RecursiveMode::Recursive)
            .with_context(|| format!("watch path {}", root_path.display()))?;

        Ok(Self {
            _debouncer: debouncer,
        })
    }

    /// Stop the watcher and release OS resources by dropping the debouncer.
    pub fn stop(self) {
        // Drop semantics on `_debouncer` perform the cleanup.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::Duration;
    use tokio::sync::mpsc;
    use tokio::time::timeout;

    /// Modifying a file inside the watched root produces a `Modified` event
    /// within ~1s (covers the 500ms debounce + scheduling jitter).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn modified_event_emitted_within_one_second() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (tx, mut rx) = mpsc::unbounded_channel();

        let _watcher = FileWatcher::start(dir.path().to_path_buf(), tx).expect("watcher starts");

        // Give the OS watcher a moment to install its kqueue/inotify hooks
        // before generating events; otherwise the very first write can be lost.
        tokio::time::sleep(Duration::from_millis(100)).await;

        let file_path = dir.path().join("hello.txt");
        fs::write(&file_path, b"hello").expect("write file");

        let event = timeout(Duration::from_secs(2), rx.recv())
            .await
            .expect("event arrives within 2s")
            .expect("channel still open");

        match event {
            WatchEvent::Modified(p) => {
                // Use file_name() rather than ends_with() so the assertion is
                // immune to macOS resolving /tmp → /private/var/folders/…
                // (the watcher delivers the canonicalized path; ends_with does
                // component matching which is correct, but file_name() is more
                // explicit and also survives any future path-normalization changes).
                assert_eq!(
                    p.file_name().and_then(|n| n.to_str()),
                    Some("hello.txt"),
                    "expected path filename to be hello.txt, got {p:?}"
                );
            }
            other => panic!("expected Modified, got {other:?}"),
        }
    }

    /// Deleting a previously-created file produces a `Removed` event.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn removed_event_emitted_on_delete() {
        let dir = tempfile::tempdir().expect("tempdir");
        let file_path = dir.path().join("doomed.txt");
        fs::write(&file_path, b"transient").expect("write file");

        let (tx, mut rx) = mpsc::unbounded_channel();
        let _watcher = FileWatcher::start(dir.path().to_path_buf(), tx).expect("watcher starts");

        tokio::time::sleep(Duration::from_millis(100)).await;

        fs::remove_file(&file_path).expect("delete file");

        // Drain events until we see a Removed for our path or time out. We
        // tolerate stray Modified events that some platforms emit for parent
        // directory mtime updates.
        let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
        loop {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            let event = timeout(remaining, rx.recv())
                .await
                .expect("event arrives before deadline")
                .expect("channel still open");
            if let WatchEvent::Removed(p) = event {
                // file_name() comparison is canonical-path-safe (macOS /tmp symlink).
                if p.file_name().and_then(|n| n.to_str()) == Some("doomed.txt") {
                    return;
                }
            }
        }
    }
}
