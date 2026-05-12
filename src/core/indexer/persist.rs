//! Persistence + snapshot helpers for [`CodeIndexer`].
//!
//! Why: snapshotting the chunk corpus and HNSW graph to disk is a self-contained
//! concern (issue #85). Lifting it out of the god `impl CodeIndexer` block keeps
//! the search/ingest paths focused on their own state mutation logic.
//! What: holds `save_chunks_to_disk`, `load_chunks_from_disk`, `save_vector_store`,
//! `set_store`, the BM25 document-text helper, and the coalesced background
//! persist task (`spawn_incremental_persist`).
//! Test: covered by `test_save_chunks_roundtrip`, `test_load_chunks_missing_file_returns_zero`,
//! and `test_persist_coalesces_concurrent_calls` in `indexer::tests`.

use std::sync::atomic::Ordering;
use std::sync::Arc;

use anyhow::{Context, Result};

use crate::core::chunker::RawChunk;
use crate::core::entity::RawEntity;
use crate::core::store::VectorStore;

use super::{ChunkSnapshot, CodeIndexer};

impl CodeIndexer {
    /// Snapshot the in-memory chunk corpus + entities to disk as JSON.
    ///
    /// Why (issue #85): on graceful shutdown (and incrementally after each
    /// committed batch) we persist the corpus so a restart can rebuild BM25
    /// and the symbol graph without re-parsing the source tree. Pairs with
    /// [`VectorStore::save_to`] which persists the HNSW vectors.
    /// What: copies chunks + entities under read locks (releasing them before
    /// the I/O), then writes JSON atomically via tmp + rename. Empty corpus
    /// is still written so the on-disk file accurately reflects state.
    /// Test: see `tests::test_save_chunks_roundtrip`.
    pub async fn save_chunks_to_disk(&self, path: &std::path::Path) -> Result<()> {
        // Snapshot under read locks, then drop them before doing I/O so
        // concurrent searches never block on the JSON serialize.
        let chunks_vec: Vec<RawChunk> = {
            let chunks = self.chunks.read().await;
            chunks.values().cloned().collect()
        };
        let entities_vec: Vec<(String, Vec<RawEntity>)> = {
            let entities = self.entities.read().await;
            entities
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect()
        };
        let snapshot = ChunkSnapshot {
            version: 1,
            chunks: chunks_vec,
            entities: entities_vec,
        };
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create parent of {}", path.display()))?;
        }
        let tmp = path.with_extension("json.tmp");
        let bytes = serde_json::to_vec(&snapshot).context("serialize chunk corpus snapshot")?;
        std::fs::write(&tmp, &bytes).with_context(|| format!("write {}", tmp.display()))?;
        std::fs::rename(&tmp, path).with_context(|| format!("rename to {}", path.display()))?;
        Ok(())
    }

    /// Restore the chunk corpus + entities from a previous snapshot. After
    /// load, rebuilds BM25 + the symbol graph so the search pipeline is
    /// immediately usable. The HNSW vectors must be restored separately via
    /// `UsearchStore::load_from` before this is called.
    ///
    /// Why (issue #85): the daemon's `restore_indexes` startup hook calls
    /// this so registered indexes come back warm without re-embedding.
    /// What: reads the JSON snapshot, repopulates `chunks` + `entities`,
    /// runs `commit_bm25_batch` against the restored chunks to refill the
    /// posting list, then rebuilds the symbol graph. Returns the number of
    /// chunks restored. Missing/corrupt file → `Ok(0)` (graceful fallback).
    /// Test: see `tests::test_save_chunks_roundtrip`.
    pub async fn load_chunks_from_disk(&self, path: &std::path::Path) -> Result<usize> {
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(0),
            Err(e) => return Err(e).with_context(|| format!("read {}", path.display())),
        };
        let snapshot: ChunkSnapshot = match serde_json::from_slice(&bytes) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(
                    "chunk snapshot at {} is corrupt ({e}) — starting with empty corpus",
                    path.display()
                );
                return Ok(0);
            }
        };

        let total = snapshot.chunks.len();
        // Phase 1: refill BM25 from the restored corpus before publishing the
        // chunks map so concurrent reads can't observe a half-state.
        {
            let mut bm25 = self.bm25.write().await;
            for chunk in &snapshot.chunks {
                let text = Self::bm25_doc_text(chunk);
                bm25.upsert_document(&chunk.id, &text);
            }
        }
        // Phase 2: publish chunks under a single write lock.
        {
            let mut corpus = self.chunks.write().await;
            for chunk in snapshot.chunks {
                corpus.insert(chunk.id.clone(), chunk);
            }
        }
        // Phase 3: publish entities.
        {
            let mut emap = self.entities.write().await;
            for (file, ents) in snapshot.entities {
                emap.insert(file, ents);
            }
        }
        // Phase 4: rebuild the symbol graph so KG expansion works on the
        // restored corpus immediately. Cheap relative to re-embedding.
        self.rebuild_symbol_graph().await;
        tracing::info!(
            "restored {} chunks for index '{}' from {}",
            total,
            self.index_id,
            path.display()
        );
        Ok(total)
    }

    /// Snapshot the HNSW vector store, if one is wired. Best-effort: returns
    /// `Ok(false)` if no store is attached (BM25-only mode) so callers can
    /// chain without checking.
    pub async fn save_vector_store(&self, path: &std::path::Path) -> Result<bool> {
        let Some(store) = &self.store else {
            return Ok(false);
        };
        store.save_to(path).await?;
        Ok(true)
    }

    /// Install a pre-loaded `VectorStore` (typically a restored `UsearchStore`)
    /// onto this indexer. Used by the warm-boot path so the persisted HNSW
    /// graph is wired in before `load_chunks_from_disk` runs.
    pub fn set_store(&mut self, store: Arc<dyn VectorStore>) {
        self.store = Some(store);
    }

    /// Compose the BM25 document text for a chunk: body + virtual_terms,
    /// matching the layout the per-query rebuild used to construct.
    pub(super) fn bm25_doc_text(chunk: &RawChunk) -> String {
        if chunk.virtual_terms.is_empty() {
            chunk.content.clone()
        } else {
            let mut s = String::with_capacity(
                chunk.content.len()
                    + chunk
                        .virtual_terms
                        .iter()
                        .map(|t| t.len() + 1)
                        .sum::<usize>(),
            );
            s.push_str(&chunk.content);
            for t in &chunk.virtual_terms {
                s.push(' ');
                s.push_str(t);
            }
            s
        }
    }

    /// Spawn a background task that snapshots the HNSW graph + chunk corpus
    /// for this index to disk. Best-effort: a failure is logged but never
    /// returned to the caller — persistence is a "backup", not the source of
    /// truth, so a partial save can't corrupt live state.
    ///
    /// Why: called from `commit_parsed_batch` so incremental progress is
    /// preserved across crashes. The actual save runs on a detached task so
    /// the commit path returns immediately.
    /// What: skips when the daemon's data dir is unresolvable (tests, broken
    /// HOME env). Snapshots HNSW (via `VectorStore::save_to`) and chunks (via
    /// `save_chunks_to_disk`) concurrently with regular search traffic — both
    /// snapshot under read locks before doing I/O.
    /// Test: covered by integration tests that mutate an index then assert
    /// the on-disk file appears within a short timeout.
    pub(super) fn spawn_incremental_persist(&self) {
        // Memory-explosion fix: coalesce concurrent calls so at most ONE
        // persist task is alive per index. Each task allocates ~1× the corpus
        // footprint (clone all RawChunks + serialize to JSON bytes); without
        // this guard, a 600-batch reindex stacked 600 such tasks and the
        // daemon was OOM-killed at 46–174 GB RSS.
        //
        // Protocol:
        //   1. Every caller sets `dirty = true` (publishes "there is new
        //      state worth persisting").
        //   2. Every caller try-acquires `in_flight` via CAS false→true.
        //      On failure (a task is already running), the caller returns
        //      immediately — the in-flight task will see `dirty` when it
        //      finishes its current snapshot and loop once more.
        //   3. The winning caller spawns the persist task, which loops:
        //      clear `dirty`, snapshot+save, then check `dirty` again.
        //      When `dirty` is still false after a snapshot, release
        //      `in_flight` and exit.
        self.persist_state.dirty.store(true, Ordering::Release);
        if self
            .persist_state
            .in_flight
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            // Another task is already running and will pick up the new state
            // via the `dirty` flag we just set.
            return;
        }

        let index_id = self.index_id.clone();
        let store = self.store.clone();
        let chunks = self.chunks.clone();
        let entities = self.entities.clone();
        let persist_state = self.persist_state.clone();
        tokio::spawn(async move {
            // Re-resolve paths in the task so the persistence layer's path
            // resolution failures don't crash the commit caller.
            let chunks_path = match crate::service::persistence::chunks_path(&index_id) {
                Ok(p) => p,
                Err(e) => {
                    tracing::debug!(
                        "incremental persist: cannot resolve chunks path for '{index_id}': {e}"
                    );
                    persist_state.in_flight.store(false, Ordering::Release);
                    return;
                }
            };
            let hnsw_path = match crate::service::persistence::hnsw_path(&index_id) {
                Ok(p) => p,
                Err(e) => {
                    tracing::debug!(
                        "incremental persist: cannot resolve hnsw path for '{index_id}': {e}"
                    );
                    persist_state.in_flight.store(false, Ordering::Release);
                    return;
                }
            };

            // Coalescing loop: snapshot+save while `dirty` keeps being set.
            // Bound the loop so a pathological caller can't pin us forever
            // (each iteration is bounded by I/O latency, but we also cap at
            // a small constant to ensure forward progress on the reindex
            // hot loop's behalf).
            const MAX_COALESCED_ITERATIONS: u32 = 8;
            for _ in 0..MAX_COALESCED_ITERATIONS {
                // Clear `dirty` *before* snapshotting so any commit that
                // races in after we start reading is guaranteed to set it
                // again — ensuring we don't miss it.
                persist_state.dirty.store(false, Ordering::Release);

                // Save HNSW first (large, parallel-friendly).
                if let Some(store) = &store {
                    if let Err(e) = store.save_to(&hnsw_path).await {
                        tracing::warn!(
                            "incremental persist: failed to save HNSW for '{index_id}': {e}"
                        );
                    }
                }

                // Snapshot chunks + entities under read locks. We scope the
                // clones tightly so the Vec<RawChunk> is dropped before the
                // next loop iteration; serde_json::to_vec is run inside a
                // spawn_blocking so the ~hundreds-of-MB JSON build doesn't
                // block a runtime worker thread.
                let chunks_vec: Vec<RawChunk> = {
                    let g = chunks.read().await;
                    g.values().cloned().collect()
                };
                let entities_vec: Vec<(String, Vec<RawEntity>)> = {
                    let g = entities.read().await;
                    g.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
                };
                let snapshot = ChunkSnapshot {
                    version: 1,
                    chunks: chunks_vec,
                    entities: entities_vec,
                };
                if let Some(parent) = chunks_path.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                let tmp = chunks_path.with_extension("json.tmp");
                let chunks_path_inner = chunks_path.clone();
                let index_id_inner = index_id.clone();
                // Serialize + write on a blocking worker so we don't pin a
                // runtime worker for hundreds of ms on large corpora. Move
                // `snapshot` in so it's dropped on the blocking thread
                // immediately after `to_vec` returns — the peak allocation
                // is `snapshot + bytes` for the duration of `to_vec`, not
                // `snapshot + bytes` for the full file write.
                let join = tokio::task::spawn_blocking(move || -> std::io::Result<()> {
                    let bytes = match serde_json::to_vec(&snapshot) {
                        Ok(b) => b,
                        Err(e) => {
                            tracing::warn!(
                                "incremental persist: serialize chunks failed for \
                                 '{index_id_inner}': {e}"
                            );
                            return Ok(()); // non-fatal
                        }
                    };
                    // Drop `snapshot` explicitly — we no longer need the
                    // cloned Vec<RawChunk> now that `bytes` holds the
                    // serialized form. This is the single biggest peak-RAM
                    // savings: without the drop, both `snapshot` (clones)
                    // and `bytes` (JSON) live simultaneously.
                    // (Implicit drop at end of `to_vec` call — `snapshot`
                    // is moved into `to_vec` then dropped at the call
                    // boundary, so it's already gone here.)
                    std::fs::write(&tmp, &bytes)?;
                    std::fs::rename(&tmp, &chunks_path_inner)?;
                    Ok(())
                })
                .await;
                match join {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => {
                        tracing::warn!("incremental persist: I/O failed for '{index_id}': {e}")
                    }
                    Err(e) => tracing::warn!(
                        "incremental persist: blocking task panicked for '{index_id}': {e}"
                    ),
                }

                // If no new commits arrived during the snapshot, we're
                // done. Release in_flight under Release ordering so the
                // next caller's CAS sees the cleared state.
                if !persist_state.dirty.load(Ordering::Acquire) {
                    persist_state.in_flight.store(false, Ordering::Release);
                    return;
                }
                // Otherwise loop: another commit landed while we were
                // saving, so its state needs flushing too.
            }
            // Hit the iteration cap. Drop in_flight so future commits can
            // start a fresh persist; we logged a debug above per iteration.
            tracing::debug!(
                "incremental persist: coalesce cap reached for '{index_id}' \
                 (more commits arriving than we can flush)"
            );
            persist_state.in_flight.store(false, Ordering::Release);
        });
    }
}
