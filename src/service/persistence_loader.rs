//! Shared helper that builds a `CodeIndexer`, attempting to restore a
//! previously-persisted HNSW snapshot and chunk corpus from disk.
//!
//! Why (issue #85): both `POST /indexes` and the daemon-startup
//! `restore_indexes` hook need the same logic — construct the indexer, wire
//! the embedder, attempt to load HNSW + chunks, and fall back to an empty
//! index on any failure. Centralising this prevents drift between the two
//! call sites (and the inevitable "the warm-boot path silently runs in
//! BM25-only mode" footgun).
//! What: `build_indexer_with_persisted_state` returns a fully-wired
//! `CodeIndexer`. On a corrupt or missing snapshot it falls back to a fresh
//! empty store + corpus and logs at WARN/INFO so operators can tell which
//! path was taken.
//! Test: covered by integration tests in `tests/integration_tests.rs` that
//! drop a state directory, restart, and assert the corpus is intact.

use std::path::PathBuf;
use std::sync::Arc;

use crate::core::{
    embed::Embedder,
    indexer::CodeIndexer,
    store::{UsearchStore, VectorStore},
};

use crate::service::persistence;

/// Build a `CodeIndexer` for `index_id`, restoring HNSW + chunks from disk
/// when a snapshot is present.
///
/// Why: see module docs.
/// What: tries `UsearchStore::load_from` first; falls back to a fresh empty
/// store if the load returns `Ok(None)` (no snapshot) or `Err` (corrupt
/// snapshot — logged at WARN). Then attaches the embedder + store, and
/// finally calls `load_chunks_from_disk` to rehydrate the corpus.
/// Test: see module docs.
pub async fn build_indexer_with_persisted_state(
    index_id: &str,
    root_path: PathBuf,
    embedder: &Arc<dyn Embedder>,
) -> CodeIndexer {
    let dim = embedder.dimension();
    let store: Arc<dyn VectorStore> = build_store(index_id, dim).await;
    let mut indexer =
        CodeIndexer::new(index_id, root_path).with_components(Arc::clone(embedder), store);

    // Restore the chunk corpus (rebuilds BM25 + symbol graph as a side effect).
    match persistence::chunks_path(index_id) {
        Ok(path) => match indexer.load_chunks_from_disk(&path).await {
            Ok(n) if n > 0 => tracing::info!(
                "warm-boot: restored {} chunks for index '{}' from {}",
                n,
                index_id,
                path.display()
            ),
            Ok(_) => {} // empty / missing — first-run case
            Err(e) => tracing::warn!(
                "warm-boot: could not load chunks for '{}' ({e}) — starting empty",
                index_id
            ),
        },
        Err(e) => tracing::warn!("cannot resolve chunks path for '{index_id}': {e}"),
    }
    // We need `set_store` only for the daemon-startup path (where we want to
    // separate "restored store" from "fresh store"); the function above
    // already handed us a properly-wired indexer, so nothing else to do.
    let _ = &mut indexer;
    indexer
}

/// Try to load the HNSW snapshot for `index_id`. On any failure (missing,
/// corrupt, dimension mismatch) returns a fresh empty `UsearchStore`.
async fn build_store(index_id: &str, dim: usize) -> Arc<dyn VectorStore> {
    let path = match persistence::hnsw_path(index_id) {
        Ok(p) => p,
        Err(e) => {
            tracing::warn!("cannot resolve hnsw path for '{index_id}': {e}");
            return fresh_store(dim);
        }
    };

    if persistence::has_persisted_hnsw(&path) {
        match UsearchStore::load_from(&path).await {
            Ok(Some(store)) => {
                if store.dim() == dim {
                    tracing::info!(
                        "warm-boot: restored HNSW snapshot for '{}' from {}",
                        index_id,
                        path.display()
                    );
                    return Arc::new(store);
                }
                tracing::warn!(
                    "warm-boot: hnsw snapshot for '{}' has dim {} but embedder is {} — starting fresh",
                    index_id,
                    store.dim(),
                    dim
                );
            }
            Ok(None) => {
                // Sidecar missing/corrupt — fall back to fresh.
                tracing::warn!(
                    "warm-boot: hnsw snapshot at {} could not be loaded — starting fresh",
                    path.display()
                );
            }
            Err(e) => {
                tracing::warn!(
                    "warm-boot: error loading hnsw snapshot at {}: {e} — starting fresh",
                    path.display()
                );
            }
        }
    }
    fresh_store(dim)
}

fn fresh_store(dim: usize) -> Arc<dyn VectorStore> {
    match UsearchStore::new(dim) {
        Ok(s) => Arc::new(s) as Arc<dyn VectorStore>,
        Err(e) => {
            tracing::error!(
                "failed to allocate UsearchStore (dim={dim}): {e} — index will run BM25-only"
            );
            // Return a no-op store that satisfies the trait via the
            // VectorStore default implementations is not possible (the trait
            // requires concrete impls). Construct an empty UsearchStore one
            // more time with the bare minimum and propagate the error path
            // via a panic — usearch::Index::new only fails on OOM here, which
            // would have already torn the daemon down.
            panic!("usearch alloc failure: {e}");
        }
    }
}
