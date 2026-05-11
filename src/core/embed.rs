//! Embedding abstraction — thin facade over the shared `trusty-embedder` crate.
//!
//! Why: The `Embedder` trait + `FastEmbedder` + `MockEmbedder` previously
//! lived in this crate. They've been moved to the shared `trusty-embedder`
//! crate so trusty-memory and trusty-search ship the same implementation
//! (LRU cache, ORT warmup, deterministic mock). This module keeps the
//! existing in-crate `Embedder` trait shape (`embed(&str)` + `embed_batch(&[&str])`)
//! so the rest of trusty-search compiles unchanged.
//! What: A local `Embedder` trait that mirrors the historic API, plus a
//! blanket-impl adapter that delegates to the shared `trusty_embedder::Embedder`.
//! `FastEmbedder` and `MockEmbedder` are re-exports.
//! Test: existing indexer / concept_cluster tests exercise this surface;
//! shared-crate behaviour is covered upstream in `trusty-embedder`.

use anyhow::Result;
use async_trait::async_trait;

pub use trusty_embedder::{FastEmbedder, EMBED_DIM};

#[cfg(any(test, feature = "test-support"))]
pub use trusty_embedder::MockEmbedder;

/// trusty-search-flavoured embedder trait.
///
/// Why: Historic call sites pass `&str` / `&[&str]` directly. The shared
/// `trusty_embedder::Embedder` settled on `&[String]` as its primitive (it
/// owns the LRU cache key, so it needs owned strings anyway). This trait
/// preserves the old surface — every `&str` is cloned into a `String` on
/// the way down, which matches what the old per-call code did internally.
/// What: an async `embed(&str) -> Vec<f32>` and `embed_batch(&[&str]) ->
/// Vec<Vec<f32>>`, plus `dimension()`.
/// Test: covered indirectly via every `CodeIndexer` test that runs against
/// `MockEmbedder`.
#[async_trait]
pub trait Embedder: Send + Sync {
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
    fn dimension(&self) -> usize;
}

/// Adapter: every shared `trusty_embedder::Embedder` automatically implements
/// the in-crate trait via owned-string conversion.
#[async_trait]
impl<E> Embedder for E
where
    E: trusty_embedder::Embedder,
{
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        trusty_embedder::embed_one(self, text).await
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let owned: Vec<String> = texts.iter().map(|s| (*s).to_owned()).collect();
        trusty_embedder::Embedder::embed_batch(self, &owned).await
    }

    fn dimension(&self) -> usize {
        trusty_embedder::Embedder::dimension(self)
    }
}
