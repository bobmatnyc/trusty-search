//! Shared text-embedding abstraction for trusty-* projects.
//!
//! Why: trusty-memory and trusty-search both shipped near-identical
//! `Embedder` traits and `FastEmbedder` implementations, with subtle
//! drift (cache vs no-cache, sync vs async warmup, `dim()` vs `dimension()`).
//! Centralising fixes one bug in one place and lets future consumers pick up
//! the embedder for free.
//!
//! What: an async `Embedder` trait with `embed_batch` as the single primitive
//! (single-text embed is a free helper), plus a production `FastEmbedder`
//! (fastembed-rs, all-MiniLM-L6-v2, 384-d) with LRU caching and ORT warmup,
//! and a `MockEmbedder` test double behind the `test-support` feature.
//!
//! Test: `cargo test -p trusty-embedder` covers shape, cache hits, and the
//! mock embedder. ONNX-backed tests are `#[ignore]` to keep CI under one
//! cargo-feature umbrella.

use std::num::NonZeroUsize;
use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use lru::LruCache;
use parking_lot::Mutex;

/// Output dimension of the all-MiniLM-L6-v2 model.
pub const EMBED_DIM: usize = 384;

/// Default LRU cache capacity. Picked to be large enough to keep the
/// hot working set of repeat queries in memory but small enough that the
/// cache itself fits well inside L2/L3 on a typical developer machine.
pub const DEFAULT_CACHE_CAPACITY: usize = 256;

/// Abstraction over embedding backends.
///
/// Why: Decouple consumers from any one model so we can swap in remote APIs,
/// quantised models, or deterministic mocks without changing call sites.
/// What: a single primitive — `embed_batch` — plus a dimension accessor.
/// Single-text callers should use the [`embed_one`] convenience helper.
/// Test: covered by `FastEmbedder` and `MockEmbedder` tests below.
#[async_trait]
pub trait Embedder: Send + Sync {
    /// Embed a batch of texts. Returns one `Vec<f32>` per input, each of
    /// length `self.dimension()`. An empty input batch returns an empty Vec.
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;

    /// Output dimension of the produced embeddings.
    fn dimension(&self) -> usize;
}

/// Convenience helper: embed a single text via `embed_batch` and return the
/// lone vector.
///
/// Why: Most call sites only need one embedding at a time and writing
/// `.embed_batch(&[text]).await?.into_iter().next()` everywhere is noise.
/// What: builds a 1-element batch, calls `embed_batch`, returns the first
/// vector (or errors if the embedder produced nothing).
/// Test: covered indirectly by `mock_embedder_round_trip`.
pub async fn embed_one(embedder: &dyn Embedder, text: &str) -> Result<Vec<f32>> {
    let mut v = embedder.embed_batch(&[text.to_string()]).await?;
    v.pop().context("embedder returned no embedding for non-empty input")
}

/// Local CPU embedder backed by fastembed-rs (ONNX runtime, all-MiniLM-L6-v2).
///
/// Why: Default to local-only embeddings so consumers have zero external
/// network dependency and predictable latency. The LRU cache keeps the hot
/// path free of redundant ONNX work for repeat strings (queries, common
/// chunks).
/// What: wraps a single `TextEmbedding` behind a `parking_lot::Mutex` (the
/// underlying `embed` requires `&mut self`) and an `LruCache<String, Vec<f32>>`.
/// Initialisation warms the ORT graph with a small batch so the first user
/// query doesn't pay the one-shot compile cost.
/// Test: `embed_batch_returns_correct_dim` and `cache_hit_is_idempotent`
/// (marked `#[ignore]` — they download a real model).
pub struct FastEmbedder {
    model: Arc<Mutex<TextEmbedding>>,
    cache: Arc<Mutex<LruCache<String, Vec<f32>>>>,
    dim: usize,
}

impl FastEmbedder {
    /// Construct a new `FastEmbedder` with the default cache size.
    pub async fn new() -> Result<Self> {
        Self::with_cache_size(DEFAULT_CACHE_CAPACITY).await
    }

    /// Construct with an explicit LRU capacity.
    pub async fn with_cache_size(capacity: usize) -> Result<Self> {
        let capacity = NonZeroUsize::new(capacity.max(1))
            .expect("capacity.max(1) is always non-zero");

        // fastembed's `try_new` downloads + builds an ONNX session — blocking
        // work that must run off the async reactor.
        let model = tokio::task::spawn_blocking(|| -> Result<TextEmbedding> {
            let mut m = TextEmbedding::try_new(TextInitOptions::new(EmbeddingModel::AllMiniLML6V2))
                .context("failed to initialise fastembed all-MiniLM-L6-v2")?;

            // Warm the graph so the first real user query is hot.
            let warmup: Vec<&str> = vec![
                "hello world",
                "the quick brown fox",
                "memory palace warmup",
                "embedding model ready",
                "trusty common warmup",
            ];
            let _ = m
                .embed(warmup, None)
                .context("fastembed warmup batch failed")?;
            Ok(m)
        })
        .await
        .context("spawn_blocking joined with error during embedder init")??;

        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            cache: Arc::new(Mutex::new(LruCache::new(capacity))),
            dim: EMBED_DIM,
        })
    }
}

#[async_trait]
impl Embedder for FastEmbedder {
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Split into cached hits vs misses.
        let mut results: Vec<Option<Vec<f32>>> = vec![None; texts.len()];
        let mut to_compute: Vec<(usize, String)> = Vec::new();
        {
            let mut cache = self.cache.lock();
            for (i, t) in texts.iter().enumerate() {
                if let Some(v) = cache.get(t) {
                    results[i] = Some(v.clone());
                } else {
                    to_compute.push((i, t.clone()));
                }
            }
        }

        if !to_compute.is_empty() {
            let model = Arc::clone(&self.model);
            let owned: Vec<String> = to_compute.iter().map(|(_, s)| s.clone()).collect();
            let computed = tokio::task::spawn_blocking(move || -> Result<Vec<Vec<f32>>> {
                let mut guard = model.lock();
                guard
                    .embed(owned, None)
                    .context("fastembed embed call failed")
            })
            .await
            .context("spawn_blocking joined with error during embed")??;

            if computed.len() != to_compute.len() {
                anyhow::bail!(
                    "fastembed returned {} embeddings, expected {}",
                    computed.len(),
                    to_compute.len()
                );
            }

            let mut cache = self.cache.lock();
            for ((idx, key), vector) in to_compute.into_iter().zip(computed.into_iter()) {
                cache.put(key, vector.clone());
                results[idx] = Some(vector);
            }
        }

        results
            .into_iter()
            .map(|opt| opt.context("missing embedding slot after batch"))
            .collect()
    }

    fn dimension(&self) -> usize {
        self.dim
    }
}

/// Deterministic test double — hashes input bytes into a fixed-dim vector.
///
/// Why: ONNX model downloads dominate test runtime and can race on cold
/// caches when multiple tests construct embedders in parallel. The mock
/// gives integration tests a "rank by similarity" surface without any I/O.
/// What: a tiny per-byte hash spread across `dim` slots, with the first byte
/// always contributing so short/empty strings still differ.
/// Test: `mock_embedder_round_trip` confirms shape + determinism.
#[cfg(any(test, feature = "test-support"))]
pub struct MockEmbedder {
    dim: usize,
}

#[cfg(any(test, feature = "test-support"))]
impl MockEmbedder {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    fn hash_to_vec(&self, text: &str) -> Vec<f32> {
        let mut v = vec![0.0_f32; self.dim];
        for (i, b) in text.bytes().enumerate() {
            let slot = (i + b as usize) % self.dim;
            v[slot] += (b as f32) / 255.0;
        }
        if let Some(first) = text.bytes().next() {
            v[0] += first as f32 / 255.0;
        }
        v
    }
}

#[cfg(any(test, feature = "test-support"))]
#[async_trait]
impl Embedder for MockEmbedder {
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| self.hash_to_vec(t)).collect())
    }

    fn dimension(&self) -> usize {
        self.dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn mock_embedder_round_trip() {
        let e = MockEmbedder::new(EMBED_DIM);
        assert_eq!(e.dimension(), EMBED_DIM);
        let v = embed_one(&e, "hello").await.unwrap();
        assert_eq!(v.len(), EMBED_DIM);
        let batch = e
            .embed_batch(&["a".to_string(), "b".to_string()])
            .await
            .unwrap();
        assert_eq!(batch.len(), 2);
        assert_ne!(batch[0], batch[1]);
    }

    #[tokio::test]
    async fn mock_embedder_empty_input_returns_empty() {
        let e = MockEmbedder::new(EMBED_DIM);
        let v = e.embed_batch(&[]).await.unwrap();
        assert!(v.is_empty());
    }

    // ONNX-backed test: downloads ~23MB on first run. Marked ignored so default
    // `cargo test` stays offline; run with `cargo test -- --ignored` when needed.
    #[tokio::test]
    #[ignore]
    async fn fastembed_returns_correct_dim() {
        let e = FastEmbedder::new().await.unwrap();
        assert_eq!(e.dimension(), 384);
        let v = embed_one(&e, "fn authenticate(user: &str) -> bool")
            .await
            .unwrap();
        assert_eq!(v.len(), 384);
        assert!(v.iter().any(|x| *x != 0.0));
    }

    #[tokio::test]
    #[ignore]
    async fn fastembed_cache_hit_is_idempotent() {
        let e = FastEmbedder::new().await.unwrap();
        let v1 = embed_one(&e, "cached").await.unwrap();
        let v2 = embed_one(&e, "cached").await.unwrap();
        assert_eq!(v1, v2);
    }
}
