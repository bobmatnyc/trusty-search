use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use async_trait::async_trait;
use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use lru::LruCache;

/// Embedding dimensionality for the all-MiniLM-L6-v2 model.
const EMBED_DIM: usize = 384;

/// LRU cache capacity (entries). 256 keeps recently-issued query/text embeddings hot
/// so repeat searches skip the ONNX call entirely.
const CACHE_CAPACITY: usize = 256;

#[async_trait]
pub trait Embedder: Send + Sync {
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
    fn dimension(&self) -> usize;
}

/// FastEmbedder: fastembed-rs ONNX runtime (all-MiniLM-L6-v2, 384-dim, SIMD-accelerated).
///
/// Why: provides local embeddings with no API key requirement, AVX2/NEON-accelerated,
/// suitable for the hot path of `search_code` queries and chunk indexing.
/// What: wraps fastembed `TextEmbedding` behind a `Mutex` (its `embed` takes `&mut self`)
/// and adds an LRU cache to bypass the ONNX call for repeated inputs.
/// Test: see the `tests` module — covers dimensions, batch behavior, and cache hits.
pub struct FastEmbedder {
    model: Arc<Mutex<TextEmbedding>>,
    cache: Arc<Mutex<LruCache<String, Vec<f32>>>>,
    dim: usize,
}

impl FastEmbedder {
    /// Initialize the embedder, downloading the model on first run (~23MB into
    /// the fastembed cache directory under `~/.cache/`).
    pub async fn new() -> Result<Self> {
        // fastembed's `try_new` is blocking (downloads + ONNX session init),
        // so run it on the blocking pool to keep the async runtime responsive.
        let model = tokio::task::spawn_blocking(|| {
            TextEmbedding::try_new(TextInitOptions::new(EmbeddingModel::AllMiniLML6V2))
        })
        .await
        .context("fastembed init task panicked")?
        .context("failed to initialize fastembed TextEmbedding")?;

        let capacity =
            NonZeroUsize::new(CACHE_CAPACITY).expect("CACHE_CAPACITY must be non-zero");

        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            cache: Arc::new(Mutex::new(LruCache::new(capacity))),
            dim: EMBED_DIM,
        })
    }
}

#[async_trait]
impl Embedder for FastEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        // Cache check
        if let Some(cached) = self.cache.lock().expect("cache mutex poisoned").get(text) {
            return Ok(cached.clone());
        }

        let model = Arc::clone(&self.model);
        let owned = text.to_owned();
        let owned_for_compute = owned.clone();

        let mut vectors = tokio::task::spawn_blocking(move || -> Result<Vec<Vec<f32>>> {
            let mut guard = model.lock().expect("embedder mutex poisoned");
            guard
                .embed(vec![owned_for_compute], None)
                .context("fastembed embed failed")
        })
        .await
        .context("fastembed embed task panicked")??;

        let vector = vectors.pop().context("fastembed returned no embedding")?;

        self.cache
            .lock()
            .expect("cache mutex poisoned")
            .put(owned, vector.clone());

        Ok(vector)
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        // Pre-allocate result slots and identify which texts need computation.
        let mut results: Vec<Option<Vec<f32>>> = vec![None; texts.len()];
        let mut to_compute: Vec<(usize, String)> = Vec::new();

        {
            let mut cache = self.cache.lock().expect("cache mutex poisoned");
            for (i, t) in texts.iter().enumerate() {
                if let Some(v) = cache.get(*t) {
                    results[i] = Some(v.clone());
                } else {
                    to_compute.push((i, (*t).to_owned()));
                }
            }
        }

        if !to_compute.is_empty() {
            let owned_texts: Vec<String> = to_compute.iter().map(|(_, s)| s.clone()).collect();
            let model = Arc::clone(&self.model);

            let computed = tokio::task::spawn_blocking(move || -> Result<Vec<Vec<f32>>> {
                let mut guard = model.lock().expect("embedder mutex poisoned");
                guard
                    .embed(owned_texts, None)
                    .context("fastembed batch embed failed")
            })
            .await
            .context("fastembed embed_batch task panicked")??;

            if computed.len() != to_compute.len() {
                anyhow::bail!(
                    "fastembed returned {} embeddings, expected {}",
                    computed.len(),
                    to_compute.len()
                );
            }

            let mut cache = self.cache.lock().expect("cache mutex poisoned");
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_embed_returns_384_dims() {
        // This will download the model on first run (~23MB)
        let embedder = FastEmbedder::new().await.expect("embedder init");
        let v = embedder
            .embed("fn authenticate(user: &str) -> bool")
            .await
            .expect("embed");
        assert_eq!(v.len(), 384);
        // Embedding should be non-zero
        assert!(v.iter().any(|&x| x != 0.0));
    }

    #[tokio::test]
    async fn test_embed_batch() {
        let embedder = FastEmbedder::new().await.expect("embedder init");
        let texts = vec!["hello world", "fn main() {}", "struct Foo;"];
        let vecs = embedder.embed_batch(&texts).await.expect("batch embed");
        assert_eq!(vecs.len(), 3);
        assert!(vecs.iter().all(|v| v.len() == 384));
    }

    #[tokio::test]
    async fn test_lru_cache_hit() {
        let embedder = FastEmbedder::new().await.expect("embedder init");
        let text = "cached query";
        let v1 = embedder.embed(text).await.expect("embed 1");
        let v2 = embedder.embed(text).await.expect("embed 2 (cache hit)");
        assert_eq!(v1, v2);
    }
}
