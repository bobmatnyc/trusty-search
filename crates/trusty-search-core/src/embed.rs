use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use async_trait::async_trait;
use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use std::path::PathBuf;
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

/// Resolve the persistent model cache directory: `~/.cache/trusty-search/models/`.
///
/// Why: fastembed defaults to `.fastembed_cache` relative to the current working
/// directory, so the model is re-downloaded whenever `trusty-search start` is run
/// from a different directory. An absolute, user-scoped cache path guarantees a
/// single download per machine.
/// What: creates the directory if absent, then returns the `PathBuf`.
/// Test: see `tests::model_cache_dir_is_absolute`.
fn model_cache_dir() -> Result<PathBuf> {
    let base = dirs::cache_dir()
        .context("could not determine platform cache directory (HOME not set?)")?;
    let dir = base.join("trusty-search").join("models");
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("create model cache dir {}", dir.display()))?;
    Ok(dir)
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
    /// Initialize the embedder, downloading the model on first run (~23 MB into
    /// `~/.cache/trusty-search/models/`). Subsequent runs load from that cache
    /// directory and skip the download entirely.
    pub async fn new() -> Result<Self> {
        let cache_dir = model_cache_dir()?;

        // fastembed's `try_new` is blocking (downloads + ONNX session init),
        // so run it on the blocking pool to keep the async runtime responsive.
        let model = tokio::task::spawn_blocking(move || {
            TextEmbedding::try_new(
                TextInitOptions::new(EmbeddingModel::AllMiniLML6V2)
                    .with_cache_dir(cache_dir),
            )
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

/// Test-only deterministic embedder. Hashes the input bytes into a fixed-dim
/// pseudo-vector. Not for production use — there is no semantic structure here,
/// only deterministic output so integration tests can exercise the pipeline
/// without paying the ONNX model-download / inference cost.
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
        // Tiny deterministic hash → vector. Distributes byte indices into the
        // dim slots, so distinct strings produce distinct (but cosine-comparable)
        // vectors. Adequate for "rank by similarity" assertions.
        let mut v = vec![0.0_f32; self.dim];
        for (i, b) in text.bytes().enumerate() {
            let slot = (i + b as usize) % self.dim;
            v[slot] += (b as f32) / 255.0;
        }
        // Always-include the first byte so empty/short strings still differ.
        if let Some(first) = text.bytes().next() {
            v[0] += first as f32 / 255.0;
        }
        v
    }
}

#[cfg(any(test, feature = "test-support"))]
#[async_trait]
impl Embedder for MockEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        Ok(self.hash_to_vec(text))
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| self.hash_to_vec(t)).collect())
    }

    fn dimension(&self) -> usize {
        self.dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    /// Tests that construct `FastEmbedder::new()` are serialised with `#[serial]`
    /// because fastembed/hf_hub uses a per-blob `.lock` file when loading the
    /// ONNX model. Running three concurrent constructors on the same model
    /// causes lock-acquisition failures on macOS and Linux.
    #[tokio::test]
    #[serial]
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
    #[serial]
    async fn test_embed_batch() {
        let embedder = FastEmbedder::new().await.expect("embedder init");
        let texts = vec!["hello world", "fn main() {}", "struct Foo;"];
        let vecs = embedder.embed_batch(&texts).await.expect("batch embed");
        assert_eq!(vecs.len(), 3);
        assert!(vecs.iter().all(|v| v.len() == 384));
    }

    #[tokio::test]
    #[serial]
    async fn test_lru_cache_hit() {
        let embedder = FastEmbedder::new().await.expect("embedder init");
        let text = "cached query";
        let v1 = embedder.embed(text).await.expect("embed 1");
        let v2 = embedder.embed(text).await.expect("embed 2 (cache hit)");
        assert_eq!(v1, v2);
    }
}
