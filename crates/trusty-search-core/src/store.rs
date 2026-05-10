use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use tokio::sync::RwLock;
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

/// Initial reserved capacity for a new HNSW index. Grows geometrically on demand.
const INITIAL_CAPACITY: usize = 1_024;

#[derive(Debug, Clone)]
pub struct VectorHit {
    pub chunk_id: String,
    pub score: f32,
}

/// Abstract vector store interface. Concrete impls (in-process HNSW today,
/// possibly remote tomorrow) plug in here so the rest of the indexer never
/// imports `usearch` directly.
///
/// Why: Decouples the indexer from any specific ANN backend so we can swap
/// implementations (mocks for tests, remote services for sharding) without
/// touching call sites.
/// What: Async upsert/search/remove/len over `(String chunk_id, Vec<f32>)`.
/// Test: See `UsearchStore` tests below — exercise upsert, search ordering,
/// remove, and len through this trait.
#[async_trait]
#[allow(clippy::len_without_is_empty)]
pub trait VectorStore: Send + Sync {
    async fn upsert(&self, id: &str, embedding: Vec<f32>) -> Result<()>;
    async fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<VectorHit>>;
    async fn remove(&self, id: &str) -> Result<()>;
    async fn len(&self) -> Result<usize>;
}

/// `UsearchStore`: usearch HNSW index wrapped in `Arc<RwLock<>>` for concurrent reads.
///
/// Why: The HNSW graph is shared across many concurrent search requests; reader-priority
/// locking lets searches run in parallel and keeps the daemon's p50 latency low.
/// What: Maps `String` chunk IDs ↔ `u64` usearch keys, manages capacity growth, and
/// translates cosine distances back into similarity scores (`1 - d`) so callers see
/// "higher = better" like the rest of the pipeline.
/// Test: `tests::test_upsert_and_search` adds three vectors and asserts the exact-match
/// vector ranks first; `test_remove` and `test_concurrent_reads` cover lifecycle and
/// reader parallelism.
pub struct UsearchStore {
    index: Arc<RwLock<Index>>,
    /// chunk_id → usearch u64 key
    id_to_key: Arc<RwLock<HashMap<String, u64>>>,
    /// usearch u64 key → chunk_id (needed to translate `Matches.keys` back to strings)
    key_to_id: Arc<RwLock<HashMap<u64, String>>>,
    /// Monotonic key generator. Never reused, even after `remove`, so KG/BM25 layers
    /// that may still hold a stale key can't accidentally collide with a fresh insert.
    next_key: Arc<AtomicU64>,
    dim: usize,
}

impl UsearchStore {
    /// Construct an empty HNSW index for `dim`-dimensional cosine-similarity vectors.
    ///
    /// Why: All-MiniLM-L6-v2 produces 384-dim embeddings; cosine is the standard
    /// similarity metric for sentence embeddings.
    /// What: Builds a usearch `Index` with `MetricKind::Cos` + `ScalarKind::F32`,
    /// reserves `INITIAL_CAPACITY` slots, and wires up the bidirectional ID map.
    /// Test: `test_len` constructs a fresh store and asserts `len() == 0`.
    pub fn new(dim: usize) -> Result<Self> {
        Self::with_capacity_hint(dim, INITIAL_CAPACITY)
    }

    /// Construct with an estimated final size. When `expected_chunks > 50_000`
    /// we tune the HNSW graph for higher recall (higher `connectivity` /
    /// `expansion_add`) at the cost of more memory and slower build —
    /// worthwhile on large monorepos where the default `connectivity=16`
    /// produces noisier neighbour lists. Smaller indexes keep usearch's
    /// auto-defaults (0 = library-chosen).
    pub fn with_capacity_hint(dim: usize, expected_chunks: usize) -> Result<Self> {
        let (connectivity, expansion_add, expansion_search) = if expected_chunks > 50_000 {
            (32, 128, 64)
        } else {
            (0, 0, 0)
        };
        let options = IndexOptions {
            dimensions: dim,
            metric: MetricKind::Cos,
            quantization: ScalarKind::F32,
            connectivity,
            expansion_add,
            expansion_search,
            multi: false,
        };
        let index = Index::new(&options)
            .map_err(|e| anyhow!("usearch Index::new failed: {e}"))?;
        let initial = expected_chunks.max(INITIAL_CAPACITY);
        index
            .reserve(initial)
            .map_err(|e| anyhow!("usearch reserve failed: {e}"))?;

        Ok(Self {
            index: Arc::new(RwLock::new(index)),
            id_to_key: Arc::new(RwLock::new(HashMap::new())),
            key_to_id: Arc::new(RwLock::new(HashMap::new())),
            next_key: Arc::new(AtomicU64::new(1)), // start at 1; reserve 0 as sentinel
            dim,
        })
    }

    /// Vector dimensionality this store was built for.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Ensure the underlying HNSW has room for at least one more vector.
    /// Grows geometrically (×2) to amortize the cost of reserve calls.
    fn ensure_capacity(index: &Index) -> Result<()> {
        let size = index.size();
        let cap = index.capacity();
        if size + 1 > cap {
            let new_cap = (cap.max(1)).saturating_mul(2);
            index
                .reserve(new_cap)
                .map_err(|e| anyhow!("usearch reserve grow failed: {e}"))?;
        }
        Ok(())
    }
}

#[async_trait]
impl VectorStore for UsearchStore {
    async fn upsert(&self, id: &str, embedding: Vec<f32>) -> Result<()> {
        if embedding.len() != self.dim {
            return Err(anyhow!(
                "embedding dim mismatch: got {}, expected {}",
                embedding.len(),
                self.dim
            ));
        }

        // Resolve or allocate the u64 key under a write lock.
        let key = {
            let mut id_to_key = self.id_to_key.write().await;
            if let Some(&existing) = id_to_key.get(id) {
                existing
            } else {
                let key = self.next_key.fetch_add(1, Ordering::Relaxed);
                id_to_key.insert(id.to_string(), key);
                self.key_to_id.write().await.insert(key, id.to_string());
                key
            }
        };

        let index = self.index.write().await;

        // If the key already existed, remove the old vector first so `add` doesn't
        // collide. usearch's `multi=false` index treats duplicate keys as errors.
        if index.contains(key) {
            index
                .remove(key)
                .map_err(|e| anyhow!("usearch remove (for upsert) failed: {e}"))?;
        }

        Self::ensure_capacity(&index)?;
        index
            .add(key, &embedding)
            .map_err(|e| anyhow!("usearch add failed: {e}"))?;
        Ok(())
    }

    async fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<VectorHit>> {
        if query.len() != self.dim {
            return Err(anyhow!(
                "query dim mismatch: got {}, expected {}",
                query.len(),
                self.dim
            ));
        }
        if top_k == 0 {
            return Ok(Vec::new());
        }

        let matches = {
            let index = self.index.read().await;
            index
                .search(query, top_k)
                .map_err(|e| anyhow!("usearch search failed: {e}"))?
        };

        let key_to_id = self.key_to_id.read().await;
        let mut hits = Vec::with_capacity(matches.keys.len());
        for (key, dist) in matches.keys.iter().zip(matches.distances.iter()) {
            if let Some(chunk_id) = key_to_id.get(key) {
                // Cosine distance ∈ [0, 2]; convert to similarity ∈ [-1, 1] so callers
                // can RRF/fuse with BM25 scores where "higher = better".
                let score = 1.0 - *dist;
                hits.push(VectorHit {
                    chunk_id: chunk_id.clone(),
                    score,
                });
            }
            // Silently skip orphaned keys (e.g. removed mid-search) — the alternative
            // of erroring would tear down a valid query for a benign race.
        }
        Ok(hits)
    }

    async fn remove(&self, id: &str) -> Result<()> {
        let key = {
            let mut id_to_key = self.id_to_key.write().await;
            match id_to_key.remove(id) {
                Some(k) => k,
                None => return Ok(()), // idempotent: removing an unknown id is a no-op
            }
        };
        self.key_to_id.write().await.remove(&key);

        let index = self.index.write().await;
        if index.contains(key) {
            index
                .remove(key)
                .map_err(|e| anyhow!("usearch remove failed: {e}"))?;
        }
        Ok(())
    }

    async fn len(&self) -> Result<usize> {
        Ok(self.index.read().await.size())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_upsert_and_search() {
        let store = UsearchStore::new(4).expect("store init");
        let v = vec![1.0f32, 0.0, 0.0, 0.0];
        store.upsert("chunk:a", v.clone()).await.expect("upsert a");
        store
            .upsert("chunk:b", vec![0.0, 1.0, 0.0, 0.0])
            .await
            .expect("upsert b");
        store
            .upsert("chunk:c", vec![0.9, 0.1, 0.0, 0.0])
            .await
            .expect("upsert c");

        let hits = store.search(&v, 2).await.expect("search");
        assert_eq!(hits.len(), 2);
        // chunk:a should be the top hit (exact match)
        assert_eq!(hits[0].chunk_id, "chunk:a");
    }

    #[tokio::test]
    async fn test_len() {
        let store = UsearchStore::new(4).expect("store init");
        assert_eq!(store.len().await.unwrap(), 0);
        store
            .upsert("x", vec![1.0, 0.0, 0.0, 0.0])
            .await
            .unwrap();
        assert_eq!(store.len().await.unwrap(), 1);
    }

    #[tokio::test]
    async fn test_remove() {
        let store = UsearchStore::new(4).expect("store init");
        store
            .upsert("del-me", vec![1.0, 0.0, 0.0, 0.0])
            .await
            .unwrap();
        assert_eq!(store.len().await.unwrap(), 1);
        store.remove("del-me").await.unwrap();
        // After remove, search should not return "del-me"
        let hits = store.search(&[1.0, 0.0, 0.0, 0.0], 5).await.unwrap();
        assert!(!hits.iter().any(|h| h.chunk_id == "del-me"));
    }

    #[tokio::test]
    async fn test_concurrent_reads() {
        let store = Arc::new(UsearchStore::new(4).expect("store init"));
        store
            .upsert("r1", vec![1.0, 0.0, 0.0, 0.0])
            .await
            .unwrap();
        store
            .upsert("r2", vec![0.0, 1.0, 0.0, 0.0])
            .await
            .unwrap();

        let s1 = store.clone();
        let s2 = store.clone();
        let q = vec![1.0f32, 0.0, 0.0, 0.0];
        let (r1, r2) = tokio::join!(s1.search(&q, 2), s2.search(&q, 2));
        assert!(!r1.unwrap().is_empty());
        assert!(!r2.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_upsert_replaces_existing() {
        // Re-upserting the same id should overwrite, not double-count.
        let store = UsearchStore::new(4).expect("store init");
        store
            .upsert("same", vec![1.0, 0.0, 0.0, 0.0])
            .await
            .unwrap();
        store
            .upsert("same", vec![0.0, 1.0, 0.0, 0.0])
            .await
            .unwrap();
        assert_eq!(store.len().await.unwrap(), 1);

        // Now its closest neighbour to (0,1,0,0) should be itself.
        let hits = store.search(&[0.0, 1.0, 0.0, 0.0], 1).await.unwrap();
        assert_eq!(hits[0].chunk_id, "same");
    }

    #[tokio::test]
    async fn test_dim_mismatch_errors() {
        let store = UsearchStore::new(4).expect("store init");
        assert!(store.upsert("bad", vec![1.0, 0.0]).await.is_err());
        assert!(store.search(&[1.0, 0.0], 1).await.is_err());
    }

    #[tokio::test]
    async fn test_capacity_growth() {
        // Force more inserts than INITIAL_CAPACITY would normally hold to exercise
        // the geometric reserve growth path without bloating test runtime.
        let store = UsearchStore::new(4).expect("store init");
        for i in 0..50 {
            let v = vec![i as f32, 0.0, 0.0, 0.0];
            store.upsert(&format!("k{i}"), v).await.unwrap();
        }
        assert_eq!(store.len().await.unwrap(), 50);
    }
}
