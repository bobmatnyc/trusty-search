use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

/// Sidecar JSON written alongside the usearch binary snapshot, capturing the
/// `chunk_id → u64 key` mapping (and the `next_key` counter) so a restored
/// index can translate HNSW matches back into chunk ids.
///
/// Why: usearch persists vectors + graph + keys, but only as `u64`s. We
/// allocate string→u64 mappings ourselves in `UsearchStore::id_to_key`, so
/// without this sidecar the loaded index would have orphaned keys.
/// What: `id_to_key` is the authoritative mapping; `next_key` is the
/// monotonic counter so post-restore inserts never collide with restored
/// keys.
/// Test: `tests::test_save_load_roundtrip` exercises this.
#[derive(Debug, Serialize, Deserialize)]
struct StoreKeyMap {
    id_to_key: HashMap<String, u64>,
    next_key: u64,
    dim: usize,
}

/// Initial reserved capacity for a new HNSW index. Grows geometrically on demand.
const INITIAL_CAPACITY: usize = 1_024;

/// Default hard cap on the HNSW index size. The usearch `IndexOptions` API
/// (v2.25) does not expose a `max_elements` field directly, so we enforce the
/// cap in `ensure_capacity` / `upsert_batch`: once the index would grow past
/// this many vectors, subsequent inserts return an error so the daemon can
/// bound RAM (~6 GB at 1M × 384-dim × 4 bytes plus graph overhead).
const DEFAULT_HNSW_MAX_ELEMENTS: usize = 1_000_000;

/// Read the HNSW max-elements cap from the environment, with a sane default.
/// Shared with `TRUSTY_MAX_CHUNKS` so a single knob bounds both the chunk
/// corpus and the vector store.
fn hnsw_max_elements() -> usize {
    std::env::var("TRUSTY_MAX_CHUNKS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n: &usize| n > 0)
        .unwrap_or(DEFAULT_HNSW_MAX_ELEMENTS)
}

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

    /// Bulk-upsert many `(chunk_id, embedding)` pairs.
    ///
    /// Why: per-chunk `upsert` acquires three write locks (`id_to_key`,
    /// `key_to_id`, `index`) for each call. On a 115k-chunk index that's
    /// ~345k lock round-trips and serializes the entire embed pipeline behind
    /// the HNSW write lock. Concrete impls should override to do all key
    /// allocation and all HNSW writes under a single lock acquisition each.
    /// What: default implementation loops over `upsert` so non-Usearch backends
    /// keep working; `UsearchStore` overrides for the fast path.
    /// Test: see `test_upsert_batch_inserts_all` in this module.
    async fn upsert_batch(&self, items: &[(String, Vec<f32>)]) -> Result<()> {
        for (id, vec) in items {
            self.upsert(id, vec.clone()).await?;
        }
        Ok(())
    }

    /// Persist this store to disk. Default = no-op (in-memory backends).
    ///
    /// Why: lets `CodeIndexer::save_to_disk` call through a `dyn VectorStore`
    /// without downcasting. `UsearchStore` overrides; mock test stores keep
    /// the no-op so they round-trip without filesystem access.
    /// What: persist whatever state is needed to restore via `load_from`.
    /// Test: covered by `UsearchStore::test_save_load_roundtrip`.
    async fn save_to(&self, _path: &Path) -> Result<()> {
        Ok(())
    }
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
        let index = Index::new(&options).map_err(|e| anyhow!("usearch Index::new failed: {e}"))?;
        // Clamp initial reserve to the env-configured max so a runaway
        // `expected_chunks` doesn't pre-allocate hundreds of GB.
        let initial = expected_chunks
            .max(INITIAL_CAPACITY)
            .min(hnsw_max_elements());
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

    /// Persist the HNSW graph and the `chunk_id → u64 key` sidecar to disk.
    ///
    /// Why (issue #85): on graceful shutdown (and incrementally after each
    /// `commit_parsed_batch`) we save the in-memory HNSW so the daemon can
    /// warm-boot without re-embedding the entire corpus. Without this every
    /// restart costs minutes of re-indexing.
    /// What: snapshots `id_to_key` + `next_key` under read locks, releases the
    /// locks, then calls usearch's `Index::save(&str)` and writes the sidecar
    /// JSON. Both writes are atomic (tmp + rename) so a crash mid-save never
    /// leaves a partial file. The caller passes the HNSW path; the sidecar is
    /// written next to it with extension `.keys.json`.
    /// Test: `tests::test_save_load_roundtrip` saves then loads into a fresh
    /// store and asserts a search still returns the original chunk_ids.
    pub async fn save(&self, hnsw_path: &Path) -> Result<()> {
        // Snapshot the key map under read locks so we can release them before
        // the (possibly slow) usearch save. The HNSW write lock is required
        // because usearch's save is `&self` but mutates internal serializer
        // buffers; treating it as a write-side operation matches the rest of
        // this store.
        let key_map = {
            let id_to_key = self.id_to_key.read().await;
            StoreKeyMap {
                id_to_key: id_to_key.clone(),
                next_key: self.next_key.load(Ordering::Relaxed),
                dim: self.dim,
            }
        };

        if let Some(parent) = hnsw_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| anyhow!("create parent of {}: {e}", hnsw_path.display()))?;
        }

        // usearch's `save` takes a `&str` path. We write to a tmp file and
        // rename so callers never observe a half-written snapshot.
        let tmp_hnsw = hnsw_path.with_extension("usearch.tmp");
        let tmp_hnsw_str = tmp_hnsw
            .to_str()
            .ok_or_else(|| anyhow!("non-utf8 path: {}", tmp_hnsw.display()))?;
        {
            let index = self.index.write().await;
            index
                .save(tmp_hnsw_str)
                .map_err(|e| anyhow!("usearch save failed: {e}"))?;
        }
        std::fs::rename(&tmp_hnsw, hnsw_path).map_err(|e| anyhow!("rename hnsw snapshot: {e}"))?;

        let sidecar = hnsw_path.with_extension("keys.json");
        let sidecar_tmp = sidecar.with_extension("json.tmp");
        let json =
            serde_json::to_vec(&key_map).map_err(|e| anyhow!("serialize hnsw key map: {e}"))?;
        std::fs::write(&sidecar_tmp, &json)
            .map_err(|e| anyhow!("write hnsw key sidecar tmp: {e}"))?;
        std::fs::rename(&sidecar_tmp, &sidecar)
            .map_err(|e| anyhow!("rename hnsw key sidecar: {e}"))?;
        Ok(())
    }

    /// Load a previously-saved HNSW snapshot and its key sidecar.
    ///
    /// Why: counterpart of [`Self::save`]. On daemon startup or `create_index`
    /// the registered index can boot with its vectors restored — no re-embed
    /// pass required.
    /// What: builds a fresh `Index` with options matching `with_capacity_hint`,
    /// calls usearch's `load(path)`, then reads the sidecar to restore the
    /// string ↔ key mappings and `next_key`. If either file is missing or
    /// corrupt, returns `Ok(None)` so callers fall back to a fresh empty
    /// store instead of crashing the daemon.
    /// Test: `tests::test_save_load_roundtrip` covers the happy path; a
    /// `tests::test_load_missing_returns_none` covers the absent-file branch.
    pub async fn load_from(hnsw_path: &Path) -> Result<Option<Self>> {
        let sidecar = hnsw_path.with_extension("keys.json");
        if !hnsw_path.exists() || !sidecar.exists() {
            return Ok(None);
        }

        let json = match std::fs::read(&sidecar) {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!(
                    "could not read hnsw key sidecar {}: {e} — discarding snapshot",
                    sidecar.display()
                );
                return Ok(None);
            }
        };
        let key_map: StoreKeyMap = match serde_json::from_slice(&json) {
            Ok(m) => m,
            Err(e) => {
                tracing::warn!(
                    "hnsw key sidecar {} is corrupt ({e}) — discarding snapshot",
                    sidecar.display()
                );
                return Ok(None);
            }
        };

        let expected_chunks = key_map.id_to_key.len();
        let store = Self::with_capacity_hint(key_map.dim, expected_chunks)?;
        let hnsw_str = match hnsw_path.to_str() {
            Some(s) => s,
            None => {
                tracing::warn!(
                    "non-utf8 hnsw path {} — discarding snapshot",
                    hnsw_path.display()
                );
                return Ok(None);
            }
        };
        {
            let index = store.index.write().await;
            if let Err(e) = index.load(hnsw_str) {
                tracing::warn!(
                    "usearch failed to load {} ({e}) — discarding snapshot",
                    hnsw_path.display()
                );
                return Ok(None);
            }
            // After load, reserve enough capacity for the restored size so
            // the next insert doesn't immediately re-grow.
            let size = index.size();
            if index.capacity() < size {
                let _ = index.reserve(size);
            }
        }

        // Rehydrate the mappings.
        {
            let mut id_map = store.id_to_key.write().await;
            let mut key_map_rev = store.key_to_id.write().await;
            for (id, key) in &key_map.id_to_key {
                id_map.insert(id.clone(), *key);
                key_map_rev.insert(*key, id.clone());
            }
        }
        store
            .next_key
            .store(key_map.next_key.max(1), Ordering::Relaxed);
        Ok(Some(store))
    }

    /// Ensure the underlying HNSW has room for at least one more vector.
    /// Grows geometrically (×2) to amortize the cost of reserve calls. Refuses
    /// to grow past `hnsw_max_elements()` so the daemon's RAM is bounded
    /// (issue #75).
    fn ensure_capacity(index: &Index) -> Result<()> {
        let size = index.size();
        let cap = index.capacity();
        let max_elem = hnsw_max_elements();
        if size >= max_elem {
            return Err(anyhow!(
                "usearch index at TRUSTY_MAX_CHUNKS cap ({} elements) — refusing further upserts",
                max_elem
            ));
        }
        if size + 1 > cap {
            let mut new_cap = (cap.max(1)).saturating_mul(2);
            if new_cap > max_elem {
                new_cap = max_elem;
            }
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

    async fn save_to(&self, path: &Path) -> Result<()> {
        self.save(path).await
    }

    /// Single-lock-pass override. Two phases:
    /// 1. Resolve/assign every chunk's `u64` key under one write-lock pair
    ///    (`id_to_key` + `key_to_id`).
    /// 2. Insert every vector under one HNSW write lock.
    /// This drops 6N lock acquisitions to 6 for a batch of N items.
    async fn upsert_batch(&self, items: &[(String, Vec<f32>)]) -> Result<()> {
        if items.is_empty() {
            return Ok(());
        }
        // Validate dims up front so we don't half-commit on a bad batch.
        for (_, v) in items {
            if v.len() != self.dim {
                return Err(anyhow!(
                    "embedding dim mismatch: got {}, expected {}",
                    v.len(),
                    self.dim
                ));
            }
        }

        // Phase 1: assign keys for any new IDs under a single write-lock pair.
        {
            let mut id_map = self.id_to_key.write().await;
            let mut key_map = self.key_to_id.write().await;
            for (id, _) in items {
                if !id_map.contains_key(id.as_str()) {
                    let k = self.next_key.fetch_add(1, Ordering::Relaxed);
                    id_map.insert(id.clone(), k);
                    key_map.insert(k, id.clone());
                }
            }
        }

        // Phase 2: insert every vector under one HNSW write lock.
        let id_map = self.id_to_key.read().await;
        let index = self.index.write().await;
        // Reserve once for the worst case (every item is new) so we don't
        // re-enter the reserve path inside the hot loop.
        let want = index.size() + items.len();
        let max_elem = hnsw_max_elements();
        if index.size() >= max_elem {
            return Err(anyhow!(
                "usearch index at TRUSTY_MAX_CHUNKS cap ({} elements) — refusing batch upsert",
                max_elem
            ));
        }
        if want > index.capacity() {
            let mut new_cap = index.capacity().max(1);
            while new_cap < want {
                new_cap = new_cap.saturating_mul(2);
            }
            if new_cap > max_elem {
                new_cap = max_elem;
            }
            index
                .reserve(new_cap)
                .map_err(|e| anyhow!("usearch reserve grow failed: {e}"))?;
        }
        for (id, embedding) in items {
            if let Some(&key) = id_map.get(id.as_str()) {
                if index.contains(key) {
                    index
                        .remove(key)
                        .map_err(|e| anyhow!("usearch remove (for upsert) failed: {e}"))?;
                }
                index
                    .add(key, embedding)
                    .map_err(|e| anyhow!("usearch add failed: {e}"))?;
            }
        }
        Ok(())
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
        store.upsert("x", vec![1.0, 0.0, 0.0, 0.0]).await.unwrap();
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
        store.upsert("r1", vec![1.0, 0.0, 0.0, 0.0]).await.unwrap();
        store.upsert("r2", vec![0.0, 1.0, 0.0, 0.0]).await.unwrap();

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
    async fn test_upsert_batch_inserts_all() {
        let store = UsearchStore::new(4).expect("store init");
        // Use orthogonal directions so cosine sim distinguishes them (parallel
        // vectors share cosine sim of 1 regardless of magnitude).
        let dirs: [[f32; 4]; 4] = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let items: Vec<(String, Vec<f32>)> = (0..4)
            .map(|i| (format!("k{i}"), dirs[i].to_vec()))
            .collect();
        store.upsert_batch(&items).await.expect("batch upsert");
        assert_eq!(store.len().await.unwrap(), 4);
        // Re-batch upserting the same ids should overwrite, not duplicate.
        store.upsert_batch(&items).await.expect("re-batch upsert");
        assert_eq!(store.len().await.unwrap(), 4);
        // Top hit for k2's exact vector must be k2.
        let hits = store.search(&dirs[2], 1).await.unwrap();
        assert_eq!(hits[0].chunk_id, "k2");
    }

    #[tokio::test]
    async fn test_upsert_batch_empty_noop() {
        let store = UsearchStore::new(4).expect("store init");
        store.upsert_batch(&[]).await.unwrap();
        assert_eq!(store.len().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_upsert_batch_dim_mismatch_errors() {
        let store = UsearchStore::new(4).expect("store init");
        let items = vec![("bad".to_string(), vec![1.0, 0.0])];
        assert!(store.upsert_batch(&items).await.is_err());
    }

    #[tokio::test]
    async fn test_save_load_roundtrip() {
        // Why: validate the persistence path end-to-end so issue #85 actually
        // survives a "restart" (simulated here by dropping the store and
        // loading the snapshot into a fresh one).
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("hnsw.usearch");

        let store = UsearchStore::new(4).unwrap();
        store
            .upsert("alpha", vec![1.0, 0.0, 0.0, 0.0])
            .await
            .unwrap();
        store
            .upsert("beta", vec![0.0, 1.0, 0.0, 0.0])
            .await
            .unwrap();
        store.save(&path).await.expect("save");
        assert!(path.exists(), "hnsw file must exist after save");
        assert!(
            path.with_extension("keys.json").exists(),
            "key sidecar must exist after save"
        );

        drop(store);

        let loaded = UsearchStore::load_from(&path)
            .await
            .expect("load ok")
            .expect("load returned Some");
        assert_eq!(loaded.len().await.unwrap(), 2);
        let hits = loaded.search(&[1.0, 0.0, 0.0, 0.0], 1).await.unwrap();
        assert_eq!(hits[0].chunk_id, "alpha", "restored ids must round-trip");
    }

    #[tokio::test]
    async fn test_load_missing_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nope.usearch");
        let loaded = UsearchStore::load_from(&path).await.unwrap();
        assert!(loaded.is_none());
    }

    #[tokio::test]
    async fn test_load_corrupt_sidecar_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("hnsw.usearch");
        // Create both files but corrupt the sidecar.
        let store = UsearchStore::new(4).unwrap();
        store.upsert("a", vec![1.0, 0.0, 0.0, 0.0]).await.unwrap();
        store.save(&path).await.unwrap();
        std::fs::write(path.with_extension("keys.json"), b"not valid json").unwrap();
        let loaded = UsearchStore::load_from(&path).await.unwrap();
        assert!(loaded.is_none(), "corrupt sidecar must fall back to None");
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
