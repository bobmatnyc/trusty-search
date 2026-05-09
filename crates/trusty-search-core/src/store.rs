use anyhow::Result;
use async_trait::async_trait;

#[derive(Debug, Clone)]
pub struct VectorHit {
    pub chunk_id: String,
    pub score: f32,
}

#[async_trait]
pub trait VectorStore: Send + Sync {
    async fn upsert(&self, id: &str, embedding: Vec<f32>) -> Result<()>;
    async fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<VectorHit>>;
    async fn remove(&self, id: &str) -> Result<()>;
    async fn len(&self) -> Result<usize>;
}

/// UsearchStore: usearch HNSW index wrapped in Arc<RwLock<>> for concurrent reads.
/// Many concurrent readers never block each other; writes acquire a brief exclusive lock.
pub struct UsearchStore;
