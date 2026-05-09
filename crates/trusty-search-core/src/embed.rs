use anyhow::Result;
use async_trait::async_trait;

#[async_trait]
pub trait Embedder: Send + Sync {
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
    fn dimension(&self) -> usize;
}

/// FastEmbedder: fastembed-rs ONNX runtime (all-MiniLM-L6-v2, 384-dim, SIMD-accelerated).
/// No API key required. AVX2/NEON where available.
pub struct FastEmbedder {
    dim: usize,
}

impl FastEmbedder {
    pub async fn new() -> Result<Self> {
        // TODO: initialize fastembed TextEmbedding with EmbeddingModel::AllMiniLML6V2
        Ok(Self { dim: 384 })
    }
}

#[async_trait]
impl Embedder for FastEmbedder {
    async fn embed(&self, _text: &str) -> Result<Vec<f32>> {
        // TODO: tokio::task::spawn_blocking(|| model.embed(vec![text], None))
        Ok(vec![0.0f32; self.dim])
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|_| vec![0.0f32; self.dim]).collect())
    }

    fn dimension(&self) -> usize { self.dim }
}
