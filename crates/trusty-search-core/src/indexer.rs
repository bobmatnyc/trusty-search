use crate::classifier::QueryClassifier;
use serde::{Deserialize, Serialize};

/// A search result returned to callers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeChunk {
    /// Collision-safe ID: "{path}:{start}:{end}"
    pub id: String,
    pub file: String,
    pub start_line: usize,
    pub end_line: usize,
    pub content: String,
    pub function_name: Option<String>,
    pub score: f32,
    /// Compact 7-line snippet for token-efficient output
    pub compact_snippet: Option<String>,
    /// How this result was found: "hybrid", "hybrid+kg", "bm25", "vector", "fallback:ripgrep"
    pub match_reason: String,
}

/// Query parameters for hybrid search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchQuery {
    pub text: String,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    #[serde(default = "default_true")]
    pub expand_graph: bool,
    #[serde(default = "default_true")]
    pub compact: bool,
}

fn default_top_k() -> usize { 10 }
fn default_true() -> bool { true }

/// CodeIndexer: HNSW + BM25 + RRF fusion + KG expansion + LRU query cache.
/// Wrapped in Arc<RwLock<>> for concurrent reads across axum handlers.
pub struct CodeIndexer {
    pub index_id: String,
    pub root_path: std::path::PathBuf,
    chunk_count: usize,
}

impl CodeIndexer {
    pub fn new(index_id: impl Into<String>, root_path: impl Into<std::path::PathBuf>) -> Self {
        Self {
            index_id: index_id.into(),
            root_path: root_path.into(),
            chunk_count: 0,
        }
    }

    pub fn chunk_count(&self) -> usize { self.chunk_count }

    /// Hybrid search: classify intent → route weights → HNSW + BM25 → RRF → KG expand.
    pub async fn search(&self, query: &SearchQuery) -> anyhow::Result<Vec<CodeChunk>> {
        let intent = QueryClassifier::classify(&query.text);
        let (alpha, beta, _use_kg_first) = intent.weights();
        tracing::debug!("query='{}' intent={:?} alpha={} beta={}", query.text, intent, alpha, beta);
        // TODO: implement full hybrid pipeline
        let _ = (alpha, beta);
        Ok(Vec::new())
    }
}
