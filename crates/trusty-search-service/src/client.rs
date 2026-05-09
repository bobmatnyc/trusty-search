use anyhow::Result;
use trusty_search_core::indexer::{CodeChunk, SearchQuery};

/// HTTP client for the trusty-search daemon.
pub struct SearchClient {
    base_url: String,
    client: reqwest::Client,
}

impl SearchClient {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            client: reqwest::Client::new(),
        }
    }

    pub async fn health(&self) -> Result<bool> {
        let resp = self.client.get(format!("{}/health", self.base_url)).send().await?;
        Ok(resp.status().is_success())
    }

    pub async fn search(&self, index_id: &str, query: SearchQuery) -> Result<Vec<CodeChunk>> {
        let resp = self.client
            .post(format!("{}/indexes/{}/search", self.base_url, index_id))
            .json(&query)
            .send()
            .await?;
        Ok(resp.json().await?)
    }
}
