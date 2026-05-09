use crate::indexer::CodeIndexer;
use dashmap::DashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct IndexId(pub String);

impl IndexId {
    pub fn new(s: impl Into<String>) -> Self { Self(s.into()) }
}

impl std::fmt::Display for IndexId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

pub struct IndexHandle {
    pub id: IndexId,
    pub indexer: Arc<RwLock<CodeIndexer>>,
    pub root_path: std::path::PathBuf,
}

/// Machine-wide index registry. DashMap = concurrent, shard-locked.
/// Multiple axum handlers can read different indexes simultaneously.
#[derive(Default, Clone)]
pub struct IndexRegistry {
    indexes: Arc<DashMap<IndexId, Arc<IndexHandle>>>,
}

impl IndexRegistry {
    pub fn new() -> Self { Self::default() }

    pub fn register(&self, handle: IndexHandle) -> Arc<IndexHandle> {
        let handle = Arc::new(handle);
        self.indexes.insert(handle.id.clone(), Arc::clone(&handle));
        handle
    }

    pub fn get(&self, id: &IndexId) -> Option<Arc<IndexHandle>> {
        self.indexes.get(id).map(|r| Arc::clone(&*r))
    }

    pub fn list(&self) -> Vec<IndexId> {
        self.indexes.iter().map(|r| r.key().clone()).collect()
    }

    pub fn len(&self) -> usize { self.indexes.len() }
    pub fn is_empty(&self) -> bool { self.indexes.is_empty() }
}
