use crate::core::indexer::CodeIndexer;
use dashmap::DashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct IndexId(pub String);

impl IndexId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
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

    /// Subtrees (absolute paths) to restrict indexing to. Empty = walk the
    /// entire `root_path`. Sourced from `trusty-search.yaml`'s `paths:` field.
    ///
    /// Why: large polyrepos need to split a single tree into multiple logical
    /// indexes (e.g. `api/` vs `ui/`). Storing the absolute subtree set on the
    /// handle lets the reindex walker prune entire directories without
    /// per-file path arithmetic.
    pub include_paths: Vec<std::path::PathBuf>,

    /// Glob patterns to exclude (on top of the built-in `SKIP_DIRS` /
    /// `should_skip_path` checks). Each pattern is run through
    /// `repo_config::path_matches_any_glob`.
    pub exclude_globs: Vec<String>,

    /// File extension allow-list (without leading dot, e.g. `["rs", "py"]`).
    /// Empty = all supported extensions are indexed.
    pub extensions: Vec<String>,

    /// Domain-specific vocabulary fed to `QueryClassifier::classify_with_domain`
    /// at search time. Empty = standard classifier behaviour.
    pub domain_terms: Vec<String>,
}

impl IndexHandle {
    /// Construct a handle with empty filter/domain fields. Convenience for the
    /// many call sites (warm-boot, tests) that don't carry repo-level config.
    pub fn bare(
        id: IndexId,
        indexer: Arc<RwLock<CodeIndexer>>,
        root_path: std::path::PathBuf,
    ) -> Self {
        Self {
            id,
            indexer,
            root_path,
            include_paths: Vec::new(),
            exclude_globs: Vec::new(),
            extensions: Vec::new(),
            domain_terms: Vec::new(),
        }
    }
}

/// Machine-wide index registry. DashMap = concurrent, shard-locked.
/// Multiple axum handlers can read different indexes simultaneously.
#[derive(Default, Clone)]
pub struct IndexRegistry {
    indexes: Arc<DashMap<IndexId, Arc<IndexHandle>>>,
}

impl IndexRegistry {
    pub fn new() -> Self {
        Self::default()
    }

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

    /// Drop an index from the registry. Returns true if the entry existed.
    ///
    /// Why: `DELETE /indexes/:id` (admin UI) needs a way to evict an index
    /// without restarting the daemon.
    /// What: shard-locked remove via DashMap; the previous `Arc<IndexHandle>`
    /// is dropped when the last reader finishes (RwLock readers from in-flight
    /// search requests keep it alive briefly, which is safe).
    /// Test: register → unregister → get returns None.
    pub fn unregister(&self, id: &IndexId) -> bool {
        self.indexes.remove(id).is_some()
    }

    pub fn len(&self) -> usize {
        self.indexes.len()
    }
    pub fn is_empty(&self) -> bool {
        self.indexes.is_empty()
    }
}
