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

    /// Glob patterns matched against the *immediate subdirectory name* under
    /// `root_path`. When non-empty, the reindex walker keeps only files
    /// whose first path component (relative to `root_path`) matches one of
    /// these patterns. Issue #111.
    ///
    /// Why: large polyrepo roots (e.g. a directory of cloned Git repos) need
    /// a quick way to scope an index to a subset of sibling repositories
    /// without enumerating every absolute subtree. Glob patterns (`*` only,
    /// no `**`) match repo-name shapes like `common-*` or `duetto-common*`.
    /// What: A `Vec<String>` of glob patterns; empty = scan the whole
    /// `root_path` (current behaviour, no regression).
    /// Test: `path_filter_matches_immediate_subdir` covers the glob logic;
    /// `reindex_honours_path_filter` covers the end-to-end walk.
    pub path_filter: Vec<String>,

    /// Embedded semantic fingerprint of the index's root-level metadata
    /// (`README.md`, `CLAUDE.md`, `Cargo.toml`, …) — issue #112.
    ///
    /// Why: cross-index fan-out (`POST /search`) needs a cheap way to
    /// weight or skip indexes based on query relevance to each project's
    /// description. Storing a single pre-computed embedding here lets the
    /// fan-out handler compute cosine similarity against the query
    /// embedding in O(d) per index, rather than running a full per-index
    /// search probe.
    /// What: `None` when no recognised metadata file was found (cosine
    /// weight defaults to neutral 1.0 in the router); `Some(vec)` carries a
    /// `dim`-length unit-ish vector produced by the same embedder used for
    /// chunks. Populated by [`crate::service::context_inference`] at the
    /// end of every reindex.
    /// Test: `context_embedding_*` tests in
    /// `crate::service::context_inference::tests`.
    pub context_embedding: Arc<RwLock<Option<Vec<f32>>>>,

    /// Truncated (≤500 char) human-readable preview of the metadata
    /// summary that produced [`Self::context_embedding`]. Surfaced via
    /// `GET /indexes/:id/status` for operator visibility.
    pub context_summary: Arc<RwLock<Option<String>>>,
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
            path_filter: Vec::new(),
            context_embedding: Arc::new(RwLock::new(None)),
            context_summary: Arc::new(RwLock::new(None)),
        }
    }
}

/// Match a file path against a set of glob patterns applied to the immediate
/// subdirectory of `root_path` (issue #111).
///
/// Why: `path_filter` on a registered index lets callers restrict indexing
/// to a subset of immediate subdirectories — usually repo names inside a
/// polyrepo. The match runs against the *first path component* of
/// `path.strip_prefix(root_path)` so the filter never has to know about
/// the deeper file tree.
/// What: returns `true` when `patterns` is empty (no filtering) or when at
/// least one pattern in `patterns` matches the immediate subdir's basename
/// via `glob::Pattern` (`*` wildcards, no `**`). Files that live directly in
/// `root_path` (depth 0) are kept only if a pattern equals `"."` or matches
/// the empty string — typically they aren't there for polyrepo layouts but
/// we err on the side of keeping them when no pattern is supplied.
/// Test: `path_filter_matches_immediate_subdir` in this module.
pub fn path_matches_filter(
    path: &std::path::Path,
    root_path: &std::path::Path,
    patterns: &[String],
) -> bool {
    if patterns.is_empty() {
        return true;
    }
    let Ok(rel) = path.strip_prefix(root_path) else {
        // Path is outside the configured root; treat as non-match so the
        // walker drops it.
        return false;
    };
    let first_component = rel.components().next().and_then(|c| c.as_os_str().to_str());
    let Some(subdir) = first_component else {
        return false;
    };
    for pat in patterns {
        let Ok(pattern) = glob::Pattern::new(pat) else {
            // Malformed pattern: log once via tracing and fall back to a
            // literal string compare so a typo doesn't silently drop every
            // file.
            tracing::warn!("path_filter pattern '{pat}' is not a valid glob; using exact match");
            if pat == subdir {
                return true;
            }
            continue;
        };
        if pattern.matches(subdir) {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Verify the glob matcher only inspects the *first* path component
    /// (the immediate subdir name) and supports `*` wildcards.
    #[test]
    fn path_filter_matches_immediate_subdir() {
        let root = PathBuf::from("/data/repos");

        // Empty pattern set ⇒ everything passes.
        assert!(path_matches_filter(
            &PathBuf::from("/data/repos/anything/src/lib.rs"),
            &root,
            &[],
        ));

        // Exact match on immediate subdir.
        let patterns = vec!["duetto-common".to_string()];
        assert!(path_matches_filter(
            &PathBuf::from("/data/repos/duetto-common/src/lib.rs"),
            &root,
            &patterns,
        ));
        assert!(!path_matches_filter(
            &PathBuf::from("/data/repos/other/src/lib.rs"),
            &root,
            &patterns,
        ));

        // Glob wildcard with `*`.
        let patterns = vec!["common-*".to_string(), "duetto-common*".to_string()];
        assert!(path_matches_filter(
            &PathBuf::from("/data/repos/common-utils/foo.rs"),
            &root,
            &patterns,
        ));
        assert!(path_matches_filter(
            &PathBuf::from("/data/repos/duetto-common-events/lib.rs"),
            &root,
            &patterns,
        ));
        assert!(!path_matches_filter(
            &PathBuf::from("/data/repos/totally-other/lib.rs"),
            &root,
            &patterns,
        ));

        // Path outside root → not matched.
        assert!(!path_matches_filter(
            &PathBuf::from("/elsewhere/duetto-common/lib.rs"),
            &root,
            &patterns,
        ));
    }

    /// Multiple matching patterns: any one match is enough.
    #[test]
    fn path_filter_matches_any_pattern() {
        let root = PathBuf::from("/repos");
        let patterns = vec!["api".to_string(), "frontend".to_string()];
        assert!(path_matches_filter(
            &PathBuf::from("/repos/api/handlers.rs"),
            &root,
            &patterns,
        ));
        assert!(path_matches_filter(
            &PathBuf::from("/repos/frontend/app.tsx"),
            &root,
            &patterns,
        ));
        assert!(!path_matches_filter(
            &PathBuf::from("/repos/docs/README.md"),
            &root,
            &patterns,
        ));
    }

    /// Malformed glob pattern: falls back to literal match so a typo never
    /// silently drops every file. The non-matching path still returns false.
    #[test]
    fn path_filter_malformed_pattern_falls_back_to_exact() {
        let root = PathBuf::from("/r");
        // `[` opens a glob character class that is never closed.
        let patterns = vec!["[unclosed".to_string()];
        assert!(path_matches_filter(
            &PathBuf::from("/r/[unclosed/file.rs"),
            &root,
            &patterns,
        ));
        assert!(!path_matches_filter(
            &PathBuf::from("/r/other/file.rs"),
            &root,
            &patterns,
        ));
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
