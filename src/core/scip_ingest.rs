//! SCIP protobuf ingest for LSP-quality entity data.
//!
//! Why: tree-sitter extraction is fast and dependency-free, but it cannot
//! resolve cross-file symbol references with the fidelity of a real LSP /
//! compiler index. Projects that already produce a SCIP index in CI
//! (`scip-rust`, `scip-python`, `scip-typescript`, etc.) can feed that index
//! into `trusty-search` for higher-fidelity entity and edge data.
//!
//! What: this module defines the `CodeEntityIndex` trait — the minimal
//! interface the indexer/KG layer consumes — plus a `ScipIndex` implementation.
//! `ScipIndex::from_refs` is the fully-testable path: callers (or future SCIP
//! parser code) hand it a list of `ScipEntityRef` + `ScipEdge` and it produces
//! a `CodeEntityIndex` directly. `ScipIndex::from_scip` is reserved for native
//! protobuf parsing once the `scip` (or `prost`) dependency is wired in.
//!
//! Dependency note: at this stage we deliberately avoid pulling the `scip`
//! crate (and its transitive protobuf toolchain) into the default build. The
//! trait + `from_refs` constructor is the load-bearing surface; native
//! `.scip` decode can be added later without changing this API.

use crate::core::entity::{fact_hash_str, EdgeKind, EntityType, RawEntity};

/// A cross-file entity reference materialised from a SCIP document.
///
/// Maps roughly onto a single SCIP `Occurrence` joined with its
/// `SymbolInformation`: the symbol string, a human-readable display name, the
/// source file/line span, and whether this occurrence is the symbol's
/// definition site.
#[derive(Debug, Clone)]
pub struct ScipEntityRef {
    /// SCIP symbol string, e.g. `"rust-analyzer cargo package.module/Struct#"`.
    pub symbol: String,
    /// Human-readable display name (typically the local identifier).
    pub display_name: String,
    /// Source file path (project-relative, matching how chunks are stored).
    pub file: String,
    /// 1-based start line of the occurrence.
    pub start_line: usize,
    /// 1-based end line of the occurrence (inclusive).
    pub end_line: usize,
    /// `true` when this occurrence is the symbol's definition site.
    pub is_definition: bool,
}

/// A SCIP-derived edge between two symbols.
#[derive(Debug, Clone)]
pub struct ScipEdge {
    pub from_symbol: String,
    pub to_symbol: String,
    pub kind: EdgeKind,
}

/// Trait for code entity indexes backed by different sources (tree-sitter,
/// SCIP, future LSP integrations).
///
/// The indexer / KG layer only needs the flat `entities` and `edges` views,
/// so the trait stays intentionally small.
pub trait CodeEntityIndex: Send + Sync {
    /// All entities the source produced.
    fn entities(&self) -> &[RawEntity];
    /// All edges as `(from_symbol, kind, to_symbol)` triples.
    fn edges(&self) -> &[(String, EdgeKind, String)];
}

/// A SCIP-backed entity index.
///
/// Constructed either by parsing a `.scip` protobuf file (`from_scip`) or
/// from an in-memory list of references (`from_refs`).
pub struct ScipIndex {
    entities: Vec<RawEntity>,
    edges: Vec<(String, EdgeKind, String)>,
}

impl ScipIndex {
    /// Load from a `.scip` protobuf file on disk.
    ///
    /// Returns `Err` if the file does not exist or cannot be read. Once a
    /// protobuf decoder (the `scip` crate or hand-rolled `prost` types) is
    /// wired in, this should populate `entities` and `edges` from the SCIP
    /// `Document`/`Occurrence`/`SymbolInformation` records.
    ///
    /// Currently: validates that the file is readable, then returns an
    /// `unimplemented` error. The trait + `from_refs` path is the load-bearing
    /// deliverable; native parsing lands in a follow-up.
    pub fn from_scip(path: &std::path::Path) -> anyhow::Result<Self> {
        // Surface a clear "file missing" error before anything else — tests
        // and tooling rely on this distinguishing existence from format.
        let _bytes = std::fs::read(path)
            .map_err(|e| anyhow::anyhow!("failed to read SCIP file {}: {e}", path.display()))?;

        // TODO: decode `_bytes` with the `scip` crate (or prost-generated
        // types) and populate `entities` / `edges`. See module docs.
        anyhow::bail!(
            "SCIP protobuf decoding is not yet implemented; use ScipIndex::from_refs for now"
        )
    }

    /// Build a `ScipIndex` from a pre-resolved list of references and edges.
    ///
    /// This is the fully-testable construction path and the one the rest of
    /// the codebase will exercise until native `.scip` decoding is wired in.
    /// Each `ScipEntityRef` becomes a `RawEntity`:
    ///
    /// - definition occurrences → `EntityType::NamedType`
    /// - reference occurrences  → `EntityType::ModulePath`
    /// - id = `"{file}:{start_line}:{hash(symbol)}"` (collision-safe, stable)
    pub fn from_refs(refs: Vec<ScipEntityRef>, edges: Vec<ScipEdge>) -> Self {
        let entities = refs
            .iter()
            .map(|r| RawEntity {
                id: format!("{}:{}:{}", r.file, r.start_line, fact_hash_str(&r.symbol)),
                entity_type: if r.is_definition {
                    EntityType::NamedType
                } else {
                    EntityType::ModulePath
                },
                text: r.display_name.clone(),
                span: (r.start_line, r.end_line),
                file: r.file.clone(),
                line: r.start_line,
            })
            .collect();
        let edge_tuples = edges
            .into_iter()
            .map(|e| (e.from_symbol, e.kind, e.to_symbol))
            .collect();
        Self {
            entities,
            edges: edge_tuples,
        }
    }
}

impl CodeEntityIndex for ScipIndex {
    fn entities(&self) -> &[RawEntity] {
        &self.entities
    }
    fn edges(&self) -> &[(String, EdgeKind, String)] {
        &self.edges
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scip_index_from_refs() {
        let refs = vec![ScipEntityRef {
            symbol: "rust-analyzer cargo foo/Bar#".into(),
            display_name: "Bar".into(),
            file: "src/bar.rs".into(),
            start_line: 10,
            end_line: 20,
            is_definition: true,
        }];
        let idx = ScipIndex::from_refs(refs, vec![]);
        assert_eq!(idx.entities().len(), 1);
        assert_eq!(idx.entities()[0].text, "Bar");
        assert_eq!(idx.edges().len(), 0);
    }

    #[test]
    fn scip_from_missing_file_returns_err() {
        let result = ScipIndex::from_scip(std::path::Path::new("/nonexistent/index.scip"));
        assert!(result.is_err());
    }

    #[test]
    fn scip_index_classifies_def_vs_ref() {
        let refs = vec![
            ScipEntityRef {
                symbol: "scip-rust cargo crate/Foo#".into(),
                display_name: "Foo".into(),
                file: "src/foo.rs".into(),
                start_line: 1,
                end_line: 1,
                is_definition: true,
            },
            ScipEntityRef {
                symbol: "scip-rust cargo crate/Foo#".into(),
                display_name: "Foo".into(),
                file: "src/use_foo.rs".into(),
                start_line: 5,
                end_line: 5,
                is_definition: false,
            },
        ];
        let edges = vec![ScipEdge {
            from_symbol: "scip-rust cargo crate/use_foo#".into(),
            to_symbol: "scip-rust cargo crate/Foo#".into(),
            kind: EdgeKind::UsesType,
        }];
        let idx = ScipIndex::from_refs(refs, edges);

        assert_eq!(idx.entities().len(), 2);
        assert!(matches!(
            idx.entities()[0].entity_type,
            EntityType::NamedType
        ));
        assert!(matches!(
            idx.entities()[1].entity_type,
            EntityType::ModulePath
        ));
        assert_eq!(idx.edges().len(), 1);
        assert!(matches!(idx.edges()[0].1, EdgeKind::UsesType));
    }
}
