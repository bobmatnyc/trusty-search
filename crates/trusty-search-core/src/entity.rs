//! Entity extraction from tree-sitter parse trees.
//!
//! Why: search and KG quality benefit from a typed view of the program's
//! "things" — type names, trait bounds, module paths, error sites, derives,
//! string literals and test relations. The chunker walks the AST anyway, so
//! emitting a flat `Vec<RawEntity>` in the same pass is essentially free.
//!
//! What: this module defines `EntityType`, `RawEntity`, and language-specific
//! extractors. The Rust extractor implements the full taxonomy from issue #17;
//! other languages currently extract `NamedType` and `ModulePath` only and
//! emit a `tracing::debug!` note for the unimplemented entity kinds.
//!
//! Test: see `#[cfg(test)]` in `chunker.rs` (`test_rust_entity_named_types`),
//! which round-trips a Rust source string through `chunk_ast` and asserts the
//! `NamedType` set.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tree_sitter::{Node, Tree};

/// Taxonomy of program entities surfaced from the AST.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum EntityType {
    /// Type identifiers (`Arc`, `Vec`, `CodeChunk`, …).
    NamedType,
    /// Trait bound expressions (`Send + Sync`, `Serialize`, …).
    TraitBound,
    /// Module paths (`crate::indexer::CodeIndexer`, `std::sync::Arc`).
    ModulePath,
    /// Error/panic call sites: `bail!`, `anyhow!`, `panic!`, `unwrap`.
    ErrorVariant,
    /// Identifiers referenced from `#[test]` function bodies.
    TestRelation,
    /// Doc-comment derived concept (NLP phrase / keyword).
    DocConcept,
    /// Attribute annotations (`#[derive(...)]`, `#[cfg(...)]`).
    Annotation,
    /// String literals longer than 10 characters.
    LiteralString,
    /// `type Foo = Bar` aliases.
    TypeAlias,
    /// Top-level `const`/`static` symbol.
    ConstantSymbol,
    /// Top-level `use` of a non-stdlib, non-self/super/crate path.
    ExternalCrate,
    /// Cluster of co-occurring concepts (Phase C).
    ConceptCluster,
    /// Free-form natural-language phrase pulled from docs/comments.
    NaturalLanguagePhrase,
}

impl EntityType {
    fn as_str(&self) -> &'static str {
        match self {
            Self::NamedType => "NamedType",
            Self::TraitBound => "TraitBound",
            Self::ModulePath => "ModulePath",
            Self::ErrorVariant => "ErrorVariant",
            Self::TestRelation => "TestRelation",
            Self::DocConcept => "DocConcept",
            Self::Annotation => "Annotation",
            Self::LiteralString => "LiteralString",
            Self::TypeAlias => "TypeAlias",
            Self::ConstantSymbol => "ConstantSymbol",
            Self::ExternalCrate => "ExternalCrate",
            Self::ConceptCluster => "ConceptCluster",
            Self::NaturalLanguagePhrase => "NaturalLanguagePhrase",
        }
    }
}

/// Edge kinds for the `SymbolGraph` knowledge graph.
///
/// Phase A = structural (tree-sitter derived)
/// Phase B = test-relation
/// Phase C = doc/concept
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum EdgeKind {
    // Call graph
    /// Caller → callee.
    CallsFunction,
    /// Callee → caller (reverse index of `CallsFunction`).
    CalledByFunction,
    // Phase A — structural
    Implements,
    UsesType,
    Derives,
    ModuleContains,
    ReExports,
    RaisesError,
    Configures,
    // Phase B — test relations
    TestedBy,
    TestUsesFixture,
    CoOccursInTest,
    // Phase C — docs / concepts
    Documents,
    ReferencesConcept,
    Aliases,
    ErrorDescribes,
}

impl EdgeKind {
    /// Score multiplier for KG expansion. Higher = more relevant when ranking
    /// neighbours discovered by walking this edge.
    pub fn score_multiplier(&self) -> f32 {
        match self {
            EdgeKind::Implements => 0.85,
            EdgeKind::UsesType => 0.75,
            EdgeKind::TestedBy => 0.80,
            EdgeKind::Documents => 0.65,
            EdgeKind::ReferencesConcept => 0.60,
            // Remaining edges use the legacy flat KG-expansion multiplier.
            _ => 0.70,
        }
    }
}

/// redb table name constants for entity storage.
pub mod tables {
    /// `entity_id (str) -> RawEntity (bincode/json)`
    pub const ENTITIES: &str = "entities";
    /// `(from_entity_id, edge_kind, to_entity_id) -> ()`
    pub const ENTITY_EDGES: &str = "entity_edges";
    /// `chunk_id -> Vec<entity_id>`
    pub const CHUNK_ENTITIES: &str = "chunk_entities";
    /// `entity_id -> Vec<chunk_id>` (reverse index of `CHUNK_ENTITIES`)
    pub const ENTITY_CHUNKS: &str = "entity_chunks";
}

/// One extracted entity, anchored to a byte span and source line.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawEntity {
    /// Stable hash of (entity_type, text, file).
    pub id: String,
    pub entity_type: EntityType,
    pub text: String,
    pub span: (usize, usize),
    pub file: String,
    pub line: usize,
}

impl RawEntity {
    fn new(entity_type: EntityType, text: String, span: (usize, usize), file: &str, line: usize) -> Self {
        let mut h = Sha256::new();
        h.update(entity_type.as_str().as_bytes());
        h.update(b"\0");
        h.update(text.as_bytes());
        h.update(b"\0");
        h.update(file.as_bytes());
        let id = format!("{:x}", h.finalize());
        Self {
            id,
            entity_type,
            text,
            span,
            file: file.to_string(),
            line,
        }
    }
}

/// Slice the source text for a node and return it as an owned string.
fn node_text(node: Node<'_>, src: &[u8]) -> String {
    std::str::from_utf8(&src[node.start_byte()..node.end_byte()])
        .unwrap_or("")
        .to_string()
}

/// Public entry point: walk `tree` and emit entities for `lang`.
pub fn extract_entities(tree: &Tree, src: &[u8], file: &str, lang: &str) -> Vec<RawEntity> {
    match lang {
        "rust" => extract_rust(tree, src, file),
        // Other languages: NamedType + ModulePath best-effort.
        "python" | "javascript" | "typescript" | "go" | "java" | "c" | "cpp" => {
            tracing::debug!("entity extraction not fully implemented for {lang}");
            extract_universal(tree, src, file)
        }
        _ => Vec::new(),
    }
}

/// Universal extractor: looks for `type_identifier`-ish nodes and `scoped_identifier`-ish
/// chains. Used as a stub for non-Rust languages.
fn extract_universal(tree: &Tree, src: &[u8], file: &str) -> Vec<RawEntity> {
    let mut out = Vec::new();
    let mut stack: Vec<Node> = vec![tree.root_node()];
    while let Some(node) = stack.pop() {
        let kind = node.kind();
        if kind == "type_identifier" || kind == "type" {
            let text = node_text(node, src);
            if !text.is_empty() {
                out.push(RawEntity::new(
                    EntityType::NamedType,
                    text,
                    (node.start_byte(), node.end_byte()),
                    file,
                    node.start_position().row + 1,
                ));
            }
        } else if kind == "scoped_identifier" || kind == "qualified_identifier" {
            let text = node_text(node, src);
            if text.contains("::") || text.contains('.') {
                out.push(RawEntity::new(
                    EntityType::ModulePath,
                    text,
                    (node.start_byte(), node.end_byte()),
                    file,
                    node.start_position().row + 1,
                ));
            }
        }
        let mut walker = node.walk();
        for child in node.children(&mut walker) {
            stack.push(child);
        }
    }
    out
}

/// Rust extractor. Implements the full taxonomy from issue #17.
fn extract_rust(tree: &Tree, src: &[u8], file: &str) -> Vec<RawEntity> {
    let mut out = Vec::new();
    let root = tree.root_node();

    // Top-level `use` declarations: classify ExternalCrate vs ModulePath.
    let mut top_cursor = root.walk();
    for child in root.children(&mut top_cursor) {
        if child.kind() == "use_declaration" {
            let text = node_text(child, src);
            // first path segment after `use ` and before `::` or whitespace
            let trimmed = text.trim_start_matches("use ").trim_end_matches(';').trim();
            let first = trimmed
                .split(|c: char| c == ':' || c.is_whitespace() || c == '{' || c == ',')
                .find(|s| !s.is_empty())
                .unwrap_or("");
            let line = child.start_position().row + 1;
            let span = (child.start_byte(), child.end_byte());
            if !first.is_empty()
                && !matches!(first, "crate" | "super" | "self" | "std" | "core" | "alloc")
            {
                out.push(RawEntity::new(
                    EntityType::ExternalCrate,
                    first.to_string(),
                    span,
                    file,
                    line,
                ));
            }
            out.push(RawEntity::new(
                EntityType::ModulePath,
                trimmed.to_string(),
                span,
                file,
                line,
            ));
        }
    }

    // Recursive walk for the rest.
    walk_rust(root, src, file, false, &mut out);
    out
}

fn walk_rust(node: Node<'_>, src: &[u8], file: &str, in_test_fn: bool, out: &mut Vec<RawEntity>) {
    let kind = node.kind();
    let line = node.start_position().row + 1;
    let span = (node.start_byte(), node.end_byte());

    match kind {
        "type_identifier" => {
            let t = node_text(node, src);
            if !t.is_empty() {
                out.push(RawEntity::new(EntityType::NamedType, t, span, file, line));
            }
        }
        "trait_bounds" => {
            let t = node_text(node, src);
            out.push(RawEntity::new(EntityType::TraitBound, t, span, file, line));
        }
        "scoped_identifier" => {
            let t = node_text(node, src);
            if t.contains("::") {
                out.push(RawEntity::new(EntityType::ModulePath, t, span, file, line));
            }
        }
        "macro_invocation" => {
            // e.g. `bail!(...)`, `anyhow::bail!(...)`, `panic!(...)`. The `macro`
            // field can be either an `identifier` or a `scoped_identifier`; we
            // care about the final segment.
            if let Some(name_node) = node.child_by_field_name("macro") {
                let name = node_text(name_node, src);
                let last = name.rsplit("::").next().unwrap_or(&name).trim();
                if matches!(last, "bail" | "anyhow" | "panic" | "unwrap" | "expect") {
                    let t = node_text(node, src);
                    out.push(RawEntity::new(EntityType::ErrorVariant, t, span, file, line));
                }
            }
        }
        "call_expression" => {
            // `.unwrap()` and `.expect()` method calls also count.
            if let Some(func) = node.child_by_field_name("function") {
                let txt = node_text(func, src);
                let last = txt.rsplit('.').next().unwrap_or(&txt);
                if matches!(last, "unwrap" | "expect") {
                    let t = node_text(node, src);
                    out.push(RawEntity::new(EntityType::ErrorVariant, t, span, file, line));
                }
            }
        }
        "attribute_item" | "inner_attribute_item" => {
            let t = node_text(node, src);
            out.push(RawEntity::new(EntityType::Annotation, t, span, file, line));
        }
        "string_literal" => {
            let t = node_text(node, src);
            // Strip surrounding quotes for length check.
            let inner = t.trim_matches('"');
            if inner.len() > 10 {
                out.push(RawEntity::new(EntityType::LiteralString, t, span, file, line));
            }
        }
        "type_item" => {
            let t = node_text(node, src);
            out.push(RawEntity::new(EntityType::TypeAlias, t, span, file, line));
        }
        "identifier" if in_test_fn => {
            let t = node_text(node, src);
            if !t.is_empty() {
                out.push(RawEntity::new(EntityType::TestRelation, t, span, file, line));
            }
        }
        _ => {}
    }

    // Detect entry into a test function so identifiers inside count as TestRelation.
    let entering_test_fn = kind == "function_item" && function_has_test_attr(node, src);

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        walk_rust(child, src, file, in_test_fn || entering_test_fn, out);
    }
}

/// Returns true if any preceding `attribute_item` sibling on this `function_item`
/// includes the `test` attribute. Tree-sitter-rust attaches attributes as
/// previous siblings, not as children of `function_item`.
fn function_has_test_attr(node: Node<'_>, src: &[u8]) -> bool {
    let mut prev = node.prev_sibling();
    while let Some(p) = prev {
        let k = p.kind();
        if k == "attribute_item" || k == "inner_attribute_item" {
            let t = node_text(p, src);
            if t.contains("test") {
                return true;
            }
            prev = p.prev_sibling();
        } else if k == "line_comment" || k == "block_comment" {
            prev = p.prev_sibling();
        } else {
            break;
        }
    }
    false
}
