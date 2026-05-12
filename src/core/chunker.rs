//! AST-aware code chunker built on tree-sitter.
//!
//! Why: a sliding-window chunker fragments declarations and produces noisy
//! BM25/vector candidates because a single function may straddle two windows.
//! AST-aware chunking yields one chunk per top-level declaration, making
//! `function_name`, `chunk_type`, and `calls` accurate enough to drive both
//! semantic search and the knowledge-graph CALLS edges (#5, #17).
//!
//! What: `chunk_ast(file, content, language) -> (Vec<RawChunk>, Vec<RawEntity>)`
//! parses with tree-sitter, walks top-level declarations into chunks, populates
//! per-chunk fields (calls, inherits_from, nlp_keywords, …), splits oversized
//! chunks into sub-chunks with stable parent IDs, and emits a flat entity list
//! in the same pass. Unknown extensions fall back to `chunk_text()`.
//!
//! Test: see `#[cfg(test)]` below — covers function/method chunking, qualified
//! method names, calls extraction, named-type entities, large-function
//! splitting, unknown-language fallback, and doc-comment NLP keywords.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use tree_sitter::{Language, Node, Parser};

use crate::core::entity::{extract_entities, RawEntity};

/// Coarse classification of an AST chunk.
///
/// `Default` is `Unknown` so chunks deserialized from older index versions
/// (which lacked `chunk_type`) round-trip cleanly.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum ChunkType {
    #[default]
    Unknown,
    Function,
    Method,
    Class,
    Struct,
    Impl,
    Module,
    Trait,
    Enum,
    Test,
    Constant,
    TypeAlias,
    Docstring,
    /// Free-form code that doesn't fit a more specific category.
    FreeCode,
    /// Legacy alias for `FreeCode` — retained for backwards-compatible deserialization.
    Code,
}

impl ChunkType {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Unknown => "Unknown",
            Self::Function => "Function",
            Self::Method => "Method",
            Self::Class => "Class",
            Self::Struct => "Struct",
            Self::Impl => "Impl",
            Self::Module => "Module",
            Self::Trait => "Trait",
            Self::Enum => "Enum",
            Self::Test => "Test",
            Self::Constant => "Constant",
            Self::TypeAlias => "TypeAlias",
            Self::Docstring => "Docstring",
            Self::FreeCode => "FreeCode",
            Self::Code => "Code",
        }
    }
}

/// A code chunk extracted from a source file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawChunk {
    pub id: String,
    pub file: String,
    pub start_line: usize,
    pub end_line: usize,
    pub content: String,
    pub function_name: Option<String>,
    pub language: Option<String>,

    // Issue #4 / #17 additions
    pub chunk_type: ChunkType,
    pub calls: Vec<String>,
    pub inherits_from: Vec<String>,
    pub chunk_depth: usize,
    pub parent_chunk_id: Option<String>,
    pub child_chunk_ids: Vec<String>,
    pub nlp_keywords: Vec<String>,
    pub nlp_code_refs: Vec<String>,

    /// Entity-derived virtual terms appended to this chunk's BM25 document
    /// at index time (issue #19). Not displayed to users; used only to give
    /// BM25 extra surface area to match symbolic queries against.
    #[serde(default)]
    pub virtual_terms: Vec<String>,
}

impl RawChunk {
    /// Build a generic `Code` chunk — used by `chunk_text` and the unknown-extension fallback.
    fn generic(
        id: String,
        file: String,
        start_line: usize,
        end_line: usize,
        content: String,
    ) -> Self {
        Self {
            id,
            file,
            start_line,
            end_line,
            content,
            function_name: None,
            language: None,
            chunk_type: ChunkType::Code,
            calls: Vec::new(),
            inherits_from: Vec::new(),
            chunk_depth: 0,
            parent_chunk_id: None,
            child_chunk_ids: Vec::new(),
            nlp_keywords: Vec::new(),
            nlp_code_refs: Vec::new(),
            virtual_terms: Vec::new(),
        }
    }
}

/// Maximum lines for a single AST chunk before we split into sub-chunks.
const MAX_CHUNK_LINES: usize = 200;
/// Sub-chunk window (used when splitting oversized AST chunks).
const SUB_CHUNK_WINDOW: usize = 100;
/// Sub-chunk stride.
const SUB_CHUNK_STRIDE: usize = 50;

/// Overlapping sliding-window chunker. Retained for unknown extensions and as
/// the backing routine for sub-chunking oversized AST chunks.
pub fn chunk_text(file: &str, content: &str, window: usize, stride: usize) -> Vec<RawChunk> {
    let lines: Vec<&str> = content.lines().collect();
    let mut chunks = Vec::new();
    let mut start = 0usize;

    while start < lines.len() {
        let end = (start + window).min(lines.len());
        let text = lines[start..end].join("\n");
        chunks.push(RawChunk::generic(
            format!("{}:{}:{}", file, start + 1, end),
            file.to_string(),
            start + 1,
            end,
            text,
        ));
        if end == lines.len() {
            break;
        }
        start += stride;
    }
    chunks
}

/// Map a file extension to a (language_tag, tree-sitter `Language`).
fn language_for(file: &str) -> Option<(&'static str, Language)> {
    let ext = std::path::Path::new(file)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let (tag, lang_fn): (&'static str, tree_sitter_language::LanguageFn) = match ext.as_str() {
        "rs" => ("rust", tree_sitter_rust::LANGUAGE),
        "py" => ("python", tree_sitter_python::LANGUAGE),
        "js" | "mjs" | "cjs" | "jsx" => ("javascript", tree_sitter_javascript::LANGUAGE),
        "ts" => ("typescript", tree_sitter_typescript::LANGUAGE_TYPESCRIPT),
        "tsx" => ("typescript", tree_sitter_typescript::LANGUAGE_TSX),
        "go" => ("go", tree_sitter_go::LANGUAGE),
        "java" => ("java", tree_sitter_java::LANGUAGE),
        "c" | "h" => ("c", tree_sitter_c::LANGUAGE),
        "cpp" | "cc" | "cxx" | "hpp" | "hh" | "hxx" => ("cpp", tree_sitter_cpp::LANGUAGE),
        "rb" => ("ruby", tree_sitter_ruby::LANGUAGE),
        "php" => ("php", tree_sitter_php::LANGUAGE_PHP),
        "scala" => ("scala", tree_sitter_scala::LANGUAGE),
        "cs" => ("csharp", tree_sitter_c_sharp::LANGUAGE),
        "kt" | "kts" => ("kotlin", tree_sitter_kotlin_ng::LANGUAGE),
        "swift" => ("swift", tree_sitter_swift::LANGUAGE),
        _ => return None,
    };
    Some((tag, lang_fn.into()))
}

/// Compute byte ranges → 1-based line numbers from the source bytes.
fn line_for_byte(line_offsets: &[usize], byte: usize) -> usize {
    // line_offsets[i] = byte offset of the start of (1-based) line i+1
    match line_offsets.binary_search(&byte) {
        Ok(i) => i + 1,
        Err(i) => i.max(1),
    }
}

fn build_line_offsets(src: &[u8]) -> Vec<usize> {
    let mut v = vec![0usize];
    for (i, b) in src.iter().enumerate() {
        if *b == b'\n' {
            v.push(i + 1);
        }
    }
    v
}

/// Stable, content-aware chunk ID. Falls back to position when no name is available.
fn make_chunk_id(
    file: &str,
    chunk_type: &ChunkType,
    name: &str,
    start_line: usize,
    end_line: usize,
) -> String {
    if name.is_empty() {
        format!("{file}:{start_line}:{end_line}")
    } else {
        format!("{file}::{}::{name}::{start_line}", chunk_type.as_str())
    }
}

/// Find the first `name` field child of an AST node (works across most
/// tree-sitter grammars where declarations expose a `name` field).
fn name_of(node: Node<'_>, src: &[u8]) -> String {
    if let Some(n) = node.child_by_field_name("name") {
        return std::str::from_utf8(&src[n.start_byte()..n.end_byte()])
            .unwrap_or("")
            .to_string();
    }
    String::new()
}

/// Collect call expressions reachable inside `node` without descending into
/// nested function/method bodies (so a parent function doesn't claim its
/// inner-fn's calls).
fn collect_calls(node: Node<'_>, src: &[u8], lang: &str) -> Vec<String> {
    let mut out: HashSet<String> = HashSet::new();
    let mut stack: Vec<Node> = vec![node];
    while let Some(n) = stack.pop() {
        let kind = n.kind();
        let is_fn_kind = matches!(
            (lang, kind),
            ("rust", "function_item")
                | ("python", "function_definition")
                | ("javascript", "function_declaration")
                | ("typescript", "function_declaration")
                | ("go", "function_declaration")
                | ("java", "method_declaration")
                | ("c" | "cpp", "function_definition")
                | ("ruby", "method")
                | ("ruby", "singleton_method")
                | ("php", "function_definition")
                | ("php", "method_declaration")
                | ("scala", "function_definition")
                | ("csharp", "method_declaration")
                | ("csharp", "constructor_declaration")
                | ("kotlin", "function_declaration")
                | ("kotlin", "secondary_constructor")
                | ("swift", "function_declaration")
                | ("swift", "init_declaration")
                | ("swift", "protocol_function_declaration")
        );

        // Don't descend into nested function bodies (we treat them as their own chunks).
        if is_fn_kind && n.id() != node.id() {
            continue;
        }

        // Tree-sitter call node names per language.
        let is_call = matches!(
            (lang, kind),
            ("rust", "call_expression")
                | ("python", "call")
                | ("javascript" | "typescript", "call_expression")
                | ("go", "call_expression")
                | ("java", "method_invocation")
                | ("c" | "cpp", "call_expression")
                | ("ruby", "call")
                | ("php", "function_call_expression")
                | ("php", "member_call_expression")
                | ("php", "scoped_call_expression")
                | ("php", "nullsafe_member_call_expression")
                | ("scala", "call_expression")
                | ("csharp", "invocation_expression")
                | ("kotlin", "call_expression")
                | ("swift", "call_expression")
        );

        if is_call {
            // function/method name field varies; try common ones.
            let callee = n
                .child_by_field_name("function")
                .or_else(|| n.child_by_field_name("name"))
                .or_else(|| n.child(0));
            if let Some(c) = callee {
                let raw = std::str::from_utf8(&src[c.start_byte()..c.end_byte()])
                    .unwrap_or("")
                    .to_string();
                // Reduce `foo::bar::baz` and `obj.method` to the last identifier
                // segment for simple-name matching, but also keep the full path
                // so KG resolution can prefer qualified matches.
                let simple = raw
                    .rsplit(['.', ':'])
                    .next()
                    .unwrap_or(&raw)
                    .trim()
                    .to_string();
                if !simple.is_empty() {
                    out.insert(simple);
                }
            }
        }

        let mut cursor = n.walk();
        for child in n.children(&mut cursor) {
            stack.push(child);
        }
    }
    let mut v: Vec<String> = out.into_iter().collect();
    v.sort();
    v
}

/// Collect inherited names: trait list for `impl` blocks, base types for `struct_item`.
fn collect_inherits(node: Node<'_>, src: &[u8], lang: &str) -> Vec<String> {
    let mut out = Vec::new();
    if lang == "rust" && node.kind() == "impl_item" {
        // `impl Trait for Type` — `trait` field is the trait name (if present).
        if let Some(t) = node.child_by_field_name("trait") {
            out.push(
                std::str::from_utf8(&src[t.start_byte()..t.end_byte()])
                    .unwrap_or("")
                    .to_string(),
            );
        }
    } else if lang == "python" && node.kind() == "class_definition" {
        if let Some(s) = node.child_by_field_name("superclasses") {
            let txt = std::str::from_utf8(&src[s.start_byte()..s.end_byte()])
                .unwrap_or("")
                .trim_matches(|c: char| c == '(' || c == ')')
                .to_string();
            for part in txt.split(',') {
                let p = part.trim();
                if !p.is_empty() {
                    out.push(p.to_string());
                }
            }
        }
    } else if lang == "scala"
        && matches!(
            node.kind(),
            "class_definition" | "object_definition" | "trait_definition"
        )
    {
        // Phase 2 (issue #55): Scala's `extends T1 with T2 with T3` is encoded
        // as an `extends_clause` containing one or more `type_identifier`
        // children. Each becomes an `Implements` edge in the SymbolGraph.
        let mut cur = node.walk();
        for child in node.children(&mut cur) {
            if child.kind() != "extends_clause" {
                continue;
            }
            let mut cur2 = child.walk();
            for sub in child.children(&mut cur2) {
                if sub.kind() == "type_identifier" {
                    let t = std::str::from_utf8(&src[sub.start_byte()..sub.end_byte()])
                        .unwrap_or("")
                        .to_string();
                    if !t.is_empty() {
                        out.push(t);
                    }
                }
            }
        }
    } else if lang == "php"
        && matches!(
            node.kind(),
            "class_declaration" | "interface_declaration" | "trait_declaration"
        )
    {
        // Phase 2 (issue #49): PHP exposes inheritance via two siblings of
        // `class_declaration`:
        //   - `base_clause`  → `extends Parent` (single parent class)
        //   - `class_interface_clause` → `implements I1, I2` (interfaces)
        // For `interface_declaration`, `base_clause` carries `extends I1, I2`
        // (interfaces can extend multiple). We walk both kinds and extract
        // every `name`-kind child as a parent symbol.
        let mut cur = node.walk();
        for child in node.children(&mut cur) {
            if !matches!(child.kind(), "base_clause" | "class_interface_clause") {
                continue;
            }
            let mut cur2 = child.walk();
            for sub in child.children(&mut cur2) {
                // `name` is the type identifier; ignore keyword tokens like
                // `extends`/`implements` and punctuation `,`.
                if sub.kind() == "name" || sub.kind() == "qualified_name" {
                    let t = std::str::from_utf8(&src[sub.start_byte()..sub.end_byte()])
                        .unwrap_or("")
                        .to_string();
                    if !t.is_empty() {
                        out.push(t);
                    }
                }
            }
        }
    }
    out.retain(|s| !s.is_empty());
    out
}

/// Pull doc-comment text immediately preceding `node` (Rust `///` and `//!`).
fn preceding_doc_comments(node: Node<'_>, src: &[u8]) -> String {
    let mut buf = String::new();
    let mut prev = node.prev_sibling();
    while let Some(p) = prev {
        if p.kind() == "line_comment" || p.kind() == "block_comment" {
            let txt = std::str::from_utf8(&src[p.start_byte()..p.end_byte()]).unwrap_or("");
            if txt.starts_with("///") || txt.starts_with("//!") || txt.starts_with("/**") {
                buf.insert_str(0, txt);
                buf.insert(0, '\n');
            }
            prev = p.prev_sibling();
        } else {
            break;
        }
    }
    buf
}

/// Cheap noun-phrase-ish keyword extraction from doc comments.
fn nlp_from_doc(doc: &str) -> (Vec<String>, Vec<String>) {
    let mut keywords: Vec<String> = Vec::new();
    let mut code_refs: Vec<String> = Vec::new();
    let mut in_backticks = false;
    let mut buf = String::new();
    // Backtick-delimited code refs.
    for ch in doc.chars() {
        if ch == '`' {
            if in_backticks && !buf.is_empty() {
                code_refs.push(buf.clone());
            }
            buf.clear();
            in_backticks = !in_backticks;
        } else if in_backticks {
            buf.push(ch);
        }
    }
    // Title-case or all-caps acronym words of length >= 3, outside backticks.
    let mut depth = 0;
    for word in doc.split(|c: char| !c.is_alphanumeric() && c != '_') {
        if word.contains('`') {
            depth = if depth == 0 { 1 } else { 0 };
            continue;
        }
        if word.len() < 3 {
            continue;
        }
        let first = word.chars().next().unwrap();
        let all_upper = word
            .chars()
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit());
        let title =
            first.is_ascii_uppercase() && word.chars().skip(1).any(|c| c.is_ascii_lowercase());
        if all_upper || title {
            keywords.push(word.to_string());
        }
    }
    keywords.sort();
    keywords.dedup();
    code_refs.sort();
    code_refs.dedup();
    (keywords, code_refs)
}

/// Swift's tree-sitter grammar folds `struct`, `enum`, and `extension`
/// declarations into the same `class_declaration` node kind, distinguished
/// by the first child being the keyword token (kind `class`/`struct`/etc.).
/// Map that keyword to the appropriate `ChunkType`.
fn swift_class_decl_kind(node: Node<'_>) -> ChunkType {
    let kw = node
        .child(0)
        .map(|c| c.kind().to_string())
        .unwrap_or_default();
    match kw.as_str() {
        "struct" => ChunkType::Struct,
        "enum" => ChunkType::Enum,
        "extension" => ChunkType::Module,
        _ => ChunkType::Class, // includes `class`, `actor`, fallback
    }
}

/// Per-language: AST node kinds we promote to top-level chunks, plus their
/// (default `ChunkType`, parent-context-overrides).
fn classify_node(lang: &str, node: Node<'_>) -> Option<ChunkType> {
    let kind = node.kind();
    let parent_kind = node.parent().map(|p| p.kind()).unwrap_or("");
    Some(match (lang, kind) {
        ("rust", "function_item") => {
            // Method if inside `impl_item` / `trait_item`.
            if matches!(parent_kind, "declaration_list" | "impl_item" | "trait_item")
                || ancestor_kind(node, "impl_item").is_some()
            {
                ChunkType::Method
            } else {
                ChunkType::Function
            }
        }
        ("rust", "impl_item") => ChunkType::Impl,
        ("rust", "struct_item") => ChunkType::Class,
        ("rust", "trait_item") => ChunkType::Trait,
        ("rust", "enum_item") => ChunkType::Enum,
        ("rust", "mod_item") => ChunkType::Module,

        ("python", "function_definition") => {
            if ancestor_kind(node, "class_definition").is_some() {
                ChunkType::Method
            } else {
                ChunkType::Function
            }
        }
        ("python", "class_definition") => ChunkType::Class,
        ("python", "decorated_definition") => return None, // descend to inner

        ("javascript" | "typescript", "function_declaration") => ChunkType::Function,
        ("javascript" | "typescript", "class_declaration") => ChunkType::Class,
        ("javascript" | "typescript", "method_definition") => ChunkType::Method,

        ("go", "function_declaration") => ChunkType::Function,
        ("go", "method_declaration") => ChunkType::Method,
        ("go", "type_declaration") => ChunkType::Class,

        ("java", "method_declaration") => ChunkType::Method,
        ("java", "class_declaration") => ChunkType::Class,
        ("java", "interface_declaration") => ChunkType::Trait,

        ("c" | "cpp", "function_definition") => ChunkType::Function,
        ("cpp", "class_specifier") => ChunkType::Class,
        ("c" | "cpp", "struct_specifier") => ChunkType::Class,

        ("ruby", "method") => ChunkType::Function,
        ("ruby", "singleton_method") => ChunkType::Method,
        ("ruby", "module") => ChunkType::Module,
        ("ruby", "class") => ChunkType::Class,

        ("php", "function_definition") => ChunkType::Function,
        ("php", "method_declaration") => ChunkType::Method,
        ("php", "class_declaration") => ChunkType::Class,
        ("php", "interface_declaration") => ChunkType::Trait,
        ("php", "trait_declaration") => ChunkType::Trait,
        ("php", "namespace_definition") => ChunkType::Module,

        ("scala", "function_definition") => {
            // Phase 2 (issue #55): a `def` inside a class/object/trait template
            // body is a method; standalone defs (top-level or inside another
            // function) are plain functions.
            if ancestor_kind(node, "class_definition").is_some()
                || ancestor_kind(node, "object_definition").is_some()
                || ancestor_kind(node, "trait_definition").is_some()
            {
                ChunkType::Method
            } else {
                ChunkType::Function
            }
        }
        ("scala", "class_definition") => ChunkType::Class,
        ("scala", "object_definition") => ChunkType::Class,
        ("scala", "trait_definition") => ChunkType::Trait,

        // C#
        ("csharp", "method_declaration") => {
            if ancestor_kind(node, "class_declaration").is_some()
                || ancestor_kind(node, "interface_declaration").is_some()
                || ancestor_kind(node, "struct_declaration").is_some()
            {
                ChunkType::Method
            } else {
                ChunkType::Function
            }
        }
        ("csharp", "constructor_declaration") => ChunkType::Method,
        ("csharp", "class_declaration") => ChunkType::Class,
        ("csharp", "interface_declaration") => ChunkType::Trait,
        ("csharp", "struct_declaration") => ChunkType::Class,
        ("csharp", "namespace_declaration") => ChunkType::Module,
        ("csharp", "enum_declaration") => ChunkType::Enum,

        // Kotlin (tree-sitter-kotlin-ng grammar)
        ("kotlin", "function_declaration") => {
            if ancestor_kind(node, "class_declaration").is_some()
                || ancestor_kind(node, "object_declaration").is_some()
            {
                ChunkType::Method
            } else {
                ChunkType::Function
            }
        }
        ("kotlin", "secondary_constructor") => ChunkType::Method,
        ("kotlin", "class_declaration") => ChunkType::Class,
        ("kotlin", "object_declaration") => ChunkType::Class,
        ("kotlin", "companion_object") => ChunkType::Class,
        ("kotlin", "interface_declaration") => ChunkType::Trait,

        // Swift: tree-sitter-swift folds struct/enum/extension into
        // `class_declaration`, distinguished by the keyword token at child(0).
        ("swift", "class_declaration") => swift_class_decl_kind(node),
        ("swift", "protocol_declaration") => ChunkType::Trait,
        ("swift", "function_declaration") | ("swift", "protocol_function_declaration") => {
            if ancestor_kind(node, "class_declaration").is_some()
                || ancestor_kind(node, "protocol_declaration").is_some()
            {
                ChunkType::Method
            } else {
                ChunkType::Function
            }
        }
        ("swift", "init_declaration") => ChunkType::Method,

        _ => return None,
    })
}

fn ancestor_kind<'a>(node: Node<'a>, kind: &str) -> Option<Node<'a>> {
    let mut cur = node.parent();
    while let Some(c) = cur {
        if c.kind() == kind {
            return Some(c);
        }
        cur = c.parent();
    }
    None
}

/// For Rust methods: walk up to the enclosing `impl_item` and grab its `type` field.
fn rust_impl_type_name(node: Node<'_>, src: &[u8]) -> Option<String> {
    let imp = ancestor_kind(node, "impl_item")?;
    let t = imp.child_by_field_name("type")?;
    Some(
        std::str::from_utf8(&src[t.start_byte()..t.end_byte()])
            .unwrap_or("")
            .to_string(),
    )
}

/// For Scala methods (issue #55): walk up to the enclosing
/// `class_definition` / `object_definition` / `trait_definition` and grab its
/// `name` field so the method can be qualified as `ClassName::methodName`.
///
/// Why: Phase 2 caller-scoped call edges need stable, container-qualified
/// symbol names. Without qualification, `Foo.bar` and `Baz.bar` collide in the
/// symbol graph and `callers_of("bar")` returns false positives.
/// What: returns the first ancestor template owner's `name` field text, or
/// None for top-level / standalone defs.
/// Test: covered by `test_scala_method_qualified_name`.
fn scala_enclosing_class_name(node: Node<'_>, src: &[u8]) -> Option<String> {
    let owner = ancestor_kind_any(
        node,
        &["class_definition", "object_definition", "trait_definition"],
    )?;
    let n = owner.child_by_field_name("name")?;
    Some(
        std::str::from_utf8(&src[n.start_byte()..n.end_byte()])
            .unwrap_or("")
            .to_string(),
    )
}

/// For PHP methods (issue #49): walk up to the enclosing
/// `class_declaration` / `interface_declaration` / `trait_declaration` and
/// grab its `name` field so the method can be qualified as
/// `ClassName::methodName`.
///
/// Why: same rationale as `scala_enclosing_class_name` — symbol-graph
/// uniqueness across classes that share method names like `handle` or `run`.
/// What: returns the first ancestor declaration's `name` field text, or None
/// for free functions.
/// Test: covered by `test_php_method_qualified_name`.
fn php_enclosing_class_name(node: Node<'_>, src: &[u8]) -> Option<String> {
    let owner = ancestor_kind_any(
        node,
        &[
            "class_declaration",
            "interface_declaration",
            "trait_declaration",
        ],
    )?;
    let n = owner.child_by_field_name("name")?;
    Some(
        std::str::from_utf8(&src[n.start_byte()..n.end_byte()])
            .unwrap_or("")
            .to_string(),
    )
}

/// Like `ancestor_kind`, but accepts a slice of candidate node kinds.
///
/// Why: Scala and PHP each have several container kinds (class/object/trait
/// for Scala; class/interface/trait for PHP) that can own a method. A single
/// `ancestor_kind` call per kind would do three linear walks; this collapses
/// them into one.
/// What: walks up from `node` and returns the first ancestor whose kind appears
/// in `kinds`.
/// Test: indirectly covered by `test_scala_method_qualified_name` and
/// `test_php_method_qualified_name`.
fn ancestor_kind_any<'a>(node: Node<'a>, kinds: &[&str]) -> Option<Node<'a>> {
    let mut cur = node.parent();
    while let Some(c) = cur {
        if kinds.contains(&c.kind()) {
            return Some(c);
        }
        cur = c.parent();
    }
    None
}

/// Maximum lines for a JSON file to be indexed as a single chunk. Files
/// larger than this are skipped (JSON is hard to chunk meaningfully).
const JSON_MAX_LINES: usize = 500;

/// Maximum lines per plaintext / log chunk. Long paragraphs are split.
const PLAINTEXT_MAX_LINES: usize = 50;

/// Structured document chunker.
///
/// Why: code-aware chunking is the wrong tool for prose, config, and log
/// formats; sliding-window chunking shreds heading structure. Format-aware
/// chunking yields semantically coherent BM25/vector candidates.
/// What: dispatches on extension to per-format chunkers:
///   - md/mdx  → section-per-heading
///   - yaml/yml → top-level key sections
///   - toml    → `[section]` blocks
///   - json    → whole file if < 500 lines, otherwise skip
///   - txt/log → blank-line paragraphs, capped at 50 lines/chunk
///   - xml     → top-level child elements
///
/// Returns `None` for unknown extensions so the caller can fall back to the
/// sliding-window chunker.
///
/// Test: see `test_chunk_markdown_*`, `test_chunk_yaml_*`, etc. below.
pub fn chunk_document(file: &str, content: &str) -> Option<Vec<RawChunk>> {
    let ext = std::path::Path::new(file)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let chunks = match ext.as_str() {
        "md" | "mdx" => chunk_markdown(file, content),
        "yaml" | "yml" => chunk_yaml(file, content),
        "toml" => chunk_toml(file, content),
        "json" => chunk_json(file, content)?,
        "txt" | "log" => chunk_plaintext(file, content),
        "xml" => chunk_xml(file, content),
        _ => return None,
    };
    Some(chunks)
}

/// Build a generic document chunk with a specific language tag and chunk type.
fn document_chunk(
    file: &str,
    start_line: usize,
    end_line: usize,
    content: String,
    function_name: Option<String>,
    language: &str,
    chunk_type: ChunkType,
) -> RawChunk {
    let id = match &function_name {
        Some(name) if !name.is_empty() => {
            format!("{file}::{}::{name}::{start_line}", chunk_type.as_str())
        }
        _ => format!("{file}:{start_line}:{end_line}"),
    };
    RawChunk {
        id,
        file: file.to_string(),
        start_line,
        end_line,
        content,
        function_name,
        language: Some(language.to_string()),
        chunk_type,
        calls: Vec::new(),
        inherits_from: Vec::new(),
        chunk_depth: 0,
        parent_chunk_id: None,
        child_chunk_ids: Vec::new(),
        nlp_keywords: Vec::new(),
        nlp_code_refs: Vec::new(),
        virtual_terms: Vec::new(),
    }
}

/// Markdown: split on `^#+ ` headings. Each heading + its body becomes one
/// chunk. Content before the first heading becomes a leading chunk.
fn chunk_markdown(file: &str, content: &str) -> Vec<RawChunk> {
    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        return Vec::new();
    }
    let mut out: Vec<RawChunk> = Vec::new();
    let mut section_start = 0usize;
    let mut section_heading: Option<String> = None;
    let mut in_code_fence = false;

    let flush = |out: &mut Vec<RawChunk>,
                 start: usize,
                 end: usize,
                 heading: &Option<String>,
                 lines: &[&str]| {
        if start >= end {
            return;
        }
        let text = lines[start..end].join("\n");
        if text.trim().is_empty() {
            return;
        }
        out.push(document_chunk(
            file,
            start + 1,
            end,
            text,
            heading.clone(),
            "markdown",
            ChunkType::Docstring,
        ));
    };

    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim_start();
        // Track fenced code blocks so we don't treat `#` inside code as headings.
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            in_code_fence = !in_code_fence;
            continue;
        }
        if in_code_fence {
            continue;
        }
        if trimmed.starts_with('#') {
            // Heading line — flush previous section.
            flush(&mut out, section_start, i, &section_heading, &lines);
            // Extract heading text (strip leading #'s and whitespace).
            let heading = trimmed.trim_start_matches('#').trim().to_string();
            section_heading = if heading.is_empty() {
                None
            } else {
                Some(heading)
            };
            section_start = i;
        }
    }
    // Final section.
    flush(
        &mut out,
        section_start,
        lines.len(),
        &section_heading,
        &lines,
    );

    if out.is_empty() {
        // No content matched: fall back to a single whole-file chunk.
        out.push(document_chunk(
            file,
            1,
            lines.len(),
            content.to_string(),
            None,
            "markdown",
            ChunkType::Docstring,
        ));
    }
    out
}

/// YAML: split on top-level keys (lines starting at column 0 with `key:`).
/// Comments and blank lines are bundled with the following key's section.
fn chunk_yaml(file: &str, content: &str) -> Vec<RawChunk> {
    chunk_by_top_level_key(file, content, "yaml", |line| {
        let trimmed = line.trim_end();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            return None;
        }
        // Top-level YAML key: starts at col 0, not indented, contains ':'.
        if !line.starts_with(|c: char| c.is_whitespace() || c == '-') {
            if let Some(idx) = trimmed.find(':') {
                let key = trimmed[..idx].trim();
                if !key.is_empty() && !key.contains(' ') {
                    return Some(key.to_string());
                }
            }
        }
        None
    })
}

/// TOML: split on `[section]` and `[[array.section]]` headers at column 0.
fn chunk_toml(file: &str, content: &str) -> Vec<RawChunk> {
    chunk_by_top_level_key(file, content, "toml", |line| {
        let trimmed = line.trim_end();
        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            let inner = trimmed
                .trim_start_matches('[')
                .trim_end_matches(']')
                .trim_start_matches('[')
                .trim_end_matches(']')
                .trim()
                .to_string();
            if !inner.is_empty() {
                return Some(inner);
            }
        }
        None
    })
}

/// Generic top-level-key chunker. `header_of(line)` returns `Some(name)` when
/// the line starts a new section. Content before the first header is emitted
/// as a leading "preamble" chunk.
fn chunk_by_top_level_key(
    file: &str,
    content: &str,
    language: &str,
    header_of: impl Fn(&str) -> Option<String>,
) -> Vec<RawChunk> {
    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        return Vec::new();
    }
    let mut out: Vec<RawChunk> = Vec::new();
    let mut section_start = 0usize;
    let mut section_name: Option<String> = None;

    let flush = |out: &mut Vec<RawChunk>,
                 start: usize,
                 end: usize,
                 name: &Option<String>,
                 lines: &[&str]| {
        if start >= end {
            return;
        }
        let text = lines[start..end].join("\n");
        if text.trim().is_empty() {
            return;
        }
        out.push(document_chunk(
            file,
            start + 1,
            end,
            text,
            name.clone(),
            language,
            ChunkType::Constant,
        ));
    };

    for (i, line) in lines.iter().enumerate() {
        if let Some(name) = header_of(line) {
            flush(&mut out, section_start, i, &section_name, &lines);
            section_name = Some(name);
            section_start = i;
        }
    }
    flush(&mut out, section_start, lines.len(), &section_name, &lines);

    if out.is_empty() {
        out.push(document_chunk(
            file,
            1,
            lines.len(),
            content.to_string(),
            None,
            language,
            ChunkType::Constant,
        ));
    }
    out
}

/// JSON: if the file has fewer than `JSON_MAX_LINES` lines, emit a single
/// whole-file chunk. Otherwise return `Some(empty)` to signal "skip indexing".
fn chunk_json(file: &str, content: &str) -> Option<Vec<RawChunk>> {
    let line_count = content.lines().count();
    if line_count == 0 {
        return Some(Vec::new());
    }
    if line_count >= JSON_MAX_LINES {
        // Skip large JSON: it's effectively un-chunkable and dominates BM25
        // with structural punctuation noise.
        return Some(Vec::new());
    }
    Some(vec![document_chunk(
        file,
        1,
        line_count,
        content.to_string(),
        None,
        "json",
        ChunkType::Constant,
    )])
}

/// Plaintext / logs: split on blank-line paragraphs, cap at
/// `PLAINTEXT_MAX_LINES` per chunk. Paragraphs longer than the cap are split
/// into successive fixed-size sub-chunks.
fn chunk_plaintext(file: &str, content: &str) -> Vec<RawChunk> {
    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        return Vec::new();
    }
    let lang = match std::path::Path::new(file)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase()
        .as_str()
    {
        "log" => "log",
        _ => "text",
    };
    let mut out: Vec<RawChunk> = Vec::new();
    let mut buf_start: Option<usize> = None;

    let push_buf =
        |out: &mut Vec<RawChunk>, start: usize, end: usize, lines: &[&str], lang: &str| {
            // Split into fixed PLAINTEXT_MAX_LINES windows (no overlap).
            let mut s = start;
            while s < end {
                let e = (s + PLAINTEXT_MAX_LINES).min(end);
                let text = lines[s..e].join("\n");
                if !text.trim().is_empty() {
                    out.push(document_chunk(
                        file,
                        s + 1,
                        e,
                        text,
                        None,
                        lang,
                        ChunkType::Code,
                    ));
                }
                s = e;
            }
        };

    for (i, line) in lines.iter().enumerate() {
        if line.trim().is_empty() {
            if let Some(start) = buf_start.take() {
                push_buf(&mut out, start, i, &lines, lang);
            }
        } else if buf_start.is_none() {
            buf_start = Some(i);
        }
    }
    if let Some(start) = buf_start {
        push_buf(&mut out, start, lines.len(), &lines, lang);
    }

    if out.is_empty() {
        out.push(document_chunk(
            file,
            1,
            lines.len(),
            content.to_string(),
            None,
            lang,
            ChunkType::Code,
        ));
    }
    out
}

/// XML: split on top-level child elements via a minimal depth-tracking parser.
/// Each direct child of the root becomes one chunk; the XML prolog and root
/// open/close tags are emitted as separate trivial chunks if present.
fn chunk_xml(file: &str, content: &str) -> Vec<RawChunk> {
    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        return Vec::new();
    }

    // Walk lines tracking element-open depth (excluding self-closing and
    // closing tags). Depth==1 is "inside root, at top of children".
    let mut out: Vec<RawChunk> = Vec::new();
    let mut depth: i32 = 0;
    let mut child_start: Option<usize> = None;
    let mut child_name: Option<String> = None;

    for (i, line) in lines.iter().enumerate() {
        let opens = count_xml_opens(line);
        let closes = count_xml_closes(line);

        // If we're at depth 1 with no active child and this line opens a new
        // element, start tracking.
        if depth == 1 && child_start.is_none() && opens > closes {
            child_start = Some(i);
            child_name = first_xml_tag_name(line);
        }

        let prev_depth = depth;
        depth += opens as i32;
        depth -= closes as i32;

        // Closed a top-level child: emit chunk.
        if let Some(start) = child_start {
            if depth <= 1 && prev_depth >= 1 && i >= start {
                let text = lines[start..=i].join("\n");
                if !text.trim().is_empty() {
                    out.push(document_chunk(
                        file,
                        start + 1,
                        i + 1,
                        text,
                        child_name.clone(),
                        "xml",
                        ChunkType::Class,
                    ));
                }
                child_start = None;
                child_name = None;
            }
        }
    }

    if out.is_empty() {
        out.push(document_chunk(
            file,
            1,
            lines.len(),
            content.to_string(),
            None,
            "xml",
            ChunkType::Class,
        ));
    }
    out
}

/// Count element-opening tags on a line, excluding self-closing (`<foo/>`),
/// closing tags (`</foo>`), prolog (`<?xml ... ?>`), comments, and DOCTYPE.
fn count_xml_opens(line: &str) -> usize {
    let mut count = 0usize;
    let bytes = line.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'<' {
            // Skip prolog/comment/doctype/closing.
            let rest = &line[i..];
            if rest.starts_with("<?")
                || rest.starts_with("<!--")
                || rest.starts_with("<!")
                || rest.starts_with("</")
            {
                i += 1;
                continue;
            }
            // Find the matching `>` and check if it's self-closing.
            if let Some(close) = rest.find('>') {
                let tag = &rest[..=close];
                if !tag.ends_with("/>") {
                    count += 1;
                }
                i += close + 1;
                continue;
            }
        }
        i += 1;
    }
    count
}

/// Count element-closing tags (`</foo>`) on a line.
fn count_xml_closes(line: &str) -> usize {
    line.matches("</").count()
}

/// Extract the first opening tag name from a line, e.g. `<book id="1">` → `book`.
fn first_xml_tag_name(line: &str) -> Option<String> {
    let start = line.find('<')?;
    let rest = &line[start + 1..];
    if rest.starts_with('?') || rest.starts_with('!') || rest.starts_with('/') {
        return None;
    }
    let end = rest
        .find(|c: char| c.is_whitespace() || c == '>' || c == '/')
        .unwrap_or(rest.len());
    let name = rest[..end].trim();
    if name.is_empty() {
        None
    } else {
        Some(name.to_string())
    }
}

/// AST-aware entry point. Returns chunks and entities produced from a single
/// parse pass. For structured documents (md, yaml, toml, json, xml, txt, log)
/// dispatches to `chunk_document`. Falls back to `chunk_text` for unknown
/// extensions.
pub fn chunk_ast(file: &str, content: &str) -> (Vec<RawChunk>, Vec<RawEntity>) {
    let Some((lang, language)) = language_for(file) else {
        // Try structured-document chunkers (markdown, yaml, toml, json, xml,
        // plaintext, logs). These return None for unknown extensions and we
        // fall back to the sliding-window chunker.
        if let Some(chunks) = chunk_document(file, content) {
            return (chunks, Vec::new());
        }
        return (chunk_text(file, content, 150, 50), Vec::new());
    };

    let mut parser = Parser::new();
    if parser.set_language(&language).is_err() {
        tracing::warn!(
            "failed to set tree-sitter language for {file}; falling back to sliding-window"
        );
        return (chunk_text(file, content, 150, 50), Vec::new());
    }

    let src = content.as_bytes();
    let Some(tree) = parser.parse(src, None) else {
        return (chunk_text(file, content, 150, 50), Vec::new());
    };

    let line_offsets = build_line_offsets(src);
    let mut chunks: Vec<RawChunk> = Vec::new();
    walk_for_chunks(
        tree.root_node(),
        src,
        file,
        lang,
        &line_offsets,
        0,
        &mut chunks,
    );

    if chunks.is_empty() {
        // Source had no recognisable declarations: fall back to a single Code chunk.
        let total_lines = content.lines().count().max(1);
        chunks.push(RawChunk::generic(
            format!("{file}:1:{total_lines}"),
            file.to_string(),
            1,
            total_lines,
            content.to_string(),
        ));
        if let Some(c) = chunks.first_mut() {
            c.language = Some(lang.to_string());
        }
    }

    // Split oversized chunks; produces sub-chunks with `parent_chunk_id`.
    let split = split_oversized(chunks);

    // Entities (single pass over the same tree).
    let entities = extract_entities(&tree, src, file, lang);

    (split, entities)
}

fn walk_for_chunks(
    node: Node<'_>,
    src: &[u8],
    file: &str,
    lang: &str,
    line_offsets: &[usize],
    depth: usize,
    out: &mut Vec<RawChunk>,
) {
    // Try to classify this node.
    if let Some(chunk_type) = classify_node(lang, node) {
        let start_byte = node.start_byte();
        let end_byte = node.end_byte();
        let start_line = line_for_byte(line_offsets, start_byte);
        let end_line = line_for_byte(line_offsets, end_byte.saturating_sub(1));
        let content = std::str::from_utf8(&src[start_byte..end_byte])
            .unwrap_or("")
            .to_string();
        let mut name = name_of(node, src);

        // For Rust methods, qualify with the impl type: `Foo::bar`.
        if lang == "rust" && chunk_type == ChunkType::Method {
            if let Some(ty) = rust_impl_type_name(node, src) {
                if !name.is_empty() {
                    name = format!("{ty}::{name}");
                }
            }
        }

        // Phase 2 (issue #55): qualify Scala methods with their enclosing
        // class/object/trait, e.g. `Foo::bar`. Mirrors the Rust impl-type
        // qualification above so SymbolGraph caller edges resolve unambiguously
        // across classes that share method names.
        if lang == "scala" && chunk_type == ChunkType::Method {
            if let Some(ty) = scala_enclosing_class_name(node, src) {
                if !name.is_empty() && !ty.is_empty() {
                    name = format!("{ty}::{name}");
                }
            }
        }

        // Phase 2 (issue #49): qualify PHP methods with their enclosing
        // class/interface/trait, e.g. `Foo::doIt`. Same rationale as Scala.
        if lang == "php" && chunk_type == ChunkType::Method {
            if let Some(ty) = php_enclosing_class_name(node, src) {
                if !name.is_empty() && !ty.is_empty() {
                    name = format!("{ty}::{name}");
                }
            }
        }

        let calls = collect_calls(node, src, lang);
        let inherits_from = collect_inherits(node, src, lang);
        let doc = preceding_doc_comments(node, src);
        let (nlp_keywords, nlp_code_refs) = nlp_from_doc(&doc);

        let id = make_chunk_id(file, &chunk_type, &name, start_line, end_line);
        out.push(RawChunk {
            id,
            file: file.to_string(),
            start_line,
            end_line,
            content,
            function_name: if name.is_empty() { None } else { Some(name) },
            language: Some(lang.to_string()),
            chunk_type,
            calls,
            inherits_from,
            chunk_depth: depth,
            parent_chunk_id: None,
            child_chunk_ids: Vec::new(),
            nlp_keywords,
            nlp_code_refs,
            virtual_terms: Vec::new(),
        });

        // Descend into impl/class/module to capture their methods/inner items,
        // but don't recurse into function/method bodies (no inner-fn chunks).
        let recurse = !matches!(
            node.kind(),
            "function_item"
                | "function_declaration"
                | "function_definition"
                | "method_declaration"
                | "method_definition"
                | "constructor_declaration"
                | "secondary_constructor"
                | "init_declaration"
                | "protocol_function_declaration"
        );
        if recurse {
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                walk_for_chunks(child, src, file, lang, line_offsets, depth + 1, out);
            }
        }
        return;
    }

    // Not a chunk-producing node: continue walking.
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        walk_for_chunks(child, src, file, lang, line_offsets, depth, out);
    }
}

/// If a chunk exceeds `MAX_CHUNK_LINES`, replace it with sliding sub-chunks
/// that keep `parent_chunk_id` pointing back at the AST chunk.
fn split_oversized(chunks: Vec<RawChunk>) -> Vec<RawChunk> {
    let mut out: Vec<RawChunk> = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        let line_count = chunk.end_line.saturating_sub(chunk.start_line) + 1;
        if line_count <= MAX_CHUNK_LINES {
            out.push(chunk);
            continue;
        }

        let parent_id = chunk.id.clone();
        let mut child_ids: Vec<String> = Vec::new();

        let lines: Vec<&str> = chunk.content.lines().collect();
        let mut start = 0usize;
        let mut sub_idx = 0usize;
        while start < lines.len() {
            let end = (start + SUB_CHUNK_WINDOW).min(lines.len());
            let text = lines[start..end].join("\n");
            let sub_id = format!("{parent_id}::sub::{sub_idx}");
            child_ids.push(sub_id.clone());
            out.push(RawChunk {
                id: sub_id,
                file: chunk.file.clone(),
                start_line: chunk.start_line + start,
                end_line: chunk.start_line + end - 1,
                content: text,
                function_name: chunk.function_name.clone(),
                language: chunk.language.clone(),
                chunk_type: chunk.chunk_type.clone(),
                calls: Vec::new(),
                inherits_from: Vec::new(),
                chunk_depth: chunk.chunk_depth,
                parent_chunk_id: Some(parent_id.clone()),
                child_chunk_ids: Vec::new(),
                nlp_keywords: Vec::new(),
                nlp_code_refs: Vec::new(),
                virtual_terms: Vec::new(),
            });
            if end == lines.len() {
                break;
            }
            start += SUB_CHUNK_STRIDE;
            sub_idx += 1;
        }

        // Keep the umbrella parent chunk too, with its child IDs filled in.
        let mut parent = chunk;
        parent.child_chunk_ids = child_ids;
        out.push(parent);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_overlapping_chunks() {
        let content = (1..=200)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let chunks = chunk_text("test.txt", &content, 150, 50);
        assert!(chunks.len() >= 2);
        assert_eq!(chunks[0].start_line, 1);
        assert_eq!(chunks[1].start_line, 51);
    }

    #[test]
    fn test_chunk_id_format() {
        let chunks = chunk_text("src/main.txt", "line1\nline2\nline3", 150, 50);
        assert!(chunks[0].id.starts_with("src/main.txt:"));
    }

    #[test]
    fn test_rust_function_chunking() {
        let src = r#"
fn alpha() {}

fn beta() -> i32 { 1 }

fn gamma(x: i32) -> i32 { x + 1 }
"#;
        let (chunks, _ents) = chunk_ast("a.rs", src);
        let fns: Vec<&RawChunk> = chunks
            .iter()
            .filter(|c| c.chunk_type == ChunkType::Function)
            .collect();
        assert_eq!(fns.len(), 3, "expected 3 function chunks, got {fns:?}");
        let names: Vec<_> = fns
            .iter()
            .map(|c| c.function_name.clone().unwrap_or_default())
            .collect();
        assert!(names.contains(&"alpha".to_string()));
        assert!(names.contains(&"beta".to_string()));
        assert!(names.contains(&"gamma".to_string()));
    }

    #[test]
    fn test_rust_impl_method_qualified_name() {
        let src = r#"
struct Foo;
impl Foo {
    fn bar(&self) {}
}
"#;
        let (chunks, _) = chunk_ast("foo.rs", src);
        let method = chunks
            .iter()
            .find(|c| c.chunk_type == ChunkType::Method)
            .expect("expected at least one Method chunk");
        assert_eq!(method.function_name.as_deref(), Some("Foo::bar"));
    }

    #[test]
    fn test_rust_calls_extraction() {
        let src = r#"
fn main() {
    foo();
    bar(1, 2);
}
fn foo() {}
fn bar(_a: i32, _b: i32) {}
"#;
        let (chunks, _) = chunk_ast("m.rs", src);
        let main_chunk = chunks
            .iter()
            .find(|c| c.function_name.as_deref() == Some("main"))
            .expect("main chunk");
        assert!(
            main_chunk.calls.contains(&"foo".to_string()),
            "calls={:?}",
            main_chunk.calls
        );
        assert!(
            main_chunk.calls.contains(&"bar".to_string()),
            "calls={:?}",
            main_chunk.calls
        );
    }

    #[test]
    fn test_rust_entity_named_types() {
        let src = r#"
use std::sync::Arc;
fn f() {
    let _x: Arc<Vec<String>> = Arc::new(Vec::new());
}
"#;
        let (_chunks, entities) = chunk_ast("t.rs", src);
        let named: Vec<&str> = entities
            .iter()
            .filter(|e| e.entity_type == crate::core::entity::EntityType::NamedType)
            .map(|e| e.text.as_str())
            .collect();
        assert!(named.contains(&"Arc"), "named_types={named:?}");
        assert!(named.contains(&"Vec"), "named_types={named:?}");
        assert!(named.contains(&"String"), "named_types={named:?}");
    }

    #[test]
    fn test_large_function_splits() {
        // 250-line function body
        let mut body = String::new();
        for i in 0..250 {
            body.push_str(&format!("    let _v{i} = {i};\n"));
        }
        let src = format!("fn huge() {{\n{body}}}\n");
        let (chunks, _) = chunk_ast("h.rs", &src);
        let subs: Vec<&RawChunk> = chunks
            .iter()
            .filter(|c| c.parent_chunk_id.is_some())
            .collect();
        assert!(
            !subs.is_empty(),
            "expected sub-chunks for 250-line fn, got {chunks:#?}"
        );
        let parent_id = subs[0].parent_chunk_id.clone().unwrap();
        let parent = chunks
            .iter()
            .find(|c| c.id == parent_id)
            .expect("parent retained");
        assert!(!parent.child_chunk_ids.is_empty());
    }

    #[test]
    fn test_unknown_language_fallback() {
        // Use an unknown extension (no document chunker matches) to verify the
        // sliding-window fallback path.
        let content = "hello world\nfoo bar\nbaz";
        let (chunks, entities) = chunk_ast("notes.unknownext", content);
        assert!(entities.is_empty());
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].chunk_type, ChunkType::Code);
    }

    #[test]
    fn test_chunk_markdown_sections() {
        let content = "# Title\n\nintro\n\n## Section A\n\nbody a\n\n## Section B\n\nbody b\n";
        let chunks = chunk_markdown("doc.md", content);
        assert!(
            chunks.len() >= 2,
            "expected multiple sections, got {chunks:#?}"
        );
        let names: Vec<_> = chunks
            .iter()
            .filter_map(|c| c.function_name.clone())
            .collect();
        assert!(names.iter().any(|n| n == "Section A"), "names={names:?}");
        assert!(names.iter().any(|n| n == "Section B"), "names={names:?}");
        for c in &chunks {
            assert_eq!(c.language.as_deref(), Some("markdown"));
            assert_eq!(c.chunk_type, ChunkType::Docstring);
        }
    }

    #[test]
    fn test_chunk_markdown_ignores_hash_in_code_fence() {
        let content = "# Real Heading\n\nintro\n\n```\n## not a heading\n```\n\n## Next\n\nx\n";
        let chunks = chunk_markdown("doc.md", content);
        let names: Vec<_> = chunks
            .iter()
            .filter_map(|c| c.function_name.clone())
            .collect();
        assert!(names.iter().any(|n| n == "Real Heading"));
        assert!(names.iter().any(|n| n == "Next"));
        assert!(
            !names.iter().any(|n| n == "not a heading"),
            "should not split on # inside fenced code block: {names:?}"
        );
    }

    #[test]
    fn test_chunk_yaml_top_level_keys() {
        let content = "name: foo\nversion: 1.0\n\ndeps:\n  - a\n  - b\n\nscripts:\n  build: x\n";
        let chunks = chunk_yaml("conf.yaml", content);
        let names: Vec<_> = chunks
            .iter()
            .filter_map(|c| c.function_name.clone())
            .collect();
        assert!(names.iter().any(|n| n == "name"), "names={names:?}");
        assert!(names.iter().any(|n| n == "deps"), "names={names:?}");
        assert!(names.iter().any(|n| n == "scripts"), "names={names:?}");
        for c in &chunks {
            assert_eq!(c.language.as_deref(), Some("yaml"));
        }
    }

    #[test]
    fn test_chunk_toml_sections() {
        let content = "[package]\nname = \"foo\"\nversion = \"1.0\"\n\n[dependencies]\nserde = \"1\"\n\n[[bin]]\nname = \"x\"\n";
        let chunks = chunk_toml("Cargo.toml", content);
        let names: Vec<_> = chunks
            .iter()
            .filter_map(|c| c.function_name.clone())
            .collect();
        assert!(names.iter().any(|n| n == "package"), "names={names:?}");
        assert!(names.iter().any(|n| n == "dependencies"), "names={names:?}");
        assert!(names.iter().any(|n| n == "bin"), "names={names:?}");
    }

    #[test]
    fn test_chunk_json_small_file_single_chunk() {
        let content = "{\n  \"name\": \"foo\",\n  \"version\": \"1.0\"\n}\n";
        let chunks = chunk_json("a.json", content).expect("Some result");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].language.as_deref(), Some("json"));
    }

    #[test]
    fn test_chunk_json_large_file_skipped() {
        let big = (0..600)
            .map(|i| format!("  \"k{i}\": {i},"))
            .collect::<Vec<_>>()
            .join("\n");
        let content = format!("{{\n{big}\n}}\n");
        let chunks = chunk_json("big.json", &content).expect("Some result");
        assert!(chunks.is_empty(), "expected large JSON to be skipped");
    }

    #[test]
    fn test_chunk_plaintext_paragraphs() {
        let content = "First paragraph line 1.\nFirst paragraph line 2.\n\nSecond paragraph line 1.\nSecond paragraph line 2.\n\nThird paragraph.\n";
        let chunks = chunk_plaintext("note.txt", content);
        assert_eq!(
            chunks.len(),
            3,
            "expected one chunk per paragraph, got {chunks:#?}"
        );
        for c in &chunks {
            assert_eq!(c.language.as_deref(), Some("text"));
        }
    }

    #[test]
    fn test_chunk_plaintext_caps_at_50_lines() {
        let content = (1..=130)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let chunks = chunk_plaintext("big.log", &content);
        assert!(
            chunks.len() >= 3,
            "expected at least 3 chunks for 130-line paragraph, got {}",
            chunks.len()
        );
        for c in &chunks {
            let line_count = c.end_line.saturating_sub(c.start_line) + 1;
            assert!(line_count <= 50, "chunk too large: {line_count} lines");
            assert_eq!(c.language.as_deref(), Some("log"));
        }
    }

    #[test]
    fn test_chunk_xml_top_level_children() {
        let content = "<?xml version=\"1.0\"?>\n<library>\n  <book id=\"1\">\n    <title>A</title>\n  </book>\n  <book id=\"2\">\n    <title>B</title>\n  </book>\n  <magazine>\n    <title>C</title>\n  </magazine>\n</library>\n";
        let chunks = chunk_xml("data.xml", content);
        let names: Vec<_> = chunks
            .iter()
            .filter_map(|c| c.function_name.clone())
            .collect();
        assert!(
            names.iter().filter(|n| *n == "book").count() >= 2,
            "names={names:?}"
        );
        assert!(names.iter().any(|n| n == "magazine"), "names={names:?}");
        for c in &chunks {
            assert_eq!(c.language.as_deref(), Some("xml"));
        }
    }

    #[test]
    fn test_chunk_document_dispatch() {
        // Verify chunk_ast routes structured documents through chunk_document.
        let md_content = "# Hello\n\nworld\n";
        let (md_chunks, _) = chunk_ast("readme.md", md_content);
        assert!(md_chunks
            .iter()
            .any(|c| c.language.as_deref() == Some("markdown")));

        let yaml_content = "key: value\n";
        let (yaml_chunks, _) = chunk_ast("conf.yml", yaml_content);
        assert!(yaml_chunks
            .iter()
            .any(|c| c.language.as_deref() == Some("yaml")));

        let toml_content = "[section]\nx = 1\n";
        let (toml_chunks, _) = chunk_ast("a.toml", toml_content);
        assert!(toml_chunks
            .iter()
            .any(|c| c.language.as_deref() == Some("toml")));
    }

    #[test]
    fn test_nlp_code_refs() {
        let src = r#"
/// Wraps the `CodeIndexer` to expose hybrid search.
fn make() {}
"#;
        let (chunks, _) = chunk_ast("d.rs", src);
        let f = chunks
            .iter()
            .find(|c| c.function_name.as_deref() == Some("make"))
            .unwrap();
        assert!(
            f.nlp_code_refs.iter().any(|k| k == "CodeIndexer"),
            "code_refs={:?}",
            f.nlp_code_refs
        );
    }

    #[test]
    fn test_entity_external_crate() {
        let src = r#"
use usearch::Index;
fn f() {}
"#;
        let (_chunks, ents) = chunk_ast("u.rs", src);
        let exts: Vec<&str> = ents
            .iter()
            .filter(|e| e.entity_type == crate::core::entity::EntityType::ExternalCrate)
            .map(|e| e.text.as_str())
            .collect();
        assert!(exts.contains(&"usearch"), "external_crates={exts:?}");
    }

    #[test]
    fn test_entity_error_variant() {
        let src = r#"
fn f() -> Result<(), anyhow::Error> {
    anyhow::bail!("index not found");
}
"#;
        let (_chunks, ents) = chunk_ast("e.rs", src);
        let any_err = ents
            .iter()
            .any(|e| e.entity_type == crate::core::entity::EntityType::ErrorVariant);
        assert!(
            any_err,
            "expected at least one ErrorVariant entity, got {ents:#?}"
        );
    }

    #[test]
    fn test_csharp_chunking() {
        let src = r#"
namespace MyApp {
    class Foo {
        public void Bar() { Baz(); this.Qux(); }
        public Foo() {}
    }
    interface IThing { void Do(); }
}
"#;
        let (chunks, _) = chunk_ast("a.cs", src);
        // Expect: namespace (Module), class Foo (Class), Bar (Method),
        //   ctor (Method), IThing (Trait), Do (Method).
        let classes: Vec<&RawChunk> = chunks
            .iter()
            .filter(|c| c.chunk_type == ChunkType::Class)
            .collect();
        assert!(
            classes
                .iter()
                .any(|c| c.function_name.as_deref() == Some("Foo")),
            "expected class Foo, got {chunks:#?}"
        );
        let traits: Vec<&RawChunk> = chunks
            .iter()
            .filter(|c| c.chunk_type == ChunkType::Trait)
            .collect();
        assert!(
            traits
                .iter()
                .any(|c| c.function_name.as_deref() == Some("IThing")),
            "expected interface IThing as Trait"
        );
        let bar = chunks
            .iter()
            .find(|c| c.function_name.as_deref() == Some("Bar"))
            .expect("Bar method chunk");
        assert_eq!(bar.chunk_type, ChunkType::Method);
        assert!(
            bar.calls.contains(&"Baz".to_string()),
            "calls={:?}",
            bar.calls
        );
        assert!(
            bar.calls.contains(&"Qux".to_string()),
            "calls={:?}",
            bar.calls
        );
    }

    #[test]
    fn test_kotlin_chunking() {
        // Avoid the top-level `package` statement which the kotlin-ng grammar
        // parses oddly without a following file body terminator; the chunker
        // still walks into ERROR-recovered subtrees, but the cleaner case
        // exercises the happy path.
        let src = r#"
class Foo {
    fun bar() { baz(); this.qux() }
}
object Singleton {
    fun run() { other() }
}
"#;
        let (chunks, _) = chunk_ast("a.kt", src);
        assert!(
            chunks
                .iter()
                .any(|c| c.function_name.as_deref() == Some("Foo")
                    && c.chunk_type == ChunkType::Class),
            "expected class Foo, got {chunks:#?}"
        );
        let bar = chunks
            .iter()
            .find(|c| c.function_name.as_deref() == Some("bar"))
            .expect("bar method chunk");
        assert_eq!(bar.chunk_type, ChunkType::Method);
        assert!(
            bar.calls.contains(&"baz".to_string()),
            "calls={:?}",
            bar.calls
        );
        assert!(
            bar.calls.contains(&"qux".to_string()),
            "calls={:?}",
            bar.calls
        );
    }

    #[test]
    fn test_swift_chunking() {
        let src = r#"
class Foo {
    func bar() { baz(); self.qux() }
    init() {}
}
struct S {}
enum E { case a }
protocol P { func d() }
extension Foo { func ext() {} }
"#;
        let (chunks, _) = chunk_ast("a.swift", src);
        // class Foo
        assert!(
            chunks
                .iter()
                .any(|c| c.function_name.as_deref() == Some("Foo")
                    && c.chunk_type == ChunkType::Class),
            "expected class Foo, got {chunks:#?}"
        );
        // struct S
        assert!(
            chunks
                .iter()
                .any(|c| c.function_name.as_deref() == Some("S")
                    && c.chunk_type == ChunkType::Struct),
            "expected struct S"
        );
        // enum E
        assert!(
            chunks.iter().any(|c| c.function_name.as_deref() == Some("E")
                && c.chunk_type == ChunkType::Enum),
            "expected enum E"
        );
        // protocol P → Trait
        assert!(
            chunks.iter().any(
                |c| c.function_name.as_deref() == Some("P") && c.chunk_type == ChunkType::Trait
            ),
            "expected protocol P as Trait"
        );
        // extension Foo → Module
        assert!(
            chunks
                .iter()
                .any(|c| c.chunk_type == ChunkType::Module
                    && c.function_name.as_deref() == Some("Foo")),
            "expected extension Foo as Module"
        );
        // method calls
        let bar = chunks
            .iter()
            .find(|c| c.function_name.as_deref() == Some("bar"))
            .expect("bar method chunk");
        assert!(
            bar.calls.contains(&"baz".to_string()),
            "calls={:?}",
            bar.calls
        );
        assert!(
            bar.calls.contains(&"qux".to_string()),
            "calls={:?}",
            bar.calls
        );
    }

    #[test]
    fn test_nlp_keywords_from_doc_comments() {
        let src = r#"
/// Implements the RRF fusion algorithm.
fn fuse() {}
"#;
        let (chunks, _) = chunk_ast("d.rs", src);
        let f = chunks
            .iter()
            .find(|c| c.function_name.as_deref() == Some("fuse"))
            .unwrap();
        assert!(
            f.nlp_keywords.iter().any(|k| k == "RRF"),
            "keywords={:?}",
            f.nlp_keywords
        );
        assert!(
            f.nlp_keywords.iter().any(|k| k == "Implements"),
            "keywords={:?}",
            f.nlp_keywords
        );
    }

    // ----- Scala Phase 2 (issue #55) -----

    #[test]
    fn test_scala_method_qualified_name() {
        // Why: SymbolGraph caller edges need `ClassName::methodName` so that
        // two classes with a `run` method don't share a single graph node.
        // What: a class method is chunked as `Foo::bar`, a top-level def as `freefn`.
        // Test: assert both chunks emit the expected qualified / unqualified names.
        let src = r#"
class Foo extends Bar with Mixin {
  def bar(): Unit = baz()
}
object O {
  def run(): Unit = other()
}
def freefn(): Unit = ()
"#;
        let (chunks, _) = chunk_ast("a.scala", src);
        let bar = chunks
            .iter()
            .find(|c| c.function_name.as_deref() == Some("Foo::bar"))
            .expect("expected qualified method Foo::bar, got: {chunks:#?}");
        assert_eq!(bar.chunk_type, ChunkType::Method);
        let run = chunks
            .iter()
            .find(|c| c.function_name.as_deref() == Some("O::run"))
            .expect("expected qualified method O::run");
        assert_eq!(run.chunk_type, ChunkType::Method);
        // Top-level def remains unqualified.
        assert!(
            chunks
                .iter()
                .any(|c| c.function_name.as_deref() == Some("freefn")
                    && c.chunk_type == ChunkType::Function),
            "expected unqualified Function freefn, got {chunks:#?}"
        );
    }

    #[test]
    fn test_scala_caller_scoped_call_edges() {
        // Why: Phase 2 needs caller-scoped call edges so `who calls baz?`
        // returns `Foo::bar`, not the whole file.
        // What: `Foo::bar`'s `calls` field includes `baz`, and the call is
        // attached to the method chunk (not the class).
        // Test: assert `calls` membership on the method chunk.
        let src = r#"
class Foo {
  def bar(): Unit = {
    baz()
    this.qux()
  }
}
"#;
        let (chunks, _) = chunk_ast("a.scala", src);
        let bar = chunks
            .iter()
            .find(|c| c.function_name.as_deref() == Some("Foo::bar"))
            .expect("Foo::bar chunk");
        assert!(
            bar.calls.contains(&"baz".to_string()),
            "calls={:?}",
            bar.calls
        );
        assert!(
            bar.calls.contains(&"qux".to_string()),
            "calls={:?}",
            bar.calls
        );
    }

    #[test]
    fn test_scala_extends_and_with_emit_inherits() {
        // Why: `extends T1 with T2 with T3` describes a layered Scala class
        // mixin chain; Phase 2 turns each parent into an `Implements` edge so
        // intent-gated KG expansion can surface the parent.
        // What: `inherits_from` on the class chunk lists all three parents.
        // Test: assert membership.
        let src = r#"
class Foo extends Bar with Mixin with Other {
  def m(): Unit = ()
}
"#;
        let (chunks, _) = chunk_ast("a.scala", src);
        let foo = chunks
            .iter()
            .find(|c| c.function_name.as_deref() == Some("Foo") && c.chunk_type == ChunkType::Class)
            .expect("Foo class chunk");
        for parent in ["Bar", "Mixin", "Other"] {
            assert!(
                foo.inherits_from.iter().any(|p| p == parent),
                "expected parent {parent} in inherits_from={:?}",
                foo.inherits_from
            );
        }
    }

    #[test]
    fn test_scala_symbol_graph_resolves_caller() {
        // Why: end-to-end check that the chunker output, once fed to
        // SymbolGraph::build_from_chunks, yields a usable caller→callee edge.
        // What: build the graph from two scala chunks and assert
        // `callers_of("baz")` returns the qualified method.
        // Test: integrates chunker + symbol_graph for Phase 2.
        use crate::core::symbol_graph::SymbolGraph;
        let src = r#"
class Foo {
  def bar(): Unit = baz()
}
def baz(): Unit = ()
"#;
        let (chunks, _) = chunk_ast("s.scala", src);
        let tuples: Vec<_> = chunks
            .iter()
            .map(|c| {
                (
                    c.id.clone(),
                    c.file.clone(),
                    c.function_name.clone(),
                    c.calls.clone(),
                    c.inherits_from.clone(),
                    c.chunk_type.clone(),
                )
            })
            .collect();
        let g = SymbolGraph::build_from_chunks(&tuples);
        let callers = g.callers_of("baz", 1);
        assert!(
            callers.iter().any(|(s, _)| s == "Foo::bar"),
            "callers={callers:?}"
        );
    }

    // ----- PHP Phase 2 (issue #49) -----

    #[test]
    fn test_php_method_qualified_name() {
        // Why: same rationale as Scala — class-qualified method names avoid
        // symbol collisions in the call graph.
        // What: `Foo::doIt` is the chunk's function_name; a free function in
        // the same file remains unqualified.
        // Test: assert both forms.
        let src = r#"<?php
class Foo extends Bar implements I1, I2 {
    public function doIt(): void {
        $this->helper();
    }
}
function freefn(): void {}
"#;
        let (chunks, _) = chunk_ast("a.php", src);
        let doit = chunks
            .iter()
            .find(|c| c.function_name.as_deref() == Some("Foo::doIt"))
            .expect("expected qualified Foo::doIt, got: {chunks:#?}");
        assert_eq!(doit.chunk_type, ChunkType::Method);
        assert!(
            chunks
                .iter()
                .any(|c| c.function_name.as_deref() == Some("freefn")
                    && c.chunk_type == ChunkType::Function),
            "expected unqualified Function freefn"
        );
    }

    #[test]
    fn test_php_caller_scoped_call_edges() {
        // Why: caller-scoped edges must capture all three PHP call shapes
        // (`$this->m()`, `Class::m()`, `func()`).
        // What: assert each callee appears in the method's `calls` field.
        let src = r#"<?php
class Foo {
    public function doIt(): void {
        $this->helper();
        Foo::staticCall();
        regularFunc();
    }
}
"#;
        let (chunks, _) = chunk_ast("a.php", src);
        let doit = chunks
            .iter()
            .find(|c| c.function_name.as_deref() == Some("Foo::doIt"))
            .expect("Foo::doIt chunk");
        for callee in ["helper", "staticCall", "regularFunc"] {
            assert!(
                doit.calls.iter().any(|c| c == callee),
                "expected callee {callee} in calls={:?}",
                doit.calls
            );
        }
    }

    #[test]
    fn test_php_implements_and_extends_emit_inherits() {
        // Why: PHP's `class Foo extends Bar implements I1, I2` carries one
        // parent class plus N interfaces; Phase 2 emits one `Implements` edge
        // for each.
        // What: assert all three names appear in `inherits_from`.
        let src = r#"<?php
class Foo extends Bar implements I1, I2 {}
"#;
        let (chunks, _) = chunk_ast("a.php", src);
        let foo = chunks
            .iter()
            .find(|c| c.function_name.as_deref() == Some("Foo") && c.chunk_type == ChunkType::Class)
            .expect("Foo class chunk");
        for parent in ["Bar", "I1", "I2"] {
            assert!(
                foo.inherits_from.iter().any(|p| p == parent),
                "expected parent {parent} in inherits_from={:?}",
                foo.inherits_from
            );
        }
    }

    #[test]
    fn test_php_interface_extends_emits_inherits() {
        // Why: PHP interfaces can extend multiple interfaces; the grammar
        // packages those parents in a `base_clause` (same shape as a class's
        // extends clause).
        // What: `interface Child extends P1, P2` → inherits_from = [P1, P2].
        let src = r#"<?php
interface Child extends P1, P2 {}
"#;
        let (chunks, _) = chunk_ast("a.php", src);
        let child = chunks
            .iter()
            .find(|c| {
                c.function_name.as_deref() == Some("Child") && c.chunk_type == ChunkType::Trait
            })
            .expect("Child interface (chunked as Trait)");
        for parent in ["P1", "P2"] {
            assert!(
                child.inherits_from.iter().any(|p| p == parent),
                "expected parent {parent} in inherits_from={:?}",
                child.inherits_from
            );
        }
    }

    #[test]
    fn test_php_symbol_graph_resolves_caller() {
        // Why: end-to-end Phase 2 integration: chunker → symbol_graph yields
        // a usable PHP caller→callee edge for KG expansion.
        // What: assert `callers_of("helper")` returns `Foo::doIt`.
        use crate::core::symbol_graph::SymbolGraph;
        let src = r#"<?php
class Foo {
    public function doIt(): void {
        $this->helper();
    }
    public function helper(): void {}
}
"#;
        let (chunks, _) = chunk_ast("p.php", src);
        let tuples: Vec<_> = chunks
            .iter()
            .map(|c| {
                (
                    c.id.clone(),
                    c.file.clone(),
                    c.function_name.clone(),
                    c.calls.clone(),
                    c.inherits_from.clone(),
                    c.chunk_type.clone(),
                )
            })
            .collect();
        let g = SymbolGraph::build_from_chunks(&tuples);
        // `helper` resolves to `Foo::helper` via the suffix lookup.
        let callers = g.callers_of("Foo::helper", 1);
        assert!(
            callers.iter().any(|(s, _)| s == "Foo::doIt"),
            "callers={callers:?}"
        );
    }

    /// Issue #90: end-to-end check that a small Rust snippet — parsed by
    /// `chunk_ast`, fed into `SymbolGraph::build_from_chunks` — produces
    /// non-zero symbols and a usable caller→callee edge. This is the
    /// regression test for the silent KG-skip bug where reindexes that
    /// breached the memory limit during embedding left the graph at 0/0.
    ///
    /// Why: prevents future skips/bugs along the chunker→graph integration
    /// from re-introducing the same failure mode.
    /// What: two free functions where `alpha` calls `beta`; the resulting
    /// graph must contain both symbols and `callers_of("beta")` must include
    /// `alpha`.
    /// Test: this is the test.
    #[test]
    fn test_rust_symbol_graph_resolves_caller() {
        use crate::core::symbol_graph::SymbolGraph;
        let src = "fn alpha() { beta(); }\nfn beta() {}\n";
        let (chunks, _) = chunk_ast("a.rs", src);
        let tuples: Vec<_> = chunks
            .iter()
            .map(|c| {
                (
                    c.id.clone(),
                    c.file.clone(),
                    c.function_name.clone(),
                    c.calls.clone(),
                    c.inherits_from.clone(),
                    c.chunk_type.clone(),
                )
            })
            .collect();
        let g = SymbolGraph::build_from_chunks(&tuples);
        assert!(
            g.node_count() >= 2,
            "expected >= 2 symbol nodes for alpha+beta, got {} (chunks={:#?})",
            g.node_count(),
            chunks
                .iter()
                .map(|c| (c.function_name.clone(), c.calls.clone()))
                .collect::<Vec<_>>(),
        );
        let callers = g.callers_of("beta", 1);
        assert!(
            callers.iter().any(|(s, _)| s == "alpha"),
            "expected alpha among callers of beta, got {callers:?}"
        );
    }
}
