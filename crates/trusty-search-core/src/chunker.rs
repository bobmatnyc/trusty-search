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

use crate::entity::{extract_entities, RawEntity};

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

        ("scala", "function_definition") => ChunkType::Function,
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

/// AST-aware entry point. Returns chunks and entities produced from a single
/// parse pass. Falls back to `chunk_text` for unknown extensions.
pub fn chunk_ast(file: &str, content: &str) -> (Vec<RawChunk>, Vec<RawEntity>) {
    let Some((lang, language)) = language_for(file) else {
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
            .filter(|e| e.entity_type == crate::entity::EntityType::NamedType)
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
        let content = "hello world\nfoo bar\nbaz";
        let (chunks, entities) = chunk_ast("notes.txt", content);
        assert!(entities.is_empty());
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].chunk_type, ChunkType::Code);
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
            .filter(|e| e.entity_type == crate::entity::EntityType::ExternalCrate)
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
            .any(|e| e.entity_type == crate::entity::EntityType::ErrorVariant);
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
            classes.iter().any(|c| c.function_name.as_deref() == Some("Foo")),
            "expected class Foo, got {chunks:#?}"
        );
        let traits: Vec<&RawChunk> = chunks
            .iter()
            .filter(|c| c.chunk_type == ChunkType::Trait)
            .collect();
        assert!(
            traits.iter().any(|c| c.function_name.as_deref() == Some("IThing")),
            "expected interface IThing as Trait"
        );
        let bar = chunks
            .iter()
            .find(|c| c.function_name.as_deref() == Some("Bar"))
            .expect("Bar method chunk");
        assert_eq!(bar.chunk_type, ChunkType::Method);
        assert!(bar.calls.contains(&"Baz".to_string()), "calls={:?}", bar.calls);
        assert!(bar.calls.contains(&"Qux".to_string()), "calls={:?}", bar.calls);
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
            chunks.iter().any(|c| c.function_name.as_deref() == Some("Foo")
                && c.chunk_type == ChunkType::Class),
            "expected class Foo, got {chunks:#?}"
        );
        let bar = chunks
            .iter()
            .find(|c| c.function_name.as_deref() == Some("bar"))
            .expect("bar method chunk");
        assert_eq!(bar.chunk_type, ChunkType::Method);
        assert!(bar.calls.contains(&"baz".to_string()), "calls={:?}", bar.calls);
        assert!(bar.calls.contains(&"qux".to_string()), "calls={:?}", bar.calls);
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
            chunks.iter().any(|c| c.function_name.as_deref() == Some("Foo")
                && c.chunk_type == ChunkType::Class),
            "expected class Foo, got {chunks:#?}"
        );
        // struct S
        assert!(
            chunks.iter().any(|c| c.function_name.as_deref() == Some("S")
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
            chunks.iter().any(|c| c.function_name.as_deref() == Some("P")
                && c.chunk_type == ChunkType::Trait),
            "expected protocol P as Trait"
        );
        // extension Foo → Module
        assert!(
            chunks
                .iter()
                .any(|c| c.chunk_type == ChunkType::Module && c.function_name.as_deref() == Some("Foo")),
            "expected extension Foo as Module"
        );
        // method calls
        let bar = chunks
            .iter()
            .find(|c| c.function_name.as_deref() == Some("bar"))
            .expect("bar method chunk");
        assert!(bar.calls.contains(&"baz".to_string()), "calls={:?}", bar.calls);
        assert!(bar.calls.contains(&"qux".to_string()), "calls={:?}", bar.calls);
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
}
