//! Per-grammar inheritance extraction for the AST chunker.
//!
//! Why: `collect_inherits` previously dispatched on `(lang, node.kind())` for
//! Rust, Python, Scala, and PHP inside a single 80-line function. Splitting
//! the per-language extractors apart keeps each grammar's quirks isolated and
//! makes adding new grammars (Kotlin, Swift, …) a matter of writing one more
//! `collect_<lang>_inherits` helper.
//! What: `collect_inherits` is the public entry point; it dispatches on
//! `(lang, node.kind())` to the per-grammar helpers below.
//! Test: covered by `test_scala_extends_and_with_emit_inherits`,
//! `test_php_implements_and_extends_emit_inherits`,
//! `test_php_interface_extends_emits_inherits`, and the Python/Rust paths via
//! `test_rust_impl_method_qualified_name`.

use tree_sitter::Node;

/// Collect inherited names: trait list for `impl` blocks, base types for `struct_item`,
/// `extends`/`with` for Scala, `extends`/`implements` for PHP, and `class(Parent)` for
/// Python.
pub(super) fn collect_inherits(node: Node<'_>, src: &[u8], lang: &str) -> Vec<String> {
    let mut out = Vec::new();
    match (lang, node.kind()) {
        ("rust", "impl_item") => collect_rust_impl_inherits(node, src, &mut out),
        ("python", "class_definition") => collect_python_class_inherits(node, src, &mut out),
        ("scala", "class_definition" | "object_definition" | "trait_definition") => {
            collect_scala_template_inherits(node, src, &mut out);
        }
        ("php", "class_declaration" | "interface_declaration" | "trait_declaration") => {
            collect_php_class_inherits(node, src, &mut out);
        }
        _ => {}
    }
    out.retain(|s| !s.is_empty());
    out
}

/// Rust `impl Trait for Type` — the `trait` field is the trait name (if present).
fn collect_rust_impl_inherits(node: Node<'_>, src: &[u8], out: &mut Vec<String>) {
    if let Some(t) = node.child_by_field_name("trait") {
        out.push(
            std::str::from_utf8(&src[t.start_byte()..t.end_byte()])
                .unwrap_or("")
                .to_string(),
        );
    }
}

/// Python `class Foo(A, B):` — pull each name from the `superclasses` field.
fn collect_python_class_inherits(node: Node<'_>, src: &[u8], out: &mut Vec<String>) {
    let Some(s) = node.child_by_field_name("superclasses") else {
        return;
    };
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

/// Scala `extends T1 with T2 with T3` — encoded as an `extends_clause`
/// containing one or more `type_identifier` children. Each becomes an
/// `Implements` edge in the SymbolGraph.
///
/// Phase 2 (issue #55).
fn collect_scala_template_inherits(node: Node<'_>, src: &[u8], out: &mut Vec<String>) {
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
}

/// PHP `class Foo extends Bar implements I1, I2` — exposed via two siblings
/// of `class_declaration`:
///   - `base_clause`  → `extends Parent` (single parent class)
///   - `class_interface_clause` → `implements I1, I2` (interfaces)
///
/// For `interface_declaration`, `base_clause` carries `extends I1, I2`
/// (interfaces can extend multiple). We walk both kinds and extract every
/// `name`-kind child as a parent symbol.
///
/// Phase 2 (issue #49).
fn collect_php_class_inherits(node: Node<'_>, src: &[u8], out: &mut Vec<String>) {
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
