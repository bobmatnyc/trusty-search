//! Shared "what index are we operating on?" helpers used by the project-scoped
//! CLI subcommands (`search`, `watch`, `add`, `remove`, `reindex`).
//!
//! Why: every project subcommand needs the same precedence rules: an explicit
//! `--index` flag wins, otherwise auto-detect from CWD; and the same user-facing
//! warning when we fall back to the bare directory name. Centralising removes
//! duplication and shrinks `main.rs`.
//! What: two thin helpers wrapping `crate::detect`.
//! Test: covered indirectly by `cargo run -- search foo` (explicit index path)
//! and `cargo run -- search foo` from inside this repo (auto-detect path).

use crate::detect::{detect_project, DetectionMethod};
use colored::Colorize;

/// Resolve the effective index ID: explicit `--index` flag wins, otherwise
/// auto-detect from CWD via `detect_project`.
///
/// Why: every project-scoped command needs the same precedence rules.
/// What: returns `(index_id, warned)` where `warned` is true when we fell back
/// to the CWD basename and should print a warning.
/// Test: With explicit Some("foo") → returns ("foo", false). With None inside
/// this repo → returns ("trusty-search", false) (detected via .git).
pub fn resolve_index(explicit: &Option<String>) -> (String, bool) {
    if let Some(id) = explicit {
        return (id.clone(), false);
    }
    let cwd = std::env::current_dir().unwrap_or_default();
    let ctx = detect_project(&cwd);
    let warned = matches!(ctx.detection_method, DetectionMethod::Fallback);
    (ctx.index_id, warned)
}

/// Why: Make fallback detection visible so users know to run `init`.
/// What: Prints a one-line yellow warning to stderr if `warned` is true.
/// Test: Call with warned=true and capture stderr → contains "⚠".
pub fn print_index_header(index_id: &str, warned: bool) {
    if warned {
        eprintln!(
            "{} No .git or .trusty-search found — using directory name '{}'. \
             Run `trusty-search init` to register this project.",
            "⚠".yellow(),
            index_id
        );
    }
}
