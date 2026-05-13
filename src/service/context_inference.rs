//! Index-context inference from root-level metadata files (issue #112).
//!
//! Why: Cross-index fan-out (`POST /search`) wants to weight or skip indexes
//! based on how relevant the query is to each project. Re-embedding every
//! query into a "project description" semantic space lets the daemon route
//! intelligently before paying the per-index search cost. The cheapest
//! source of such a description is the metadata files developers already
//! maintain (`README.md`, `CLAUDE.md`, `Cargo.toml`, `package.json`,
//! `pyproject.toml`, `go.mod`, `.trusty-search.yaml`).
//!
//! What: a single entry-point [`scrape_metadata_summary`] that reads any of
//! the supported files under `root_path`, extracts the high-signal fields
//! (package name, description, keywords, the top of `README.md`, full
//! `CLAUDE.md` / `AGENTS.md`), and concatenates them into a single string
//! capped at [`MAX_SUMMARY_CHARS`]. Returns `None` when no metadata was
//! found.
//!
//! Test: `scrape_*` unit tests in this module cover each file type, the
//! truncation behaviour, the precedence ordering, and the no-metadata
//! fallback.

use std::path::Path;

/// Maximum characters in the concatenated summary returned by
/// [`scrape_metadata_summary`]. The downstream embedder is fine with longer
/// inputs (it truncates internally), but bounding here keeps RAM and embed
/// latency predictable.
pub const MAX_SUMMARY_CHARS: usize = 3000;

/// Cap for the short summary stored on `IndexHandle.context_summary` and
/// surfaced via `GET /indexes/:id/status`. 500 chars is enough to identify
/// a project but small enough to embed in a status JSON without bloat.
pub const MAX_DISPLAY_SUMMARY_CHARS: usize = 500;

/// Cap on the README.md slice we feed into the summary. The README header
/// is usually the project's elevator pitch, so 2000 chars covers the
/// abstract + a couple of feature bullets without dragging in API tables.
const README_SLICE_CHARS: usize = 2000;

/// Walk `root_path` for known metadata files and return a concatenated
/// summary string suitable for embedding. Returns `None` when none of the
/// recognised files exist (the caller treats this as "no context
/// embedding" → cosine weight defaults to 1.0).
///
/// Why: see module doc. The precedence here is by signal strength: the
/// README header gives the human-language pitch first, then explicit AI
/// briefs (`CLAUDE.md` / `AGENTS.md`), then structured manifests
/// (Cargo/npm/pyproject/go.mod), and finally the project-local
/// `.trusty-search.yaml` description (lowest priority because it's usually
/// the noisiest / least curated).
/// What: returns `Some(summary)` capped at [`MAX_SUMMARY_CHARS`] or `None`
/// when no recognised metadata file was found.
/// Test: `scrape_combines_multiple_sources_in_priority_order` covers the
/// ordering; `scrape_returns_none_when_no_metadata` covers the fallback.
pub fn scrape_metadata_summary(root_path: &Path) -> Option<String> {
    let mut parts: Vec<String> = Vec::new();

    if let Some(s) = read_readme_head(root_path) {
        parts.push(format!("README: {s}"));
    }
    if let Some(s) = read_full_text(root_path, "CLAUDE.md") {
        parts.push(format!("CLAUDE.md: {s}"));
    }
    if let Some(s) = read_full_text(root_path, "AGENTS.md") {
        parts.push(format!("AGENTS.md: {s}"));
    }
    if let Some(s) = read_cargo_toml(root_path) {
        parts.push(format!("Cargo: {s}"));
    }
    if let Some(s) = read_package_json(root_path) {
        parts.push(format!("package.json: {s}"));
    }
    if let Some(s) = read_pyproject_toml(root_path) {
        parts.push(format!("pyproject: {s}"));
    }
    if let Some(s) = read_go_mod(root_path) {
        parts.push(format!("go.mod: {s}"));
    }
    if let Some(s) = read_trusty_yaml_description(root_path) {
        parts.push(format!("trusty-search.yaml: {s}"));
    }

    if parts.is_empty() {
        return None;
    }

    let mut joined = parts.join("\n\n");
    if joined.chars().count() > MAX_SUMMARY_CHARS {
        joined = truncate_chars(&joined, MAX_SUMMARY_CHARS);
    }
    Some(joined)
}

/// Truncate a short version of `summary` suitable for status JSON display.
///
/// Why: callers want a human-readable preview without the full 3000-char
/// embed input.
/// What: returns at most [`MAX_DISPLAY_SUMMARY_CHARS`] characters.
/// Test: `display_summary_truncates`.
pub fn make_display_summary(summary: &str) -> String {
    truncate_chars(summary, MAX_DISPLAY_SUMMARY_CHARS)
}

fn truncate_chars(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    s.chars().take(max).collect()
}

fn read_readme_head(root: &Path) -> Option<String> {
    let p = root.join("README.md");
    let text = std::fs::read_to_string(&p).ok()?;
    let head = truncate_chars(text.trim(), README_SLICE_CHARS);
    if head.is_empty() {
        None
    } else {
        Some(head)
    }
}

fn read_full_text(root: &Path, file_name: &str) -> Option<String> {
    let p = root.join(file_name);
    let text = std::fs::read_to_string(&p).ok()?;
    let trimmed = text.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn read_cargo_toml(root: &Path) -> Option<String> {
    let text = std::fs::read_to_string(root.join("Cargo.toml")).ok()?;
    let value: toml::Value = toml::from_str(&text).ok()?;
    let pkg = value.get("package")?.as_table()?;
    let name = pkg.get("name").and_then(|v| v.as_str());
    let description = pkg.get("description").and_then(|v| v.as_str());
    join_optional(&[name, description])
}

fn read_package_json(root: &Path) -> Option<String> {
    let text = std::fs::read_to_string(root.join("package.json")).ok()?;
    let value: serde_json::Value = serde_json::from_str(&text).ok()?;
    let name = value.get("name").and_then(|v| v.as_str());
    let description = value.get("description").and_then(|v| v.as_str());
    let keywords = value.get("keywords").and_then(|v| v.as_array()).map(|arr| {
        arr.iter()
            .filter_map(|v| v.as_str())
            .collect::<Vec<&str>>()
            .join(", ")
    });
    let keywords_str = keywords.as_deref();
    join_optional(&[name, description, keywords_str])
}

fn read_pyproject_toml(root: &Path) -> Option<String> {
    let text = std::fs::read_to_string(root.join("pyproject.toml")).ok()?;
    let value: toml::Value = toml::from_str(&text).ok()?;
    // Prefer PEP 621 `[project]` over `[tool.poetry]`.
    let (name, description) = if let Some(project) = value.get("project").and_then(|v| v.as_table())
    {
        (
            project.get("name").and_then(|v| v.as_str()),
            project.get("description").and_then(|v| v.as_str()),
        )
    } else if let Some(poetry) = value
        .get("tool")
        .and_then(|v| v.as_table())
        .and_then(|t| t.get("poetry"))
        .and_then(|v| v.as_table())
    {
        (
            poetry.get("name").and_then(|v| v.as_str()),
            poetry.get("description").and_then(|v| v.as_str()),
        )
    } else {
        (None, None)
    };
    join_optional(&[name, description])
}

fn read_go_mod(root: &Path) -> Option<String> {
    let text = std::fs::read_to_string(root.join("go.mod")).ok()?;
    for line in text.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("module ") {
            let module = rest.trim();
            if !module.is_empty() {
                return Some(module.to_string());
            }
        }
    }
    None
}

fn read_trusty_yaml_description(root: &Path) -> Option<String> {
    // Both legacy and current names; either may carry a `description:` field.
    for name in &[".trusty-search.yaml", "trusty-search.yaml"] {
        let p = root.join(name);
        let Ok(text) = std::fs::read_to_string(&p) else {
            continue;
        };
        for line in text.lines() {
            let trimmed = line.trim();
            if let Some(rest) = trimmed.strip_prefix("description:") {
                let val = rest.trim().trim_matches('"').trim_matches('\'').trim();
                if !val.is_empty() {
                    return Some(val.to_string());
                }
            }
        }
    }
    None
}

fn join_optional(parts: &[Option<&str>]) -> Option<String> {
    let collected: Vec<&str> = parts
        .iter()
        .filter_map(|p| p.map(|s| s.trim()).filter(|s| !s.is_empty()))
        .collect();
    if collected.is_empty() {
        None
    } else {
        Some(collected.join(" — "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn scrape_returns_none_when_no_metadata() {
        let tmp = tempdir().expect("tempdir");
        assert!(scrape_metadata_summary(tmp.path()).is_none());
    }

    #[test]
    fn scrape_reads_readme_head() {
        let tmp = tempdir().expect("tempdir");
        fs::write(
            tmp.path().join("README.md"),
            "# trusty-search\n\nBlazingly fast hybrid code search service.\n",
        )
        .unwrap();
        let summary = scrape_metadata_summary(tmp.path()).expect("summary");
        assert!(summary.contains("README:"));
        assert!(summary.contains("trusty-search"));
        assert!(summary.contains("hybrid code search"));
    }

    #[test]
    fn scrape_reads_cargo_toml_package() {
        let tmp = tempdir().expect("tempdir");
        fs::write(
            tmp.path().join("Cargo.toml"),
            r#"[package]
name = "trusty-search"
description = "Hybrid code search"
version = "0.1.0"
"#,
        )
        .unwrap();
        let summary = scrape_metadata_summary(tmp.path()).expect("summary");
        assert!(summary.contains("Cargo:"));
        assert!(summary.contains("trusty-search"));
        assert!(summary.contains("Hybrid code search"));
        // Version should not leak.
        assert!(!summary.contains("0.1.0"));
    }

    #[test]
    fn scrape_reads_package_json() {
        let tmp = tempdir().expect("tempdir");
        fs::write(
            tmp.path().join("package.json"),
            r#"{
  "name": "my-app",
  "description": "Frontend app",
  "keywords": ["react", "ssr"],
  "version": "1.0.0"
}"#,
        )
        .unwrap();
        let summary = scrape_metadata_summary(tmp.path()).expect("summary");
        assert!(summary.contains("my-app"));
        assert!(summary.contains("Frontend app"));
        assert!(summary.contains("react"));
        assert!(summary.contains("ssr"));
    }

    #[test]
    fn scrape_reads_pyproject_project_table() {
        let tmp = tempdir().expect("tempdir");
        fs::write(
            tmp.path().join("pyproject.toml"),
            r#"[project]
name = "myproj"
description = "Python data tools"
"#,
        )
        .unwrap();
        let summary = scrape_metadata_summary(tmp.path()).expect("summary");
        assert!(summary.contains("myproj"));
        assert!(summary.contains("Python data tools"));
    }

    #[test]
    fn scrape_reads_pyproject_poetry_table() {
        let tmp = tempdir().expect("tempdir");
        fs::write(
            tmp.path().join("pyproject.toml"),
            r#"[tool.poetry]
name = "legacy-poetry"
description = "Old style"
"#,
        )
        .unwrap();
        let summary = scrape_metadata_summary(tmp.path()).expect("summary");
        assert!(summary.contains("legacy-poetry"));
        assert!(summary.contains("Old style"));
    }

    #[test]
    fn scrape_reads_go_mod_module_line() {
        let tmp = tempdir().expect("tempdir");
        fs::write(
            tmp.path().join("go.mod"),
            "module github.com/example/proj\n\ngo 1.21\n",
        )
        .unwrap();
        let summary = scrape_metadata_summary(tmp.path()).expect("summary");
        assert!(summary.contains("github.com/example/proj"));
    }

    #[test]
    fn scrape_reads_trusty_yaml_description() {
        let tmp = tempdir().expect("tempdir");
        fs::write(
            tmp.path().join("trusty-search.yaml"),
            "description: A multi-index polyrepo\nname: foo\n",
        )
        .unwrap();
        let summary = scrape_metadata_summary(tmp.path()).expect("summary");
        assert!(summary.contains("A multi-index polyrepo"));
    }

    #[test]
    fn scrape_truncates_to_max_summary_chars() {
        let tmp = tempdir().expect("tempdir");
        // Build a README well over 3000 chars.
        let big = "x".repeat(8000);
        fs::write(tmp.path().join("README.md"), &big).unwrap();
        let summary = scrape_metadata_summary(tmp.path()).expect("summary");
        assert!(summary.chars().count() <= MAX_SUMMARY_CHARS);
    }

    #[test]
    fn scrape_combines_multiple_sources_in_priority_order() {
        let tmp = tempdir().expect("tempdir");
        fs::write(tmp.path().join("README.md"), "Pitch line.").unwrap();
        fs::write(tmp.path().join("CLAUDE.md"), "AI brief.").unwrap();
        fs::write(
            tmp.path().join("Cargo.toml"),
            "[package]\nname = \"p\"\ndescription = \"d\"\n",
        )
        .unwrap();

        let summary = scrape_metadata_summary(tmp.path()).expect("summary");
        let readme_idx = summary.find("README:").unwrap();
        let claude_idx = summary.find("CLAUDE.md:").unwrap();
        let cargo_idx = summary.find("Cargo:").unwrap();
        assert!(readme_idx < claude_idx);
        assert!(claude_idx < cargo_idx);
    }

    #[test]
    fn display_summary_truncates() {
        let full = "a".repeat(2000);
        let short = make_display_summary(&full);
        assert!(short.chars().count() <= MAX_DISPLAY_SUMMARY_CHARS);
    }

    #[test]
    fn empty_readme_does_not_contribute() {
        let tmp = tempdir().expect("tempdir");
        fs::write(tmp.path().join("README.md"), "   \n\n").unwrap();
        // Nothing else present; should be None.
        assert!(scrape_metadata_summary(tmp.path()).is_none());
    }

    /// Sanity-check the cosine helper this module's downstream caller
    /// depends on for fan-out routing. The math lives in
    /// `core::mmr::cosine_similarity`; this test pins the contract:
    /// identity → 1.0, orthogonal → 0.0, mismatched lengths → 0.0.
    #[test]
    fn cosine_helper_contract() {
        use crate::core::mmr::cosine_similarity;
        assert!((cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!((cosine_similarity(&[1.0, 0.0], &[0.0, 1.0])).abs() < 1e-6);
        assert_eq!(cosine_similarity(&[1.0, 0.0], &[1.0, 0.0, 0.0]), 0.0);
    }
}
