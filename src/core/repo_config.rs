//! Repo-level configuration parsed from `<repo_root>/trusty-search.yaml`.
//!
//! Why: large polyrepos (Duetto's monolith, mono-style ML repos) need to expose
//! several logically-separate indexes — backend Java APIs vs. frontend
//! TypeScript dashboards — without dumping every file into one giant index.
//! A repo-level YAML lets project owners declare these slices once and have
//! `trusty-search index` apply them automatically.
//!
//! What: `RepoConfig` parses a YAML manifest containing one or more named
//! `IndexConfig` slices. Each slice carries:
//!   - `paths`: subtrees to walk (defaults to the whole repo)
//!   - `exclude`: glob patterns to skip on top of the built-in ignores
//!   - `languages`: extension allow-list (`rust`, `typescript`, ...)
//!   - `domain_terms`: per-repo vocabulary fed into the intent classifier so
//!     queries containing those terms route to the right intent
//!
//! Test: see the `#[cfg(test)]` block at the bottom — covers YAML round-trip,
//! missing-file handling, path resolution, language mapping, and glob matching.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// File name auto-detected at the repo root.
pub const CONFIG_FILENAME: &str = "trusty-search.yaml";

/// Top-level repo configuration. See module docs.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RepoConfig {
    #[serde(default = "default_version")]
    pub version: u32,
    pub indexes: Vec<IndexConfig>,
}

/// One named index slice within a repo.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IndexConfig {
    pub name: String,

    /// Subtrees to index, relative to the repo root. Empty / missing → walk
    /// the entire repo root (`["."]`).
    #[serde(default)]
    pub paths: Vec<String>,

    /// Glob patterns to exclude (on top of built-in `SKIP_DIRS`).
    #[serde(default)]
    pub exclude: Vec<String>,

    /// Restrict to these language identifiers (see [`language_to_exts`]).
    /// Empty → all supported extensions.
    #[serde(default)]
    pub languages: Vec<String>,

    /// Domain-specific terms injected into the intent classifier for this
    /// index. Queries containing any of these terms (case-insensitive) are
    /// nudged toward `Definition` intent when no other pattern matched.
    #[serde(default)]
    pub domain_terms: Vec<String>,
}

fn default_version() -> u32 {
    1
}

impl RepoConfig {
    /// Load `<root>/trusty-search.yaml`. Returns `Ok(None)` if the file is
    /// absent (the common case — most repos won't have one).
    ///
    /// Why: callers (the `index` CLI handler) need to cleanly distinguish
    /// "no config, fall back to single-index" from "config present but
    /// malformed, fail loudly".
    /// What: stat → read → `serde_yaml::from_str`. Surfaces parse errors via
    /// `anyhow::Error` with file path context.
    /// Test: `test_load_valid_yaml`, `test_load_missing_yaml_returns_none`,
    /// `test_load_malformed_yaml_errors`.
    pub fn load(root: &Path) -> anyhow::Result<Option<Self>> {
        let path = root.join(CONFIG_FILENAME);
        if !path.exists() {
            return Ok(None);
        }
        let raw = std::fs::read_to_string(&path)
            .map_err(|e| anyhow::anyhow!("failed to read {}: {e}", path.display()))?;
        let cfg: Self = serde_yaml::from_str(&raw)
            .map_err(|e| anyhow::anyhow!("failed to parse {}: {e}", path.display()))?;
        Ok(Some(cfg))
    }

    /// Resolve `IndexConfig::paths` to absolute paths under `root`. Empty
    /// `paths` means "the entire repo" → `vec![root.to_path_buf()]`.
    ///
    /// Why: the file walker needs absolute paths to feed `WalkDir`, but users
    /// write `paths:` entries relative to the repo root.
    /// What: joins each relative entry with `root`. `.` and empty strings
    /// also normalise to `root`.
    /// Test: `test_resolved_paths_default_is_root`, `test_resolved_paths_multiple`.
    pub fn resolved_paths(cfg: &IndexConfig, root: &Path) -> Vec<PathBuf> {
        if cfg.paths.is_empty() {
            return vec![root.to_path_buf()];
        }
        cfg.paths
            .iter()
            .map(|p| {
                let trimmed = p.trim();
                if trimmed.is_empty() || trimmed == "." {
                    root.to_path_buf()
                } else {
                    root.join(trimmed)
                }
            })
            .collect()
    }

    /// Resolve language identifiers to file-extension allow-list. Empty input
    /// returns an empty allow-list (interpreted as "all extensions allowed"
    /// by the caller).
    pub fn resolved_extensions(cfg: &IndexConfig) -> Vec<&'static str> {
        let mut out: Vec<&'static str> = Vec::new();
        for lang in &cfg.languages {
            out.extend(language_to_exts(lang));
        }
        out
    }
}

/// Map a language name (case-insensitive) to its file extensions.
///
/// Why: users write `languages: [typescript]` but the walker filters on the
/// raw extension (`.ts`, `.tsx`). One place to maintain the mapping.
/// What: returns the extensions (without the leading dot) for the language,
/// or an empty slice for unknown languages (caller should warn).
/// Test: `test_language_to_exts_known`, `test_language_to_exts_unknown`.
pub fn language_to_exts(lang: &str) -> &'static [&'static str] {
    match lang.to_ascii_lowercase().as_str() {
        "rust" | "rs" => &["rs"],
        "python" | "py" => &["py"],
        "typescript" | "ts" => &["ts", "tsx"],
        "javascript" | "js" => &["js", "jsx", "mjs", "cjs"],
        "go" => &["go"],
        "java" => &["java"],
        "c" => &["c", "h"],
        "cpp" | "c++" | "cxx" => &["cpp", "cc", "cxx", "hpp", "hh", "hxx"],
        _ => &[],
    }
}

/// Return `true` if any glob in `excludes` matches `path` (relative to root)
/// or `path` directly. Caller is responsible for choosing a stable form.
///
/// Why: `IndexConfig::exclude` patterns target both file basenames
/// (`"**/__tests__/**"`) and partial paths (`"selenium/"`). The `glob` crate's
/// `Pattern` handles both via the standard glob syntax.
/// What: parses each pattern once and tests with `Pattern::matches`. Patterns
/// that fail to parse are skipped with a warning.
/// Test: `test_glob_match_basic`, `test_glob_match_recursive`.
pub fn path_matches_any_glob(path: &Path, excludes: &[String]) -> bool {
    if excludes.is_empty() {
        return false;
    }
    let s = path.to_string_lossy();
    for pat in excludes {
        match glob::Pattern::new(pat) {
            Ok(p) => {
                if p.matches(&s) {
                    return true;
                }
                // Also try matching just the file name / any trailing segment
                // so that `selenium/` matches `<root>/api/selenium/foo.py`.
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    if p.matches(name) {
                        return true;
                    }
                }
                // And the `**/<pat>` form so plain `selenium/` works against
                // arbitrarily-nested paths.
                if !pat.starts_with("**/") {
                    let alt = format!("**/{}", pat.trim_end_matches('/'));
                    if let Ok(p2) = glob::Pattern::new(&alt) {
                        if p2.matches(&s) {
                            return true;
                        }
                    }
                    let alt2 = format!("**/{}/**", pat.trim_end_matches('/'));
                    if let Ok(p3) = glob::Pattern::new(&alt2) {
                        if p3.matches(&s) {
                            return true;
                        }
                    }
                }
            }
            Err(e) => {
                tracing::warn!("ignoring invalid exclude glob {pat:?}: {e}");
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    fn write_yaml(dir: &Path, body: &str) {
        fs::write(dir.join(CONFIG_FILENAME), body).unwrap();
    }

    #[test]
    fn test_load_valid_yaml() {
        let tmp = tempdir().unwrap();
        write_yaml(
            tmp.path(),
            r#"
version: 1
indexes:
  - name: api
    paths: [src/api]
    exclude: ["**/__tests__/**"]
    languages: [java, python]
    domain_terms: ["PMS", "rate strategy"]
  - name: ui
    paths: [ui/src]
    languages: [typescript]
"#,
        );
        let cfg = RepoConfig::load(tmp.path())
            .unwrap()
            .expect("config present");
        assert_eq!(cfg.version, 1);
        assert_eq!(cfg.indexes.len(), 2);
        assert_eq!(cfg.indexes[0].name, "api");
        assert_eq!(cfg.indexes[0].paths, vec!["src/api".to_string()]);
        assert_eq!(cfg.indexes[0].languages, vec!["java", "python"]);
        assert_eq!(
            cfg.indexes[0].domain_terms,
            vec!["PMS".to_string(), "rate strategy".to_string()]
        );
        assert_eq!(cfg.indexes[1].name, "ui");
        assert_eq!(cfg.indexes[1].languages, vec!["typescript"]);
    }

    #[test]
    fn test_load_missing_yaml_returns_none() {
        let tmp = tempdir().unwrap();
        let res = RepoConfig::load(tmp.path()).unwrap();
        assert!(res.is_none());
    }

    #[test]
    fn test_load_malformed_yaml_errors() {
        let tmp = tempdir().unwrap();
        write_yaml(tmp.path(), "not: valid: yaml: : :");
        let res = RepoConfig::load(tmp.path());
        assert!(res.is_err(), "expected parse error for malformed yaml");
    }

    #[test]
    fn test_resolved_paths_default_is_root() {
        let cfg = IndexConfig {
            name: "x".into(),
            paths: vec![],
            exclude: vec![],
            languages: vec![],
            domain_terms: vec![],
        };
        let root = Path::new("/tmp/repo");
        let resolved = RepoConfig::resolved_paths(&cfg, root);
        assert_eq!(resolved, vec![root.to_path_buf()]);
    }

    #[test]
    fn test_resolved_paths_multiple() {
        let cfg = IndexConfig {
            name: "x".into(),
            paths: vec!["src/api".into(), "services/".into(), ".".into()],
            exclude: vec![],
            languages: vec![],
            domain_terms: vec![],
        };
        let root = Path::new("/tmp/repo");
        let resolved = RepoConfig::resolved_paths(&cfg, root);
        assert_eq!(
            resolved,
            vec![
                PathBuf::from("/tmp/repo/src/api"),
                PathBuf::from("/tmp/repo/services/"),
                PathBuf::from("/tmp/repo"),
            ]
        );
    }

    #[test]
    fn test_language_to_exts_known() {
        assert_eq!(language_to_exts("rust"), &["rs"]);
        assert_eq!(language_to_exts("Python"), &["py"]);
        assert_eq!(language_to_exts("typescript"), &["ts", "tsx"]);
        assert_eq!(language_to_exts("javascript"), &["js", "jsx", "mjs", "cjs"]);
        assert_eq!(
            language_to_exts("c++"),
            &["cpp", "cc", "cxx", "hpp", "hh", "hxx"]
        );
    }

    #[test]
    fn test_language_to_exts_unknown() {
        let empty: &[&str] = &[];
        assert_eq!(language_to_exts("cobol"), empty);
        assert_eq!(language_to_exts(""), empty);
    }

    #[test]
    fn test_glob_match_basic() {
        let excludes = vec!["**/__tests__/**".to_string()];
        assert!(path_matches_any_glob(
            Path::new("/repo/src/api/__tests__/foo.py"),
            &excludes
        ));
        assert!(!path_matches_any_glob(
            Path::new("/repo/src/api/foo.py"),
            &excludes
        ));
    }

    #[test]
    fn test_glob_match_partial_segment() {
        let excludes = vec!["selenium/".to_string()];
        assert!(path_matches_any_glob(
            Path::new("/repo/api/selenium/foo.py"),
            &excludes
        ));
    }

    #[test]
    fn test_resolved_extensions() {
        let cfg = IndexConfig {
            name: "x".into(),
            paths: vec![],
            exclude: vec![],
            languages: vec!["typescript".into(), "javascript".into()],
            domain_terms: vec![],
        };
        let exts = RepoConfig::resolved_extensions(&cfg);
        assert!(exts.contains(&"ts"));
        assert!(exts.contains(&"tsx"));
        assert!(exts.contains(&"js"));
        assert!(exts.contains(&"jsx"));
    }
}
