//! Source file walker for `reindex`.
//!
//! Why: Reindexing a project requires enumerating all source files under a
//! root directory while skipping VCS/build/dependency dirs and binary noise.
//! What: `walk_source_files(root)` returns every source file (filtered by
//! [`SOURCE_EXTS`]) below `root`, skipping any directory whose name appears
//! in [`SKIP_DIRS`], plus [`should_skip_path`] and [`should_skip_content`]
//! for minification / size guards applied at read time.
//! Test: see the unit tests at the bottom.

use std::path::{Path, PathBuf};
use walkdir::WalkDir;

/// Source file extensions that the indexer can handle.
///
/// Java-heavy build files (`.gradle`, `.groovy`) and Kotlin (`.kt`, `.kts`)
/// fall back to the unknown-language sliding-window chunker since their
/// tree-sitter grammars aren't compiled in.
pub const SOURCE_EXTS: &[&str] = &[
    "rs", "py", "ts", "tsx", "js", "jsx", "mjs", "cjs", "go", "java", "c", "cpp",
    "h", "hpp", "cs", "rb", "php", "swift", "kt", "kts", "scala", "groovy", "gradle",
    "sh", "yaml", "yml", "toml", "json", "md",
];

/// Directory names to skip when walking. Matched on basename only.
///
/// Java/Gradle and JS/TS build outputs are aggressively pruned — these directories
/// contain machine-generated code that bloats the index without adding signal.
pub const SKIP_DIRS: &[&str] = &[
    ".git",
    "target",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    "dist",
    "build",
    ".next",
    ".nuxt",
    ".svelte-kit",
    "vendor",
    ".cargo",
    "coverage",
    ".nyc_output",
    ".pytest_cache",
    ".mypy_cache",
    // Java / Gradle / Maven build artefacts
    ".gradle",
    ".m2",
    "out",
    "bin",
    "classes",
    "generated",
    "generated-sources",
    "generated-test-sources",
    // IDE / metadata
    ".idea",
    ".vscode",
    // AI tooling / agent metadata — high-density prose that outranks source code in BM25
    ".claude",
    ".claude-mpm",
    ".open-mpm",
    ".cursor",
    ".aider",
    ".continue",
    ".obsidian",
];

/// File extensions that are always binary or non-parseable. These are skipped
/// before the file is opened, so the walker never reads their bytes.
const BINARY_EXTS: &[&str] = &[
    "wasm", "so", "dylib", "dll", "exe",
    "pdf", "png", "jpg", "jpeg", "gif", "ico", "webp",
    "zip", "tar", "gz", "bz2", "xz", "7z", "rar",
    "ttf", "otf", "woff", "woff2",
    "mp3", "mp4", "mov", "avi", "mkv",
    "db", "sqlite", "lock",
    "pyc", "class", "o", "a",
];

/// Files larger than this are silently skipped. 1 MiB is generous for a source
/// file; anything bigger is almost certainly generated or a data blob.
pub const MAX_FILE_BYTES: u64 = 1_048_576; // 1 MiB

/// If a JS/CSS file's longest line exceeds this and the file has fewer than
/// [`MIN_LINES_FOR_READABLE_JS`] total lines, it is treated as minified.
const MAX_LINE_LEN_FOR_MINIFIED: usize = 500;
/// A readable JS file is expected to have at least this many lines when the
/// longest line is already suspiciously long.
const MIN_LINES_FOR_READABLE_JS: usize = 5;

/// Return `true` when `path` should be silently excluded from indexing based
/// solely on the file name / extension — without reading file contents.
///
/// Covers:
/// - Binary / non-text extensions ([`BINARY_EXTS`]).
/// - Minified or bundled JS/CSS (`*.min.js`, `*.min.css`, `*.bundle.js`,
///   `*.bundle.css`, `*.chunk.js`).
/// - Hashed bundle filenames like `index-ahKOasfG.js` (8+ hex/alnum chars
///   before the `.js` extension, preceded by a `-`).
/// - Files larger than [`MAX_FILE_BYTES`] (requires `fs::metadata` — only
///   called when extension checks pass).
pub fn should_skip_path(path: &Path) -> bool {
    let file_name = match path.file_name().and_then(|n| n.to_str()) {
        Some(n) => n,
        None => return true, // no name → skip
    };
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    // Binary / non-text extensions.
    if BINARY_EXTS.iter().any(|b| *b == ext) {
        return true;
    }

    // Minified / bundled filename patterns.
    // *.min.js, *.min.css
    if file_name.ends_with(".min.js")
        || file_name.ends_with(".min.css")
        // *.bundle.js, *.bundle.css
        || file_name.ends_with(".bundle.js")
        || file_name.ends_with(".bundle.css")
        // *.chunk.js
        || file_name.ends_with(".chunk.js")
    {
        return true;
    }

    // Hashed bundle: name like `<stem>-<8+ alnum chars>.<js|css>`.
    // Only applied to JS and CSS since other extensions (e.g. .ts) are never
    // emitted as hashed bundles by bundlers.
    if ext == "js" || ext == "css" {
        if let Some(stem) = path
            .file_stem()
            .and_then(|s| s.to_str())
        {
            if is_hashed_bundle_stem(stem) {
                return true;
            }
        }
    }

    // File size guard — avoids reading giant generated files.
    if let Ok(meta) = std::fs::metadata(path) {
        if meta.len() > MAX_FILE_BYTES {
            return true;
        }
    }

    false
}

/// Return `true` when the *contents* of a JS file look minified even though
/// the filename didn't match a known minified pattern.
///
/// Heuristic: a `.js` (or `.mjs`/`.cjs`) file is treated as minified when it
/// has fewer than [`MIN_LINES_FOR_READABLE_JS`] lines **and** its longest line
/// exceeds [`MAX_LINE_LEN_FOR_MINIFIED`] characters. This catches minified
/// bundles that ship without a `.min.js` suffix.
pub fn should_skip_content(path: &Path, content: &str) -> bool {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    if !matches!(ext.as_str(), "js" | "mjs" | "cjs") {
        return false;
    }

    let line_count = content.lines().count();
    if line_count >= MIN_LINES_FOR_READABLE_JS {
        return false;
    }

    content
        .lines()
        .any(|l| l.len() > MAX_LINE_LEN_FOR_MINIFIED)
}

/// Helper: return `true` when `stem` ends with `-<8+ alnum chars>`, which is
/// the naming convention used by Vite, webpack, and Rollup for hashed bundles
/// (e.g. `index-ahKOasfG`, `vendor-1a2b3c4d5e6f7a8b`).
fn is_hashed_bundle_stem(stem: &str) -> bool {
    // Find the last '-' separator.
    let Some(dash_pos) = stem.rfind('-') else {
        return false;
    };
    let hash_part = &stem[dash_pos + 1..];
    // Must be ≥ 8 characters, all alphanumeric.
    hash_part.len() >= 8 && hash_part.chars().all(|c| c.is_ascii_alphanumeric())
}

/// Result of a walk. `skipped_dirs` reports how many entries were pruned by
/// [`SKIP_DIRS`] — useful for surfacing skipped paths in CLI output later.
pub struct WalkResult {
    pub files: Vec<PathBuf>,
    pub skipped_dirs: usize,
}

/// Walk `root` recursively and return every source file under it.
///
/// - Filtered by [`SOURCE_EXTS`].
/// - Skips any directory whose basename is in [`SKIP_DIRS`].
/// - Skips files matching [`should_skip_path`] (minified/binary/large).
/// - Skips entries that fail to stat (does not error out the whole walk).
pub fn walk_source_files(root: &Path) -> WalkResult {
    let mut files = Vec::new();
    let mut skipped_dirs = 0usize;

    let walker = WalkDir::new(root).follow_links(false).into_iter();
    let walker = walker.filter_entry(|e| {
        if e.depth() == 0 {
            return true;
        }
        if e.file_type().is_dir() {
            let name = e.file_name().to_string_lossy();
            if SKIP_DIRS.iter().any(|d| *d == name) {
                return false;
            }
        }
        true
    });

    for entry in walker {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => {
                skipped_dirs += 1;
                continue;
            }
        };
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        // Extension allow-list: only known source extensions enter the indexer.
        let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
            continue;
        };
        if !SOURCE_EXTS.iter().any(|e| e.eq_ignore_ascii_case(ext)) {
            continue;
        }
        // Path-level skip: minified filenames, binaries, large files.
        if should_skip_path(path) {
            continue;
        }
        files.push(path.to_path_buf());
    }

    WalkResult { files, skipped_dirs }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn finds_source_files_and_skips_dirs() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path();
        fs::create_dir_all(root.join("src")).unwrap();
        fs::create_dir_all(root.join("target/debug")).unwrap();
        fs::create_dir_all(root.join("node_modules/foo")).unwrap();
        fs::write(root.join("src/main.rs"), "fn main() {}").unwrap();
        fs::write(root.join("src/lib.py"), "x = 1").unwrap();
        fs::write(root.join("README.md"), "# hi").unwrap();
        fs::write(root.join("target/debug/build.o"), b"\0\0").unwrap();
        fs::write(root.join("node_modules/foo/index.js"), "// no").unwrap();
        fs::write(root.join("binary.bin"), b"\0\0").unwrap();

        let result = walk_source_files(root);
        let names: Vec<String> = result
            .files
            .iter()
            .filter_map(|p| p.file_name().map(|n| n.to_string_lossy().to_string()))
            .collect();
        assert!(names.contains(&"main.rs".to_string()));
        assert!(names.contains(&"lib.py".to_string()));
        assert!(names.contains(&"README.md".to_string()));
        assert!(!names.contains(&"build.o".to_string()));
        assert!(!names.contains(&"index.js".to_string()));
        assert!(!names.contains(&"binary.bin".to_string()));
    }

    // --- should_skip_path tests ---

    #[test]
    fn test_skips_min_js() {
        assert!(should_skip_path(Path::new("foo.min.js")));
        assert!(should_skip_path(Path::new("path/to/app.min.js")));
    }

    #[test]
    fn test_skips_min_css() {
        assert!(should_skip_path(Path::new("styles.min.css")));
    }

    #[test]
    fn test_skips_bundle_js() {
        assert!(should_skip_path(Path::new("app.bundle.js")));
        assert!(should_skip_path(Path::new("vendor.bundle.css")));
    }

    #[test]
    fn test_skips_chunk_js() {
        assert!(should_skip_path(Path::new("runtime.chunk.js")));
    }

    #[test]
    fn test_skips_hashed_bundle() {
        // 8+ alphanumeric characters after the last '-'
        assert!(should_skip_path(Path::new("index-ahKOasfG.js")));
        assert!(should_skip_path(Path::new("vendor-1a2b3c4d5e6f7a8b.js")));
        assert!(should_skip_path(Path::new("src/assets/main-AbCdEfGh.js")));
    }

    #[test]
    fn test_hashed_bundle_too_short_not_skipped() {
        // Only 7 chars after '-' — not long enough to be a hash.
        assert!(!should_skip_path(Path::new("foo-abcdefg.js")));
    }

    #[test]
    fn test_keeps_normal_js() {
        assert!(!should_skip_path(Path::new("utils.js")));
        assert!(!should_skip_path(Path::new("main.js")));
        assert!(!should_skip_path(Path::new("src/components/button.js")));
    }

    #[test]
    fn test_skips_node_modules_dir() {
        // Path-based dir skipping happens in the walker; should_skip_path
        // only looks at the file name, not ancestor dirs. But we verify that
        // the walker does exclude node_modules content end-to-end.
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path();
        fs::create_dir_all(root.join("node_modules/lodash")).unwrap();
        fs::write(root.join("node_modules/lodash/index.js"), "module.exports={}").unwrap();
        fs::write(root.join("real.js"), "export const x = 1;").unwrap();

        let result = walk_source_files(root);
        let names: Vec<String> = result
            .files
            .iter()
            .filter_map(|p| p.file_name().map(|n| n.to_string_lossy().to_string()))
            .collect();
        assert!(!names.contains(&"index.js".to_string()), "node_modules must be excluded");
        assert!(names.contains(&"real.js".to_string()));
    }

    #[test]
    fn test_skips_large_file() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let big_path = tmp.path().join("huge.js");
        // Write just over 1 MiB.
        let big_content = "x".repeat((MAX_FILE_BYTES + 1) as usize);
        fs::write(&big_path, big_content.as_bytes()).unwrap();
        assert!(should_skip_path(&big_path));
    }

    #[test]
    fn test_keeps_small_file() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let small_path = tmp.path().join("small.js");
        fs::write(&small_path, b"const x = 1;").unwrap();
        assert!(!should_skip_path(&small_path));
    }

    // --- should_skip_content tests ---

    #[test]
    fn test_skip_content_detects_minified_js() {
        // A single line longer than 500 chars with fewer than 5 total lines.
        let minified = "a".repeat(501);
        assert!(should_skip_content(Path::new("bundle.js"), &minified));
    }

    #[test]
    fn test_skip_content_allows_normal_js() {
        let normal = "const x = 1;\nconst y = 2;\nconst z = 3;\nconst w = 4;\nconst v = 5;\n";
        assert!(!should_skip_content(Path::new("app.js"), normal));
    }

    #[test]
    fn test_skip_content_ignores_non_js() {
        // Even with a very long line, non-JS files are not skipped by this check.
        let long_line = "x".repeat(1000);
        assert!(!should_skip_content(Path::new("data.rs"), &long_line));
        assert!(!should_skip_content(Path::new("query.py"), &long_line));
    }

    #[test]
    fn test_skip_content_mjs_cjs() {
        let minified = "a".repeat(501);
        assert!(should_skip_content(Path::new("mod.mjs"), &minified));
        assert!(should_skip_content(Path::new("mod.cjs"), &minified));
    }

    #[test]
    fn test_skip_content_multiline_js_not_skipped() {
        // 5+ lines, even if one line is long.
        let content = format!(
            "line1\nline2\nline3\nline4\nline5\n{}\n",
            "x".repeat(600)
        );
        assert!(!should_skip_content(Path::new("ok.js"), &content));
    }
}
