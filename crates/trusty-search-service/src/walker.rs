//! Source file walker for `reindex`.
//!
//! Why: Reindexing a project requires enumerating all source files under a
//! root directory while skipping VCS/build/dependency dirs and binary noise.
//! What: `walk_source_files(root)` returns every source file (filtered by
//! [`SOURCE_EXTS`]) below `root`, skipping any directory whose name appears
//! in [`SKIP_DIRS`].
//! Test: see the unit tests at the bottom.

use std::path::{Path, PathBuf};
use walkdir::WalkDir;

/// Source file extensions that the indexer can handle.
pub const SOURCE_EXTS: &[&str] = &[
    "rs", "py", "ts", "tsx", "js", "jsx", "go", "java", "c", "cpp", "h", "hpp",
    "cs", "rb", "swift", "kt", "scala", "sh", "yaml", "yml", "toml", "json", "md",
];

/// Directory names to skip when walking. Matched on basename only.
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
    "vendor",
    ".cargo",
    "coverage",
    ".pytest_cache",
    ".mypy_cache",
];

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
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            if SOURCE_EXTS.iter().any(|e| e.eq_ignore_ascii_case(ext)) {
                files.push(path.to_path_buf());
            }
        }
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
}
