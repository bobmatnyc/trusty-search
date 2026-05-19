//! Handler for `trusty-search integrate cursor`.
//!
//! Why: Cursor IDE consumes MCP servers via `~/.cursor/mcp.json` (global) and
//! `.cursor/mcp.json` (project-level), and teaches its AI when to use a tool
//! via `.cursor/rules/*.mdc` files. A user installing trusty-search wants a
//! single command that wires both up so the search tools are immediately
//! usable in any Cursor session — without hand-editing JSON.
//! What: `handle_integrate` dispatches on the [`IntegrateTarget`] (today only
//! `cursor`) and performs three idempotent, atomic file operations: upsert the
//! global MCP config, upsert the project MCP config, and write the project
//! rules `.mdc` file. Every write is backed up first and uses a temp-rename
//! to avoid partially-written files.
//! Test: `cargo run -- integrate cursor --dry-run` prints the planned changes
//! without touching disk; the unit tests below cover the JSON upsert,
//! backup-path generation, and MDC rendering.

use anyhow::{anyhow, Result};
use clap::ValueEnum;
use colored::Colorize;
use serde_json::{Map, Value};
use std::path::{Path, PathBuf};

// ── String resources ───────────────────────────────────────────────────────

/// The canonical key written under `mcpServers` for the trusty-search entry.
const TRUSTY_KEY: &str = "trusty-search";

/// Human-readable description embedded in the MCP server entry.
const TRUSTY_DESCRIPTION: &str = "Hybrid code search — BM25 + vector + knowledge graph";

/// File name of the Cursor MCP config (both global and project scoped).
const MCP_FILE_NAME: &str = "mcp.json";

/// Relative path (under a project root) of the generated rules file.
const RULES_REL_PATH: &str = ".cursor/rules/trusty-search.mdc";

/// Sub-directory of `~/.cursor/` where pre-modification backups are stored.
const BACKUP_DIR_NAME: &str = ".mcp-installer-backups";

/// Verbatim contents of the generated `.cursor/rules/trusty-search.mdc` file.
///
/// Why: the rules file teaches Cursor's AI when to reach for the trusty-search
/// MCP tools; pinning the exact text as a constant keeps the renderer and the
/// idempotency test in agreement.
const RULES_MDC_CONTENT: &str = r#"---
description: |
  trusty-search is available as an MCP tool for hybrid code search.
  Use it for semantic, lexical, and graph-expanded queries over the indexed codebase.
globs:
  - "**/*"
alwaysApply: true
---

# trusty-search Code Search

This project has trusty-search MCP tools available. Prefer them over grep for non-trivial queries.

- `search_code` — hybrid BM25 + vector search with KG expansion (best for most queries)
- `search_similar` — find code similar to a given file/function
- `reindex` — trigger a full reindex of this project
- `index_status` — check chunk count and index health
- `search_health` — confirm the daemon is running

## When to use
- Finding function definitions → `search_code "fn <name>"` with Definition intent
- Exploring callers of a function → `search_code "<name> callers"` with Usage intent
- Conceptual queries ("how does auth work") → `search_code` with Conceptual intent
- Finding similar implementations → `search_similar`
"#;

// ── CLI surface ─────────────────────────────────────────────────────────────

/// The editor/IDE that `integrate` is wiring trusty-search into.
///
/// Why: model the integration target as an enum (validated at parse time by
/// clap) so future targets — `vscode`, `zed` — slot in without changing the
/// CLI shape.
/// What: a single variant today — `cursor`.
/// Test: `cargo run -- integrate bogus` → clap rejects with a usage hint.
#[derive(Debug, Clone, ValueEnum)]
pub enum IntegrateTarget {
    /// Wire trusty-search into Cursor IDE (MCP config + `.mdc` rules).
    Cursor,
}

// ── Result types ────────────────────────────────────────────────────────────

/// Outcome of attempting to write or update one file.
///
/// Why: the confirmation table must distinguish a real write from an
/// idempotent skip and a backup-then-write, and `--dry-run` needs a fourth
/// "planned" state so nothing is reported as actually changed.
/// What: enumerates the terminal states of every file operation.
/// Test: unit tests assert each variant via the upsert / render helpers.
#[derive(Debug, PartialEq, Eq)]
pub enum FileStatus {
    /// The file was created fresh (did not previously exist).
    Created,
    /// The file existed and was updated; the inner path is the backup written.
    Updated(PathBuf),
    /// The file already contained the desired content — left untouched.
    Skipped,
    /// `--dry-run`: the change was computed but not written.
    Planned,
    /// An IO/parse error occurred.
    Failed(String),
}

/// Result of one file operation: the target path plus its terminal status.
///
/// Why: pairs the file path with its outcome so the confirmation-table
/// renderer can print one line per file.
/// What: returned by every file-writing helper in this module.
/// Test: unit tests inspect `status` after running the helpers on fixtures.
#[derive(Debug)]
pub struct IntegrateOutcome {
    /// The file the operation targeted.
    pub path: PathBuf,
    /// What happened to it.
    pub status: FileStatus,
}

impl IntegrateOutcome {
    /// Build an outcome with a [`FileStatus::Failed`] status.
    fn failed(path: &Path, msg: impl Into<String>) -> Self {
        Self {
            path: path.to_path_buf(),
            status: FileStatus::Failed(msg.into()),
        }
    }
}

// ── Entry point ─────────────────────────────────────────────────────────────

/// Entry point for `trusty-search integrate`.
///
/// Why: a single command that wires trusty-search into an editor so its
/// search tools are usable without hand-editing config.
/// What: dispatches on `target`, then runs the global MCP, project MCP, and
/// rules-file phases as gated by the `--global-only` / `--project-only` /
/// `--no-rules` flags. Prints a confirmation table and a restart hint.
/// Test: `integrate cursor --dry-run` prints the planned changes; the
/// per-file helpers it orchestrates are unit-tested below.
pub fn handle_integrate(
    target: IntegrateTarget,
    dry_run: bool,
    global_only: bool,
    project_only: bool,
    no_rules: bool,
    force: bool,
) -> Result<()> {
    // `target` has one variant today; the match keeps future targets explicit.
    match target {
        IntegrateTarget::Cursor => {}
    }

    if global_only && project_only {
        return Err(anyhow!(
            "--global-only and --project-only are mutually exclusive"
        ));
    }

    if dry_run {
        println!("{} Dry run — no files will be written.\n", "·".dimmed());
    }

    println!("🔌 Integrating trusty-search into Cursor IDE…\n");

    let home =
        dirs::home_dir().ok_or_else(|| anyhow!("could not determine home directory"))?;
    let mut outcomes: Vec<IntegrateOutcome> = Vec::new();

    // Phase 1 — global MCP config (~/.cursor/mcp.json).
    if !project_only {
        let global_path = home.join(".cursor").join(MCP_FILE_NAME);
        outcomes.push(upsert_mcp_config(&global_path, &home, dry_run));
    }

    // Phase 2 — project MCP config (.cursor/mcp.json in CWD).
    if !global_only {
        let cwd = std::env::current_dir()
            .map_err(|e| anyhow!("could not determine current directory: {e}"))?;
        let project_path = cwd.join(".cursor").join(MCP_FILE_NAME);
        outcomes.push(upsert_mcp_config(&project_path, &home, dry_run));

        // Phase 3 — project rules file (.cursor/rules/trusty-search.mdc).
        if !no_rules {
            let rules_path = cwd.join(RULES_REL_PATH);
            outcomes.push(write_rules_file(&rules_path, dry_run, force));
        }
    }

    print_summary(&outcomes);

    let any_failed = outcomes
        .iter()
        .any(|o| matches!(o.status, FileStatus::Failed(_)));

    println!();
    if dry_run {
        println!(
            "{} Dry run complete — re-run without --dry-run to apply.",
            "·".dimmed()
        );
    } else if any_failed {
        return Err(anyhow!("one or more files could not be written"));
    } else {
        println!(
            "{} Restart Cursor (or reload MCP servers) to activate trusty-search.",
            "✓".green()
        );
    }

    Ok(())
}

// ── MCP config upsert ───────────────────────────────────────────────────────

/// Upsert the `trusty-search` entry into a Cursor `mcp.json` file.
///
/// Why: this is the load-bearing surgery — it must create the file if absent,
/// preserve every unrelated key if present, be idempotent when the
/// `trusty-search` key already exists, and never corrupt the file on failure.
/// What: parses the file (or starts from `{}`), upserts `mcpServers.trusty-search`,
/// then — unless `dry_run` — backs the original up and writes atomically.
/// Test: `test_upsert_creates_file`, `test_upsert_preserves_other_keys`, and
/// `test_upsert_idempotent` cover the three cases.
pub fn upsert_mcp_config(path: &Path, home: &Path, dry_run: bool) -> IntegrateOutcome {
    let existed = path.exists();

    // Load the existing document, or start from an empty object.
    let (mut root, original) = if existed {
        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(e) => return IntegrateOutcome::failed(path, format!("read: {e}")),
        };
        let value: Value = match serde_json::from_str(&content) {
            Ok(v) => v,
            Err(e) => return IntegrateOutcome::failed(path, format!("parse: {e}")),
        };
        (value, Some(content))
    } else {
        (Value::Object(Map::new()), None)
    };

    // Ensure the root is an object so `mcpServers` can be nested under it.
    let obj = match root.as_object_mut() {
        Some(o) => o,
        None => {
            return IntegrateOutcome::failed(path, "top-level JSON value is not an object")
        }
    };

    let changed = upsert_trusty_entry(obj);

    // Idempotency: the entry already matched — leave the file untouched.
    if !changed {
        return IntegrateOutcome {
            path: path.to_path_buf(),
            status: FileStatus::Skipped,
        };
    }

    if dry_run {
        return IntegrateOutcome {
            path: path.to_path_buf(),
            status: FileStatus::Planned,
        };
    }

    // Persist: back up any existing file, then write atomically.
    let backup = match original {
        Some(ref content) => match write_backup(path, content, home) {
            Ok(b) => Some(b),
            Err(e) => return IntegrateOutcome::failed(path, format!("backup: {e}")),
        },
        None => None,
    };

    let pretty = match serde_json::to_string_pretty(&root) {
        Ok(p) => p,
        Err(e) => return IntegrateOutcome::failed(path, format!("serialize: {e}")),
    };
    if let Err(e) = write_atomic(path, &format!("{pretty}\n")) {
        return IntegrateOutcome::failed(path, format!("write: {e}"));
    }

    IntegrateOutcome {
        path: path.to_path_buf(),
        status: match backup {
            Some(b) => FileStatus::Updated(b),
            None => FileStatus::Created,
        },
    }
}

/// Upsert the canonical `trusty-search` entry into an MCP document's root map.
///
/// Why: separating the pure JSON transformation from the IO makes it directly
/// unit-testable and lets `upsert_mcp_config` decide whether anything changed.
/// Taking a `&mut Map` (not a `Value`) keeps the function panic-free — the
/// caller has already proven the root is an object.
/// What: ensures a `mcpServers` object exists, then inserts/replaces the
/// `trusty-search` key. Returns `true` if the document was modified, `false`
/// if it already held the exact desired entry (idempotency signal).
/// Test: `test_upsert_idempotent` asserts a second call returns `false`.
fn upsert_trusty_entry(obj: &mut Map<String, Value>) -> bool {
    let desired = trusty_server_entry();

    // Ensure `mcpServers` exists and is an object; reset it if it is not.
    let needs_servers = !obj
        .get("mcpServers")
        .map(Value::is_object)
        .unwrap_or(false);
    if needs_servers {
        obj.insert("mcpServers".to_string(), Value::Object(Map::new()));
    }

    // `mcpServers` is now guaranteed present and an object — but resolve it
    // via a graceful match rather than `.expect()` to keep this panic-free.
    let servers = match obj.get_mut("mcpServers").and_then(Value::as_object_mut) {
        Some(s) => s,
        // Unreachable given the block above, but a no-panic fallback is free.
        None => return false,
    };

    // Idempotency: if the present entry already equals the desired one and we
    // did not have to create the servers block, nothing changed.
    if !needs_servers && servers.get(TRUSTY_KEY) == Some(&desired) {
        return false;
    }

    servers.insert(TRUSTY_KEY.to_string(), desired);
    true
}

/// Build the canonical `trusty-search` MCP server JSON entry.
///
/// Why: centralizes the one true shape of the entry so the upsert logic and
/// the tests agree on `command` / `args` / `description`.
/// What: returns
/// `{"command": "trusty-search", "args": ["serve"], "description": "…"}`.
/// Test: `test_upsert_creates_file` asserts the inserted value's fields.
fn trusty_server_entry() -> Value {
    let mut entry = Map::new();
    entry.insert("command".to_string(), Value::String(TRUSTY_KEY.to_string()));
    entry.insert(
        "args".to_string(),
        Value::Array(vec![Value::String("serve".to_string())]),
    );
    entry.insert(
        "description".to_string(),
        Value::String(TRUSTY_DESCRIPTION.to_string()),
    );
    Value::Object(entry)
}

// ── Rules file ──────────────────────────────────────────────────────────────

/// Write the `.cursor/rules/trusty-search.mdc` rules file.
///
/// Why: the rules file teaches Cursor's AI when to use the trusty-search MCP
/// tools. It must be idempotent — never clobber a user-edited file — unless
/// `force` is set.
/// What: creates parent directories, then writes [`RULES_MDC_CONTENT`]
/// atomically. Skips if the file already exists and `force` is false.
/// Test: `test_write_rules_creates_file` and `test_write_rules_idempotent`.
pub fn write_rules_file(path: &Path, dry_run: bool, force: bool) -> IntegrateOutcome {
    let existed = path.exists();

    // Idempotency: an existing rules file is left alone unless --force.
    if existed && !force {
        return IntegrateOutcome {
            path: path.to_path_buf(),
            status: FileStatus::Skipped,
        };
    }

    if dry_run {
        return IntegrateOutcome {
            path: path.to_path_buf(),
            status: FileStatus::Planned,
        };
    }

    if let Some(parent) = path.parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            return IntegrateOutcome::failed(path, format!("mkdir {}: {e}", parent.display()));
        }
    }

    if let Err(e) = write_atomic(path, RULES_MDC_CONTENT) {
        return IntegrateOutcome::failed(path, format!("write: {e}"));
    }

    IntegrateOutcome {
        path: path.to_path_buf(),
        // An overwrite-with-force still reports as Created (no JSON backup
        // applies to the rules file — git history is the recovery path).
        status: FileStatus::Created,
    }
}

// ── IO helpers ──────────────────────────────────────────────────────────────

/// Atomically write `content` to `path` via a temp-file rename.
///
/// Why: a half-written config file would break the user's Cursor MCP setup;
/// writing to `<path>.tmp` then renaming guarantees readers never observe a
/// partial file.
/// What: writes `<path>.tmp`, then renames it over `path`.
/// Test: exercised by `test_upsert_creates_file` (the post-rename file exists
/// and parses).
fn write_atomic(path: &Path, content: &str) -> Result<()> {
    let tmp = sibling_with_suffix(path, "tmp")?;
    std::fs::write(&tmp, content)
        .map_err(|e| anyhow!("write temp {}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path)
        .map_err(|e| anyhow!("rename {} → {}: {e}", tmp.display(), path.display()))?;
    Ok(())
}

/// Copy the existing file's bytes into the timestamped backup directory.
///
/// Why: before modifying an MCP config, the original must be recoverable; the
/// spec mandates `~/.cursor/.mcp-installer-backups/<name>.<TS>.backup`.
/// What: ensures the backup directory exists, then writes `original` to a
/// timestamped path inside it. Returns the backup path.
/// Test: `test_backup_path_generation` asserts the path shape.
fn write_backup(path: &Path, original: &str, home: &Path) -> Result<PathBuf> {
    let backup_dir = home.join(".cursor").join(BACKUP_DIR_NAME);
    std::fs::create_dir_all(&backup_dir)
        .map_err(|e| anyhow!("create backup dir {}: {e}", backup_dir.display()))?;
    let backup_path = backup_path_for(path, &backup_dir, &now_timestamp());
    std::fs::write(&backup_path, original)
        .map_err(|e| anyhow!("write backup {}: {e}", backup_path.display()))?;
    Ok(backup_path)
}

/// Produce the timestamped backup path for `source` inside `backup_dir`.
///
/// Why: backup-path construction is pure and must be deterministic so it can
/// be unit-tested without touching the filesystem clock.
/// What: returns `<backup_dir>/<source filename>.<timestamp>.backup`.
/// Test: `test_backup_path_generation` asserts the resulting filename.
fn backup_path_for(source: &Path, backup_dir: &Path, timestamp: &str) -> PathBuf {
    let name = source
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(MCP_FILE_NAME);
    backup_dir.join(format!("{name}.{timestamp}.backup"))
}

/// Current local time formatted as `YYYYMMDD_HHMMSS`.
///
/// Why: the spec fixes the backup-suffix format; isolating it keeps
/// `backup_path_for` pure and testable.
/// What: returns e.g. `20260519_143005`.
/// Test: covered indirectly — `test_backup_path_generation` injects a fixed
/// timestamp instead of calling this.
fn now_timestamp() -> String {
    chrono::Local::now().format("%Y%m%d_%H%M%S").to_string()
}

/// Build a sibling path by appending `.<suffix>` to the full filename.
///
/// Why: `Path::with_extension` only replaces the final extension, so a file
/// like `trusty-search.mdc` would lose `mdc`. Appending to the whole name
/// yields `trusty-search.mdc.tmp` as intended.
/// What: returns `path` with `.<suffix>` glued onto the file name.
/// Test: exercised by `write_atomic` in `test_upsert_creates_file`.
fn sibling_with_suffix(path: &Path, suffix: &str) -> Result<PathBuf> {
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| anyhow!("path has no file name: {}", path.display()))?;
    Ok(path.with_file_name(format!("{name}.{suffix}")))
}

// ── Rendering ───────────────────────────────────────────────────────────────

/// Print the confirmation table summarising every file operation.
///
/// Why: the user needs an at-a-glance report of what was written, skipped, or
/// backed up.
/// What: prints one line per outcome with a status glyph and any backup path.
/// Test: visual — exercised by `integrate cursor --dry-run`.
fn print_summary(outcomes: &[IntegrateOutcome]) {
    println!("{}", "  File                                       Status".bold());
    println!("  {}", "─".repeat(60).dimmed());
    for o in outcomes {
        let path = o.path.display().to_string();
        match &o.status {
            FileStatus::Created => {
                println!("  {} {}  {}", "✓".green(), path, "created".green());
            }
            FileStatus::Updated(backup) => {
                println!(
                    "  {} {}  {}\n      {} {}",
                    "✓".green(),
                    path,
                    "updated".green(),
                    "backup →".dimmed(),
                    backup.display().to_string().dimmed()
                );
            }
            FileStatus::Skipped => {
                println!(
                    "  {} {}  {}",
                    "↻".cyan(),
                    path.dimmed(),
                    "already configured (skipped)".dimmed()
                );
            }
            FileStatus::Planned => {
                println!("  {} {}  {}", "·".dimmed(), path, "would write".cyan());
            }
            FileStatus::Failed(msg) => {
                println!(
                    "  {} {}  {}",
                    "✗".red(),
                    path.dimmed(),
                    format!("failed: {msg}").red()
                );
            }
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Why: integrating into a project that has no `~/.cursor/mcp.json` yet
    /// must create the file with a well-formed `trusty-search` entry.
    #[test]
    fn test_upsert_creates_file() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join(".cursor").join("mcp.json");
        std::fs::create_dir_all(path.parent().unwrap()).expect("mkdir");

        let outcome = upsert_mcp_config(&path, tmp.path(), false);
        assert_eq!(outcome.status, FileStatus::Created);
        assert!(path.exists(), "mcp.json should have been created");

        let written: Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let entry = &written["mcpServers"]["trusty-search"];
        assert_eq!(entry["command"], "trusty-search");
        assert_eq!(entry["args"][0], "serve");
        assert_eq!(entry["description"], TRUSTY_DESCRIPTION);
    }

    /// Why: upserting must never destroy unrelated MCP servers or top-level
    /// keys the user has configured.
    #[test]
    fn test_upsert_preserves_other_keys() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("mcp.json");
        let input = serde_json::json!({
            "someTopLevel": 42,
            "mcpServers": {
                "other-server": { "command": "other", "args": [] }
            }
        });
        std::fs::write(&path, serde_json::to_string_pretty(&input).unwrap()).unwrap();

        let outcome = upsert_mcp_config(&path, tmp.path(), false);
        assert!(matches!(outcome.status, FileStatus::Updated(_)));

        let written: Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let servers = written["mcpServers"].as_object().unwrap();
        assert!(servers.contains_key("other-server"), "unrelated server dropped");
        assert!(servers.contains_key("trusty-search"), "trusty key missing");
        assert_eq!(written["someTopLevel"], 42, "unrelated top-level key dropped");

        // The Updated backup path must live in the backup dir and exist.
        if let FileStatus::Updated(backup) = outcome.status {
            assert!(backup.exists(), "backup file missing");
            assert!(
                backup.starts_with(tmp.path().join(".cursor").join(BACKUP_DIR_NAME)),
                "backup not in backup dir: {}",
                backup.display()
            );
        }
    }

    /// Why: re-running `integrate` on an already-configured file must be a
    /// graceful no-op so the command is safe to run repeatedly.
    #[test]
    fn test_upsert_idempotent() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("mcp.json");

        // First run creates the file.
        let first = upsert_mcp_config(&path, tmp.path(), false);
        assert_eq!(first.status, FileStatus::Created);

        // Second run must skip — the entry already matches.
        let before = std::fs::read_to_string(&path).unwrap();
        let second = upsert_mcp_config(&path, tmp.path(), false);
        assert_eq!(second.status, FileStatus::Skipped);
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            before,
            "idempotent run must leave the file byte-for-byte unchanged"
        );
    }

    /// Why: `--dry-run` must compute the change but never touch disk.
    #[test]
    fn test_upsert_dry_run_writes_nothing() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("mcp.json");

        let outcome = upsert_mcp_config(&path, tmp.path(), true);
        assert_eq!(outcome.status, FileStatus::Planned);
        assert!(!path.exists(), "dry run must not create the file");
    }

    /// Why: a corrupt entry (wrong shape) must be repaired, not skipped.
    #[test]
    fn test_upsert_replaces_stale_entry() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("mcp.json");
        let input = serde_json::json!({
            "mcpServers": {
                "trusty-search": { "command": "old-binary", "args": [] }
            }
        });
        std::fs::write(&path, serde_json::to_string_pretty(&input).unwrap()).unwrap();

        let outcome = upsert_mcp_config(&path, tmp.path(), false);
        assert!(
            matches!(outcome.status, FileStatus::Updated(_)),
            "stale entry should be rewritten"
        );

        let written: Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(written["mcpServers"]["trusty-search"]["command"], "trusty-search");
    }

    /// Why: the rules file must be created with the exact spec'd MDC content.
    #[test]
    fn test_write_rules_creates_file() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join(".cursor/rules/trusty-search.mdc");

        let outcome = write_rules_file(&path, false, false);
        assert_eq!(outcome.status, FileStatus::Created);
        assert!(path.exists(), "rules file should have been created");

        let content = std::fs::read_to_string(&path).unwrap();
        assert_eq!(content, RULES_MDC_CONTENT);
        assert!(content.starts_with("---\n"), "missing MDC front-matter");
        assert!(content.contains("alwaysApply: true"), "missing alwaysApply");
        assert!(content.contains("search_code"), "missing tool reference");
    }

    /// Why: an existing rules file must be left alone unless `--force`.
    #[test]
    fn test_write_rules_idempotent() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join(".cursor/rules/trusty-search.mdc");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, "USER EDITED").unwrap();

        let outcome = write_rules_file(&path, false, false);
        assert_eq!(outcome.status, FileStatus::Skipped);
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "USER EDITED",
            "non-force run must not overwrite an existing rules file"
        );
    }

    /// Why: `--force` must overwrite a user-edited rules file with the
    /// canonical content.
    #[test]
    fn test_write_rules_force_overwrites() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join(".cursor/rules/trusty-search.mdc");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, "USER EDITED").unwrap();

        let outcome = write_rules_file(&path, false, true);
        assert_eq!(outcome.status, FileStatus::Created);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), RULES_MDC_CONTENT);
    }

    /// Why: `--dry-run` must not create the rules file.
    #[test]
    fn test_write_rules_dry_run_writes_nothing() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join(".cursor/rules/trusty-search.mdc");

        let outcome = write_rules_file(&path, true, false);
        assert_eq!(outcome.status, FileStatus::Planned);
        assert!(!path.exists(), "dry run must not create the rules file");
    }

    /// Why: the backup path must match the spec'd
    /// `<name>.<YYYYMMDD_HHMMSS>.backup` shape inside the backup dir.
    #[test]
    fn test_backup_path_generation() {
        let source = Path::new("/home/u/.cursor/mcp.json");
        let backup_dir = Path::new("/home/u/.cursor/.mcp-installer-backups");
        let path = backup_path_for(source, backup_dir, "20260519_143005");

        assert_eq!(
            path,
            backup_dir.join("mcp.json.20260519_143005.backup"),
            "backup path shape mismatch"
        );
    }

    /// Why: the live timestamp must always be the fixed-width
    /// `YYYYMMDD_HHMMSS` form so backups sort chronologically.
    #[test]
    fn test_now_timestamp_shape() {
        let ts = now_timestamp();
        assert_eq!(ts.len(), 15, "timestamp must be YYYYMMDD_HHMMSS: {ts}");
        assert_eq!(&ts[8..9], "_", "timestamp must separate date and time with _");
        assert!(
            ts.chars().enumerate().all(|(i, c)| if i == 8 {
                c == '_'
            } else {
                c.is_ascii_digit()
            }),
            "timestamp must be digits + one underscore: {ts}"
        );
    }

    /// Why: `upsert_trusty_entry` must signal "no change" on a second call so
    /// the IO layer can skip the write.
    #[test]
    fn test_upsert_entry_returns_change_flag() {
        let mut doc: Map<String, Value> = Map::new();
        assert!(
            upsert_trusty_entry(&mut doc),
            "first upsert must change the doc"
        );
        assert!(
            !upsert_trusty_entry(&mut doc),
            "second upsert on an identical doc must report no change"
        );
    }
}
