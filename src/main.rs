//! trusty-search CLI binary.
//!
//! Why: Single entry point that exposes both project-scoped commands
//! (`search`, `watch`, `status`, `init`, `add`, `remove`, `reindex`) which
//! auto-detect the index from the current working directory, and global
//! commands (`list`, `query`, `health`, `start`, `stop`, `serve`, `completions`)
//! that operate across the registry or manage the daemon.
//!
//! What: Parses CLI args via clap, resolves the active index via
//! `detect::detect_project`, and dispatches to placeholder handlers that
//! will be wired to the daemon HTTP API in subsequent issues.
//!
//! Test: `cargo run -- --help` → renders grouped command list with aliases.
//! `cargo run -- status` from inside this repo → prints `[trusty-search]`
//! detected via `.git`. `cargo test --workspace` → all tests pass.

mod detect;

use anyhow::Result;
use clap::{CommandFactory, Parser, Subcommand, ValueEnum};
use clap_complete::{generate, Shell};
use colored::Colorize;
use detect::{detect_project, DetectionMethod};
use std::io;

/// Machine-wide hybrid code search — BM25 + vector + knowledge graph.
///
/// Run from inside any project and trusty-search auto-detects the index.
/// Use `trusty-search start` to start the background service first.
#[derive(Parser)]
#[command(
    name = "trusty-search",
    version,
    author,
    propagate_version = true,
    subcommand_required = true,
    arg_required_else_help = true,
)]
struct Cli {
    /// Override the auto-detected project index
    #[arg(short = 'i', long, global = true, env = "TRUSTY_INDEX")]
    index: Option<String>,

    /// Output results as JSON
    #[arg(long, global = true)]
    json: bool,

    /// Enable debug logging
    #[arg(short, long, global = true)]
    verbose: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    // ── Project commands (auto-detect index from CWD) ──────────────────────
    /// Hybrid search in current project  [alias: s]
    ///
    /// Examples:
    ///   trusty-search search "fn authenticate"
    ///   trusty-search search "error handling" --intent conceptual
    ///   trusty-search search "TODO FIXME" --intent bugdebt --top-k 20
    #[command(alias = "s", display_order = 1)]
    Search {
        /// Search query (natural language or code)
        query: String,

        /// Number of results to return
        #[arg(short = 'k', long, default_value = "10")]
        top_k: usize,

        /// Show full chunk content instead of compact snippet
        #[arg(short, long)]
        full: bool,

        /// Force query intent classification
        #[arg(long, value_enum)]
        intent: Option<IntentArg>,

        /// Skip knowledge graph expansion
        #[arg(long)]
        no_kg: bool,

        /// Pagination offset
        #[arg(long, default_value = "0")]
        offset: usize,

        /// Max token budget for results
        #[arg(long, default_value = "8000")]
        budget: u32,
    },

    /// Watch for changes and keep index updated  [alias: w]
    ///
    /// Examples:
    ///   trusty-search watch
    ///   trusty-search watch ~/Projects/myapp
    #[command(alias = "w", display_order = 2)]
    Watch {
        /// Directory to watch (default: auto-detected project root)
        path: Option<std::path::PathBuf>,
    },

    /// Show index stats for current project  [alias: st]
    ///
    /// Examples:
    ///   trusty-search status
    ///   trusty-search status --json
    #[command(alias = "st", display_order = 3)]
    Status,

    /// Register current directory as a named index  [alias: i]
    ///
    /// Creates a .trusty-search marker file and registers with the daemon.
    ///
    /// Examples:
    ///   trusty-search init
    ///   trusty-search init ~/Projects/myapp --name myapp-prod
    #[command(alias = "i", display_order = 4)]
    Init {
        /// Directory to register (default: CWD)
        path: Option<std::path::PathBuf>,

        /// Index name (default: directory basename)
        #[arg(short, long)]
        name: Option<String>,

        /// Additional glob exclusion patterns
        #[arg(long)]
        exclude: Vec<String>,
    },

    /// Add or update a single file in the index
    ///
    /// Examples:
    ///   trusty-search add src/main.rs
    #[command(display_order = 5)]
    Add {
        /// File to index
        file: std::path::PathBuf,
    },

    /// Remove a file from the index  [alias: rm]
    ///
    /// Examples:
    ///   trusty-search remove src/old.rs
    #[command(alias = "rm", display_order = 6)]
    Remove {
        /// File to remove
        file: std::path::PathBuf,
    },

    /// Full reindex of current project (fire-and-forget)
    ///
    /// Examples:
    ///   trusty-search reindex
    ///   trusty-search reindex ~/Projects/myapp
    #[command(display_order = 7)]
    Reindex {
        /// Directory to reindex (default: auto-detected project root)
        path: Option<std::path::PathBuf>,
    },

    // ── Global / multi-index commands ─────────────────────────────────────
    /// List all registered indexes with stats  [alias: ls]
    ///
    /// Examples:
    ///   trusty-search list
    ///   trusty-search list --json
    #[command(alias = "ls", display_order = 10)]
    List,

    /// Search across all or named indexes  [alias: q]
    ///
    /// Examples:
    ///   trusty-search query "fn authenticate" --indexes "*"
    ///   trusty-search query "database pool" --indexes proj-a,proj-b
    #[command(alias = "q", display_order = 11)]
    Query {
        /// Search query
        query: String,

        /// Indexes to search: "*" for all, or comma-separated names
        #[arg(long, default_value = "*")]
        indexes: String,

        /// Number of results
        #[arg(short = 'k', long, default_value = "10")]
        top_k: usize,

        /// Show full chunk content
        #[arg(short, long)]
        full: bool,
    },

    /// Check daemon liveness and version
    ///
    /// Examples:
    ///   trusty-search health
    #[command(display_order = 12)]
    Health,

    // ── Service commands ──────────────────────────────────────────────────
    /// Start the background HTTP daemon
    ///
    /// Examples:
    ///   trusty-search start
    ///   trusty-search start --port 7878
    #[command(display_order = 20)]
    Start {
        /// Port to listen on (default: 7878, auto-selects next if busy)
        #[arg(long, default_value = "7878")]
        port: u16,
    },

    /// Stop the running background daemon
    ///
    /// Sends SIGTERM to the daemon process and waits for clean shutdown.
    ///
    /// Examples:
    ///   trusty-search stop
    #[command(display_order = 21)]
    Stop,

    /// Start MCP stdio server for Claude Code integration
    ///
    /// Examples:
    ///   trusty-search serve
    ///   trusty-search serve --http 0.0.0.0:8080
    #[command(display_order = 22)]
    Serve {
        /// Start HTTP/SSE mode instead of stdio
        #[arg(long)]
        http: Option<String>,
    },

    /// Migrate mcp-vector-search project(s) to trusty-search
    ///
    /// Reads `.mcp-vector-search/config.json` from each project, derives an
    /// index name from the project root's basename, and POSTs to the daemon
    /// to create + reindex the project.
    ///
    /// Examples:
    ///   trusty-search convert project           # convert current project
    ///   trusty-search convert all               # convert every project on this machine
    ///   trusty-search convert all --dry-run     # preview without changes
    #[command(display_order = 25)]
    Convert {
        /// What to convert: "project" (CWD) or "all" (machine-wide scan)
        #[arg(value_name = "TARGET")]
        target: ConvertTarget,

        /// Show what would be converted without contacting the daemon
        #[arg(long)]
        dry_run: bool,

        /// Maximum concurrent conversions for "all"
        #[arg(long, default_value = "4")]
        concurrency: usize,
    },

    /// Open the web management UI in the default browser
    ///
    /// Why: One-command access to the embedded admin panel — auto-detects
    /// a running daemon (or asks the user to start one) and pops the
    /// browser at `/ui`.
    /// Examples:
    ///   trusty-search ui
    ///   trusty-search ui --port 7878
    #[command(display_order = 23)]
    Ui {
        /// Port the daemon is listening on (default: read port file or 7878)
        #[arg(long)]
        port: Option<u16>,
    },

    /// Generate shell completion script
    ///
    /// Examples:
    ///   trusty-search completions zsh > ~/.zsh/completions/_trusty-search
    ///   trusty-search completions bash >> ~/.bashrc
    #[command(display_order = 30)]
    Completions {
        /// Shell to generate completions for
        #[arg(value_enum)]
        shell: Shell,
    },
}

/// Why: Allow users to override `QueryClassifier`'s automatic intent detection
/// when they know the intent up-front (e.g. searching for TODO comments).
/// What: Mirrors `trusty_search_core::QueryIntent` for the CLI surface.
/// Test: `cargo run -- search foo --intent conceptual --help` parses without error.
#[derive(Debug, Clone, ValueEnum)]
enum IntentArg {
    Definition,
    Usage,
    Conceptual,
    Bugdebt,
    Unknown,
}

/// Why: `convert` accepts a discrete operating mode, so model it as an enum
/// rather than a free-form string. Validated at parse time by clap.
/// What: `Project` operates on the CWD; `All` walks the user's home tree
/// looking for `.mcp-vector-search/config.json` files.
/// Test: `cargo run -- convert bogus` → clap rejects with usage hint.
#[derive(Debug, Clone, ValueEnum)]
enum ConvertTarget {
    /// Convert the project in the current directory (or any parent)
    Project,
    /// Convert every mcp-vector-search project on this machine
    All,
}

// ── Helpers ───────────────────────────────────────────────────────────────

/// Resolve the effective index ID: explicit `--index` flag wins, otherwise
/// auto-detect from CWD via `detect_project`.
///
/// Why: Every project-scoped command needs the same precedence rules.
/// What: Returns `(index_id, warned)` where `warned` is true when we fell back
/// to the CWD basename and should print a warning.
/// Test: With explicit Some("foo") → returns ("foo", false). With None inside
/// this repo → returns ("trusty-search", false) (detected via .git).
fn resolve_index(explicit: &Option<String>) -> (String, bool) {
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
fn print_index_header(index_id: &str, warned: bool) {
    if warned {
        eprintln!(
            "{} No .git or .trusty-search found — using directory name '{}'. \
             Run `trusty-search init` to register this project.",
            "⚠".yellow(),
            index_id
        );
    }
}

/// Resolve the daemon's base URL from the port file written by
/// `trusty-search start`. Falls back to `7878` when the file is missing,
/// so `serve` works out-of-the-box if the user starts the daemon on its
/// default port.
///
/// Why: stdio MCP servers are spawned by Claude Code and have no way to
/// learn the daemon port other than this file.
/// What: returns `http://127.0.0.1:{port}` (no trailing slash).
fn daemon_base_url() -> String {
    let port = daemon_port_path()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|s| s.trim().parse::<u16>().ok())
        .unwrap_or(7878);
    format!("http://127.0.0.1:{port}")
}

/// Path to `~/.local/share/trusty-search/daemon.port` (or platform equivalent).
fn daemon_port_path() -> Option<std::path::PathBuf> {
    dirs::data_local_dir().map(|d| d.join("trusty-search").join("daemon.port"))
}

/// Index a single file via the daemon's `/indexes/:id/index-file` endpoint.
async fn index_single_file(
    client: &reqwest::Client,
    base: &str,
    index_id: &str,
    file: &std::path::Path,
) -> Result<()> {
    let content = tokio::fs::read_to_string(file)
        .await
        .map_err(|e| anyhow::anyhow!("read {}: {e}", file.display()))?;
    let url = format!("{}/indexes/{}/index-file", base, index_id);
    let body = serde_json::json!({
        "path": file.display().to_string(),
        "content": content,
    });
    let resp = client.post(&url).json(&body).send().await?;
    if !resp.status().is_success() {
        anyhow::bail!("daemon returned {} for {}", resp.status(), url);
    }
    Ok(())
}

/// Handle `trusty-search add <path>`: a single file goes to `index-file`;
/// a directory walks `walk_source_files` and indexes every match.
async fn add_path(index_id: &str, path: &std::path::Path) -> Result<()> {
    let base = daemon_base_url();
    let client = reqwest::Client::new();

    if path.is_dir() {
        let walk = trusty_search_service::walker::walk_source_files(path);
        println!(
            "{} [{}] indexing {} files under {}",
            "→".cyan(),
            index_id,
            walk.files.len(),
            path.display()
        );
        let mut ok = 0usize;
        let mut err = 0usize;
        for f in &walk.files {
            match index_single_file(&client, &base, index_id, f).await {
                Ok(()) => ok += 1,
                Err(e) => {
                    eprintln!("  {} {}: {e}", "⚠".yellow(), f.display());
                    err += 1;
                }
            }
        }
        println!(
            "{} indexed {} files ({} errors)",
            "✓".green(),
            ok,
            err
        );
        Ok(())
    } else {
        index_single_file(&client, &base, index_id, path).await?;
        println!("{} [{}] {}", "→".cyan(), index_id, path.display());
        Ok(())
    }
}

/// Drive a reindex: POST /reindex, then connect to the SSE stream and render
/// progress.
async fn run_reindex(index_id: &str, root_path: &std::path::Path) -> Result<()> {
    use std::io::Write;

    let base = daemon_base_url();
    let client = reqwest::Client::new();

    let kickoff_url = format!("{}/indexes/{}/reindex", base, index_id);
    let kickoff_body = serde_json::json!({ "root_path": root_path });
    let kickoff = client
        .post(&kickoff_url)
        .json(&kickoff_body)
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("could not reach daemon at {base}: {e}"))?;

    if kickoff.status() == reqwest::StatusCode::NOT_FOUND {
        eprintln!(
            "{} index '{}' is not registered on the daemon — run `trusty-search init` first",
            "✗".red(),
            index_id
        );
        std::process::exit(1);
    }
    if !kickoff.status().is_success() {
        anyhow::bail!("daemon returned {} for reindex kickoff", kickoff.status());
    }

    let kickoff_body: serde_json::Value =
        kickoff.json().await.unwrap_or_else(|_| serde_json::json!({}));
    let stream_path = kickoff_body
        .get("stream_url")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("/indexes/{}/reindex/stream", index_id));
    let stream_url = format!("{}{}", base, stream_path);

    let resp = client
        .get(&stream_url)
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("could not connect to SSE stream {stream_url}: {e}"))?;
    if !resp.status().is_success() {
        eprintln!(
            "{} reindex stream returned {} — daemon may be an older version that doesn't support /reindex/stream",
            "✗".red(),
            resp.status()
        );
        std::process::exit(1);
    }
    let mut resp = resp;

    let mut buf = String::new();
    let mut done = false;
    while !done {
        let chunk = match resp.chunk().await {
            Ok(Some(c)) => c,
            Ok(None) => break,
            Err(e) => {
                eprintln!("\n{} stream read error: {e}", "⚠".yellow());
                break;
            }
        };
        let text = match std::str::from_utf8(&chunk) {
            Ok(s) => s,
            Err(_) => continue,
        };
        buf.push_str(text);

        // SSE events are separated by blank lines. Process each complete one.
        while let Some(idx) = buf.find("\n\n") {
            let event_block: String = buf.drain(..idx + 2).collect();
            for line in event_block.lines() {
                let Some(json_str) = line.strip_prefix("data: ").or_else(|| line.strip_prefix("data:")) else {
                    continue;
                };
                let evt: serde_json::Value = match serde_json::from_str(json_str.trim()) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                match evt.get("event").and_then(|v| v.as_str()) {
                    Some("start") => {
                        let total = evt
                            .get("total_files")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0);
                        println!(
                            "{} Reindexing {} files in '{}'…",
                            "⟳".cyan(),
                            total,
                            index_id
                        );
                    }
                    Some("progress") => {
                        let indexed = evt
                            .get("indexed")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0) as usize;
                        let total = evt
                            .get("total_files")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(1)
                            .max(1) as usize;
                        let file = evt.get("file").and_then(|v| v.as_str()).unwrap_or("");
                        let pct = (indexed * 100) / total;
                        let trimmed: String = if file.len() > 50 {
                            let start = file.len() - 50;
                            format!("…{}", &file[start..])
                        } else {
                            file.to_string()
                        };
                        print!("\r  [{:>3}%] {}/{} — {:<60}", pct, indexed, total, trimmed);
                        let _ = std::io::stdout().flush();
                    }
                    Some("complete") => {
                        let indexed = evt
                            .get("indexed")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0);
                        let chunks = evt
                            .get("total_chunks")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0);
                        let ms = evt
                            .get("elapsed_ms")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0);
                        let errors = evt
                            .get("errors")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0);
                        println!(
                            "\n{} Done: {} files, {} chunks in {}ms ({} errors)",
                            "✓".green(),
                            indexed,
                            chunks,
                            ms,
                            errors
                        );
                        done = true;
                    }
                    Some("error") => {
                        let msg = evt
                            .get("message")
                            .and_then(|v| v.as_str())
                            .unwrap_or("unknown");
                        let file = evt.get("file").and_then(|v| v.as_str()).unwrap_or("");
                        eprintln!("\n{}  {}: {}", "⚠".yellow(), file, msg);
                    }
                    _ => {}
                }
            }
        }
    }

    Ok(())
}

// ── Convert helpers (mcp-vector-search → trusty-search migration) ─────────

/// Subset of mcp-vector-search's `config.json` we care about.
///
/// Why: only `project_root` is needed to derive an index name and reindex
/// path. Every other field (file_extensions, embedding_model, ...) is
/// re-derived from the project tree at index time.
/// What: serde-deserialized from the JSON config.
/// Test: parse a config containing extra unknown fields → succeeds.
#[derive(Debug, serde::Deserialize)]
struct MvsConfig {
    project_root: std::path::PathBuf,
}

/// Walk up from `start` looking for `.mcp-vector-search/config.json`.
///
/// Why: the user may invoke `convert project` from a subdirectory of the
/// project; mirror git's discovery behaviour.
/// What: returns `Some(path)` of the first config found, else `None`.
/// Test: in a directory with no config and no `.mcp-vector-search` ancestors
/// → returns `None`.
fn find_mvs_config(start: &std::path::Path) -> Option<std::path::PathBuf> {
    let mut dir = start.to_path_buf();
    loop {
        let candidate = dir.join(".mcp-vector-search").join("config.json");
        if candidate.exists() {
            return Some(candidate);
        }
        if !dir.pop() {
            return None;
        }
    }
}

/// Find every `*/.mcp-vector-search/config.json` under the user's home dir.
///
/// Why: `convert all` needs to enumerate every project that mcp-vector-search
/// has ever indexed. Capping depth at 6 keeps the scan well under a second
/// even on dense home directories while covering typical repo layouts
/// (`~/Projects/foo`, `~/Clients/x/projects/y`, etc.).
/// What: returns absolute paths to each `config.json`. Skips errors silently.
/// Test: scan home → returns >0 paths on a machine with mcp-vector-search
/// installed; deterministic for a given filesystem state.
fn find_all_mvs_configs() -> Vec<std::path::PathBuf> {
    let home = match dirs::home_dir() {
        Some(h) => h,
        None => return Vec::new(),
    };
    let mut configs = Vec::new();
    for entry in walkdir::WalkDir::new(&home)
        .max_depth(6)
        .follow_links(false)
        .into_iter()
        .filter_entry(|e| {
            // Skip obvious noise that can't contain user projects but bloats
            // the walk: hidden caches, language toolchains, OS junk.
            let name = e.file_name().to_string_lossy();
            !matches!(
                name.as_ref(),
                "node_modules"
                    | ".git"
                    | "target"
                    | "Library"
                    | ".cache"
                    | ".cargo"
                    | ".rustup"
                    | ".npm"
                    | ".pnpm"
                    | ".pyenv"
                    | ".nvm"
                    | "venv"
                    | ".venv"
                    | "__pycache__"
            )
        })
        .filter_map(|e| e.ok())
    {
        if entry.file_name() == "config.json"
            && entry
                .path()
                .parent()
                .and_then(|p| p.file_name())
                .map(|n| n == ".mcp-vector-search")
                .unwrap_or(false)
        {
            configs.push(entry.path().to_path_buf());
        }
    }
    configs
}

/// Parse a mcp-vector-search config and derive `(project_root, index_name)`.
///
/// Why: the trusty-search index id is the lowercased basename of the project
/// root with spaces replaced by hyphens — same convention used by `init`.
/// What: returns the canonical pair. Errors propagate as anyhow.
/// Test: project_root="/Users/x/My Project" → ("/Users/x/My Project", "my-project").
fn parse_mvs_config(config_path: &std::path::Path) -> Result<(std::path::PathBuf, String)> {
    let content = std::fs::read_to_string(config_path)
        .map_err(|e| anyhow::anyhow!("read {}: {e}", config_path.display()))?;
    let config: MvsConfig = serde_json::from_str(&content)
        .map_err(|e| anyhow::anyhow!("parse {}: {e}", config_path.display()))?;
    let name = config
        .project_root
        .file_name()
        .map(|n| n.to_string_lossy().to_lowercase().replace(' ', "-"))
        .unwrap_or_else(|| "project".to_string());
    Ok((config.project_root, name))
}

#[derive(Debug)]
enum ConvertStatus {
    Queued,
    AlreadyRegistered,
    DryRun,
    Failed(String),
}

#[derive(Debug)]
struct ConvertResult {
    name: String,
    path: std::path::PathBuf,
    status: ConvertStatus,
}

/// Convert one project: register it with the daemon (idempotent) and trigger
/// a reindex.
///
/// Why: the daemon's `POST /indexes` returns `{created: false, reason: "already exists"}`
/// for known indexes and `{created: true}` for new ones — both are 200 OK,
/// so we read the body to distinguish the cases for reporting.
/// What: returns a `ConvertResult` capturing the outcome. Network errors
/// surface as `Failed`.
/// Test: dry_run=true → returns `DryRun` without any HTTP traffic.
async fn convert_one(
    project_root: std::path::PathBuf,
    index_name: String,
    base_url: &str,
    dry_run: bool,
) -> ConvertResult {
    if dry_run {
        return ConvertResult {
            name: index_name,
            path: project_root,
            status: ConvertStatus::DryRun,
        };
    }

    let client = reqwest::Client::new();

    // 1. Register the index. 200 with body.created=false means it already
    //    existed — still proceed to reindex so the user gets a fresh build.
    let create_url = format!("{base_url}/indexes");
    let create_resp = client
        .post(&create_url)
        .json(&serde_json::json!({
            "id": index_name,
            "root_path": project_root,
        }))
        .send()
        .await;

    let already_existed = match create_resp {
        Ok(resp) if resp.status().is_success() => {
            let body: serde_json::Value = resp.json().await.unwrap_or_else(|_| serde_json::json!({}));
            !body.get("created").and_then(|v| v.as_bool()).unwrap_or(true)
        }
        Ok(resp) => {
            return ConvertResult {
                name: index_name,
                path: project_root,
                status: ConvertStatus::Failed(format!("create returned {}", resp.status())),
            };
        }
        Err(e) => {
            return ConvertResult {
                name: index_name,
                path: project_root,
                status: ConvertStatus::Failed(format!("create error: {e}")),
            };
        }
    };

    // 2. Kick off reindex (fire-and-forget — we don't follow the SSE stream
    //    here because `convert all` may have many parallel migrations).
    let reindex_url = format!("{base_url}/indexes/{index_name}/reindex");
    let reindex_resp = client
        .post(&reindex_url)
        .json(&serde_json::json!({ "root_path": project_root }))
        .send()
        .await;

    match reindex_resp {
        Ok(resp) if resp.status().is_success() => ConvertResult {
            name: index_name,
            path: project_root,
            status: if already_existed {
                ConvertStatus::AlreadyRegistered
            } else {
                ConvertStatus::Queued
            },
        },
        Ok(resp) => ConvertResult {
            name: index_name,
            path: project_root,
            status: ConvertStatus::Failed(format!("reindex returned {}", resp.status())),
        },
        Err(e) => ConvertResult {
            name: index_name,
            path: project_root,
            status: ConvertStatus::Failed(format!("reindex error: {e}")),
        },
    }
}

/// Render one ConvertResult line for the `convert all` table.
fn print_convert_line(idx: usize, total: usize, r: &ConvertResult) {
    let prefix = format!("[{}/{}]", idx, total);
    let path = r.path.display().to_string();
    match &r.status {
        ConvertStatus::Queued => {
            println!(
                "  {} {} {:<24} → {}",
                prefix.dimmed(),
                "✓".green(),
                r.name,
                path.dimmed()
            );
        }
        ConvertStatus::AlreadyRegistered => {
            println!(
                "  {} {} {:<24} → {} {}",
                prefix.dimmed(),
                "↻".cyan(),
                r.name,
                path.dimmed(),
                "(already registered, reindexing)".dimmed()
            );
        }
        ConvertStatus::DryRun => {
            println!("  {} {:<24} {}", prefix.dimmed(), r.name, path.dimmed());
        }
        ConvertStatus::Failed(msg) => {
            println!(
                "  {} {} {:<24} → {} {}",
                prefix.dimmed(),
                "✗".red(),
                r.name,
                path.dimmed(),
                format!("({})", msg).red()
            );
        }
    }
}

// ── Main ──────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::from_filename(".env.local").ok();

    let cli = Cli::parse();

    if cli.verbose {
        // SAFETY: env mutation happens before any threads are spawned that
        // observe it. tracing-subscriber reads RUST_LOG on init below.
        std::env::set_var("RUST_LOG", "debug");
    }
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_target(false)
        .compact()
        .init();

    let no_color = std::env::var("NO_COLOR").is_ok()
        || std::env::var("TERM").as_deref() == Ok("dumb");
    if no_color {
        colored::control::set_override(false);
    }

    match cli.command {
        Commands::Search {
            query,
            top_k,
            full: _,
            intent: _,
            no_kg: _,
            offset: _,
            budget: _,
        } => {
            let (index_id, warned) = resolve_index(&cli.index);
            print_index_header(&index_id, warned);
            println!(
                "{} {} {} {}",
                "→".cyan(),
                format!("[{}]", index_id).dimmed(),
                query.bold(),
                format!("(top-{})", top_k).dimmed()
            );
            println!(
                "{}",
                "  Daemon connection not yet implemented — see issue #3".yellow()
            );
        }

        Commands::Watch { path } => {
            let (index_id, warned) = resolve_index(&cli.index);
            print_index_header(&index_id, warned);
            let watch_path = path.unwrap_or_else(|| {
                let cwd = std::env::current_dir().unwrap_or_default();
                detect_project(&cwd).root_path
            });
            println!(
                "{} Watching {} as index {}",
                "◉".green(),
                watch_path.display().to_string().cyan(),
                format!("'{}'", index_id).bold()
            );
            println!(
                "{}",
                "  FileWatcher not yet implemented — see issue #6".yellow()
            );
        }

        Commands::Status => {
            // Why: a multi-index status overview is more useful than a single-index
            // dump — the user wants to know "is the daemon up, and what's it serving".
            // Calls /health + /indexes + per-index /status.
            let base = daemon_base_url();
            let client = reqwest::Client::new();

            let health = client.get(format!("{}/health", base)).send().await;
            let health_body: serde_json::Value = match health {
                Ok(r) if r.status().is_success() => r.json().await.unwrap_or_else(|_| serde_json::json!({})),
                _ => {
                    if cli.json {
                        println!(r#"{{"daemon":"not_running"}}"#);
                    } else {
                        eprintln!("{} Daemon: not running ({})", "✗".red(), base);
                    }
                    std::process::exit(1);
                }
            };

            let list = client.get(format!("{}/indexes", base)).send().await;
            let list_body: serde_json::Value = match list {
                Ok(r) if r.status().is_success() => r.json().await.unwrap_or_else(|_| serde_json::json!({})),
                _ => serde_json::json!({"indexes": []}),
            };
            let empty: Vec<serde_json::Value> = Vec::new();
            let names: Vec<String> = list_body
                .get("indexes")
                .and_then(|v| v.as_array())
                .unwrap_or(&empty)
                .iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect();

            // Fetch per-index status concurrently.
            let mut joinset = tokio::task::JoinSet::new();
            for name in &names {
                let n = name.clone();
                let url = format!("{}/indexes/{}/status", base, n);
                let c = client.clone();
                joinset.spawn(async move {
                    let body: serde_json::Value = match c.get(&url).send().await {
                        Ok(r) if r.status().is_success() => {
                            r.json().await.unwrap_or_else(|_| serde_json::json!({}))
                        }
                        _ => serde_json::json!({}),
                    };
                    (n, body)
                });
            }
            let mut per_index: Vec<(String, serde_json::Value)> = Vec::new();
            while let Some(j) = joinset.join_next().await {
                if let Ok(pair) = j {
                    per_index.push(pair);
                }
            }
            per_index.sort_by(|a, b| a.0.cmp(&b.0));

            if cli.json {
                let arr: Vec<serde_json::Value> = per_index
                    .iter()
                    .map(|(n, b)| serde_json::json!({"id": n, "status": b}))
                    .collect();
                println!(
                    "{}",
                    serde_json::json!({
                        "daemon": "running",
                        "url": base,
                        "version": health_body.get("version").cloned().unwrap_or(serde_json::json!(null)),
                        "indexes": arr,
                    })
                );
            } else {
                println!(
                    "{} running ({})",
                    "Daemon:".bold(),
                    base.cyan()
                );
                if per_index.is_empty() {
                    println!("{} {}", "Indexes:".bold(), "(none)".dimmed());
                } else {
                    println!("{}", "Indexes:".bold());
                    for (name, body) in &per_index {
                        let chunks = body
                            .get("chunk_count")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0);
                        let files = body
                            .get("file_count")
                            .and_then(|v| v.as_u64());
                        match files {
                            Some(f) => println!(
                                "  {:<24} — {} chunks, {} files",
                                name.bold(),
                                chunks,
                                f
                            ),
                            None => println!(
                                "  {:<24} — {} chunks",
                                name.bold(),
                                chunks
                            ),
                        }
                    }
                }
            }
        }

        Commands::Init {
            path,
            name,
            exclude,
        } => {
            let cwd = std::env::current_dir().unwrap_or_default();
            let project_path = path.unwrap_or(cwd);
            let index_name = name.unwrap_or_else(|| {
                project_path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned()
            });

            let marker = project_path.join(".trusty-search");
            if !marker.exists() {
                std::fs::write(&marker, format!("index = \"{}\"\n", index_name))?;
                println!("{} Created {}", "✓".green(), marker.display());
            } else {
                println!("{} {} already exists", "·".dimmed(), marker.display());
            }

            if !exclude.is_empty() {
                println!("{} Extra exclusions: {}", "·".dimmed(), exclude.join(", "));
            }

            // Why: previously we printed "Registered" without contacting the daemon
            // — misleading because the daemon had no idea about this index.
            // Now: POST /indexes (idempotent) and report truthfully.
            let base = daemon_base_url();
            let client = reqwest::Client::new();
            let create_url = format!("{}/indexes", base);
            let create_body = serde_json::json!({
                "id": index_name,
                "root_path": project_path,
            });
            match client.post(&create_url).json(&create_body).send().await {
                Ok(resp) if resp.status().is_success() => {
                    let body: serde_json::Value =
                        resp.json().await.unwrap_or_else(|_| serde_json::json!({}));
                    let created = body
                        .get("created")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    if created {
                        println!(
                            "{} Registered '{}' with daemon at {}",
                            "✓".green(),
                            index_name.bold(),
                            project_path.display()
                        );
                    } else {
                        println!(
                            "{} '{}' already registered with daemon",
                            "↻".cyan(),
                            index_name.bold()
                        );
                    }
                    println!(
                        "  Run {} to index this project.",
                        "trusty-search reindex".cyan()
                    );
                }
                Ok(resp) => {
                    eprintln!(
                        "{} daemon returned {} for /indexes — index will need to be re-registered when daemon is healthy",
                        "⚠".yellow(),
                        resp.status()
                    );
                }
                Err(_) => {
                    println!(
                        "{} Daemon not running — index will be created when daemon starts.",
                        "·".dimmed()
                    );
                    println!(
                        "  Start with {} then run {}.",
                        "trusty-search start".cyan(),
                        "trusty-search reindex".cyan()
                    );
                }
            }
        }

        Commands::Add { file } => {
            let (index_id, warned) = resolve_index(&cli.index);
            print_index_header(&index_id, warned);
            add_path(&index_id, &file).await?;
        }

        Commands::Remove { file } => {
            let (index_id, warned) = resolve_index(&cli.index);
            print_index_header(&index_id, warned);
            let base = daemon_base_url();
            let url = format!("{}/indexes/{}/remove-file", base, index_id);
            let client = reqwest::Client::new();
            let body = serde_json::json!({ "path": file.display().to_string() });
            match client.post(&url).json(&body).send().await {
                Ok(resp) if resp.status().is_success() => {
                    println!("{} [{}] removed {}", "−".red(), index_id, file.display());
                }
                Ok(resp) => {
                    eprintln!("{} daemon returned {} for {}", "✗".red(), resp.status(), url);
                    std::process::exit(1);
                }
                Err(e) => {
                    eprintln!("{} could not reach daemon at {}: {e}", "✗".red(), base);
                    std::process::exit(1);
                }
            }
        }

        Commands::Reindex { path } => {
            let (index_id, warned) = resolve_index(&cli.index);
            print_index_header(&index_id, warned);
            let reindex_path = path.unwrap_or_else(|| {
                let cwd = std::env::current_dir().unwrap_or_default();
                detect_project(&cwd).root_path
            });
            run_reindex(&index_id, &reindex_path).await?;
        }

        Commands::List => {
            let base = daemon_base_url();
            let url = format!("{}/indexes", base);
            match reqwest::get(&url).await {
                Ok(resp) if resp.status().is_success() => {
                    let body: serde_json::Value =
                        resp.json().await.unwrap_or_else(|_| serde_json::json!({}));
                    if cli.json {
                        println!("{}", body);
                    } else {
                        println!("{}", "Registered indexes:".bold());
                        let empty: Vec<serde_json::Value> = Vec::new();
                        let arr = body
                            .get("indexes")
                            .and_then(|v| v.as_array())
                            .unwrap_or(&empty);
                        if arr.is_empty() {
                            println!("  {}", "(none)".dimmed());
                        } else {
                            for v in arr {
                                if let Some(s) = v.as_str() {
                                    println!("  • {}", s);
                                }
                            }
                        }
                    }
                }
                Ok(resp) => {
                    eprintln!("{} daemon returned {}", "✗".red(), resp.status());
                    std::process::exit(1);
                }
                Err(e) => {
                    eprintln!("{} could not reach daemon at {}: {e}", "✗".red(), base);
                    std::process::exit(1);
                }
            }
        }

        Commands::Query {
            query,
            indexes,
            top_k,
            full,
        } => {
            // Why: resolve which index to search.
            // Precedence: --index flag > --indexes (single name) > auto-detect (if "*" and one index).
            // For multi-index "*" with several indexes registered, require explicit choice
            // because the daemon's search endpoint is single-index-scoped.
            let base = daemon_base_url();
            let client = reqwest::Client::new();

            let target_id: String = if let Some(id) = cli.index.as_ref() {
                id.clone()
            } else if indexes != "*" && !indexes.contains(',') {
                indexes.clone()
            } else {
                // Try to resolve by listing.
                let resp = client.get(format!("{}/indexes", base)).send().await;
                match resp {
                    Ok(r) if r.status().is_success() => {
                        let body: serde_json::Value =
                            r.json().await.unwrap_or_else(|_| serde_json::json!({}));
                        let empty: Vec<serde_json::Value> = Vec::new();
                        let names: Vec<String> = body
                            .get("indexes")
                            .and_then(|v| v.as_array())
                            .unwrap_or(&empty)
                            .iter()
                            .filter_map(|v| v.as_str().map(|s| s.to_string()))
                            .collect();
                        if names.len() == 1 {
                            names.into_iter().next().unwrap()
                        } else {
                            eprintln!(
                                "{} Multiple indexes registered — please pass --index <id>: {}",
                                "✗".red(),
                                names.join(", ")
                            );
                            std::process::exit(1);
                        }
                    }
                    _ => {
                        eprintln!(
                            "{} could not reach daemon at {}",
                            "✗".red(),
                            base
                        );
                        std::process::exit(1);
                    }
                }
            };

            let url = format!("{}/indexes/{}/search", base, target_id);
            let body = serde_json::json!({"text": query, "top_k": top_k});
            let resp = client.post(&url).json(&body).send().await;
            let body_json: serde_json::Value = match resp {
                Ok(r) if r.status().is_success() => {
                    r.json().await.unwrap_or_else(|_| serde_json::json!({}))
                }
                Ok(r) if r.status() == reqwest::StatusCode::NOT_FOUND => {
                    eprintln!(
                        "{} index '{}' not found on daemon",
                        "✗".red(),
                        target_id
                    );
                    std::process::exit(1);
                }
                Ok(r) => {
                    eprintln!("{} daemon returned {}", "✗".red(), r.status());
                    std::process::exit(1);
                }
                Err(e) => {
                    eprintln!("{} could not reach daemon at {}: {e}", "✗".red(), base);
                    std::process::exit(1);
                }
            };

            if cli.json {
                println!("{}", body_json);
            } else {
                let empty: Vec<serde_json::Value> = Vec::new();
                let results = body_json
                    .get("results")
                    .and_then(|v| v.as_array())
                    .unwrap_or(&empty);
                let intent = body_json
                    .get("intent")
                    .and_then(|v| v.as_str())
                    .unwrap_or("?");
                let latency = body_json
                    .get("latency_ms")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                println!(
                    "{} [{}] {} {}",
                    "→".cyan(),
                    target_id.dimmed(),
                    query.bold(),
                    format!("(intent={}, {}ms, {} results)", intent, latency, results.len()).dimmed()
                );
                if results.is_empty() {
                    println!("  {}", "(no matches)".dimmed());
                }
                for (i, r) in results.iter().enumerate() {
                    let file = r.get("file").and_then(|v| v.as_str()).unwrap_or("?");
                    let start = r.get("start_line").and_then(|v| v.as_u64()).unwrap_or(0);
                    let end = r.get("end_line").and_then(|v| v.as_u64()).unwrap_or(0);
                    let score = r.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let reason = r
                        .get("match_reason")
                        .and_then(|v| v.as_str())
                        .unwrap_or("?");
                    println!(
                        "[{}] {}:{}-{}  {}",
                        i + 1,
                        file,
                        start,
                        end,
                        format!("(score: {:.3}, {})", score, reason).dimmed()
                    );
                    let snippet = if full {
                        r.get("content").and_then(|v| v.as_str()).unwrap_or("")
                    } else {
                        r.get("compact_snippet")
                            .and_then(|v| v.as_str())
                            .or_else(|| r.get("content").and_then(|v| v.as_str()))
                            .unwrap_or("")
                    };
                    for line in snippet.lines().take(if full { usize::MAX } else { 7 }) {
                        println!("    {}", line);
                    }
                    if !full && snippet.lines().count() > 7 {
                        println!("    {}", "...".dimmed());
                    }
                }
            }
        }

        Commands::Health => {
            let url = format!("{}/health", daemon_base_url());
            match reqwest::get(&url).await {
                Ok(resp) if resp.status().is_success() => {
                    let body: serde_json::Value =
                        resp.json().await.unwrap_or_else(|_| serde_json::json!({}));
                    if cli.json {
                        println!("{}", body);
                    } else {
                        println!(
                            "{} daemon ok at {} (version {})",
                            "✓".green(),
                            daemon_base_url().cyan(),
                            body.get("version")
                                .and_then(|v| v.as_str())
                                .unwrap_or("?")
                        );
                    }
                }
                Ok(resp) => {
                    eprintln!("{} daemon returned {}", "✗".red(), resp.status());
                    std::process::exit(1);
                }
                Err(e) => {
                    eprintln!(
                        "{} could not reach daemon at {}: {e}",
                        "✗".red(),
                        daemon_base_url()
                    );
                    std::process::exit(1);
                }
            }
        }

        Commands::Start { port } => {
            // Open the canonical facts store next to the daemon lockfile.
            // Why: facts persist across daemon restarts and are scoped per-machine
            // (single install). Falling back to `None` keeps the daemon usable if
            // the data dir is read-only — `/facts` endpoints will return 503.
            let facts = match dirs::data_local_dir() {
                Some(d) => {
                    let dir = d.join("trusty-search");
                    if let Err(e) = std::fs::create_dir_all(&dir) {
                        tracing::warn!("could not create facts dir {}: {e}", dir.display());
                        None
                    } else {
                        match trusty_search_core::FactStore::open(&dir.join("facts.redb")) {
                            Ok(s) => Some(s),
                            Err(e) => {
                                tracing::warn!("could not open facts store: {e}");
                                None
                            }
                        }
                    }
                }
                None => None,
            };
            // Bug A fix: build a single FastEmbedder up front and share it
            // across every index registered during the daemon's lifetime.
            // Without this, `create_index_handler` constructs a BM25-only
            // `CodeIndexer` and the HNSW lane silently contributes nothing
            // — the symptom seen in the 115k-chunk benchmark where every
            // result returned `match_reason: "bm25"`.
            let embedder: Option<std::sync::Arc<dyn trusty_search_core::Embedder>> =
                match trusty_search_core::FastEmbedder::new().await {
                    Ok(e) => Some(std::sync::Arc::new(e)),
                    Err(e) => {
                        tracing::warn!(
                            "FastEmbedder init failed ({e}); daemon falling back to BM25-only mode"
                        );
                        None
                    }
                };
            let mut state = trusty_search_service::SearchAppState::new(
                trusty_search_core::registry::IndexRegistry::new(),
                facts,
            );
            if let Some(e) = embedder {
                state = state.with_embedder(e);
            }
            match trusty_search_service::run_daemon(state, port).await {
                Ok(()) => {}
                Err(trusty_search_service::DaemonError::AlreadyRunning(p)) => {
                    eprintln!(
                        "{} another trusty-search daemon is already running (lock at {})",
                        "✗".red(),
                        p.display()
                    );
                    std::process::exit(1);
                }
                Err(e) => {
                    eprintln!("{} daemon failed: {e}", "✗".red());
                    std::process::exit(1);
                }
            }
        }

        Commands::Stop => {
            // The daemon writes its PID into the fs4 lockfile at startup
            // (see trusty-search-service/src/daemon.rs). Read the PID, send
            // SIGTERM, then poll for the port file to disappear as a signal
            // that shutdown completed cleanly.
            let lock_path = dirs::data_local_dir()
                .map(|d| d.join("trusty-search").join("daemon.lock"));
            let port_path = daemon_port_path();

            let pid = lock_path
                .as_ref()
                .and_then(|p| std::fs::read_to_string(p).ok())
                .and_then(|s| s.trim().parse::<u32>().ok());

            match pid {
                None => {
                    eprintln!("{} No daemon running (no PID file)", "✗".red());
                    std::process::exit(1);
                }
                Some(pid) => {
                    println!("{} Stopping daemon (PID {})…", "⟳".cyan(), pid);
                    let status = std::process::Command::new("kill")
                        .arg("-TERM")
                        .arg(pid.to_string())
                        .status();
                    match status {
                        Ok(s) if s.success() => {
                            // Poll up to 5s for the port file to disappear.
                            for _ in 0..50 {
                                std::thread::sleep(std::time::Duration::from_millis(100));
                                if port_path.as_ref().map(|p| !p.exists()).unwrap_or(true) {
                                    println!("{} Daemon stopped", "✓".green());
                                    return Ok(());
                                }
                            }
                            println!(
                                "{} Daemon may still be shutting down",
                                "⚠".yellow()
                            );
                        }
                        _ => {
                            eprintln!(
                                "{} Failed to send SIGTERM (process may already be gone)",
                                "✗".red()
                            );
                            std::process::exit(1);
                        }
                    }
                }
            }
        }

        Commands::Serve { http } => {
            let daemon_url = daemon_base_url();
            let server = trusty_search_mcp::McpServer::new(daemon_url.clone());
            if let Some(addr) = http {
                eprintln!(
                    "{} MCP HTTP/SSE on {} → daemon {}",
                    "◉".green(),
                    addr.cyan(),
                    daemon_url.dimmed()
                );
                let listener = tokio::net::TcpListener::bind(&addr).await?;
                let app = trusty_search_mcp::sse::router(server);
                axum::serve(listener, app).await?;
            } else {
                eprintln!(
                    "{} MCP stdio → daemon {}",
                    "◉".green(),
                    daemon_url.dimmed()
                );
                trusty_search_mcp::stdio::run(server).await?;
            }
        }

        Commands::Convert {
            target,
            dry_run,
            concurrency,
        } => {
            let base = daemon_base_url();

            match target {
                ConvertTarget::Project => {
                    let cwd = std::env::current_dir()?;
                    let config_path = find_mvs_config(&cwd).ok_or_else(|| {
                        anyhow::anyhow!(
                            "No .mcp-vector-search/config.json found in {} or any parent directory",
                            cwd.display()
                        )
                    })?;
                    let (root, name) = parse_mvs_config(&config_path)?;
                    if dry_run {
                        println!(
                            "{} Dry run — would convert '{}' ({})",
                            "·".dimmed(),
                            name.bold(),
                            root.display()
                        );
                    } else {
                        println!(
                            "{} Converting '{}' ({})…",
                            "⟳".cyan(),
                            name.bold(),
                            root.display()
                        );
                        let result = convert_one(root, name, &base, false).await;
                        match &result.status {
                            ConvertStatus::Queued => {
                                println!(
                                    "{} Queued for reindex — watch progress with: {}",
                                    "✓".green(),
                                    "trusty-search status".cyan()
                                );
                            }
                            ConvertStatus::AlreadyRegistered => {
                                println!(
                                    "{} Already registered — reindex queued",
                                    "↻".cyan()
                                );
                            }
                            ConvertStatus::Failed(msg) => {
                                eprintln!("{} Conversion failed: {}", "✗".red(), msg);
                                std::process::exit(1);
                            }
                            ConvertStatus::DryRun => unreachable!(),
                        }
                    }
                }

                ConvertTarget::All => {
                    let home_display = dirs::home_dir()
                        .map(|h| h.display().to_string())
                        .unwrap_or_else(|| "$HOME".to_string());
                    println!(
                        "🔍 Scanning for mcp-vector-search projects under {}…",
                        home_display
                    );
                    let configs = find_all_mvs_configs();
                    if configs.is_empty() {
                        println!("{} No mcp-vector-search projects found.", "·".dimmed());
                        return Ok(());
                    }

                    if dry_run {
                        println!(
                            "{} Dry run — would convert {} projects:\n",
                            "·".dimmed(),
                            configs.len()
                        );
                    } else {
                        println!(
                            "{} Found {} projects. Converting (max {} concurrent)…\n",
                            "·".dimmed(),
                            configs.len(),
                            concurrency
                        );
                    }

                    let total = configs.len();
                    let sem = std::sync::Arc::new(tokio::sync::Semaphore::new(concurrency.max(1)));
                    let base = std::sync::Arc::new(base);
                    let mut tasks = tokio::task::JoinSet::new();

                    for (i, config_path) in configs.into_iter().enumerate() {
                        let sem = sem.clone();
                        let base = base.clone();
                        tasks.spawn(async move {
                            // Acquire permit inside the task so JoinSet limits
                            // concurrency cleanly without us pre-allocating
                            // futures that all immediately try to fire.
                            let _permit = sem.acquire_owned().await.ok();
                            let parsed = parse_mvs_config(&config_path);
                            let result = match parsed {
                                Ok((root, name)) => {
                                    convert_one(root, name, &base, dry_run).await
                                }
                                Err(e) => ConvertResult {
                                    name: config_path.display().to_string(),
                                    path: config_path.clone(),
                                    status: ConvertStatus::Failed(format!("parse: {e}")),
                                },
                            };
                            (i + 1, result)
                        });
                    }

                    let mut queued = 0usize;
                    let mut already = 0usize;
                    let mut dry = 0usize;
                    let mut failed = 0usize;

                    // Collect-then-sort so output is deterministic instead of
                    // racy. For 69 projects this is trivially small.
                    let mut results: Vec<(usize, ConvertResult)> = Vec::with_capacity(total);
                    while let Some(joined) = tasks.join_next().await {
                        match joined {
                            Ok((i, r)) => results.push((i, r)),
                            Err(e) => eprintln!("{} task panicked: {e}", "✗".red()),
                        }
                    }
                    results.sort_by_key(|(i, _)| *i);

                    for (i, r) in &results {
                        print_convert_line(*i, total, r);
                        match r.status {
                            ConvertStatus::Queued => queued += 1,
                            ConvertStatus::AlreadyRegistered => already += 1,
                            ConvertStatus::DryRun => dry += 1,
                            ConvertStatus::Failed(_) => failed += 1,
                        }
                    }

                    println!();
                    if dry_run {
                        println!(
                            "{} Dry run complete: {} projects",
                            "·".dimmed(),
                            dry
                        );
                    } else {
                        println!(
                            "{} Summary: {} queued, {} already registered (reindexing), {} failed",
                            "✓".green(),
                            queued,
                            already,
                            failed
                        );
                        println!(
                            "  Reindexing in background. Run {} to see progress.",
                            "trusty-search list".cyan()
                        );
                    }
                }
            }
        }

        Commands::Ui { port } => {
            // Resolve port: explicit > port file > 7878.
            let port = port
                .or_else(|| {
                    daemon_port_path()
                        .and_then(|p| std::fs::read_to_string(p).ok())
                        .and_then(|s| s.trim().parse::<u16>().ok())
                })
                .unwrap_or(7878);
            let url = format!("http://127.0.0.1:{port}/ui");

            // Probe the daemon — if it's not running, surface a friendly
            // hint instead of a confusing browser error page.
            let probe_url = format!("http://127.0.0.1:{port}/health");
            let healthy = reqwest::get(&probe_url)
                .await
                .ok()
                .map(|r| r.status().is_success())
                .unwrap_or(false);
            if !healthy {
                eprintln!(
                    "{} Daemon not reachable at {}. Run {} first.",
                    "✗".red(),
                    format!("http://127.0.0.1:{port}").cyan(),
                    "trusty-search start".cyan(),
                );
                std::process::exit(1);
            }

            println!("{} Opening {} …", "◉".green(), url.cyan());
            if let Err(e) = open::that(&url) {
                eprintln!(
                    "{} could not launch browser ({e}). Open this URL manually: {}",
                    "⚠".yellow(),
                    url
                );
            }
        }

        Commands::Completions { shell } => {
            let mut cmd = Cli::command();
            let name = cmd.get_name().to_string();
            generate(shell, &mut cmd, name, &mut io::stdout());
        }
    }

    Ok(())
}
