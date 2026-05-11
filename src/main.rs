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
use eventsource_stream::Eventsource;
use futures_util::stream::StreamExt;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::io;
use std::time::Duration;

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
    arg_required_else_help = true
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

    /// Show daemon status and all index stats  [alias: st]
    ///
    /// Shows daemon liveness, version, and per-index chunk counts.
    /// `health` produces the same output (kept for backward compatibility).
    ///
    /// Examples:
    ///   trusty-search status
    ///   trusty-search status --json
    #[command(alias = "st", display_order = 3)]
    Status,

    /// Register and index a project in one step  [alias: idx]
    ///
    /// Registers the index with the daemon if needed, then runs a reindex
    /// with a live progress bar. Skips the reindex if the index already has
    /// chunks indexed (use --force to override).
    ///
    /// Examples:
    ///   trusty-search index                   # CWD, name from basename
    ///   trusty-search index ~/Projects/myapp
    ///   trusty-search index --force           # full reindex even if up-to-date
    #[command(alias = "idx", display_order = 4)]
    Index {
        /// Directory to register and index (default: CWD)
        path: Option<std::path::PathBuf>,

        /// Index name (default: directory basename)
        #[arg(short, long)]
        name: Option<String>,

        /// Force a full reindex even if the index already has chunks
        #[arg(short, long)]
        force: bool,
    },

    /// Register current directory as a named index (see `index`)
    ///
    /// Kept for backward compatibility. Prefer `trusty-search index`, which
    /// registers AND indexes in one step.
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

    /// Full reindex of current project (see `index --force`)
    ///
    /// Streams progress via SSE and renders a live progress bar. Prefer
    /// `trusty-search index --force` which also handles registration.
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

    /// Check daemon liveness (alias for `status`)
    ///
    /// Kept for backward compatibility. Both `health` and `status` produce
    /// the same rich output: daemon URL, version, and per-index chunk counts.
    ///
    /// Examples:
    ///   trusty-search health
    #[command(display_order = 12)]
    Health,

    // ── Service commands ──────────────────────────────────────────────────
    /// Start the HTTP daemon
    ///
    /// By default, runs the daemon inline in the current process (blocks until
    /// SIGTERM/SIGINT). The `--foreground` flag is accepted for clarity when the
    /// process is supervised by launchd, systemd, or Docker — these supervisors
    /// require the managed binary to remain in the foreground rather than forking.
    ///
    /// Examples:
    ///   trusty-search start
    ///   trusty-search start --port 7878
    ///   trusty-search start --foreground --port 7878   # launchd / systemd
    #[command(display_order = 20)]
    Start {
        /// Port to listen on (default: 7878, auto-selects next if busy)
        #[arg(long, default_value = "7878")]
        port: u16,

        /// Run in the foreground instead of forking a background daemon.
        ///
        /// Use this when the process is managed by launchd, systemd, or Docker.
        /// Note: the daemon already runs inline by default (no fork is performed),
        /// so this flag is currently a no-op accepted for forward-compatibility
        /// and to make the launchd/systemd contract explicit in ProgramArguments.
        #[arg(long, default_value_t = false)]
        foreground: bool,
    },

    /// Stop the running background daemon
    ///
    /// Sends SIGTERM to the daemon process and waits for clean shutdown.
    ///
    /// Examples:
    ///   trusty-search stop
    #[command(display_order = 21)]
    Stop,

    /// Start MCP server (HTTP by default; add --no-http for stdio-only)
    ///
    /// HTTP/SSE binds to 127.0.0.1:<port> (port 0 = OS-chosen free port) and
    /// the bound address is written to `~/.trusty-search/http_addr` so
    /// `trusty-search dashboard` and other clients can discover it. Stdio
    /// MCP is always served on the process's stdin/stdout for Claude Code.
    ///
    /// Use `--no-http` to skip the HTTP listener (Claude Code MCP hook path
    /// — Claude Code pipes JSON-RPC directly and doesn't need a panel).
    ///
    /// Examples:
    ///   trusty-search serve                       # MCP stdio + HTTP on :0
    ///   trusty-search serve --port 7878           # MCP stdio + HTTP on :7878
    ///   trusty-search serve --no-http             # MCP stdio only (Claude hook)
    ///   trusty-search serve --http 0.0.0.0:8080   # legacy: explicit bind addr
    #[command(display_order = 22)]
    Serve {
        /// Disable the HTTP listener (MCP stdio only).
        ///
        /// Use this when wired into a Claude Code MCP hook: Claude Code pipes
        /// JSON-RPC over stdin/stdout, so the HTTP admin panel is unnecessary
        /// and binding it just wastes a port.
        #[arg(long, default_value_t = false)]
        no_http: bool,

        /// Port for the HTTP/SSE MCP transport (default: 0 = OS picks).
        ///
        /// Ignored when `--no-http` is set.
        #[arg(long, default_value_t = 0)]
        port: u16,

        /// Legacy: explicit "host:port" bind address. When set, overrides
        /// `--port`. Kept for backward compatibility with older docs.
        #[arg(long)]
        http: Option<String>,
    },

    /// Manage the macOS launchd service (install/uninstall/status/logs)
    ///
    /// Installs a LaunchAgent plist at
    /// `~/Library/LaunchAgents/com.trusty.trusty-search.plist` that runs the
    /// daemon in the foreground under launchd supervision. Not supported on
    /// Linux / Windows — the subcommand exits 1 with a clear message.
    ///
    /// Examples:
    ///   trusty-search service install
    ///   trusty-search service status
    ///   trusty-search service logs
    ///   trusty-search service uninstall
    #[command(display_order = 24)]
    Service {
        #[command(subcommand)]
        action: ServiceAction,
    },

    /// Open the admin panel of the running daemon in the default browser
    ///
    /// Reads `~/.trusty-search/http_addr` to discover the daemon, then opens
    /// `http://<addr>/ui` in the default browser. Falls back to printing the
    /// URL if the browser fails to launch. Errors clearly if no daemon is
    /// running (no discovery file).
    ///
    /// Examples:
    ///   trusty-search dashboard
    #[command(display_order = 23)]
    Dashboard,

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

    /// Diagnose configuration, model cache, and index health
    ///
    /// Checks each component and reports ✓ / ✗ / ⚠ for each. Exit code 0
    /// when all checks pass or only warnings; exit code 1 when any error is
    /// found. Pass --fix to attempt automatic repair of fixable problems.
    ///
    /// Examples:
    ///   trusty-search doctor
    ///   trusty-search doctor --fix
    #[command(display_order = 28)]
    Doctor {
        /// Attempt to fix detected problems automatically
        #[arg(long)]
        fix: bool,
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

/// Subcommands for `trusty-search service` (macOS launchd integration).
///
/// Why: launchd is the canonical way to keep a long-lived foreground service
/// alive on macOS — it survives logout, restarts on crash, and integrates with
/// `launchctl` for diagnostics. Wrapping the plist mechanics in `service`
/// subcommands keeps users from having to hand-edit XML.
/// What: each variant maps to one `launchctl` operation.
/// Test: `cargo run -- service --help` lists the four actions; on Linux,
/// any action prints "not supported" and exits 1.
#[derive(Debug, Clone, Subcommand)]
enum ServiceAction {
    /// Install the LaunchAgent plist and load it
    Install,
    /// Unload the LaunchAgent and remove the plist
    Uninstall,
    /// Show launchd status for the agent
    Status,
    /// Tail the launchd stdout / stderr logs
    Logs,
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

/// Resolve the daemon's base URL.
///
/// Why: stdio MCP servers and CLI subcommands need to find the running
/// daemon without configuration. We check the canonical
/// `~/.trusty-search/http_addr` first (the new address-discovery contract,
/// aligned with trusty-memory), then fall back to the legacy port file
/// (`~/.local/share/trusty-search/daemon.port`) for backward compatibility,
/// and finally to `127.0.0.1:7878` if neither exists.
/// What: returns `http://{host}:{port}` (no trailing slash).
fn daemon_base_url() -> String {
    if let Some(addr) = read_http_addr_file() {
        return format!("http://{addr}");
    }
    let port = daemon_port_path()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|s| s.trim().parse::<u16>().ok())
        .unwrap_or(7878);
    format!("http://127.0.0.1:{port}")
}

/// Read the canonical address-discovery file. Returns `Some("host:port")`
/// when the daemon has written it; `None` otherwise.
fn read_http_addr_file() -> Option<String> {
    let path = http_addr_path()?;
    let raw = std::fs::read_to_string(&path).ok()?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// Path to `~/.trusty-search/http_addr` — the canonical address-discovery
/// file. Mirrors `trusty_search_service::daemon::http_addr_path` so the CLI
/// doesn't need to depend on the service crate for path resolution.
fn http_addr_path() -> Option<std::path::PathBuf> {
    dirs::home_dir().map(|h| h.join(".trusty-search").join("http_addr"))
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
    let client = trusty_common::server::daemon_http_client()?;

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
        println!("{} indexed {} files ({} errors)", "✓".green(), ok, err);
        Ok(())
    } else {
        index_single_file(&client, &base, index_id, path).await?;
        println!("{} [{}] {}", "→".cyan(), index_id, path.display());
        Ok(())
    }
}

/// Format a millisecond elapsed time as `Xm Ys` (or `Ys` if < 1 minute).
fn fmt_elapsed(ms: u64) -> String {
    let secs = ms / 1000;
    if secs >= 60 {
        format!("{}m {:02}s", secs / 60, secs % 60)
    } else if secs > 0 {
        format!("{}s", secs)
    } else {
        format!("{}ms", ms)
    }
}

/// Format an elapsed seconds count as `Xm Ys` (or `Ys`).
fn fmt_secs(secs: u64) -> String {
    if secs >= 60 {
        format!("{}m {:02}s", secs / 60, secs % 60)
    } else {
        format!("{}s", secs)
    }
}

/// Multi-line live progress display for a reindex.
///
/// Why: a single-line `ProgressBar` can't simultaneously show file progress,
/// chunk count, skipped count, speed, and elapsed/ETA. `MultiProgress` stacks
/// three lines (header / files bar / stats) that update independently.
///
/// Layout:
///   ⟳ Indexing <index>
///     [████████░░░░] 7,234/14,445 files (50%) — ETA 50s
///     Chunks: 58,402  Skipped: 12  Speed: 142 files/s  Elapsed: 50s  ETA: ~50s
struct ReindexUi {
    /// Held to keep the MultiProgress draw target alive for the bars' lifetime.
    #[allow(dead_code)]
    multi: MultiProgress,
    header: ProgressBar,
    files: ProgressBar,
    stats: ProgressBar,
}

impl ReindexUi {
    fn new(index_id: &str) -> Self {
        let multi = MultiProgress::new();

        let header = multi.add(ProgressBar::new(1));
        if let Ok(s) = ProgressStyle::with_template("{spinner:.cyan} {msg}") {
            header.set_style(s);
        }
        header.set_message(format!("Indexing {}", index_id.bold()));
        header.enable_steady_tick(Duration::from_millis(120));

        let files = multi.add(ProgressBar::new(1));
        if let Ok(s) = ProgressStyle::with_template(
            "  [{bar:40.cyan/blue}] {pos}/{len} files ({percent}%) — ETA {eta}",
        ) {
            files.set_style(s.progress_chars("█░ "));
        }

        let stats = multi.add(ProgressBar::new(1));
        if let Ok(s) = ProgressStyle::with_template("  {msg}") {
            stats.set_style(s);
        }
        stats.set_message("Waiting for daemon…".to_string());

        Self {
            multi,
            header,
            files,
            stats,
        }
    }

    fn set_total(&self, total: u64) {
        self.files.set_length(total.max(1));
    }

    fn set_position(&self, indexed: u64) {
        self.files.set_position(indexed);
    }

    fn update_stats(&self, indexed: u64, total_chunks: u64, skipped: u64, elapsed_secs: u64) {
        let files_per_sec = if elapsed_secs > 0 {
            indexed / elapsed_secs
        } else {
            0
        };
        self.stats.set_message(format!(
            "Chunks: {chunks}  Skipped: {skipped}  Speed: {fps} files/s  Elapsed: {elapsed}",
            chunks = format_with_commas(total_chunks),
            skipped = format_with_commas(skipped),
            fps = files_per_sec,
            elapsed = fmt_secs(elapsed_secs),
        ));
    }

    fn finish(self, final_msg: String) {
        self.files.finish_and_clear();
        self.stats.finish_and_clear();
        self.header.finish_with_message(final_msg);
    }

    fn abandon(self, final_msg: String) {
        self.files.abandon();
        self.stats.abandon();
        self.header.abandon_with_message(final_msg);
    }
}

/// Options controlling reindex CLI behaviour.
#[derive(Debug, Clone, Copy, Default)]
struct ReindexOptions {
    /// After the reindex completes, fetch `/status` and issue a sanity-check
    /// search to verify the index is healthy. Enabled by `--force` to give
    /// the user a blue-green-style safety net.
    ///
    /// Note: the daemon's reindex is NOT atomic blue-green — it mutates the
    /// in-memory index in place via a write lock per batch (see
    /// `crates/trusty-search-service/src/reindex.rs::spawn_reindex` —
    /// `index_files_batch_no_rebuild` adds chunks per-batch). If verify fails
    /// after a `--force`, the index is already in its new (possibly broken)
    /// state. We surface that fact loudly so the user can manually re-run.
    verify_after: bool,
    /// Chunk count snapshot taken before the reindex started, used to print
    /// "(was N)" in the final verify message.
    prior_chunk_count: Option<u64>,
    /// Forwarded to the daemon as `"force": true` in the reindex kickoff body.
    /// Set by `index --force` so the daemon clears its content-hash cache and
    /// re-embeds every file (otherwise unchanged files would be skipped on a
    /// warm daemon and `--force` would have no effect).
    force: bool,
}

/// Outcome of a reindex run, captured for the post-verify step and the final
/// summary line. `indexed` includes skipped files (the daemon emits one
/// `indexed++` per file regardless of whether it was hashed-skip or re-embedded).
#[derive(Debug, Default, Clone, Copy)]
struct ReindexOutcome {
    indexed: u64,
    total_chunks: u64,
    skipped: u64,
    errors: u64,
    elapsed_ms: u64,
    completed: bool,
}

/// Plain reindex (no post-verify). Used by the non-force `index` command, the
/// bare `reindex` command, and the doctor auto-repair path. The daemon's
/// hash-skip optimization (see `reindex.rs::hash_content`) means unchanged
/// files are cheap, so calling this even when nothing changed is fine.
async fn run_reindex(index_id: &str, root_path: &std::path::Path) -> Result<()> {
    run_reindex_with(index_id, root_path, ReindexOptions::default())
        .await
        .map(|_| ())
}

/// `index --force` reindex: snapshot the prior chunk count, kick off a full
/// reindex, and run a post-reindex health check. Exits 1 if the new index
/// looks unhealthy (no chunks or empty sanity query).
async fn run_reindex_force(index_id: &str, root_path: &std::path::Path) -> Result<()> {
    let prior = fetch_chunk_count(index_id).await;
    let opts = ReindexOptions {
        verify_after: true,
        prior_chunk_count: prior,
        force: true,
    };
    run_reindex_with(index_id, root_path, opts)
        .await
        .map(|_| ())
}

/// Drive a reindex: POST /reindex, then connect to the SSE stream and render
/// progress with an indicatif `MultiProgress` layout (header + files bar +
/// stats line). A wall-clock ticker keeps the stats line moving even when
/// SSE events are sparse (e.g. the embedder is mid-batch).
async fn run_reindex_with(
    index_id: &str,
    root_path: &std::path::Path,
    opts: ReindexOptions,
) -> Result<ReindexOutcome> {
    let base = daemon_base_url();
    let client = trusty_common::server::daemon_http_client()?;

    let kickoff_url = format!("{}/indexes/{}/reindex", base, index_id);
    let kickoff_body = serde_json::json!({
        "root_path": root_path,
        "force": opts.force,
    });
    let kickoff = client
        .post(&kickoff_url)
        .json(&kickoff_body)
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("could not reach daemon at {base}: {e}"))?;

    if kickoff.status() == reqwest::StatusCode::NOT_FOUND {
        eprintln!(
            "{} index '{}' is not registered on the daemon — run `trusty-search index` first",
            "✗".red(),
            index_id
        );
        std::process::exit(1);
    }
    if !kickoff.status().is_success() {
        anyhow::bail!("daemon returned {} for reindex kickoff", kickoff.status());
    }

    let kickoff_body: serde_json::Value = kickoff
        .json()
        .await
        .unwrap_or_else(|_| serde_json::json!({}));
    let stream_path = kickoff_body
        .get("stream_url")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("/indexes/{}/reindex/stream", index_id));
    let stream_url = format!("{}{}", base, stream_path);

    // SSE streams must NOT use the short request timeout from
    // `daemon_http_client()` (currently 5s) — a large repo reindex can run for
    // minutes. We build a dedicated client with only a connect timeout so the
    // byte stream stays open until the daemon emits the `complete` event.
    let sse_client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(5))
        .timeout(Duration::MAX)
        .build()
        .map_err(|e| anyhow::anyhow!("could not build SSE client: {e}"))?;
    let resp = sse_client
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
    // MultiProgress UI: header + files bar + stats line. Built eagerly so
    // the user sees something during the 1–2 second daemon warmup before the
    // first SSE event arrives.
    let ui = ReindexUi::new(index_id);

    // Atomics shared with the wall-clock ticker. The ticker refreshes the
    // stats line every second so the user sees movement even when the SSE
    // stream is idle (e.g. mid-batch embedding of 256 chunks).
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::Arc as StdArc;
    let started = std::time::Instant::now();
    let indexed_now = StdArc::new(AtomicU64::new(0));
    let chunks_now = StdArc::new(AtomicU64::new(0));
    let skipped_now = StdArc::new(AtomicU64::new(0));
    let tick_done = StdArc::new(AtomicBool::new(false));

    let ticker = {
        let indexed_now = indexed_now.clone();
        let chunks_now = chunks_now.clone();
        let skipped_now = skipped_now.clone();
        let tick_done = tick_done.clone();
        let stats_bar = ui.stats.clone();
        let files_bar = ui.files.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(1));
            interval.tick().await; // discard immediate tick
            loop {
                interval.tick().await;
                if tick_done.load(Ordering::Acquire) {
                    break;
                }
                let elapsed = started.elapsed().as_secs();
                let indexed = indexed_now.load(Ordering::Acquire);
                let chunks = chunks_now.load(Ordering::Acquire);
                let skipped = skipped_now.load(Ordering::Acquire);
                let fps = if elapsed > 0 { indexed / elapsed } else { 0 };
                let total = files_bar.length().unwrap_or(0);
                let eta = if fps > 0 && total > indexed {
                    fmt_secs((total - indexed) / fps)
                } else {
                    "?".to_string()
                };
                stats_bar.set_message(format!(
                    "Chunks: {chunks}  Skipped: {skipped}  Speed: {fps} files/s  Elapsed: {elapsed}s  ETA: ~{eta}",
                    chunks = format_with_commas(chunks),
                    skipped = format_with_commas(skipped),
                    fps = fps,
                    elapsed = elapsed,
                    eta = eta,
                ));
            }
        })
    };

    let mut outcome = ReindexOutcome::default();
    let mut done = false;

    // `eventsource-stream` handles SSE framing. The daemon emits these event
    // types (see `crates/trusty-search-service/src/reindex.rs::spawn_reindex`):
    //   - start:    total_files, index_id, root_path
    //   - batch:    batch_files, batch_chunks, indexed, total_files, elapsed_ms
    //   - skip:     file, indexed, total_files (hash matched OR minified)
    //   - error:    message, file (or files for a batch failure)
    //   - complete: indexed, total_chunks, skipped, errors, elapsed_ms
    let byte_stream = resp.bytes_stream();
    let stream = byte_stream.eventsource();
    tokio::pin!(stream);
    while !done {
        let event = match stream.next().await {
            Some(Ok(e)) => e,
            Some(Err(e)) => {
                ui.stats
                    .println(format!("{} stream read error: {e}", "⚠".yellow()));
                break;
            }
            None => break,
        };

        let evt: serde_json::Value = match serde_json::from_str(event.data.trim()) {
            Ok(v) => v,
            Err(_) => continue,
        };
        match evt.get("event").and_then(|v| v.as_str()) {
            Some("start") => {
                let total = evt.get("total_files").and_then(|v| v.as_u64()).unwrap_or(0);
                ui.set_total(total);
            }
            Some("batch") => {
                let indexed = evt.get("indexed").and_then(|v| v.as_u64()).unwrap_or(0);
                let batch_chunks = evt
                    .get("batch_chunks")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                let total = evt.get("total_files").and_then(|v| v.as_u64()).unwrap_or(0);
                if total > 0 && ui.files.length() != Some(total.max(1)) {
                    ui.set_total(total);
                }
                indexed_now.store(indexed, Ordering::Release);
                let new_chunks =
                    chunks_now.fetch_add(batch_chunks, Ordering::AcqRel) + batch_chunks;
                ui.set_position(indexed);
                ui.update_stats(
                    indexed,
                    new_chunks,
                    skipped_now.load(Ordering::Acquire),
                    started.elapsed().as_secs(),
                );
            }
            Some("skip") => {
                let indexed = evt.get("indexed").and_then(|v| v.as_u64()).unwrap_or(0);
                indexed_now.store(indexed, Ordering::Release);
                let skipped = skipped_now.fetch_add(1, Ordering::AcqRel) + 1;
                ui.set_position(indexed);
                ui.update_stats(
                    indexed,
                    chunks_now.load(Ordering::Acquire),
                    skipped,
                    started.elapsed().as_secs(),
                );
            }
            Some("complete") => {
                outcome.indexed = evt.get("indexed").and_then(|v| v.as_u64()).unwrap_or(0);
                outcome.total_chunks = evt
                    .get("total_chunks")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                outcome.skipped = evt
                    .get("skipped")
                    .and_then(|v| v.as_u64())
                    .unwrap_or_else(|| skipped_now.load(Ordering::Acquire));
                outcome.errors = evt.get("errors").and_then(|v| v.as_u64()).unwrap_or(0);
                outcome.elapsed_ms = evt.get("elapsed_ms").and_then(|v| v.as_u64()).unwrap_or(0);
                outcome.completed = true;
                ui.set_position(outcome.indexed);
                done = true;
            }
            Some("error") => {
                let msg = evt
                    .get("message")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");
                let file = evt.get("file").and_then(|v| v.as_str()).unwrap_or("");
                ui.stats
                    .println(format!("{}  {}: {}", "⚠".yellow(), file, msg));
            }
            _ => {}
        }
    }

    // Stop the ticker before finishing the UI so it doesn't overwrite the
    // final message during the brief window between finish() and shutdown.
    tick_done.store(true, Ordering::Release);
    let _ = ticker.await;

    if !outcome.completed {
        ui.abandon(format!(
            "{} Reindex stream ended without completion event",
            "⚠".yellow()
        ));
        anyhow::bail!("reindex did not complete");
    }

    // Final headline. We distinguish three cases:
    //   1. errors > 0          → show error count + unchanged count
    //   2. nothing changed     → "is up to date" message (Improvement 3)
    //   3. some files changed  → "Indexed N changed files" with unchanged tally
    let elapsed = fmt_elapsed(outcome.elapsed_ms);
    let changed = outcome.indexed.saturating_sub(outcome.skipped);
    let final_msg = if outcome.errors > 0 {
        format!(
            "{} Indexed {} files → {} chunks  [took {}, {} errors, {} unchanged]",
            "✓".green(),
            format_with_commas(changed),
            format_with_commas(outcome.total_chunks),
            elapsed,
            outcome.errors,
            format_with_commas(outcome.skipped),
        )
    } else if changed == 0 && outcome.indexed > 0 {
        format!(
            "{} '{}' is up to date ({} chunks, {} files — no changes detected)  [took {}]",
            "✓".green(),
            index_id,
            format_with_commas(outcome.total_chunks),
            format_with_commas(outcome.indexed),
            elapsed,
        )
    } else {
        format!(
            "{} Indexed {} changed file{} → {} chunks  [took {}, {} unchanged]",
            "✓".green(),
            format_with_commas(changed),
            if changed == 1 { "" } else { "s" },
            format_with_commas(outcome.total_chunks),
            elapsed,
            format_with_commas(outcome.skipped),
        )
    };
    ui.finish(final_msg);

    // ── Post-reindex health check (blue-green safety net) ─────────────────
    if opts.verify_after {
        verify_reindex_health(&client, &base, index_id, &outcome, opts.prior_chunk_count).await?;
    }

    Ok(outcome)
}

/// After a `--force` reindex, fetch the new chunk count and run a sanity
/// query. Exits 1 if either looks wrong.
///
/// Why: the daemon's reindex mutates the in-memory `CodeIndexer` in place
/// (no shadow slot — see `reindex.rs::spawn_reindex`, which writes each batch
/// directly into the live indexer via `index_files_batch_no_rebuild`). If the
/// rebuild produces a broken index, the only signal the user has is "search
/// returns nothing" hours later. This check surfaces that immediately.
async fn verify_reindex_health(
    client: &reqwest::Client,
    base: &str,
    index_id: &str,
    outcome: &ReindexOutcome,
    prior: Option<u64>,
) -> Result<()> {
    // 1) Chunk count via /status.
    let status_url = format!("{}/indexes/{}/status", base, index_id);
    let new_chunks = match client.get(&status_url).send().await {
        Ok(r) if r.status().is_success() => r
            .json::<serde_json::Value>()
            .await
            .ok()
            .and_then(|v| v.get("chunk_count").and_then(|n| n.as_u64()))
            .unwrap_or(0),
        _ => 0,
    };

    // 2) Sanity query: pick something that hits virtually any source tree
    //    (`fn` matches Rust; `function` JS/TS; `def` Python; etc.). One hit
    //    in any single probe is enough to consider the index queryable.
    let search_url = format!("{}/indexes/{}/search", base, index_id);
    let probes = ["fn", "function", "def", "class", "the"];
    let mut got_hit = false;
    for probe in probes {
        let body = serde_json::json!({ "text": probe, "top_k": 1 });
        if let Ok(resp) = client.post(&search_url).json(&body).send().await {
            if resp.status().is_success() {
                if let Ok(json) = resp.json::<serde_json::Value>().await {
                    let n = json
                        .get("results")
                        .and_then(|r| r.as_array())
                        .map(|a| a.len())
                        .unwrap_or(0);
                    if n > 0 {
                        got_hit = true;
                        break;
                    }
                }
            }
        }
    }

    let healthy = new_chunks > 0 && got_hit && outcome.errors == 0;
    let was = prior
        .map(|p| format!(" (was {})", format_with_commas(p)))
        .unwrap_or_default();
    if healthy {
        println!(
            "{} Reindex complete: {} chunks{}",
            "✓".green(),
            format_with_commas(new_chunks),
            was
        );
        Ok(())
    } else {
        eprintln!(
            "{} Reindex produced unhealthy index: {} chunks{}, sanity query {} — old index NOT preserved (daemon reindex is in-place; see crates/trusty-search-service/src/reindex.rs)",
            "✗".red(),
            format_with_commas(new_chunks),
            was,
            if got_hit { "ok" } else { "returned 0 results" }
        );
        std::process::exit(1);
    }
}

/// Register an index with the daemon (idempotent).
///
/// Why: factored out of `Init` and `Index` because both flows need the same
/// "POST /indexes, parse `created`" dance.
/// What: returns `Ok((created, daemon_reachable))`. `daemon_reachable=false`
/// surfaces network failures distinctly from "registered but already existed".
async fn register_index_with_daemon(
    index_name: &str,
    project_path: &std::path::Path,
) -> Result<(bool, bool)> {
    let base = daemon_base_url();
    let client = trusty_common::server::daemon_http_client()?;
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
            Ok((created, true))
        }
        Ok(resp) => {
            anyhow::bail!("daemon returned {} for POST /indexes", resp.status());
        }
        Err(_) => Ok((false, false)),
    }
}

/// Fetch chunk count for an index via /status. Returns `None` if the daemon
/// is unreachable or the index isn't registered.
async fn fetch_chunk_count(index_id: &str) -> Option<u64> {
    let base = daemon_base_url();
    let url = format!("{}/indexes/{}/status", base, index_id);
    let client = trusty_common::server::daemon_http_client().ok()?;
    let resp = client.get(&url).send().await.ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let body: serde_json::Value = resp.json().await.ok()?;
    body.get("chunk_count").and_then(|v| v.as_u64())
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

    let client = match trusty_common::server::daemon_http_client() {
        Ok(c) => c,
        Err(e) => {
            return ConvertResult {
                name: index_name,
                path: project_root,
                status: ConvertStatus::Failed(format!("failed to build HTTP client: {e}")),
            };
        }
    };

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
            let body: serde_json::Value =
                resp.json().await.unwrap_or_else(|_| serde_json::json!({}));
            !body
                .get("created")
                .and_then(|v| v.as_bool())
                .unwrap_or(true)
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

// ── Status / Health helper ────────────────────────────────────────────────

/// Format a u64 with locale-style thousands separators (e.g. 115585 → "115,585").
///
/// Why: chunk counts for large repos (100k+) are hard to read without commas.
/// What: groups digits in threes from the right, separated by ",".
/// Test: 0 → "0", 1000 → "1,000", 115585 → "115,585".
fn format_with_commas(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::with_capacity(s.len() + s.len() / 3);
    for (i, ch) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(ch);
    }
    result.chars().rev().collect()
}

/// Shared handler for `status` and `health` — both show the same rich output.
///
/// Why: removing duplication between the two CLI aliases; a single source of
/// truth for the daemon+indexes display logic.
/// What: queries `/health` + `/indexes` + per-index `/status`, then prints
/// or emits JSON. Exits 1 when the daemon is not reachable.
async fn run_status(json: bool) -> Result<()> {
    let base = daemon_base_url();
    let client = trusty_common::server::daemon_http_client()?;

    let health = client.get(format!("{}/health", base)).send().await;
    let health_body: serde_json::Value = match health {
        Ok(r) if r.status().is_success() => {
            r.json().await.unwrap_or_else(|_| serde_json::json!({}))
        }
        _ => {
            if json {
                println!(r#"{{"daemon":"not_running"}}"#);
            } else {
                eprintln!(
                    "{} Daemon not running  (start with `trusty-search start`)",
                    "✗".red()
                );
            }
            std::process::exit(1);
        }
    };

    let list = client.get(format!("{}/indexes", base)).send().await;
    let list_body: serde_json::Value = match list {
        Ok(r) if r.status().is_success() => {
            r.json().await.unwrap_or_else(|_| serde_json::json!({}))
        }
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

    if json {
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
        let version = health_body
            .get("version")
            .and_then(|v| v.as_str())
            .unwrap_or("?");
        println!(
            "{} Daemon running  {}  v{}",
            "✓".green(),
            base.cyan(),
            version
        );
        if per_index.is_empty() {
            println!("{}", "Indexes:".bold());
            println!("  {}", "(none)".dimmed());
        } else {
            println!("{}", "Indexes:".bold());
            for (name, body) in &per_index {
                let chunks = body
                    .get("chunk_count")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                let root = body.get("root_path").and_then(|v| v.as_str()).unwrap_or("");
                let chunks_fmt = format_with_commas(chunks);
                if root.is_empty() {
                    println!("  {:<16} {:>12} chunks", name.bold(), chunks_fmt,);
                } else {
                    println!(
                        "  {:<16} {:>12} chunks  {}",
                        name.bold(),
                        chunks_fmt,
                        root.dimmed()
                    );
                }
            }
        }
    }
    Ok(())
}

// ── Doctor ────────────────────────────────────────────────────────────────

/// Return the directory where fastembed caches ONNX models.
///
/// Why: fastembed uses `FASTEMBED_CACHE_DIR` env var when set, otherwise
/// `.fastembed_cache` relative to the process CWD. For the daemon the CWD
/// is wherever the user launched it — so we check the env var first, then
/// fall back to the cache path next to the trusty-search data dir.
fn fastembed_cache_dir() -> std::path::PathBuf {
    if let Ok(s) = std::env::var("FASTEMBED_CACHE_DIR") {
        return std::path::PathBuf::from(s);
    }
    // fastembed's default is ".fastembed_cache" in the process CWD at the
    // time TextEmbedding::try_new() is called. For doctor we look in the
    // most likely candidate: the data dir used by the daemon.
    if let Some(d) = dirs::data_local_dir() {
        let candidate = d.join("trusty-search").join(".fastembed_cache");
        if candidate.exists() {
            return candidate;
        }
    }
    // Also check next to the binary (common dev setup).
    if let Ok(exe) = std::env::current_exe() {
        let candidate = exe
            .parent()
            .unwrap_or(std::path::Path::new("."))
            .join(".fastembed_cache");
        if candidate.exists() {
            return candidate;
        }
    }
    // Final fallback: relative to CWD (matches fastembed default).
    std::path::PathBuf::from(".fastembed_cache")
}

/// Compute total byte size of a directory tree (best-effort; ignores errors).
fn dir_size_bytes(path: &std::path::Path) -> u64 {
    let mut total = 0u64;
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_file() {
                total += std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
            } else if p.is_dir() {
                total += dir_size_bytes(&p);
            }
        }
    }
    total
}

/// Format bytes as a human-readable string (MB / KB / B).
fn fmt_bytes(bytes: u64) -> String {
    if bytes >= 1_000_000 {
        format!("{:.0}MB", bytes as f64 / 1_000_000.0)
    } else if bytes >= 1_000 {
        format!("{:.0}KB", bytes as f64 / 1_000.0)
    } else {
        format!("{}B", bytes)
    }
}

/// Check whether a TCP port is open (non-blocking connect with 500 ms timeout).
async fn port_reachable(host: &str, port: u16) -> bool {
    let addr = format!("{}:{}", host, port);
    tokio::time::timeout(
        std::time::Duration::from_millis(500),
        tokio::net::TcpStream::connect(&addr),
    )
    .await
    .ok()
    .and_then(|r| r.ok())
    .is_some()
}

/// Read the daemon port from the port file (or return 7878).
fn read_daemon_port() -> u16 {
    daemon_port_path()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|s| s.trim().parse::<u16>().ok())
        .unwrap_or(7878)
}

/// Outcome of a single doctor check.
#[derive(Debug, Clone, PartialEq)]
enum CheckResult {
    /// Check passed.
    Ok(String),
    /// Non-fatal issue; doctor continues.
    Warn(String),
    /// Fatal issue; counted as an error.
    Error(String),
}

impl CheckResult {
    fn print(&self) {
        match self {
            CheckResult::Ok(msg) => println!("{} {}", "✓".green(), msg),
            CheckResult::Warn(msg) => println!("{} {}", "⚠".yellow(), msg),
            CheckResult::Error(msg) => println!("{} {}", "✗".red(), msg),
        }
    }

    fn is_error(&self) -> bool {
        matches!(self, CheckResult::Error(_))
    }

    fn is_warn(&self) -> bool {
        matches!(self, CheckResult::Warn(_))
    }
}

/// Represents an index that has no chunks (fixable via reindex).
#[derive(Debug)]
struct EmptyIndex {
    name: String,
    root_path: String,
}

/// Run the full doctor diagnostic suite and return (checks, empty_indexes).
async fn run_doctor_checks() -> (Vec<CheckResult>, Vec<EmptyIndex>) {
    let mut checks: Vec<CheckResult> = Vec::new();
    let mut empty_indexes: Vec<EmptyIndex> = Vec::new();

    // ── 1. Daemon liveness ────────────────────────────────────────────────
    let port = read_daemon_port();
    let base = daemon_base_url();
    let client = match trusty_common::server::daemon_http_client() {
        Ok(c) => c,
        Err(e) => {
            checks.push(CheckResult::Error(format!(
                "failed to build HTTP client: {e}"
            )));
            return (checks, empty_indexes);
        }
    };

    let health_result = client.get(format!("{}/health", base)).send().await;

    let (daemon_running, daemon_version) = match health_result {
        Ok(r) if r.status().is_success() => {
            let body: serde_json::Value = r.json().await.unwrap_or_else(|_| serde_json::json!({}));
            let ver = body
                .get("version")
                .and_then(|v| v.as_str())
                .unwrap_or("?")
                .to_string();
            (true, ver)
        }
        _ => (false, String::new()),
    };

    if daemon_running {
        checks.push(CheckResult::Ok(format!(
            "Daemon running at {} (v{})",
            base, daemon_version
        )));
    } else {
        checks.push(CheckResult::Error(
            "Daemon not running — run `trusty-search start`".to_string(),
        ));
    }

    // ── 2. Model cache ────────────────────────────────────────────────────
    let model_cache = fastembed_cache_dir();
    let model_name = "all-MiniLM-L6-v2";
    let model_subdir = model_cache.join("models--Qdrant--all-MiniLM-L6-v2-onnx");
    if model_subdir.exists() {
        let size = dir_size_bytes(&model_cache);
        checks.push(CheckResult::Ok(format!(
            "Model cache: {} ({}, {})",
            model_cache.display(),
            fmt_bytes(size),
            model_name
        )));
    } else if model_cache.exists() {
        checks.push(CheckResult::Warn(format!(
            "Model cache directory exists ({}) but {} not found — will download on first start",
            model_cache.display(),
            model_name
        )));
    } else {
        checks.push(CheckResult::Warn(
            "Model not cached — will download on first `trusty-search start`".to_string(),
        ));
    }

    // ── 3. Data directory ─────────────────────────────────────────────────
    let data_dir = dirs::data_local_dir()
        .map(|d| d.join("trusty-search"))
        .unwrap_or_else(|| std::path::PathBuf::from("~/.local/share/trusty-search"));

    if data_dir.exists() {
        // Probe writability with a temp file.
        let probe = data_dir.join(".write_probe");
        let writable = std::fs::write(&probe, b"").is_ok();
        let _ = std::fs::remove_file(&probe);
        if writable {
            checks.push(CheckResult::Ok(format!(
                "Data directory: {} (writable)",
                data_dir.display()
            )));
        } else {
            checks.push(CheckResult::Error(format!(
                "Data directory {} is not writable",
                data_dir.display()
            )));
        }
    } else {
        checks.push(CheckResult::Warn(format!(
            "Data directory {} does not exist (will be created on first start)",
            data_dir.display()
        )));
    }

    // ── 4. Lock file ──────────────────────────────────────────────────────
    let lock_path = data_dir.join("daemon.lock");
    if lock_path.exists() {
        // Read the PID stored in the lockfile and check if that process is alive.
        let pid_opt = std::fs::read_to_string(&lock_path)
            .ok()
            .and_then(|s| s.trim().parse::<u32>().ok());
        match pid_opt {
            Some(pid) => {
                // POSIX: kill(pid, 0) — check existence without sending a signal.
                let alive = unsafe { libc::kill(pid as libc::pid_t, 0) } == 0;
                if alive {
                    if daemon_running {
                        checks.push(CheckResult::Ok(format!(
                            "Lock file: healthy (PID {} is running)",
                            pid
                        )));
                    } else {
                        checks.push(CheckResult::Warn(format!(
                            "Lock file contains PID {} which is alive but /health failed",
                            pid
                        )));
                    }
                } else {
                    checks.push(CheckResult::Warn(format!(
                        "Stale lock file: PID {} is not running ({})",
                        pid,
                        lock_path.display()
                    )));
                }
            }
            None => {
                checks.push(CheckResult::Warn(format!(
                    "Lock file exists but contains no valid PID ({})",
                    lock_path.display()
                )));
            }
        }
    } else {
        checks.push(CheckResult::Ok("Lock file: healthy (no stale lock)".into()));
    }

    // ── 5. Indexes ────────────────────────────────────────────────────────
    if daemon_running {
        let list = client.get(format!("{}/indexes", base)).send().await;
        let list_body: serde_json::Value = match list {
            Ok(r) if r.status().is_success() => {
                r.json().await.unwrap_or_else(|_| serde_json::json!({}))
            }
            _ => serde_json::json!({"indexes": []}),
        };
        let empty_arr: Vec<serde_json::Value> = Vec::new();
        let names: Vec<String> = list_body
            .get("indexes")
            .and_then(|v| v.as_array())
            .unwrap_or(&empty_arr)
            .iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect();

        if names.is_empty() {
            checks.push(CheckResult::Warn(
                "No indexes registered — run `trusty-search index` to add a project".into(),
            ));
        } else {
            // Fetch status for each index concurrently.
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

            let zero_count = per_index
                .iter()
                .filter(|(_, b)| b.get("chunk_count").and_then(|v| v.as_u64()).unwrap_or(0) == 0)
                .count();

            if zero_count == 0 {
                checks.push(CheckResult::Ok(format!(
                    "{} index{} registered, all have chunks",
                    per_index.len(),
                    if per_index.len() == 1 { "" } else { "es" }
                )));
            } else {
                checks.push(CheckResult::Warn(format!(
                    "{} index{} registered, {} {} no chunks yet:",
                    per_index.len(),
                    if per_index.len() == 1 { "" } else { "es" },
                    zero_count,
                    if zero_count == 1 { "has" } else { "have" }
                )));
            }

            // Print per-index breakdown (indented).
            for (name, body) in &per_index {
                let chunks = body
                    .get("chunk_count")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                let root = body
                    .get("root_path")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let chunks_fmt = format_with_commas(chunks);
                if chunks == 0 {
                    println!(
                        "    {} {:<16} {:>12} chunks  {} — run `trusty-search index` to populate",
                        "⚠".yellow(),
                        name.bold(),
                        chunks_fmt,
                        root.dimmed()
                    );
                    empty_indexes.push(EmptyIndex {
                        name: name.clone(),
                        root_path: root,
                    });
                } else {
                    println!(
                        "    {} {:<16} {:>12} chunks  {}",
                        "✓".green(),
                        name.bold(),
                        chunks_fmt,
                        root.dimmed()
                    );
                }
            }
        }
    } else {
        checks.push(CheckResult::Warn(
            "Indexes: skipped (daemon not running)".into(),
        ));
    }

    // ── 6. Port reachability ──────────────────────────────────────────────
    if port_reachable("127.0.0.1", port).await {
        checks.push(CheckResult::Ok(format!("Port {} is reachable", port)));
    } else {
        checks.push(CheckResult::Error(format!(
            "Port {} is not reachable",
            port
        )));
    }

    (checks, empty_indexes)
}

/// Remove a stale lock file and report the outcome.
fn fix_stale_lock(data_dir: &std::path::Path) {
    let lock_path = data_dir.join("daemon.lock");
    if lock_path.exists() {
        let pid_opt = std::fs::read_to_string(&lock_path)
            .ok()
            .and_then(|s| s.trim().parse::<u32>().ok());
        let stale = pid_opt
            .map(|pid| unsafe { libc::kill(pid as libc::pid_t, 0) } != 0)
            .unwrap_or(true);
        if stale {
            match std::fs::remove_file(&lock_path) {
                Ok(()) => println!(
                    "  {} Removed stale lock file {}",
                    "✓".green(),
                    lock_path.display()
                ),
                Err(e) => println!(
                    "  {} Could not remove lock file {}: {e}",
                    "✗".red(),
                    lock_path.display()
                ),
            }
        } else {
            println!(
                "  {} Lock file is held by a live process — not removing",
                "⚠".yellow()
            );
        }
    }
}

// ── Dashboard ─────────────────────────────────────────────────────────────

/// Open the admin panel of the running daemon in the default browser.
///
/// Why: provides a one-command path from "is the daemon up?" to "show me the
/// UI" without the user having to memorize ports or paths.
/// What: reads `~/.trusty-search/http_addr`; if missing, errors with a clear
/// hint. Otherwise opens `http://<addr>/ui` (falling back to printing the
/// URL when `open` fails — e.g. headless environments).
/// Test: with no daemon running → returns Err with "no daemon running"
/// message. With a fake `http_addr` file → prints URL.
fn run_dashboard() -> Result<()> {
    let Some(path) = http_addr_path() else {
        anyhow::bail!("could not resolve $HOME — set HOME and try again");
    };
    let addr = match std::fs::read_to_string(&path) {
        Ok(s) => s.trim().to_string(),
        Err(_) => {
            eprintln!(
                "{} No daemon running ({} not found). Start one with {}.",
                "✗".red(),
                path.display(),
                "trusty-search start".cyan()
            );
            std::process::exit(1);
        }
    };
    if addr.is_empty() {
        anyhow::bail!("{} is empty — daemon may be shutting down", path.display());
    }
    let url = format!("http://{addr}/ui");
    println!("{} Opening {} …", "◉".green(), url.cyan());
    if let Err(e) = open::that(&url) {
        eprintln!(
            "{} could not launch browser ({e}). Open this URL manually: {}",
            "⚠".yellow(),
            url
        );
    }
    Ok(())
}

// ── Service (macOS launchd) ───────────────────────────────────────────────

/// Reverse-DNS label for the LaunchAgent. Used as the plist filename and the
/// `Label` key — both must match for `launchctl` lookups to work.
#[cfg(target_os = "macos")]
const LAUNCHD_LABEL: &str = "com.trusty.trusty-search";

/// Dispatch a `trusty-search service <action>` invocation.
///
/// Why: launchd is macOS-specific; on other platforms we exit cleanly with a
/// clear message rather than emitting confusing plist errors.
/// What: macOS routes to `service_install` / `service_uninstall` /
/// `service_status` / `service_logs`. Non-macOS prints "not supported" and
/// exits 1.
/// Test: on Linux, every action returns Err with the platform message;
/// on macOS, `service status` runs `launchctl list` without crashing.
fn run_service_action(action: &ServiceAction) -> Result<()> {
    #[cfg(target_os = "macos")]
    {
        match action {
            ServiceAction::Install => service_install(),
            ServiceAction::Uninstall => service_uninstall(),
            ServiceAction::Status => service_status(),
            ServiceAction::Logs => service_logs(),
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = action;
        eprintln!(
            "{} `trusty-search service` is not supported on this platform — \
             use your distro's service manager (systemd, OpenRC, etc.) directly.",
            "✗".red()
        );
        std::process::exit(1);
    }
}

#[cfg(target_os = "macos")]
fn launchd_plist_path() -> Result<std::path::PathBuf> {
    let home = dirs::home_dir().ok_or_else(|| anyhow::anyhow!("could not resolve $HOME"))?;
    Ok(home
        .join("Library")
        .join("LaunchAgents")
        .join(format!("{LAUNCHD_LABEL}.plist")))
}

#[cfg(target_os = "macos")]
fn launchd_log_dir() -> Result<std::path::PathBuf> {
    let home = dirs::home_dir().ok_or_else(|| anyhow::anyhow!("could not resolve $HOME"))?;
    let dir = home.join("Library").join("Logs").join("trusty-search");
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// Render the LaunchAgent plist body. Foreground mode (launchd owns lifecycle).
#[cfg(target_os = "macos")]
fn launchd_plist_body(exe: &std::path::Path, log_dir: &std::path::Path) -> String {
    let exe = exe.display();
    let stdout = log_dir.join("stdout.log");
    let stderr = log_dir.join("stderr.log");
    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{LAUNCHD_LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{exe}</string>
        <string>start</string>
        <string>--foreground</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{}</string>
    <key>StandardErrorPath</key>
    <string>{}</string>
    <key>ProcessType</key>
    <string>Interactive</string>
</dict>
</plist>
"#,
        stdout.display(),
        stderr.display(),
    )
}

#[cfg(target_os = "macos")]
fn service_install() -> Result<()> {
    let exe = std::env::current_exe()
        .map_err(|e| anyhow::anyhow!("could not resolve current exe: {e}"))?;
    let plist_path = launchd_plist_path()?;
    if let Some(parent) = plist_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let log_dir = launchd_log_dir()?;
    let body = launchd_plist_body(&exe, &log_dir);
    std::fs::write(&plist_path, body)
        .map_err(|e| anyhow::anyhow!("write {}: {e}", plist_path.display()))?;
    println!(
        "{} Wrote LaunchAgent plist: {}",
        "✓".green(),
        plist_path.display()
    );

    // Bootstrap into the GUI domain of the current user. `bootout` first
    // (ignoring errors) so a re-install replaces a previously-loaded agent
    // cleanly.
    let uid = unsafe { libc::getuid() };
    let domain = format!("gui/{uid}");
    let _ = std::process::Command::new("launchctl")
        .args(["bootout", &domain])
        .arg(&plist_path)
        .status();
    let status = std::process::Command::new("launchctl")
        .args(["bootstrap", &domain])
        .arg(&plist_path)
        .status()
        .map_err(|e| anyhow::anyhow!("launchctl bootstrap failed: {e}"))?;
    if !status.success() {
        anyhow::bail!("launchctl bootstrap exited with {status}");
    }
    println!(
        "{} Loaded {} into {} — daemon will start automatically.",
        "✓".green(),
        LAUNCHD_LABEL,
        domain
    );
    println!(
        "  Logs:    {}\n  Status:  {}",
        log_dir.display().to_string().dimmed(),
        "trusty-search service status".cyan(),
    );
    Ok(())
}

#[cfg(target_os = "macos")]
fn service_uninstall() -> Result<()> {
    let plist_path = launchd_plist_path()?;
    let uid = unsafe { libc::getuid() };
    let domain = format!("gui/{uid}");
    if plist_path.exists() {
        let _ = std::process::Command::new("launchctl")
            .args(["bootout", &domain])
            .arg(&plist_path)
            .status();
        std::fs::remove_file(&plist_path)
            .map_err(|e| anyhow::anyhow!("remove {}: {e}", plist_path.display()))?;
        println!(
            "{} Unloaded and removed {}",
            "✓".green(),
            plist_path.display()
        );
    } else {
        println!(
            "{} {} not installed — nothing to do",
            "·".dimmed(),
            plist_path.display()
        );
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn service_status() -> Result<()> {
    let uid = unsafe { libc::getuid() };
    let target = format!("gui/{uid}/{LAUNCHD_LABEL}");
    let output = std::process::Command::new("launchctl")
        .args(["print", &target])
        .output()
        .map_err(|e| anyhow::anyhow!("launchctl print failed: {e}"))?;
    if output.status.success() {
        println!("{}", String::from_utf8_lossy(&output.stdout));
    } else {
        // `launchctl print` exits non-zero when the service isn't loaded.
        eprintln!(
            "{} {} is not loaded ({})",
            "✗".red(),
            target,
            String::from_utf8_lossy(&output.stderr).trim()
        );
        eprintln!("  Install with: {}", "trusty-search service install".cyan());
        std::process::exit(1);
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn service_logs() -> Result<()> {
    let log_dir = launchd_log_dir()?;
    let stdout = log_dir.join("stdout.log");
    let stderr = log_dir.join("stderr.log");
    if !stdout.exists() && !stderr.exists() {
        eprintln!(
            "{} No logs at {} yet — start the service first.",
            "·".dimmed(),
            log_dir.display()
        );
        return Ok(());
    }
    // Defer to `tail -F` so the user gets a familiar follow-mode experience
    // and we don't have to re-implement log rotation handling.
    let status = std::process::Command::new("tail")
        .arg("-F")
        .arg(&stdout)
        .arg(&stderr)
        .status()
        .map_err(|e| anyhow::anyhow!("tail failed: {e}"))?;
    if !status.success() {
        anyhow::bail!("tail exited with {status}");
    }
    Ok(())
}

// ── Main ──────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::from_filename(".env.local").ok();

    let cli = Cli::parse();

    // Tracing init + NO_COLOR handling via shared trusty-common helpers.
    trusty_common::init_tracing(if cli.verbose { 2 } else { 0 });
    trusty_common::maybe_disable_color(false);

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
            run_status(cli.json).await?;
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
            match register_index_with_daemon(&index_name, &project_path).await {
                Ok((true, _)) => {
                    println!(
                        "{} Registered '{}' with daemon at {}",
                        "✓".green(),
                        index_name.bold(),
                        project_path.display()
                    );
                    println!(
                        "  Run {} to index this project.",
                        "trusty-search index".cyan()
                    );
                }
                Ok((false, true)) => {
                    println!(
                        "{} '{}' already registered with daemon",
                        "↻".cyan(),
                        index_name.bold()
                    );
                    println!(
                        "  Run {} to index this project.",
                        "trusty-search index".cyan()
                    );
                }
                Ok((_, false)) => {
                    println!(
                        "{} Daemon not running — index will be created when daemon starts.",
                        "·".dimmed()
                    );
                    println!(
                        "  Start with {} then run {}.",
                        "trusty-search start".cyan(),
                        "trusty-search index".cyan()
                    );
                }
                Err(e) => {
                    eprintln!(
                        "{} {} — index will need to be re-registered when daemon is healthy",
                        "⚠".yellow(),
                        e
                    );
                }
            }
        }

        Commands::Index { path, name, force } => {
            let cwd = std::env::current_dir().unwrap_or_default();
            let project_path = path.unwrap_or(cwd);
            let index_name = name.unwrap_or_else(|| {
                project_path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned()
            });

            // 1. Register with daemon (idempotent). Surface a clear error if
            //    the daemon is unreachable — `index` is useless without it.
            let (created, daemon_reachable) =
                match register_index_with_daemon(&index_name, &project_path).await {
                    Ok(tuple) => tuple,
                    Err(e) => {
                        eprintln!("{} {}", "✗".red(), e);
                        std::process::exit(1);
                    }
                };
            if !daemon_reachable {
                eprintln!(
                    "{} Daemon not reachable at {}. Start it with {}.",
                    "✗".red(),
                    daemon_base_url().cyan(),
                    "trusty-search start".cyan(),
                );
                std::process::exit(1);
            }

            if created {
                println!(
                    "{} '{}' registered at {}",
                    "✓".green(),
                    index_name.bold(),
                    project_path.display()
                );
            }

            // 2. Run the reindex. The daemon's hash-skip optimization
            //    (see `reindex.rs::hash_content`) re-reads file content but
            //    skips re-embedding when the SHA-256 matches the previous
            //    run, so calling reindex even when nothing has changed is
            //    cheap. The final summary line tells the user whether any
            //    files actually changed (Improvement 3).
            //
            //    `--force` adds a post-reindex health check (chunk count +
            //    sanity query) so the user gets immediate feedback if the
            //    rebuild produced an empty/broken index.
            if force {
                run_reindex_force(&index_name, &project_path).await?;
            } else {
                run_reindex(&index_name, &project_path).await?;
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
            let client = trusty_common::server::daemon_http_client()?;
            let body = serde_json::json!({ "path": file.display().to_string() });
            match client.post(&url).json(&body).send().await {
                Ok(resp) if resp.status().is_success() => {
                    println!("{} [{}] removed {}", "−".red(), index_id, file.display());
                }
                Ok(resp) => {
                    eprintln!(
                        "{} daemon returned {} for {}",
                        "✗".red(),
                        resp.status(),
                        url
                    );
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
            let list_client = trusty_common::server::daemon_http_client()?;
            match list_client.get(&url).send().await {
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
            let client = trusty_common::server::daemon_http_client()?;

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
                        eprintln!("{} could not reach daemon at {}", "✗".red(), base);
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
                    eprintln!("{} index '{}' not found on daemon", "✗".red(), target_id);
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
                    format!(
                        "(intent={}, {}ms, {} results)",
                        intent,
                        latency,
                        results.len()
                    )
                    .dimmed()
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

        // `health` is an alias registered on the `status` subcommand, so
        // this arm catches the bare `Commands::Health` variant which is kept
        // for backward-compat with any scripts that invoke it directly.
        Commands::Health => {
            run_status(cli.json).await?;
        }

        Commands::Start { port, foreground } => {
            // `foreground` is currently a no-op: `run_daemon` already runs inline
            // and never forks. The flag is accepted so launchd/systemd plists can
            // declare the supervised contract explicitly in ProgramArguments
            // (see ~/Library/LaunchAgents/com.bobmatnyc.trusty-search.plist).
            // If a background-fork path is ever added, gate it on `!foreground`.
            let _ = foreground;
            // Fast-path: bail before loading the 86 MB embedding model when
            // another daemon is already running.  The lock check is ~1 ms;
            // FastEmbedder::new() can take several seconds on first run.
            if let Some(lock_path) = trusty_search_service::is_already_running() {
                eprintln!(
                    "{} another trusty-search daemon is already running (lock at {})",
                    "✗".red(),
                    lock_path.display()
                );
                std::process::exit(1);
            }

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
            let lock_path =
                dirs::data_local_dir().map(|d| d.join("trusty-search").join("daemon.lock"));
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
                            println!("{} Daemon may still be shutting down", "⚠".yellow());
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

        Commands::Serve {
            no_http,
            port,
            http,
        } => {
            let daemon_url = daemon_base_url();

            // Resolve the HTTP bind address. Precedence:
            //   1. `--no-http`              → disabled
            //   2. legacy `--http <addr>`   → explicit bind
            //   3. `--port <p>`             → 127.0.0.1:p (p=0 → OS picks)
            let bind_addr: Option<String> = if no_http {
                None
            } else if let Some(addr) = http {
                Some(addr)
            } else {
                Some(format!("127.0.0.1:{port}"))
            };

            let server = trusty_search_mcp::McpServer::new(daemon_url.clone());

            match bind_addr {
                Some(addr) => {
                    // Bind first so we can report the OS-chosen port when 0.
                    let listener = tokio::net::TcpListener::bind(&addr).await?;
                    let local = listener.local_addr()?;

                    // Write `~/.trusty-search/http_addr` so `trusty-search
                    // dashboard` (and other clients) can find this MCP
                    // server's HTTP transport. Best-effort: a missing $HOME
                    // is reported but doesn't abort.
                    let addr_file = http_addr_path();
                    if let Some(ref path) = addr_file {
                        if let Some(parent) = path.parent() {
                            let _ = std::fs::create_dir_all(parent);
                        }
                        match std::fs::write(path, format!("{local}\n")) {
                            Ok(()) => {}
                            Err(e) => {
                                eprintln!(
                                    "{} could not write {}: {e}",
                                    "⚠".yellow(),
                                    path.display()
                                );
                            }
                        }
                    }

                    eprintln!(
                        "trusty-search v{} — HTTP admin panel: http://{}",
                        env!("CARGO_PKG_VERSION"),
                        local,
                    );
                    eprintln!(
                        "{} MCP HTTP/SSE on {} → daemon {}",
                        "◉".green(),
                        local.to_string().cyan(),
                        daemon_url.dimmed()
                    );

                    let app = trusty_search_mcp::sse::router(server);
                    let serve_result = axum::serve(listener, app).await;

                    // Clean up the discovery file regardless of the serve
                    // outcome so a crashed `serve` doesn't leave a stale
                    // pointer.
                    if let Some(path) = addr_file {
                        let _ = std::fs::remove_file(&path);
                    }
                    serve_result?;
                }
                None => {
                    eprintln!(
                        "{} MCP stdio (no HTTP) → daemon {}",
                        "◉".green(),
                        daemon_url.dimmed()
                    );
                    trusty_search_mcp::stdio::run(server).await?;
                }
            }
        }

        Commands::Service { action } => {
            run_service_action(&action)?;
        }

        Commands::Dashboard => {
            run_dashboard()?;
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
                                println!("{} Already registered — reindex queued", "↻".cyan());
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
                                Ok((root, name)) => convert_one(root, name, &base, dry_run).await,
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
                        println!("{} Dry run complete: {} projects", "·".dimmed(), dry);
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
            let ui_probe_client = trusty_common::server::daemon_http_client()?;
            let healthy = ui_probe_client
                .get(&probe_url)
                .send()
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

        Commands::Doctor { fix } => {
            println!("\ntrusty-search doctor\n");
            println!("Checking configuration...\n");

            let (checks, empty_indexes) = run_doctor_checks().await;

            // Print all checks (index sub-lines were already printed inline
            // by run_doctor_checks, so we skip the index summary line itself
            // to avoid double-printing — it carries the per-index detail).
            for check in &checks {
                check.print();
            }

            let errors = checks.iter().filter(|c| c.is_error()).count();
            let warnings = checks.iter().filter(|c| c.is_warn()).count();

            // ── Fix mode ──────────────────────────────────────────────────
            if fix {
                let mut fixed_any = false;

                // Fix 1: Stale lock file.
                let data_dir = dirs::data_local_dir()
                    .map(|d| d.join("trusty-search"))
                    .unwrap_or_else(|| std::path::PathBuf::from("~/.local/share/trusty-search"));
                let lock_path = data_dir.join("daemon.lock");
                if lock_path.exists() {
                    let pid_opt = std::fs::read_to_string(&lock_path)
                        .ok()
                        .and_then(|s| s.trim().parse::<u32>().ok());
                    let stale = pid_opt
                        .map(|pid| unsafe { libc::kill(pid as libc::pid_t, 0) } != 0)
                        .unwrap_or(true);
                    if stale {
                        println!("\nFixing issues...");
                        fix_stale_lock(&data_dir);
                        fixed_any = true;
                    }
                }

                // Fix 2: Zero-chunk indexes — reindex each one.
                if !empty_indexes.is_empty() {
                    if !fixed_any {
                        println!("\nFixing issues...");
                        fixed_any = true;
                    }
                    for idx in &empty_indexes {
                        let root = if idx.root_path.is_empty() {
                            eprintln!(
                                "  {} '{}' has no root_path — cannot auto-fix; run `trusty-search index` manually",
                                "⚠".yellow(),
                                idx.name
                            );
                            continue;
                        } else {
                            std::path::PathBuf::from(&idx.root_path)
                        };
                        println!("  Indexing '{}'...", idx.name);
                        match run_reindex(&idx.name, &root).await {
                            Ok(()) => println!("  {} '{}' done", "✓".green(), idx.name),
                            Err(e) => println!("  {} '{}' failed: {e}", "✗".red(), idx.name),
                        }
                    }
                }

                // Fix 3: Missing model — cannot pre-download, just inform.
                let has_model_warn = checks.iter().any(|c| {
                    matches!(c, CheckResult::Warn(msg) if msg.contains("not cached") || msg.contains("not found"))
                });
                if has_model_warn {
                    if !fixed_any {
                        println!("\nFixing issues...");
                    }
                    println!(
                        "  {} Model downloads automatically on `trusty-search start` — no manual action needed",
                        "·".dimmed()
                    );
                }
            }

            // ── Summary ───────────────────────────────────────────────────
            println!();
            if errors == 0 && warnings == 0 {
                println!("{}", "Everything looks good!".green().bold());
            } else {
                if errors > 0 || warnings > 0 {
                    println!(
                        "Issues found: {} warning{}, {} error{}",
                        warnings,
                        if warnings == 1 { "" } else { "s" },
                        errors,
                        if errors == 1 { "" } else { "s" }
                    );
                }
                if !fix {
                    println!(
                        "Run {} to attempt automatic repair.",
                        "trusty-search doctor --fix".cyan()
                    );
                }
            }

            if errors > 0 {
                std::process::exit(1);
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
