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

mod commands;
mod detect;
mod doctor;

// Re-export the library's modules into the binary's `crate::` namespace so
// existing `crate::core::*` / `crate::service::*` / `crate::mcp::*` imports
// in `commands/*.rs` resolve without churn after the workspace consolidation.
pub(crate) use trusty_search::{core, mcp, service};

pub(crate) use doctor::run_doctor_checks;

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

        /// SSE stream timeout in seconds (default: 600). Increase for very large repos.
        #[arg(long, default_value_t = 600)]
        timeout: u64,
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

        /// SSE stream timeout in seconds (default: 600). Increase for very large repos.
        #[arg(long, default_value_t = 600)]
        timeout: u64,
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
    /// Why: Provides quick access to the embedded web management UI.
    /// What: Discovers the daemon URL and opens it in the system browser.
    /// Test: `cargo run -- dashboard` (also `dash` / `ui`) opens the browser
    /// when a daemon is running; otherwise auto-starts the daemon first.
    ///
    /// Examples:
    ///   trusty-search dashboard
    ///   trusty-search dash
    ///   trusty-search ui
    #[command(display_order = 23, aliases = ["dash", "ui"])]
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
/// What: Mirrors `crate::core::QueryIntent` for the CLI surface.
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
/// file. Mirrors `crate::service::daemon::http_addr_path` so the CLI
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
        let walk = crate::service::walker::walk_source_files(path);
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

/// Print per-subsystem indexing time breakdown after a successful reindex.
///
/// Why: gives the operator proof that each subsystem (parse, embed, BM25, KG)
/// actually ran and how long each took. The vector-count check is the
/// smoking-gun signal for the "embedder silently fell back to BM25" failure
/// mode — printed as a loud warning so it can never go unnoticed.
/// What: 4-line breakdown, plus a 5th warning line when `vector_count == 0`
/// despite non-zero chunks (the BM25-only-mode signal).
/// Test: call with synthetic timings where vector_count==0 and total_chunks>0;
/// assert the warning line is printed.
fn print_timing_breakdown(t: &ReindexTimings, total_chunks: u64) {
    println!(
        "  {} {:>7}  ({} chunks)",
        "Parse+chunk:".dimmed(),
        fmt_elapsed(t.parse_ms),
        format_with_commas(total_chunks),
    );
    if t.vector_count == 0 && total_chunks > 0 {
        println!(
            "  {} {}",
            "Embed (HNSW):".dimmed(),
            "SKIPPED (embedder unavailable — BM25-only mode)"
                .yellow()
                .bold(),
        );
    } else {
        println!(
            "  {} {:>7}  ({} vectors)",
            "Embed (HNSW):".dimmed(),
            fmt_elapsed(t.embed_ms),
            format_with_commas(t.vector_count),
        );
    }
    println!("  {} {:>7}", "BM25:".dimmed(), fmt_elapsed(t.bm25_ms));
    println!(
        "  {} {:>7}  ({} symbols, {} edges)",
        "KG:".dimmed(),
        fmt_elapsed(t.kg_ms),
        format_with_commas(t.symbol_count),
        format_with_commas(t.edge_count),
    );
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
///     Files: 7,234/14,445  Chunks: 58,402  Skipped: 12  Speed: 142 files/s  Elapsed: 50s  ETA: 50s
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
        let total = self.files.length().unwrap_or(0);
        let files_per_sec = indexed.checked_div(elapsed_secs).unwrap_or(0);
        let eta = if files_per_sec > 0 && total > indexed {
            fmt_secs((total - indexed) / files_per_sec)
        } else {
            "?".to_string()
        };
        self.stats.set_message(format!(
            "Files: {indexed}/{total}  Chunks: {chunks}  Skipped: {skipped}  Speed: {fps} files/s  Elapsed: {elapsed}  ETA: {eta}",
            indexed = format_with_commas(indexed),
            total = format_with_commas(total),
            chunks = format_with_commas(total_chunks),
            skipped = format_with_commas(skipped),
            fps = files_per_sec,
            elapsed = fmt_secs(elapsed_secs),
            eta = eta,
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
#[derive(Debug, Clone, Copy)]
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
    /// Maximum wall-clock seconds to wait for the SSE reindex stream to emit
    /// a `complete` event. Default: 600. Use `--timeout 0` to disable (wait
    /// forever). When the deadline is exceeded the CLI prints a warning and
    /// exits; the daemon continues indexing in the background.
    timeout_secs: u64,
}

impl Default for ReindexOptions {
    fn default() -> Self {
        Self {
            verify_after: false,
            prior_chunk_count: None,
            force: false,
            timeout_secs: 600,
        }
    }
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
    /// Per-subsystem timings captured from the daemon's `complete` event
    /// `timings` payload. `None` when the daemon is an older version that
    /// didn't emit timings — caller renders a single-line summary in that case.
    timings: Option<ReindexTimings>,
}

/// Per-subsystem indexing timings parsed from the SSE `complete` event.
///
/// Why: gives the user proof that each subsystem ran and how long each took.
/// `vector_count == 0` with `total_chunks > 0` is the smoking-gun signal that
/// the embedder silently fell back to BM25-only — surfaced as a warning in the
/// CLI breakdown so this regression can never go unnoticed.
#[derive(Debug, Default, Clone, Copy)]
struct ReindexTimings {
    parse_ms: u64,
    embed_ms: u64,
    bm25_ms: u64,
    #[allow(dead_code)]
    vector_upsert_ms: u64,
    kg_ms: u64,
    vector_count: u64,
    symbol_count: u64,
    edge_count: u64,
}

/// Plain reindex (no post-verify). Used by the non-force `index` command, the
/// bare `reindex` command, and the doctor auto-repair path. The daemon's
/// hash-skip optimization (see `reindex.rs::hash_content`) means unchanged
/// files are cheap, so calling this even when nothing changed is fine.
///
/// `timeout_secs` caps how long the CLI waits for the SSE stream's `complete`
/// event. 0 means no limit (wait forever). Default for callers that don't have
/// an explicit user-supplied value: 600.
async fn run_reindex(index_id: &str, root_path: &std::path::Path, timeout_secs: u64) -> Result<()> {
    run_reindex_with(
        index_id,
        root_path,
        ReindexOptions {
            timeout_secs,
            ..ReindexOptions::default()
        },
    )
    .await
    .map(|_| ())
}

/// `index --force` reindex: snapshot the prior chunk count, kick off a full
/// reindex, and run a post-reindex health check. Exits 1 if the new index
/// looks unhealthy (no chunks or empty sanity query).
async fn run_reindex_force(
    index_id: &str,
    root_path: &std::path::Path,
    timeout_secs: u64,
) -> Result<()> {
    let prior = fetch_chunk_count(index_id).await;
    let opts = ReindexOptions {
        verify_after: true,
        prior_chunk_count: prior,
        force: true,
        timeout_secs,
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
    //
    // The per-request reqwest timeout only governs the *connection* phase here;
    // we handle the overall stream deadline ourselves below via
    // `tokio::time::timeout` so we can print a friendly warning instead of a
    // raw timeout error.
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
                let fps = indexed.checked_div(elapsed).unwrap_or(0);
                let total = files_bar.length().unwrap_or(0);
                let eta = if fps > 0 && total > indexed {
                    fmt_secs((total - indexed) / fps)
                } else {
                    "?".to_string()
                };
                stats_bar.set_message(format!(
                    "Files: {indexed}/{total}  Chunks: {chunks}  Skipped: {skipped}  Speed: {fps} files/s  Elapsed: {elapsed}s  ETA: {eta}",
                    indexed = format_with_commas(indexed),
                    total = format_with_commas(total),
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
    let mut timed_out = false;

    // Optional wall-clock deadline for the SSE stream. `timeout_secs == 0`
    // means wait forever (legacy behaviour). Otherwise each `stream.next()`
    // is raced against `tokio::time::sleep_until(deadline)` via
    // `tokio::select!`. When the sleep wins we set `timed_out = true` and
    // break so the post-loop path can print the canonical warning.
    // The daemon continues indexing in the background.
    let deadline: Option<tokio::time::Instant> = if opts.timeout_secs > 0 {
        Some(tokio::time::Instant::now() + Duration::from_secs(opts.timeout_secs))
    } else {
        None
    };

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
        // Race the next SSE event against the optional deadline. When the
        // deadline fires `timed_out` is set and we break cleanly; the
        // post-loop section emits the warning and returns Ok.
        let maybe_event = if let Some(dl) = deadline {
            tokio::select! {
                biased;
                ev = stream.next() => ev,
                _ = tokio::time::sleep_until(dl) => {
                    timed_out = true;
                    break;
                }
            }
        } else {
            stream.next().await
        };
        let event = match maybe_event {
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
                // Per-subsystem timings (added in 0.3.11). Absent when talking
                // to an older daemon — outcome.timings stays `None` and the
                // CLI falls back to the legacy single-line summary.
                if let Some(t) = evt.get("timings") {
                    let get = |k: &str| t.get(k).and_then(|v| v.as_u64()).unwrap_or(0);
                    outcome.timings = Some(ReindexTimings {
                        parse_ms: get("parse_ms"),
                        embed_ms: get("embed_ms"),
                        bm25_ms: get("bm25_ms"),
                        vector_upsert_ms: get("vector_upsert_ms"),
                        kg_ms: get("kg_ms"),
                        vector_count: get("vector_count"),
                        symbol_count: get("symbol_count"),
                        edge_count: get("edge_count"),
                    });
                }
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

    if timed_out {
        // The SSE deadline fired before the daemon emitted `complete`. The
        // daemon is still indexing in the background. Print the canonical
        // warning (exact text the issue tracker refers to) and return Ok so
        // callers don't treat this as a hard error.
        ui.abandon(format!(
            "{} trusty-search index timed out after {}s — continuing; re-run later if needed",
            "⚠".yellow(),
            opts.timeout_secs,
        ));
        eprintln!(
            "{} Daemon is still indexing in the background. \
             Use `trusty-search status` or re-run `trusty-search index` to check progress. \
             Pass `--timeout <seconds>` to wait longer (e.g. `--timeout 1200`).",
            "ℹ".cyan()
        );
        return Ok(outcome);
    }

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

    // ── Per-subsystem timing breakdown (issue: silent BM25 fallback) ──────
    // We render this AFTER `ui.finish` so the indicatif `MultiProgress`
    // doesn't redraw over our printed lines. Skipped entirely when talking
    // to a daemon older than 0.3.11 (no `timings` block in the SSE
    // `complete` event).
    if let Some(t) = outcome.timings {
        print_timing_breakdown(&t, outcome.total_chunks);
    }

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

/// Why: separates the network probe from the result-formatting so the
/// formatting check can be tested without async/HTTP.
/// What: returns `(running, version)` by hitting `/health`.
/// Test: with daemon running, returns `(true, "x.y.z")`; without, `(false, "")`.
async fn probe_daemon_health(client: &reqwest::Client, base: &str) -> (bool, String) {
    let health_result = client.get(format!("{}/health", base)).send().await;
    match health_result {
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
    }
}

/// Why: pure formatting of the daemon liveness verdict.
/// What: Ok with version string when running, Error otherwise.
/// Test: pass `(true, "1.2")` — get Ok variant; `(false, "")` — get Error.
fn check_daemon_running(running: bool, base: &str, version: &str) -> CheckResult {
    if running {
        CheckResult::Ok(format!("Daemon running at {} (v{})", base, version))
    } else {
        CheckResult::Error("Daemon not running — run `trusty-search start`".to_string())
    }
}

/// Why: model-cache detection is independent of network state, so isolating
/// it lets the doctor produce useful output even with the daemon down.
/// What: inspects `~/Library/Caches/...` for the ONNX model subdir and
/// returns Ok with size / Warn otherwise.
/// Test: with model present, returns Ok and message contains "Model cache:";
/// without, returns Warn.
fn check_model_cache() -> CheckResult {
    let model_cache = fastembed_cache_dir();
    let model_name = "all-MiniLM-L6-v2";
    let model_subdir = model_cache.join("models--Qdrant--all-MiniLM-L6-v2-onnx");
    if model_subdir.exists() {
        let size = dir_size_bytes(&model_cache);
        CheckResult::Ok(format!(
            "Model cache: {} ({}, {})",
            model_cache.display(),
            fmt_bytes(size),
            model_name
        ))
    } else if model_cache.exists() {
        CheckResult::Warn(format!(
            "Model cache directory exists ({}) but {} not found — will download on first start",
            model_cache.display(),
            model_name
        ))
    } else {
        CheckResult::Warn(
            "Model not cached — will download on first `trusty-search start`".to_string(),
        )
    }
}

/// Why: data-dir path is used by multiple checks; centralizing the resolution
/// keeps the path logic in one place.
/// What: returns the per-user data dir (`~/Library/Application Support/...` on
/// macOS, falling back to a placeholder string on platforms with no default).
/// Test: on macOS/Linux, returns a path ending in `trusty-search`.
fn doctor_data_dir() -> std::path::PathBuf {
    dirs::data_local_dir()
        .map(|d| d.join("trusty-search"))
        .unwrap_or_else(|| std::path::PathBuf::from("~/.local/share/trusty-search"))
}

/// Why: data-dir writability is required for daemon startup, so a dedicated
/// check makes the failure mode visible at doctor time.
/// What: probes existence + writability via a temp file; returns Ok/Error/Warn.
/// Test: a writable temp dir yields Ok; a non-existent path yields Warn.
fn check_data_dir(data_dir: &std::path::Path) -> CheckResult {
    if !data_dir.exists() {
        return CheckResult::Warn(format!(
            "Data directory {} does not exist (will be created on first start)",
            data_dir.display()
        ));
    }
    let probe = data_dir.join(".write_probe");
    let writable = std::fs::write(&probe, b"").is_ok();
    let _ = std::fs::remove_file(&probe);
    if writable {
        CheckResult::Ok(format!("Data directory: {} (writable)", data_dir.display()))
    } else {
        CheckResult::Error(format!(
            "Data directory {} is not writable",
            data_dir.display()
        ))
    }
}

/// Why: stale lockfiles block daemon startup; surfacing them at doctor time
/// (with PID liveness info) lets `--fix` clean them up automatically.
/// What: reads `daemon.lock`, parses PID, checks if alive via `kill(pid, 0)`,
/// and emits Ok/Warn accordingly. Cross-references `daemon_running` to flag
/// suspicious "alive PID but /health fails" states.
/// Test: an absent lockfile returns Ok ("no stale lock"); a lockfile with a
/// dead PID returns Warn ("Stale lock file: ...").
fn check_lock_file(data_dir: &std::path::Path, daemon_running: bool) -> CheckResult {
    let lock_path = data_dir.join("daemon.lock");
    if !lock_path.exists() {
        return CheckResult::Ok("Lock file: healthy (no stale lock)".into());
    }
    let pid_opt = std::fs::read_to_string(&lock_path)
        .ok()
        .and_then(|s| s.trim().parse::<u32>().ok());
    let Some(pid) = pid_opt else {
        return CheckResult::Warn(format!(
            "Lock file exists but contains no valid PID ({})",
            lock_path.display()
        ));
    };
    // POSIX: kill(pid, 0) — check existence without sending a signal.
    let alive = unsafe { libc::kill(pid as libc::pid_t, 0) } == 0;
    if !alive {
        return CheckResult::Warn(format!(
            "Stale lock file: PID {} is not running ({})",
            pid,
            lock_path.display()
        ));
    }
    if daemon_running {
        CheckResult::Ok(format!("Lock file: healthy (PID {} is running)", pid))
    } else {
        CheckResult::Warn(format!(
            "Lock file contains PID {} which is alive but /health failed",
            pid
        ))
    }
}

/// Why: parsing the `/indexes` listing is its own concern — failure to reach
/// the daemon should silently degrade to an empty list (the index summary
/// will then report "no indexes").
/// What: GETs `/indexes` and extracts the `indexes: [..]` string array.
/// Test: a 200 with `{"indexes": ["a", "b"]}` returns `["a", "b"]`; any error
/// returns `[]`.
async fn fetch_index_names(client: &reqwest::Client, base: &str) -> Vec<String> {
    let list = client.get(format!("{}/indexes", base)).send().await;
    let list_body: serde_json::Value = match list {
        Ok(r) if r.status().is_success() => {
            r.json().await.unwrap_or_else(|_| serde_json::json!({}))
        }
        _ => serde_json::json!({"indexes": []}),
    };
    let empty_arr: Vec<serde_json::Value> = Vec::new();
    list_body
        .get("indexes")
        .and_then(|v| v.as_array())
        .unwrap_or(&empty_arr)
        .iter()
        .filter_map(|v| v.as_str().map(|s| s.to_string()))
        .collect()
}

/// Why: fan-out of status calls keeps doctor responsive on installs with many
/// indexes; doing this serially would multiply latency by N.
/// What: concurrently fetches `/indexes/:name/status` for each name and
/// returns the results sorted by name.
/// Test: with three indexes registered, returns three entries sorted
/// alphabetically.
async fn fetch_index_statuses(
    client: &reqwest::Client,
    base: &str,
    names: &[String],
) -> Vec<(String, serde_json::Value)> {
    let mut joinset = tokio::task::JoinSet::new();
    for name in names {
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
    per_index
}

/// Why: the summary line wording depends on multiple count-derived plurals;
/// isolating the string-building keeps the orchestrator clean.
/// What: returns Ok when all indexes have chunks, Warn when at least one is
/// empty (with the count formatted).
/// Test: `(2, 0)` → Ok "2 indexes registered, all have chunks"; `(2, 1)` →
/// Warn "2 indexes registered, 1 has no chunks yet:".
fn summarize_indexes(total: usize, zero_count: usize) -> CheckResult {
    if zero_count == 0 {
        CheckResult::Ok(format!(
            "{} index{} registered, all have chunks",
            total,
            if total == 1 { "" } else { "es" }
        ))
    } else {
        CheckResult::Warn(format!(
            "{} index{} registered, {} {} no chunks yet:",
            total,
            if total == 1 { "" } else { "es" },
            zero_count,
            if zero_count == 1 { "has" } else { "have" }
        ))
    }
}

/// Why: printing per-index detail is a side-effect distinct from the
/// CheckResult summary; separating it keeps the summary function pure.
/// What: prints one indented line per index (with chunk count and root path),
/// and records empty indexes in `empty_indexes` so `--fix` can repair them.
/// Test: with one populated and one empty index, two lines print and one
/// `EmptyIndex` is pushed.
fn print_index_breakdown(
    per_index: &[(String, serde_json::Value)],
    empty_indexes: &mut Vec<EmptyIndex>,
) {
    for (name, body) in per_index {
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

/// Why: TCP-reachability probe is independent of HTTP and from the rest of
/// the doctor logic; lifting it keeps the orchestrator's control flow flat.
/// What: returns Ok when 127.0.0.1:port accepts a connection, Error otherwise.
/// Test: with daemon listening on the read port, returns Ok; with daemon
/// stopped, returns Error.
async fn check_port_reachable(port: u16) -> CheckResult {
    if port_reachable("127.0.0.1", port).await {
        CheckResult::Ok(format!("Port {} is reachable", port))
    } else {
        CheckResult::Error(format!("Port {} is not reachable", port))
    }
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

/// Render the `<key>EnvironmentVariables</key>` plist fragment.
///
/// Why: launchd re-spawns the daemon without the user's shell environment.
/// Embedding env vars directly in the plist provides a belt-and-suspenders
/// guarantee for operator tunables, and pins `HF_HOME` to the user's standard
/// Hugging Face cache directory so fastembed-rs never inherits a non-standard
/// or read-only `HF_HOME` that was set in an earlier shell session (fixes #86).
/// What: always emits an `HF_HOME` entry resolved at install time, plus any
/// `PERSISTED_ENV_VARS` that are currently set.
/// Test: call `launchd_env_vars_plist()` with HOME set; assert output contains
/// `<key>HF_HOME</key>` and the resolved path ends in `.cache/huggingface`.
#[cfg(target_os = "macos")]
fn launchd_env_vars_plist() -> String {
    use crate::service::PERSISTED_ENV_VARS;

    let xml_escape = |s: &str| -> String {
        s.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
    };

    let mut pairs: Vec<String> = Vec::new();

    // Always pin HF_HOME to $HOME/.cache/huggingface resolved at install time.
    // fastembed-rs follows HF_HOME when present; if it points at a read-only
    // location (e.g. a previous admin install) the embedder silently falls
    // back to BM25-only mode. Setting it here guarantees the correct writable
    // path regardless of what the operator's shell had in HF_HOME.
    if let Some(home) = dirs::home_dir() {
        let hf_home = home.join(".cache").join("huggingface");
        let escaped = xml_escape(&hf_home.display().to_string());
        pairs.push(format!(
            "        <key>HF_HOME</key>\n        <string>{escaped}</string>"
        ));
    }

    // Append operator tunables (TRUSTY_* vars) that are currently set.
    for key in PERSISTED_ENV_VARS {
        if let Ok(val) = std::env::var(key) {
            let escaped = xml_escape(&val);
            pairs.push(format!(
                "        <key>{key}</key>\n        <string>{escaped}</string>"
            ));
        }
    }

    if pairs.is_empty() {
        String::new()
    } else {
        format!(
            "    <key>EnvironmentVariables</key>\n    <dict>\n{}\n    </dict>\n",
            pairs.join("\n")
        )
    }
}

/// Render the LaunchAgent plist body. Foreground mode (launchd owns lifecycle).
#[cfg(target_os = "macos")]
fn launchd_plist_body(exe: &std::path::Path, log_dir: &std::path::Path) -> String {
    let exe = exe.display();
    let stdout = log_dir.join("stdout.log");
    let stderr = log_dir.join("stderr.log");
    let env_vars_section = launchd_env_vars_plist();
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
{env_vars_section}    <key>RunAtLoad</key>
    <true/>
    <!-- KeepAlive=SuccessfulExit:false means launchd only restarts the daemon
         on a non-zero exit. The `start` command exits 0 when a live daemon is
         already running (idempotent fast-path); without this, launchd would
         immediately re-spawn and crash-loop on the existing lockfile. -->
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>
    <key>ThrottleInterval</key>
    <integer>30</integer>
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
            commands::search::handle_search(&cli.index, query, top_k).await?;
        }

        Commands::Watch { path } => {
            commands::watch::handle_watch(&cli.index, path).await?;
        }

        Commands::Status => {
            commands::status::handle_status(cli.json).await?;
        }

        Commands::Init {
            path,
            name,
            exclude,
        } => {
            commands::init::handle_init(path, name, exclude).await?;
        }

        Commands::Index {
            path,
            name,
            force,
            timeout,
        } => {
            commands::index::handle_index(path, name, force, timeout).await?;
        }

        Commands::Add { file } => {
            commands::add::handle_add(&cli.index, file).await?;
        }

        Commands::Remove { file } => {
            commands::remove::handle_remove(&cli.index, file).await?;
        }

        Commands::Reindex { path, timeout } => {
            commands::reindex::handle_reindex(&cli.index, path, timeout).await?;
        }

        Commands::List => {
            commands::list::handle_list(cli.json).await?;
        }

        Commands::Query {
            query,
            indexes,
            top_k,
            full,
        } => {
            commands::query::handle_query(&cli.index, cli.json, query, indexes, top_k, full)
                .await?;
        }

        // `health` is an alias registered on the `status` subcommand, so
        // this arm catches the bare `Commands::Health` variant which is kept
        // for backward-compat with any scripts that invoke it directly.
        Commands::Health => {
            commands::status::handle_status(cli.json).await?;
        }

        Commands::Start { port, foreground } => {
            commands::start::handle_start(port, foreground).await?;
        }

        Commands::Stop => {
            commands::stop::handle_stop().await?;
        }

        Commands::Serve {
            no_http,
            port,
            http,
        } => {
            commands::serve::handle_serve(no_http, port, http).await?;
        }

        Commands::Service { action } => {
            commands::service::handle_service(&action)?;
        }

        Commands::Dashboard => {
            commands::dashboard::handle_dashboard().await?;
        }

        Commands::Convert {
            target,
            dry_run,
            concurrency,
        } => {
            commands::convert::handle_convert(target, dry_run, concurrency).await?;
        }

        Commands::Doctor { fix } => {
            commands::doctor::handle_doctor(fix).await?;
        }

        Commands::Completions { shell } => {
            let mut cmd = Cli::command();
            let name = cmd.get_name().to_string();
            generate(shell, &mut cmd, name, &mut io::stdout());
        }
    }

    Ok(())
}
