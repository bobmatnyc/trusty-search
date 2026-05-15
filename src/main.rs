//! trusty-search CLI binary.
//!
//! Why: Single entry point that exposes both project-scoped commands
//! (`search`, `watch`, `status`, `init`, `add`, `remove`, `reindex`) which
//! auto-detect the index from the current working directory, and global
//! commands (`list`, `query`, `health`, `start`, `stop`, `serve`, `completions`)
//! that operate across the registry or manage the daemon.
//!
//! What: Parses CLI args via clap and dispatches to the per-subcommand
//! handlers in [`crate::commands`]. All implementation lives in those
//! modules; this file is purely a clap-to-handler shim plus a top-level
//! error printer.
//!
//! Test: `cargo run -- --help` → renders grouped command list with aliases.
//! `cargo run -- status` from inside this repo → prints `[trusty-search]`
//! detected via `.git`. `cargo test --workspace` → all tests pass.

mod commands;
mod detect;

// Re-export the library's modules into the binary's `crate::` namespace so
// existing `crate::core::*` / `crate::service::*` / `crate::mcp::*` imports
// in `commands/*.rs` resolve without churn after the workspace consolidation.
pub(crate) use trusty_search::{core, mcp, service};

use anyhow::Result;
use clap::{CommandFactory, Parser, Subcommand, ValueEnum};
use clap_complete::{generate, Shell};
use colored::Colorize;
use commands::convert::ConvertTarget;
use commands::service::ServiceAction;
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
    /// By default, self-spawns a detached background copy of itself (with
    /// `--foreground`) and returns immediately, so the daemon survives the
    /// caller's terminal closing (e.g. tmux pane SIGHUP, `make patch`).
    /// Use `--foreground` when the process is supervised by launchd, systemd,
    /// or Docker — those supervisors require the managed binary to stay in
    /// the foreground rather than forking.
    ///
    /// Examples:
    ///   trusty-search start
    ///   trusty-search start --port 7878
    ///   trusty-search start --foreground --port 7878   # launchd / systemd
    #[command(display_order = 20)]
    Start {
        /// Port to listen on (default: 7878, auto-selects next if busy)
        #[arg(long, default_value_t = trusty_search::service::DEFAULT_PORT)]
        port: u16,

        /// Run in the foreground instead of forking a background daemon.
        ///
        /// Default (`trusty-search start`): self-spawns a detached child with
        /// `--foreground` and returns immediately, so the daemon survives the
        /// caller's terminal closing (e.g. tmux pane SIGHUP). Use this flag
        /// when the process is managed by launchd, systemd, or Docker — those
        /// supervisors require the managed binary to stay in the foreground.
        #[arg(long, default_value_t = false)]
        foreground: bool,

        /// Embedding execution device: `auto` (default), `cpu`, or `gpu`.
        ///
        /// - `auto`: prefer CUDA on Linux/Windows (binary must be built with
        ///   `--features cuda`), then CoreML on Apple Silicon, otherwise CPU.
        /// - `cpu`: force CPU even when a GPU is available — useful for A/B
        ///   benchmarking or freeing the GPU for another workload.
        /// - `gpu`: require GPU acceleration; exit 1 if no GPU EP can be
        ///   initialised. Useful on a dedicated GPU indexing node where
        ///   silent CPU fallback would mean a 10× slower reindex.
        ///
        /// Implemented as the `TRUSTY_DEVICE` env var, which the embedder
        /// reads at session-init time. Set explicitly to override the daemon
        /// default.
        #[arg(long, value_parser = ["auto", "cpu", "gpu"], default_value = "auto")]
        device: String,
    },

    /// Stop the running background daemon
    ///
    /// Sends SIGTERM to the daemon process and waits for clean shutdown.
    ///
    /// Examples:
    ///   trusty-search stop
    #[command(display_order = 21)]
    Stop,

    /// Start MCP server (stdio by default; add --with-http for an HTTP listener)
    ///
    /// Stdio MCP is always served on the process's stdin/stdout for Claude
    /// Code, which pipes JSON-RPC directly and needs nothing more — so the
    /// HTTP listener is OFF by default (issue #123).
    ///
    /// Pass `--with-http` to additionally bind an HTTP/SSE transport on
    /// 127.0.0.1:<port> (port 0 = OS-chosen free port). The bound address is
    /// written to `~/.trusty-search/mcp_http_addr` so admin clients can
    /// discover it.
    ///
    /// Examples:
    ///   trusty-search serve                       # MCP stdio only (Claude hook)
    ///   trusty-search serve --with-http           # MCP stdio + HTTP on :0
    ///   trusty-search serve --with-http --port 7878  # MCP stdio + HTTP on :7878
    ///   trusty-search serve --http 0.0.0.0:8080   # legacy: explicit bind addr
    #[command(display_order = 22)]
    Serve {
        /// Enable the HTTP/SSE listener in addition to MCP stdio.
        ///
        /// Off by default: Claude Code MCP hooks pipe JSON-RPC over
        /// stdin/stdout and never need the HTTP admin panel, so binding it
        /// would just waste a port. Opt in with this flag when you want the
        /// HTTP transport (e.g. for the browser admin panel).
        #[arg(long, default_value_t = false)]
        with_http: bool,

        /// Port for the HTTP/SSE MCP transport (default: 0 = OS picks).
        ///
        /// Only used when `--with-http` is set.
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

#[tokio::main]
async fn main() {
    // Central error-printer + exit-code chooser. Why: command handlers are now
    // testable units that return `Result<()>` instead of calling `process::exit`
    // directly (issue #104). Print the chain compactly with the red ✗ prefix
    // operators already recognize, then exit 1.
    if let Err(e) = run().await {
        let msg = format!("{:#}", e);
        if !msg.is_empty() {
            eprintln!("{} {}", "✗".red(), msg);
        }
        std::process::exit(1);
    }
}

async fn run() -> Result<()> {
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

        Commands::Start {
            port,
            foreground,
            device,
        } => {
            commands::start::handle_start(port, foreground, &device).await?;
        }

        Commands::Stop => {
            commands::stop::handle_stop().await?;
        }

        Commands::Serve {
            with_http,
            port,
            http,
        } => {
            commands::serve::handle_serve(with_http, port, http).await?;
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
