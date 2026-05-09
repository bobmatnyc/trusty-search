//! trusty-search CLI binary.
//!
//! Why: Single entry point that exposes both project-scoped commands
//! (`search`, `watch`, `status`, `init`, `add`, `remove`, `reindex`) which
//! auto-detect the index from the current working directory, and global
//! commands (`list`, `query`, `health`, `daemon`, `serve`, `completions`)
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
/// Use `trusty-search daemon` to start the background service first.
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
    /// Start background HTTP daemon
    ///
    /// Examples:
    ///   trusty-search daemon
    ///   trusty-search daemon --port 7878
    ///   trusty-search daemon --stop
    #[command(display_order = 20)]
    Daemon {
        /// Port to listen on (0 = auto-select)
        #[arg(long, default_value = "7878")]
        port: u16,

        /// Stop the running daemon
        #[arg(long)]
        stop: bool,
    },

    /// Start MCP stdio server for Claude Code integration
    ///
    /// Examples:
    ///   trusty-search serve
    ///   trusty-search serve --http 0.0.0.0:8080
    #[command(display_order = 21)]
    Serve {
        /// Start HTTP/SSE mode instead of stdio
        #[arg(long)]
        http: Option<String>,
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
/// `trusty-search daemon`. Falls back to `7878` when the file is missing,
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
            let (index_id, warned) = resolve_index(&cli.index);
            print_index_header(&index_id, warned);
            if cli.json {
                println!(
                    r#"{{"index_id":"{}","status":"not_implemented"}}"#,
                    index_id
                );
            } else {
                println!("{} {}", "Index:".bold(), index_id);
                println!(
                    "{}",
                    "  Status endpoint not yet implemented — see issue #8".yellow()
                );
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

            println!(
                "{} Registered '{}' at {}",
                "✓".green(),
                index_name.bold(),
                project_path.display()
            );
            println!(
                "  Run {} to index this project.",
                "trusty-search reindex".cyan()
            );
        }

        Commands::Add { file } => {
            let (index_id, warned) = resolve_index(&cli.index);
            print_index_header(&index_id, warned);
            println!("{} [{}] {}", "→".cyan(), index_id, file.display());
            println!(
                "{}",
                "  File add not yet implemented — see issue #3".yellow()
            );
        }

        Commands::Remove { file } => {
            let (index_id, warned) = resolve_index(&cli.index);
            print_index_header(&index_id, warned);
            println!("{} [{}] {}", "−".red(), index_id, file.display());
            println!(
                "{}",
                "  File remove not yet implemented — see issue #3".yellow()
            );
        }

        Commands::Reindex { path } => {
            let (index_id, warned) = resolve_index(&cli.index);
            print_index_header(&index_id, warned);
            let reindex_path = path.unwrap_or_else(|| {
                let cwd = std::env::current_dir().unwrap_or_default();
                detect_project(&cwd).root_path
            });
            println!(
                "{} Reindexing {} as '{}'…",
                "⟳".cyan(),
                reindex_path.display(),
                index_id
            );
            println!(
                "{}",
                "  Reindex not yet implemented — see issue #3".yellow()
            );
        }

        Commands::List => {
            if cli.json {
                println!(r#"{{"indexes":[],"note":"not_implemented"}}"#);
            } else {
                println!("{}", "Registered indexes:".bold());
                println!(
                    "{}",
                    "  List endpoint not yet implemented — see issue #8".yellow()
                );
            }
        }

        Commands::Query {
            query,
            indexes,
            top_k,
            full,
        } => {
            println!(
                "{} {} {} {} {}",
                "→".cyan(),
                format!("[{}]", indexes).dimmed(),
                query.bold(),
                format!("(top-{})", top_k).dimmed(),
                if full { "(full)" } else { "" }.dimmed()
            );
            println!(
                "{}",
                "  Cross-project search not yet implemented — see issue #10".yellow()
            );
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

        Commands::Daemon { port, stop } => {
            if stop {
                eprintln!(
                    "{} `daemon --stop` is not implemented — send SIGTERM/SIGINT instead",
                    "·".dimmed()
                );
                std::process::exit(2);
            }
            let state = trusty_search_service::SearchAppState {
                registry: trusty_search_core::registry::IndexRegistry::new(),
            };
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

        Commands::Completions { shell } => {
            let mut cmd = Cli::command();
            let name = cmd.get_name().to_string();
            generate(shell, &mut cmd, name, &mut io::stdout());
        }
    }

    Ok(())
}
