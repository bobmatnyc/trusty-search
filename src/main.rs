use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "trusty-search", version, about = "Machine-wide hybrid code search service (BM25 + vector + KG)")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start MCP stdio server
    Serve {
        #[arg(long)]
        http: Option<String>,
    },
    /// Run as background HTTP daemon
    Daemon {
        #[arg(long, default_value = "0")]
        port: u16,
    },
    /// Index a directory
    Index {
        path: std::path::PathBuf,
        #[arg(long)]
        name: Option<String>,
    },
    /// Run a search query
    Query {
        text: String,
        #[arg(long)]
        index: Option<String>,
        #[arg(long, default_value = "10")]
        top_k: usize,
        #[arg(long)]
        json: bool,
    },
    /// Show daemon status and index stats
    Status,
    /// Watch a directory for changes and keep index updated
    Watch {
        path: std::path::PathBuf,
        #[arg(long)]
        name: Option<String>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::from_filename(".env.local").ok();
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Serve { http } => {
            tracing::info!("Starting trusty-search MCP server (http={:?})", http);
            println!("trusty-search MCP server starting...");
            std::future::pending::<()>().await;
        }
        Commands::Daemon { port } => {
            tracing::info!("Starting trusty-search daemon on port {}", port);
            println!("trusty-search daemon starting...");
            std::future::pending::<()>().await;
        }
        Commands::Index { path, name } => {
            let name = name.unwrap_or_else(|| path.file_name().unwrap_or_default().to_string_lossy().into_owned());
            println!("Indexing '{}' as '{}'", path.display(), name);
        }
        Commands::Query { text, index, top_k, json } => {
            let index_name = index.unwrap_or_else(|| "default".to_string());
            println!("Searching index '{}' for: {} (top_k={}{})", index_name, text, top_k, if json { ", json" } else { "" });
        }
        Commands::Status => println!("trusty-search status: not yet implemented"),
        Commands::Watch { path, name } => {
            let name = name.unwrap_or_else(|| path.file_name().unwrap_or_default().to_string_lossy().into_owned());
            println!("Watching '{}' as '{}'", path.display(), name);
        }
    }

    Ok(())
}
