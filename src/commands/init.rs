//! Handler for `trusty-search init` (register an index without indexing).

use super::reindex_engine::register_index_with_daemon;
use anyhow::Result;
use colored::Colorize;

/// Why: extracted from `main()` so the registration flow is testable in isolation
/// and `main()` stays a thin dispatcher. Behaviour is unchanged from the
/// pre-refactor inline implementation.
/// What: writes a `.trusty-search` marker file (idempotent), then registers the
/// index with the daemon. Distinguishes "registered", "already registered",
/// and "daemon not running" in the user-facing output.
/// Test: `cargo run -- init /tmp/empty` creates a marker and reports daemon
/// status truthfully (registered / already-registered / daemon-down).
pub async fn handle_init(
    path: Option<std::path::PathBuf>,
    name: Option<String>,
    exclude: Vec<String>,
) -> Result<()> {
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
    Ok(())
}
