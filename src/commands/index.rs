//! Handler for `trusty-search index` (register + reindex in one step).

use crate::{daemon_base_url, register_index_with_daemon, run_reindex, run_reindex_force};
use anyhow::Result;
use colored::Colorize;

/// Why: register-then-reindex is the primary onboarding flow. Extracted so the
/// `main()` dispatcher doesn't have to know about both helpers.
/// What: 1) idempotent `POST /indexes`; bails when daemon unreachable. 2) runs
/// the regular reindex or the `--force` variant (which adds a post-reindex
/// health check).
/// Test: `cargo run -- index --force` against a healthy daemon prints the
/// registration line then drives the SSE progress bar.
pub async fn handle_index(
    path: Option<std::path::PathBuf>,
    name: Option<String>,
    force: bool,
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

    // 0. Auto-start the daemon if needed. `index` is useless without it,
    //    so we proactively boot it rather than dump a confusing connection
    //    error on the user.
    crate::commands::daemon_guard::ensure_daemon_running_or_exit(&daemon_base_url()).await;

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
    Ok(())
}
