//! Handler for `trusty-search index` (register + reindex in one step).
//!
//! Why: register-then-reindex is the primary onboarding flow. When a repo
//! contains a `trusty-search.yaml` file (issue: repo-level config), we
//! transparently fan out into one register+reindex pass per declared index
//! so a single `trusty-search index` command can populate multiple named
//! slices (e.g. `duetto-api` and `duetto-ui`).

use crate::core::repo_config::{RepoConfig, CONFIG_FILENAME};
use crate::{daemon_base_url, register_index_with_daemon, run_reindex, run_reindex_force};
use anyhow::Result;
use colored::Colorize;

/// Entry point for `trusty-search index`.
///
/// Why: register-then-reindex is the primary onboarding flow. With a
/// `trusty-search.yaml` present, this dispatches into a multi-index pass;
/// otherwise it falls back to the single-index behaviour.
/// What:
/// 1. Auto-start the daemon if needed.
/// 2. Look for `<path>/trusty-search.yaml`. If present, ignore `--name` and
///    register+reindex each declared index sequentially.
/// 3. Otherwise, register one index with name = `--name` or dirname.
///
/// Test: `cargo run -- index --force` against a healthy daemon prints the
/// registration line then drives the SSE progress bar. With a yaml at
/// `<path>/trusty-search.yaml`, it iterates each declared name.
///
/// `timeout_secs` is forwarded to the SSE stream reader; 0 = no limit.
pub async fn handle_index(
    path: Option<std::path::PathBuf>,
    name: Option<String>,
    force: bool,
    timeout_secs: u64,
) -> Result<()> {
    let cwd = std::env::current_dir().unwrap_or_default();
    let project_path = path.unwrap_or(cwd);

    // 0. Auto-start the daemon if needed. `index` is useless without it,
    //    so we proactively boot it rather than dump a confusing connection
    //    error on the user.
    crate::commands::daemon_guard::ensure_daemon_running_or_exit(&daemon_base_url()).await;

    // 1. Repo-level config detection. `trusty-search.yaml` at the project root
    //    declares one or more named indexes; when present it overrides the
    //    `--name` flag and we register each declared slice in turn.
    match RepoConfig::load(&project_path) {
        Ok(Some(cfg)) => {
            println!(
                "{} loaded {} ({} index{} declared)",
                "→".cyan(),
                CONFIG_FILENAME.bold(),
                cfg.indexes.len(),
                if cfg.indexes.len() == 1 { "" } else { "es" },
            );
            if name.is_some() {
                eprintln!(
                    "{} --name is ignored when {} is present",
                    "ℹ".yellow(),
                    CONFIG_FILENAME
                );
            }
            for idx in &cfg.indexes {
                index_one(&idx.name, &project_path, force, timeout_secs).await?;
            }
            return Ok(());
        }
        Ok(None) => {
            // No config; fall through to single-index path.
        }
        Err(e) => {
            eprintln!(
                "{} could not parse {}: {e}",
                "✗".red(),
                CONFIG_FILENAME.bold()
            );
            std::process::exit(1);
        }
    }

    let index_name = name.unwrap_or_else(|| {
        project_path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned()
    });
    index_one(&index_name, &project_path, force, timeout_secs).await
}

/// Register one named index and run a reindex against it.
///
/// Why: extracted so both the single-index and yaml-multi-index paths share
/// exactly the same registration + reindex sequence (and error handling).
/// What: idempotent `POST /indexes` followed by reindex (or force-reindex).
/// Test: covered indirectly by `handle_index` tests above.
async fn index_one(
    index_name: &str,
    project_path: &std::path::Path,
    force: bool,
    timeout_secs: u64,
) -> Result<()> {
    let (created, daemon_reachable) =
        match register_index_with_daemon(index_name, project_path).await {
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

    if force {
        run_reindex_force(index_name, project_path, timeout_secs).await?;
    } else {
        run_reindex(index_name, project_path, timeout_secs).await?;
    }
    Ok(())
}
