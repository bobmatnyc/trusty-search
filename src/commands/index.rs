//! Handler for `trusty-search index` (register + reindex in one step).
//!
//! Why: register-then-reindex is the primary onboarding flow. When a repo
//! contains a `trusty-search.yaml` file (issue: repo-level config), we
//! transparently fan out into one register+reindex pass per declared index
//! so a single `trusty-search index` command can populate multiple named
//! slices (e.g. `duetto-api` and `duetto-ui`).

use super::daemon_utils::daemon_base_url;
use super::reindex_engine::{
    register_index_with_daemon, register_index_with_daemon_filtered, run_reindex,
    run_reindex_force, RegisterFilters,
};
use crate::core::repo_config::{language_to_exts, IndexConfig, RepoConfig, CONFIG_FILENAME};
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
    crate::commands::daemon_guard::ensure_daemon_running_or_exit(&daemon_base_url()).await?;

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
                let filters = filters_from_index_config(idx);
                index_one_with_filters(&idx.name, &project_path, force, timeout_secs, &filters)
                    .await?;
            }
            return Ok(());
        }
        Ok(None) => {
            // No config; fall through to single-index path.
        }
        Err(e) => {
            anyhow::bail!("could not parse {}: {e}", CONFIG_FILENAME);
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

/// Map a parsed `IndexConfig` to the daemon-bound `RegisterFilters`.
///
/// Why: `IndexConfig` uses ergonomic YAML names (`paths`, `exclude`,
/// `languages`); the daemon expects the resolved shape (`include_paths`,
/// `exclude_globs`, `extensions`, `domain_terms`). One place to translate.
/// What: clones `paths` and `exclude` verbatim, expands `languages` to file
/// extensions via [`language_to_exts`], passes `domain_terms` through.
/// Test: see `tests::filters_from_index_config_translates_languages` in
/// `src/core/repo_config.rs`.
pub(crate) fn filters_from_index_config(idx: &IndexConfig) -> RegisterFilters {
    let mut extensions: Vec<String> = Vec::new();
    for lang in &idx.languages {
        for e in language_to_exts(lang) {
            extensions.push((*e).to_string());
        }
    }
    extensions.sort();
    extensions.dedup();
    RegisterFilters {
        include_paths: idx.paths.clone(),
        exclude_globs: idx.exclude.clone(),
        extensions,
        domain_terms: idx.domain_terms.clone(),
    }
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
    index_one_with_filters(
        index_name,
        project_path,
        force,
        timeout_secs,
        &RegisterFilters::default(),
    )
    .await
}

/// Filter-aware version of `index_one`. The yaml multi-index path uses this
/// to forward per-index `paths`/`exclude`/`languages`/`domain_terms` to the
/// daemon.
async fn index_one_with_filters(
    index_name: &str,
    project_path: &std::path::Path,
    force: bool,
    timeout_secs: u64,
    filters: &RegisterFilters,
) -> Result<()> {
    let result = if filters.include_paths.is_empty()
        && filters.exclude_globs.is_empty()
        && filters.extensions.is_empty()
        && filters.domain_terms.is_empty()
    {
        register_index_with_daemon(index_name, project_path).await
    } else {
        register_index_with_daemon_filtered(index_name, project_path, filters).await
    };
    let (created, daemon_reachable) = result?;
    if !daemon_reachable {
        anyhow::bail!(
            "Daemon not reachable at {}. Start it with `trusty-search start`.",
            daemon_base_url(),
        );
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
