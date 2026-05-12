//! Handler for `trusty-search convert` — migrate mcp-vector-search projects.

use crate::{
    convert_one, daemon_base_url, find_all_mvs_configs, find_mvs_config, parse_mvs_config,
    print_convert_line, ConvertResult, ConvertStatus, ConvertTarget,
};
use anyhow::Result;
use colored::Colorize;

/// Why: the convert flow has two distinct sub-cases (single project / all
/// projects) plus dry-run handling and bounded-concurrency fan-out, totalling
/// ~150 lines that bloated `main()`. Extracted as-is to keep behaviour
/// identical while shrinking the dispatcher.
/// What: dispatches on `ConvertTarget`; `Project` converts the mcp-vector-search
/// config found by walking parents of `cwd`; `All` walks `$HOME` for every
/// `.mcp-vector-search/config.json` and converts them in parallel (semaphore
/// bounded by `concurrency`).
/// Test: `cargo run -- convert project --dry-run` from inside an
/// mcp-vector-search repo prints the would-convert line; `convert all
/// --dry-run` enumerates every detected project.
pub async fn handle_convert(
    target: ConvertTarget,
    dry_run: bool,
    concurrency: usize,
) -> Result<()> {
    let base = daemon_base_url();
    crate::commands::daemon_guard::ensure_daemon_running_or_exit(&base).await?;

    match target {
        ConvertTarget::Project => handle_convert_project(dry_run, &base).await,
        ConvertTarget::All => handle_convert_all(dry_run, concurrency, base).await,
    }
}

/// Convert the mcp-vector-search project rooted at (or above) the cwd.
async fn handle_convert_project(dry_run: bool, base: &str) -> Result<()> {
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
        return Ok(());
    }

    println!(
        "{} Converting '{}' ({})…",
        "⟳".cyan(),
        name.bold(),
        root.display()
    );
    let result = convert_one(root, name, base, false).await;
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
            anyhow::bail!("Conversion failed: {}", msg);
        }
        ConvertStatus::DryRun => unreachable!(),
    }
    Ok(())
}

/// Convert every mcp-vector-search project found under `$HOME`, fanning out
/// with `tokio::task::JoinSet` and bounding concurrency by `concurrency`.
async fn handle_convert_all(dry_run: bool, concurrency: usize, base: String) -> Result<()> {
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
            // Acquire permit inside the task so JoinSet limits concurrency
            // cleanly without us pre-allocating futures that all immediately
            // try to fire.
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

    // Collect-then-sort so output is deterministic instead of racy. For
    // 69 projects this is trivially small.
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
    Ok(())
}
