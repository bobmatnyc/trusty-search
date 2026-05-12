//! Handler for `trusty-search doctor` — 6-check diagnostic + auto-repair.

use super::daemon_utils::daemon_base_url;
use super::doctor_checks::{fix_stale_lock, CheckResult, EmptyIndex};
use super::doctor_pipeline::run_doctor_checks;
use super::reindex_engine::run_reindex;
use anyhow::Result;
use colored::Colorize;

/// Why: extracted from `main()`. The diagnostic runs six independent checks
/// and the `--fix` branch has three sub-fixes (stale lock, empty indexes,
/// missing model). Keeping all of that inline pushed `main()` past
/// cyclomatic-complexity 100; lifting it out gives `doctor` its own scope.
/// What: prints check results, optionally repairs (stale lockfile, empty
/// indexes via `run_reindex`, missing model is informational only), then
/// prints a summary. Exits 1 if any check is an error.
/// Test: `cargo run -- doctor` on a healthy install prints
/// "Everything looks good!"; with a stale lockfile and `--fix`, the lockfile
/// is removed and the summary reports the repair.
pub async fn handle_doctor(fix: bool) -> Result<()> {
    println!("\ntrusty-search doctor\n");
    println!("Checking configuration...\n");

    crate::commands::daemon_guard::ensure_daemon_running_or_exit(&daemon_base_url()).await?;

    let (checks, empty_indexes) = run_doctor_checks().await;

    // Print all checks (index sub-lines were already printed inline by
    // run_doctor_checks, so we skip the index summary line itself to avoid
    // double-printing — it carries the per-index detail).
    for check in &checks {
        check.print();
    }

    let errors = checks.iter().filter(|c| c.is_error()).count();
    let warnings = checks.iter().filter(|c| c.is_warn()).count();

    if fix {
        apply_fixes(&checks, &empty_indexes).await;
    }

    print_summary(errors, warnings, fix);

    if errors > 0 {
        // Non-zero exit code without printing another line (summary already
        // showed the error count). Empty-message bail keeps `main()`'s
        // central ✗ printer silent for this exit path.
        return Err(anyhow::anyhow!(""));
    }
    Ok(())
}

/// Run the three auto-repair branches: stale lockfile removal, reindex of
/// zero-chunk indexes, informational note for missing models.
async fn apply_fixes(checks: &[CheckResult], empty_indexes: &[EmptyIndex]) {
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
            .map(|pid| {
                nix::sys::signal::kill(nix::unistd::Pid::from_raw(pid as i32), None).is_err()
            })
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
        for idx in empty_indexes {
            if idx.root_path.is_empty() {
                eprintln!(
                    "  {} '{}' has no root_path — cannot auto-fix; run `trusty-search index` manually",
                    "⚠".yellow(),
                    idx.name
                );
                continue;
            }
            let root = std::path::PathBuf::from(&idx.root_path);
            println!("  Indexing '{}'...", idx.name);
            // Doctor auto-repair uses the default 600s timeout so very large
            // repos still get a reasonable window without blocking forever.
            match run_reindex(&idx.name, &root, 600).await {
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

/// Print the trailing summary line(s): green "all good", or a count of
/// warnings/errors plus the hint to re-run with `--fix`.
fn print_summary(errors: usize, warnings: usize, fix: bool) {
    println!();
    if errors == 0 && warnings == 0 {
        println!("{}", "Everything looks good!".green().bold());
        return;
    }
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
