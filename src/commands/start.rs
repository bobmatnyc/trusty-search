//! Handler for `trusty-search start` — boots the HTTP daemon.

use anyhow::Result;
use colored::Colorize;

/// Build a shared `FastEmbedder` for every index registered during the
/// daemon's lifetime.
///
/// Why (Bug A fix): without this, `create_index_handler` constructs a BM25-only
/// `CodeIndexer` and the HNSW lane silently contributes nothing — the symptom
/// seen in the 115k-chunk benchmark where every result returned
/// `match_reason: "bm25"`.
async fn build_embedder() -> Option<std::sync::Arc<dyn crate::core::Embedder>> {
    match crate::core::FastEmbedder::new().await {
        Ok(e) => Some(std::sync::Arc::new(e)),
        Err(e) => {
            tracing::warn!("FastEmbedder init failed ({e}); daemon falling back to BM25-only mode");
            None
        }
    }
}

/// Why: extracted from `main()`. The boot sequence is intricate (lockfile probe,
/// embedder, app state) and benefits from being its own unit. Facts storage
/// moved to trusty-analyzer (issue #40).
/// What: probes the lockfile fast-path, then constructs `SearchAppState` and
/// hands off to `run_daemon`. Maps `DaemonError::AlreadyRunning` to a friendly
/// exit-1 message.
/// Test: run twice in a row — the second invocation must exit 1 with the
/// "another daemon is already running" message.
pub async fn handle_start(port: u16, foreground: bool) -> Result<()> {
    // `foreground` is currently a no-op: `run_daemon` already runs inline
    // and never forks. The flag is accepted so launchd/systemd plists can
    // declare the supervised contract explicitly in ProgramArguments
    // (see ~/Library/LaunchAgents/com.bobmatnyc.trusty-search.plist).
    // If a background-fork path is ever added, gate it on `!foreground`.
    let _ = foreground;
    // Fast-path: bail before loading the 86 MB embedding model when
    // another daemon is already running.  The lock check is ~1 ms;
    // FastEmbedder::new() can take several seconds on first run.
    //
    // Bug fix (launchd crash-loop): if the lockfile exists and the recorded
    // PID is *alive*, exit 0 — launchd treats any non-zero exit as a crash
    // and re-spawns after ThrottleInterval, producing an infinite loop when
    // the daemon is already running. If the PID is dead (stale lock), fall
    // through to `run_daemon`, whose `acquire_lock` removes the stale file
    // and retries on our behalf.
    if let Some(pid) = crate::service::running_daemon_pid() {
        tracing::info!("daemon already running (pid {pid}), exiting cleanly");
        eprintln!(
            "{} trusty-search daemon already running (pid {pid}); nothing to do",
            "✓".green()
        );
        return Ok(());
    }
    // Rare race: the lock is held but the PID-aliveness check returned None
    // (lockfile may contain garbage or be mid-write by a sibling launch).
    // Fall through to `run_daemon` — its `acquire_lock` will either succeed
    // (lock now free) or return AlreadyRunning, handled below.

    let embedder = build_embedder().await;
    let cfg = crate::service::load_user_config();

    let mut state =
        crate::service::SearchAppState::new(crate::core::registry::IndexRegistry::new())
            .with_local_model(cfg.local_model)
            .with_openrouter_model(cfg.openrouter_model);
    if let Some(e) = embedder {
        state = state.with_embedder(e);
    }
    match crate::service::run_daemon(state, port).await {
        Ok(()) => {}
        Err(crate::service::DaemonError::AlreadyRunning(p)) => {
            // `acquire_lock` returns AlreadyRunning only after confirming the
            // recorded PID is alive (it removes stale lockfiles automatically).
            // Exit 0 so launchd does not treat this as a crash and re-spawn.
            tracing::info!(
                "daemon already running (lock at {}), exiting cleanly",
                p.display()
            );
            eprintln!(
                "{} trusty-search daemon already running (lock at {}); nothing to do",
                "✓".green(),
                p.display()
            );
            return Ok(());
        }
        Err(e) => {
            eprintln!("{} daemon failed: {e}", "✗".red());
            std::process::exit(1);
        }
    }
    Ok(())
}
