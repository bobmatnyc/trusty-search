//! Handler for `trusty-search start` — boots the HTTP daemon.

use anyhow::Result;
use colored::Colorize;

/// Build the canonical `FactStore` next to the daemon lockfile.
///
/// Why: facts persist across daemon restarts and are scoped per-machine
/// (single install). Falling back to `None` keeps the daemon usable if the
/// data dir is read-only — `/facts` endpoints will return 503.
fn open_facts_store() -> Option<trusty_search_core::FactStore> {
    let dir = dirs::data_local_dir()?.join("trusty-search");
    if let Err(e) = std::fs::create_dir_all(&dir) {
        tracing::warn!("could not create facts dir {}: {e}", dir.display());
        return None;
    }
    match trusty_search_core::FactStore::open(&dir.join("facts.redb")) {
        Ok(s) => Some(s),
        Err(e) => {
            tracing::warn!("could not open facts store: {e}");
            None
        }
    }
}

/// Build a shared `FastEmbedder` for every index registered during the
/// daemon's lifetime.
///
/// Why (Bug A fix): without this, `create_index_handler` constructs a BM25-only
/// `CodeIndexer` and the HNSW lane silently contributes nothing — the symptom
/// seen in the 115k-chunk benchmark where every result returned
/// `match_reason: "bm25"`.
async fn build_embedder() -> Option<std::sync::Arc<dyn trusty_search_core::Embedder>> {
    match trusty_search_core::FastEmbedder::new().await {
        Ok(e) => Some(std::sync::Arc::new(e)),
        Err(e) => {
            tracing::warn!(
                "FastEmbedder init failed ({e}); daemon falling back to BM25-only mode"
            );
            None
        }
    }
}

/// Why: extracted from `main()`. The boot sequence is intricate (lockfile probe,
/// facts store, embedder, app state) and benefits from being its own unit.
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
    if let Some(lock_path) = trusty_search_service::is_already_running() {
        eprintln!(
            "{} another trusty-search daemon is already running (lock at {})",
            "✗".red(),
            lock_path.display()
        );
        std::process::exit(1);
    }

    let facts = open_facts_store();
    let embedder = build_embedder().await;

    let mut state = trusty_search_service::SearchAppState::new(
        trusty_search_core::registry::IndexRegistry::new(),
        facts,
    );
    if let Some(e) = embedder {
        state = state.with_embedder(e);
    }
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
    Ok(())
}
