//! Handler for `trusty-search start` — boots the HTTP daemon.

use anyhow::Result;
use colored::Colorize;
use std::sync::Arc;

use crate::core::registry::{IndexHandle, IndexId};
use crate::service::persistence::load_index_registry;
use crate::service::persistence_loader::build_indexer_with_persisted_state;
use crate::service::SearchAppState;

/// Restore every index recorded in `indexes.toml` by re-registering it on the
/// in-memory registry. For each entry we attempt to load the persisted HNSW
/// snapshot and chunk corpus from disk so the index comes back warm (no
/// re-indexing required).
///
/// Why (issue #85): before this hook, the daemon had no way to remember
/// which projects were registered — every restart required the user to run
/// `trusty-search index <path>` again. Now the registry is durable and
/// HNSW + chunks are restored automatically.
/// What: iterates registry entries, skips any that the in-memory registry
/// already has (idempotent — `create_index` may have raced ahead), then
/// constructs an `IndexHandle` via the shared `build_indexer_with_persisted_state`
/// helper.
/// Test: integration test in `tests/integration_tests.rs` that writes a
/// registry file, calls this hook, and asserts the registry list matches.
async fn restore_indexes(state: &SearchAppState, embedder: &Arc<dyn crate::core::Embedder>) {
    let entries = match load_index_registry() {
        Ok(e) => e,
        Err(e) => {
            tracing::warn!("could not read indexes.toml at startup: {e}");
            return;
        }
    };
    if entries.is_empty() {
        return;
    }
    tracing::info!(
        "warm-boot: restoring {} index registration(s) from indexes.toml",
        entries.len()
    );
    for entry in entries {
        let id = IndexId::new(entry.id.clone());
        if state.registry.get(&id).is_some() {
            // A live create_index handler beat us to it — skip.
            continue;
        }
        let mut indexer =
            build_indexer_with_persisted_state(&entry.id, entry.root_path.clone(), embedder).await;
        // Restore per-index filters and domain vocabulary from indexes.toml.
        // Resolve `include_paths` to absolute under `root_path` so the reindex
        // walker can prune without per-call path arithmetic. `.` and empty
        // entries collapse to "walk the whole root".
        let include_paths: Vec<std::path::PathBuf> = entry
            .include_paths
            .iter()
            .filter(|p| !p.trim().is_empty() && p.trim() != ".")
            .map(|p| entry.root_path.join(p.trim()))
            .collect();
        let extensions: Vec<String> = entry
            .extensions
            .iter()
            .map(|e| e.trim_start_matches('.').to_string())
            .filter(|e| !e.is_empty())
            .collect();
        indexer.set_domain_terms(entry.domain_terms.clone());
        let handle = IndexHandle {
            id: id.clone(),
            indexer: Arc::new(tokio::sync::RwLock::new(indexer)),
            root_path: entry.root_path,
            include_paths,
            exclude_globs: entry.exclude_globs,
            extensions,
            domain_terms: entry.domain_terms,
        };
        state.registry.register(handle);
    }
}

/// Build a shared `FastEmbedder` for every index registered during the
/// daemon's lifetime.
///
/// Why (Bug A fix): without this, `create_index_handler` constructs a BM25-only
/// `CodeIndexer` and the HNSW lane silently contributes nothing — the symptom
/// seen in the 115k-chunk benchmark where every result returned
/// `match_reason: "bm25"`.
///
/// Why (blocking init): previously a failure here returned `None` and the
/// daemon continued in BM25-only mode without any visible signal — operators
/// only noticed when every search returned `match_reason: "bm25"` and an
/// entire 17k-file repo "indexed" in 12 seconds (no ONNX work happened).
/// Now: success is logged at INFO with the embedding dimension so operators
/// can confirm the model loaded; failure is logged at ERROR and propagated
/// so the daemon exits non-zero rather than silently degrading.
/// Test: run `trusty-search start` with `RUST_LOG=info` — the log MUST
/// contain `embedder initialized: dim=384` before any HTTP request is
/// accepted. Force a failure (e.g. delete the model cache while offline)
/// and the daemon must exit non-zero, not start in BM25 mode.
async fn build_embedder() -> Result<std::sync::Arc<dyn crate::core::Embedder>> {
    let embedder = crate::core::FastEmbedder::new().await.map_err(|e| {
        tracing::error!("FastEmbedder init failed: {e:#}");
        anyhow::anyhow!("FastEmbedder init failed: {e}")
    })?;
    let dim = <crate::core::FastEmbedder as crate::core::Embedder>::dimension(&embedder);
    let provider = embedder.provider();
    let metal_hint = match provider {
        trusty_embedder::ExecutionProvider::CoreML => " (Metal GPU / ANE)",
        trusty_embedder::ExecutionProvider::Cuda => " (CUDA GPU)",
        trusty_embedder::ExecutionProvider::Cpu => "",
    };
    tracing::info!(
        "embedder initialized: model=AllMiniLML6V2(Q) dim={dim} provider={provider}{metal_hint}"
    );
    Ok(std::sync::Arc::new(embedder))
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
    // Persist any memory-limit env vars set in the current shell so that
    // launchd restarts (which run without the user's shell environment) pick
    // them up via `daemon.env`. This runs before the fast-path check so the
    // file is always refreshed when `start` is explicitly invoked.
    crate::service::save_daemon_env();

    // Source daemon.env into the current environment (env var > file > default).
    // This is primarily for the first `start` after upgrading, when the file
    // may already exist from a previous run.
    crate::service::load_daemon_env();

    // Auto-tune memory caps based on detected system RAM (issue: memory-tier
    // autosizing). Precedence: explicit env var > daemon.env (just loaded)
    // > tier default. `MemoryPolicy::detect()` writes the resolved values
    // back into the process env so existing readers (indexer, bm25,
    // symbol_graph, memguard, store) pick them up automatically.
    //
    // SAFETY: invoked before tokio spawns any indexing workers — env mutation
    // here is on the runtime's main worker thread but no other thread is
    // reading these vars yet. Same invariant `load_daemon_env` relies on.
    let policy = crate::core::MemoryPolicy::detect();

    // Hard 16 GB minimum: trusty-search is designed for developer workstations.
    // Machines with less than 16 GB cannot safely index large codebases —
    // ONNX model load, HNSW resident memory, and indexing batches will OOM
    // under realistic workloads. Exit before binding any port or loading
    // the embedder so the operator gets a clear, actionable error.
    const MIN_RAM_MB: u64 = 16 * 1024;
    if policy.total_ram_mb < MIN_RAM_MB {
        eprintln!(
            "error: trusty-search requires at least 16 GB of RAM.\n\
             Detected: {} MB ({:.1} GB)\n\
             Indexing large codebases on machines with less memory is not supported.",
            policy.total_ram_mb,
            policy.total_ram_mb as f64 / 1024.0
        );
        std::process::exit(1);
    }

    policy.log_summary();
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

    // Issue #81: detect orphan daemons whose PIDs are NOT recorded in the
    // lockfile (e.g. a previous `start` whose lockfile was deleted or
    // overwritten by a launchd respawn). These orphans keep their HNSW
    // indexes resident — we observed a 73 GB RSS orphan daemon left running
    // when `stop` only knew about the lockfile PID. Reap them now so we
    // don't end up with two daemons fighting over `bind_with_auto_port`.
    let orphans = crate::commands::stop::find_daemon_pids();
    if !orphans.is_empty() {
        tracing::warn!(
            "found {} existing trusty-search daemon process(es) not tracked by lockfile: {:?} — terminating before start",
            orphans.len(),
            orphans
        );
        eprintln!(
            "{} found {} existing trusty-search daemon process(es) not tracked by lockfile — stopping them first",
            "⚠".yellow(),
            orphans.len()
        );
        for pid in &orphans {
            #[cfg(unix)]
            unsafe {
                libc::kill(*pid as libc::pid_t, libc::SIGTERM);
            }
        }
        // Give them 3 s to exit cleanly, then SIGKILL stragglers.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3);
        loop {
            std::thread::sleep(std::time::Duration::from_millis(100));
            #[cfg(unix)]
            let any_alive = orphans
                .iter()
                .any(|p| unsafe { libc::kill(*p as libc::pid_t, 0) } == 0);
            #[cfg(not(unix))]
            let any_alive = false;
            if !any_alive || std::time::Instant::now() >= deadline {
                break;
            }
        }
        #[cfg(unix)]
        for pid in &orphans {
            if unsafe { libc::kill(*pid as libc::pid_t, 0) } == 0 {
                tracing::warn!("orphan pid {pid} ignored SIGTERM — sending SIGKILL");
                unsafe {
                    libc::kill(*pid as libc::pid_t, libc::SIGKILL);
                }
            }
        }
        // Clear stale lock/port files left behind by the killed orphans.
        if let Ok(lock) = crate::service::daemon_lock_path() {
            let _ = std::fs::remove_file(&lock);
        }
        if let Some(port) = crate::daemon_port_path() {
            let _ = std::fs::remove_file(&port);
        }
    }

    // Why (v0.3.12 fix — deferred embedder init): previously `build_embedder()`
    // was awaited before the HTTP listener bound, so the daemon's port stayed
    // closed for 15–30 s on first run while ONNX/CoreML loaded the model.
    // That blew past the 10 s readiness budget in `daemon_guard.rs` and made
    // `trusty-search index` think the daemon had failed to start. Now: we
    // construct the `SearchAppState` immediately, kick off model loading on
    // a background task, and let `run_daemon` bind the HTTP port right away.
    // Handlers that need the embedder return `503 Service Unavailable` until
    // `state.install_embedder()` flips the watch channel.
    let cfg = crate::service::load_user_config();

    let state = crate::service::SearchAppState::new(crate::core::registry::IndexRegistry::new())
        .with_local_model(cfg.local_model)
        .with_openrouter_model(cfg.openrouter_model)
        .with_openrouter_api_key(cfg.openrouter_api_key);

    // Spawn embedder load on a background task; the daemon's HTTP server
    // starts serving requests in parallel. On success, `install_embedder`
    // populates the slot and flips the readiness watch so the next inbound
    // request transitions out of the "initializing" branch. On failure, we
    // log loudly but leave the daemon running in BM25-only mode — operators
    // can `/health`-check `embedder: "unavailable"` and intervene. We can't
    // exit the process here without racing the HTTP server's shutdown path.
    let install_state = state.clone();
    tokio::spawn(async move {
        match build_embedder().await {
            Ok(embedder) => {
                install_state.install_embedder(Arc::clone(&embedder)).await;
                tracing::info!("embedder ready — vector lane online");
                // Issue #85: now that the embedder is ready, restore every
                // index recorded in `indexes.toml`. We do this after embedder
                // init so the restored indexes get a fully-wired hybrid
                // pipeline (HNSW vectors + BM25), not a BM25-only fallback.
                restore_indexes(&install_state, &embedder).await;
            }
            Err(e) => {
                tracing::error!(
                    "embedder failed to initialize: {e:#} — daemon will continue in BM25-only mode"
                );
                eprintln!(
                    "{} embedder failed to initialize: {e}\n\
                     Daemon is up but running BM25-only. Check the model cache at \
                     ~/Library/Caches/trusty-search/models/ and network access.",
                    "✗".red()
                );
            }
        }
    });
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
