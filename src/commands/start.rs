//! Handler for `trusty-search start` — boots the HTTP daemon.

use anyhow::Result;
use colored::Colorize;
use std::sync::Arc;
use std::time::Duration;

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
            path_filter: entry.path_filter,
            context_embedding: Arc::new(tokio::sync::RwLock::new(None)),
            context_summary: Arc::new(tokio::sync::RwLock::new(None)),
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
    tune_batch_size_for_provider(provider);
    Ok(std::sync::Arc::new(embedder))
}

/// When the resolved execution provider is a GPU, retune `TRUSTY_MAX_BATCH_SIZE`
/// upward so ONNX dispatches use the GPU efficiently.
///
/// Why (issue #113): the CPU batch-size formula (≈55 MB transient ORT arena
/// per batch slot, clamped to `[32, 512]`) is sized to keep the *CPU* ORT
/// path under the soft RSS cap. On a CUDA GPU the arena lives in device
/// memory and the per-slot transient is much smaller, so the CPU-tuned
/// default (e.g. 128 on Medium tier) starves the GPU — most wall-clock is
/// spent on host↔device round-trips instead of compute. Bumping the default
/// to 512 cuts host↔device transitions ~4× and is what turns a 5-hour
/// reindex into a ~30-minute one on a Tesla T4 (40 k files, INT8 model).
/// What: if a GPU EP is active AND the operator did NOT opt out by setting
/// `TRUSTY_MAX_BATCH_SIZE_EXPLICIT=1`, retune `TRUSTY_MAX_BATCH_SIZE` to 512.
/// Test: on a CUDA-enabled binary the startup log shows
/// `gpu_batch_tuning: TRUSTY_MAX_BATCH_SIZE=512 (was N)`; running with
/// `TRUSTY_MAX_BATCH_SIZE_EXPLICIT=1 TRUSTY_MAX_BATCH_SIZE=256 trusty-search start`
/// keeps 256.
fn tune_batch_size_for_provider(provider: trusty_embedder::ExecutionProvider) {
    const GPU_BATCH_DEFAULT: usize = 512;

    let is_gpu = matches!(
        provider,
        trusty_embedder::ExecutionProvider::Cuda | trusty_embedder::ExecutionProvider::CoreML
    );
    if !is_gpu {
        return;
    }

    if std::env::var("TRUSTY_MAX_BATCH_SIZE_EXPLICIT")
        .map(|v| v == "1")
        .unwrap_or(false)
    {
        tracing::info!(
            "gpu_batch_tuning: TRUSTY_MAX_BATCH_SIZE_EXPLICIT=1 set, leaving batch size unchanged"
        );
        return;
    }

    let current = std::env::var("TRUSTY_MAX_BATCH_SIZE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(128);
    if current >= GPU_BATCH_DEFAULT {
        return;
    }

    // SAFETY: invoked on the main thread before any indexing worker has
    // started. Same invariant `MemoryPolicy::apply_to_env` relies on.
    unsafe {
        std::env::set_var("TRUSTY_MAX_BATCH_SIZE", GPU_BATCH_DEFAULT.to_string());
    }
    tracing::info!(
        "gpu_batch_tuning: provider={provider} → TRUSTY_MAX_BATCH_SIZE={GPU_BATCH_DEFAULT} (was {current}); \
         set TRUSTY_MAX_BATCH_SIZE_EXPLICIT=1 to keep your value"
    );
}

/// Why: extracted from `main()`. The boot sequence is intricate (lockfile probe,
/// embedder, app state) and benefits from being its own unit. Facts storage
/// moved to trusty-analyzer (issue #40).
/// What: probes the lockfile fast-path, then constructs `SearchAppState` and
/// hands off to `run_daemon`. Maps `DaemonError::AlreadyRunning` to a friendly
/// exit-1 message.
/// Test: run twice in a row — the second invocation must exit 1 with the
/// "another daemon is already running" message.
pub async fn handle_start(port: u16, foreground: bool, device: &str) -> Result<()> {
    // Background self-spawn path: when invoked without `--foreground`, fork a
    // detached copy of ourselves with `--foreground` and return immediately.
    //
    // Why: prior to this, `run_daemon().await` blocked the current process
    // forever and held the controlling terminal. Running `trusty-search start`
    // from inside a tmux pane (e.g. via `make patch`) meant a SIGHUP on pane
    // close would kill the daemon and tear down the whole tmux session. By
    // detaching stdio to `/dev/null` and returning, we let the caller (shell,
    // make, tmux) continue without owning the daemon's lifetime.
    //
    // launchd/systemd/Docker users (and `trusty-search service`) always pass
    // `--foreground` and stay on the inline path so the supervisor can manage
    // the process lifecycle.
    if !foreground {
        // Validate `--device` here too so a typo doesn't silently spawn a
        // child that will then immediately exit non-zero.
        match device {
            "auto" | "cpu" | "gpu" => {}
            other => {
                anyhow::bail!("invalid --device value '{other}'; expected one of: auto, cpu, gpu")
            }
        }
        let exe = std::env::current_exe()
            .map_err(|e| anyhow::anyhow!("could not resolve current_exe: {e}"))?;
        let mut cmd = std::process::Command::new(&exe);
        cmd.arg("start")
            .arg("--foreground")
            .arg("--port")
            .arg(port.to_string())
            .arg("--device")
            .arg(device)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null());
        let child = cmd
            .spawn()
            .map_err(|e| anyhow::anyhow!("could not spawn detached daemon: {e}"))?;
        let pid = child.id();
        eprintln!(
            "{} Daemon starting in background (pid {pid}). Run `trusty-search status` to verify readiness.",
            "✓".green()
        );
        return Ok(());
    }

    // Translate the `--device` CLI flag into the `TRUSTY_DEVICE` env var that
    // `trusty-embedder` reads at session-init time. An explicit shell-level
    // `TRUSTY_DEVICE` always wins so launchd/systemd unit files can set the
    // policy in one place; the CLI flag only fills in when no env value is
    // already present. `auto` is a no-op (existing behaviour: try GPU EPs in
    // priority order, fall back to CPU on failure).
    //
    // SAFETY: invoked on the main thread before tokio spawns any workers.
    // Same invariant `MemoryPolicy::apply_to_env` relies on.
    if std::env::var_os("TRUSTY_DEVICE").is_none() {
        match device {
            "cpu" | "gpu" => unsafe {
                std::env::set_var("TRUSTY_DEVICE", device);
            },
            "auto" => {}
            other => {
                anyhow::bail!("invalid --device value '{other}'; expected one of: auto, cpu, gpu")
            }
        }
    }

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
        anyhow::bail!(
            "trusty-search requires at least 16 GB of RAM.\n\
             Detected: {} MB ({:.1} GB)\n\
             Indexing large codebases on machines with less memory is not supported.",
            policy.total_ram_mb,
            policy.total_ram_mb as f64 / 1024.0
        );
    }

    policy.log_summary();
    // We arrive here only when `foreground == true` (the background path
    // returned earlier via self-spawn). The flag is consumed by the spawn
    // branch above, so it is unused from this point onward.
    let _ = foreground;
    // Fast-path: bail before loading the 86 MB embedding model when
    // another daemon is already running.  The lock check is ~1 ms;
    // FastEmbedder::new() can take several seconds on first run.
    //
    // Issue #87 (Option A): if the lockfile exists and the recorded PID is
    // *alive*, print an actionable warning and exit 1 so the caller knows
    // they must stop the daemon first before installing a new binary.
    // Replacing the binary on-disk while the daemon is running causes macOS
    // 26.3+ to SIGKILL the process with "Code Signature Invalid".
    //
    // NOTE: this is intentionally exit(1), NOT exit(0).  The previous exit(0)
    // was kept for launchd's crash-loop guard; that guard is now handled by
    // the `DaemonError::AlreadyRunning` branch below (which still exits 0)
    // because launchd only calls `start` without a prior `stop`.  A direct
    // user invocation of `trusty-search start` when the daemon is already up
    // should be a visible, actionable error.
    //
    // If the PID is dead (stale lock), fall through to `run_daemon`, whose
    // `acquire_lock` removes the stale file and retries on our behalf.
    if let Some(pid) = crate::service::running_daemon_pid() {
        tracing::warn!("daemon already running (pid {pid}); refusing to start a second instance");
        anyhow::bail!(
            "Daemon already running (pid {pid}).\n\
             Stop it first with `trusty-search stop`, then re-run `trusty-search start`.\n\
             Replacing the binary while the daemon is running causes macOS to SIGKILL \
             the process (Code Signature Invalid)."
        );
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
        #[cfg(unix)]
        for pid in &orphans {
            let _ = nix::sys::signal::kill(
                nix::unistd::Pid::from_raw(*pid as i32),
                nix::sys::signal::Signal::SIGTERM,
            );
        }
        // Give them 3 s to exit cleanly, then SIGKILL stragglers.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3);
        loop {
            std::thread::sleep(std::time::Duration::from_millis(100));
            #[cfg(unix)]
            let any_alive = orphans.iter().any(|p| {
                nix::sys::signal::kill(nix::unistd::Pid::from_raw(*p as i32), None).is_ok()
            });
            #[cfg(not(unix))]
            let any_alive = false;
            if !any_alive || std::time::Instant::now() >= deadline {
                break;
            }
        }
        #[cfg(unix)]
        for pid in &orphans {
            if nix::sys::signal::kill(nix::unistd::Pid::from_raw(*pid as i32), None).is_ok() {
                tracing::warn!("orphan pid {pid} ignored SIGTERM — sending SIGKILL");
                let _ = nix::sys::signal::kill(
                    nix::unistd::Pid::from_raw(*pid as i32),
                    nix::sys::signal::Signal::SIGKILL,
                );
            }
        }
        // Clear stale lock/port files left behind by the killed orphans.
        if let Ok(lock) = crate::service::daemon_lock_path() {
            let _ = std::fs::remove_file(&lock);
        }
        if let Some(port) = super::daemon_utils::daemon_port_path() {
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
    // Why (issue #121): on Intel Xeon AVX-512 hosts, `ort-2.0.0-rc.12`'s CPU
    // session init can block the calling thread indefinitely with no error,
    // no timeout, and no panic. Running `build_embedder().await` on a normal
    // Tokio task means the blocking native code parks a Tokio worker thread
    // forever, and the surrounding `tokio::spawn` future never makes progress.
    //
    // Fix has two parts:
    //   1. Run the ORT/fastembed init on a dedicated `spawn_blocking` thread
    //      so a hung init only consumes one blocking-pool thread instead of
    //      starving the async executor.
    //   2. Race the init against `TRUSTY_EMBEDDER_INIT_TIMEOUT_SECS` (default
    //      60 s). If init does not complete in time, transition the embedder
    //      to an `"error"` state (visible via `/health`) so callers stop
    //      polling and `POST /indexes` fails fast instead of dangling forever.
    //
    // The dedicated thread is intentionally NOT joined on timeout — the ORT
    // call is hung in native code we cannot abort safely, so we leak the
    // thread rather than risk an unsound abort. The daemon keeps running in
    // BM25-only mode; the operator is expected to restart with a working ORT
    // configuration after seeing the error.
    let init_timeout_secs: u64 = std::env::var("TRUSTY_EMBEDDER_INIT_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(60);

    let install_state = state.clone();
    tokio::spawn(async move {
        let runtime_handle = tokio::runtime::Handle::current();
        let init_handle = tokio::task::spawn_blocking(move || {
            // `build_embedder` is async only because `FastEmbedder::new()` is;
            // the heavy lifting (ONNX session init) happens in synchronous
            // native code inside ORT, so isolating it on a blocking thread is
            // both correct and necessary.
            runtime_handle.block_on(build_embedder())
        });

        let init_result = tokio::time::timeout(
            Duration::from_secs(init_timeout_secs),
            init_handle,
        )
        .await;

        match init_result {
            Ok(Ok(Ok(embedder))) => {
                install_state.install_embedder(Arc::clone(&embedder)).await;
                tracing::info!("embedder ready — vector lane online");
                // Issue #85: restore every index recorded in `indexes.toml`
                // now that we have a fully-wired hybrid pipeline.
                restore_indexes(&install_state, &embedder).await;
            }
            Ok(Ok(Err(e))) => {
                let msg = format!("FastEmbedder init failed: {e}");
                install_state.install_embedder_error(msg.clone()).await;
                eprintln!(
                    "{} embedder failed to initialize: {e}\n\
                     Daemon is up but running BM25-only. Check the model cache at \
                     ~/Library/Caches/trusty-search/models/ and network access.",
                    "✗".red()
                );
            }
            Ok(Err(join_err)) => {
                let msg = format!("embedder init task panicked: {join_err}");
                install_state.install_embedder_error(msg).await;
            }
            Err(_elapsed) => {
                // Timeout fired. The blocking init thread is intentionally
                // leaked — we cannot safely cancel native ORT code that is
                // already stuck. The daemon keeps serving HTTP in
                // BM25-only mode and surfaces the error via `/health`.
                tracing::error!(
                    "embedder init timed out after {init_timeout_secs}s — \
                     ONNX session did not start (try \
                     TRUSTY_EMBEDDER_INIT_TIMEOUT_SECS=120 or check ORT \
                     compatibility, see GitHub #121)"
                );
                let msg = format!("init timed out after {init_timeout_secs}s");
                install_state.install_embedder_error(msg).await;
                eprintln!(
                    "{} embedder init timed out after {init_timeout_secs}s — \
                     see GitHub #121. Daemon is up in BM25-only mode.",
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
        Err(e) => anyhow::bail!("daemon failed: {e}"),
    }
    Ok(())
}
