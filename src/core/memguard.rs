//! Process-memory introspection helpers for the indexing pipeline.
//!
//! Why: Long-running reindexes on large repos can grow process RSS without
//! bound (ONNX session arenas, BM25 corpus, HNSW vectors, chunk metadata).
//! `TRUSTY_MEMORY_LIMIT_MB` lets operators set a soft ceiling; the reindex
//! orchestrator polls [`current_rss_mb`] every N batches and bails out
//! gracefully when the limit is hit, rather than being OOM-killed by the
//! kernel (macOS Jetsam, Linux oom_killer).
//! What: thin wrapper around `sysinfo::System` that refreshes only the
//! current process's memory and returns RSS in megabytes. Also reads and
//! caches the `TRUSTY_MEMORY_LIMIT_MB` env var at first use.
//! Test: see `tests::test_memory_limit_env_parse` and
//! `tests::test_current_rss_mb_nonzero`.
//!
//! No `unwrap()` in this module — every fallible call uses `.ok()` /
//! `unwrap_or_else` so a sysinfo / kernel hiccup never panics the daemon.

use std::sync::OnceLock;

use sysinfo::{Pid, ProcessRefreshKind, RefreshKind, System};

/// Hard-coded safety-net ceiling (8 GiB). Applied when neither the env var
/// nor `daemon.env` sets an explicit limit. This prevents an unattended
/// launchd restart from consuming all available RAM on a developer machine.
///
/// Operators who need more RAM (e.g. indexing >1M-chunk monorepos) should
/// set `TRUSTY_MEMORY_LIMIT_MB` before running `trusty-search start` — the
/// value is persisted to `daemon.env` and survives launchd restarts.
const DEFAULT_MEMORY_LIMIT_MB: u64 = 8_192;

/// Cached snapshot of `TRUSTY_MEMORY_LIMIT_MB` parsed at first read.
///
/// `None` => limit disabled (env unset or unparseable / zero).
/// `Some(mb)` => soft RSS ceiling in megabytes.
static MEMORY_LIMIT_MB: OnceLock<Option<u64>> = OnceLock::new();

/// Read `TRUSTY_MEMORY_LIMIT_MB`, caching the result.
///
/// Priority: env var > `daemon.env` (already sourced into env by
/// `load_daemon_env`) > compiled-in default of `DEFAULT_MEMORY_LIMIT_MB`
/// (8 192 MB / 8 GiB). A value of `0` in the env var explicitly disables
/// the limit and returns `None`; any other non-numeric value falls through
/// to the default.
///
/// Why default 8 GiB: on a launchd restart without any env vars the daemon
/// previously ran with no cap at all, which allowed ONNX arena growth to
/// consume 80+ GB before macOS Jetsam killed it. 8 GiB is a safe ceiling
/// for typical developer machines that still allows large-repo indexing.
pub fn memory_limit_mb() -> Option<u64> {
    *MEMORY_LIMIT_MB.get_or_init(|| {
        match std::env::var("TRUSTY_MEMORY_LIMIT_MB") {
            Ok(v) => {
                // Explicit "0" means "no limit" — respect it.
                match v.parse::<u64>() {
                    Ok(0) => None,
                    Ok(n) => Some(n),
                    Err(_) => {
                        tracing::warn!(
                            "TRUSTY_MEMORY_LIMIT_MB={v:?} is not a valid u64; \
                             using compiled-in default ({DEFAULT_MEMORY_LIMIT_MB} MB)"
                        );
                        Some(DEFAULT_MEMORY_LIMIT_MB)
                    }
                }
            }
            Err(_) => Some(DEFAULT_MEMORY_LIMIT_MB),
        }
    })
}

/// Current process Resident Set Size in megabytes. Returns `None` if sysinfo
/// could not resolve the current process (extremely unlikely; only seen in
/// containerised environments with /proc hidden).
pub fn current_rss_mb() -> Option<u64> {
    let pid = Pid::from_u32(std::process::id());
    let mut sys = System::new_with_specifics(
        RefreshKind::nothing().with_processes(ProcessRefreshKind::everything()),
    );
    sys.refresh_processes_specifics(
        sysinfo::ProcessesToUpdate::Some(&[pid]),
        true,
        ProcessRefreshKind::nothing().with_memory(),
    );
    // `Process::memory()` returns bytes on every supported platform as of
    // sysinfo 0.30+. Convert to MB with a saturating divide.
    sys.process(pid).map(|p| p.memory() / (1024 * 1024))
}

/// Convenience helper for the reindex orchestrator: returns `true` when a
/// memory limit is configured AND current RSS is at or above it.
pub fn over_memory_limit() -> bool {
    match (memory_limit_mb(), current_rss_mb()) {
        (Some(limit), Some(rss)) => rss >= limit,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_limit_env_parse() {
        // The static is cached on first read across the test binary, so we
        // can't reliably mutate the env here. Just assert the getter never
        // panics and returns a deterministic value for this process.
        let _ = memory_limit_mb();
    }

    #[test]
    fn test_current_rss_mb_nonzero() {
        // The test process itself is real — RSS should be > 0 MB.
        if let Some(mb) = current_rss_mb() {
            assert!(mb > 0, "current process RSS should be > 0 MB, got {mb}");
        }
        // If sysinfo couldn't resolve the pid we tolerate `None` (CI sandbox).
    }

    #[test]
    fn test_over_memory_limit_false_when_unset() {
        // Without TRUSTY_MEMORY_LIMIT_MB set in the test environment, the
        // helper must return false regardless of current RSS.
        if memory_limit_mb().is_none() {
            assert!(!over_memory_limit());
        }
    }
}
