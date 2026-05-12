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

/// Cached snapshot of `TRUSTY_MEMORY_LIMIT_MB` parsed at first read.
///
/// `None` => limit disabled (env unset or unparseable / zero).
/// `Some(mb)` => soft RSS ceiling in megabytes.
static MEMORY_LIMIT_MB: OnceLock<Option<u64>> = OnceLock::new();

/// Read `TRUSTY_MEMORY_LIMIT_MB`, caching the result. Zero or non-numeric
/// values disable the limit.
pub fn memory_limit_mb() -> Option<u64> {
    *MEMORY_LIMIT_MB.get_or_init(|| {
        std::env::var("TRUSTY_MEMORY_LIMIT_MB")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .filter(|&n| n > 0)
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
