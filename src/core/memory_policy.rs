//! Auto-tuned memory caps based on detected system RAM.
//!
//! Why: Static defaults for `TRUSTY_MAX_CHUNKS`, `TRUSTY_EMBEDDING_CACHE`,
//! `TRUSTY_MAX_BATCH_SIZE`, `TRUSTY_BM25_CORPUS_CAP`, `TRUSTY_MAX_KG_NODES`,
//! and `TRUSTY_MEMORY_LIMIT_MB` cannot fit every host: on an 8 GB laptop they
//! risk OOM; on a 192 GB workstation they're needlessly conservative. This
//! module detects total physical RAM at startup, selects a memory tier, and
//! computes sensible default caps. Env vars always override.
//! What: provides [`MemoryPolicy::detect`] which (1) reads total RAM via
//! platform-specific syscalls (`sysctl hw.memsize` on macOS, `/proc/meminfo`
//! on Linux), (2) classifies into a [`MemoryTier`], (3) starts with the
//! tier's default caps, (4) overrides any field whose env var is set, and
//! (5) writes the resolved values back into the process environment so
//! existing module-level readers (in `indexer.rs`, `bm25.rs`, `symbol_graph.rs`,
//! `memguard.rs`, `store.rs`) pick them up automatically.
//! Test: see the `tests` module — tier selection table, env override behaviour,
//! and a smoke test that RAM detection returns a non-zero value on the host
//! running the test suite.
//!
//! Refactor note (transitional): we set env vars after detection so existing
//! readers don't need to change. Callers may instead read fields from
//! [`MemoryPolicy`] directly, which is the preferred long-term path.

use std::fmt;

/// Hard-coded fallback when RAM detection fails (8 GiB worth of MB). Logged
/// as a warning when used.
const FALLBACK_RAM_MB: u64 = 8 * 1024;

/// Memory tier selected based on total system RAM. The tier picks default
/// caps; env vars override individual fields.
///
/// Note: trusty-search requires at least 16 GB of RAM. The daemon startup
/// path (`commands::start`) hard-exits before reaching tier selection on any
/// host with less than 16 GB, so sub-16 GB tiers are deliberately absent
/// from this enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryTier {
    /// 16–31 GB total RAM (minimum supported configuration).
    Medium,
    /// 32–63 GB total RAM.
    Large,
    /// >= 64 GB total RAM.
    XLarge,
}

impl fmt::Display for MemoryTier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            MemoryTier::Medium => "Medium",
            MemoryTier::Large => "Large",
            MemoryTier::XLarge => "XLarge",
        })
    }
}

impl MemoryTier {
    /// Pick a tier from total RAM in megabytes.
    ///
    /// Why: tier selection drives default caps. The daemon enforces a 16 GB
    /// minimum at startup, so anything < 16 GB should never reach this
    /// function in normal operation. As a defensive fallback (e.g. tests,
    /// library consumers bypassing `commands::start`), values < 16 GB map to
    /// `Medium` so the policy remains well-defined.
    pub fn from_total_ram_mb(total_ram_mb: u64) -> Self {
        // GB boundaries: 16–31 Medium, 32–63 Large, >=64 XLarge.
        // < 16 GB: defensive fallback to Medium (the daemon exits before
        // reaching here on under-spec hosts).
        let gb = total_ram_mb / 1024;
        match gb {
            0..=31 => MemoryTier::Medium,
            32..=63 => MemoryTier::Large,
            _ => MemoryTier::XLarge,
        }
    }

    /// Default caps for this tier.
    fn defaults(self) -> TierDefaults {
        match self {
            MemoryTier::Medium => TierDefaults {
                memory_limit_mb: 4_096,
                max_chunks: 200_000,
                embedding_cache: 5_000,
                max_batch_size: 256,
                bm25_corpus_cap: 100_000,
                max_kg_nodes: 150_000,
            },
            MemoryTier::Large => TierDefaults {
                memory_limit_mb: 8_192,
                max_chunks: 400_000,
                embedding_cache: 10_000,
                max_batch_size: 512,
                bm25_corpus_cap: 200_000,
                max_kg_nodes: 300_000,
            },
            MemoryTier::XLarge => TierDefaults {
                memory_limit_mb: 16_384,
                max_chunks: 800_000,
                embedding_cache: 20_000,
                max_batch_size: 512,
                bm25_corpus_cap: 400_000,
                max_kg_nodes: 500_000,
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct TierDefaults {
    memory_limit_mb: usize,
    max_chunks: usize,
    embedding_cache: usize,
    max_batch_size: usize,
    bm25_corpus_cap: usize,
    max_kg_nodes: usize,
}

/// Resolved memory caps for this daemon process. Constructed by
/// [`MemoryPolicy::detect`].
#[derive(Debug, Clone, Copy)]
pub struct MemoryPolicy {
    pub total_ram_mb: u64,
    pub tier: MemoryTier,
    pub memory_limit_mb: usize,
    pub max_chunks: usize,
    pub embedding_cache: usize,
    pub max_batch_size: usize,
    pub bm25_corpus_cap: usize,
    pub max_kg_nodes: usize,
}

impl MemoryPolicy {
    /// Detect total system RAM, pick a tier, apply env-var overrides, and
    /// return the resolved policy.
    ///
    /// Why: single source of truth for memory caps at daemon startup.
    /// What: runs platform RAM detection, selects a [`MemoryTier`], starts
    /// from the tier's defaults, then overrides any field whose corresponding
    /// `TRUSTY_*` env var is set to a parseable value. As a transitional
    /// measure (so existing scattered env-var readers Just Work) it also
    /// writes every resolved field back into the process environment.
    /// Test: see `test_tier_selection`, `test_env_override`, and
    /// `test_ram_detection_returns_nonzero`.
    pub fn detect() -> Self {
        let total_ram_mb = detect_total_ram_mb().unwrap_or_else(|| {
            tracing::warn!(
                "memory_policy: could not detect total system RAM — \
                 falling back to {FALLBACK_RAM_MB} MB (Medium tier defaults)"
            );
            FALLBACK_RAM_MB
        });
        Self::from_total_ram_mb(total_ram_mb)
    }

    /// Like [`Self::detect`] but with a caller-supplied RAM value. Useful for
    /// tests and for callers that have already measured RAM.
    pub fn from_total_ram_mb(total_ram_mb: u64) -> Self {
        let tier = MemoryTier::from_total_ram_mb(total_ram_mb);
        let d = tier.defaults();

        let policy = MemoryPolicy {
            total_ram_mb,
            tier,
            memory_limit_mb: env_override_usize("TRUSTY_MEMORY_LIMIT_MB", d.memory_limit_mb),
            max_chunks: env_override_usize("TRUSTY_MAX_CHUNKS", d.max_chunks),
            embedding_cache: env_override_usize("TRUSTY_EMBEDDING_CACHE", d.embedding_cache),
            max_batch_size: env_override_usize("TRUSTY_MAX_BATCH_SIZE", d.max_batch_size),
            bm25_corpus_cap: env_override_usize("TRUSTY_BM25_CORPUS_CAP", d.bm25_corpus_cap),
            max_kg_nodes: env_override_usize("TRUSTY_MAX_KG_NODES", d.max_kg_nodes),
        };

        // Transitional: stamp resolved values back into the env so existing
        // module-level readers (in indexer.rs, bm25.rs, symbol_graph.rs,
        // memguard.rs, store.rs) pick up the auto-tuned defaults without
        // each having to learn about MemoryPolicy.
        policy.apply_to_env();
        policy
    }

    /// Write every field back into the process environment. Idempotent; safe
    /// to call before any worker thread has read its env-cached cap.
    ///
    /// SAFETY: This must run before the daemon spawns any threads that read
    /// these env vars (e.g. before tokio workers start indexing). Calling
    /// `std::env::set_var` from a multi-threaded context is unsound on some
    /// platforms (see the std docs); `MemoryPolicy::detect()` is intended to
    /// be invoked once in `main` before the runtime is built.
    pub fn apply_to_env(&self) {
        // SAFETY: see doc comment — caller must invoke before threading begins.
        unsafe {
            std::env::set_var("TRUSTY_MEMORY_LIMIT_MB", self.memory_limit_mb.to_string());
            std::env::set_var("TRUSTY_MAX_CHUNKS", self.max_chunks.to_string());
            std::env::set_var("TRUSTY_EMBEDDING_CACHE", self.embedding_cache.to_string());
            std::env::set_var("TRUSTY_MAX_BATCH_SIZE", self.max_batch_size.to_string());
            std::env::set_var("TRUSTY_BM25_CORPUS_CAP", self.bm25_corpus_cap.to_string());
            std::env::set_var("TRUSTY_MAX_KG_NODES", self.max_kg_nodes.to_string());
        }
    }

    /// Pretty-print the resolved policy in two compact log lines suitable for
    /// `tracing::info!` at daemon startup.
    pub fn log_summary(&self) {
        let gb = self.total_ram_mb / 1024;
        tracing::info!("trusty-search: detected {} GB RAM → tier={}", gb, self.tier);
        tracing::info!(
            "  MEMORY_LIMIT_MB={}  MAX_CHUNKS={}  EMBEDDING_CACHE={}  \
             MAX_BATCH_SIZE={}  BM25_CORPUS_CAP={}  MAX_KG_NODES={}",
            self.memory_limit_mb,
            self.max_chunks,
            self.embedding_cache,
            self.max_batch_size,
            self.bm25_corpus_cap,
            self.max_kg_nodes,
        );
    }
}

/// Read a `TRUSTY_*` env var as `usize`; fall back to `default` when unset
/// or unparseable. A warning is logged on parse failure to surface typos.
fn env_override_usize(name: &str, default: usize) -> usize {
    match std::env::var(name) {
        Ok(v) => match v.parse::<usize>() {
            Ok(n) => n,
            Err(_) => {
                tracing::warn!(
                    "memory_policy: {name}={v:?} is not a valid usize; \
                     using tier default ({default})"
                );
                default
            }
        },
        Err(_) => default,
    }
}

/// Detect total physical RAM in megabytes. Returns `None` if the platform
/// path is not implemented or the detection command failed.
///
/// Why: tier selection drives every memory cap; we'd rather fall back to the
/// conservative Tiny tier than guess wrong on an unsupported OS.
/// What: dispatches to a `#[cfg]`-gated platform implementation
/// (`sysctl hw.memsize` on macOS, `/proc/meminfo` parsing on Linux).
/// Test: `test_ram_detection_returns_nonzero` asserts > 0 on the host
/// running the suite (CI runs Linux/macOS, both supported).
pub fn detect_total_ram_mb() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        detect_macos_ram_mb()
    }
    #[cfg(target_os = "linux")]
    {
        detect_linux_ram_mb()
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        None
    }
}

#[cfg(target_os = "macos")]
fn detect_macos_ram_mb() -> Option<u64> {
    use std::process::Command;
    // `sysctl -n hw.memsize` prints the byte count on its own line.
    let output = Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    let bytes: u64 = text.trim().parse().ok()?;
    Some(bytes / (1024 * 1024))
}

#[cfg(target_os = "linux")]
fn detect_linux_ram_mb() -> Option<u64> {
    // /proc/meminfo `MemTotal: NNNNN kB` (always kB, even on aarch64).
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            // rest looks like "  16384000 kB"
            let mut parts = rest.split_whitespace();
            let kb: u64 = parts.next()?.parse().ok()?;
            return Some(kb / 1024);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tier_selection() {
        // Boundary table: 16 GB Medium, 32 GB Large, 64 GB XLarge.
        // The daemon enforces a 16 GB hard minimum at startup, so sub-16 GB
        // RAM should never reach tier selection in normal operation. If it
        // does (e.g. tests, library consumers), we return Medium as a safe
        // fallback rather than panic.
        assert_eq!(MemoryTier::from_total_ram_mb(16 * 1024), MemoryTier::Medium);
        assert_eq!(MemoryTier::from_total_ram_mb(31 * 1024), MemoryTier::Medium);
        assert_eq!(MemoryTier::from_total_ram_mb(32 * 1024), MemoryTier::Large);
        assert_eq!(MemoryTier::from_total_ram_mb(63 * 1024), MemoryTier::Large);
        assert_eq!(MemoryTier::from_total_ram_mb(64 * 1024), MemoryTier::XLarge);
        assert_eq!(
            MemoryTier::from_total_ram_mb(192 * 1024),
            MemoryTier::XLarge
        );

        // Defensive fallback: sub-16 GB should not be reachable in production
        // (the daemon exits at startup), but the tier function must still
        // return something well-defined. We map to Medium.
        assert_eq!(MemoryTier::from_total_ram_mb(15 * 1024), MemoryTier::Medium);
        assert_eq!(MemoryTier::from_total_ram_mb(8 * 1024), MemoryTier::Medium);
        assert_eq!(MemoryTier::from_total_ram_mb(4 * 1024), MemoryTier::Medium);
    }

    #[test]
    fn test_tier_defaults_table() {
        // Spot-check the documented Memory Tier Table from the design doc.
        let medium = MemoryTier::Medium.defaults();
        assert_eq!(medium.memory_limit_mb, 4_096);
        assert_eq!(medium.max_chunks, 200_000);

        let xl = MemoryTier::XLarge.defaults();
        assert_eq!(xl.memory_limit_mb, 16_384);
        assert_eq!(xl.max_chunks, 800_000);
        assert_eq!(xl.embedding_cache, 20_000);
        assert_eq!(xl.max_kg_nodes, 500_000);
    }

    /// Verify that an env-var override beats the tier default.
    ///
    /// Note: `from_total_ram_mb` calls `apply_to_env`, which mutates the
    /// process env. We restore the prior values at the end of the test to
    /// avoid bleeding into other tests in the same binary. We do not run
    /// this concurrently with other env-mutating tests in this module —
    /// `cargo test` runs tests in a single module on different threads, so
    /// callers in CI rely on `--test-threads=1` only if they extend this
    /// module with more env-touching tests.
    #[test]
    fn test_env_override() {
        // Save & override.
        let prior = std::env::var("TRUSTY_MAX_CHUNKS").ok();
        // SAFETY: tests run single-threaded within this module's env block.
        unsafe {
            std::env::set_var("TRUSTY_MAX_CHUNKS", "42");
        }

        // 16 GB → Medium tier (default max_chunks = 200_000). Env should win.
        let policy = MemoryPolicy::from_total_ram_mb(16 * 1024);
        assert_eq!(policy.tier, MemoryTier::Medium);
        assert_eq!(policy.max_chunks, 42);

        // Restore.
        // SAFETY: same as above.
        unsafe {
            match prior {
                Some(v) => std::env::set_var("TRUSTY_MAX_CHUNKS", v),
                None => std::env::remove_var("TRUSTY_MAX_CHUNKS"),
            }
        }
    }

    #[test]
    fn test_ram_detection_returns_nonzero() {
        // Best-effort: on macOS/Linux CI hosts this must return a real value.
        // On other platforms (none in our CI matrix today) the function
        // returns None and we skip the assertion rather than fail.
        if let Some(mb) = detect_total_ram_mb() {
            assert!(mb > 0, "detected RAM should be > 0, got {mb}");
            // Sanity ceiling: no host in our deployment fleet has > 4 TB.
            assert!(
                mb < 4 * 1024 * 1024,
                "detected RAM implausibly large: {mb} MB"
            );
        }
    }
}
