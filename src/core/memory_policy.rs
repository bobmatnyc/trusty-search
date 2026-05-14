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

/// Empirically measured ORT transient arena cost per `embed_batch` slot in MB.
///
/// Why: ORT allocates working memory proportional to `batch_size × emb_dim ×
/// seq_len` during each `embed_batch` call. The previous estimate of 55 MB
/// per slot was measured on a small synthetic test corpus with short chunks
/// and turned out to be a severe underestimate on real workloads. Production
/// reindex on a 128 GB host showed peak RSS of 26–94 GB at the XLarge-tier
/// default of 223 slots — i.e. ~150–400 MB per slot for typical code files,
/// not 55. We now use 200 MB/slot as a realistic upper-bound estimate
/// reflecting actual large-chunk behaviour. The between-batch RSS poller in
/// `memguard` cannot catch this intra-call spike (it polls AFTER each batch,
/// not during), so batch size MUST be bounded up-front from the configured
/// memory limit. See issue #95 and the 94 GB reindex incident report.
const EMBED_MB_PER_BATCH_SLOT: u64 = 200;

/// Reserve this fraction of `memory_limit_mb` for the ORT transient arena
/// when computing `max_batch_size`. The remaining 25% accounts for the
/// resident process working set (HNSW, BM25 corpus, chunk cache, redb, etc.).
const EMBED_ARENA_BUDGET_NUM: u64 = 75;
const EMBED_ARENA_BUDGET_DEN: u64 = 100;

/// Fraction of total system RAM allocated to `memory_limit_mb` (the soft cap
/// on the indexing pipeline's working set). Why 25%: the daemon shares the
/// host with the user's editor, browser, language servers, OS, and frequently
/// other dev daemons. Reserving 75% for everything else has empirically kept
/// reindex runs from triggering OOM-killer cascades on workstations between
/// 16 GB and 256 GB. See issue #120 (104 GB reindex on a 128 GB host with a
/// hardcoded 128 GB plist override).
const MEMORY_LIMIT_FRACTION_NUM: u64 = 25;
const MEMORY_LIMIT_FRACTION_DEN: u64 = 100;

/// Absolute minimum `memory_limit_mb` (1 GB). On hosts smaller than 4 GB the
/// 25% rule would otherwise drop below where the indexer can meaningfully run.
const MEMORY_LIMIT_FLOOR_MB: u64 = 1_024;

/// Absolute maximum `memory_limit_mb` (64 GB). Even on 256 GB workstations we
/// cap the soft limit here — beyond this point the bottleneck is no longer
/// RAM but ORT transient-arena behaviour and HNSW serialization, and very
/// large limits encourage configurations that are hard to reason about.
const MEMORY_LIMIT_CEIL_MB: u64 = 65_536;

/// Ratio of `max_chunks` to `memory_limit_mb` (chunks per MB of soft limit).
/// Derived from the historical Medium tier (200 000 chunks / 4 096 MB ≈ 49)
/// — see the prior tier table. Why preserve it: this ratio reflects empirical
/// HNSW + redb overhead per chunk in steady state.
const CHUNKS_PER_MB: u64 = 50;

/// Floor / ceiling for the computed `max_chunks`. Match the prior tier table
/// endpoints (Tiny → 50 000, XLarge → 800 000) so behavior on previously
/// supported hosts is unchanged.
const MAX_CHUNKS_FLOOR: usize = 50_000;
const MAX_CHUNKS_CEIL: usize = 800_000;

/// Compute `memory_limit_mb` proportional to detected system RAM.
///
/// Why: prior to issue #120 the XLarge tier capped the soft limit at 16 GB
/// regardless of host size, so a 128 GB box was indistinguishable from a
/// 64 GB box — and a launchd plist override pushed it to 128 GB, allowing a
/// reindex to consume 104 GB and OOM-kill the tmux server. The fix is to
/// scale the limit with available RAM: 25% of host RAM, clamped to
/// [`MEMORY_LIMIT_FLOOR_MB`, `MEMORY_LIMIT_CEIL_MB`].
/// What: `clamp(total_ram_mb * 0.25, 1024, 65536)`. Examples: 16 GB → 4 GB,
/// 32 GB → 8 GB, 64 GB → 16 GB, 128 GB → 32 GB, 256 GB → 64 GB (ceiling).
/// Test: `test_compute_memory_limit_from_ram` covers the table and clamps.
fn compute_memory_limit_mb(total_ram_mb: u64) -> usize {
    let raw = total_ram_mb * MEMORY_LIMIT_FRACTION_NUM / MEMORY_LIMIT_FRACTION_DEN;
    raw.clamp(MEMORY_LIMIT_FLOOR_MB, MEMORY_LIMIT_CEIL_MB) as usize
}

/// Compute `max_chunks` proportional to `memory_limit_mb`.
///
/// Why: chunk capacity should scale with the working-set budget, not with
/// fixed tier buckets. At ~50 chunks/MB (the historical Medium-tier ratio)
/// every MB of soft limit corresponds to one chunk of HNSW + redb overhead
/// in steady state.
/// What: `clamp(memory_limit_mb * 50, 50_000, 800_000)`.
/// Test: `test_compute_max_chunks_from_limit` covers the tier table.
fn compute_max_chunks(memory_limit_mb: usize) -> usize {
    let raw = (memory_limit_mb as u64) * CHUNKS_PER_MB;
    (raw as usize).clamp(MAX_CHUNKS_FLOOR, MAX_CHUNKS_CEIL)
}

/// Floor for the computed batch size. Below this throughput collapses but the
/// process is still functional. Lowered from 32 → 8 to give the formula room
/// to recommend safer values on memory-constrained hosts.
const MIN_COMPUTED_BATCH_SIZE: usize = 8;

/// Ceiling for the computed batch size. Lowered from 512 → 64 (issue: 94 GB
/// reindex spike). At 200 MB/slot, even 64 slots × 200 MB = 12.8 GB transient
/// peak — already close to the Medium tier's 4 GB soft cap, so anything
/// higher would be reckless on the smallest supported host. GPU paths that
/// genuinely need 512 explicitly opt out via `TRUSTY_MAX_BATCH_SIZE_EXPLICIT=1`
/// (see `commands::start::tune_batch_size_for_provider`).
const MAX_COMPUTED_BATCH_SIZE: usize = 64;

/// Compute the safe `max_batch_size` for a given memory limit so that the ORT
/// transient arena (≈ `EMBED_MB_PER_BATCH_SLOT` per slot) stays within
/// `memory_limit_mb × budget_fraction`. Clamped to
/// `[MIN_COMPUTED_BATCH_SIZE, MAX_COMPUTED_BATCH_SIZE]`.
///
/// Why: see `EMBED_MB_PER_BATCH_SLOT` doc — intra-call spikes are invisible to
/// the RSS poller, so batch size must be sized from the limit, not against it.
/// What: `floor(memory_limit_mb * 0.75 / 200)`, clamped to `[8, 64]`. With the
/// corrected 200 MB/slot estimate this yields: Medium (4 GB) → 15, Large
/// (8 GB) → 30, XLarge (16 GB) → 61. The previous formula (55 MB/slot,
/// `[32, 512]`) produced 55/111/223 which blew through every soft cap on
/// real workloads — see the 94 GB reindex incident report.
/// Test: `test_compute_max_batch_size_from_limit` covers the tier table and
/// the clamp endpoints.
fn compute_max_batch_size(memory_limit_mb: usize) -> usize {
    let budget_mb = (memory_limit_mb as u64) * EMBED_ARENA_BUDGET_NUM / EMBED_ARENA_BUDGET_DEN;
    let raw = (budget_mb / EMBED_MB_PER_BATCH_SLOT) as usize;
    raw.clamp(MIN_COMPUTED_BATCH_SIZE, MAX_COMPUTED_BATCH_SIZE)
}

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
    /// Tier-specific hard cap on `max_batch_size`. Conservative bound that
    /// protects against runaway env-var overrides on memory-constrained hosts
    /// (issue #89).
    ///
    /// Why: `TRUSTY_MAX_BATCH_SIZE` is a runtime knob. An operator who sets it
    /// to 2048 on a 16 GB box (the Medium tier) will trigger the same ORT
    /// transient-arena spike that auto-tuning was designed to prevent. The
    /// auto-derived defaults (Medium=15, Large=30, XLarge=61) already sit
    /// well below these caps, so this hard ceiling only kicks in for explicit
    /// overrides — exactly the case where additional safety is warranted.
    /// What: Medium=16, Large=32, XLarge=64. Lowered from {64, 128, 256}
    /// after the 94 GB reindex incident (issue: 55→200 MB/slot correction):
    /// at the corrected 200 MB/slot, even 64 slots × 200 MB = 12.8 GB
    /// intra-call peak, which is already aggressive on Medium's 4 GB soft cap.
    /// The arena disable in trusty-embedder (`with_arena_allocator(false)`)
    /// is the primary mitigation; this cap is defense in depth.
    /// Test: `test_tier_batch_size_hard_cap` covers the table.
    pub fn batch_size_hard_cap(self) -> usize {
        match self {
            MemoryTier::Medium => 16,
            MemoryTier::Large => 32,
            MemoryTier::XLarge => 64,
        }
    }

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

    /// Default caps for this tier given a precomputed proportional
    /// `memory_limit_mb` (see [`compute_memory_limit_mb`]).
    ///
    /// `max_batch_size` is derived from `memory_limit_mb` via
    /// [`compute_max_batch_size`] so the ORT transient arena (≈200 MB per
    /// batch slot) cannot exceed 75% of the configured soft cap. See issue #95.
    /// `max_chunks` is derived from `memory_limit_mb` via [`compute_max_chunks`]
    /// so capacity scales with the working-set budget rather than fixed tier
    /// buckets (issue #120). The remaining fields (`embedding_cache`,
    /// `bm25_corpus_cap`, `max_kg_nodes`) keep their tier-based defaults since
    /// they're driven more by index size than absolute RAM.
    fn defaults(self, memory_limit_mb: usize) -> TierDefaults {
        let (embedding_cache, bm25_corpus_cap, max_kg_nodes) = match self {
            MemoryTier::Medium => (5_000, 100_000, 150_000),
            MemoryTier::Large => (10_000, 200_000, 300_000),
            MemoryTier::XLarge => (20_000, 400_000, 500_000),
        };
        TierDefaults {
            memory_limit_mb,
            max_chunks: compute_max_chunks(memory_limit_mb),
            embedding_cache,
            max_batch_size: compute_max_batch_size(memory_limit_mb),
            bm25_corpus_cap,
            max_kg_nodes,
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
        // Compute the proportional memory limit (25% of system RAM, clamped)
        // BEFORE selecting tier-keyed defaults so max_chunks / max_batch_size
        // scale with the actual host RAM rather than a fixed tier bucket.
        // See issue #120 (104 GB reindex on a 128 GB host).
        let proportional_limit_mb = compute_memory_limit_mb(total_ram_mb);
        let d = tier.defaults(proportional_limit_mb);

        // Resolve memory limit first so the derived max_batch_size default
        // tracks any TRUSTY_MEMORY_LIMIT_MB override. TRUSTY_MAX_BATCH_SIZE
        // still wins when explicitly set. See issue #95.
        let memory_limit_mb = env_override_usize("TRUSTY_MEMORY_LIMIT_MB", d.memory_limit_mb);
        let derived_batch_size = if memory_limit_mb == d.memory_limit_mb {
            d.max_batch_size
        } else {
            compute_max_batch_size(memory_limit_mb)
        };
        // Recompute max_chunks if the operator overrode the memory limit so
        // chunk capacity tracks the actual configured limit (env var for
        // max_chunks still wins below via env_override_usize).
        let derived_max_chunks = if memory_limit_mb == d.memory_limit_mb {
            d.max_chunks
        } else {
            compute_max_chunks(memory_limit_mb)
        };

        // Apply tier-based hard cap on max_batch_size (issue #89). The cap
        // only constrains env-var overrides; auto-derived defaults are already
        // safely below it.
        //
        // Escape hatch: `TRUSTY_MAX_BATCH_SIZE_EXPLICIT=1` opts the operator
        // out of the tier hard cap, honoring `TRUSTY_MAX_BATCH_SIZE` verbatim.
        // Why: the GPU tuning path (commands::start::tune_batch_size_for_provider)
        // needs to set 512 to feed CUDA efficiently, and other power users may
        // have measured their own per-slot cost on specific workloads. This
        // mirrors the GPU path's existing semantics and makes the flag apply
        // universally (CPU + GPU), not just at GPU init. See the 94 GB reindex
        // incident report.
        let explicit = std::env::var("TRUSTY_MAX_BATCH_SIZE_EXPLICIT")
            .map(|v| v == "1")
            .unwrap_or(false);
        let env_set = std::env::var("TRUSTY_MAX_BATCH_SIZE").is_ok();
        let raw_batch_size = env_override_usize("TRUSTY_MAX_BATCH_SIZE", derived_batch_size);
        let batch_cap = tier.batch_size_hard_cap();
        let max_batch_size = if explicit && env_set {
            tracing::warn!(
                "memory_policy: TRUSTY_MAX_BATCH_SIZE_EXPLICIT=1 — honoring \
                 TRUSTY_MAX_BATCH_SIZE={} verbatim and bypassing tier {} hard cap of {}. \
                 Ensure you have measured the actual ORT transient-arena cost per slot \
                 on your workload (defaults assume 200 MB/slot).",
                raw_batch_size,
                tier,
                batch_cap,
            );
            raw_batch_size
        } else if raw_batch_size > batch_cap {
            tracing::warn!(
                "memory_policy: TRUSTY_MAX_BATCH_SIZE={} exceeds tier {} hard cap of {}; \
                 clamping to protect against ORT transient-arena spike (issue #89). \
                 Set TRUSTY_MAX_BATCH_SIZE_EXPLICIT=1 to bypass this clamp.",
                raw_batch_size,
                tier,
                batch_cap,
            );
            batch_cap
        } else {
            raw_batch_size
        };

        let policy = MemoryPolicy {
            total_ram_mb,
            tier,
            memory_limit_mb,
            max_chunks: env_override_usize("TRUSTY_MAX_CHUNKS", derived_max_chunks),
            embedding_cache: env_override_usize("TRUSTY_EMBEDDING_CACHE", d.embedding_cache),
            max_batch_size,
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
        let proportional = compute_memory_limit_mb(self.total_ram_mb);
        tracing::info!(
            "trusty-search: detected {} GB RAM → tier={} (proportional memory_limit_mb={}, 25% of RAM clamped to [{}, {}])",
            gb,
            self.tier,
            proportional,
            MEMORY_LIMIT_FLOOR_MB,
            MEMORY_LIMIT_CEIL_MB,
        );
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
    use std::sync::Mutex;

    /// Serialize env-mutating tests within this module. Cargo runs tests on
    /// multiple threads by default and `std::env::set_var` is process-global,
    /// so without this guard a concurrent test can stomp on the env vars a
    /// sibling test relies on (e.g. `TRUSTY_MAX_BATCH_SIZE_EXPLICIT`).
    static ENV_LOCK: Mutex<()> = Mutex::new(());

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
        // Spot-check the documented Memory Tier Table. As of issue #120 the
        // tier's defaults are parameterised by a proportional memory_limit_mb
        // (25% of host RAM, clamped). We feed each tier its representative
        // host size and assert the derived caps match the table.

        // 16 GB host → Medium → proportional limit = 4 GB.
        let medium = MemoryTier::Medium.defaults(compute_memory_limit_mb(16 * 1024));
        assert_eq!(medium.memory_limit_mb, 4_096);
        // max_chunks = clamp(4096 * 50, 50_000, 800_000) = 204_800
        assert_eq!(medium.max_chunks, 204_800);
        // max_batch_size = floor(4096 * 0.75 / 200) = floor(15.36) = 15
        assert_eq!(medium.max_batch_size, 15);
        assert_eq!(medium.embedding_cache, 5_000);

        // 32 GB host → Large → proportional limit = 8 GB.
        let large = MemoryTier::Large.defaults(compute_memory_limit_mb(32 * 1024));
        assert_eq!(large.memory_limit_mb, 8_192);
        // max_chunks = clamp(8192 * 50, 50_000, 800_000) = 409_600
        assert_eq!(large.max_chunks, 409_600);
        // max_batch_size = floor(8192 * 0.75 / 200) = floor(30.72) = 30
        assert_eq!(large.max_batch_size, 30);

        // 64 GB host → XLarge → proportional limit = 16 GB.
        let xl = MemoryTier::XLarge.defaults(compute_memory_limit_mb(64 * 1024));
        assert_eq!(xl.memory_limit_mb, 16_384);
        // max_chunks = clamp(16384 * 50, 50_000, 800_000) = 800_000 (ceiling)
        assert_eq!(xl.max_chunks, 800_000);
        assert_eq!(xl.embedding_cache, 20_000);
        assert_eq!(xl.max_kg_nodes, 500_000);
        // max_batch_size = floor(16384 * 0.75 / 200) = floor(61.44) = 61
        assert_eq!(xl.max_batch_size, 61);

        // 128 GB host → XLarge → proportional limit = 32 GB (previously
        // capped at 16 GB regardless of host size — issue #120).
        let huge = MemoryTier::XLarge.defaults(compute_memory_limit_mb(128 * 1024));
        assert_eq!(huge.memory_limit_mb, 32 * 1024);

        // 256 GB host → XLarge → proportional limit = 64 GB (ceiling).
        let max_host = MemoryTier::XLarge.defaults(compute_memory_limit_mb(256 * 1024));
        assert_eq!(max_host.memory_limit_mb, MEMORY_LIMIT_CEIL_MB as usize);
    }

    #[test]
    fn test_compute_memory_limit_from_ram() {
        // Issue #120: memory_limit_mb is 25% of system RAM, clamped to
        // [1 GB, 64 GB]. Examples for the tier table:
        assert_eq!(compute_memory_limit_mb(16 * 1024), 4 * 1024); // 16 GB → 4 GB
        assert_eq!(compute_memory_limit_mb(32 * 1024), 8 * 1024); // 32 GB → 8 GB
        assert_eq!(compute_memory_limit_mb(64 * 1024), 16 * 1024); // 64 GB → 16 GB
        assert_eq!(compute_memory_limit_mb(128 * 1024), 32 * 1024); // 128 GB → 32 GB
        assert_eq!(compute_memory_limit_mb(192 * 1024), 48 * 1024); // 192 GB → 48 GB

        // Ceiling clamp at 64 GB.
        assert_eq!(compute_memory_limit_mb(256 * 1024), 64 * 1024);
        assert_eq!(compute_memory_limit_mb(1024 * 1024), 64 * 1024); // 1 TB host

        // Floor clamp at 1 GB.
        assert_eq!(compute_memory_limit_mb(0), 1_024);
        assert_eq!(compute_memory_limit_mb(2 * 1024), 1_024); // 2 GB → floored at 1 GB
        assert_eq!(compute_memory_limit_mb(4 * 1024), 1_024); // 4 GB → exactly 1 GB
    }

    #[test]
    fn test_compute_max_chunks_from_limit() {
        // max_chunks = clamp(memory_limit_mb * 50, 50_000, 800_000).
        assert_eq!(compute_max_chunks(4_096), 204_800);
        assert_eq!(compute_max_chunks(8_192), 409_600);
        assert_eq!(compute_max_chunks(16_384), 800_000); // ceiling
        assert_eq!(compute_max_chunks(32_768), 800_000); // ceiling
        assert_eq!(compute_max_chunks(65_536), 800_000); // ceiling
                                                         // Floor clamp: tiny limits still produce a usable chunk capacity.
        assert_eq!(compute_max_chunks(0), MAX_CHUNKS_FLOOR);
        assert_eq!(compute_max_chunks(500), MAX_CHUNKS_FLOOR);
    }

    #[test]
    fn test_memory_limit_scales_proportionally_across_xlarge_hosts() {
        // Regression test for issue #120: two XLarge hosts of different sizes
        // must produce different memory limits (the old code returned 16 GB
        // for both 64 GB and 128 GB boxes).
        //
        // Hold ENV_LOCK because `from_total_ram_mb` writes the resolved values
        // back into the process env via `apply_to_env`. Without the lock,
        // other tests in this module can stomp on TRUSTY_MEMORY_LIMIT_MB
        // between the two calls below.
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // Save & clear TRUSTY_MEMORY_LIMIT_MB so an earlier test's
        // `apply_to_env` write doesn't override the proportional defaults.
        let prior = std::env::var("TRUSTY_MEMORY_LIMIT_MB").ok();
        // SAFETY: tests run single-threaded within this module's env block.
        unsafe {
            std::env::remove_var("TRUSTY_MEMORY_LIMIT_MB");
        }
        let p64 = MemoryPolicy::from_total_ram_mb(64 * 1024);
        // The 64 GB call wrote TRUSTY_MEMORY_LIMIT_MB=16384 back into the env;
        // clear it again so the 128 GB call sees a clean slate.
        // SAFETY: same as above.
        unsafe {
            std::env::remove_var("TRUSTY_MEMORY_LIMIT_MB");
        }
        let p128 = MemoryPolicy::from_total_ram_mb(128 * 1024);
        // Restore prior value.
        // SAFETY: same as above.
        unsafe {
            match prior {
                Some(v) => std::env::set_var("TRUSTY_MEMORY_LIMIT_MB", v),
                None => std::env::remove_var("TRUSTY_MEMORY_LIMIT_MB"),
            }
        }
        assert_eq!(p64.tier, MemoryTier::XLarge);
        assert_eq!(p128.tier, MemoryTier::XLarge);
        assert!(
            p128.memory_limit_mb > p64.memory_limit_mb,
            "128 GB host ({} MB) should have a larger memory_limit_mb than \
             a 64 GB host ({} MB) — see issue #120",
            p128.memory_limit_mb,
            p64.memory_limit_mb,
        );
        // Specifically: 128 GB → 32 GB limit; 64 GB → 16 GB limit.
        assert_eq!(p64.memory_limit_mb, 16 * 1024);
        assert_eq!(p128.memory_limit_mb, 32 * 1024);
    }

    #[test]
    fn test_compute_max_batch_size_from_limit() {
        // Tier table with corrected 200 MB/slot estimate (was 55).
        assert_eq!(compute_max_batch_size(4_096), 15);
        assert_eq!(compute_max_batch_size(8_192), 30);
        assert_eq!(compute_max_batch_size(16_384), 61);

        // Floor clamp: tiny limits still produce a usable batch size.
        assert_eq!(compute_max_batch_size(0), MIN_COMPUTED_BATCH_SIZE);
        // 1024 MB → floor(1024 * 0.75 / 200) = floor(3.84) = 3 → clamped to 8
        assert_eq!(compute_max_batch_size(1_024), MIN_COMPUTED_BATCH_SIZE);

        // Ceiling clamp: enormous limits don't push past new [8, 64] cap.
        assert_eq!(compute_max_batch_size(64_000), MAX_COMPUTED_BATCH_SIZE);
        assert_eq!(compute_max_batch_size(1_000_000), MAX_COMPUTED_BATCH_SIZE);
        // Just above where the raw formula reaches 64:
        // floor(17_067 * 0.75 / 200) = 64, anything above stays clamped at 64.
        assert_eq!(compute_max_batch_size(20_000), MAX_COMPUTED_BATCH_SIZE);
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
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
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
    fn test_tier_batch_size_hard_cap() {
        // Issue #89: tier-specific batch-size hard caps protect against
        // runaway TRUSTY_MAX_BATCH_SIZE overrides on memory-constrained hosts.
        // Lowered after the 94 GB reindex incident (55→200 MB/slot correction).
        assert_eq!(MemoryTier::Medium.batch_size_hard_cap(), 16);
        assert_eq!(MemoryTier::Large.batch_size_hard_cap(), 32);
        assert_eq!(MemoryTier::XLarge.batch_size_hard_cap(), 64);
    }

    #[test]
    fn test_batch_size_env_override_clamped_by_hard_cap() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // Save & override.
        let prior = std::env::var("TRUSTY_MAX_BATCH_SIZE").ok();
        let prior_explicit = std::env::var("TRUSTY_MAX_BATCH_SIZE_EXPLICIT").ok();
        // SAFETY: tests run single-threaded within this module's env block.
        unsafe {
            std::env::set_var("TRUSTY_MAX_BATCH_SIZE", "2048");
            // Ensure the explicit-bypass flag is unset for this test.
            std::env::remove_var("TRUSTY_MAX_BATCH_SIZE_EXPLICIT");
        }

        // 16 GB → Medium tier, hard cap = 16. Env value of 2048 must be
        // clamped down to the tier cap.
        let policy = MemoryPolicy::from_total_ram_mb(16 * 1024);
        assert_eq!(policy.tier, MemoryTier::Medium);
        assert_eq!(
            policy.max_batch_size, 16,
            "Medium tier must clamp TRUSTY_MAX_BATCH_SIZE=2048 down to 16"
        );

        // Restore.
        // SAFETY: same as above.
        unsafe {
            match prior {
                Some(v) => std::env::set_var("TRUSTY_MAX_BATCH_SIZE", v),
                None => std::env::remove_var("TRUSTY_MAX_BATCH_SIZE"),
            }
            match prior_explicit {
                Some(v) => std::env::set_var("TRUSTY_MAX_BATCH_SIZE_EXPLICIT", v),
                None => std::env::remove_var("TRUSTY_MAX_BATCH_SIZE_EXPLICIT"),
            }
        }
    }

    #[test]
    fn test_batch_size_explicit_flag_bypasses_clamp() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // Save & override.
        let prior = std::env::var("TRUSTY_MAX_BATCH_SIZE").ok();
        let prior_explicit = std::env::var("TRUSTY_MAX_BATCH_SIZE_EXPLICIT").ok();
        let prior_mem = std::env::var("TRUSTY_MEMORY_LIMIT_MB").ok();
        // SAFETY: tests run single-threaded within this module's env block.
        unsafe {
            std::env::set_var("TRUSTY_MAX_BATCH_SIZE", "512");
            std::env::set_var("TRUSTY_MAX_BATCH_SIZE_EXPLICIT", "1");
            // Clear leftover memory-limit env from sibling tests so the
            // proportional default for 16 GB host applies cleanly.
            std::env::remove_var("TRUSTY_MEMORY_LIMIT_MB");
        }

        // 16 GB → Medium tier, hard cap = 16. With EXPLICIT=1 the operator's
        // 512 must be honored verbatim (GPU path, expert opt-out).
        let policy = MemoryPolicy::from_total_ram_mb(16 * 1024);
        assert_eq!(policy.tier, MemoryTier::Medium);
        assert_eq!(
            policy.max_batch_size, 512,
            "TRUSTY_MAX_BATCH_SIZE_EXPLICIT=1 must bypass the tier hard cap"
        );

        // Restore.
        // SAFETY: same as above.
        unsafe {
            match prior {
                Some(v) => std::env::set_var("TRUSTY_MAX_BATCH_SIZE", v),
                None => std::env::remove_var("TRUSTY_MAX_BATCH_SIZE"),
            }
            match prior_explicit {
                Some(v) => std::env::set_var("TRUSTY_MAX_BATCH_SIZE_EXPLICIT", v),
                None => std::env::remove_var("TRUSTY_MAX_BATCH_SIZE_EXPLICIT"),
            }
            match prior_mem {
                Some(v) => std::env::set_var("TRUSTY_MEMORY_LIMIT_MB", v),
                None => std::env::remove_var("TRUSTY_MEMORY_LIMIT_MB"),
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
