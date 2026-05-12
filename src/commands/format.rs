//! Shared formatting helpers used across multiple CLI subcommands.
//!
//! Why: `format_with_commas`, `fmt_elapsed`, `fmt_secs`, `fmt_bytes`, and
//! `dir_size_bytes` were defined inline in `main.rs` but consumed by reindex
//! progress rendering, status output, and doctor output. Lifting them into a
//! dedicated module shrinks `main.rs` and makes the helpers independently
//! testable.
//! What: pure formatting functions plus a recursive directory-size walker.
//! Test: `cargo test --workspace` — covered indirectly by every CLI subcommand
//! that renders byte counts, elapsed times, or chunk counts.

/// Format a u64 with locale-style thousands separators (e.g. 115585 → "115,585").
///
/// Why: chunk counts for large repos (100k+) are hard to read without commas.
/// What: groups digits in threes from the right, separated by ",".
/// Test: 0 → "0", 1000 → "1,000", 115585 → "115,585".
pub fn format_with_commas(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::with_capacity(s.len() + s.len() / 3);
    for (i, ch) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(ch);
    }
    result.chars().rev().collect()
}

/// Format a millisecond elapsed time as `Xm Ys` (or `Ys` if < 1 minute).
pub fn fmt_elapsed(ms: u64) -> String {
    let secs = ms / 1000;
    if secs >= 60 {
        format!("{}m {:02}s", secs / 60, secs % 60)
    } else if secs > 0 {
        format!("{}s", secs)
    } else {
        format!("{}ms", ms)
    }
}

/// Format an elapsed seconds count as `Xm Ys` (or `Ys`).
pub fn fmt_secs(secs: u64) -> String {
    if secs >= 60 {
        format!("{}m {:02}s", secs / 60, secs % 60)
    } else {
        format!("{}s", secs)
    }
}

/// Format bytes as a human-readable string (MB / KB / B).
pub fn fmt_bytes(bytes: u64) -> String {
    if bytes >= 1_000_000 {
        format!("{:.0}MB", bytes as f64 / 1_000_000.0)
    } else if bytes >= 1_000 {
        format!("{:.0}KB", bytes as f64 / 1_000.0)
    } else {
        format!("{}B", bytes)
    }
}

/// Compute total byte size of a directory tree (best-effort; ignores errors).
pub fn dir_size_bytes(path: &std::path::Path) -> u64 {
    let mut total = 0u64;
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_file() {
                total += std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
            } else if p.is_dir() {
                total += dir_size_bytes(&p);
            }
        }
    }
    total
}
