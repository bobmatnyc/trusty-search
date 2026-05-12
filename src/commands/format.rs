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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_with_commas_zero() {
        assert_eq!(format_with_commas(0), "0");
    }

    #[test]
    fn format_with_commas_small() {
        assert_eq!(format_with_commas(42), "42");
        assert_eq!(format_with_commas(999), "999");
    }

    #[test]
    fn format_with_commas_thousands_boundary() {
        assert_eq!(format_with_commas(1_000), "1,000");
        assert_eq!(format_with_commas(10_000), "10,000");
        assert_eq!(format_with_commas(115_585), "115,585");
    }

    #[test]
    fn format_with_commas_millions_and_billions() {
        assert_eq!(format_with_commas(1_000_000), "1,000,000");
        assert_eq!(format_with_commas(1_234_567_890), "1,234,567,890");
    }

    #[test]
    fn fmt_elapsed_milliseconds_branch() {
        // < 1s → milliseconds branch.
        assert_eq!(fmt_elapsed(0), "0ms");
        assert_eq!(fmt_elapsed(250), "250ms");
        assert_eq!(fmt_elapsed(999), "999ms");
    }

    #[test]
    fn fmt_elapsed_seconds_branch() {
        // 1s <= t < 60s → seconds branch.
        assert_eq!(fmt_elapsed(1_000), "1s");
        assert_eq!(fmt_elapsed(45_000), "45s");
        assert_eq!(fmt_elapsed(59_999), "59s");
    }

    #[test]
    fn fmt_elapsed_minutes_branch() {
        assert_eq!(fmt_elapsed(60_000), "1m 00s");
        assert_eq!(fmt_elapsed(75_000), "1m 15s");
        assert_eq!(fmt_elapsed(3_600_000), "60m 00s");
    }

    #[test]
    fn fmt_secs_short() {
        assert_eq!(fmt_secs(0), "0s");
        assert_eq!(fmt_secs(59), "59s");
    }

    #[test]
    fn fmt_secs_long() {
        assert_eq!(fmt_secs(60), "1m 00s");
        assert_eq!(fmt_secs(125), "2m 05s");
    }

    #[test]
    fn fmt_bytes_three_branches() {
        // Bytes branch (< 1KB).
        assert_eq!(fmt_bytes(0), "0B");
        assert_eq!(fmt_bytes(999), "999B");
        // KB branch (>= 1KB, < 1MB).
        assert_eq!(fmt_bytes(1_000), "1KB");
        assert_eq!(fmt_bytes(999_999), "1000KB");
        // MB branch.
        assert_eq!(fmt_bytes(1_000_000), "1MB");
        assert_eq!(fmt_bytes(25_500_000), "26MB");
    }

    #[test]
    fn dir_size_bytes_missing_returns_zero() {
        // Non-existent path silently returns 0 (best-effort contract).
        let p = std::path::Path::new("/definitely/does/not/exist/trusty-search-test-xyz");
        assert_eq!(dir_size_bytes(p), 0);
    }

    #[test]
    fn dir_size_bytes_sums_files() {
        // Create a tempdir with two known-size files in a subdir.
        let tmp =
            std::env::temp_dir().join(format!("trusty-search-dir-size-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(tmp.join("sub")).unwrap();
        std::fs::write(tmp.join("a.bin"), vec![0u8; 100]).unwrap();
        std::fs::write(tmp.join("sub").join("b.bin"), vec![0u8; 250]).unwrap();
        let total = dir_size_bytes(&tmp);
        assert_eq!(total, 350);
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
