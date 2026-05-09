//! Per-chunk git blame integration with temporal decay scoring (issue #30).
//!
//! Why: Recent code is more likely to be relevant to "what's broken now"
//! questions, while stale code may carry hidden assumptions. Capturing
//! blame metadata at index time lets the search pipeline (or downstream
//! tooling) bias results by recency without paying a per-query git cost.
//!
//! What: [`blame_chunk`] shells out to `git log -L` to fetch the most-recent
//! commit touching a chunk's line range. Returns `None` for files that aren't
//! tracked by git (or when `git` isn't on PATH) so non-git workflows degrade
//! gracefully. [`temporal_decay`] computes `exp(-lambda * days)`.
//!
//! Test: see `#[cfg(test)]` below — `temporal_decay` half-life math, and
//! a non-tracked tempfile returning `None` rather than panicking.
//!
//! Note on cost: spawning `git` per chunk is acceptable for index-time work
//! (one-off, parallelizable). If this ever moves into the hot search loop the
//! caller should batch blames per file or cache them.

use std::path::Path;
use std::process::Command;

use chrono::{DateTime, NaiveDate, Utc};
use serde::{Deserialize, Serialize};

/// Git blame metadata for a single chunk's line range.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ChunkBlame {
    /// Short commit hash (12 chars) or empty if not tracked.
    pub commit_hash: String,
    pub author_email: String,
    /// ISO 8601 date string, e.g. "2026-05-09".
    pub last_modified: String,
    /// Days between `last_modified` and the time `blame_chunk` ran.
    pub days_since_modified: u32,
}

/// Default temporal decay constant. λ=0.01 → half-life ≈ ln(2)/0.01 ≈ 69.3 days.
pub const DEFAULT_LAMBDA: f32 = 0.01;

/// Compute temporal decay score: `exp(-lambda * days)`.
///
/// Returns 1.0 for fresh code (days=0) and decays exponentially. With
/// [`DEFAULT_LAMBDA`] the score halves every ~70 days.
pub fn temporal_decay(days: u32, lambda: f32) -> f32 {
    (-lambda * days as f32).exp()
}

/// Fetch git blame for a line range in a file.
///
/// Returns `None` when:
/// - the file isn't tracked by git
/// - git isn't on PATH
/// - `git log -L` fails or produces unparseable output
///
/// Does *not* panic on any error path — non-git workflows must keep indexing.
pub fn blame_chunk(file: &Path, start_line: usize, end_line: usize) -> Option<ChunkBlame> {
    if start_line == 0 || end_line < start_line {
        return None;
    }

    // Run git from the file's parent so relative paths resolve and we don't
    // need to know the repo root.
    let cwd = file.parent().unwrap_or_else(|| Path::new("."));
    let file_arg = file.file_name()?.to_str()?;

    // 1) Verify the file is tracked. `git ls-files --error-unmatch <path>`
    //    exits non-zero if not tracked or if not in a repo.
    let tracked = Command::new("git")
        .current_dir(cwd)
        .args(["ls-files", "--error-unmatch", file_arg])
        .output()
        .ok()?;
    if !tracked.status.success() {
        return None;
    }

    // 2) Fetch the most recent commit touching this line range.
    //    Format: "<hash>|<author_email>|<author_iso_date>"
    let line_spec = format!("{start_line},{end_line}:{file_arg}");
    let out = Command::new("git")
        .current_dir(cwd)
        .args([
            "log",
            "-L",
            &line_spec,
            "--format=%H|%ae|%aI",
            "-s", // suppress diffs; we only want the header
            "-1",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&out.stdout);
    let header = stdout.lines().find(|l| !l.is_empty())?;
    let mut parts = header.splitn(3, '|');
    let hash = parts.next()?.trim();
    let email = parts.next()?.trim();
    let iso = parts.next()?.trim();

    let commit_dt: DateTime<Utc> = DateTime::parse_from_rfc3339(iso).ok()?.with_timezone(&Utc);
    let date_only = commit_dt.date_naive();
    let today: NaiveDate = Utc::now().date_naive();
    let days = (today - date_only).num_days().max(0) as u32;

    Some(ChunkBlame {
        commit_hash: hash.chars().take(12).collect(),
        author_email: email.to_string(),
        last_modified: date_only.to_string(),
        days_since_modified: days,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn temporal_decay_at_zero_is_one() {
        assert!((temporal_decay(0, 0.01) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn temporal_decay_half_life_around_69_days() {
        // ln(2)/0.01 ≈ 69.3
        let s = temporal_decay(69, 0.01);
        assert!(
            (s - 0.5).abs() < 0.1,
            "expected ~0.5 at 69 days, got {s}"
        );
    }

    #[test]
    fn temporal_decay_monotone_decreasing() {
        assert!(temporal_decay(0, 0.01) > temporal_decay(10, 0.01));
        assert!(temporal_decay(10, 0.01) > temporal_decay(100, 0.01));
    }

    #[test]
    fn blame_returns_none_for_untracked_tempfile() {
        // Write a file in /tmp that is definitely not tracked by git. Must
        // return None, never panic — non-git workflows depend on this.
        let mut path = std::env::temp_dir();
        path.push(format!(
            "trusty_search_blame_test_{}.txt",
            std::process::id()
        ));
        {
            let mut f = std::fs::File::create(&path).expect("create tmpfile");
            writeln!(f, "fn untracked() {{}}").unwrap();
        }
        let result = blame_chunk(&path, 1, 1);
        let _ = std::fs::remove_file(&path);
        assert!(
            result.is_none(),
            "untracked file must return None, got {result:?}"
        );
    }

    #[test]
    fn blame_returns_none_for_invalid_line_range() {
        let path = std::env::temp_dir().join("nonexistent.txt");
        assert!(blame_chunk(&path, 0, 0).is_none());
        assert!(blame_chunk(&path, 5, 3).is_none());
    }
}
