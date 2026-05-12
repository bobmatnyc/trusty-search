//! Reindex orchestration shared by `index`, `reindex`, `add`, `convert`, and
//! the doctor auto-repair path.
//!
//! Why: driving a daemon-side reindex involves several distinct pieces — the
//! progress UI (`ReindexUi`), the options and outcome record types, the SSE
//! event loop in `run_reindex_with`, the post-reindex health check, and the
//! companion file-level helpers (`index_single_file`, `add_path`,
//! `register_index_with_daemon{,_filtered}`, `fetch_chunk_count`). Keeping
//! them inline in `main.rs` pushed it past 2.7k lines; co-locating them here
//! drops `main.rs` to a thin dispatcher.
//! What: public surface mirrors the previous `main.rs` items so existing
//! callers in `commands/*` only have to change their `use` paths.
//! Test: `cargo test --workspace` — every reindex-driven integration test
//! continues to pass; the refactor is purely structural.

use super::daemon_utils::daemon_base_url;
use super::format::{fmt_elapsed, fmt_secs, format_with_commas};
use anyhow::Result;
use colored::Colorize;
use eventsource_stream::Eventsource;
use futures_util::stream::StreamExt;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::time::Duration;

/// Print per-subsystem indexing time breakdown after a successful reindex.
///
/// Why: gives the operator proof that each subsystem (parse, embed, BM25, KG)
/// actually ran and how long each took. The vector-count check is the
/// smoking-gun signal for the "embedder silently fell back to BM25" failure
/// mode — printed as a loud warning so it can never go unnoticed.
/// What: 4-line breakdown, plus a 5th warning line when `vector_count == 0`
/// despite non-zero chunks (the BM25-only-mode signal).
/// Test: call with synthetic timings where vector_count==0 and total_chunks>0;
/// assert the warning line is printed.
pub fn print_timing_breakdown(t: &ReindexTimings, total_chunks: u64) {
    println!(
        "  {} {:>7}  ({} chunks)",
        "Parse+chunk:".dimmed(),
        fmt_elapsed(t.parse_ms),
        format_with_commas(total_chunks),
    );
    if t.vector_count == 0 && total_chunks > 0 {
        println!(
            "  {} {}",
            "Embed (HNSW):".dimmed(),
            "SKIPPED (embedder unavailable — BM25-only mode)"
                .yellow()
                .bold(),
        );
    } else {
        println!(
            "  {} {:>7}  ({} vectors)",
            "Embed (HNSW):".dimmed(),
            fmt_elapsed(t.embed_ms),
            format_with_commas(t.vector_count),
        );
    }
    println!("  {} {:>7}", "BM25:".dimmed(), fmt_elapsed(t.bm25_ms));
    println!(
        "  {} {:>7}  ({} symbols, {} edges)",
        "KG:".dimmed(),
        fmt_elapsed(t.kg_ms),
        format_with_commas(t.symbol_count),
        format_with_commas(t.edge_count),
    );
}

/// Index a single file via the daemon's `/indexes/:id/index-file` endpoint.
pub async fn index_single_file(
    client: &reqwest::Client,
    base: &str,
    index_id: &str,
    file: &std::path::Path,
) -> Result<()> {
    let content = tokio::fs::read_to_string(file)
        .await
        .map_err(|e| anyhow::anyhow!("read {}: {e}", file.display()))?;
    let url = format!("{}/indexes/{}/index-file", base, index_id);
    let body = serde_json::json!({
        "path": file.display().to_string(),
        "content": content,
    });
    let resp = client.post(&url).json(&body).send().await?;
    if !resp.status().is_success() {
        anyhow::bail!("daemon returned {} for {}", resp.status(), url);
    }
    Ok(())
}

/// Handle `trusty-search add <path>`: a single file goes to `index-file`;
/// a directory walks `walk_source_files` and indexes every match.
pub async fn add_path(index_id: &str, path: &std::path::Path) -> Result<()> {
    let base = daemon_base_url();
    let client = trusty_common::server::daemon_http_client()?;

    if path.is_dir() {
        let walk = crate::service::walker::walk_source_files(path);
        println!(
            "{} [{}] indexing {} files under {}",
            "→".cyan(),
            index_id,
            walk.files.len(),
            path.display()
        );
        let mut ok = 0usize;
        let mut err = 0usize;
        for f in &walk.files {
            match index_single_file(&client, &base, index_id, f).await {
                Ok(()) => ok += 1,
                Err(e) => {
                    eprintln!("  {} {}: {e}", "⚠".yellow(), f.display());
                    err += 1;
                }
            }
        }
        println!("{} indexed {} files ({} errors)", "✓".green(), ok, err);
        Ok(())
    } else {
        index_single_file(&client, &base, index_id, path).await?;
        println!("{} [{}] {}", "→".cyan(), index_id, path.display());
        Ok(())
    }
}

/// Multi-line live progress display for a reindex.
///
/// Why: a single-line `ProgressBar` can't simultaneously show file progress,
/// chunk count, skipped count, speed, and elapsed/ETA. `MultiProgress` stacks
/// three lines (header / files bar / stats) that update independently.
///
/// Layout:
///   ⟳ Indexing <index>
///     [████████░░░░] 7,234/14,445 files (50%) — ETA 50s
///     Files: 7,234/14,445  Chunks: 58,402  Skipped: 12  Speed: 142 files/s  Elapsed: 50s  ETA: 50s
struct ReindexUi {
    /// Held to keep the MultiProgress draw target alive for the bars' lifetime.
    #[allow(dead_code)]
    multi: MultiProgress,
    header: ProgressBar,
    files: ProgressBar,
    stats: ProgressBar,
}

impl ReindexUi {
    fn new(index_id: &str) -> Self {
        let multi = MultiProgress::new();

        let header = multi.add(ProgressBar::new(1));
        if let Ok(s) = ProgressStyle::with_template("{spinner:.cyan} {msg}") {
            header.set_style(s);
        }
        header.set_message(format!("Indexing {}", index_id.bold()));
        header.enable_steady_tick(Duration::from_millis(120));

        let files = multi.add(ProgressBar::new(1));
        if let Ok(s) = ProgressStyle::with_template(
            "  [{bar:40.cyan/blue}] {pos}/{len} files ({percent}%) — ETA {eta}",
        ) {
            files.set_style(s.progress_chars("█░ "));
        }

        let stats = multi.add(ProgressBar::new(1));
        if let Ok(s) = ProgressStyle::with_template("  {msg}") {
            stats.set_style(s);
        }
        stats.set_message("Waiting for daemon…".to_string());

        Self {
            multi,
            header,
            files,
            stats,
        }
    }

    fn set_total(&self, total: u64) {
        self.files.set_length(total.max(1));
    }

    fn set_position(&self, indexed: u64) {
        self.files.set_position(indexed);
    }

    fn update_stats(&self, indexed: u64, total_chunks: u64, skipped: u64, elapsed_secs: u64) {
        let total = self.files.length().unwrap_or(0);
        let files_per_sec = indexed.checked_div(elapsed_secs).unwrap_or(0);
        let eta = if files_per_sec > 0 && total > indexed {
            fmt_secs((total - indexed) / files_per_sec)
        } else {
            "?".to_string()
        };
        self.stats.set_message(format!(
            "Files: {indexed}/{total}  Chunks: {chunks}  Skipped: {skipped}  Speed: {fps} files/s  Elapsed: {elapsed}  ETA: {eta}",
            indexed = format_with_commas(indexed),
            total = format_with_commas(total),
            chunks = format_with_commas(total_chunks),
            skipped = format_with_commas(skipped),
            fps = files_per_sec,
            elapsed = fmt_secs(elapsed_secs),
            eta = eta,
        ));
    }

    fn finish(self, final_msg: String) {
        self.files.finish_and_clear();
        self.stats.finish_and_clear();
        self.header.finish_with_message(final_msg);
    }

    fn abandon(self, final_msg: String) {
        self.files.abandon();
        self.stats.abandon();
        self.header.abandon_with_message(final_msg);
    }
}

/// Options controlling reindex CLI behaviour.
#[derive(Debug, Clone, Copy)]
pub struct ReindexOptions {
    /// After the reindex completes, fetch `/status` and issue a sanity-check
    /// search to verify the index is healthy. Enabled by `--force` to give
    /// the user a blue-green-style safety net.
    ///
    /// Note: the daemon's reindex is NOT atomic blue-green — it mutates the
    /// in-memory index in place via a write lock per batch (see
    /// `crates/trusty-search-service/src/reindex.rs::spawn_reindex` —
    /// `index_files_batch_no_rebuild` adds chunks per-batch). If verify fails
    /// after a `--force`, the index is already in its new (possibly broken)
    /// state. We surface that fact loudly so the user can manually re-run.
    pub verify_after: bool,
    /// Chunk count snapshot taken before the reindex started, used to print
    /// "(was N)" in the final verify message.
    pub prior_chunk_count: Option<u64>,
    /// Forwarded to the daemon as `"force": true` in the reindex kickoff body.
    /// Set by `index --force` so the daemon clears its content-hash cache and
    /// re-embeds every file (otherwise unchanged files would be skipped on a
    /// warm daemon and `--force` would have no effect).
    pub force: bool,
    /// Maximum wall-clock seconds to wait for the SSE reindex stream to emit
    /// a `complete` event. Default: 600. Use `--timeout 0` to disable (wait
    /// forever). When the deadline is exceeded the CLI prints a warning and
    /// exits; the daemon continues indexing in the background.
    pub timeout_secs: u64,
}

impl Default for ReindexOptions {
    fn default() -> Self {
        Self {
            verify_after: false,
            prior_chunk_count: None,
            force: false,
            timeout_secs: 600,
        }
    }
}

/// Outcome of a reindex run, captured for the post-verify step and the final
/// summary line. `indexed` includes skipped files (the daemon emits one
/// `indexed++` per file regardless of whether it was hashed-skip or re-embedded).
#[derive(Debug, Default, Clone, Copy)]
pub struct ReindexOutcome {
    pub indexed: u64,
    pub total_chunks: u64,
    pub skipped: u64,
    pub errors: u64,
    pub elapsed_ms: u64,
    pub completed: bool,
    /// Per-subsystem timings captured from the daemon's `complete` event
    /// `timings` payload. `None` when the daemon is an older version that
    /// didn't emit timings — caller renders a single-line summary in that case.
    pub timings: Option<ReindexTimings>,
}

/// Per-subsystem indexing timings parsed from the SSE `complete` event.
///
/// Why: gives the user proof that each subsystem ran and how long each took.
/// `vector_count == 0` with `total_chunks > 0` is the smoking-gun signal that
/// the embedder silently fell back to BM25-only — surfaced as a warning in the
/// CLI breakdown so this regression can never go unnoticed.
#[derive(Debug, Default, Clone, Copy)]
pub struct ReindexTimings {
    pub parse_ms: u64,
    pub embed_ms: u64,
    pub bm25_ms: u64,
    #[allow(dead_code)]
    pub vector_upsert_ms: u64,
    pub kg_ms: u64,
    pub vector_count: u64,
    pub symbol_count: u64,
    pub edge_count: u64,
}

/// Plain reindex (no post-verify). Used by the non-force `index` command, the
/// bare `reindex` command, and the doctor auto-repair path. The daemon's
/// hash-skip optimization (see `reindex.rs::hash_content`) means unchanged
/// files are cheap, so calling this even when nothing changed is fine.
///
/// `timeout_secs` caps how long the CLI waits for the SSE stream's `complete`
/// event. 0 means no limit (wait forever). Default for callers that don't have
/// an explicit user-supplied value: 600.
pub async fn run_reindex(
    index_id: &str,
    root_path: &std::path::Path,
    timeout_secs: u64,
) -> Result<()> {
    run_reindex_with(
        index_id,
        root_path,
        ReindexOptions {
            timeout_secs,
            ..ReindexOptions::default()
        },
    )
    .await
    .map(|_| ())
}

/// `index --force` reindex: snapshot the prior chunk count, kick off a full
/// reindex, and run a post-reindex health check. Exits 1 if the new index
/// looks unhealthy (no chunks or empty sanity query).
pub async fn run_reindex_force(
    index_id: &str,
    root_path: &std::path::Path,
    timeout_secs: u64,
) -> Result<()> {
    let prior = fetch_chunk_count(index_id).await;
    let opts = ReindexOptions {
        verify_after: true,
        prior_chunk_count: prior,
        force: true,
        timeout_secs,
    };
    run_reindex_with(index_id, root_path, opts)
        .await
        .map(|_| ())
}

/// Drive a reindex: POST /reindex, then connect to the SSE stream and render
/// progress with an indicatif `MultiProgress` layout (header + files bar +
/// stats line). A wall-clock ticker keeps the stats line moving even when
/// SSE events are sparse (e.g. the embedder is mid-batch).
pub async fn run_reindex_with(
    index_id: &str,
    root_path: &std::path::Path,
    opts: ReindexOptions,
) -> Result<ReindexOutcome> {
    let base = daemon_base_url();
    let client = trusty_common::server::daemon_http_client()?;

    let kickoff_url = format!("{}/indexes/{}/reindex", base, index_id);
    let kickoff_body = serde_json::json!({
        "root_path": root_path,
        "force": opts.force,
    });
    let kickoff = client
        .post(&kickoff_url)
        .json(&kickoff_body)
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("could not reach daemon at {base}: {e}"))?;

    if kickoff.status() == reqwest::StatusCode::NOT_FOUND {
        anyhow::bail!(
            "index '{}' is not registered on the daemon — run `trusty-search index` first",
            index_id
        );
    }
    if !kickoff.status().is_success() {
        anyhow::bail!("daemon returned {} for reindex kickoff", kickoff.status());
    }

    let kickoff_body: serde_json::Value = kickoff
        .json()
        .await
        .unwrap_or_else(|_| serde_json::json!({}));
    let stream_path = kickoff_body
        .get("stream_url")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("/indexes/{}/reindex/stream", index_id));
    let stream_url = format!("{}{}", base, stream_path);

    // SSE streams must NOT use the short request timeout from
    // `daemon_http_client()` (currently 5s) — a large repo reindex can run for
    // minutes. We build a dedicated client with only a connect timeout so the
    // byte stream stays open until the daemon emits the `complete` event.
    //
    // The per-request reqwest timeout only governs the *connection* phase here;
    // we handle the overall stream deadline ourselves below via
    // `tokio::time::timeout` so we can print a friendly warning instead of a
    // raw timeout error.
    let sse_client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(5))
        .timeout(Duration::MAX)
        .build()
        .map_err(|e| anyhow::anyhow!("could not build SSE client: {e}"))?;
    let resp = sse_client
        .get(&stream_url)
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("could not connect to SSE stream {stream_url}: {e}"))?;
    if !resp.status().is_success() {
        anyhow::bail!(
            "reindex stream returned {} — daemon may be an older version that doesn't support /reindex/stream",
            resp.status()
        );
    }
    // MultiProgress UI: header + files bar + stats line. Built eagerly so
    // the user sees something during the 1–2 second daemon warmup before the
    // first SSE event arrives.
    let ui = ReindexUi::new(index_id);

    // Atomics shared with the wall-clock ticker. The ticker refreshes the
    // stats line every second so the user sees movement even when the SSE
    // stream is idle (e.g. mid-batch embedding of 256 chunks).
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::Arc as StdArc;
    let started = std::time::Instant::now();
    let indexed_now = StdArc::new(AtomicU64::new(0));
    let chunks_now = StdArc::new(AtomicU64::new(0));
    let skipped_now = StdArc::new(AtomicU64::new(0));
    let tick_done = StdArc::new(AtomicBool::new(false));

    let ticker = {
        let indexed_now = indexed_now.clone();
        let chunks_now = chunks_now.clone();
        let skipped_now = skipped_now.clone();
        let tick_done = tick_done.clone();
        let stats_bar = ui.stats.clone();
        let files_bar = ui.files.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(1));
            interval.tick().await; // discard immediate tick
            loop {
                interval.tick().await;
                if tick_done.load(Ordering::Acquire) {
                    break;
                }
                let elapsed = started.elapsed().as_secs();
                let indexed = indexed_now.load(Ordering::Acquire);
                let chunks = chunks_now.load(Ordering::Acquire);
                let skipped = skipped_now.load(Ordering::Acquire);
                let fps = indexed.checked_div(elapsed).unwrap_or(0);
                let total = files_bar.length().unwrap_or(0);
                let eta = if fps > 0 && total > indexed {
                    fmt_secs((total - indexed) / fps)
                } else {
                    "?".to_string()
                };
                stats_bar.set_message(format!(
                    "Files: {indexed}/{total}  Chunks: {chunks}  Skipped: {skipped}  Speed: {fps} files/s  Elapsed: {elapsed}s  ETA: {eta}",
                    indexed = format_with_commas(indexed),
                    total = format_with_commas(total),
                    chunks = format_with_commas(chunks),
                    skipped = format_with_commas(skipped),
                    fps = fps,
                    elapsed = elapsed,
                    eta = eta,
                ));
            }
        })
    };

    let mut outcome = ReindexOutcome::default();
    let mut done = false;
    let mut timed_out = false;

    // Optional wall-clock deadline for the SSE stream. `timeout_secs == 0`
    // means wait forever (legacy behaviour). Otherwise each `stream.next()`
    // is raced against `tokio::time::sleep_until(deadline)` via
    // `tokio::select!`. When the sleep wins we set `timed_out = true` and
    // break so the post-loop path can print the canonical warning.
    // The daemon continues indexing in the background.
    let deadline: Option<tokio::time::Instant> = if opts.timeout_secs > 0 {
        Some(tokio::time::Instant::now() + Duration::from_secs(opts.timeout_secs))
    } else {
        None
    };

    // `eventsource-stream` handles SSE framing. The daemon emits these event
    // types (see `crates/trusty-search-service/src/reindex.rs::spawn_reindex`):
    //   - start:    total_files, index_id, root_path
    //   - batch:    batch_files, batch_chunks, indexed, total_files, elapsed_ms
    //   - skip:     file, indexed, total_files (hash matched OR minified)
    //   - error:    message, file (or files for a batch failure)
    //   - complete: indexed, total_chunks, skipped, errors, elapsed_ms
    let byte_stream = resp.bytes_stream();
    let stream = byte_stream.eventsource();
    tokio::pin!(stream);
    while !done {
        // Race the next SSE event against the optional deadline. When the
        // deadline fires `timed_out` is set and we break cleanly; the
        // post-loop section emits the warning and returns Ok.
        let maybe_event = if let Some(dl) = deadline {
            tokio::select! {
                biased;
                ev = stream.next() => ev,
                _ = tokio::time::sleep_until(dl) => {
                    timed_out = true;
                    break;
                }
            }
        } else {
            stream.next().await
        };
        let event = match maybe_event {
            Some(Ok(e)) => e,
            Some(Err(e)) => {
                ui.stats
                    .println(format!("{} stream read error: {e}", "⚠".yellow()));
                break;
            }
            None => break,
        };

        let evt: serde_json::Value = match serde_json::from_str(event.data.trim()) {
            Ok(v) => v,
            Err(_) => continue,
        };
        match evt.get("event").and_then(|v| v.as_str()) {
            Some("start") => {
                let total = evt.get("total_files").and_then(|v| v.as_u64()).unwrap_or(0);
                ui.set_total(total);
            }
            Some("batch") => {
                let indexed = evt.get("indexed").and_then(|v| v.as_u64()).unwrap_or(0);
                let batch_chunks = evt
                    .get("batch_chunks")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                let total = evt.get("total_files").and_then(|v| v.as_u64()).unwrap_or(0);
                if total > 0 && ui.files.length() != Some(total.max(1)) {
                    ui.set_total(total);
                }
                indexed_now.store(indexed, Ordering::Release);
                let new_chunks =
                    chunks_now.fetch_add(batch_chunks, Ordering::AcqRel) + batch_chunks;
                ui.set_position(indexed);
                ui.update_stats(
                    indexed,
                    new_chunks,
                    skipped_now.load(Ordering::Acquire),
                    started.elapsed().as_secs(),
                );
            }
            Some("skip") => {
                let indexed = evt.get("indexed").and_then(|v| v.as_u64()).unwrap_or(0);
                indexed_now.store(indexed, Ordering::Release);
                let skipped = skipped_now.fetch_add(1, Ordering::AcqRel) + 1;
                ui.set_position(indexed);
                ui.update_stats(
                    indexed,
                    chunks_now.load(Ordering::Acquire),
                    skipped,
                    started.elapsed().as_secs(),
                );
            }
            Some("complete") => {
                outcome.indexed = evt.get("indexed").and_then(|v| v.as_u64()).unwrap_or(0);
                outcome.total_chunks = evt
                    .get("total_chunks")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                outcome.skipped = evt
                    .get("skipped")
                    .and_then(|v| v.as_u64())
                    .unwrap_or_else(|| skipped_now.load(Ordering::Acquire));
                outcome.errors = evt.get("errors").and_then(|v| v.as_u64()).unwrap_or(0);
                outcome.elapsed_ms = evt.get("elapsed_ms").and_then(|v| v.as_u64()).unwrap_or(0);
                // Per-subsystem timings (added in 0.3.11). Absent when talking
                // to an older daemon — outcome.timings stays `None` and the
                // CLI falls back to the legacy single-line summary.
                if let Some(t) = evt.get("timings") {
                    let get = |k: &str| t.get(k).and_then(|v| v.as_u64()).unwrap_or(0);
                    outcome.timings = Some(ReindexTimings {
                        parse_ms: get("parse_ms"),
                        embed_ms: get("embed_ms"),
                        bm25_ms: get("bm25_ms"),
                        vector_upsert_ms: get("vector_upsert_ms"),
                        kg_ms: get("kg_ms"),
                        vector_count: get("vector_count"),
                        symbol_count: get("symbol_count"),
                        edge_count: get("edge_count"),
                    });
                }
                outcome.completed = true;
                ui.set_position(outcome.indexed);
                done = true;
            }
            Some("error") => {
                let msg = evt
                    .get("message")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");
                let file = evt.get("file").and_then(|v| v.as_str()).unwrap_or("");
                ui.stats
                    .println(format!("{}  {}: {}", "⚠".yellow(), file, msg));
            }
            _ => {}
        }
    }

    // Stop the ticker before finishing the UI so it doesn't overwrite the
    // final message during the brief window between finish() and shutdown.
    tick_done.store(true, Ordering::Release);
    let _ = ticker.await;

    if timed_out {
        // The SSE deadline fired before the daemon emitted `complete`. The
        // daemon is still indexing in the background. Print the canonical
        // warning (exact text the issue tracker refers to) and return Ok so
        // callers don't treat this as a hard error.
        ui.abandon(format!(
            "{} trusty-search index timed out after {}s — continuing; re-run later if needed",
            "⚠".yellow(),
            opts.timeout_secs,
        ));
        eprintln!(
            "{} Daemon is still indexing in the background. \
             Use `trusty-search status` or re-run `trusty-search index` to check progress. \
             Pass `--timeout <seconds>` to wait longer (e.g. `--timeout 1200`).",
            "ℹ".cyan()
        );
        return Ok(outcome);
    }

    if !outcome.completed {
        ui.abandon(format!(
            "{} Reindex stream ended without completion event",
            "⚠".yellow()
        ));
        anyhow::bail!("reindex did not complete");
    }

    // Final headline. We distinguish three cases:
    //   1. errors > 0          → show error count + unchanged count
    //   2. nothing changed     → "is up to date" message (Improvement 3)
    //   3. some files changed  → "Indexed N changed files" with unchanged tally
    let elapsed = fmt_elapsed(outcome.elapsed_ms);
    let changed = outcome.indexed.saturating_sub(outcome.skipped);
    let final_msg = if outcome.errors > 0 {
        format!(
            "{} Indexed {} files → {} chunks  [took {}, {} errors, {} unchanged]",
            "✓".green(),
            format_with_commas(changed),
            format_with_commas(outcome.total_chunks),
            elapsed,
            outcome.errors,
            format_with_commas(outcome.skipped),
        )
    } else if changed == 0 && outcome.indexed > 0 {
        format!(
            "{} '{}' is up to date ({} chunks, {} files — no changes detected)  [took {}]",
            "✓".green(),
            index_id,
            format_with_commas(outcome.total_chunks),
            format_with_commas(outcome.indexed),
            elapsed,
        )
    } else {
        format!(
            "{} Indexed {} changed file{} → {} chunks  [took {}, {} unchanged]",
            "✓".green(),
            format_with_commas(changed),
            if changed == 1 { "" } else { "s" },
            format_with_commas(outcome.total_chunks),
            elapsed,
            format_with_commas(outcome.skipped),
        )
    };
    ui.finish(final_msg);

    // ── Per-subsystem timing breakdown (issue: silent BM25 fallback) ──────
    // We render this AFTER `ui.finish` so the indicatif `MultiProgress`
    // doesn't redraw over our printed lines. Skipped entirely when talking
    // to a daemon older than 0.3.11 (no `timings` block in the SSE
    // `complete` event).
    if let Some(t) = outcome.timings {
        print_timing_breakdown(&t, outcome.total_chunks);
    }

    // ── Post-reindex health check (blue-green safety net) ─────────────────
    if opts.verify_after {
        verify_reindex_health(&client, &base, index_id, &outcome, opts.prior_chunk_count).await?;
    }

    Ok(outcome)
}

/// After a `--force` reindex, fetch the new chunk count and run a sanity
/// query. Exits 1 if either looks wrong.
///
/// Why: the daemon's reindex mutates the in-memory `CodeIndexer` in place
/// (no shadow slot — see `reindex.rs::spawn_reindex`, which writes each batch
/// directly into the live indexer via `index_files_batch_no_rebuild`). If the
/// rebuild produces a broken index, the only signal the user has is "search
/// returns nothing" hours later. This check surfaces that immediately.
async fn verify_reindex_health(
    client: &reqwest::Client,
    base: &str,
    index_id: &str,
    outcome: &ReindexOutcome,
    prior: Option<u64>,
) -> Result<()> {
    // 1) Chunk count via /status.
    let status_url = format!("{}/indexes/{}/status", base, index_id);
    let new_chunks = match client.get(&status_url).send().await {
        Ok(r) if r.status().is_success() => r
            .json::<serde_json::Value>()
            .await
            .ok()
            .and_then(|v| v.get("chunk_count").and_then(|n| n.as_u64()))
            .unwrap_or(0),
        _ => 0,
    };

    // 2) Sanity query: pick something that hits virtually any source tree
    //    (`fn` matches Rust; `function` JS/TS; `def` Python; etc.). One hit
    //    in any single probe is enough to consider the index queryable.
    let search_url = format!("{}/indexes/{}/search", base, index_id);
    let probes = ["fn", "function", "def", "class", "the"];
    let mut got_hit = false;
    for probe in probes {
        let body = serde_json::json!({ "text": probe, "top_k": 1 });
        if let Ok(resp) = client.post(&search_url).json(&body).send().await {
            if resp.status().is_success() {
                if let Ok(json) = resp.json::<serde_json::Value>().await {
                    let n = json
                        .get("results")
                        .and_then(|r| r.as_array())
                        .map(|a| a.len())
                        .unwrap_or(0);
                    if n > 0 {
                        got_hit = true;
                        break;
                    }
                }
            }
        }
    }

    let healthy = new_chunks > 0 && got_hit && outcome.errors == 0;
    let was = prior
        .map(|p| format!(" (was {})", format_with_commas(p)))
        .unwrap_or_default();
    if healthy {
        println!(
            "{} Reindex complete: {} chunks{}",
            "✓".green(),
            format_with_commas(new_chunks),
            was
        );
        Ok(())
    } else {
        anyhow::bail!(
            "Reindex produced unhealthy index: {} chunks{}, sanity query {} — old index NOT preserved (daemon reindex is in-place; see crates/trusty-search-service/src/reindex.rs)",
            format_with_commas(new_chunks),
            was,
            if got_hit { "ok" } else { "returned 0 results" }
        );
    }
}

/// Register an index with the daemon (idempotent).
///
/// Why: factored out of `Init` and `Index` because both flows need the same
/// "POST /indexes, parse `created`" dance.
/// What: returns `Ok((created, daemon_reachable))`. `daemon_reachable=false`
/// surfaces network failures distinctly from "registered but already existed".
pub async fn register_index_with_daemon(
    index_name: &str,
    project_path: &std::path::Path,
) -> Result<(bool, bool)> {
    register_index_with_daemon_filtered(index_name, project_path, &RegisterFilters::default()).await
}

/// Optional repo-config filters carried in `POST /indexes` request bodies.
///
/// Why: `trusty-search.yaml` declares per-index filter sets (`paths`,
/// `exclude`, `languages`, `domain_terms`). The CLI loads the YAML and
/// forwards the resolved values to the daemon when registering each
/// index so the daemon stores them on the `IndexHandle` and applies them
/// to subsequent reindex + search calls.
/// What: thin struct carrying the four fields. `Default` = empty everywhere,
/// which keeps the original single-index path unchanged.
/// Test: `commands::index::handle_index` populates this from `IndexConfig`.
#[derive(Debug, Default)]
pub struct RegisterFilters {
    pub include_paths: Vec<String>,
    pub exclude_globs: Vec<String>,
    pub extensions: Vec<String>,
    pub domain_terms: Vec<String>,
}

/// Variant of [`register_index_with_daemon`] that forwards filter/domain
/// fields in the request body so the daemon can store them on the handle.
pub async fn register_index_with_daemon_filtered(
    index_name: &str,
    project_path: &std::path::Path,
    filters: &RegisterFilters,
) -> Result<(bool, bool)> {
    let base = daemon_base_url();
    let client = trusty_common::server::daemon_http_client()?;
    let create_url = format!("{}/indexes", base);
    let mut create_body = serde_json::json!({
        "id": index_name,
        "root_path": project_path,
    });
    // Only attach filter fields when non-empty. This keeps the wire format
    // identical for the single-index case (no `trusty-search.yaml`) and lets
    // older daemons reject unknown fields cleanly (they're `Option<Vec<…>>`
    // on the daemon side, so this is forward-compatible anyway).
    if !filters.include_paths.is_empty() {
        create_body["include_paths"] = serde_json::json!(filters.include_paths);
    }
    if !filters.exclude_globs.is_empty() {
        create_body["exclude_globs"] = serde_json::json!(filters.exclude_globs);
    }
    if !filters.extensions.is_empty() {
        create_body["extensions"] = serde_json::json!(filters.extensions);
    }
    if !filters.domain_terms.is_empty() {
        create_body["domain_terms"] = serde_json::json!(filters.domain_terms);
    }
    match client.post(&create_url).json(&create_body).send().await {
        Ok(resp) if resp.status().is_success() => {
            let body: serde_json::Value =
                resp.json().await.unwrap_or_else(|_| serde_json::json!({}));
            let created = body
                .get("created")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            Ok((created, true))
        }
        Ok(resp) => {
            anyhow::bail!("daemon returned {} for POST /indexes", resp.status());
        }
        Err(_) => Ok((false, false)),
    }
}

/// Fetch chunk count for an index via /status. Returns `None` if the daemon
/// is unreachable or the index isn't registered.
pub async fn fetch_chunk_count(index_id: &str) -> Option<u64> {
    let base = daemon_base_url();
    let url = format!("{}/indexes/{}/status", base, index_id);
    let client = trusty_common::server::daemon_http_client().ok()?;
    let resp = client.get(&url).send().await.ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let body: serde_json::Value = resp.json().await.ok()?;
    body.get("chunk_count").and_then(|v| v.as_u64())
}
