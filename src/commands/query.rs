//! Handler for `trusty-search query`.

use crate::daemon_base_url;
use anyhow::Result;
use colored::Colorize;

/// Resolve which index to search against the daemon. Precedence:
/// `--index` flag > `--indexes <single name>` > auto-resolve via `/indexes`
/// (only when exactly one index is registered).
///
/// Why: factored out so the main query path stays linear. Bails (exit 1)
/// with a helpful message when ambiguous.
async fn resolve_target_id(
    explicit_index: &Option<String>,
    indexes: &str,
    client: &reqwest::Client,
    base: &str,
) -> String {
    if let Some(id) = explicit_index {
        return id.clone();
    }
    if indexes != "*" && !indexes.contains(',') {
        return indexes.to_string();
    }
    // Try to resolve by listing.
    let resp = client.get(format!("{}/indexes", base)).send().await;
    match resp {
        Ok(r) if r.status().is_success() => {
            let body: serde_json::Value = r.json().await.unwrap_or_else(|_| serde_json::json!({}));
            let empty: Vec<serde_json::Value> = Vec::new();
            let names: Vec<String> = body
                .get("indexes")
                .and_then(|v| v.as_array())
                .unwrap_or(&empty)
                .iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect();
            if names.len() == 1 {
                names.into_iter().next().unwrap()
            } else {
                eprintln!(
                    "{} Multiple indexes registered — please pass --index <id>: {}",
                    "✗".red(),
                    names.join(", ")
                );
                std::process::exit(1);
            }
        }
        _ => {
            eprintln!("{} could not reach daemon at {}", "✗".red(), base);
            std::process::exit(1);
        }
    }
}

/// Render the human-readable result list for a `query` response.
fn render_text(query: &str, target_id: &str, body_json: &serde_json::Value, full: bool) {
    let empty: Vec<serde_json::Value> = Vec::new();
    let results = body_json
        .get("results")
        .and_then(|v| v.as_array())
        .unwrap_or(&empty);
    let intent = body_json
        .get("intent")
        .and_then(|v| v.as_str())
        .unwrap_or("?");
    let latency = body_json
        .get("latency_ms")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    println!(
        "{} [{}] {} {}",
        "→".cyan(),
        target_id.dimmed(),
        query.bold(),
        format!(
            "(intent={}, {}ms, {} results)",
            intent,
            latency,
            results.len()
        )
        .dimmed()
    );
    if results.is_empty() {
        println!("  {}", "(no matches)".dimmed());
    }
    for (i, r) in results.iter().enumerate() {
        let file = r.get("file").and_then(|v| v.as_str()).unwrap_or("?");
        let start = r.get("start_line").and_then(|v| v.as_u64()).unwrap_or(0);
        let end = r.get("end_line").and_then(|v| v.as_u64()).unwrap_or(0);
        let score = r.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let reason = r
            .get("match_reason")
            .and_then(|v| v.as_str())
            .unwrap_or("?");
        println!(
            "[{}] {}:{}-{}  {}",
            i + 1,
            file,
            start,
            end,
            format!("(score: {:.3}, {})", score, reason).dimmed()
        );
        let snippet = if full {
            r.get("content").and_then(|v| v.as_str()).unwrap_or("")
        } else {
            r.get("compact_snippet")
                .and_then(|v| v.as_str())
                .or_else(|| r.get("content").and_then(|v| v.as_str()))
                .unwrap_or("")
        };
        for line in snippet.lines().take(if full { usize::MAX } else { 7 }) {
            println!("    {}", line);
        }
        if !full && snippet.lines().count() > 7 {
            println!("    {}", "...".dimmed());
        }
    }
}

/// Why: extracted from `main()`; behaviour unchanged.
/// What: resolves target index, POSTs to `/indexes/<id>/search`, then renders
/// JSON or the compact text view.
/// Test: `cargo run -- query "fn main" -k 5` returns 5 hits for a registered
/// repo.
pub async fn handle_query(
    explicit_index: &Option<String>,
    global_json: bool,
    query: String,
    indexes: String,
    top_k: usize,
    full: bool,
) -> Result<()> {
    let base = daemon_base_url();
    crate::commands::daemon_guard::ensure_daemon_running_or_exit(&base).await;
    let client = trusty_common::server::daemon_http_client()?;

    let target_id = resolve_target_id(explicit_index, &indexes, &client, &base).await;

    let url = format!("{}/indexes/{}/search", base, target_id);
    let body = serde_json::json!({"text": query, "top_k": top_k});
    let resp = client.post(&url).json(&body).send().await;
    let body_json: serde_json::Value = match resp {
        Ok(r) if r.status().is_success() => {
            r.json().await.unwrap_or_else(|_| serde_json::json!({}))
        }
        Ok(r) if r.status() == reqwest::StatusCode::NOT_FOUND => {
            eprintln!("{} index '{}' not found on daemon", "✗".red(), target_id);
            std::process::exit(1);
        }
        Ok(r) => {
            eprintln!("{} daemon returned {}", "✗".red(), r.status());
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("{} could not reach daemon at {}: {e}", "✗".red(), base);
            std::process::exit(1);
        }
    };

    if global_json {
        println!("{}", body_json);
    } else {
        render_text(&query, &target_id, &body_json, full);
    }
    Ok(())
}
