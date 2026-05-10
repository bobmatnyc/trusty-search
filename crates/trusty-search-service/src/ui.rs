//! Embedded Svelte UI server.
//!
//! Why: We ship a single `trusty-search` binary that serves the management
//! UI without requiring users to run a separate static-file server. The
//! Svelte build output (`ui/dist/`) is baked into the binary at compile time
//! via `include_dir!`, so the daemon is fully self-contained.
//!
//! What: Two route handlers serving the SPA:
//!   - `GET /ui`       → index.html with runtime config injected
//!   - `GET /ui/*path` → static asset, falling back to index.html for
//!     client-side routes (e.g. `/ui/search`).
//!
//! Plus the OpenRouter-proxying `POST /chat` endpoint.
//!
//! Test: `cargo test -p trusty-search-service ui::` exercises the path
//! resolver against the embedded directory.

use axum::{
    body::Body,
    extract::{Path as AxumPath, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use include_dir::{include_dir, Dir};
use serde::Deserialize;
use std::sync::Arc;

use crate::server::SearchAppState;

/// Why: `include_dir!` walks at compile time and embeds every byte. We point
/// it at `../../../ui/dist` (relative to this crate's `src/`).
/// What: `UI_DIR` is a static reference to the compiled tree. If `ui/dist`
/// doesn't exist at compile time, the macro embeds an empty directory and
/// the handlers return 404 — the daemon still builds and runs.
/// Test: `cargo build` after `npm run build` produces a binary that, when
/// run, serves `/ui` with the SPA shell.
static UI_DIR: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/../../ui/dist");

/// Inject runtime configuration into index.html before serving.
///
/// Why: The browser needs to know (a) the daemon port (so it can reach
/// the API at the right host:port even when the UI is opened directly via
/// `file://` for local dev), and (b) whether the OpenRouter chat lane is
/// enabled. We can't bake these in at compile time because they're chosen
/// at runtime.
/// What: Replaces the placeholder boot script with one that sets both
/// globals before the bundle loads.
/// Test: After serving, `view-source:` shows the correct port literal.
fn inject_runtime_config(html: &str, port: u16, openrouter_enabled: bool) -> String {
    let inject = format!(
        "<script>\n\
         window.__DAEMON_PORT__ = {};\n\
         window.__OPENROUTER_ENABLED__ = {};\n\
         </script>",
        port,
        if openrouter_enabled { "true" } else { "false" }
    );
    // Insert just before </head> so the inline script runs before the
    // bundle. If </head> isn't found (shouldn't happen with vite output),
    // prepend to keep behavior safe.
    if let Some(idx) = html.find("</head>") {
        let mut out = String::with_capacity(html.len() + inject.len());
        out.push_str(&html[..idx]);
        out.push_str(&inject);
        out.push_str(&html[idx..]);
        out
    } else {
        format!("{inject}{html}")
    }
}

/// Serve `index.html` at `/ui`.
pub async fn ui_index_handler(State(state): State<Arc<SearchAppState>>) -> Response {
    serve_index(&state).await
}

/// Serve any file under `/ui/*path`, falling back to index.html for SPA
/// routes that don't map to a real file.
pub async fn ui_asset_handler(
    State(state): State<Arc<SearchAppState>>,
    AxumPath(path): AxumPath<String>,
) -> Response {
    // Strip leading slashes — include_dir paths are relative.
    let trimmed = path.trim_start_matches('/');
    if let Some(file) = UI_DIR.get_file(trimmed) {
        let mime = mime_for(trimmed);
        return Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, mime)
            .header(header::CACHE_CONTROL, cache_control_for(trimmed))
            .body(Body::from(file.contents()))
            .unwrap();
    }
    // SPA fallback.
    serve_index(&state).await
}

async fn serve_index(state: &SearchAppState) -> Response {
    let Some(index_file) = UI_DIR.get_file("index.html") else {
        return (
            StatusCode::NOT_FOUND,
            "UI assets not bundled — run `npm run build` in ui/ before `cargo build`.",
        )
            .into_response();
    };
    let html_bytes = index_file.contents();
    let html = std::str::from_utf8(html_bytes).unwrap_or_default();
    let port = state.daemon_port.unwrap_or(7878);
    let body = inject_runtime_config(html, port, state.openrouter_enabled);
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/html; charset=utf-8")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from(body))
        .unwrap()
}

fn mime_for(path: &str) -> &'static str {
    let ext = path.rsplit('.').next().unwrap_or("");
    match ext {
        "html" => "text/html; charset=utf-8",
        "js" | "mjs" => "application/javascript; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "json" => "application/json",
        "svg" => "image/svg+xml",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "ico" => "image/x-icon",
        "woff" => "font/woff",
        "woff2" => "font/woff2",
        "map" => "application/json",
        _ => "application/octet-stream",
    }
}

fn cache_control_for(path: &str) -> &'static str {
    // Vite hashes asset filenames, so /assets/* is safe to cache aggressively.
    if path.starts_with("assets/") {
        "public, max-age=31536000, immutable"
    } else {
        "no-cache"
    }
}

// ── Chat endpoint ──────────────────────────────────────────────────────────

/// Inbound payload for `POST /chat`.
///
/// Why: The browser doesn't see the OpenRouter API key — the daemon proxies
/// the request server-side using `OPENROUTER_API_KEY` from the environment.
/// What: Caller supplies `index_id` (the collection to ground the question
/// in), the new `message`, and prior `history`. The handler runs a search
/// to gather context, then forwards a chat completion request.
/// Test: With `OPENROUTER_API_KEY` unset → returns 503 + `{error}`.
#[derive(Deserialize)]
pub struct ChatRequest {
    pub index_id: String,
    pub message: String,
    #[serde(default)]
    pub history: Vec<ChatMessage>,
}

#[derive(Deserialize, serde::Serialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

pub async fn chat_handler(
    State(state): State<Arc<SearchAppState>>,
    Json(req): Json<ChatRequest>,
) -> Response {
    let Some(api_key) = std::env::var("OPENROUTER_API_KEY").ok() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": "OpenRouter not configured"})),
        )
            .into_response();
    };

    // 1. Search the index for context (best-effort — empty context is fine).
    let context_snippet = match search_for_context(state.as_ref(), &req.index_id, &req.message).await {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!("chat: search for context failed: {e}");
            String::new()
        }
    };

    // 2. Build messages: system prompt with context, then history, then new user message.
    let system = format!(
        "You are a code-aware assistant for the '{}' codebase. \
         Answer the user's question using the search results below as primary context. \
         If the context doesn't cover the question, say so honestly.\n\n\
         === Search Context ===\n{}\n=== End Context ===",
        req.index_id, context_snippet
    );

    let mut messages: Vec<serde_json::Value> = Vec::new();
    messages.push(serde_json::json!({"role": "system", "content": system}));
    for m in &req.history {
        messages.push(serde_json::json!({"role": m.role, "content": m.content}));
    }
    messages.push(serde_json::json!({"role": "user", "content": req.message}));

    // 3. Forward to OpenRouter.
    let client = reqwest::Client::new();
    let body = serde_json::json!({
        "model": "anthropic/claude-3.5-sonnet",
        "messages": messages,
    });
    let resp = match client
        .post("https://openrouter.ai/api/v1/chat/completions")
        .header("Authorization", format!("Bearer {api_key}"))
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::BAD_GATEWAY,
                Json(serde_json::json!({"error": format!("openrouter request failed: {e}")})),
            )
                .into_response()
        }
    };

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return (
            StatusCode::BAD_GATEWAY,
            Json(serde_json::json!({"error": format!("openrouter {status}: {text}")})),
        )
            .into_response();
    }

    let json: serde_json::Value = match resp.json().await {
        Ok(j) => j,
        Err(e) => {
            return (
                StatusCode::BAD_GATEWAY,
                Json(serde_json::json!({"error": format!("openrouter response decode: {e}")})),
            )
                .into_response()
        }
    };

    let reply = json
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .unwrap_or("(no reply)")
        .to_string();

    Json(serde_json::json!({ "reply": reply, "raw": json })).into_response()
}

async fn search_for_context(
    state: &SearchAppState,
    index_id: &str,
    query: &str,
) -> Result<String, String> {
    use trusty_search_core::{indexer::SearchQuery, registry::IndexId};
    let id = IndexId::new(index_id.to_string());
    let handle = state.registry.get(&id).ok_or_else(|| "index not found".to_string())?;
    let q = SearchQuery {
        text: query.to_string(),
        top_k: 5,
        expand_graph: true,
        compact: true,
    };
    let indexer = handle.indexer.read().await;
    let results = indexer.search(&q).await.map_err(|e| e.to_string())?;
    let mut out = String::new();
    for (i, r) in results.iter().enumerate() {
        out.push_str(&format!(
            "\n--- Result {} ({}:{}-{}, score {:.3}) ---\n",
            i + 1,
            r.file,
            r.start_line,
            r.end_line,
            r.score
        ));
        let snippet = r
            .compact_snippet
            .as_deref()
            .unwrap_or(&r.content);
        out.push_str(snippet);
        out.push('\n');
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Why: Verify the runtime config injection lands before </head> so the
    /// inline globals execute before the bundle.
    /// What: Inject into a minimal HTML and assert the script is present.
    /// Test: this test.
    #[test]
    fn inject_runtime_config_inserts_before_head_close() {
        let html = "<html><head><title>x</title></head><body></body></html>";
        let out = inject_runtime_config(html, 7878, true);
        let script_idx = out.find("__DAEMON_PORT__").expect("port global injected");
        let head_close = out.find("</head>").expect("head close present");
        assert!(script_idx < head_close, "script must be inside <head>");
        assert!(out.contains("window.__OPENROUTER_ENABLED__ = true"));
    }

    #[test]
    fn inject_runtime_config_handles_missing_head() {
        let html = "<html><body></body></html>";
        let out = inject_runtime_config(html, 1234, false);
        assert!(out.starts_with("<script>"));
        assert!(out.contains("window.__DAEMON_PORT__ = 1234"));
    }

    #[test]
    fn mime_for_known_extensions() {
        assert_eq!(mime_for("index.html"), "text/html; charset=utf-8");
        assert_eq!(mime_for("a/b.js"), "application/javascript; charset=utf-8");
        assert_eq!(mime_for("a/b.css"), "text/css; charset=utf-8");
        assert_eq!(mime_for("nope"), "application/octet-stream");
    }

    #[test]
    fn cache_control_assets_are_immutable() {
        assert!(cache_control_for("assets/x.js").contains("immutable"));
        assert_eq!(cache_control_for("index.html"), "no-cache");
    }
}
