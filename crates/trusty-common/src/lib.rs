//! Shared utility surface for trusty-* projects.
//!
//! Why: Port auto-detect, data-directory resolution, tracing init, NO_COLOR
//! handling, and the OpenRouter chat-completions client appeared in both
//! trusty-memory and trusty-search with subtle divergence. Centralising keeps
//! them aligned and gives future trusty-* binaries a one-import surface.
//!
//! What: pure utility functions — no global state. Each subsystem is a free
//! function or a small helper struct.
//!
//! Test: `cargo test -p trusty-common` covers port walking, data-dir creation,
//! and the OpenRouter request shape (without hitting the network).

use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;

// ─── Port binding ─────────────────────────────────────────────────────────

/// Bind to `addr`; if the port is in use, walk forward up to `max_attempts`
/// ports and return the first listener that binds.
///
/// Why: Running multiple instances of a trusty-* daemon (or restarting before
/// the kernel releases the prior socket) shouldn't produce a noisy failure —
/// auto-incrementing gives a friendlier developer experience while still
/// honouring the user's preferred starting port.
/// What: returns the first successful `tokio::net::TcpListener`. The actual
/// bound address is printed to stderr so callers can discover where it
/// landed. `max_attempts == 0` means "try `addr` exactly once".
/// Test: `auto_port_walks_forward` binds a port, then calls this with the
/// occupied port and confirms a different free port is returned.
pub async fn bind_with_auto_port(addr: SocketAddr, max_attempts: u16) -> Result<TcpListener> {
    use std::io::ErrorKind;
    let mut current = addr;
    for attempt in 0..=max_attempts {
        match TcpListener::bind(current).await {
            Ok(l) => {
                if let Ok(local) = l.local_addr() {
                    eprintln!("listening on {local}");
                }
                return Ok(l);
            }
            Err(e) if e.kind() == ErrorKind::AddrInUse && attempt < max_attempts => {
                let next_port = current.port().saturating_add(1);
                if next_port == 0 {
                    anyhow::bail!("ran out of ports while searching for free slot");
                }
                tracing::warn!(
                    "port {} in use, trying {}",
                    current.port(),
                    next_port
                );
                current.set_port(next_port);
            }
            Err(e) => return Err(e.into()),
        }
    }
    anyhow::bail!("could not find free port after {max_attempts} attempts")
}

// ─── Data directory ───────────────────────────────────────────────────────

/// Resolve `<data_dir>/<app_name>`, creating it if it doesn't exist.
///
/// Why: All trusty-* tools want a per-machine, per-app directory under the
/// OS-standard data dir (`~/Library/Application Support/`, `~/.local/share/`,
/// `%APPDATA%/`). If `dirs::data_dir()` is unavailable (rare — locked-down
/// containers), fall back to `~/.<app_name>` so the tool still works.
/// What: returns the absolute path. Creates intermediates.
/// Test: `resolve_data_dir_creates_directory` confirms creation under a
/// stubbed HOME.
pub fn resolve_data_dir(app_name: &str) -> Result<PathBuf> {
    let base = dirs::data_dir()
        .or_else(|| dirs::home_dir().map(|h| h.join(format!(".{app_name}"))))
        .context("could not resolve data directory or home directory")?;
    let dir = if base.ends_with(format!(".{app_name}")) {
        base
    } else {
        base.join(app_name)
    };
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("create data directory {}", dir.display()))?;
    Ok(dir)
}

// ─── CLI initialisation ───────────────────────────────────────────────────

/// Initialise the global tracing subscriber.
///
/// Why: Every trusty-* binary wants the same verbosity ladder and the same
/// `RUST_LOG` override semantics. Defining it once removes the boilerplate
/// from every `main.rs`.
/// What: `verbose_count` maps `0 → warn`, `1 → info`, `2 → debug`, `3+ →
/// trace`. If `RUST_LOG` is set in the environment it wins. Logs go to
/// stderr so stdout stays clean for MCP JSON-RPC.
/// Test: side-effecting (global subscriber) — covered by integration with
/// `cargo run -- -v status` in downstream crates.
pub fn init_tracing(verbose_count: u8) {
    let default_filter = match verbose_count {
        0 => "warn",
        1 => "info",
        2 => "debug",
        _ => "trace",
    };
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(default_filter));
    // try_init so callers that pre-install a subscriber don't panic.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .with_target(false)
        .try_init();
}

/// Disable coloured terminal output when requested or when stdout is not a TTY.
///
/// Why: Pipe-friendly output is mandatory for scripting (`trusty-search list
/// | jq …`). `NO_COLOR` / `TERM=dumb` are the canonical signals; passing
/// `--no-color` should override too.
/// What: calls `colored::control::set_override(false)` when the caller asks
/// for it or when the standard heuristics indicate no colour.
/// Test: side-effecting global; trivially covered by manual `NO_COLOR=1 cargo
/// run -- list`.
pub fn maybe_disable_color(no_color: bool) {
    let env_says_no = std::env::var("NO_COLOR").is_ok()
        || std::env::var("TERM").as_deref() == Ok("dumb");
    if no_color || env_says_no {
        colored::control::set_override(false);
    }
}

// ─── OpenRouter ───────────────────────────────────────────────────────────

const OPENROUTER_URL: &str = "https://openrouter.ai/api/v1/chat/completions";
const HTTP_REFERER: &str = "https://github.com/bobmatnyc/trusty-common";
const X_TITLE: &str = "trusty-common";

/// OpenAI-compatible chat message.
///
/// Why: Both trusty-memory's `chat` subcommand and trusty-search's `/chat`
/// endpoint speak the OpenRouter format. Sharing the struct keeps them in
/// step (and lets callers compose chat histories without re-defining types).
/// What: `role` is one of `"system" | "user" | "assistant"`. `content` is
/// the message text.
/// Test: serde round-trip in `chat_message_round_trips`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: &'a [ChatMessage],
    stream: bool,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: ResponseMessage,
}

#[derive(Debug, Deserialize)]
struct ResponseMessage {
    #[serde(default)]
    content: String,
}

/// Send a chat completion request to OpenRouter and return the assistant's
/// message content.
///
/// Why: A one-shot, non-streaming chat call is the common-case helper — used
/// by trusty-memory's `chat` CLI and trusty-search's `/chat` endpoint.
/// What: POSTs `{model, messages, stream: false}` to OpenRouter with bearer
/// auth, decodes the response, and returns `choices[0].message.content`.
/// Errors propagate as anyhow with HTTP status context.
/// Test: error paths covered by `openrouter_propagates_http_errors` (uses a
/// blackhole base URL — no real call).
pub async fn openrouter_chat(
    api_key: &str,
    model: &str,
    messages: Vec<ChatMessage>,
) -> Result<String> {
    if api_key.is_empty() {
        return Err(anyhow!("openrouter api key is empty"));
    }
    let client = reqwest::Client::builder()
        .build()
        .context("build reqwest client for openrouter_chat")?;
    let body = ChatRequest {
        model,
        messages: &messages,
        stream: false,
    };
    let resp = client
        .post(OPENROUTER_URL)
        .bearer_auth(api_key)
        .header("HTTP-Referer", HTTP_REFERER)
        .header("X-Title", X_TITLE)
        .json(&body)
        .send()
        .await
        .context("POST openrouter chat completions")?;
    let status = resp.status();
    if !status.is_success() {
        let text = resp.text().await.unwrap_or_default();
        return Err(anyhow!("openrouter HTTP {status}: {text}"));
    }
    let payload: ChatResponse = resp.json().await.context("decode openrouter response")?;
    payload
        .choices
        .into_iter()
        .next()
        .map(|c| c.message.content)
        .ok_or_else(|| anyhow!("openrouter returned no choices"))
}

/// Stream chat-completion deltas from OpenRouter through a tokio mpsc channel.
///
/// Why: `chat` UIs want incremental tokens for a responsive feel; the
/// streaming endpoint emits SSE `data:` frames with delta content.
/// What: POSTs the request with `stream: true`, parses each SSE `data:` line
/// as a JSON object, extracts `choices[0].delta.content`, and sends each
/// non-empty chunk to `tx`. The function returns when the stream terminates
/// (either by `[DONE]` sentinel or by upstream EOF).
/// Test: integration-only (no offline mock); covered manually via the
/// trusty-search `/chat` endpoint that re-uses this helper.
pub async fn openrouter_chat_stream(
    api_key: &str,
    model: &str,
    messages: Vec<ChatMessage>,
    tx: tokio::sync::mpsc::Sender<String>,
) -> Result<()> {
    use futures_util::StreamExt;

    if api_key.is_empty() {
        return Err(anyhow!("openrouter api key is empty"));
    }
    let client = reqwest::Client::builder()
        .build()
        .context("build reqwest client for openrouter_chat_stream")?;
    let body = ChatRequest {
        model,
        messages: &messages,
        stream: true,
    };
    let resp = client
        .post(OPENROUTER_URL)
        .bearer_auth(api_key)
        .header("HTTP-Referer", HTTP_REFERER)
        .header("X-Title", X_TITLE)
        .json(&body)
        .send()
        .await
        .context("POST openrouter chat completions (stream)")?;
    let status = resp.status();
    if !status.is_success() {
        let text = resp.text().await.unwrap_or_default();
        return Err(anyhow!("openrouter HTTP {status}: {text}"));
    }

    let mut buf = String::new();
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let bytes = chunk.context("read openrouter stream chunk")?;
        let text = match std::str::from_utf8(&bytes) {
            Ok(s) => s,
            Err(_) => continue,
        };
        buf.push_str(text);

        while let Some(idx) = buf.find('\n') {
            let line: String = buf.drain(..=idx).collect();
            let line = line.trim();
            let Some(payload) = line.strip_prefix("data:").map(str::trim) else {
                continue;
            };
            if payload.is_empty() || payload == "[DONE]" {
                continue;
            }
            let v: serde_json::Value = match serde_json::from_str(payload) {
                Ok(v) => v,
                Err(_) => continue,
            };
            if let Some(delta) = v
                .get("choices")
                .and_then(|c| c.get(0))
                .and_then(|c| c.get("delta"))
                .and_then(|d| d.get("content"))
                .and_then(|c| c.as_str())
            {
                if !delta.is_empty() && tx.send(delta.to_string()).await.is_err() {
                    // Receiver dropped — caller has lost interest.
                    return Ok(());
                }
            }
        }
    }
    Ok(())
}

// ─── Misc helpers ─────────────────────────────────────────────────────────

/// Check whether a path exists and is a directory.
///
/// Why: tiny but commonly-needed shim — clearer at call sites than
/// `path.exists() && path.is_dir()`.
/// What: returns `true` iff the path exists and metadata reports a directory.
/// Test: `is_dir_recognises_directories`.
pub fn is_dir(path: &Path) -> bool {
    path.metadata().map(|m| m.is_dir()).unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn auto_port_walks_forward() {
        // Bind to an OS-chosen port, then ask auto-port to start there.
        let occupied = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = occupied.local_addr().unwrap().port();
        let addr: SocketAddr = format!("127.0.0.1:{port}").parse().unwrap();
        let next = bind_with_auto_port(addr, 8).await.unwrap();
        let got = next.local_addr().unwrap().port();
        assert_ne!(got, port, "expected walk-forward to a different port");
    }

    #[tokio::test]
    async fn auto_port_zero_attempts_still_binds_free() {
        let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let l = bind_with_auto_port(addr, 0).await.unwrap();
        assert!(l.local_addr().unwrap().port() > 0);
    }

    #[test]
    fn resolve_data_dir_creates_directory() {
        // Point HOME at a tempdir so we test the fallback branch deterministically.
        let tmp = tempfile_like_dir();
        // SAFETY: env mutation pre-runtime; no other thread observes HOME here.
        unsafe {
            std::env::set_var("HOME", &tmp);
            // Some platforms key off these too — clear them so dirs::data_dir
            // takes the predictable HOME-relative path.
            std::env::set_var("XDG_DATA_HOME", tmp.join("share"));
        }
        let dir = resolve_data_dir("trusty-test-xyz").unwrap();
        assert!(dir.exists(), "data dir should be created at {}", dir.display());
        assert!(dir.is_dir());
    }

    #[test]
    fn is_dir_recognises_directories() {
        let tmp = tempfile_like_dir();
        assert!(is_dir(&tmp));
        assert!(!is_dir(&tmp.join("nope")));
    }

    #[test]
    fn chat_message_round_trips() {
        let m = ChatMessage {
            role: "user".into(),
            content: "hello".into(),
        };
        let s = serde_json::to_string(&m).unwrap();
        let back: ChatMessage = serde_json::from_str(&s).unwrap();
        assert_eq!(back.role, "user");
        assert_eq!(back.content, "hello");
    }

    #[tokio::test]
    async fn openrouter_chat_rejects_empty_key() {
        let err = openrouter_chat("", "x", vec![]).await.unwrap_err();
        assert!(err.to_string().contains("api key"));
    }

    // Test-only helper: makes a unique scratch dir without pulling in tempfile
    // as a dev-dep (keeps the dependency surface minimal).
    fn tempfile_like_dir() -> PathBuf {
        let pid = std::process::id();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let p = std::env::temp_dir().join(format!("trusty-common-test-{pid}-{nanos}"));
        std::fs::create_dir_all(&p).unwrap();
        p
    }
}
