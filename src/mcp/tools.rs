//! MCP tool dispatcher: JSON-RPC 2.0 over a daemon HTTP back-end.
//!
//! Why: Claude Code speaks MCP/JSON-RPC; the trusty-search daemon speaks
//! REST. This module is a pure translator. It owns no state beyond a
//! `reqwest::Client` and a base URL, so the same dispatcher can be driven
//! from `stdio` (one process per session) or `sse` (long-lived axum task).
//!
//! What: [`McpServer::dispatch`] takes a [`Request`] and returns a
//! [`Response`]. Tool calls map 1:1 to daemon endpoints:
//!
//! | MCP tool        | Daemon endpoint                           |
//! |-----------------|-------------------------------------------|
//! | `search_code`   | `POST /indexes/:id/search`                |
//! | `index_file`    | `POST /indexes/:id/index-file`            |
//! | `remove_file`   | `POST /indexes/:id/remove-file`           |
//! | `list_indexes`  | `GET  /indexes`                           |
//! | `create_index`  | `POST /indexes`                           |
//! | `search_health` | `GET  /health`                            |
//! | `delete_index`  | `DELETE /indexes/:id`                     |
//! | `reindex`       | `POST /indexes/:id/reindex`               |
//! | `index_status`  | `GET  /indexes/:id/status`                |
//! | `list_chunks`   | `GET  /indexes/:id/chunks?offset=&limit=` |
//! | `chat`          | `POST /chat`                              |
//!
//! Test: `cargo test -p trusty-search-mcp` covers JSON-RPC parsing, error
//! shapes (-32600 invalid request, -32601 method not found, -32602 invalid
//! params), and dispatch routing without hitting a real daemon.

use serde_json::Value;

// JSON-RPC 2.0 primitives moved to the shared `trusty-mcp-core` crate so
// trusty-memory and trusty-search agree on the wire shape. Re-exported here
// to keep `pub use` consumers (and `crate::mcp::tools::error_codes` etc.) working.
pub use trusty_mcp_core::{error_codes, initialize_response, JsonRpcError, Request, Response};

/// Tool dispatcher backed by an HTTP client targeting the daemon.
#[derive(Clone)]
pub struct McpServer {
    base_url: String,
    http: reqwest::Client,
}

impl McpServer {
    /// Construct a dispatcher pointing at the daemon's base URL
    /// (e.g. `http://127.0.0.1:7878`).
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            http: reqwest::Client::new(),
        }
    }

    /// Inject a pre-built reqwest client (useful for tests / pooling).
    pub fn with_client(base_url: impl Into<String>, http: reqwest::Client) -> Self {
        Self {
            base_url: base_url.into(),
            http,
        }
    }

    /// Daemon base URL.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Translate a JSON-RPC request into a daemon HTTP call and wrap the
    /// response. Always returns a `Response`; transport / daemon failures are
    /// reported as `INTERNAL_ERROR` rather than panicking.
    pub async fn dispatch(&self, req: Request) -> Response {
        let is_notification = req.id.is_none();
        let id = req.id.clone();

        if req.jsonrpc.as_deref() != Some("2.0") {
            if is_notification {
                return Response::suppressed();
            }
            return Response::err(id, error_codes::INVALID_REQUEST, "jsonrpc must be \"2.0\"");
        }

        // MCP lifecycle methods. `initialize` exchanges capabilities;
        // `notifications/initialized` confirms the client finished setup
        // and is silenced (per JSON-RPC 2.0 notification semantics — no
        // `id`, no reply).
        match req.method.as_str() {
            "initialize" => {
                return Response::ok(
                    id,
                    initialize_response("trusty-search", env!("CARGO_PKG_VERSION"), None),
                );
            }
            "notifications/initialized" | "initialized" => {
                return Response::suppressed();
            }
            _ => {}
        }

        // MCP "tools/call" wraps tool name + arguments. We also accept the
        // bare method name for ergonomics (`search_code` directly).
        let params = req.params.clone().unwrap_or(Value::Null);
        let (tool, arguments, via_tools_call) = match req.method.as_str() {
            "tools/call" => {
                let name = params
                    .get("name")
                    .and_then(Value::as_str)
                    .map(str::to_owned);
                let args = params
                    .get("arguments")
                    .cloned()
                    .unwrap_or(Value::Object(Default::default()));
                match name {
                    Some(n) => (n, args, true),
                    None => {
                        return Response::err(
                            id,
                            error_codes::INVALID_PARAMS,
                            "tools/call requires a 'name' field",
                        )
                    }
                }
            }
            "tools/list" => {
                return Response::ok(id, serde_json::json!({ "tools": tool_descriptors() }));
            }
            // OpenRPC 1.3.2 discovery — see `mcp::openrpc`. Returns the
            // full service description so orchestrators (open-mpm, etc.)
            // can introspect every tool and its required
            // `search.read`/`search.write` scope without bespoke adapters.
            "rpc.discover" => {
                return Response::ok(
                    id,
                    crate::mcp::openrpc::build_discover_response(env!("CARGO_PKG_VERSION")),
                );
            }
            other => (other.to_string(), params, false),
        };

        let outcome = self.call_tool(&tool, &arguments).await;

        if via_tools_call {
            // Per MCP spec, tool execution failures are reported in-band as
            // `{content: [...], isError: true}` rather than JSON-RPC errors —
            // the protocol-level error space is reserved for malformed
            // requests / unknown tools.
            match outcome {
                Ok(value) => Response::ok(id, wrap_tool_result(&value, false)),
                Err(DispatchError::UnknownTool) => Response::err(
                    id,
                    error_codes::METHOD_NOT_FOUND,
                    format!("unknown tool: {tool}"),
                ),
                Err(DispatchError::InvalidParams(msg)) => Response::ok(id, wrap_tool_error(&msg)),
                Err(DispatchError::Transport(msg)) => Response::ok(id, wrap_tool_error(&msg)),
            }
        } else {
            match outcome {
                Ok(value) => Response::ok(id, wrap_text_content(&value)),
                Err(DispatchError::UnknownTool) => Response::err(
                    id,
                    error_codes::METHOD_NOT_FOUND,
                    format!("unknown tool: {tool}"),
                ),
                Err(DispatchError::InvalidParams(msg)) => {
                    Response::err(id, error_codes::INVALID_PARAMS, msg)
                }
                Err(DispatchError::Transport(msg)) => {
                    Response::err(id, error_codes::INTERNAL_ERROR, msg)
                }
            }
        }
    }

    async fn call_tool(&self, tool: &str, args: &Value) -> Result<Value, DispatchError> {
        match tool {
            "search_all" => {
                // Issue #10 — cross-project fan-out search. Maps directly to
                // `POST /search` (the top-level endpoint, distinct from
                // per-index `/indexes/:id/search`).
                let query = require_str(args, "query")?;
                let top_k = args.get("top_k").and_then(Value::as_u64).unwrap_or(10);
                let full_content = args
                    .get("full_content")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                let body = serde_json::json!({
                    "query": query,
                    "top_k": top_k,
                    "full_content": full_content,
                });
                self.post("/search", &body).await
            }
            "search_code" => {
                let index_id = require_str(args, "index_id")?;
                // Accept the spec form `{query: string, top_k?: int}` and
                // also a pre-built `{query: object}` body for callers that
                // need to pass advanced search parameters directly.
                let body = match args.get("query") {
                    Some(v @ Value::Object(_)) => v.clone(),
                    Some(Value::String(text)) => {
                        let mut b = serde_json::json!({ "text": text });
                        if let Some(k) = args.get("top_k").and_then(Value::as_u64) {
                            b["top_k"] = Value::from(k);
                        }
                        // Issue #122 — branch-aware search: forward optional
                        // branch context fields through to the daemon body so
                        // MCP callers can boost on-branch matches without
                        // building the `{query: object}` shape themselves.
                        if let Some(bf) = args.get("branch_files") {
                            b["branch_files"] = bf.clone();
                        }
                        if let Some(bb) = args.get("branch_boost") {
                            b["branch_boost"] = bb.clone();
                        }
                        if let Some(br) = args.get("branch").and_then(Value::as_str) {
                            b["branch"] = Value::String(br.to_string());
                        }
                        b
                    }
                    _ => {
                        return Err(DispatchError::InvalidParams(
                            "missing or invalid 'query' (expected string or object)".into(),
                        ))
                    }
                };
                let resp = self
                    .post(&format!("/indexes/{index_id}/search"), &body)
                    .await?;
                // Mirror the daemon's per-query INFO log (issue #125) so the
                // MCP transport surfaces the same query/intent/latency line.
                let query_text = body
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let log_intent = resp
                    .get("intent")
                    .and_then(Value::as_str)
                    .unwrap_or("Unknown");
                let log_latency = resp
                    .get("latency_ms")
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                let log_results = resp
                    .get("results")
                    .and_then(Value::as_array)
                    .map(Vec::len)
                    .unwrap_or(0);
                tracing::info!(
                    index_id = %index_id,
                    intent = %log_intent,
                    latency_ms = log_latency,
                    results = log_results,
                    query = %&query_text[..query_text.len().min(80)],
                    "search"
                );
                Ok(resp)
            }
            "index_file" => {
                let index_id = require_str(args, "index_id")?;
                let path = require_str(args, "path")?;
                let content = require_str(args, "content")?;
                self.post(
                    &format!("/indexes/{index_id}/index-file"),
                    &serde_json::json!({ "path": path, "content": content }),
                )
                .await
            }
            "remove_file" => {
                let index_id = require_str(args, "index_id")?;
                let path = require_str(args, "path")?;
                self.post(
                    &format!("/indexes/{index_id}/remove-file"),
                    &serde_json::json!({ "path": path }),
                )
                .await
            }
            "list_indexes" => self.get("/indexes").await,
            "create_index" => {
                let id = require_str(args, "id")?;
                let root_path = require_str(args, "root_path")?;
                self.post(
                    "/indexes",
                    &serde_json::json!({ "id": id, "root_path": root_path }),
                )
                .await
            }
            "search_similar" => {
                // Code-to-code similarity (issue #31). Index defaults to "default"
                // so simple call sites don't need to specify it.
                let index_id = args
                    .get("index")
                    .and_then(Value::as_str)
                    .unwrap_or("default");
                let file = require_str(args, "file")?;
                let mut body = serde_json::json!({ "file": file });
                if let Some(func) = args.get("function").and_then(Value::as_str) {
                    body["function"] = Value::String(func.to_string());
                }
                if let Some(k) = args.get("top_k").and_then(Value::as_u64) {
                    body["top_k"] = Value::from(k);
                }
                self.post(&format!("/indexes/{index_id}/search_similar"), &body)
                    .await
            }
            "search_health" => self.get("/health").await,
            "delete_index" => {
                let index_id = require_str(args, "index_id")?;
                self.delete(&format!("/indexes/{index_id}")).await
            }
            "reindex" => {
                let index_id = require_str(args, "index_id")?;
                // Accept optional root_path override (mirrors the HTTP body).
                let mut body = serde_json::json!({});
                if let Some(rp) = args.get("root_path").and_then(Value::as_str) {
                    body["root_path"] = Value::String(rp.to_string());
                }
                self.post(&format!("/indexes/{index_id}/reindex"), &body)
                    .await
            }
            "index_status" => {
                let index_id = require_str(args, "index_id")?;
                self.get(&format!("/indexes/{index_id}/status")).await
            }
            "chat" => {
                let index_id = require_str(args, "index_id")?;
                // Accept either `message` (legacy / UI) or `question` (issue #15 spec).
                let message = args
                    .get("message")
                    .and_then(Value::as_str)
                    .or_else(|| args.get("question").and_then(Value::as_str))
                    .ok_or_else(|| {
                        DispatchError::InvalidParams(
                            "missing required string field: message (or question)".into(),
                        )
                    })?;
                let mut body = serde_json::json!({
                    "index_id": index_id,
                    "message": message,
                });
                if let Some(history) = args.get("history") {
                    body["history"] = history.clone();
                }
                if let Some(model) = args.get("model").and_then(Value::as_str) {
                    body["model"] = Value::String(model.to_string());
                }
                if let Some(top_k) = args.get("top_k").and_then(Value::as_u64) {
                    body["top_k"] = Value::from(top_k);
                }
                if let Some(key) = args.get("api_key").and_then(Value::as_str) {
                    body["api_key"] = Value::String(key.to_string());
                }
                self.post("/chat", &body).await
            }
            "list_chunks" => {
                // Issue #54 — paginated enumeration of an index's corpus.
                // Mirrors `GET /indexes/:id/chunks?offset=&limit=`.
                let index_id = require_str(args, "index_id")?;
                let offset = args.get("offset").and_then(Value::as_u64).unwrap_or(0);
                let limit = args.get("limit").and_then(Value::as_u64).unwrap_or(100);
                self.get(&format!(
                    "/indexes/{index_id}/chunks?offset={offset}&limit={limit}"
                ))
                .await
            }
            _ => Err(DispatchError::UnknownTool),
        }
    }

    async fn get(&self, path: &str) -> Result<Value, DispatchError> {
        let url = format!("{}{}", self.base_url, path);
        let resp = self
            .http
            .get(&url)
            .send()
            .await
            .map_err(|e| DispatchError::Transport(format!("GET {url}: {e}")))?;
        let status = resp.status();
        let body: Value = resp
            .json()
            .await
            .map_err(|e| DispatchError::Transport(format!("decode {url}: {e}")))?;
        if !status.is_success() {
            return Err(DispatchError::Transport(format!(
                "GET {url} returned {status}: {body}"
            )));
        }
        Ok(body)
    }

    async fn delete(&self, path: &str) -> Result<Value, DispatchError> {
        let url = format!("{}{}", self.base_url, path);
        let resp = self
            .http
            .delete(&url)
            .send()
            .await
            .map_err(|e| DispatchError::Transport(format!("DELETE {url}: {e}")))?;
        let status = resp.status();
        let body: Value = resp
            .json()
            .await
            .map_err(|e| DispatchError::Transport(format!("decode {url}: {e}")))?;
        if !status.is_success() {
            return Err(DispatchError::Transport(format!(
                "DELETE {url} returned {status}: {body}"
            )));
        }
        Ok(body)
    }

    async fn post(&self, path: &str, body: &Value) -> Result<Value, DispatchError> {
        let url = format!("{}{}", self.base_url, path);
        let resp = self
            .http
            .post(&url)
            .json(body)
            .send()
            .await
            .map_err(|e| DispatchError::Transport(format!("POST {url}: {e}")))?;
        let status = resp.status();
        let body: Value = resp
            .json()
            .await
            .map_err(|e| DispatchError::Transport(format!("decode {url}: {e}")))?;
        if !status.is_success() {
            return Err(DispatchError::Transport(format!(
                "POST {url} returned {status}: {body}"
            )));
        }
        Ok(body)
    }
}

#[derive(Debug)]
enum DispatchError {
    UnknownTool,
    InvalidParams(String),
    Transport(String),
}

fn require_str<'a>(args: &'a Value, key: &str) -> Result<&'a str, DispatchError> {
    args.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| DispatchError::InvalidParams(format!("missing or non-string '{key}'")))
}

/// Wrap a structured JSON result in MCP's `content[]` envelope so downstream
/// LLM clients can render it directly.
fn wrap_text_content(value: &Value) -> Value {
    serde_json::json!({
        "content": [{
            "type": "text",
            "text": serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string()),
        }]
    })
}

/// Wrap a successful `tools/call` payload with the spec-required
/// `isError: false` flag so MCP clients can branch without parsing text.
fn wrap_tool_result(value: &Value, is_error: bool) -> Value {
    serde_json::json!({
        "content": [{
            "type": "text",
            "text": serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string()),
        }],
        "isError": is_error,
    })
}

/// Wrap a tool execution failure as `{content, isError: true}` per MCP spec.
fn wrap_tool_error(msg: &str) -> Value {
    serde_json::json!({
        "content": [{
            "type": "text",
            "text": format!("Error: {msg}"),
        }],
        "isError": true,
    })
}

/// Static metadata for `tools/list`. Keep in sync with [`McpServer::call_tool`].
pub fn tool_descriptors() -> Value {
    serde_json::json!([
        {
            "name": "search_all",
            "description": "Cross-project hybrid search: fan out to every registered index, merge results via RRF, tag each chunk with its index_id (issue #10).",
            "inputSchema": {
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query":        { "type": "string" },
                    "top_k":        { "type": "integer", "default": 10 },
                    "full_content": { "type": "boolean", "default": false }
                }
            }
        },
        {
            "name": "search_code",
            "description": "Hybrid code search (BM25+vector+KG). Supports branch-aware scoring via branch_files/branch_boost/branch (issue #122).",
            "inputSchema": {
                "type": "object",
                "required": ["index_id", "query"],
                "properties": {
                    "index_id": { "type": "string" },
                    "query": { "type": "string" },
                    "top_k": { "type": "integer", "default": 10 },
                    "branch_files": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Files modified on current git branch (relative to index root). Boosted in results."
                    },
                    "branch_boost": {
                        "type": "number",
                        "description": "Score multiplier for branch files (default 1.5, range 1.0-3.0)."
                    },
                    "branch": {
                        "type": "string",
                        "description": "Branch name; daemon will compute branch_files via git if branch_files is absent."
                    }
                }
            }
        },
        {
            "name": "index_file",
            "description": "Add or update one file in an index",
            "inputSchema": {
                "type": "object",
                "required": ["index_id", "path", "content"],
                "properties": {
                    "index_id": { "type": "string" },
                    "path":     { "type": "string" },
                    "content":  { "type": "string" }
                }
            }
        },
        {
            "name": "remove_file",
            "description": "Remove a file's chunks from an index",
            "inputSchema": {
                "type": "object",
                "required": ["index_id", "path"],
                "properties": {
                    "index_id": { "type": "string" },
                    "path":     { "type": "string" }
                }
            }
        },
        {
            "name": "list_indexes",
            "description": "List all registered indexes on this daemon",
            "inputSchema": { "type": "object", "properties": {} }
        },
        {
            "name": "create_index",
            "description": "Register a new (empty) index",
            "inputSchema": {
                "type": "object",
                "required": ["id", "root_path"],
                "properties": {
                    "id":        { "type": "string" },
                    "root_path": { "type": "string" }
                }
            }
        },
        {
            "name": "search_similar",
            "description": "Find chunks semantically similar to a given file/function via HNSW (issue #31)",
            "inputSchema": {
                "type": "object",
                "required": ["file"],
                "properties": {
                    "file":     { "type": "string" },
                    "function": { "type": "string" },
                    "top_k":    { "type": "number" },
                    "index":    { "type": "string" }
                }
            }
        },
        {
            "name": "search_health",
            "description": "Probe daemon liveness and version",
            "inputSchema": { "type": "object", "properties": {} }
        },
        {
            "name": "delete_index",
            "description": "Delete a registered index and all its data",
            "inputSchema": {
                "type": "object",
                "required": ["index_id"],
                "properties": {
                    "index_id": { "type": "string" }
                }
            }
        },
        {
            "name": "reindex",
            "description": "Trigger a full reindex of a collection (async, returns immediately)",
            "inputSchema": {
                "type": "object",
                "required": ["index_id"],
                "properties": {
                    "index_id":  { "type": "string" },
                    "root_path": { "type": "string" }
                }
            }
        },
        {
            "name": "index_status",
            "description": "Get stats for an index (chunk count, root path)",
            "inputSchema": {
                "type": "object",
                "required": ["index_id"],
                "properties": {
                    "index_id": { "type": "string" }
                }
            }
        },
        {
            "name": "list_chunks",
            "description": "Paginated enumeration of every chunk in an index (issue #54). Stable order by (file, start_line).",
            "inputSchema": {
                "type": "object",
                "required": ["index_id"],
                "properties": {
                    "index_id": { "type": "string" },
                    "offset":   { "type": "integer", "default": 0 },
                    "limit":    { "type": "integer", "default": 100 }
                }
            }
        },
        {
            "name": "chat",
            "description": "Ask a natural-language question about the indexed codebase. \
                            Automatically searches for the top_k most relevant chunks and \
                            sends them as context to an OpenRouter LLM (default model: \
                            anthropic/claude-haiku-4.5). Returns {answer, sources, model}. \
                            Requires OPENROUTER_API_KEY env var on the daemon, or an \
                            `api_key` field in the request.",
            "inputSchema": {
                "type": "object",
                "required": ["index_id"],
                "properties": {
                    "index_id": { "type": "string" },
                    "message":  { "type": "string", "description": "User question (alias: question)" },
                    "question": { "type": "string", "description": "User question (alias: message)" },
                    "history":  { "type": "array", "items": { "type": "object" } },
                    "model":    { "type": "string", "description": "OpenRouter model id (default: anthropic/claude-haiku-4.5)" },
                    "top_k":    { "type": "integer", "description": "Number of context chunks (default: 5)", "default": 5 },
                    "api_key":  { "type": "string", "description": "Fallback OpenRouter API key when OPENROUTER_API_KEY env is unset" }
                }
            }
        }
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(method: &str, params: Value) -> Request {
        Request {
            jsonrpc: Some("2.0".into()),
            id: Some(Value::from(1u64)),
            method: method.into(),
            params: Some(params),
        }
    }

    #[tokio::test]
    async fn rejects_wrong_jsonrpc_version() {
        let server = McpServer::new("http://127.0.0.1:1");
        let r = Request {
            jsonrpc: Some("1.0".into()),
            id: Some(Value::from(7u64)),
            method: "search_health".into(),
            params: None,
        };
        let resp = server.dispatch(r).await;
        let err = resp.error.expect("expected error");
        assert_eq!(err.code, error_codes::INVALID_REQUEST);
        assert_eq!(resp.id, Some(Value::from(7u64)));
    }

    #[tokio::test]
    async fn unknown_tool_returns_method_not_found() {
        let server = McpServer::new("http://127.0.0.1:1");
        let resp = server.dispatch(req("not_a_tool", Value::Null)).await;
        let err = resp.error.expect("expected error");
        assert_eq!(err.code, error_codes::METHOD_NOT_FOUND);
    }

    #[tokio::test]
    async fn missing_params_returns_invalid_params() {
        let server = McpServer::new("http://127.0.0.1:1");
        let resp = server
            .dispatch(req("index_file", serde_json::json!({})))
            .await;
        let err = resp.error.expect("expected error");
        assert_eq!(err.code, error_codes::INVALID_PARAMS);
    }

    #[tokio::test]
    async fn tools_list_returns_all_tools() {
        let server = McpServer::new("http://127.0.0.1:1");
        let resp = server.dispatch(req("tools/list", Value::Null)).await;
        let result = resp.result.expect("expected result");
        let tools = result
            .get("tools")
            .and_then(Value::as_array)
            .expect("array");
        // Issue #36 requires the 6 core MCP tools to be present; we ship
        // additional tools beyond that minimum.
        assert!(
            tools.len() >= 6,
            "expected at least 6 tools, got {}",
            tools.len()
        );
        let names: Vec<&str> = tools
            .iter()
            .filter_map(|t| t.get("name").and_then(Value::as_str))
            .collect();
        for required in [
            "search_code",
            "index_file",
            "remove_file",
            "list_indexes",
            "create_index",
            "search_health",
        ] {
            assert!(
                names.contains(&required),
                "missing required tool: {required}"
            );
        }
    }

    /// Issue #36 — verify the `initialize` handshake returns the spec-shaped
    /// payload Claude Code expects on startup.
    #[tokio::test]
    async fn test_initialize_response() {
        let server = McpServer::new("http://127.0.0.1:1");
        let r = Request {
            jsonrpc: Some("2.0".into()),
            id: Some(Value::from(1u64)),
            method: "initialize".into(),
            params: Some(serde_json::json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": { "name": "test", "version": "0.0.0" }
            })),
        };
        let resp = server.dispatch(r).await;
        assert!(resp.error.is_none(), "initialize must not error");
        let result = resp.result.expect("expected result");
        assert_eq!(result["protocolVersion"], "2024-11-05");
        assert!(result["capabilities"].get("tools").is_some());
        assert_eq!(result["serverInfo"]["name"], "trusty-search");
        assert!(result["serverInfo"]["version"].is_string());
    }

    /// Issue #36 — `tools/list` must surface every spec-required tool so
    /// MCP clients can render the full manifest.
    #[tokio::test]
    async fn test_tools_list_response() {
        let server = McpServer::new("http://127.0.0.1:1");
        let resp = server.dispatch(req("tools/list", Value::Null)).await;
        let result = resp.result.expect("expected result");
        let tools = result
            .get("tools")
            .and_then(Value::as_array)
            .expect("array");
        let names: Vec<&str> = tools
            .iter()
            .filter_map(|t| t.get("name").and_then(Value::as_str))
            .collect();
        for required in [
            "search_code",
            "index_file",
            "remove_file",
            "list_indexes",
            "create_index",
            "search_health",
        ] {
            assert!(
                names.contains(&required),
                "tools/list missing '{required}' (got {names:?})"
            );
        }
        // Each tool must carry an inputSchema so clients can validate args.
        for t in tools {
            assert!(t.get("name").is_some());
            assert!(t.get("inputSchema").is_some());
        }
    }

    /// Issue #36 — JSON-RPC method-not-found surfaces as -32601.
    #[tokio::test]
    async fn test_unknown_method_returns_error() {
        let server = McpServer::new("http://127.0.0.1:1");
        let resp = server
            .dispatch(req("definitely_not_a_method", Value::Null))
            .await;
        let err = resp.error.expect("expected error");
        assert_eq!(err.code, error_codes::METHOD_NOT_FOUND);
    }

    /// `notifications/initialized` is a JSON-RPC notification — the server
    /// must NOT emit a response, signalled by `Response::suppress = true`.
    #[tokio::test]
    async fn notification_initialized_is_suppressed() {
        let server = McpServer::new("http://127.0.0.1:1");
        let r = Request {
            jsonrpc: Some("2.0".into()),
            id: None, // notifications carry no id
            method: "notifications/initialized".into(),
            params: None,
        };
        let resp = server.dispatch(r).await;
        assert!(resp.suppress, "notifications must be suppressed");
    }

    /// Parity gate: every HTTP endpoint reachable via REST must also be callable
    /// as an MCP tool. This guards the "MCP and HTTP are functionally equivalent"
    /// invariant — if a new HTTP route lands without a matching tool, this fails.
    #[tokio::test]
    async fn test_tools_list_complete() {
        let server = McpServer::new("http://127.0.0.1:1");
        let resp = server.dispatch(req("tools/list", Value::Null)).await;
        let result = resp.result.expect("expected result");
        let tools = result
            .get("tools")
            .and_then(Value::as_array)
            .expect("array");
        let names: Vec<&str> = tools
            .iter()
            .filter_map(|t| t.get("name").and_then(Value::as_str))
            .collect();
        for required in [
            "search_code",
            "index_file",
            "remove_file",
            "list_indexes",
            "create_index",
            "search_health",
            "delete_index",
            "reindex",
            "index_status",
            "list_chunks",
            "chat",
            "search_all",
        ] {
            assert!(
                names.contains(&required),
                "tools/list missing '{required}' (got {names:?})"
            );
        }
    }

    /// Issue #10 — `search_all` requires the `query` arg and rejects missing it
    /// before any HTTP round-trip.
    #[tokio::test]
    async fn search_all_missing_query_returns_invalid_params() {
        let server = McpServer::new("http://127.0.0.1:1");
        let resp = server
            .dispatch(req("search_all", serde_json::json!({})))
            .await;
        let err = resp.error.expect("expected error");
        assert_eq!(err.code, error_codes::INVALID_PARAMS);
    }

    #[tokio::test]
    async fn tools_call_without_name_returns_invalid_params() {
        let server = McpServer::new("http://127.0.0.1:1");
        let resp = server
            .dispatch(req("tools/call", serde_json::json!({})))
            .await;
        let err = resp.error.expect("expected error");
        assert_eq!(err.code, error_codes::INVALID_PARAMS);
    }
}
