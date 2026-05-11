# trusty-search

Machine-wide, blazingly fast hybrid code search service. Single install per machine,
serves multiple named indexes (one per project) via HTTP daemon and MCP server.

> **Coordination:** Shared library patterns, consistent conventions, and CI/CD configuration for this project are managed by [trusty-common](../trusty-common). See that repo's CLAUDE.md for cross-project guidelines.

## Project Goals

- **Machine-wide service**: one install (`cargo install trusty-search`), one daemon
  per machine, serves all projects on the box
- **Multiple named indexes**: each project registers an `IndexId`; one daemon manages
  all of them concurrently
- **Hybrid search**: BM25 (lexical) + HNSW vector (semantic) + Knowledge Graph
  expansion, fused via Reciprocal Rank Fusion (RRF, k=60, parameter-free)
- **Query-type routing**: classify intent (Definition / Usage / Conceptual / BugDebt /
  Unknown) and route to optimal weighting before searching
- **MCP server**: stdio + HTTP/SSE for Claude Code integration
- **Zero cold-start**: HNSW stays hot (Duration::MAX cool-after), LRU embedding cache
  (256 entries) skips re-embedding on repeated queries
- **Native multi-request**: `Arc<SearchAppState>`, concurrent reads via `RwLock`,
  axum HTTP/2 — many concurrent readers never block each other
- **Zero dependency on trusty-memory**: bundles its own storage layer
  (redb + usearch + fastembed) — `cargo install trusty-search` works standalone

## Architecture

```
Machine-wide service (single install, one daemon per machine)
  └── IndexRegistry: DashMap<IndexId, Arc<IndexHandle>>
        └── IndexHandle
              ├── CodeIndexer: Arc<RwLock<HnswIndex>> (usearch) — concurrent reads
              │     ├── parse_and_embed_files()  — runs outside write lock (parse + embed)
              │     └── commit_parsed_batch()    — holds write lock only for redb+HNSW commit
              ├── BM25Builder: per-query, built from chunk corpus
              ├── KnowledgeGraph: Arc<SymbolGraph> (petgraph, tree-sitter derived)
              ├── FileWatcher: notify-debouncer-mini, 500ms debounce
              └── QueryCache: Arc<Mutex<LruCache<QueryHash, Vec<f32>>>> — skip embedding on repeat
```

### Query Pipeline

1. **Classify intent**: `QueryClassifier` (sub-ms regex) →
   `Definition / Usage / Conceptual / BugDebt / Unknown`
2. **Route weights**: `alpha` (vector), `beta` (BM25), `use_kg_first`
3. **Search**: 4×top_k HNSW candidates + per-query BM25 index over chunk corpus
4. **Fuse**: Reciprocal Rank Fusion (k=60, parameter-free)
5. **KG expand**: 1–2 hop `callers_of` / `callees_of` via `SymbolGraph`,
   scored at 70% of trigger chunk's RRF score
6. **Return**: compact (7-line snippet) or full chunk

### Query Intent → Routing Weights

| Intent      | alpha (vector) | beta (BM25) | use_kg_first |
|-------------|----------------|-------------|--------------|
| Definition  | 0.3            | 0.7         | false        |
| Usage       | 0.5            | 0.5         | true         |
| Conceptual  | 0.8            | 0.2         | false        |
| BugDebt     | 0.1            | 0.9         | false        |
| Unknown     | 0.6            | 0.4         | false        |

### CodeChunk

```rust
pub struct CodeChunk {
    pub id: String,                       // "{path}:{start}:{end}" — collision-safe
    pub file: String,
    pub start_line: usize,
    pub end_line: usize,
    pub content: String,
    pub function_name: Option<String>,
    pub score: f32,
    pub compact_snippet: Option<String>,  // 7-line snippet for token-efficient output
    pub match_reason: String,             // "hybrid", "hybrid+kg", "bm25", "vector", "fallback:ripgrep"
}
```

### HTTP API (axum, single daemon, multi-index)

```
GET    /health                       liveness probe
GET    /indexes                      list all registered indexes
POST   /indexes/:id/search           hybrid search query
POST   /indexes/:id/index-file       add/update a file
POST   /indexes/:id/remove-file      remove from index
POST   /indexes/:id/reindex          full reindex (fire-and-forget)
GET    /indexes/:id/reindex/stream   SSE progress stream (start/progress/complete/error)
GET    /indexes/:id/status           index stats (chunks, last-updated)
GET    /indexes/:id/chunks           paginated chunk enumeration (?offset, ?limit)
DELETE /indexes/:id                  delete an index
GET    /ui                           web management UI (Svelte, embedded)
POST   /chat                         OpenRouter proxy with search context injection
```

### MCP Tools

- `search_code` — hybrid search query
- `index_file` — add/update one file
- `remove_file` — remove one file
- `list_indexes` — enumerate registered indexes
- `create_index` — register a new index
- `search_health` — daemon liveness
- `delete_index` — delete an index
- `reindex` — trigger full reindex
- `index_status` — per-index stats
- `list_chunks` — paginated enumeration of an index's chunks
- `chat` — OpenRouter conversational Q&A

## Stack

- **Language**: Rust 2021
- **Async runtime**: tokio (full features)
- **HTTP**: axum 0.7 + tower-http (CORS, trace, gzip), HTTP/2
- **Vector store**: usearch 2.25 (HNSW), wrapped in `Arc<RwLock<>>` for concurrent reads
- **Embeddings**: fastembed 5.x (ONNX, all-MiniLM-L6-v2, 384-dim, SIMD/AVX2/NEON)
- **Lexical**: BM25 (zero-dep port from open-mpm `src/context/bm25.rs`)
- **KV store**: redb 2.6 (chunk metadata, file→chunks mapping)
- **File watching**: notify 6 + notify-debouncer-mini 0.4 (500ms debounce, fsevent)
- **Code parsing**: tree-sitter 0.24 (rust, python, js, ts, go, java, c, cpp)
- **Graph**: petgraph 0.6 (`SymbolGraph` for callers_of / callees_of)
- **Concurrency**: dashmap 5 (`IndexRegistry`), lru 0.12 (embedding cache),
  rayon 1 (parallel chunk hashing)
- **Serde**: serde + serde_json
- **Errors**: anyhow (app), thiserror (lib)
- **Tracing**: tracing + tracing-subscriber (env-filter)
- **CLI**: clap 4 (derive)
- **HTTP client**: reqwest 0.12 (rustls-tls, no native-tls dependency)
- **Progress display**: indicatif (progress bars during reindex)
- **Embedded assets**: include_dir (Svelte admin UI compiled into binary)
- **Content hashing**: sha2 (stable file fingerprints for incremental reindex skip)

## Multi-Request Design

- `Arc<SearchAppState>` shared across all axum handlers
- `DashMap<IndexId, Arc<IndexHandle>>` is shard-locked — different indexes never
  contend for locks
- `IndexHandle.indexer: Arc<RwLock<CodeIndexer>>` — reader-priority RwLock; many
  concurrent searches against the same index never block each other
- Indexing operations use `tokio::sync::Semaphore` to prevent thread-pool starvation
  (carry-over fix from open-mpm BUG-2)
- HTTP/2 multiplexing: a single client connection can issue many concurrent searches

## Performance Targets

- **Sub-10ms p50 warm query** on a 100k-chunk index
- **10× faster than ripgrep** on whole-repo conceptual queries
- **HNSW pre-warmed**: index loaded at daemon start, never paged out
  (`Duration::MAX` cool-after)
- **LRU embedding cache** (256 entries): repeated queries skip the embedder entirely
- **~2–3 min for a 14k-file repo** (4 optimizations: INT8 quantized model
  `AllMiniLML6V2Q`, batch upsert into HNSW, split lock via
  `parse_and_embed_files` / `commit_parsed_batch`, batch size 512)

## CLI

```bash
trusty-search start                                  # start HTTP daemon (background)
trusty-search stop                                   # stop daemon (SIGTERM via PID lockfile)
trusty-search index [path] [--name <id>] [--force]  # register + index (primary command)
trusty-search query <text> [--index <id>] [--top-k N] [--json]
trusty-search status                                 # daemon + index overview (alias: health)
trusty-search doctor [--fix]                         # 6-check diagnostic + auto-repair
trusty-search ui [--port N]                          # open web management UI in browser
trusty-search convert project|all [--dry-run]        # migrate from mcp-vector-search
trusty-search serve [--http <addr>]                  # MCP stdio (default) or HTTP/SSE
# Aliases preserved for backward compatibility:
trusty-search init [path]                            # alias for index
trusty-search reindex [path]                         # alias for index --force
```

## The ONE Seam from open-mpm

When integrating into open-mpm, only one cut is needed:

- `src/search/indexer.rs` imports `crate::context::bm25::Bm25Index` →
  re-export from `trusty-search-core/src/bm25.rs`
- `crate::context::indexer::tokenize` → re-export from
  `trusty-search-core/src/bm25.rs` (lives in the same module)

Everything else (the orchestrator, agent runners, REPL, ctrl) stays in open-mpm.

## Workspace Layout

```
trusty-search/
├── Cargo.toml                       workspace + bin manifest
├── CLAUDE.md                        this file
├── CHANGELOG.md
├── README.md
├── .open-mpm/agents/                pm.toml, engineer.toml
├── crates/
│   ├── trusty-search-core/          CodeIndexer, BM25, HNSW, chunking, classifier
│   ├── trusty-search-service/       axum daemon, FileWatcher, client, Svelte UI
│   └── trusty-search-mcp/           MCP server (stdio + HTTP/SSE)
├── src/main.rs                      CLI binary
└── tests/integration_tests.rs
```

### Shared Crates (external, `../trusty-common`)

Three crates extracted from this repo and published at
`github.com/bobmatnyc/trusty-common` (pinned via git tags in `Cargo.toml`):

| Crate | Contents |
|-------|----------|
| `trusty-mcp-core` | `McpRequest`/`McpResponse`/`JsonRpcError`, `run_stdio_loop`, CORS/Trace axum helpers |
| `trusty-embedder` | `Embedder` trait, `FastEmbedder` (LRU + persistent model cache), `MockEmbedder` |
| `trusty-common` | `bind_with_auto_port`, `resolve_data_dir`/`cache_dir`, `ConcurrentRegistry`, `init_tracing`, `daemon_http_client` |

## Development

```bash
# Build
cargo build

# Test
cargo test

# Run daemon with debug logging
RUST_LOG=debug cargo run -- start

# Query a registered index
cargo run -- query "fn authenticate" --index myproject

# Lint (no warnings allowed)
cargo clippy --all-targets --all-features -- -D warnings
```

## Project Status

**Phase**: Production-ready. Full hybrid search pipeline, web UI, MCP server, and
robust CLI are all functional. The project is installable as a machine-wide service
via `cargo install trusty-search`.

**Working**:
- `FastEmbedder` with fastembed-rs, LRU cache, persistent model cache (`~/Library/Caches/trusty-search/models/`)
- `UsearchStore` wired to real usearch HNSW index (add/search/remove)
- `CodeIndexer::search` end-to-end (HNSW + BM25 + RRF fusion)
- Tree-sitter AST-aware chunker (rust, python, js, ts, go, java, c, cpp)
- `EntityExtractor` Phase A structural entities (functions, classes, imports)
- `SymbolGraph` KG expansion (callers_of / callees_of, 1–2 hop, EdgeKind multipliers)
- `FileWatcher` with notify-debouncer-mini, 500ms debounce
- MCP server: full JSON-RPC 2.0 stdio + HTTP/SSE transport, 10 tools
- Daemon: auto-port, fs4 PID lockfile, graceful shutdown, persistent model cache
- Svelte 5 admin UI embedded in binary via `include_dir`
- OpenRouter chat proxy with search context injection
- SSE reindex progress streaming with replay buffer
- Incremental reindex skip via sha2 content fingerprinting
- Parallel batch indexing (rayon + 256-chunk ONNX batches)
- HNSW capacity hinting for large codebases (> 50k chunks)
- Minified JS / build-dir exclusion from indexing
- `trusty-search doctor` 6-check diagnostic with `--fix` auto-repair
- `trusty-search convert` migration from mcp-vector-search
- `indicatif` progress bars for reindex
- HTTP timeouts (2s connect / 5s request) on all daemon calls
- GitHub Actions CI + Dependabot
- 170+ tests passing; clippy clean

**Potential next steps**:
- KG Phase B: IMPORTS/INHERITS edge propagation across file boundaries
- ONNX NER: enable doc comment entity extraction when model file is present
- Benchmark regression CI gate (MRR@5 / Recall@10)
- `cargo install trusty-search` smoke test in CI
- Windows / Linux daemon path support in `trusty-common`
- Blue-green verify canary query tuning (currently uses a fixed probe string)
