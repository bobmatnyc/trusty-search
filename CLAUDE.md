# trusty-search

Machine-wide, blazingly fast hybrid code search service. Single install per machine,
serves multiple named indexes (one per project) via HTTP daemon and MCP server.

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
GET  /health                       liveness probe
GET  /indexes                      list all registered indexes
POST /indexes/:id/search           hybrid search query
POST /indexes/:id/index-file       add/update a file
POST /indexes/:id/remove-file      remove from index
POST /indexes/:id/reindex          full reindex (fire-and-forget)
GET  /indexes/:id/status           index stats (chunks, last-updated)
```

### MCP Tools

- `search_code` — hybrid search query
- `index_file` — add/update one file
- `remove_file` — remove one file
- `list_indexes` — enumerate registered indexes
- `create_index` — register a new index
- `search_health` — daemon liveness

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
- **Parallel indexing**: rayon for chunk hashing + batched fastembed calls

## CLI

```bash
trusty-search serve [--http <addr>]                  # MCP stdio (default) or HTTP/SSE
trusty-search daemon [--port <port>]                 # background HTTP daemon
trusty-search index <path> [--name <id>]             # add a project
trusty-search query <text> [--index <id>] [--top-k N] [--json]
trusty-search status                                 # daemon + index stats
trusty-search watch <path> [--name <id>]             # foreground watcher
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
├── README.md
├── .open-mpm/agents/                pm.toml, engineer.toml
├── crates/
│   ├── trusty-search-core/          CodeIndexer, BM25, HNSW, chunking, classifier
│   ├── trusty-search-service/       axum daemon, FileWatcher, client
│   └── trusty-search-mcp/           MCP server (stdio + HTTP/SSE)
├── src/main.rs                      CLI binary
└── tests/integration_tests.rs
```

## Development

```bash
# Build
cargo build

# Test
cargo test

# Run daemon with debug logging
RUST_LOG=debug cargo run -- daemon

# Query a registered index
cargo run -- query "fn authenticate" --index myproject

# Lint (no warnings allowed)
cargo clippy --all-targets --all-features -- -D warnings
```

## Project Status

**Phase**: Initial scaffolding. Workspace compiles, tests pass, classifier and BM25
are functional. HNSW indexing, KG expansion, FileWatcher, and MCP server stubs
are placeholders for the next implementation phases.

**Working**:
- Workspace builds (`cargo check` passes)
- Query classifier (regex-based intent detection)
- BM25 lexical index (ported from open-mpm)
- Sliding-window chunker
- IndexRegistry with DashMap + Arc<RwLock<CodeIndexer>>
- axum router skeleton (`/health`, `/indexes`, `/indexes/:id/search`, `/indexes/:id/status`)
- CLI subcommands wired with clap

**Next**:
1. Implement `FastEmbedder` with real fastembed-rs `TextEmbedding`
2. Wire `UsearchStore` to actual usearch `Index` (add/search/remove)
3. Implement `CodeIndexer::search` end-to-end (HNSW + BM25 + RRF)
4. Tree-sitter chunker (replace sliding-window with AST-aware chunks)
5. KG: build `SymbolGraph` from tree-sitter, add KG expansion to query pipeline
6. FileWatcher with notify-debouncer-mini + tokio channel
7. MCP server: stdio + HTTP/SSE tool dispatch
8. Daemon: TCP auto-port, fs4 PID lockfile, graceful shutdown
9. CI workflow + `cargo install trusty-search` smoke test
