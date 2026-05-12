# trusty-search

[![CI](https://github.com/bobmatnyc/trusty-search/actions/workflows/ci.yml/badge.svg)](https://github.com/bobmatnyc/trusty-search/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/trusty-search.svg)](https://crates.io/crates/trusty-search)

Machine-wide, blazingly fast hybrid code search service. One install per machine,
one always-on daemon, unlimited named indexes.

> **Note:** Code complexity, smell detection, and quality analysis have moved to
> [trusty-analyzer](https://github.com/bobmatnyc/trusty-analyzer). `trusty-search`
> focuses on hybrid code *search*; `trusty-analyzer` provides static analysis,
> complexity hotspots, and code quality grades. As of v0.2.0 the
> `complexity_hotspots`, `smells`, and `quality` HTTP endpoints are no longer
> served from this binary.

## Features

- **Machine-wide daemon** — single install (`cargo install trusty-search`),
  one process, unlimited registered indexes via `DashMap<IndexId, IndexHandle>`
- **Hybrid search** — BM25 (lexical, zero-dep port with camelCase / snake_case
  splitting) + HNSW vector (usearch 2.25, all-MiniLM-L6-v2 INT8) + Knowledge
  Graph 1–2 hop expansion, fused via Reciprocal Rank Fusion (k = 60, always-on)
- **Query intent routing** — sub-ms regex classifier routes every query to one
  of 5 intents and adjusts α / β weights and KG gating per query
- **KG symbol graph** — petgraph-backed `SymbolGraph` derived from tree-sitter
  parses, with `EdgeKind` (CALLS / IMPORTS / INHERITS / CONTAINS) score
  multipliers; KG expansion is intent-gated (Usage only)
- **Auto-tuned memory tiers** — 5 tiers (Tiny / Small / Medium / Large / XLarge)
  from < 8 GB up to 64+ GB; chunk caps, batch sizes, cache sizes, and BM25 /
  KG limits computed at daemon startup from detected RAM
- **macOS CoreML auto-detection** — on Apple Silicon the ONNX session
  registers the CoreML execution provider automatically (no `--features`
  flag needed since v0.3.13)
- **Multi-index repo support** — drop a `trusty-search.yaml` at the repo root
  to define per-directory named indexes; `trusty-search index` reads it
  automatically (see [`docs/examples/trusty-search.yaml`](docs/examples/trusty-search.yaml))
- **Incremental reindex** — sha2 content fingerprints skip unchanged files
  across daemon restarts; `--force` triggers a full rebuild
- **Zero cold-start queries** — HNSW kept hot (`Duration::MAX` cool-after),
  LRU embedding cache (256+ entries) skips re-embedding on repeat queries
- **Native multi-request** — `Arc<SearchAppState>`, reader-priority `RwLock`,
  axum HTTP/2 — many concurrent searches against the same index never block
- **MCP server** — stdio + HTTP/SSE transports, 11 tools, drop-in for Claude Code
- **Embedded Svelte 5 admin UI** — Collections, Search, Chat, Admin panels
  compiled into the binary via `include_dir!`; open with `trusty-search ui`
- **Migration path** — `trusty-search convert` reads `mcp-vector-search`
  configs and re-registers each project as a named index

## Install

```bash
# CPU-only (default, all platforms incl. macOS Intel + Apple Silicon)
cargo install trusty-search

# CUDA build (NVIDIA GPU, requires CUDA toolkit)
cargo install trusty-search --features cuda
```

On Apple Silicon the CoreML execution provider is registered automatically;
no opt-in flag is needed.

**System requirement:** 16 GB RAM minimum. The daemon performs a hard RAM
check on startup.

## Quick start

```bash
# Start the machine-wide daemon (background)
trusty-search start

# Index a project (auto-registers, skips if already indexed)
trusty-search index ~/Projects/myproj --name myproj

# Hybrid search
trusty-search query "fn authenticate" --index myproj

# Run as MCP server for Claude Code (stdio)
trusty-search serve

# Open the admin UI in your browser
trusty-search ui
```

## Memory tiers (auto-tuned at startup)

| Tier   | Total RAM  | `MEMORY_LIMIT_MB` | `MAX_CHUNKS` | `EMBEDDING_CACHE` | `MAX_BATCH_SIZE` | `BM25_CORPUS_CAP` | `MAX_KG_NODES` |
|--------|------------|-------------------|--------------|-------------------|------------------|-------------------|----------------|
| Tiny   | < 8 GB     | 1 024             | 50 000       | 500               | 64               | 20 000            | 30 000         |
| Small  | 8–15 GB    | 2 048             | 100 000      | 1 000             | 128              | 50 000            | 75 000         |
| Medium | 16–31 GB   | 4 096             | 200 000      | 5 000             | 256              | 100 000           | 150 000        |
| Large  | 32–63 GB   | 8 192             | 400 000      | 10 000            | 512              | 200 000           | 300 000        |
| XLarge | ≥ 64 GB    | 16 384            | 800 000      | 20 000            | 512              | 400 000           | 500 000        |

Env vars (`TRUSTY_MAX_CHUNKS`, `TRUSTY_EMBEDDING_CACHE`, `TRUSTY_MAX_BATCH_SIZE`,
`TRUSTY_BM25_CORPUS_CAP`, `TRUSTY_MAX_KG_NODES`, `TRUSTY_MEMORY_LIMIT_MB`)
always override the tier default. Precedence: shell env > `daemon.env` >
tier default. The resolved tier is logged at daemon startup.

## Query intent → routing weights

| Intent     | α (vector) | β (BM25) | KG-first |
|------------|------------|----------|----------|
| Definition | 0.3        | 0.7      | false    |
| Usage      | 0.5        | 0.5      | **true** |
| Conceptual | 0.8        | 0.2      | false    |
| BugDebt    | 0.1        | 0.9      | false    |
| Unknown    | 0.6        | 0.4      | false    |

The classifier is a sub-ms regex over the query text. KG expansion is gated
to `Usage` intent only — caller/callee chains are scored at 70 % of the
trigger chunk's RRF score.

## CLI

```bash
trusty-search start                                  # start HTTP daemon (background)
trusty-search stop                                   # stop daemon (SIGTERM via PID lockfile)
trusty-search index [path] [--name <id>] [--force]   # register + index (primary command)
                                                     # auto-detects ./trusty-search.yaml
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

## MCP tools

| Tool            | Description                                          |
|-----------------|------------------------------------------------------|
| `search_code`   | Hybrid search (BM25 + HNSW + KG, RRF-fused)          |
| `search_similar`| Code-to-code similarity from a seed file/function    |
| `index_file`    | Add or replace a single file in the index            |
| `remove_file`   | Remove a file and all its chunks                     |
| `list_indexes`  | Enumerate all registered indexes                     |
| `create_index`  | Register a new (empty) index                         |
| `delete_index`  | Drop an index from the registry                      |
| `reindex`       | Fire-and-forget full reindex (SSE progress)          |
| `index_status`  | Per-index chunk count and root path                  |
| `list_chunks`   | Paginated enumeration of chunks `(file, start_line)` |
| `search_health` | Daemon liveness probe                                |
| `chat`          | OpenRouter Q&A with auto-injected search context     |

## Stack

| Component       | Choice                                              |
|-----------------|-----------------------------------------------------|
| Language        | Rust 2021                                           |
| Async runtime   | tokio (full features)                               |
| HTTP            | axum 0.7 + tower-http (CORS, trace, gzip), HTTP/2   |
| Vector store    | usearch 2.25 (HNSW, in-memory, `Arc<RwLock<>>`)     |
| Embeddings      | fastembed 5.x (ONNX, all-MiniLM-L6-v2 INT8, 384-dim)|
| Lexical         | Custom BM25 (zero-dep port, camelCase splitting)    |
| KV store        | redb 2.6                                            |
| Knowledge graph | petgraph 0.6 (`SymbolGraph`)                        |
| File watching   | notify 6 + notify-debouncer-mini 0.4 (500 ms)       |
| Code parsing    | tree-sitter 0.24 (14 grammars)                      |
| Concurrency     | dashmap 5, lru 0.12, rayon 1                        |
| HTTP client     | reqwest 0.12 (rustls-tls)                           |
| CLI             | clap 4 (derive)                                     |
| UI              | Svelte 5, embedded via `include_dir!`               |
| Hashing         | sha2 (incremental reindex fingerprints)             |

## Architecture and HTTP API

See [CLAUDE.md](./CLAUDE.md) for the full HTTP endpoint catalogue, query
pipeline, multi-request design, and release process.

## Documentation

- [CLAUDE.md](./CLAUDE.md) — full architecture + HTTP API reference
- [CHANGELOG.md](./CHANGELOG.md) — release history
- [docs/examples/trusty-search.yaml](./docs/examples/trusty-search.yaml) — multi-index repo config
- [docs/research/](./docs/research/) — design + comparison documents

## License

MIT
