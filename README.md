# trusty-search

Machine-wide, blazingly fast hybrid code search service.

- **Single install per machine** (not per-project), one daemon serves all your projects
- **Multiple named indexes** — register one per project
- **Hybrid search**: BM25 (lexical) + HNSW vector (semantic) + KG expansion, fused via Reciprocal Rank Fusion
- **Query-type routing**: Definition / Usage / Conceptual / BugDebt — each gets the optimal weighting
- **MCP server**: stdio + HTTP/SSE — drop-in for Claude Code
- **Zero cold-start**: HNSW stays hot, embeddings LRU-cached, sub-10ms p50 warm queries
- **Native multi-request**: `Arc<SearchState>`, concurrent reads via `RwLock`, axum HTTP/2

## Install

```bash
cargo install trusty-search
```

## Quick start

```bash
# Start the machine-wide daemon
trusty-search daemon

# Index a project
trusty-search index ~/Projects/myproj --name myproj

# Query
trusty-search query "fn authenticate" --index myproj

# Or run as MCP server for Claude Code
trusty-search serve
```

## Architecture

See [CLAUDE.md](./CLAUDE.md) for full architecture, query pipeline, and HTTP API reference.
