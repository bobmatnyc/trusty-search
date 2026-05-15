# Changelog

All notable changes to trusty-search are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versions correspond to `Cargo.toml` patch releases.

---

## [Unreleased]

_(no unreleased changes)_

---

## [0.3.36] — 2026-05-15

### Added
- **#122** Branch-aware scoring: `branch_files` request field boosts chunks from the current branch by a configurable multiplier (default 1.5×, clamped to `[1.0, 3.0]`); results carry `on_branch: bool`; when `branch_files` is absent, the daemon shells out to `git merge-base` + `git diff --name-only` to derive the file list automatically

### Fixed
- **#121** Embedder init hang: ORT initialization now runs on a blocking thread with a configurable timeout (`TRUSTY_EMBEDDER_INIT_TIMEOUT_SECS`); a timeout surfaces as an error state rather than hanging forever
- **#120** `MEMORY_LIMIT_MB` recomputed as 25% of system RAM instead of a fixed tier cap; `TRUSTY_MEMORY_LIMIT_MB` still overrides

### Changed
- Makefile: `CLOSES` variable support in `patch` target; surgical daemon stop in `deploy` (PID lockfile + `pkill -x`) instead of broad pattern match; `kill` before deploy prevents OOM during compile; `launchctl unload` in deploy target prevents dual-daemon OOM
- Workflow: `closes #N` now required in all resolution commits

---

## [0.3.35] — 2026-05-14

### Fixed
- **#119** CoreML jetsam crash on Apple Silicon (via trusty-embedder v0.1.5 bump)
- **#118** `DELETE /indexes/:id` now persisted to `indexes.toml` so removals survive daemon restart
- Daemon now detaches from terminal when started without `--foreground`, fixing crash when the parent tmux session is killed
- ORT batch size default lowered from 200 MB/slot estimate; clamp changed to `[8, 64]` to prevent 94 GB reindex spikes

### Changed
- `TRUSTY_DEVICE` persisted to `daemon.env` so `--device cpu` survives daemon restarts
- Makefile: `deploy` target added with `CARGO_BUILD_JOBS=2` to prevent OOM kills; `cargo install` removed from `patch` target

---

## [0.3.34] — 2026-05-13

_(version bump only; internal release pipeline fix)_

---

## [0.3.33] — 2026-05-13

### Added
- OpenRPC `rpc.discover` endpoint exposed via trusty-mcp-core helpers
- `SearchMcpService` implements `ServiceDescriptor` (#115)
- Migration script for mcp-vector-search → trusty-search

### Fixed
- **#117** `serve --http` no longer clobbers the daemon's `http_addr` discovery file
- **#116** tree-sitter upgraded to 0.26 for direct linking compatibility with open-mpm
- **#114** glibc 2.34 compatibility for CUDA builds on Amazon Linux 2023
- Test flakiness in file-watcher test on macOS (stray tmpdir events)

### Changed
- trusty-mcp-core bumped to v0.1.1 for OpenRPC support
- trusty-embedder bumped to v0.1.4 for bundled-ort support

---

## [0.3.32] — 2026-05-12

### Fixed
- **#117** `serve --http` flag no longer overwrites the daemon's HTTP address discovery file, preventing the CLI from connecting to the wrong process

---

## [0.3.31] — 2026-05-12

### Added
- **#112** Index context inference and smart fan-out routing: queries against unknown or multi-index contexts are routed to the best-matching indexes automatically
- **#113** Runtime CUDA auto-detection with GPU batch size tuning: when a CUDA-capable GPU is detected, `TRUSTY_MAX_BATCH_SIZE` is auto-bumped to 512; set `TRUSTY_MAX_BATCH_SIZE_EXPLICIT=1` to keep a manually configured value

---

## [0.3.30] — 2026-05-12

### Added
- **#110** `POST /search` fan-out endpoint: search across all registered indexes in a single call, results merged by RRF score
- **#111** `path_filter` field on index registration: restrict which file paths are indexed for a given `IndexId`
- **#91** Classifier extended to match leading-acronym identifiers (`BM25Index`, `IOError`, `URLParser`)

---

## [0.3.29] — 2026-05-12

### Fixed
- `colored::Colorize` import gated to macOS only, fixing compilation on Linux

---

## [0.3.28] — 2026-05-12

### Changed
- **#97** Extracted 52 functions from `main.rs` into `commands/` modules for improved maintainability
- **#98, #109** Extracted helpers in `build.rs` and `spawn_reindex` into focused async helpers
- **#98** Reindex phases extracted into focused async helpers
- **#103** `symbol_graph` helpers extracted to reduce cyclomatic complexity
- **#101, #104** Replaced `unwrap`/`panic`/`process::exit` with proper error handling throughout

### Tests
- **#99** Unit tests added for CLI command handlers and daemon-guard paths

---

## [0.3.27] - 2026-05-12

### Fixed
- **#87** macOS SIGKILL on binary replace: `trusty-search start` now exits with an error if a daemon is already running; `make install` and `make patch` stop the daemon before reinstalling the binary
- **#82** Memory limit enforcement during reindex: tier-based hard caps on `TRUSTY_MAX_BATCH_SIZE` env-var overrides (Medium=64, Large=128, XLarge=256) prevent RSS spikes from misconfigured batch sizes; existing background RSS poller confirmed active
- **#89** ORT ONNX arena pre-allocation: confirmed mitigated by `with_arena_allocator(false)`; tier hard caps add defense-in-depth

### Improved
- **#88** Intent classifier now recognises domain-term Definition queries: PascalCase/CamelCase identifiers, and standalone "definition"/"interface"/"schema"/"type"/"enum"/"model" trigger Definition intent
- **#91** Compound noun classifier: CamelCase compound noun queries (e.g. "QueryClassifier intent classification") now route to Definition intent instead of Unknown
- **#92** Definition-intent ranking: `.md`/`.toml`/`.json`/`.yaml` files scored at 0.5× in RRF fusion for Definition intent only; source files rank first for symbol lookups
- **#94** KG expansion: results merged by score before `take(top_k)`; `hybrid+kg` match_reason now surfaces on large indexes

---

## [0.1.46] — 4 indexing speed optimizations

### Performance
- **INT8 quantized model**: switch fastembed model to `AllMiniLML6V2Q` (INT8 quantized); same 384-dim output, ~30% faster ONNX inference
- **Batch upsert**: accumulate HNSW vectors across all chunks in a reindex pass and call a single `UsearchStore::upsert_batch` instead of N individual inserts; eliminates per-chunk lock overhead
- **Split lock** (`parse_and_embed_files` / `commit_parsed_batch`): parsing + embedding now runs outside the write lock; the write lock is held only for the final redb + HNSW commit, enabling higher concurrency
- **Batch size 512**: increase ONNX batch size 256 → 512 for better GPU/NEON/AVX2 saturation
- Combined target: **< 2 min on a 14k-file repo** (down from ~2–4 min after v0.1.34)

---

## [0.1.45] — multi-line progress + blue-green verify + incremental index

### Added
- **Multi-line progress display**: `indicatif::MultiProgress` shows concurrent bars — one per active reindex stream — plus a summary line with aggregate `chunks/s`
- **Blue-green verify**: after a reindex completes, a lightweight verification pass confirms the new HNSW index answers a canary query before swapping the live handle; prevents silent corruption on large repos
- **Incremental index flag** (`--incremental`, default on): skips files whose sha2 fingerprint matches the stored value even across daemon restarts; `--force` still triggers a full rebuild

---

## [0.1.44] — async HTTP server in trusty-common

### Added
- `trusty-common`: `server` module with `with_standard_middleware` (axum-server feature) and `daemon_http_client` helper
- `trusty-search-service`: `build_router` uses `with_standard_middleware`
- `main.rs`: all daemon HTTP call sites use `trusty_common::server::daemon_http_client`

---

## [0.1.43] — HTTP timeouts (fix status hang)

### Fixed
- Add 2s connect / 5s request timeouts to all daemon HTTP calls via `daemon_client()` helper; `status`, `health`, `doctor`, `query`, and `reindex` now fail fast with a clear error instead of hanging when the daemon is not running

---

## [0.1.42] — status/health unified + doctor command

### Added
- `status` and `health` are now aliases for the same `run_status()` handler; output shows daemon version, port, and per-index chunk counts
- `trusty-search doctor`: 6-check diagnostic (daemon liveness, model cache, data-dir writability, stale lockfile, empty indexes, port reachability) with colored ✓/⚠/✗ output
- `doctor --fix`: auto-repairs stale lockfile and empty indexes via `run_reindex` with progress bar; exits 1 on any error

---

## [0.1.41] — `index` primary command + indicatif progress bar

### Added
- `trusty-search index [PATH] [--name <id>] [--force]`: auto-registers the index if absent, skips if already indexed, `--force` triggers full reindex; replaces the awkward `init` + `reindex` two-step
- `indicatif` progress bar during reindex: `⟳ Indexing {id} [████░░] {pos}/{len} files — {eta} remaining`; updates on each SSE batch event, finishes with chunk count and elapsed time
- `register_index_with_daemon()` and `fetch_chunk_count()` helpers shared between `Init` and `Index` commands
- `init` and `reindex` preserved as backward-compatible aliases

---

## [0.1.40] — wire shared crates throughout

### Refactored
- `ui.rs`: replace inline OpenRouter HTTP client with `trusty_common::openrouter_chat` (~50 lines removed)
- All three shared crates (`trusty-mcp-core`, `trusty-embedder`, `trusty-common`) fully wired into every consumer
- Pin shared crates to public git tags on `bobmatnyc/trusty-common` (v0.1.0); remove in-tree copies (net −985 LOC)

---

## [0.1.39] — exclude minified JS/build dirs from indexing

### Added
- `should_skip_path()`: skips `*.min.js/css`, `*.bundle.js`, `*.chunk.js`, hashed bundles, binary extensions, files > 1 MB
- `should_skip_content()`: heuristic minification detection for `.js/mjs/cjs` (< 5 lines with any line > 500 chars)
- `SKIP_DIRS`: `node_modules`, `dist`, `build`, `target`, `.git`, `__pycache__`, `.next`, `.nuxt`, `.svelte-kit`, `vendor`, `.gradle`, `.m2`, `coverage`, `.nyc_output`
- Reindex emits SSE `"skip"` event with `reason:"minified"` for content-skipped files
- 14 new tests covering all skip patterns

---

## [0.1.38] — shared crates (trusty-mcp-core, trusty-embedder, trusty-common)

### Added
- `trusty-mcp-core`: `McpRequest`/`McpResponse`/`JsonRpcError`, error code constants, `run_stdio_loop` generic async stdio handler, CORS/Trace axum helpers
- `trusty-embedder`: `Embedder` trait, `FastEmbedder` with LRU cache + persistent model cache dir, `EMBED_DIM=384`, `MockEmbedder` for tests, `embed_one` helper
- `trusty-common`: `bind_with_auto_port`, `resolve_data_dir`/`cache_dir`, `ConcurrentRegistry<K,V>`, `init_tracing`, `maybe_disable_color`
- All three registered in workspace; 163 tests passing

### Refactored
- Adopt shared crates and delete inlined equivalents; `trusty-search-core::embed` becomes a thin facade over the shared `Embedder` trait
- Daemon port binding goes through shared async helper; `main.rs` uses `init_tracing`/`maybe_disable_color`

---

## [0.1.37] — daemon early-exit + model cache

### Fixed
- `is_already_running()` checks lockfile before `FastEmbedder::new()` so "another daemon running" exits in < 1 ms instead of after an 86 MB model download

### Added
- `model_cache_dir()` resolves `~/Library/Caches/trusty-search/models/`; model downloads once and loads from disk on all subsequent daemon starts
- `serial_test` on embed tests prevents `hf_hub` lock-file races in parallel test runs

---

## [0.1.36] — HTTP ↔ MCP functional parity

### Added
- Four missing MCP tools added for full HTTP endpoint coverage:
  - `delete_index` ← `DELETE /indexes/:id`
  - `reindex` ← `POST /indexes/:id/reindex`
  - `index_status` ← `GET /indexes/:id/status`
  - `chat` ← `POST /chat` (OpenRouter proxy)
- `test_tools_list_complete` asserts HTTP/MCP parity; 151 tests passing

---

## [0.1.35] — Svelte admin UI + MCP stdio server

### Added
- **Web management UI** served at `GET /ui`:
  - Collections panel: list/create/delete indexes, reindex with live SSE progress
  - Search panel: single and cross-collection hybrid search, `match_reason` badges, compact/full snippet toggle
  - Chat panel: OpenRouter-backed conversational Q&A (gated by `OPENROUTER_API_KEY`)
  - Admin panel: daemon info, per-file index/remove ops, danger zone
  - Static assets embedded at compile time via `include_dir`
- `POST /chat` endpoint proxies to OpenRouter with search context injection
- `DELETE /indexes/:id` endpoint
- `trusty-search ui` subcommand: start daemon + open browser
- **MCP stdio JSON-RPC server** (full JSON-RPC 2.0 over stdin/stdout, protocol 2024-11-05):
  - `initialize` handshake, `notifications/initialized` suppressed correctly
  - `tools/list`: all 6 tools (`search_code`, `index_file`, `remove_file`, `list_indexes`, `create_index`, `search_health`)
  - `tools/call`: MCP-spec content envelope with `isError` flag
  - Graceful shutdown on stdin EOF; errors to stderr only

---

## [0.1.34] — 4× faster indexing

### Performance
- Eliminate 452 symbol-graph rebuilds per reindex: `index_files_batch_no_rebuild` defers graph rebuild to once at completion
- `resolve_callee` O(N×S) linear suffix scan replaced with O(1) hash lookup using precomputed simple-name → `NodeIndex` map
- Batch size 32 → 128 for better ONNX saturation
- `drain` RawChunk corpus instead of cloning (saves ~115k allocations per reindex)
- Expected reduction on large monorepos: ~46 min → 2–4 min

---

## [0.1.33] — hot BM25/HNSW/LRU fixes + CLI stubs

### Fixed
- **Bug A**: Wire `FastEmbedder` + `UsearchStore` in `create_index_handler`; HNSW now actually stores and returns vector results
- **Bug B**: Replace per-query BM25 rebuild with persistent `Arc<RwLock<Bm25Index>>` maintained incrementally at index time; search is O(df_i) not O(corpus)
- **Bug C**: LRU embedding cache now deduplicates across requests (was masked by Bug B)

### Added
- `status` CLI: daemon health + per-index chunk counts
- `query` CLI: `POST /indexes/:id/search` with ranked output or `--json`
- `init` now calls `POST /indexes` on the daemon (fixes misleading "Registered" message)

---

## [0.1.32] — convert command

### Added
- `trusty-search convert project|all`: migrate indexes from mcp-vector-search by reading `.mcp-vector-search/config.json` files
  - `convert project`: git-style upward discovery from CWD
  - `convert all`: scans `~` at depth 6, skipping noise dirs
  - `--dry-run`: preview without contacting the daemon
  - `--concurrency`: bounds parallel migrations via `tokio::Semaphore`
  - Idempotent: existing indexes detected via daemon `{created: false}` response

---

## [0.1.31] — large codebase performance

### Performance
- `CodeIndexer::index_files_batch`: parses N files in parallel via rayon, embeds all chunks in 256-chunk ONNX batches, takes corpus write lock once per batch
- Incremental hash skip: files whose content hash matches the previous reindex are skipped; new SSE events: `"skip"`, `"batch"` (with `chunks_per_sec`), `"complete"` now carries skipped count
- `UsearchStore::with_capacity_hint`: tunes HNSW (connectivity=32, expansion_add=128, expansion_search=64) when expected chunk count > 50k
- `.gradle`/`.groovy`/`.kts`/`.mjs`/`.cjs` added to `SOURCE_EXTS`; Java/Gradle build dirs pruned from walker

---

## [0.1.30] — start/stop CLI

### Added
- `trusty-search start`: starts the HTTP daemon (replaces `daemon`)
- `trusty-search stop`: reads PID from fs4 lockfile, sends SIGTERM, polls up to 5s for port file to disappear

---

## [0.1.29] — reindex + SSE progress streaming

### Added
- `walker::walk_source_files`: walkdir-based, skips `.git`/`target`/`node_modules`/etc.
- `POST /indexes/:id/reindex`: spawns background reindex task with optional `{root_path}` body
- `GET /indexes/:id/reindex/stream`: SSE endpoint emitting `start`/`progress`/`complete`/`error` events with replay buffer for late subscribers
- `trusty-search reindex [PATH]` CLI: connects to SSE stream, renders live percentage/file progress
- `trusty-search add <PATH>`: walks directories and indexes every source file match
- `trusty-search remove <FILE>`: calls `/indexes/:id/remove-file`
- `trusty-search list`: calls `/indexes` and renders registry

---

## [0.1.28] — SCIP ingest interface

### Added
- SCIP ingest interface with `CodeEntityIndex` trait and `from_refs` constructor ([#24])

---

## [0.1.27] — ONNX NER gated

### Added
- ONNX NER for doc comment NLP entity extraction, gated by model file presence ([#23])

---

## [0.1.26] — ConceptCluster k-means

### Added
- `ConceptCluster` entities via fastembed + linfa k-means ([#22])

---

## [0.1.25] — complexity metrics

### Added
- Complexity and code quality metrics per chunk ([#32])

---

## [0.1.24] — search_similar

### Added
- Code-to-code similarity search and `search_similar` MCP tool ([#31])

---

## [0.1.23] — git blame integration

### Added
- Git blame integration per-chunk with temporal decay scoring ([#30])

---

## [0.1.22] — benchmark harness

### Added
- Benchmark harness: MRR@5 and Recall@10 evaluation ([#25])

---

## [0.1.21] — canonical facts table

### Added
- Canonical facts table with provenance tracking and HTTP query API ([#26])

---

## [0.1.20] — MMR diversity

### Added
- MMR (Maximal Marginal Relevance) diversity pass after RRF fusion ([#28])

---

## [0.1.19] — entity-match RRF lane

### Added
- Entity-match RRF lane for exact symbol name queries ([#20])

---

## [0.1.18] — KG rich edge types

### Added
- Knowledge Graph CALLS/IMPORTS/INHERITS/CONTAINS edges derived from chunk AST data ([#33])

---

## [0.1.17] — virtual_terms in BM25

### Added
- Populate `virtual_terms` from entities and append to BM25 documents for enriched lexical matching ([#19])

---

## [0.1.16] — intent-gated KG traversal

### Added
- Intent-gated KG traversal with `EdgeKind` score multipliers ([#18])

---

## [0.1.15] — EntityExtractor Phase A

### Added
- `EntityExtractor` Phase A: structural entities (functions, classes, imports) ([#17])

---

## [0.1.14] — CodeChunk extended fields

### Added
- Extend `CodeChunk` with `chunk_type`, `calls`, `inherits_from`, `complexity_score`, `chunk_depth` ([#29])

---

## [0.1.13] — BM25 three-pass tokenizer

### Added
- Three-pass BM25 tokenizer with camelCase and snake_case splitting ([#27])

---

## [0.1.12] — QueryClassifier entity keywords

### Added
- Extend `QueryClassifier` with entity-type keyword recognition ([#21])

---

## [0.1.11] — RawEntity + EdgeKind schema

### Added
- Canonical `RawEntity` schema and `EdgeKind` enum ([#16])

---

## [0.1.10] — CI + Dependabot

### Added
- GitHub Actions CI workflow and Dependabot config ([#9])

---

## [0.1.9] — daemon + graceful shutdown

### Added
- Daemon with PID lockfile (fs4), auto-port binding, graceful shutdown ([#8])

---

## [0.1.8] — MCP server

### Added
- MCP server with stdio and HTTP/SSE transport ([#7])

---

## [0.1.7] — FileWatcher

### Added
- `FileWatcher` with notify-debouncer-mini, 500ms debounce, fsevent backend ([#6])

---

## [0.1.6] — SymbolGraph KG expansion

### Added
- Build `SymbolGraph` from tree-sitter parse output; wire KG expansion (callers_of/callees_of) into the query pipeline ([#5])

---

## [0.1.5] — AST chunker + entity extraction

### Added
- Replace sliding-window chunker with tree-sitter AST-aware chunker ([#4])
- Initial `EntityExtractor` ([#17])

---

## [0.1.4] — search pipeline

### Added
- `CodeIndexer::search` end-to-end: HNSW + BM25 + RRF fusion ([#3])

---

## [0.1.3] — CLI redesign with auto-detection

### Added
- Project auto-detection and clean CLI help structure ([#14])

---

## [0.1.2] — UsearchStore HNSW wiring

### Added
- Wire `UsearchStore` to real usearch HNSW `Index` for add/search/remove ([#2])

---

## [0.1.1] — FastEmbedder implementation

### Added
- `FastEmbedder` with fastembed-rs + LRU cache ([#1])

---

## [0.1.0] — initial scaffold

### Added
- Workspace scaffold: `trusty-search-core`, `trusty-search-service`, `trusty-search-mcp`, CLI binary
- Query classifier (regex-based intent detection)
- BM25 lexical index (ported from open-mpm)
- `IndexRegistry` with `DashMap` + `Arc<RwLock<CodeIndexer>>`
- axum router skeleton

[Unreleased]: https://github.com/bobmatnyc/trusty-search/compare/v0.3.36...HEAD
[0.3.36]: https://github.com/bobmatnyc/trusty-search/compare/v0.3.35...v0.3.36
[0.3.35]: https://github.com/bobmatnyc/trusty-search/compare/v0.3.34...v0.3.35
[0.3.34]: https://github.com/bobmatnyc/trusty-search/compare/v0.3.33...v0.3.34
[0.3.33]: https://github.com/bobmatnyc/trusty-search/compare/v0.3.32...v0.3.33
[0.3.32]: https://github.com/bobmatnyc/trusty-search/compare/v0.3.31...v0.3.32
[0.3.31]: https://github.com/bobmatnyc/trusty-search/compare/v0.3.30...v0.3.31
[0.3.30]: https://github.com/bobmatnyc/trusty-search/compare/v0.3.29...v0.3.30
[0.3.29]: https://github.com/bobmatnyc/trusty-search/compare/v0.3.28...v0.3.29
[0.3.28]: https://github.com/bobmatnyc/trusty-search/compare/v0.3.27...v0.3.28
[0.3.27]: https://github.com/bobmatnyc/trusty-search/compare/v0.3.26...v0.3.27
[0.1.46]: https://github.com/bobmatnyc/trusty-search/compare/v0.1.45...v0.1.46
[0.1.45]: https://github.com/bobmatnyc/trusty-search/compare/v0.1.44...v0.1.45
[0.1.44]: https://github.com/bobmatnyc/trusty-search/compare/v0.1.43...v0.1.44
[0.1.43]: https://github.com/bobmatnyc/trusty-search/compare/v0.1.42...v0.1.43
[0.1.42]: https://github.com/bobmatnyc/trusty-search/compare/v0.1.41...v0.1.42
[0.1.41]: https://github.com/bobmatnyc/trusty-search/compare/v0.1.40...v0.1.41
[0.1.40]: https://github.com/bobmatnyc/trusty-search/compare/v0.1.39...v0.1.40
[0.1.39]: https://github.com/bobmatnyc/trusty-search/compare/v0.1.38...v0.1.39
[0.1.38]: https://github.com/bobmatnyc/trusty-search/compare/v0.1.37...v0.1.38
[0.1.37]: https://github.com/bobmatnyc/trusty-search/compare/v0.1.36...v0.1.37
[0.1.36]: https://github.com/bobmatnyc/trusty-search/compare/v0.1.35...v0.1.36
[0.1.35]: https://github.com/bobmatnyc/trusty-search/compare/v0.1.34...v0.1.35
[0.1.34]: https://github.com/bobmatnyc/trusty-search/compare/v0.1.33...v0.1.34
[0.1.33]: https://github.com/bobmatnyc/trusty-search/compare/v0.1.32...v0.1.33
[0.1.32]: https://github.com/bobmatnyc/trusty-search/compare/v0.1.31...v0.1.32
[0.1.31]: https://github.com/bobmatnyc/trusty-search/compare/v0.1.30...v0.1.31
[0.1.30]: https://github.com/bobmatnyc/trusty-search/compare/v0.1.29...v0.1.30
[0.1.29]: https://github.com/bobmatnyc/trusty-search/compare/v0.1.28...v0.1.29
[0.1.28]: https://github.com/bobmatnyc/trusty-search/compare/v0.1.27...v0.1.28
[0.1.27]: https://github.com/bobmatnyc/trusty-search/compare/v0.1.26...v0.1.27
[0.1.26]: https://github.com/bobmatnyc/trusty-search/compare/v0.1.25...v0.1.26
[0.1.25]: https://github.com/bobmatnyc/trusty-search/compare/v0.1.24...v0.1.25
[0.1.24]: https://github.com/bobmatnyc/trusty-search/compare/v0.1.23...v0.1.24
[0.1.23]: https://github.com/bobmatnyc/trusty-search/compare/v0.1.22...v0.1.23
[0.1.22]: https://github.com/bobmatnyc/trusty-search/compare/v0.1.21...v0.1.22
[0.1.21]: https://github.com/bobmatnyc/trusty-search/compare/v0.1.20...v0.1.21
[0.1.20]: https://github.com/bobmatnyc/trusty-search/compare/v0.1.19...v0.1.20
[0.1.19]: https://github.com/bobmatnyc/trusty-search/compare/v0.1.18...v0.1.19
[0.1.18]: https://github.com/bobmatnyc/trusty-search/compare/v0.1.17...v0.1.18
[0.1.17]: https://github.com/bobmatnyc/trusty-search/compare/v0.1.16...v0.1.17
[0.1.16]: https://github.com/bobmatnyc/trusty-search/compare/v0.1.15...v0.1.16
[0.1.15]: https://github.com/bobmatnyc/trusty-search/compare/v0.1.14...v0.1.15
[0.1.14]: https://github.com/bobmatnyc/trusty-search/compare/v0.1.13...v0.1.14
[0.1.13]: https://github.com/bobmatnyc/trusty-search/compare/v0.1.12...v0.1.13
[0.1.12]: https://github.com/bobmatnyc/trusty-search/compare/v0.1.11...v0.1.12
[0.1.11]: https://github.com/bobmatnyc/trusty-search/compare/v0.1.10...v0.1.11
[0.1.10]: https://github.com/bobmatnyc/trusty-search/compare/v0.1.9...v0.1.10
[0.1.9]: https://github.com/bobmatnyc/trusty-search/compare/v0.1.8...v0.1.9
[0.1.8]: https://github.com/bobmatnyc/trusty-search/compare/v0.1.7...v0.1.8
[0.1.7]: https://github.com/bobmatnyc/trusty-search/compare/v0.1.6...v0.1.7
[0.1.6]: https://github.com/bobmatnyc/trusty-search/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/bobmatnyc/trusty-search/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/bobmatnyc/trusty-search/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/bobmatnyc/trusty-search/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/bobmatnyc/trusty-search/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/bobmatnyc/trusty-search/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/bobmatnyc/trusty-search/releases/tag/v0.1.0
