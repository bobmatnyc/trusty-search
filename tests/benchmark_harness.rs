//! Benchmark harness for entity-enriched KG search quality.
//!
//! Why: ad-hoc spot checks aren't enough to know whether a refactor regressed
//! retrieval. This harness fixes a small per-intent query set against the
//! `trusty-search-core` source tree and reports MRR@5 + Recall@10 + latency.
//! What: indexes `crates/trusty-search-core/src/` end-to-end (FastEmbedder +
//! UsearchStore + BM25 via the live `CodeIndexer::search` pipeline), runs the
//! query set, prints a per-intent table, and asserts a soft mean MRR@5 floor.
//! Test: each `#[ignore] #[tokio::test]` corresponds to one intent class.
//!
//! Run: cargo test --test benchmark_harness -- --include-ignored --nocapture
//!
//! Soft thresholds, not hard contracts: changes that drop MRR@5 below 0.3 mean
//! warrant scrutiny but the harness itself stays advisory.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use trusty_search::core::indexer::{CodeChunk, CodeIndexer, SearchQuery};
use trusty_search::core::store::{UsearchStore, VectorStore};
use trusty_search::core::{Embedder, FastEmbedder};

// ---------------------------------------------------------------------------
// Query corpora — `(query_text, expected_substring)`. The substring is matched
// case-insensitively against the chunk content (for substring queries) or the
// `function_name` field (for symbolic queries).
// ---------------------------------------------------------------------------

const DEFINITION_QUERIES: &[(&str, &str)] = &[
    ("UsearchStore", "UsearchStore"),
    ("CodeChunk struct", "CodeChunk"),
    ("QueryIntent enum", "QueryIntent"),
    ("RRF fusion", "rrf"),
    ("BM25 tokenize", "tokenize"),
    ("SymbolGraph", "SymbolGraph"),
    ("EntityExtractor", "EntityExtractor"),
    ("FastEmbedder", "FastEmbedder"),
    ("FactRecord", "FactRecord"),
    ("EdgeKind score_multiplier", "score_multiplier"),
];

const USAGE_QUERIES: &[(&str, &str)] = &[
    ("where is FastEmbedder called", "embed"),
    ("callers of kg_expand", "kg_expand"),
    ("uses of redb table", "redb"),
    ("search method calls", "search"),
    ("index_file usage", "index_file"),
    ("entity_exact_match called", "entity_exact_match"),
    ("mmr_rerank called", "mmr_rerank"),
    ("bm25 tokenize usage", "tokenize"),
    ("cosine_similarity callers", "cosine"),
    ("fact_hash usage", "fact_hash"),
];

const CONCEPTUAL_QUERIES: &[(&str, &str)] = &[
    ("how does BM25 scoring work", "bm25"),
    ("what handles concurrent reads", "rwlock"),
    ("how is RRF fusion computed", "rrf"),
    ("knowledge graph expansion", "kg"),
    ("how are embeddings cached", "cache"),
    ("intent routing weights", "alpha"),
    ("how are chunks stored", "redb"),
    ("file watching debounce", "debounce"),
    ("MCP tool dispatch", "dispatch"),
    ("entity extraction tree-sitter", "entity"),
];

const BUGDEBT_QUERIES: &[(&str, &str)] = &[
    ("panic sites in indexer", "panic"),
    ("unwrap calls in search path", "unwrap"),
    ("todo in codebase", "todo"),
    ("unimplemented stubs", "unimplemented"),
    ("error handling gaps", "error"),
    ("raises CapacityError", "capacity"),
    ("bail macro usage", "bail"),
    ("anyhow error sites", "anyhow"),
    ("missing test coverage", "test"),
    ("deprecated code", "deprecated"),
];

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/// Mean Reciprocal Rank at k: 1/(rank+1) of the first relevant hit, else 0.
fn mrr_at_k(results: &[CodeChunk], expected: &str, k: usize) -> f32 {
    let needle = expected.to_lowercase();
    results
        .iter()
        .take(k)
        .enumerate()
        .find(|(_, c)| {
            c.content.to_lowercase().contains(&needle)
                || c.function_name
                    .as_deref()
                    .unwrap_or("")
                    .to_lowercase()
                    .contains(&needle)
        })
        .map(|(i, _)| 1.0 / (i + 1) as f32)
        .unwrap_or(0.0)
}

/// Recall@k: did at least one of the top-k results contain the expected substring?
fn recall_at_k(results: &[CodeChunk], expected: &str, k: usize) -> bool {
    let needle = expected.to_lowercase();
    results
        .iter()
        .take(k)
        .any(|c| c.content.to_lowercase().contains(&needle))
}

// ---------------------------------------------------------------------------
// Indexing fixture — shared across the four benchmark tests.
// ---------------------------------------------------------------------------

/// Locate `crates/trusty-search-core/src/` relative to the workspace root so
/// the harness works regardless of which directory `cargo test` was launched in.
fn core_src_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("crates")
        .join("trusty-search-core")
        .join("src")
}

/// Build a fresh `CodeIndexer` populated with every `.rs` file under
/// `crates/trusty-search-core/src/`. Returns the indexer once population
/// completes (synchronous from the caller's perspective).
async fn build_indexer() -> CodeIndexer {
    let embedder: Arc<dyn Embedder> = Arc::new(
        FastEmbedder::new()
            .await
            .expect("init FastEmbedder (downloads model on first run)"),
    );
    let store: Arc<dyn VectorStore> = Arc::new(UsearchStore::new(384).expect("init UsearchStore"));
    let indexer = CodeIndexer::new("bench", core_src_dir()).with_components(embedder, store);

    // Walk core src and feed every .rs file through index_file.
    let src = core_src_dir();
    for entry in walkdir::WalkDir::new(&src)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("rs") {
            continue;
        }
        let rel = path
            .strip_prefix(&src)
            .unwrap_or(path)
            .to_string_lossy()
            .into_owned();
        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => continue,
        };
        if let Err(e) = indexer.index_file(&rel, &content).await {
            eprintln!("index_file({rel}) failed: {e}");
        }
    }

    indexer
}

/// Run a query set, print a results table, and return the mean MRR@5 so the
/// caller can assert on the soft floor.
async fn run_bench(label: &str, indexer: &CodeIndexer, queries: &[(&str, &str)]) -> f32 {
    println!("\n=== {label} ===");
    println!(
        "| {:<40} | {:>6} | {:>9} | {:>10} |",
        "query", "MRR@5", "Recall@10", "latency_ms"
    );
    println!("|{:-<42}|{:-<8}|{:-<11}|{:-<12}|", "", "", "", "");

    let mut mrr_sum = 0.0_f32;
    let mut recall_hits = 0_usize;
    for (q, expected) in queries {
        let sq = SearchQuery {
            text: (*q).to_string(),
            top_k: 10,
            expand_graph: true,
            compact: false,
        };
        let started = Instant::now();
        let results = indexer
            .search(&sq)
            .await
            .expect("search must not fail in bench harness");
        let latency_ms = started.elapsed().as_millis();
        let mrr = mrr_at_k(&results, expected, 5);
        let rec = recall_at_k(&results, expected, 10);
        mrr_sum += mrr;
        if rec {
            recall_hits += 1;
        }
        println!(
            "| {:<40} | {:>6.3} | {:>9} | {:>10} |",
            truncate(q, 40),
            mrr,
            if rec { "yes" } else { "no" },
            latency_ms
        );
    }

    let n = queries.len() as f32;
    let mean_mrr = mrr_sum / n;
    let recall_pct = (recall_hits as f32 / n) * 100.0;
    println!(
        "mean MRR@5 = {:.3}  |  Recall@10 = {:.0}% ({}/{})",
        mean_mrr,
        recall_pct,
        recall_hits,
        queries.len()
    );
    mean_mrr
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(n - 1).collect();
        out.push('…');
        out
    }
}

const SOFT_MRR_FLOOR: f32 = 0.3;

// ---------------------------------------------------------------------------
// Tests — `#[ignore]` so they don't run in `cargo test --workspace`.
// ---------------------------------------------------------------------------

#[ignore]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn bench_definition_queries() {
    let indexer = build_indexer().await;
    let mean = run_bench("definition", &indexer, DEFINITION_QUERIES).await;
    assert!(
        mean >= SOFT_MRR_FLOOR,
        "definition mean MRR@5 = {mean:.3} below soft floor {SOFT_MRR_FLOOR}"
    );
}

#[ignore]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn bench_usage_queries() {
    let indexer = build_indexer().await;
    let mean = run_bench("usage", &indexer, USAGE_QUERIES).await;
    assert!(
        mean >= SOFT_MRR_FLOOR,
        "usage mean MRR@5 = {mean:.3} below soft floor {SOFT_MRR_FLOOR}"
    );
}

#[ignore]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn bench_conceptual_queries() {
    let indexer = build_indexer().await;
    let mean = run_bench("conceptual", &indexer, CONCEPTUAL_QUERIES).await;
    assert!(
        mean >= SOFT_MRR_FLOOR,
        "conceptual mean MRR@5 = {mean:.3} below soft floor {SOFT_MRR_FLOOR}"
    );
}

#[ignore]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn bench_bugdebt_queries() {
    let indexer = build_indexer().await;
    let mean = run_bench("bugdebt", &indexer, BUGDEBT_QUERIES).await;
    assert!(
        mean >= SOFT_MRR_FLOOR,
        "bugdebt mean MRR@5 = {mean:.3} below soft floor {SOFT_MRR_FLOOR}"
    );
}
