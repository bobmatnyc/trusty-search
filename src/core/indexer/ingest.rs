//! Ingestion + indexing pipeline for [`CodeIndexer`].
//!
//! Why: parsing files, embedding chunks, and committing batched results into
//! the corpus / BM25 / HNSW / KG is the single largest cluster of behaviour on
//! `CodeIndexer`. Lifting it into a dedicated module keeps the search hot path
//! and the persistence module focused.
//! What: `add_chunk`, `index_file`, the NLP enrichment helper,
//! `index_files_batch[_no_rebuild]`, `parse_and_embed_files`, the parallel
//! parse + batched embed helpers, `commit_parsed_batch`, every `commit_*`
//! helper, and `rebuild_symbol_graph[_now]`.
//! Test: covered by `test_index_files_batch_*`, `test_virtual_terms_populated_from_entities`,
//! and every ingest-flavoured test in `indexer::tests`.

use std::sync::Arc;

use anyhow::{Context, Result};

use crate::core::chunker::{chunk_ast, RawChunk};
use crate::core::entity::RawEntity;
use crate::core::symbol_graph::{ChunkTuple, SymbolGraph};

use super::{
    embed_batch_size, max_chunks_per_index, populate_virtual_terms, CodeIndexer, CommitTimings,
    ParsedBatch,
};

impl CodeIndexer {
    /// Rebuild the symbol graph from the current corpus. Called after any
    /// mutation (`add_chunk`, `remove_chunk`, `index_file`). Rebuilding is
    /// O(N + E) over chunks/calls and the corpus is small + in-memory, so we
    /// favour simplicity over incremental maintenance.
    pub(super) async fn rebuild_symbol_graph(&self) {
        // Issue (180GB RSS fix): the temporary `Vec<ChunkTuple>` snapshot clones
        // every chunk's strings (id, file, function_name, calls, inherits_from)
        // and can hit 1-2 GB on a 1M-chunk corpus. We can't avoid the snapshot
        // entirely (build_from_chunks needs a slice, and we don't want to hold
        // the chunks read lock across `add_node`), but we cap snapshot size to
        // the same KG node cap so we don't allocate more than we'll actually
        // use. Chunks past the cap can't contribute new symbols anyway.
        let kg_cap = crate::core::symbol_graph::max_kg_nodes();
        let chunks = self.chunks.read().await;
        // Pre-size for the worst case. When `kg_cap == 0` (unlimited) fall back
        // to corpus size. Multiplied by 2 because the cap is on unique symbols
        // and a single function might be defined across a handful of duplicates.
        let snapshot_cap = if kg_cap == 0 {
            chunks.len()
        } else {
            // Heuristic: most chunks have a function name; cap snapshot at
            // 2× the KG node cap to leave headroom for duplicates while still
            // bounding peak allocation.
            (kg_cap.saturating_mul(2)).min(chunks.len())
        };
        let mut tuples: Vec<ChunkTuple> = Vec::with_capacity(snapshot_cap);
        for c in chunks.values() {
            if tuples.len() >= snapshot_cap {
                break;
            }
            tuples.push((
                c.id.clone(),
                c.file.clone(),
                c.function_name.clone(),
                c.calls.clone(),
                c.inherits_from.clone(),
                c.chunk_type.clone(),
            ));
        }
        drop(chunks);
        let new_graph = Arc::new(SymbolGraph::build_from_chunks(&tuples));
        // Free the snapshot immediately — it's the second-largest allocation
        // in this function and we don't need it past `build_from_chunks`.
        drop(tuples);
        *self.symbol_graph.write().await = new_graph;
    }

    /// Add (or replace) a chunk in the corpus. If an embedder + store are
    /// attached, the chunk is also embedded and upserted into the HNSW index.
    pub async fn add_chunk(&self, chunk: RawChunk) -> Result<()> {
        let id = chunk.id.clone();

        // Issue #75: hard cap per-index chunk count to bound RAM growth.
        // Upserts (existing id) are always allowed; only brand-new ids hit
        // the cap. Failing fast here keeps HNSW / BM25 / corpus in sync.
        {
            let chunks = self.chunks.read().await;
            let cap = max_chunks_per_index();
            if !chunks.contains_key(&id) && chunks.len() >= cap {
                tracing::warn!(
                    "index '{}' chunk cap ({}) reached — skipping chunk {}",
                    self.index_id,
                    cap,
                    id
                );
                return Ok(());
            }
        }

        if let (Some(embedder), Some(store)) = (&self.embedder, &self.store) {
            let vec = embedder
                .embed(&chunk.content)
                .await
                .context("embed chunk content")?;
            store
                .upsert(&id, vec.clone())
                .await
                .context("upsert chunk vector")?;
            // Cache for MMR diversity (#28). Cheap O(1) write under the corpus
            // mutation path so the search hot loop never has to re-embed.
            // LRU `put` evicts the oldest entry when at capacity.
            self.chunk_embeddings.write().await.put(id.clone(), vec);
        }

        // Maintain the persistent BM25 index. Doing this on every write keeps
        // the search path O(query_terms · postings) instead of O(corpus).
        let bm25_text = Self::bm25_doc_text(&chunk);
        self.bm25.write().await.upsert_document(&id, &bm25_text);

        self.chunks.write().await.insert(id, chunk);
        self.rebuild_symbol_graph().await;
        Ok(())
    }

    /// Parse a file with `chunk_ast`, store every chunk in the corpus, and
    /// retain the per-file entity list for later KG/entity-search phases.
    pub async fn index_file(&self, file_path: &str, content: &str) -> Result<()> {
        let (mut chunks, entities) = chunk_ast(file_path, content);

        // Issue #19: virtual_terms from entities so BM25 sees symbolic tokens
        // that don't appear literally in the chunk body.
        populate_virtual_terms(&mut chunks, &entities);

        // Snapshot chunk contents before move so we can run the ConceptCluster
        // pass below. Borrowing into the for-loop would hold the slice across
        // `await`, which `add_chunk` doesn't allow.
        let chunk_contents: Vec<String> = chunks.iter().map(|c| c.content.clone()).collect();

        for chunk in chunks {
            self.add_chunk(chunk).await?;
        }

        let all_entities = self
            .enrich_with_nlp_entities(file_path, content, &chunk_contents, entities)
            .await;

        self.entities
            .write()
            .await
            .insert(file_path.to_string(), all_entities);
        // `add_chunk` already rebuilds, but we also rebuild once more here so a
        // partial failure mid-file doesn't leave a stale graph; this is cheap.
        self.rebuild_symbol_graph().await;
        Ok(())
    }

    /// Run NER + ConceptCluster passes and merge their entities with the
    /// AST-derived base list.
    ///
    /// Why: keeps `index_file` focused on chunk persistence; isolates the two
    /// gated NLP passes (both no-ops when their respective preconditions
    /// aren't met) behind a single helper.
    /// What: extracts doc-comment NER entities, runs ConceptCluster when an
    /// embedder is wired, returns the combined entity list.
    /// Test: covered indirectly by every `index_file` integration test.
    async fn enrich_with_nlp_entities(
        &self,
        file_path: &str,
        content: &str,
        #[cfg_attr(not(feature = "clustering"), allow(unused_variables))]
        chunk_contents: &[String],
        base_entities: Vec<RawEntity>,
    ) -> Vec<RawEntity> {
        // Phase D: ONNX NER over doc comments (issue #23). Gated — no-op when
        // the model file is absent.
        let doc_text = crate::core::ner::extract_doc_comments(content);
        let ner_entities = self.ner.extract(&doc_text, file_path);
        if !ner_entities.is_empty() {
            tracing::debug!(
                "ner: {} NaturalLanguagePhrase entities for {}",
                ner_entities.len(),
                file_path
            );
        }

        let mut all_entities = base_entities;
        all_entities.extend(ner_entities);

        // Phase C: ConceptCluster entities (issue #22). Only runs when an
        // embedder is wired and the file has enough doc comments to cluster.
        // Feature-gated behind `clustering` (issue #108) to keep linfa/ndarray
        // out of default builds.
        #[cfg(feature = "clustering")]
        if let Some(embedder) = &self.embedder {
            let refs: Vec<&str> = chunk_contents.iter().map(|s| s.as_str()).collect();
            let cluster_entities = crate::core::concept_cluster::cluster_concepts_from_contents(
                &refs,
                embedder.as_ref(),
                file_path,
            )
            .await;
            if !cluster_entities.is_empty() {
                tracing::debug!(
                    "concept_cluster: {} ConceptCluster entities for {}",
                    cluster_entities.len(),
                    file_path
                );
                all_entities.extend(cluster_entities);
            }
        }

        all_entities
    }

    /// Bulk-index many files in one shot.
    ///
    /// Why: per-file `index_file` issues one ONNX `embed` call per chunk and
    /// rebuilds the symbol graph after every chunk. On a 13k-file Java
    /// monorepo that translates to ~80k serial ONNX calls and ~80k graph
    /// rebuilds — the dominant cost of a cold reindex.
    ///
    /// What:
    /// 1. Parse every file into chunks + entities in parallel via rayon.
    /// 2. Collect all chunk texts and embed them in batches of
    ///    [`EMBED_BATCH_SIZE`] — one ONNX call per batch instead of per chunk.
    /// 3. Upsert vectors + insert chunks under a single corpus write lock.
    /// 4. Rebuild the symbol graph **once** at the end.
    ///
    /// Returns the total number of chunks added across the batch. Files whose
    /// chunker returned no chunks contribute zero; per-file embed/upsert
    /// failures are surfaced as `Err` and abort the batch (the caller should
    /// fall back to per-file `index_file` for diagnostics).
    pub async fn index_files_batch(&self, files: &[(String, String)]) -> Result<usize> {
        self.index_files_batch_inner(files, false).await
    }

    /// Bulk-index variant that skips the trailing symbol graph rebuild.
    ///
    /// Why: a full reindex calls `index_files_batch` many times. Each call
    /// previously rebuilt the symbol graph (`O(N + E)` over the entire corpus
    /// with a per-edge suffix scan). On 14k files / 115k chunks that adds up
    /// to the dominant non-embedding cost. The reindex orchestrator now calls
    /// `index_files_batch_no_rebuild` per batch and rebuilds the graph **once**
    /// at the very end.
    ///
    /// Single-file paths (`add_chunk`, `index_file`, file watcher) keep the
    /// per-call rebuild for correctness — they're not in the bulk-cold-start
    /// hot path.
    pub async fn index_files_batch_no_rebuild(&self, files: &[(String, String)]) -> Result<usize> {
        self.index_files_batch_inner(files, true).await
    }

    /// Public hook for the bulk reindex orchestrator: rebuild the symbol graph
    /// once after a series of `index_files_batch_no_rebuild` calls.
    pub async fn rebuild_symbol_graph_now(&self) {
        self.rebuild_symbol_graph().await;
    }

    async fn index_files_batch_inner(
        &self,
        files: &[(String, String)],
        defer_graph_rebuild: bool,
    ) -> Result<usize> {
        if files.is_empty() {
            return Ok(0);
        }
        let parsed = self.parse_and_embed_files(files.to_vec()).await?;
        let timings = self
            .commit_parsed_batch(parsed, defer_graph_rebuild)
            .await?;
        Ok(timings.chunks)
    }

    /// Phase 1+2 of the bulk pipeline: parse files into chunks and embed them.
    ///
    /// Why: This phase does the heavy CPU/ONNX work but mutates **no shared
    /// state**. Lifting it out of the corpus write lock lets the reindex
    /// orchestrator overlap a batch's parse+embed with the previous batch's
    /// commit phase, and ensures concurrent search readers are never blocked
    /// by ONNX inference.
    /// What: parallel parse via rayon (with virtual_terms population from
    /// entities), then batched ONNX embed (`EMBED_BATCH_SIZE` chunks per
    /// `embed_batch` call). Returns a [`ParsedBatch`] ready for
    /// [`Self::commit_parsed_batch`].
    /// Test: covered indirectly by every `index_files_batch*` test.
    pub async fn parse_and_embed_files(&self, files: Vec<(String, String)>) -> Result<ParsedBatch> {
        if files.is_empty() {
            return Ok(ParsedBatch::default());
        }

        let parse_start = std::time::Instant::now();
        let parsed = Self::parse_files_parallel(files).await?;

        let mut all_chunks: Vec<RawChunk> = Vec::new();
        let mut entities_by_file: Vec<(String, Vec<RawEntity>)> = Vec::with_capacity(parsed.len());
        for (path, chunks, entities) in parsed {
            all_chunks.extend(chunks);
            entities_by_file.push((path, entities));
        }
        let parse_ms = parse_start.elapsed().as_millis() as u64;

        let embed_start = std::time::Instant::now();
        let embeddings = self.embed_chunks_in_batches(&all_chunks).await?;
        let embed_ms = embed_start.elapsed().as_millis() as u64;
        let vector_count = embeddings.iter().filter(|e| e.is_some()).count();

        Ok(ParsedBatch {
            chunks: all_chunks,
            embeddings,
            entities_by_file,
            parse_ms,
            embed_ms,
            vector_count,
        })
    }

    /// Parse every file in parallel via rayon and populate `virtual_terms`
    /// from the AST-derived entity list.
    ///
    /// Why: `chunk_ast` is sync + CPU-bound, so rayon's worker pool is a
    /// better fit than tokio tasks. Returning `(path, chunks, entities)`
    /// keeps file boundaries intact for downstream entity-map insertion.
    /// What: spawns a single blocking task that parallel-maps `chunk_ast`
    /// across every input, then populates virtual_terms per chunk.
    /// Test: covered indirectly by every `index_files_batch_*` test.
    async fn parse_files_parallel(
        files: Vec<(String, String)>,
    ) -> Result<Vec<(String, Vec<RawChunk>, Vec<RawEntity>)>> {
        use rayon::prelude::*;
        tokio::task::spawn_blocking(move || {
            files
                .par_iter()
                .map(|(path, content)| {
                    let (mut chunks, entities) = chunk_ast(path, content);
                    populate_virtual_terms(&mut chunks, &entities);
                    (path.clone(), chunks, entities)
                })
                .collect()
        })
        .await
        .context("batch parse task panicked")
    }

    /// Batched ONNX embed across every chunk's content.
    ///
    /// Why: per-chunk `embed` issues one ONNX call apiece; batching
    /// `EMBED_BATCH_SIZE` chunks per call amortizes session setup cost and
    /// caps the per-call tensor footprint (see `EMBED_BATCH_SIZE` doc for
    /// the macOS Jetsam history).
    /// What: returns `Vec<Option<Vec<f32>>>` aligned 1:1 with `chunks`,
    /// where `None` means "no embedder wired (BM25-only mode)". Fails
    /// fast if `embed_batch` returns a wrong-sized result.
    /// Test: covered indirectly by `test_index_files_batch_*`.
    async fn embed_chunks_in_batches(&self, chunks: &[RawChunk]) -> Result<Vec<Option<Vec<f32>>>> {
        let mut embeddings: Vec<Option<Vec<f32>>> = vec![None; chunks.len()];
        let (Some(embedder), Some(_store)) = (&self.embedder, &self.store) else {
            return Ok(embeddings);
        };
        let chunk_total = chunks.len();
        let batch_size = embed_batch_size();
        for batch_start in (0..chunk_total).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(chunk_total);
            let batch_texts: Vec<&str> = chunks[batch_start..batch_end]
                .iter()
                .map(|c| c.content.as_str())
                .collect();
            let batch_vecs = embedder
                .embed_batch(&batch_texts)
                .await
                .context("batch embed_batch failed")?;
            if batch_vecs.len() != batch_texts.len() {
                anyhow::bail!(
                    "embed_batch returned {} vectors, expected {}",
                    batch_vecs.len(),
                    batch_texts.len()
                );
            }
            for (offset, vec) in batch_vecs.into_iter().enumerate() {
                embeddings[batch_start + offset] = Some(vec);
            }
        }
        Ok(embeddings)
    }

    /// Phase 3+4 of the bulk pipeline: commit a [`ParsedBatch`] into the index.
    ///
    /// Why: this is the **only** phase that mutates shared state (BM25 index,
    /// corpus map, chunk_embeddings cache, HNSW store, entities map). By
    /// isolating it from the parse+embed work, the write-lock window shrinks
    /// from "minutes per batch" to "milliseconds per batch", letting concurrent
    /// searches and the next batch's parse+embed phase overlap freely.
    /// What: single-pass BM25 upsert, single-call HNSW `upsert_batch`, one
    /// corpus write lock for the whole batch, one entities write lock, then
    /// the (optional) graph rebuild.
    /// Test: covered indirectly by `test_index_files_batch_*`.
    pub async fn commit_parsed_batch(
        &self,
        parsed: ParsedBatch,
        defer_graph_rebuild: bool,
    ) -> Result<CommitTimings> {
        let ParsedBatch {
            chunks: mut all_chunks,
            mut embeddings,
            entities_by_file,
            parse_ms: _,
            embed_ms: _,
            vector_count: _,
        } = parsed;

        // Issue #N (180GB RSS fix): enforce the per-index chunk cap BEFORE
        // ingesting anything into BM25, HNSW, or the embedding cache.
        //
        // Why: previously `commit_corpus` was the only place that honoured the
        // cap. Chunks that were dropped from the corpus map still leaked into:
        //   - the HNSW vector store (via `commit_vectors_batch`)
        //   - the BM25 posting list (via `commit_bm25_batch`)
        //   - the chunk_embeddings LRU (via `commit_embeddings_cache`)
        // So on an over-cap repo, three structures grew unbounded while the
        // corpus map looked "capped". Pre-filtering here keeps every in-memory
        // structure consistent with the configured cap. Brand-new ids past the
        // cap are dropped; updates to existing ids are always allowed (they
        // don't grow the corpus).
        //
        // This is the structural fix for issue #82 — chunks dropped here never
        // allocate downstream, so RSS stays bounded by `TRUSTY_MAX_CHUNKS`.
        let cap = max_chunks_per_index();
        let pre_filter_dropped = {
            let corpus = self.chunks.read().await;
            let mut keep_mask: Vec<bool> = Vec::with_capacity(all_chunks.len());
            let mut new_count = corpus.len();
            let mut dropped = 0usize;
            for chunk in &all_chunks {
                let is_update = corpus.contains_key(&chunk.id);
                if is_update {
                    keep_mask.push(true);
                } else if new_count < cap {
                    new_count += 1;
                    keep_mask.push(true);
                } else {
                    dropped += 1;
                    keep_mask.push(false);
                }
            }
            drop(corpus);
            if dropped > 0 {
                // Rebuild chunks/embeddings in place, dropping over-cap entries
                // so they never reach the downstream structures.
                let mut kept_chunks: Vec<RawChunk> = Vec::with_capacity(all_chunks.len() - dropped);
                let mut kept_embeddings: Vec<Option<Vec<f32>>> =
                    Vec::with_capacity(all_chunks.len() - dropped);
                for ((chunk, vec_opt), keep) in all_chunks
                    .drain(..)
                    .zip(embeddings.drain(..))
                    .zip(keep_mask)
                {
                    if keep {
                        kept_chunks.push(chunk);
                        kept_embeddings.push(vec_opt);
                    }
                }
                all_chunks = kept_chunks;
                embeddings = kept_embeddings;
            }
            dropped
        };
        if pre_filter_dropped > 0 {
            tracing::warn!(
                "index '{}' chunk cap ({}) reached — pre-filtered {} chunks before commit \
                 (prevents leak into BM25/HNSW/embedding cache)",
                self.index_id,
                cap,
                pre_filter_dropped
            );
        }

        let chunk_total = all_chunks.len();
        if chunk_total == 0 {
            self.commit_entities(entities_by_file).await;
            return Ok(CommitTimings::default());
        }

        let vec_start = std::time::Instant::now();
        self.commit_vectors_batch(&all_chunks, &embeddings).await?;
        let vector_upsert_ms = vec_start.elapsed().as_millis() as u64;

        let bm25_start = std::time::Instant::now();
        self.commit_bm25_batch(&all_chunks).await;
        let bm25_ms = bm25_start.elapsed().as_millis() as u64;

        self.commit_embeddings_cache(&all_chunks, embeddings).await;
        self.commit_corpus(&mut all_chunks).await;
        self.commit_entities(entities_by_file).await;

        let kg_ms = if defer_graph_rebuild {
            0
        } else {
            let kg_start = std::time::Instant::now();
            self.rebuild_symbol_graph().await;
            kg_start.elapsed().as_millis() as u64
        };

        // Issue #85 — fire-and-forget incremental persistence. After every
        // committed batch we snapshot the HNSW graph + chunk corpus to disk
        // so a daemon crash mid-reindex preserves whatever was committed
        // (no progress is lost beyond the in-flight batch).
        //
        // Why background: `Index::save` can take 100s of ms on a large
        // corpus and we don't want the commit path (which is on the hot
        // reindex loop) to wait on filesystem I/O. We don't hold any locks
        // while spawning — the clones are cheap (Arc bumps + a path string).
        self.spawn_incremental_persist();

        Ok(CommitTimings {
            chunks: chunk_total,
            bm25_ms,
            vector_upsert_ms,
            kg_ms,
        })
    }

    /// Single batched HNSW upsert across all chunks that have an embedding.
    ///
    /// Why: drops 3N lock acquisitions to 3 for a batch of N chunks (key
    /// alloc, key rev-map, HNSW write).
    /// What: filters chunks without embeddings (BM25-only mode), delegates to
    /// `store.upsert_batch`. No-op when no store is wired or no embeddings
    /// were computed.
    /// Test: covered indirectly by `test_index_files_batch_*`.
    async fn commit_vectors_batch(
        &self,
        chunks: &[RawChunk],
        embeddings: &[Option<Vec<f32>>],
    ) -> Result<()> {
        let Some(store) = &self.store else {
            return Ok(());
        };
        let items: Vec<(String, Vec<f32>)> = chunks
            .iter()
            .zip(embeddings.iter())
            .filter_map(|(chunk, vec_opt)| vec_opt.as_ref().map(|v| (chunk.id.clone(), v.clone())))
            .collect();
        if items.is_empty() {
            return Ok(());
        }
        store
            .upsert_batch(&items)
            .await
            .context("batch upsert chunk vectors")
    }

    /// Upsert every chunk's BM25 document under a single write lock.
    ///
    /// Why: doing this **before** moving chunks into the corpus avoids a
    /// second clone of each chunk.
    /// What: holds the BM25 write lock once and walks `chunks` to upsert
    /// `body + virtual_terms` for each.
    /// Test: BM25 search correctness is covered by every search test.
    async fn commit_bm25_batch(&self, chunks: &[RawChunk]) {
        let mut bm25 = self.bm25.write().await;
        for chunk in chunks {
            let text = Self::bm25_doc_text(chunk);
            bm25.upsert_document(&chunk.id, &text);
        }
    }

    /// Cache per-chunk embeddings for MMR diversity (#28).
    ///
    /// Why: MMR needs vectors for already-ranked chunks without paying a
    /// re-embed or HNSW round-trip per candidate. Skip entirely when no
    /// embedder is wired (BM25-only mode).
    /// What: walks chunks and their (consumed) embeddings, inserts each
    /// `(id, vec)` pair under one write lock.
    /// Test: covered indirectly by `test_get_embedding_returns_some_after_indexing`.
    async fn commit_embeddings_cache(
        &self,
        chunks: &[RawChunk],
        embeddings: Vec<Option<Vec<f32>>>,
    ) {
        if self.embedder.is_none() {
            return;
        }
        let mut emb_cache = self.chunk_embeddings.write().await;
        for (chunk, vec_opt) in chunks.iter().zip(embeddings) {
            if let Some(vec) = vec_opt {
                // LRU `put` evicts the oldest entry when over capacity. Cache
                // eviction here is harmless: MMR rerank treats a missing entry
                // as zero diversity contribution.
                emb_cache.put(chunk.id.clone(), vec);
            }
        }
    }

    /// Drain `chunks` into the corpus under a single write lock.
    ///
    /// Why: single-lock insertion shrinks the write-lock window to
    /// milliseconds even for large batches.
    /// What: consumes `chunks` via `drain` so callers don't keep a stale
    /// copy after the corpus owns each one. Honours `max_chunks_per_index()`
    /// (issue #75): once the cap is reached new chunk ids are dropped (warned)
    /// while existing ids continue to be upserted.
    /// Test: covered indirectly by every search test.
    async fn commit_corpus(&self, chunks: &mut Vec<RawChunk>) {
        let cap = max_chunks_per_index();
        let mut corpus = self.chunks.write().await;
        let mut dropped = 0usize;
        for chunk in chunks.drain(..) {
            if !corpus.contains_key(&chunk.id) && corpus.len() >= cap {
                dropped += 1;
                continue;
            }
            corpus.insert(chunk.id.clone(), chunk);
        }
        if dropped > 0 {
            tracing::warn!(
                "index '{}' chunk cap ({}) reached — dropped {} new chunks in batch",
                self.index_id,
                cap,
                dropped
            );
        }
    }

    /// Insert each `(file_path, entities)` tuple into the per-file entity map.
    ///
    /// Why: factored so the early-return path (empty batch) and the main
    /// commit path share one implementation.
    /// What: holds the entities write lock once and inserts every tuple.
    /// Test: covered indirectly by `test_entity_exact_match_*`.
    async fn commit_entities(&self, entities_by_file: Vec<(String, Vec<RawEntity>)>) {
        let mut emap = self.entities.write().await;
        for (path, ents) in entities_by_file {
            emap.insert(path, ents);
        }
    }

    /// Number of chunks currently held in the corpus.
    pub fn chunk_count(&self) -> usize {
        // blocking_read is fine on a tokio worker thread for a quick stat probe;
        // we never await across this call.
        self.chunks.try_read().map(|g| g.len()).unwrap_or(0)
    }

    /// Snapshot the current symbol graph. Cheap (`Arc::clone`); intended for
    /// read-only KG queries from concurrent search handlers.
    pub async fn symbol_graph(&self) -> Arc<SymbolGraph> {
        Arc::clone(&*self.symbol_graph.read().await)
    }
}
