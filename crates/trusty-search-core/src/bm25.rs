//! BM25 index for lexical search.
//! Ported from open-mpm src/context/bm25.rs (zero external crate deps).
use std::collections::HashMap;

/// Three-pass tokenizer for code-aware BM25 (issue #27).
///
/// Pass 1 emits the raw token lowercased so an exact identifier match still
/// scores highest. Pass 2 splits camelCase / PascalCase identifiers so
/// `CodeIndexer` matches a query for `indexer`. Pass 3 splits at alpha↔digit
/// boundaries so `HTTP2Client` matches `http`, `2`, and `client`.
///
/// Outer split is on any non-alphanumeric character (including `_`) so
/// snake_case naturally falls out as separate tokens. Tokens are deduped and
/// sorted at the end so the inverted index sees a stable, unique-per-doc list.
pub fn tokenize(text: &str) -> Vec<String> {
    let mut tokens: Vec<String> = Vec::new();
    for raw in text.split(|c: char| !c.is_alphanumeric()) {
        if raw.is_empty() {
            continue;
        }

        // Pass 1: raw token lowercased.
        tokens.push(raw.to_lowercase());

        // Pass 2: camelCase / PascalCase split.
        let camel_parts = split_camel_case(raw);
        if camel_parts.len() > 1 {
            tokens.extend(camel_parts.iter().map(|s| s.to_lowercase()));
        }

        // Pass 3: alpha↔digit split.
        let digit_parts = split_on_digits(raw);
        if digit_parts.len() > 1 {
            tokens.extend(digit_parts.iter().map(|s| s.to_lowercase()));
        }
    }
    tokens.sort_unstable();
    tokens.dedup();
    tokens
}

/// Split an identifier at camelCase / PascalCase / acronym boundaries.
///
/// Boundaries:
/// - lowercase → uppercase ("codeIndexer" -> ["code", "Indexer"])
/// - uppercase run → uppercase + lowercase ("HTTPSClient" -> ["HTTPS", "Client"])
fn split_camel_case(s: &str) -> Vec<&str> {
    let bytes_len = s.len();
    let chars: Vec<(usize, char)> = s.char_indices().collect();
    if chars.len() < 2 {
        return vec![s];
    }
    let mut bounds: Vec<usize> = vec![0];
    for i in 1..chars.len() {
        let (idx, c) = chars[i];
        let (_, prev) = chars[i - 1];
        // lowercase/digit → uppercase
        let lower_to_upper = (prev.is_lowercase() || prev.is_ascii_digit()) && c.is_uppercase();
        // uppercase run → uppercase + lowercase: split before the trailing
        // uppercase that begins a new word (e.g. "HTTPSClient" → split at 'C').
        let acronym_to_word = prev.is_uppercase()
            && c.is_uppercase()
            && i + 1 < chars.len()
            && chars[i + 1].1.is_lowercase();
        if lower_to_upper || acronym_to_word {
            bounds.push(idx);
        }
    }
    bounds.push(bytes_len);
    bounds
        .windows(2)
        .map(|w| &s[w[0]..w[1]])
        .filter(|p| !p.is_empty())
        .collect()
}

/// Split at alpha↔digit transitions: "HTTP2" -> ["HTTP", "2"], "v3alpha" ->
/// ["v", "3", "alpha"].
fn split_on_digits(s: &str) -> Vec<&str> {
    let bytes_len = s.len();
    let chars: Vec<(usize, char)> = s.char_indices().collect();
    if chars.len() < 2 {
        return vec![s];
    }
    let mut bounds: Vec<usize> = vec![0];
    for i in 1..chars.len() {
        let (idx, c) = chars[i];
        let (_, prev) = chars[i - 1];
        let alpha_to_digit = prev.is_alphabetic() && c.is_ascii_digit();
        let digit_to_alpha = prev.is_ascii_digit() && c.is_alphabetic();
        if alpha_to_digit || digit_to_alpha {
            bounds.push(idx);
        }
    }
    bounds.push(bytes_len);
    bounds
        .windows(2)
        .map(|w| &s[w[0]..w[1]])
        .filter(|p| !p.is_empty())
        .collect()
}

/// Incremental BM25 index keyed by `chunk_id` (string).
///
/// Why: rebuilding the index over the entire corpus on every query is O(n) and
/// dominates p50 latency on large indexes (115k chunks → ~9.5s). We keep the
/// inverted lists hot and mutate them as chunks are added/removed.
///
/// What: each document is identified by an opaque `chunk_id`. Internally we
/// allocate a stable `usize` slot per chunk and reuse it on update; removed
/// slots are tracked in a free-list so the per-term `Vec` stays compact.
/// Per-document term frequencies are kept in `doc_terms` so `remove_document`
/// can decrement `doc_freqs` accurately without re-tokenizing.
///
/// `score_query_all` walks only the posting lists for the query terms, so a
/// k-term query touches `O(sum(df_i))` postings instead of N docs.
pub struct Bm25Index {
    k1: f32,
    b: f32,
    /// Per-term document frequency.
    doc_freqs: HashMap<String, usize>,
    /// Per-slot doc length (token count). `None` for free / removed slots.
    doc_lengths: Vec<Option<usize>>,
    /// Per-term posting list: `(slot, term_count_in_doc)`.
    inverted: HashMap<String, Vec<(usize, usize)>>,
    /// Cached avg doc length over live slots only.
    avg_doc_len: f32,
    /// chunk_id → slot. Used by upsert/remove to find an existing slot.
    id_to_slot: HashMap<String, usize>,
    /// slot → chunk_id. Used by `score_query_all` to materialize results.
    slot_to_id: Vec<Option<String>>,
    /// Free slots returned by `remove_document`, reused on next `add_document`.
    free_slots: Vec<usize>,
    /// Per-slot term list, retained so `remove_document` can update postings
    /// without re-tokenizing the original text.
    doc_terms: Vec<Option<Vec<String>>>,
    /// Number of live (non-free) slots.
    live_docs: usize,
}

impl Bm25Index {
    pub fn new() -> Self {
        Self {
            k1: 1.5,
            b: 0.75,
            doc_freqs: HashMap::new(),
            doc_lengths: Vec::new(),
            inverted: HashMap::new(),
            avg_doc_len: 0.0,
            id_to_slot: HashMap::new(),
            slot_to_id: Vec::new(),
            free_slots: Vec::new(),
            doc_terms: Vec::new(),
            live_docs: 0,
        }
    }

    /// Number of live documents currently indexed.
    pub fn len(&self) -> usize {
        self.live_docs
    }

    pub fn is_empty(&self) -> bool {
        self.live_docs == 0
    }

    /// Recompute average doc length over live slots only. O(slots).
    fn refresh_avg_doc_len(&mut self) {
        if self.live_docs == 0 {
            self.avg_doc_len = 0.0;
            return;
        }
        let total: usize = self.doc_lengths.iter().filter_map(|x| *x).sum();
        self.avg_doc_len = total as f32 / self.live_docs as f32;
    }

    /// Allocate a slot for a new chunk_id, reusing a freed slot when possible.
    fn allocate_slot(&mut self, chunk_id: &str) -> usize {
        if let Some(slot) = self.free_slots.pop() {
            self.slot_to_id[slot] = Some(chunk_id.to_string());
            self.doc_lengths[slot] = Some(0);
            self.doc_terms[slot] = Some(Vec::new());
            slot
        } else {
            let slot = self.slot_to_id.len();
            self.slot_to_id.push(Some(chunk_id.to_string()));
            self.doc_lengths.push(Some(0));
            self.doc_terms.push(Some(Vec::new()));
            slot
        }
    }

    /// Insert a chunk. If `chunk_id` already exists, the previous postings are
    /// removed first so updates are idempotent.
    pub fn upsert_document(&mut self, chunk_id: &str, text: &str) {
        if self.id_to_slot.contains_key(chunk_id) {
            self.remove_document(chunk_id);
        }
        let slot = self.allocate_slot(chunk_id);
        self.id_to_slot.insert(chunk_id.to_string(), slot);
        self.live_docs += 1;

        let tokens = tokenize(text);
        self.doc_lengths[slot] = Some(tokens.len());

        // Per-doc term counts.
        let mut term_counts: HashMap<&str, usize> = HashMap::new();
        for t in &tokens {
            *term_counts.entry(t.as_str()).or_default() += 1;
        }
        for (term, count) in term_counts {
            *self.doc_freqs.entry(term.to_string()).or_default() += 1;
            self.inverted
                .entry(term.to_string())
                .or_default()
                .push((slot, count));
        }
        self.doc_terms[slot] = Some(tokens);
        self.refresh_avg_doc_len();
    }

    /// Legacy slot-based add. Retained so the in-tree `score(query, doc_id)`
    /// API keeps working for existing tests; new callers should prefer
    /// [`upsert_document`].
    pub fn add_document(&mut self, doc_id: usize, text: &str) {
        // Map the slot-style doc_id onto a synthetic chunk_id that won't
        // collide with real ids. This keeps the legacy test surface intact
        // while the production path uses `upsert_document` exclusively.
        let synthetic = format!("__legacy:{doc_id}");
        self.upsert_document(&synthetic, text);
    }

    /// Remove a chunk by id. No-op when the id isn't present.
    pub fn remove_document(&mut self, chunk_id: &str) {
        let Some(slot) = self.id_to_slot.remove(chunk_id) else {
            return;
        };
        // Decrement doc_freqs and prune postings using the cached term list.
        if let Some(terms) = self.doc_terms[slot].take() {
            // Unique terms only — doc_freqs counts documents, not occurrences.
            let mut unique = terms.clone();
            unique.sort_unstable();
            unique.dedup();
            for term in &unique {
                if let Some(df) = self.doc_freqs.get_mut(term) {
                    *df = df.saturating_sub(1);
                    if *df == 0 {
                        self.doc_freqs.remove(term);
                    }
                }
                if let Some(postings) = self.inverted.get_mut(term) {
                    postings.retain(|(s, _)| *s != slot);
                    if postings.is_empty() {
                        self.inverted.remove(term);
                    }
                }
            }
        }
        self.doc_lengths[slot] = None;
        self.slot_to_id[slot] = None;
        self.free_slots.push(slot);
        self.live_docs = self.live_docs.saturating_sub(1);
        self.refresh_avg_doc_len();
    }

    /// Score every document that contains at least one query term, returning
    /// `(chunk_id, score)` pairs sorted by score descending. Up to `top_k`
    /// pairs are returned. Documents with score zero are filtered out.
    ///
    /// This is the production search entry point. Cost is `O(sum(df_i))`
    /// over query terms — independent of total corpus size.
    pub fn score_query_all(&self, query: &str, top_k: usize) -> Vec<(String, f32)> {
        if self.live_docs == 0 || top_k == 0 {
            return Vec::new();
        }
        let n = self.live_docs as f32;
        let avg = self.avg_doc_len.max(1.0);

        // Accumulate score per slot. HashMap is fine — the touched-slot set is
        // bounded by the union of postings for the query terms, not by N.
        let mut acc: HashMap<usize, f32> = HashMap::new();
        for term in tokenize(query) {
            let df = match self.doc_freqs.get(&term) {
                Some(d) if *d > 0 => *d as f32,
                _ => continue,
            };
            let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
            let Some(postings) = self.inverted.get(&term) else {
                continue;
            };
            for (slot, count) in postings {
                let dl = match self.doc_lengths.get(*slot).and_then(|x| *x) {
                    Some(l) => l as f32,
                    None => continue,
                };
                let tf = *count as f32;
                let tf_norm = tf * (self.k1 + 1.0)
                    / (tf + self.k1 * (1.0 - self.b + self.b * dl / avg));
                *acc.entry(*slot).or_insert(0.0) += idf * tf_norm;
            }
        }

        let mut scored: Vec<(String, f32)> = acc
            .into_iter()
            .filter(|(_, s)| *s > 0.0)
            .filter_map(|(slot, score)| {
                self.slot_to_id
                    .get(slot)
                    .and_then(|o| o.clone())
                    .map(|id| (id, score))
            })
            .collect();
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        scored.truncate(top_k);
        scored
    }

    pub fn score(&self, query: &str, doc_id: usize) -> f32 {
        let n = self.live_docs as f32;
        let dl = match self.doc_lengths.get(doc_id).and_then(|x| *x) {
            Some(l) => l as f32,
            None => return 0.0,
        };
        let mut score = 0.0f32;

        for term in tokenize(query) {
            let df = *self.doc_freqs.get(&term).unwrap_or(&0) as f32;
            if df == 0.0 { continue; }
            let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();

            let tf = self.inverted.get(&term)
                .and_then(|v| v.iter().find(|(id, _)| *id == doc_id))
                .map(|(_, c)| *c as f32)
                .unwrap_or(0.0);

            let tf_norm = tf * (self.k1 + 1.0)
                / (tf + self.k1 * (1.0 - self.b + self.b * dl / self.avg_doc_len.max(1.0)));

            score += idf * tf_norm;
        }
        score
    }
}

impl Default for Bm25Index {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bm25_scores_relevant_doc_higher() {
        let mut idx = Bm25Index::new();
        idx.add_document(0, "authentication login password secure");
        idx.add_document(1, "rendering ui components svelte");
        let s0 = idx.score("authentication", 0);
        let s1 = idx.score("authentication", 1);
        assert!(s0 > s1, "relevant doc should score higher: {s0} vs {s1}");
    }

    #[test]
    fn test_tokenize_splits_code() {
        let tokens = tokenize("fn search_hybrid(query: &str) -> Vec<Hit>");
        // snake_case parts split via outer non-alphanumeric split.
        assert!(tokens.contains(&"search".to_string()));
        assert!(tokens.contains(&"hybrid".to_string()));
        assert!(tokens.contains(&"query".to_string()));
    }

    #[test]
    fn test_tokenize_camel_case_pascal() {
        let tokens = tokenize("CodeIndexer");
        assert!(tokens.contains(&"code".to_string()), "got {tokens:?}");
        assert!(tokens.contains(&"indexer".to_string()), "got {tokens:?}");
        assert!(tokens.contains(&"codeindexer".to_string()), "got {tokens:?}");
    }

    #[test]
    fn test_tokenize_pascal_two_words() {
        let tokens = tokenize("UsearchStore");
        assert!(tokens.contains(&"usearch".to_string()), "got {tokens:?}");
        assert!(tokens.contains(&"store".to_string()), "got {tokens:?}");
    }

    #[test]
    fn test_tokenize_snake_case() {
        let tokens = tokenize("use_kg_first");
        assert!(tokens.contains(&"use".to_string()), "got {tokens:?}");
        assert!(tokens.contains(&"kg".to_string()), "got {tokens:?}");
        assert!(tokens.contains(&"first".to_string()), "got {tokens:?}");
    }

    #[test]
    fn test_tokenize_alpha_digit_split() {
        let tokens = tokenize("HTTP2Client");
        assert!(tokens.contains(&"http".to_string()), "got {tokens:?}");
        assert!(tokens.contains(&"2".to_string()), "got {tokens:?}");
        assert!(tokens.contains(&"client".to_string()), "got {tokens:?}");
    }

    #[test]
    fn test_tokenize_acronym_then_word() {
        // Pass 2 boundary: "HTTPSClient" → ["HTTPS", "Client"]
        let tokens = tokenize("HTTPSClient");
        assert!(tokens.contains(&"https".to_string()), "got {tokens:?}");
        assert!(tokens.contains(&"client".to_string()), "got {tokens:?}");
    }

    #[test]
    fn test_bm25_incremental_upsert_and_remove() {
        // Bug B regression: BM25 must support incremental upsert/remove so the
        // search hot path doesn't have to rebuild the corpus on every query.
        let mut idx = Bm25Index::new();
        idx.upsert_document("a", "authentication login password");
        idx.upsert_document("b", "rendering ui components svelte");
        idx.upsert_document("c", "database connection pool postgres");
        assert_eq!(idx.len(), 3);

        let hits = idx.score_query_all("authentication", 10);
        assert!(hits.iter().any(|(id, _)| id == "a"));
        assert!(!hits.iter().any(|(id, _)| id == "b"));

        // Removing a doc must drop it from results AND keep the rest scoring.
        idx.remove_document("a");
        assert_eq!(idx.len(), 2);
        let hits_after = idx.score_query_all("authentication", 10);
        assert!(!hits_after.iter().any(|(id, _)| id == "a"));
        let svelte_hits = idx.score_query_all("svelte", 10);
        assert!(svelte_hits.iter().any(|(id, _)| id == "b"));
    }

    #[test]
    fn test_bm25_upsert_replaces_existing_doc() {
        // Re-upserting an existing chunk_id must not double-count terms.
        let mut idx = Bm25Index::new();
        idx.upsert_document("a", "alpha beta gamma");
        idx.upsert_document("a", "delta epsilon");
        assert_eq!(idx.len(), 1);
        // "alpha" was in the first version only — must be gone.
        assert!(idx.score_query_all("alpha", 10).is_empty());
        assert!(!idx.score_query_all("delta", 10).is_empty());
    }

    #[test]
    fn test_score_query_all_returns_sorted_unique_results() {
        let mut idx = Bm25Index::new();
        idx.upsert_document("a", "search rust async tokio");
        idx.upsert_document("b", "search rust");
        idx.upsert_document("c", "unrelated content");
        let hits = idx.score_query_all("rust async", 10);
        // Must be sorted by score desc.
        for w in hits.windows(2) {
            assert!(w[0].1 >= w[1].1, "results must be sorted desc: {hits:?}");
        }
        // No duplicates.
        let mut ids: Vec<&str> = hits.iter().map(|(id, _)| id.as_str()).collect();
        ids.sort();
        let unique = ids.len();
        ids.dedup();
        assert_eq!(unique, ids.len());
    }

    #[test]
    fn test_tokenize_dedups_and_sorts() {
        let tokens = tokenize("foo foo bar");
        let foos: Vec<&String> = tokens.iter().filter(|t| t.as_str() == "foo").collect();
        assert_eq!(foos.len(), 1, "duplicates must collapse: {tokens:?}");
        let mut sorted = tokens.clone();
        sorted.sort();
        assert_eq!(tokens, sorted, "tokens must be sorted: {tokens:?}");
    }
}
