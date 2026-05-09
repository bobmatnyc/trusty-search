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

pub struct Bm25Index {
    k1: f32,
    b: f32,
    doc_freqs: HashMap<String, usize>,
    doc_lengths: Vec<usize>,
    inverted: HashMap<String, Vec<(usize, usize)>>,
    avg_doc_len: f32,
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
        }
    }

    pub fn add_document(&mut self, doc_id: usize, text: &str) {
        let tokens = tokenize(text);
        let len = tokens.len();

        // ensure doc_lengths is large enough
        if self.doc_lengths.len() <= doc_id {
            self.doc_lengths.resize(doc_id + 1, 0);
        }
        self.doc_lengths[doc_id] = len;

        let mut term_counts: HashMap<&str, usize> = HashMap::new();
        for t in &tokens {
            *term_counts.entry(t.as_str()).or_default() += 1;
        }
        for (term, count) in term_counts {
            *self.doc_freqs.entry(term.to_string()).or_default() += 1;
            self.inverted.entry(term.to_string()).or_default().push((doc_id, count));
        }

        let n = self.doc_lengths.len() as f32;
        let total: usize = self.doc_lengths.iter().sum();
        self.avg_doc_len = total as f32 / n.max(1.0);
    }

    pub fn score(&self, query: &str, doc_id: usize) -> f32 {
        let n = self.doc_lengths.len() as f32;
        let dl = *self.doc_lengths.get(doc_id).unwrap_or(&0) as f32;
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
    fn test_tokenize_dedups_and_sorts() {
        let tokens = tokenize("foo foo bar");
        let foos: Vec<&String> = tokens.iter().filter(|t| t.as_str() == "foo").collect();
        assert_eq!(foos.len(), 1, "duplicates must collapse: {tokens:?}");
        let mut sorted = tokens.clone();
        sorted.sort();
        assert_eq!(tokens, sorted, "tokens must be sorted: {tokens:?}");
    }
}
