//! BM25 index for lexical search.
//! Ported from open-mpm src/context/bm25.rs (zero external crate deps).
use std::collections::HashMap;

pub fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|t| t.len() > 2)
        .map(|t| t.to_lowercase())
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
        assert!(tokens.contains(&"search_hybrid".to_string()));
        assert!(tokens.contains(&"query".to_string()));
    }
}
