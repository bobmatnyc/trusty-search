use regex::Regex;
use std::sync::OnceLock;

#[derive(Debug, Clone, PartialEq)]
pub enum QueryIntent {
    Definition,   // BM25-heavy: alpha=0.3, beta=0.7
    Usage,        // KG-first: alpha=0.5, beta=0.5, use_kg_first=true
    Conceptual,   // vector-heavy: alpha=0.8, beta=0.2
    BugDebt,      // BM25-only: alpha=0.1, beta=0.9
    Unknown,      // balanced: alpha=0.6, beta=0.4
}

impl QueryIntent {
    pub fn weights(&self) -> (f32, f32, bool) {
        // returns (alpha_vector, beta_bm25, use_kg_first)
        match self {
            QueryIntent::Definition  => (0.3, 0.7, false),
            QueryIntent::Usage       => (0.5, 0.5, true),
            QueryIntent::Conceptual  => (0.8, 0.2, false),
            QueryIntent::BugDebt     => (0.1, 0.9, false),
            QueryIntent::Unknown     => (0.6, 0.4, false),
        }
    }
}

pub struct QueryClassifier;

static DEFINITION_RE: OnceLock<Regex> = OnceLock::new();
static USAGE_RE: OnceLock<Regex> = OnceLock::new();
static CONCEPTUAL_RE: OnceLock<Regex> = OnceLock::new();
static BUG_DEBT_RE: OnceLock<Regex> = OnceLock::new();
// Entity-relationship keyword patterns (issue #21). Matched alongside the
// existing intent regexes; a hit overrides the default to bias the query
// toward the routing best-suited for that entity relationship.
static ENTITY_DEF_RE: OnceLock<Regex> = OnceLock::new();
static ENTITY_USAGE_RE: OnceLock<Regex> = OnceLock::new();
static ENTITY_BUG_RE: OnceLock<Regex> = OnceLock::new();

impl QueryClassifier {
    pub fn classify(query: &str) -> QueryIntent {
        let def_re = DEFINITION_RE.get_or_init(|| {
            Regex::new(r"(?i)\b(fn |struct |impl |trait |enum |type |def |class |function |define)\b").unwrap()
        });
        let usage_re = USAGE_RE.get_or_init(|| {
            Regex::new(r"(?i)\b(where is|callers of|who calls|uses of|usages|called by)\b").unwrap()
        });
        let conceptual_re = CONCEPTUAL_RE.get_or_init(|| {
            Regex::new(r"(?i)\b(how does|what is|explain|overview|architecture|design|why)\b").unwrap()
        });
        let bug_re = BUG_DEBT_RE.get_or_init(|| {
            Regex::new(r"(?i)\b(TODO|FIXME|HACK|panic!|unwrap\(\)|bug|error|crash|fail)\b").unwrap()
        });
        // Entity-relationship keyword regexes (issue #21).
        let entity_def_re = ENTITY_DEF_RE.get_or_init(|| {
            Regex::new(r"(?i)\b(implements|derives from|aliased as)\b").unwrap()
        });
        let entity_usage_re = ENTITY_USAGE_RE.get_or_init(|| {
            Regex::new(r"(?i)\b(tested by|co-occurs)\b").unwrap()
        });
        let entity_bug_re = ENTITY_BUG_RE.get_or_init(|| {
            Regex::new(r"(?i)\b(raises|documented by)\b").unwrap()
        });

        // Entity-keyword hits take precedence over the generic patterns when
        // the user explicitly asks an entity-relationship question.
        if entity_usage_re.is_match(query) { return QueryIntent::Usage; }
        if entity_def_re.is_match(query) { return QueryIntent::Definition; }
        if entity_bug_re.is_match(query) { return QueryIntent::BugDebt; }

        if usage_re.is_match(query) { return QueryIntent::Usage; }
        if def_re.is_match(query) { return QueryIntent::Definition; }
        if conceptual_re.is_match(query) { return QueryIntent::Conceptual; }
        if bug_re.is_match(query) { return QueryIntent::BugDebt; }
        QueryIntent::Unknown
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_definition_intent() {
        assert_eq!(QueryClassifier::classify("fn search_hybrid"), QueryIntent::Definition);
        assert_eq!(QueryClassifier::classify("struct CodeIndexer"), QueryIntent::Definition);
    }

    #[test]
    fn test_usage_intent() {
        assert_eq!(QueryClassifier::classify("callers of search_hybrid"), QueryIntent::Usage);
        assert_eq!(QueryClassifier::classify("where is CodeIndexer used"), QueryIntent::Usage);
    }

    #[test]
    fn test_conceptual_intent() {
        assert_eq!(QueryClassifier::classify("how does the search work"), QueryIntent::Conceptual);
        assert_eq!(QueryClassifier::classify("what is BM25"), QueryIntent::Conceptual);
    }

    #[test]
    fn test_bug_debt_intent() {
        assert_eq!(QueryClassifier::classify("TODO items in search"), QueryIntent::BugDebt);
        assert_eq!(QueryClassifier::classify("FIXME authentication"), QueryIntent::BugDebt);
    }

    #[test]
    fn test_usage_beats_definition() {
        assert_eq!(QueryClassifier::classify("callers of fn search_hybrid"), QueryIntent::Usage);
    }

    #[test]
    fn test_entity_implements_is_definition() {
        assert_eq!(
            QueryClassifier::classify("which types implements Embedder"),
            QueryIntent::Definition
        );
    }

    #[test]
    fn test_entity_derives_from_is_definition() {
        assert_eq!(
            QueryClassifier::classify("structs that derives from Default"),
            QueryIntent::Definition
        );
    }

    #[test]
    fn test_entity_aliased_as_is_definition() {
        assert_eq!(
            QueryClassifier::classify("Result aliased as anyhow::Result"),
            QueryIntent::Definition
        );
    }

    #[test]
    fn test_entity_tested_by_is_usage() {
        assert_eq!(
            QueryClassifier::classify("authenticate tested by login_test"),
            QueryIntent::Usage
        );
    }

    #[test]
    fn test_entity_co_occurs_is_usage() {
        assert_eq!(
            QueryClassifier::classify("symbols that co-occurs in test fixtures"),
            QueryIntent::Usage
        );
    }

    #[test]
    fn test_entity_raises_is_bug_debt() {
        assert_eq!(
            QueryClassifier::classify("functions that raises ConfigError"),
            QueryIntent::BugDebt
        );
    }

    #[test]
    fn test_entity_documented_by_is_bug_debt() {
        assert_eq!(
            QueryClassifier::classify("ParseError documented by docs/errors.md"),
            QueryIntent::BugDebt
        );
    }
}
