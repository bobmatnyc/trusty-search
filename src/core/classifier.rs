use regex::Regex;
use std::sync::OnceLock;

#[derive(Debug, Clone, PartialEq)]
pub enum QueryIntent {
    Definition, // BM25-heavy: alpha=0.3, beta=0.7
    Usage,      // KG-first: alpha=0.5, beta=0.5, use_kg_first=true
    Conceptual, // vector-heavy: alpha=0.8, beta=0.2
    BugDebt,    // BM25-only: alpha=0.1, beta=0.9
    Unknown,    // balanced: alpha=0.6, beta=0.4
}

impl QueryIntent {
    pub fn weights(&self) -> (f32, f32, bool) {
        // returns (alpha_vector, beta_bm25, use_kg_first)
        match self {
            QueryIntent::Definition => (0.3, 0.7, false),
            QueryIntent::Usage => (0.5, 0.5, true),
            QueryIntent::Conceptual => (0.8, 0.2, false),
            QueryIntent::BugDebt => (0.1, 0.9, false),
            QueryIntent::Unknown => (0.6, 0.4, false),
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
// Domain-term definition patterns (issue #88): PascalCase identifiers followed
// by structural keywords, and standalone structural vocabulary words.
static DOMAIN_DEF_RE: OnceLock<Regex> = OnceLock::new();
// Extended bug/debt vocabulary (issue #88).
static EXTENDED_BUG_RE: OnceLock<Regex> = OnceLock::new();
// Long natural-language conceptual queries (issue #88): ≥6 words, no code tokens.
static LONG_NL_RE: OnceLock<Regex> = OnceLock::new();

impl QueryClassifier {
    /// Classify a query string into a `QueryIntent` for routing weight selection.
    ///
    /// Why: different query shapes benefit from different BM25/vector balance;
    /// classifying up-front lets the search pipeline pick optimal weights without
    /// per-result heuristics.
    /// What: applies a priority-ordered chain of regex patterns; the first match
    /// wins. Entity-relationship keywords (issue #21) and domain-term definitions
    /// (issue #88) are checked before the generic structural patterns.
    /// Test: see the `#[cfg(test)]` module below for representative examples per intent.
    pub fn classify(query: &str) -> QueryIntent {
        let def_re = DEFINITION_RE.get_or_init(|| {
            Regex::new(
                r"(?i)\b(fn |struct |impl |trait |enum |type |def |class |function |define)\b",
            )
            .unwrap()
        });
        let usage_re = USAGE_RE.get_or_init(|| {
            Regex::new(r"(?i)\b(where is|callers of|who calls|uses of|usages|called by)\b").unwrap()
        });
        let conceptual_re = CONCEPTUAL_RE.get_or_init(|| {
            Regex::new(r"(?i)\b(how does|what is|explain|overview|architecture|design|why)\b")
                .unwrap()
        });
        let bug_re = BUG_DEBT_RE.get_or_init(|| {
            Regex::new(r"(?i)\b(TODO|FIXME|HACK|panic!|unwrap\(\)|bug|error|crash|fail)\b").unwrap()
        });
        // Entity-relationship keyword regexes (issue #21).
        let entity_def_re = ENTITY_DEF_RE
            .get_or_init(|| Regex::new(r"(?i)\b(implements|derives from|aliased as)\b").unwrap());
        let entity_usage_re =
            ENTITY_USAGE_RE.get_or_init(|| Regex::new(r"(?i)\b(tested by|co-occurs)\b").unwrap());
        let entity_bug_re =
            ENTITY_BUG_RE.get_or_init(|| Regex::new(r"(?i)\b(raises|documented by)\b").unwrap());

        // Domain-term definition patterns (issue #88).
        //
        // Pattern A: a standalone structural/schema keyword used as a noun
        //   → "definition", "interface", "schema", "enum", "model" as a whole word.
        // Pattern B: a PascalCase identifier immediately followed by a structural
        //   keyword → "RoomType definition", "Hotel struct", "UserRole enum".
        let domain_def_re = DOMAIN_DEF_RE.get_or_init(|| {
            Regex::new(
                r"(?x)
                # Pattern A — standalone structural vocabulary word
                (?i)\b(definition|interface|schema|model)\b
                |
                # Pattern B — PascalCase identifier + structural keyword
                \b[A-Z][a-zA-Z0-9]+\s+(?i)(definition|struct|class|interface|type|schema|enum|trait|model)\b
                ",
            )
            .unwrap()
        });

        // Extended bug/debt vocabulary (issue #88).
        let extended_bug_re = EXTENDED_BUG_RE.get_or_init(|| {
            Regex::new(
                r"(?i)\b(error\s+handling|deprecated|legacy|missing\s+validation|hardcoded)\b",
            )
            .unwrap()
        });

        // Long natural-language conceptual query (issue #88): ≥6 whitespace-
        // separated tokens with no code punctuation (`(`, `:`, `_`, `.`).
        let long_nl_re = LONG_NL_RE.get_or_init(|| {
            // A "code token" is any character in: ( ) : _ .
            // We match queries that have ≥6 word characters separated by spaces and
            // contain none of the above punctuation.
            Regex::new(r"^[^():_.]+(?:\s+[^():_.]+){5,}$").unwrap()
        });

        // Priority chain — most-specific patterns first.

        // Entity-keyword hits take precedence over generic patterns (issue #21).
        if entity_usage_re.is_match(query) {
            return QueryIntent::Usage;
        }
        if entity_def_re.is_match(query) {
            return QueryIntent::Definition;
        }
        if entity_bug_re.is_match(query) {
            return QueryIntent::BugDebt;
        }

        // Domain-term definition patterns (issue #88) evaluated before the
        // generic `def_re` so "enum" / "interface" etc. are not missed.
        if domain_def_re.is_match(query) {
            return QueryIntent::Definition;
        }

        // Extended bug/debt vocabulary (issue #88).
        if extended_bug_re.is_match(query) {
            return QueryIntent::BugDebt;
        }

        if usage_re.is_match(query) {
            return QueryIntent::Usage;
        }
        if def_re.is_match(query) {
            return QueryIntent::Definition;
        }
        if conceptual_re.is_match(query) {
            return QueryIntent::Conceptual;
        }
        if bug_re.is_match(query) {
            return QueryIntent::BugDebt;
        }

        // Long natural-language query with no code tokens → Conceptual (issue #88).
        // Checked after all keyword patterns so that long queries containing
        // code tokens still fall through to the generic patterns.
        let trimmed = query.trim();
        if long_nl_re.is_match(trimmed) {
            return QueryIntent::Conceptual;
        }

        QueryIntent::Unknown
    }

    /// Classify with per-repo domain vocabulary.
    ///
    /// Why: repos define jargon (e.g. "PMS", "rate strategy", "RoomType") that
    /// the generic regex chain can't recognise. When `classify` returns
    /// `Unknown` and the query mentions a domain term, we nudge the result to
    /// `Definition` (the safe default — pulls the entity's defining chunk in
    /// via BM25 weight and lets the user iterate). All other intents survive
    /// untouched so explicit signals (`fn`, `callers of`, `TODO`) keep their
    /// routing.
    /// What: runs `classify`, then upgrades `Unknown` to `Definition` if any
    /// non-empty `domain_terms` entry appears case-insensitively as a substring
    /// of the query.
    /// Test: `test_domain_term_upgrades_unknown_to_definition`,
    /// `test_domain_term_does_not_override_explicit_intent`.
    pub fn classify_with_domain(query: &str, domain_terms: &[String]) -> QueryIntent {
        let base = Self::classify(query);
        if base != QueryIntent::Unknown {
            return base;
        }
        if domain_terms.is_empty() {
            return base;
        }
        let q = query.to_lowercase();
        for term in domain_terms {
            let t = term.trim();
            if t.is_empty() {
                continue;
            }
            if q.contains(&t.to_lowercase()) {
                return QueryIntent::Definition;
            }
        }
        base
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_definition_intent() {
        assert_eq!(
            QueryClassifier::classify("fn search_hybrid"),
            QueryIntent::Definition
        );
        assert_eq!(
            QueryClassifier::classify("struct CodeIndexer"),
            QueryIntent::Definition
        );
    }

    #[test]
    fn test_usage_intent() {
        assert_eq!(
            QueryClassifier::classify("callers of search_hybrid"),
            QueryIntent::Usage
        );
        assert_eq!(
            QueryClassifier::classify("where is CodeIndexer used"),
            QueryIntent::Usage
        );
    }

    #[test]
    fn test_conceptual_intent() {
        assert_eq!(
            QueryClassifier::classify("how does the search work"),
            QueryIntent::Conceptual
        );
        assert_eq!(
            QueryClassifier::classify("what is BM25"),
            QueryIntent::Conceptual
        );
    }

    #[test]
    fn test_bug_debt_intent() {
        assert_eq!(
            QueryClassifier::classify("TODO items in search"),
            QueryIntent::BugDebt
        );
        assert_eq!(
            QueryClassifier::classify("FIXME authentication"),
            QueryIntent::BugDebt
        );
    }

    #[test]
    fn test_usage_beats_definition() {
        assert_eq!(
            QueryClassifier::classify("callers of fn search_hybrid"),
            QueryIntent::Usage
        );
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

    // ── Domain-term definition tests (issue #88) ────────────────────────────

    #[test]
    fn test_domain_word_definition_is_definition() {
        assert_eq!(
            QueryClassifier::classify("RoomType definition"),
            QueryIntent::Definition
        );
    }

    #[test]
    fn test_domain_pascal_struct_is_definition() {
        assert_eq!(
            QueryClassifier::classify("Hotel struct"),
            QueryIntent::Definition
        );
    }

    #[test]
    fn test_domain_pascal_interface_is_definition() {
        assert_eq!(
            QueryClassifier::classify("BookingRepository interface"),
            QueryIntent::Definition
        );
    }

    #[test]
    fn test_domain_pascal_enum_is_definition() {
        assert_eq!(
            QueryClassifier::classify("UserRole enum"),
            QueryIntent::Definition
        );
    }

    #[test]
    fn test_standalone_schema_is_definition() {
        assert_eq!(
            QueryClassifier::classify("schema for reservations"),
            QueryIntent::Definition
        );
    }

    #[test]
    fn test_standalone_interface_is_definition() {
        assert_eq!(
            QueryClassifier::classify("interface for payment processor"),
            QueryIntent::Definition
        );
    }

    #[test]
    fn test_standalone_model_is_definition() {
        assert_eq!(
            QueryClassifier::classify("model for user accounts"),
            QueryIntent::Definition
        );
    }

    #[test]
    fn test_domain_pascal_type_is_definition() {
        assert_eq!(
            QueryClassifier::classify("ReservationStatus type"),
            QueryIntent::Definition
        );
    }

    // ── Extended BugDebt tests (issue #88) ──────────────────────────────────

    #[test]
    fn test_error_handling_is_bug_debt() {
        assert_eq!(
            QueryClassifier::classify("error handling in payment flow"),
            QueryIntent::BugDebt
        );
    }

    #[test]
    fn test_deprecated_is_bug_debt() {
        assert_eq!(
            QueryClassifier::classify("deprecated authentication methods"),
            QueryIntent::BugDebt
        );
    }

    #[test]
    fn test_legacy_is_bug_debt() {
        assert_eq!(
            QueryClassifier::classify("legacy session management code"),
            QueryIntent::BugDebt
        );
    }

    #[test]
    fn test_missing_validation_is_bug_debt() {
        assert_eq!(
            QueryClassifier::classify("missing validation in user input"),
            QueryIntent::BugDebt
        );
    }

    #[test]
    fn test_hardcoded_is_bug_debt() {
        assert_eq!(
            QueryClassifier::classify("hardcoded connection strings"),
            QueryIntent::BugDebt
        );
    }

    // ── Long natural-language conceptual tests (issue #88) ──────────────────

    #[test]
    fn test_long_nl_query_is_conceptual() {
        assert_eq!(
            QueryClassifier::classify("how the reservation system handles overbooking scenarios"),
            QueryIntent::Conceptual
        );
    }

    #[test]
    fn test_long_nl_query_without_code_tokens_is_conceptual() {
        assert_eq!(
            QueryClassifier::classify("what happens when a payment method expires"),
            QueryIntent::Conceptual
        );
    }

    #[test]
    fn test_long_query_with_code_token_not_long_nl_conceptual() {
        // Contains `_` so the long-NL path should NOT fire. The query may still
        // be classified Conceptual via "how does" in the generic pattern — that
        // is correct behaviour. This test verifies that the long-NL path requires
        // no code tokens by using a query that has no other conceptual keywords.
        let result =
            QueryClassifier::classify("the payment_processor retries failed attempts five times");
        // Contains `_` → long-NL rule is suppressed; no other keyword match → Unknown
        assert_eq!(result, QueryIntent::Unknown);
    }

    // ── Domain-term routing tests (trusty-search.yaml config) ──────────────

    #[test]
    fn test_domain_term_upgrades_unknown_to_definition() {
        // Bare "PMS integration" — no generic pattern matches → Unknown.
        // With domain_terms containing "PMS", we upgrade to Definition.
        let terms = vec!["PMS".to_string()];
        assert_eq!(
            QueryClassifier::classify("PMS integration"),
            QueryIntent::Unknown
        );
        assert_eq!(
            QueryClassifier::classify_with_domain("PMS integration", &terms),
            QueryIntent::Definition
        );
    }

    #[test]
    fn test_domain_term_case_insensitive() {
        let terms = vec!["RateStrategy".to_string()];
        // Conceptual keyword "what is" wins regardless of domain match.
        assert_eq!(
            QueryClassifier::classify_with_domain("what is the ratestrategy applies", &terms),
            QueryIntent::Conceptual
        );
        // No conceptual keyword → falls through Unknown → upgraded.
        // Lowercase usage of a PascalCase domain term still matches.
        assert_eq!(
            QueryClassifier::classify_with_domain("ratestrategy fields", &terms),
            QueryIntent::Definition
        );
    }

    #[test]
    fn test_domain_term_does_not_override_explicit_intent() {
        let terms = vec!["PMS".to_string()];
        // Explicit "callers of" → Usage; domain term must not overwrite.
        assert_eq!(
            QueryClassifier::classify_with_domain("callers of PMS handler", &terms),
            QueryIntent::Usage
        );
        // Explicit `fn` → Definition (already correct).
        assert_eq!(
            QueryClassifier::classify_with_domain("fn handle_pms", &terms),
            QueryIntent::Definition
        );
        // Explicit TODO → BugDebt.
        assert_eq!(
            QueryClassifier::classify_with_domain("TODO refactor PMS adapter", &terms),
            QueryIntent::BugDebt
        );
    }

    #[test]
    fn test_domain_term_empty_list_passthrough() {
        // With no domain terms, behaviour matches plain `classify`.
        let terms: Vec<String> = vec![];
        assert_eq!(
            QueryClassifier::classify_with_domain("PMS integration", &terms),
            QueryIntent::Unknown
        );
    }

    #[test]
    fn test_short_nl_query_not_forced_conceptual() {
        // Only 3 words — should not match the ≥6-word pattern
        let result = QueryClassifier::classify("reservation booking flow");
        // May be Unknown, not forced to Conceptual
        assert_ne!(result, QueryIntent::Conceptual);
    }
}
