//! Optional ONNX NER for extracting `NaturalLanguagePhrase` entities from doc comments.
//!
//! Why: doc comments and `///` blocks are dense with natural-language signal
//! ("async runtime", "rate limiter", "exponential backoff") that the structural
//! AST extractor can't see. A small ONNX NER model surfaces these phrases so
//! they show up in BM25 / KG lookups alongside `NamedType` / `ModulePath`.
//!
//! What: a runtime-gated extractor. `NerExtractor::try_load()` always succeeds
//! and returns a disabled extractor unless **both** are true:
//!   1. The crate was built with `--features ner` (pulls in `ort` + `tokenizers`).
//!   2. The file `~/.trusty-search/models/ner.onnx` exists.
//!
//! When disabled, `extract()` is a no-op returning `vec![]`. This lets the
//! daemon ship a single binary and opportunistically light up NER if the user
//! drops the model into place — no rebuild, no restart of unrelated code paths.
//!
//! Test: see `#[cfg(test)]` below — the gating logic is exercised without
//! requiring an ONNX model to be present.

#[cfg(feature = "ner")]
use crate::entity::EntityType;
use crate::entity::RawEntity;

/// NER model handle. Always constructible; inference only runs when both the
/// `ner` feature is compiled in *and* the ONNX model file is present at the
/// expected path. See module docs.
pub struct NerExtractor {
    enabled: bool,
    #[cfg(feature = "ner")]
    inner: Option<NerInner>,
}

#[cfg(feature = "ner")]
struct NerInner {
    session: ort::session::Session,
    tokenizer: tokenizers::Tokenizer,
}

impl NerExtractor {
    /// Attempt to load the ONNX NER model from
    /// `~/.trusty-search/models/ner.onnx`. Always returns a value: a disabled
    /// extractor when the feature is off, the model file is missing, or
    /// loading fails. Failures are logged at `warn` level and never propagate.
    pub fn try_load() -> Self {
        #[cfg(feature = "ner")]
        {
            if let Some(path) = model_path() {
                if path.exists() {
                    match Self::load_from_path(&path) {
                        Ok(ext) => return ext,
                        Err(err) => {
                            tracing::warn!(
                                "NER model present at {} but failed to load: {err:#}; \
                                 NER will be disabled",
                                path.display()
                            );
                        }
                    }
                } else {
                    tracing::debug!(
                        "NER model not found at {}; extractor disabled",
                        path.display()
                    );
                }
            }
            return Self {
                enabled: false,
                inner: None,
            };
        }
        #[cfg(not(feature = "ner"))]
        {
            tracing::debug!("NER feature not compiled in; extractor disabled");
            Self { enabled: false }
        }
    }

    /// Whether this extractor will actually run inference.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Run NER over `doc_text` and return any `NaturalLanguagePhrase` entities.
    /// Returns an empty vector when disabled or when inference fails — never
    /// panics, never propagates errors. Callers should `extend` results into
    /// the existing entity list unconditionally.
    pub fn extract(&self, doc_text: &str, file: &str) -> Vec<RawEntity> {
        if !self.enabled || doc_text.trim().is_empty() {
            return Vec::new();
        }
        #[cfg(feature = "ner")]
        {
            if let Some(inner) = &self.inner {
                return run_inference(inner, doc_text, file).unwrap_or_else(|err| {
                    tracing::debug!("NER inference failed: {err:#}");
                    Vec::new()
                });
            }
            Vec::new()
        }
        #[cfg(not(feature = "ner"))]
        {
            let _ = (doc_text, file);
            Vec::new()
        }
    }

    #[cfg(feature = "ner")]
    fn load_from_path(model_path: &std::path::Path) -> anyhow::Result<Self> {
        use anyhow::Context;
        let session = ort::session::Session::builder()
            .context("ort: builder")?
            .commit_from_file(model_path)
            .with_context(|| format!("ort: load model {}", model_path.display()))?;
        // Tokenizer expected next to the model as `tokenizer.json`. This is
        // the HF convention and keeps the on-disk layout minimal.
        let tokenizer_path = model_path.with_file_name("tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("load tokenizer {}: {e}", tokenizer_path.display()))?;
        Ok(Self {
            enabled: true,
            inner: Some(NerInner { session, tokenizer }),
        })
    }
}

/// Resolve `~/.trusty-search/models/ner.onnx` without pulling a `dirs` dep.
#[cfg(feature = "ner")]
fn model_path() -> Option<std::path::PathBuf> {
    std::env::var_os("HOME")
        .map(std::path::PathBuf::from)
        .map(|h| h.join(".trusty-search/models/ner.onnx"))
}

/// Run a real NER pass. Skeleton — returns an empty vector. Replacing this
/// with a tokenize → forward → BIO-decode pipeline is independent of the gating
/// logic the rest of the codebase relies on.
#[cfg(feature = "ner")]
fn run_inference(inner: &NerInner, doc_text: &str, file: &str) -> anyhow::Result<Vec<RawEntity>> {
    // Touch the fields so that compiling with `--features ner` doesn't warn
    // about unused inner state until the inference body lands.
    let _ = (&inner.session, &inner.tokenizer, doc_text, file);
    let _ = EntityType::NaturalLanguagePhrase;
    Ok(Vec::new())
}

/// Extract `///` and `//!` doc-comment text from a chunk's source body.
/// Used by `CodeIndexer::index_file` to build the input to `extract`.
/// Strips the leading comment markers and joins lines with spaces so the NER
/// model sees a single paragraph instead of fragmented tokens.
pub fn extract_doc_comments(content: &str) -> String {
    let mut out: Vec<&str> = Vec::new();
    for raw in content.lines() {
        let line = raw.trim_start();
        let stripped = line
            .strip_prefix("///")
            .or_else(|| line.strip_prefix("//!"));
        if let Some(rest) = stripped {
            out.push(rest.trim());
        }
    }
    out.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ner_disabled_without_model() {
        // The test environment doesn't ship the ONNX model file, so try_load
        // should always return a disabled extractor regardless of the `ner`
        // feature flag.
        let extractor = NerExtractor::try_load();
        assert!(
            !extractor.is_enabled(),
            "extractor must be disabled in tests (no ner.onnx present)"
        );
        let result = extractor.extract("async runtime", "foo.rs");
        assert!(
            result.is_empty(),
            "disabled extractor must return no entities"
        );
    }

    #[test]
    fn extract_handles_empty_input() {
        let extractor = NerExtractor::try_load();
        assert!(extractor.extract("", "foo.rs").is_empty());
        assert!(extractor.extract("   \n  ", "foo.rs").is_empty());
    }

    #[test]
    fn doc_comment_extraction_pulls_triple_slash_lines() {
        let src = "/// Async runtime hint\n\
                   //! Module-level note\n\
                   fn foo() {}\n\
                   // regular comment ignored\n\
                   /// rate limiter\n";
        let doc = extract_doc_comments(src);
        assert_eq!(doc, "Async runtime hint Module-level note rate limiter");
    }

    #[test]
    fn doc_comment_extraction_empty_when_no_doc_lines() {
        assert_eq!(extract_doc_comments("fn foo() {}\n// not a doc"), "");
    }
}
