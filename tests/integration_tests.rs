#[test]
fn test_query_classifier_smoke() {
    use trusty_search::core::classifier::{QueryClassifier, QueryIntent};
    assert_eq!(
        QueryClassifier::classify("fn authenticate"),
        QueryIntent::Definition
    );
    assert_eq!(
        QueryClassifier::classify("how does auth work"),
        QueryIntent::Conceptual
    );
}

#[test]
fn test_bm25_smoke() {
    use trusty_search::core::bm25::Bm25Index;
    let mut idx = Bm25Index::new();
    idx.add_document(0, "rust async tokio search");
    idx.add_document(1, "python django web framework");
    let s = idx.score("rust tokio", 0);
    assert!(s > 0.0);
}

#[test]
fn test_chunker_smoke() {
    use trusty_search::core::chunker::chunk_text;
    let content = "fn foo() {}\nfn bar() {}\n";
    let chunks = chunk_text("test.rs", content, 150, 50);
    assert!(!chunks.is_empty());
    assert!(chunks[0].id.starts_with("test.rs:"));
}

#[test]
fn test_index_registry() {
    use std::sync::Arc;
    use tokio::sync::RwLock;
    use trusty_search::core::{
        indexer::CodeIndexer,
        registry::{IndexHandle, IndexId, IndexRegistry},
    };

    let registry = IndexRegistry::new();
    let id = IndexId::new("test-project");
    let indexer = CodeIndexer::new("test-project", "/tmp/test");
    registry.register(IndexHandle::bare(
        id.clone(),
        Arc::new(RwLock::new(indexer)),
        "/tmp/test".into(),
    ));
    assert!(registry.get(&id).is_some());
    assert_eq!(registry.len(), 1);
}
