//! Canonical facts table with provenance tracking.
//!
//! Why: Hybrid search retrieves chunks; a fact store retains *distilled*
//! knowledge — `(subject, predicate, object)` triples derived from chunks but
//! detached from any single one. This lets callers ask "what implements X?"
//! or "what tests cover Y?" without re-ranking chunks every time.
//!
//! What: a redb-backed map from `fact_id (u64)` → `FactRecord` (JSON-encoded).
//! `fact_id` is a stable hash of `(subject, predicate, object)` so upserts
//! deduplicate and merge provenance lists rather than creating duplicates.
//!
//! Test: `tests` below covers upsert / query / dedupe / delete round-trips
//! against an in-memory tempfile redb.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use redb::{Database, ReadableTable, TableDefinition};
use serde::{Deserialize, Serialize};

/// redb table holding all facts. Key = `fact_id`, value = JSON `FactRecord`.
const FACTS_TABLE: TableDefinition<u64, &[u8]> = TableDefinition::new("facts");

/// One canonical fact about the indexed corpus.
///
/// Identity is the `(subject, predicate, object)` triple — the `id` field is a
/// hash of those three strings so re-asserting the same fact updates rather
/// than duplicates. `provenance` is a list of `chunk_id`s supporting this fact
/// and is merged on upsert. `confidence` is overwritten with the latest value
/// (callers should monotonically refine, not regress).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FactRecord {
    /// Stable hash of `(subject, predicate, object)`. Computed by `fact_hash`.
    pub id: u64,
    /// Subject of the triple, e.g. `"fn CodeIndexer::search"`.
    pub subject: String,
    /// Predicate, e.g. `"implements"`, `"raises"`, `"tested_by"`.
    pub predicate: String,
    /// Object of the triple, e.g. `"trait Searcher"`.
    pub object: String,
    /// Confidence score in [0.0, 1.0]. Latest value wins on upsert.
    pub confidence: f32,
    /// Chunk IDs supporting this fact. Merged (set-union) on upsert.
    pub provenance: Vec<String>,
    /// Index this fact came from (so we can scope queries per-project).
    pub index_id: String,
    /// Unix timestamp (seconds) at first creation.
    pub created_at: u64,
}

impl FactRecord {
    /// Build a `FactRecord` with the canonical `id` derived from its triple
    /// and `created_at` set to the current wall-clock time. `confidence` and
    /// `provenance` start empty — fill them via the builder methods or set
    /// directly before upserting.
    pub fn new(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
        index_id: impl Into<String>,
    ) -> Self {
        let subject = subject.into();
        let predicate = predicate.into();
        let object = object.into();
        let id = fact_hash(&subject, &predicate, &object);
        Self {
            id,
            subject,
            predicate,
            object,
            confidence: 1.0,
            provenance: Vec::new(),
            index_id: index_id.into(),
            created_at: now_secs(),
        }
    }

    /// Builder: set confidence in [0.0, 1.0]. Values outside the range are
    /// clamped — callers shouldn't rely on out-of-range values being preserved.
    pub fn with_confidence(mut self, c: f32) -> Self {
        self.confidence = c.clamp(0.0, 1.0);
        self
    }

    /// Builder: append `chunk_id` to provenance.
    pub fn with_provenance(mut self, chunk_id: impl Into<String>) -> Self {
        self.provenance.push(chunk_id.into());
        self
    }
}

/// Stable u64 hash of the canonical `(subject, predicate, object)` triple.
///
/// This is the identity function for facts. Using `DefaultHasher` is fine —
/// we only need stability *within a process* between `upsert` and `query`
/// calls. Across releases the hash may shift; that's acceptable because the
/// store is rebuildable from the indexed corpus.
pub fn fact_hash(subject: &str, predicate: &str, object: &str) -> u64 {
    let mut h = DefaultHasher::new();
    (subject, predicate, object).hash(&mut h);
    h.finish()
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// redb-backed store for `FactRecord`s.
///
/// Cheap to clone — internally an `Arc<Database>`.
#[derive(Clone)]
pub struct FactStore {
    db: Arc<Database>,
}

impl FactStore {
    /// Open (creating if needed) the facts database at `path`.
    pub fn open(path: &Path) -> Result<Self> {
        let db = Database::create(path).context("open facts redb")?;
        // Eagerly create the table so first-write doesn't race with first-read.
        let txn = db.begin_write().context("begin facts init txn")?;
        {
            let _t = txn
                .open_table(FACTS_TABLE)
                .context("open facts table for init")?;
        }
        txn.commit().context("commit facts init txn")?;
        Ok(Self { db: Arc::new(db) })
    }

    /// Upsert a fact. If a record with the same `id` exists, provenance is
    /// merged (set-union) and `confidence` is overwritten with the new value.
    /// `created_at` is preserved from the existing record.
    pub fn upsert(&self, mut fact: FactRecord) -> Result<()> {
        // Ensure the id matches the canonical triple, even if callers built
        // FactRecord by struct literal with a stale id.
        fact.id = fact_hash(&fact.subject, &fact.predicate, &fact.object);

        let txn = self.db.begin_write().context("begin upsert txn")?;
        {
            let mut table = txn
                .open_table(FACTS_TABLE)
                .context("open facts table for upsert")?;

            let merged = if let Some(existing_bytes) = table
                .get(fact.id)
                .context("read existing fact for merge")?
            {
                let existing: FactRecord = serde_json::from_slice(existing_bytes.value())
                    .context("decode existing fact for merge")?;
                let mut prov_set: HashSet<String> =
                    existing.provenance.into_iter().collect();
                for p in &fact.provenance {
                    prov_set.insert(p.clone());
                }
                let mut provenance: Vec<String> = prov_set.into_iter().collect();
                provenance.sort(); // deterministic order
                FactRecord {
                    id: fact.id,
                    subject: fact.subject,
                    predicate: fact.predicate,
                    object: fact.object,
                    confidence: fact.confidence,
                    provenance,
                    index_id: fact.index_id,
                    created_at: existing.created_at,
                }
            } else {
                fact
            };

            let bytes = serde_json::to_vec(&merged).context("encode fact")?;
            table
                .insert(merged.id, bytes.as_slice())
                .context("insert fact")?;
        }
        txn.commit().context("commit upsert txn")?;
        Ok(())
    }

    /// Query facts with optional filters. Any combination of filters can be
    /// supplied; `None` means "match anything" for that field.
    pub fn query(
        &self,
        subject: Option<&str>,
        predicate: Option<&str>,
        object: Option<&str>,
    ) -> Result<Vec<FactRecord>> {
        let txn = self.db.begin_read().context("begin query txn")?;
        let table = txn
            .open_table(FACTS_TABLE)
            .context("open facts table for query")?;

        let mut out = Vec::new();
        for row in table.iter().context("iter facts")? {
            let (_, v) = row.context("read fact row")?;
            let fact: FactRecord = serde_json::from_slice(v.value())
                .context("decode fact during query")?;
            if let Some(s) = subject {
                if fact.subject != s {
                    continue;
                }
            }
            if let Some(p) = predicate {
                if fact.predicate != p {
                    continue;
                }
            }
            if let Some(o) = object {
                if fact.object != o {
                    continue;
                }
            }
            out.push(fact);
        }
        Ok(out)
    }

    /// Return every fact in the store (debug / list endpoint helper).
    pub fn all(&self) -> Result<Vec<FactRecord>> {
        self.query(None, None, None)
    }

    /// Delete a fact by id. Returns `true` if a record existed and was removed.
    pub fn delete(&self, id: u64) -> Result<bool> {
        let txn = self.db.begin_write().context("begin delete txn")?;
        let removed = {
            let mut table = txn
                .open_table(FACTS_TABLE)
                .context("open facts table for delete")?;
            let was_present = table.remove(id).context("delete fact")?.is_some();
            was_present
        };
        txn.commit().context("commit delete txn")?;
        Ok(removed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_store() -> (FactStore, TempDir) {
        let tmp = TempDir::new().expect("tempdir");
        let path = tmp.path().join("facts.redb");
        let store = FactStore::open(&path).expect("open facts store");
        (store, tmp)
    }

    #[test]
    fn test_upsert_and_query_by_subject() {
        let (store, _tmp) = make_store();
        let f = FactRecord::new("fn search", "implements", "trait Searcher", "test")
            .with_confidence(0.9)
            .with_provenance("src/indexer.rs:1:10");
        store.upsert(f.clone()).unwrap();

        let hits = store.query(Some("fn search"), None, None).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].subject, "fn search");
        assert_eq!(hits[0].predicate, "implements");
        assert_eq!(hits[0].object, "trait Searcher");
    }

    #[test]
    fn test_upsert_dedupes_and_merges_provenance() {
        let (store, _tmp) = make_store();
        let a = FactRecord::new("X", "implements", "Y", "i1").with_provenance("c1");
        store.upsert(a).unwrap();
        let b = FactRecord::new("X", "implements", "Y", "i1").with_provenance("c2");
        store.upsert(b).unwrap();

        let all = store.all().unwrap();
        assert_eq!(all.len(), 1, "upsert should dedupe by triple");
        let mut prov = all[0].provenance.clone();
        prov.sort();
        assert_eq!(prov, vec!["c1".to_string(), "c2".to_string()]);
    }

    #[test]
    fn test_query_combined_filters() {
        let (store, _tmp) = make_store();
        store
            .upsert(FactRecord::new("A", "calls", "B", "i"))
            .unwrap();
        store
            .upsert(FactRecord::new("A", "calls", "C", "i"))
            .unwrap();
        store
            .upsert(FactRecord::new("D", "calls", "B", "i"))
            .unwrap();

        let hits = store.query(Some("A"), Some("calls"), None).unwrap();
        assert_eq!(hits.len(), 2);
        let hits = store.query(None, Some("calls"), Some("B")).unwrap();
        assert_eq!(hits.len(), 2);
        let hits = store.query(Some("A"), Some("calls"), Some("B")).unwrap();
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn test_delete_returns_true_then_false() {
        let (store, _tmp) = make_store();
        let f = FactRecord::new("X", "y", "Z", "i");
        let id = f.id;
        store.upsert(f).unwrap();
        assert!(store.delete(id).unwrap());
        assert!(!store.delete(id).unwrap());
        assert!(store.all().unwrap().is_empty());
    }

    #[test]
    fn test_fact_hash_is_stable_and_field_sensitive() {
        let h1 = fact_hash("a", "b", "c");
        let h2 = fact_hash("a", "b", "c");
        assert_eq!(h1, h2);
        assert_ne!(h1, fact_hash("a", "b", "d"));
        assert_ne!(h1, fact_hash("a", "x", "c"));
    }

    #[test]
    fn test_confidence_clamps() {
        let f = FactRecord::new("a", "b", "c", "i").with_confidence(2.5);
        assert_eq!(f.confidence, 1.0);
        let f = FactRecord::new("a", "b", "c", "i").with_confidence(-1.0);
        assert_eq!(f.confidence, 0.0);
    }
}
