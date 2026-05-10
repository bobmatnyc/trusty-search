//! `SymbolGraph`: petgraph-backed call graph derived from the chunk corpus.
//!
//! Why: query intent like "who calls `authenticate`?" or "what does `process_request`
//! delegate to?" can't be answered well by BM25/HNSW alone. A directed call graph
//! (caller → callee) lets the search pipeline expand around a hit, surfacing
//! adjacent code at a discounted score (KG-expansion = 0.7 × trigger RRF score).
//!
//! What: a `petgraph::DiGraph<SymbolNode, ()>` keyed by symbol name (the
//! `function_name` recorded on each `RawChunk` — qualified for Rust methods, e.g.
//! `Foo::bar`). Edges point from caller symbol to callee symbol. The graph is
//! cheap to rebuild from the corpus and is held in `Arc<SymbolGraph>` so search
//! handlers can read concurrently without locking.
//!
//! Test: see the `tests` module — covers basic build, `callers_of`, `callees_of`,
//! 1-hop and 2-hop traversal, qualified-method names, and unknown-symbol queries.

use std::collections::{HashMap, HashSet, VecDeque};

use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use petgraph::Direction;
use serde::{Deserialize, Serialize};

use crate::chunker::ChunkType;
use crate::entity::EdgeKind;

/// A node in the symbol graph. One node per defining symbol (function or method).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolNode {
    /// Defining symbol name. For Rust methods this is qualified (`Foo::bar`);
    /// for free functions it's the bare name.
    pub symbol: String,
    /// `RawChunk.id` of the chunk that defines this symbol.
    pub chunk_id: String,
    /// Source file path (for debugging / display).
    pub file: String,
}

/// Tuple shape consumed by [`SymbolGraph::build_from_chunks`].
///
/// Fields, in order: `(chunk_id, file, function_name, calls, inherits_from,
/// chunk_type)`. Aliased so the public signature stays clippy-clean (large
/// inline tuple types trip `clippy::type_complexity`).
pub type ChunkTuple = (
    String,
    String,
    Option<String>,
    Vec<String>,
    Vec<String>,
    ChunkType,
);

/// A petgraph-backed directed call graph: edge `A → B` means "A calls B".
///
/// Built from a slice of `(chunk_id, file, function_name, calls)` tuples; the
/// chunker (`chunk_ast`) is responsible for populating the `function_name` and
/// `calls` fields per chunk, so the graph just stitches them together.
#[derive(Debug, Default)]
pub struct SymbolGraph {
    graph: DiGraph<SymbolNode, EdgeKind>,
    /// Symbol name → node index. Holds the *first* definition seen if a symbol
    /// is defined twice (rare; e.g. `cfg`-gated duplicates).
    by_symbol: HashMap<String, NodeIndex>,
    /// chunk_id → symbol name, so callers can resolve a search hit to its node.
    chunk_to_symbol: HashMap<String, String>,
}

impl SymbolGraph {
    /// Construct an empty graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a graph from the chunk corpus.
    ///
    /// Each tuple is
    /// `(chunk_id, file, function_name, calls, inherits_from, chunk_type)`:
    /// - `function_name`: `None` for non-callable chunks (structs, modules, …);
    ///   such chunks contribute no node.
    /// - `calls`: simple-name callees (the chunker reduces `obj.method` and
    ///   `foo::bar` to the trailing identifier). We add a `CallsFunction` edge
    ///   per call only if the callee symbol is also defined in the corpus, so
    ///   the graph stays closed over local code (no edges pointing into the
    ///   void).
    /// - `inherits_from`: parent type names. For each parent that's defined in
    ///   the corpus, emit an `Implements` edge from the child symbol → parent.
    /// - `chunk_type`: container chunks (`Impl`, `Class`, `Struct`, `Module`)
    ///   emit `ModuleContains` edges to every other defining symbol that lives
    ///   in the same file. Coarse but cheap; nesting-depth refinement can come
    ///   later.
    pub fn build_from_chunks(chunks: &[ChunkTuple]) -> Self {
        let mut g = Self::new();

        // Pass 1: register all defining symbols.
        for (chunk_id, file, name, _calls, _inh, _ct) in chunks {
            let Some(name) = name else { continue };
            if name.is_empty() {
                continue;
            }
            // First-write-wins so chunk_to_symbol stays stable.
            if g.by_symbol.contains_key(name) {
                g.chunk_to_symbol.insert(chunk_id.clone(), name.clone());
                continue;
            }
            let idx = g.graph.add_node(SymbolNode {
                symbol: name.clone(),
                chunk_id: chunk_id.clone(),
                file: file.clone(),
            });
            g.by_symbol.insert(name.clone(), idx);
            g.chunk_to_symbol.insert(chunk_id.clone(), name.clone());
        }

        // Build a `simple_name → first-NodeIndex` lookup for qualified-symbol
        // resolution. Replaces the per-edge `O(symbols)` linear suffix scan
        // that used to live inside `resolve_callee`. On a 115k-chunk corpus
        // with thousands of qualified methods this collapses what was an
        // O(N²) build pass into O(N).
        let mut by_suffix: HashMap<&str, NodeIndex> = HashMap::new();
        for (sym, &idx) in g.by_symbol.iter() {
            if let Some(suffix) = sym.rsplit("::").next() {
                // First-write-wins to match the original semantics (the old
                // `find` returned the first qualified hit).
                by_suffix.entry(suffix).or_insert(idx);
            }
        }

        // Pass 2: add CallsFunction + Implements edges. For each call, also
        // try the simple-name suffix of a qualified caller's symbol, so
        // `Foo::bar` calling `baz` resolves even when only `baz` is in the
        // index.
        for (_chunk_id, _file, name, calls, inherits_from, _ct) in chunks {
            let Some(name) = name else { continue };
            let Some(&from) = g.by_symbol.get(name) else {
                continue;
            };
            for callee in calls {
                if let Some(to) = g.resolve_callee_fast(callee, &by_suffix) {
                    if from != to {
                        g.graph.add_edge(from, to, EdgeKind::CallsFunction);
                    }
                }
            }
            // Issue #33: INHERITS / Implements edges from `inherits_from`.
            for parent in inherits_from {
                if let Some(to) = g.resolve_callee_fast(parent, &by_suffix) {
                    if from != to {
                        g.graph.add_edge(from, to, EdgeKind::Implements);
                    }
                }
            }
        }

        // Pass 3: ModuleContains edges from container chunks (Impl / Class /
        // Struct / Module) to other defining symbols in the same file.
        // `by_file` is built lazily here so we only pay when there's at least
        // one container in the corpus.
        let has_container = chunks.iter().any(|(_, _, name, _, _, ct)| {
            name.is_some()
                && matches!(
                    ct,
                    ChunkType::Impl
                        | ChunkType::Class
                        | ChunkType::Struct
                        | ChunkType::Module
                )
        });
        if has_container {
            // file → list of (symbol, NodeIndex) for everything defined there.
            let mut by_file: HashMap<&str, Vec<(&str, NodeIndex)>> = HashMap::new();
            for (_chunk_id, file, name, _calls, _inh, _ct) in chunks {
                if let Some(name) = name {
                    if let Some(&idx) = g.by_symbol.get(name) {
                        by_file
                            .entry(file.as_str())
                            .or_default()
                            .push((name.as_str(), idx));
                    }
                }
            }
            for (_chunk_id, file, name, _calls, _inh, ct) in chunks {
                if !matches!(
                    ct,
                    ChunkType::Impl
                        | ChunkType::Class
                        | ChunkType::Struct
                        | ChunkType::Module
                ) {
                    continue;
                }
                let Some(name) = name else { continue };
                let Some(&from) = g.by_symbol.get(name) else {
                    continue;
                };
                let Some(siblings) = by_file.get(file.as_str()) else {
                    continue;
                };
                for (sib_name, sib_idx) in siblings {
                    if *sib_idx == from || *sib_name == name.as_str() {
                        continue;
                    }
                    g.graph
                        .add_edge(from, *sib_idx, EdgeKind::ModuleContains);
                }
            }
        }

        g
    }

    /// O(1) callee lookup using a precomputed `simple_name → NodeIndex` map.
    ///
    /// Why: the previous implementation linearly scanned every symbol per call
    /// edge looking for a `::callee` suffix. On a 115k-chunk corpus this was
    /// the single biggest cost in `build_from_chunks`. We now materialize the
    /// suffix map once per build and look up in O(1).
    fn resolve_callee_fast(
        &self,
        callee: &str,
        by_suffix: &HashMap<&str, NodeIndex>,
    ) -> Option<NodeIndex> {
        if let Some(&idx) = self.by_symbol.get(callee) {
            return Some(idx);
        }
        by_suffix.get(callee).copied()
    }

    /// Number of symbol nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Number of call edges in the graph.
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Look up the defining symbol for a chunk_id, if any.
    pub fn symbol_for_chunk(&self, chunk_id: &str) -> Option<&str> {
        self.chunk_to_symbol.get(chunk_id).map(|s| s.as_str())
    }

    /// BFS up to `hops` levels: symbols that (transitively) call `symbol`.
    /// Returns `Vec<(symbol, chunk_id)>` excluding `symbol` itself.
    pub fn callers_of(&self, symbol: &str, hops: usize) -> Vec<(String, String)> {
        self.bfs_neighbors(symbol, hops, Direction::Incoming)
    }

    /// BFS up to `hops` levels: symbols (transitively) called by `symbol`.
    /// Returns `Vec<(symbol, chunk_id)>` excluding `symbol` itself.
    pub fn callees_of(&self, symbol: &str, hops: usize) -> Vec<(String, String)> {
        self.bfs_neighbors(symbol, hops, Direction::Outgoing)
    }

    /// BFS up to `hops` levels, walking only edges whose `EdgeKind` is in
    /// `edge_kinds`. Returns `(symbol, chunk_id, edge_kind)` triples for each
    /// neighbour discovered (excluding `symbol` itself).
    ///
    /// Used by intent-gated KG expansion (issue #18) so each query intent
    /// traverses the subset of edge types most likely to surface relevant
    /// adjacent code (`Implements`/`UsesType` for definitions, `CallsFunction`
    /// for usage, `RaisesError` for bug-debt, …).
    pub fn neighbors_by_edge(
        &self,
        symbol: &str,
        edge_kinds: &[EdgeKind],
        hops: usize,
    ) -> Vec<(String, String, EdgeKind)> {
        let Some(&start) = self.by_symbol.get(symbol) else {
            return Vec::new();
        };
        if hops == 0 || edge_kinds.is_empty() {
            return Vec::new();
        }
        let allowed: HashSet<&EdgeKind> = edge_kinds.iter().collect();

        let mut visited: HashSet<NodeIndex> = HashSet::new();
        visited.insert(start);
        let mut queue: VecDeque<(NodeIndex, usize)> = VecDeque::new();
        queue.push_back((start, 0));
        let mut out: Vec<(String, String, EdgeKind)> = Vec::new();

        while let Some((node, depth)) = queue.pop_front() {
            if depth >= hops {
                continue;
            }
            // Walk both directions: caller→callee and callee→caller relations
            // are both useful for KG expansion.
            for dir in [Direction::Outgoing, Direction::Incoming] {
                for edge in self.graph.edges_directed(node, dir) {
                    if !allowed.contains(edge.weight()) {
                        continue;
                    }
                    let nb = match dir {
                        Direction::Outgoing => edge.target(),
                        Direction::Incoming => edge.source(),
                    };
                    if visited.insert(nb) {
                        let n = &self.graph[nb];
                        out.push((n.symbol.clone(), n.chunk_id.clone(), edge.weight().clone()));
                        queue.push_back((nb, depth + 1));
                    }
                }
            }
        }
        out
    }

    fn bfs_neighbors(&self, symbol: &str, hops: usize, dir: Direction) -> Vec<(String, String)> {
        let Some(&start) = self.by_symbol.get(symbol) else {
            return Vec::new();
        };
        if hops == 0 {
            return Vec::new();
        }

        let mut visited: HashSet<NodeIndex> = HashSet::new();
        visited.insert(start);
        let mut queue: VecDeque<(NodeIndex, usize)> = VecDeque::new();
        queue.push_back((start, 0));
        let mut out: Vec<(String, String)> = Vec::new();

        while let Some((node, depth)) = queue.pop_front() {
            if depth >= hops {
                continue;
            }
            // Only walk call-graph edges; other `EdgeKind`s belong to entity
            // expansion paths (Phase A/B/C) and shouldn't pollute callers/callees.
            for edge in self.graph.edges_directed(node, dir) {
                if edge.weight() != &EdgeKind::CallsFunction {
                    continue;
                }
                let nb = match dir {
                    Direction::Outgoing => edge.target(),
                    Direction::Incoming => edge.source(),
                };
                if visited.insert(nb) {
                    let n = &self.graph[nb];
                    out.push((n.symbol.clone(), n.chunk_id.clone()));
                    queue.push_back((nb, depth + 1));
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(id: &str, file: &str, name: Option<&str>, calls: &[&str]) -> ChunkTuple {
        chunk_full(id, file, name, calls, &[], ChunkType::Function)
    }

    fn chunk_full(
        id: &str,
        file: &str,
        name: Option<&str>,
        calls: &[&str],
        inherits_from: &[&str],
        chunk_type: ChunkType,
    ) -> ChunkTuple {
        (
            id.to_string(),
            file.to_string(),
            name.map(String::from),
            calls.iter().map(|s| s.to_string()).collect(),
            inherits_from.iter().map(|s| s.to_string()).collect(),
            chunk_type,
        )
    }

    #[test]
    fn test_build_simple_graph() {
        let chunks = vec![
            chunk("a:1", "a.rs", Some("main"), &["foo", "bar"]),
            chunk("a:2", "a.rs", Some("foo"), &["bar"]),
            chunk("a:3", "a.rs", Some("bar"), &[]),
        ];
        let g = SymbolGraph::build_from_chunks(&chunks);
        assert_eq!(g.node_count(), 3);
        // main→foo, main→bar, foo→bar = 3 edges
        assert_eq!(g.edge_count(), 3);
    }

    #[test]
    fn test_callers_of_one_hop() {
        let chunks = vec![
            chunk("m:1", "m.rs", Some("main"), &["authenticate"]),
            chunk("h:1", "h.rs", Some("login_handler"), &["authenticate"]),
            chunk("a:1", "a.rs", Some("authenticate"), &[]),
        ];
        let g = SymbolGraph::build_from_chunks(&chunks);
        let mut callers = g.callers_of("authenticate", 1);
        callers.sort();
        assert_eq!(
            callers,
            vec![
                ("login_handler".to_string(), "h:1".to_string()),
                ("main".to_string(), "m:1".to_string()),
            ]
        );
    }

    #[test]
    fn test_callees_of_one_hop() {
        let chunks = vec![
            chunk("a:1", "a.rs", Some("authenticate"), &["hash_password", "lookup_user"]),
            chunk("p:1", "p.rs", Some("hash_password"), &[]),
            chunk("u:1", "u.rs", Some("lookup_user"), &[]),
        ];
        let g = SymbolGraph::build_from_chunks(&chunks);
        let mut callees = g.callees_of("authenticate", 1);
        callees.sort();
        assert_eq!(
            callees,
            vec![
                ("hash_password".to_string(), "p:1".to_string()),
                ("lookup_user".to_string(), "u:1".to_string()),
            ]
        );
    }

    #[test]
    fn test_two_hop_traversal() {
        // a → b → c
        let chunks = vec![
            chunk("a:1", "a.rs", Some("a"), &["b"]),
            chunk("b:1", "b.rs", Some("b"), &["c"]),
            chunk("c:1", "c.rs", Some("c"), &[]),
        ];
        let g = SymbolGraph::build_from_chunks(&chunks);
        let one_hop = g.callees_of("a", 1);
        assert_eq!(one_hop.len(), 1);
        assert_eq!(one_hop[0].0, "b");

        let two_hop = g.callees_of("a", 2);
        let names: Vec<&str> = two_hop.iter().map(|(s, _)| s.as_str()).collect();
        assert!(names.contains(&"b"));
        assert!(names.contains(&"c"));
    }

    #[test]
    fn test_unknown_symbol_returns_empty() {
        let chunks = vec![chunk("a:1", "a.rs", Some("a"), &[])];
        let g = SymbolGraph::build_from_chunks(&chunks);
        assert!(g.callers_of("nonexistent", 1).is_empty());
        assert!(g.callees_of("nonexistent", 1).is_empty());
    }

    #[test]
    fn test_qualified_method_resolves_simple_callee() {
        // `Foo::bar` calls `baz`; only `Foo::bar` and `baz` are in the corpus.
        let chunks = vec![
            chunk("f:1", "f.rs", Some("Foo::bar"), &["baz"]),
            chunk("b:1", "b.rs", Some("baz"), &[]),
        ];
        let g = SymbolGraph::build_from_chunks(&chunks);
        let callers = g.callers_of("baz", 1);
        assert_eq!(callers.len(), 1);
        assert_eq!(callers[0].0, "Foo::bar");
    }

    #[test]
    fn test_simple_callee_resolves_to_qualified_definition() {
        // Caller writes `bar()`; only `Foo::bar` is defined.
        let chunks = vec![
            chunk("c:1", "c.rs", Some("caller"), &["bar"]),
            chunk("f:1", "f.rs", Some("Foo::bar"), &[]),
        ];
        let g = SymbolGraph::build_from_chunks(&chunks);
        let callees = g.callees_of("caller", 1);
        assert_eq!(callees.len(), 1);
        assert_eq!(callees[0].0, "Foo::bar");
    }

    #[test]
    fn test_chunk_with_no_function_name_is_skipped() {
        let chunks = vec![
            chunk("s:1", "s.rs", None, &[]),
            chunk("f:1", "f.rs", Some("f"), &[]),
        ];
        let g = SymbolGraph::build_from_chunks(&chunks);
        assert_eq!(g.node_count(), 1);
    }

    #[test]
    fn test_zero_hops_returns_empty() {
        let chunks = vec![
            chunk("a:1", "a.rs", Some("a"), &["b"]),
            chunk("b:1", "b.rs", Some("b"), &[]),
        ];
        let g = SymbolGraph::build_from_chunks(&chunks);
        assert!(g.callees_of("a", 0).is_empty());
    }

    #[test]
    fn test_symbol_for_chunk() {
        let chunks = vec![chunk("a:1", "a.rs", Some("alpha"), &[])];
        let g = SymbolGraph::build_from_chunks(&chunks);
        assert_eq!(g.symbol_for_chunk("a:1"), Some("alpha"));
        assert_eq!(g.symbol_for_chunk("missing"), None);
    }

    #[test]
    fn test_neighbors_by_edge_filters_by_kind() {
        // Build a graph with two edge kinds. neighbors_by_edge must only
        // return neighbours reachable via the requested kinds.
        let mut g = SymbolGraph::new();
        let a = g.graph.add_node(SymbolNode {
            symbol: "a".into(),
            chunk_id: "a:1".into(),
            file: "a.rs".into(),
        });
        let b = g.graph.add_node(SymbolNode {
            symbol: "b".into(),
            chunk_id: "b:1".into(),
            file: "b.rs".into(),
        });
        let c = g.graph.add_node(SymbolNode {
            symbol: "c".into(),
            chunk_id: "c:1".into(),
            file: "c.rs".into(),
        });
        g.by_symbol.insert("a".into(), a);
        g.by_symbol.insert("b".into(), b);
        g.by_symbol.insert("c".into(), c);
        g.graph.add_edge(a, b, EdgeKind::CallsFunction);
        g.graph.add_edge(a, c, EdgeKind::Implements);

        let calls = g.neighbors_by_edge("a", &[EdgeKind::CallsFunction], 1);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "b");

        let impls = g.neighbors_by_edge("a", &[EdgeKind::Implements], 1);
        assert_eq!(impls.len(), 1);
        assert_eq!(impls[0].0, "c");

        let both = g.neighbors_by_edge(
            "a",
            &[EdgeKind::CallsFunction, EdgeKind::Implements],
            1,
        );
        assert_eq!(both.len(), 2);

        // Empty edge set returns nothing.
        assert!(g.neighbors_by_edge("a", &[], 1).is_empty());
        // Zero hops returns nothing.
        assert!(g
            .neighbors_by_edge("a", &[EdgeKind::CallsFunction], 0)
            .is_empty());
    }

    #[test]
    fn test_calls_function_edges_present_in_graph() {
        // Issue #33: a chunk whose `calls` field lists `bar` must produce a
        // `CallsFunction` edge from the caller's symbol to bar.
        let chunks = vec![
            chunk("a:1", "a.rs", Some("alpha"), &["bar"]),
            chunk("b:1", "a.rs", Some("bar"), &[]),
        ];
        let g = SymbolGraph::build_from_chunks(&chunks);
        let calls = g.neighbors_by_edge("alpha", &[EdgeKind::CallsFunction], 1);
        assert_eq!(
            calls.len(),
            1,
            "expected exactly one CallsFunction neighbour, got {calls:?}"
        );
        assert_eq!(calls[0].0, "bar");
        assert!(matches!(calls[0].2, EdgeKind::CallsFunction));
    }

    #[test]
    fn test_inherits_from_emits_implements_edges() {
        // Issue #33: a chunk's `inherits_from` field should produce
        // `Implements` edges to each parent that's defined in the corpus.
        let chunks = vec![
            chunk_full(
                "c:1",
                "c.rs",
                Some("Child"),
                &[],
                &["Parent"],
                ChunkType::Class,
            ),
            chunk_full("p:1", "p.rs", Some("Parent"), &[], &[], ChunkType::Class),
        ];
        let g = SymbolGraph::build_from_chunks(&chunks);
        let impls = g.neighbors_by_edge("Child", &[EdgeKind::Implements], 1);
        assert_eq!(impls.len(), 1, "expected one Implements edge: {impls:?}");
        assert_eq!(impls[0].0, "Parent");
    }

    #[test]
    fn test_module_contains_edges_from_container_chunks() {
        // Issue #33: a container chunk (Impl/Class/Struct/Module) should emit
        // `ModuleContains` edges to other defining symbols in the same file.
        let chunks = vec![
            chunk_full(
                "i:1",
                "f.rs",
                Some("FooImpl"),
                &[],
                &[],
                ChunkType::Impl,
            ),
            chunk_full(
                "m:1",
                "f.rs",
                Some("method_a"),
                &[],
                &[],
                ChunkType::Method,
            ),
            chunk_full(
                "m:2",
                "f.rs",
                Some("method_b"),
                &[],
                &[],
                ChunkType::Method,
            ),
            // A symbol in a different file should NOT be contained.
            chunk_full(
                "o:1",
                "other.rs",
                Some("outside"),
                &[],
                &[],
                ChunkType::Function,
            ),
        ];
        let g = SymbolGraph::build_from_chunks(&chunks);
        let contained = g.neighbors_by_edge("FooImpl", &[EdgeKind::ModuleContains], 1);
        let names: HashSet<&str> = contained.iter().map(|(n, _, _)| n.as_str()).collect();
        assert!(names.contains("method_a"), "got {names:?}");
        assert!(names.contains("method_b"), "got {names:?}");
        assert!(!names.contains("outside"), "cross-file leak: {names:?}");
    }

    #[test]
    fn test_neighbors_by_edge_only_returns_filtered_kinds() {
        // Issue #33: a graph with mixed edge kinds — filtering by one kind
        // must not surface neighbours reachable only through other kinds.
        let chunks = vec![
            chunk_full(
                "a:1",
                "a.rs",
                Some("Alpha"),
                &["beta"],
                &["BaseAlpha"],
                ChunkType::Class,
            ),
            chunk("b:1", "a.rs", Some("beta"), &[]),
            chunk_full(
                "ba:1",
                "a.rs",
                Some("BaseAlpha"),
                &[],
                &[],
                ChunkType::Class,
            ),
        ];
        let g = SymbolGraph::build_from_chunks(&chunks);

        let calls = g.neighbors_by_edge("Alpha", &[EdgeKind::CallsFunction], 1);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "beta");
        assert!(calls.iter().all(|(_, _, k)| k == &EdgeKind::CallsFunction));

        let impls = g.neighbors_by_edge("Alpha", &[EdgeKind::Implements], 1);
        assert!(impls.iter().any(|(n, _, _)| n == "BaseAlpha"));
        assert!(impls.iter().all(|(_, _, k)| k == &EdgeKind::Implements));
    }

    #[test]
    fn test_self_call_does_not_create_self_loop() {
        // Recursive function: `f` calls `f`. We skip self-edges so KG expansion
        // doesn't surface the trigger chunk as its own neighbor.
        let chunks = vec![chunk("f:1", "f.rs", Some("f"), &["f"])];
        let g = SymbolGraph::build_from_chunks(&chunks);
        assert_eq!(g.edge_count(), 0);
    }
}
