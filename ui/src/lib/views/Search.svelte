<script>
  /*
   * Why: Operator-facing search box that fans out across every registered
   * index via the global `POST /search` endpoint. No per-index picker keeps
   * the UI simple — the daemon's RRF fan-out already returns merged top-k.
   * What: Query input + results list. Each result row renders the file
   * path, a snippet (compact_snippet or content), and the relevance score.
   * Test: With at least one index seeded, type "fn" and click Search;
   * results render with non-zero scores.
   */
  import { api } from '../api.js';
  import { getIndexes } from '../state.svelte.js';

  let query = $state('');
  let topK = $state(10);
  let results = $state([]);
  let intent = $state(null);
  let latencyMs = $state(null);
  let indexesSearched = $state([]);
  let loading = $state(false);
  let error = $state(null);

  let indexes = $derived(getIndexes());

  async function runSearch() {
    if (!query.trim()) return;
    loading = true;
    error = null;
    try {
      const body = await api.globalSearch(query.trim(), topK, false);
      results = body.results || [];
      intent = body.intent ?? null;
      latencyMs = body.latency_ms ?? null;
      indexesSearched = body.indexes_searched || [];
    } catch (e) {
      error = e.message || String(e);
      results = [];
    } finally {
      loading = false;
    }
  }

  function onKey(e) {
    if (e.key === 'Enter') runSearch();
  }

  /**
   * Why: The daemon returns either `compact_snippet` (default) or full
   * `content` (when full_content=true). Pick whichever is present and trim.
   * What: Returns a short snippet for display.
   * Test: Pass `{compact_snippet: 'abc'}`, expect 'abc'.
   */
  function snippet(chunk) {
    const raw = chunk.compact_snippet || chunk.content || '';
    if (raw.length <= 320) return raw;
    return raw.slice(0, 320) + '…';
  }
</script>

<h1 class="page-title">Search</h1>

<div class="card mb-4">
  <div class="card-body">
    <div class="search-row">
      <input
        type="text"
        class="input"
        placeholder="Search across all indexes…"
        bind:value={query}
        onkeydown={onKey}
      />
      <input
        type="number"
        class="input top-k"
        min="1"
        max="100"
        bind:value={topK}
        title="top_k"
      />
      <button
        class="btn btn-primary"
        onclick={runSearch}
        disabled={loading || !query.trim()}
      >
        {loading ? 'Searching…' : 'Search'}
      </button>
    </div>
    <div class="meta">
      {#if indexes.length === 0}
        <span class="text-muted text-sm"
          >No indexes registered — create one from the Indexes view.</span
        >
      {:else}
        <span class="text-muted text-sm">
          Searches {indexes.length} index{indexes.length === 1 ? '' : 'es'}.
        </span>
      {/if}
      {#if latencyMs !== null}
        <span class="text-muted text-sm">· {latencyMs}ms</span>
      {/if}
      {#if intent}
        <span class="badge badge-info">{intent}</span>
      {/if}
    </div>
  </div>
</div>

{#if error}
  <div class="card" style="border-color: var(--trusty-danger)">
    <div class="card-body" style="color: var(--trusty-danger)">{error}</div>
  </div>
{/if}

{#if results.length === 0 && !loading && !error}
  <div class="empty">
    {#if query.trim()}
      No results.
    {:else}
      Type a query above to search across all registered indexes.
    {/if}
  </div>
{:else}
  <div class="results">
    {#each results as r, i (r.id || i)}
      <div class="result">
        <div class="result-head">
          <div class="result-path">
            <span class="text-mono text-sm">{r.file || r.path || r.id}</span>
            {#if r.function}
              <span class="badge badge-muted">{r.function}</span>
            {/if}
            {#if r.index_id}
              <span class="badge badge-info">{r.index_id}</span>
            {/if}
            {#if r.match_reason}
              <span class="badge">{r.match_reason}</span>
            {/if}
            {#if r.start_line}
              <span class="text-muted text-xs">L{r.start_line}{r.end_line ? `–${r.end_line}` : ''}</span>
            {/if}
          </div>
          <div class="result-score">
            <span class="score-label">score</span>
            <span class="score-value">{(r.score ?? 0).toFixed(3)}</span>
          </div>
        </div>
        <pre class="snippet">{snippet(r)}</pre>
      </div>
    {/each}
  </div>
{/if}

<style>
  .page-title {
    font-size: var(--trusty-fs-xl);
    margin: 0 0 var(--trusty-space-5) 0;
    font-weight: 600;
  }
  .search-row {
    display: flex;
    gap: var(--trusty-space-2);
    align-items: stretch;
  }
  .top-k {
    width: 80px;
    flex: 0 0 80px;
  }
  .meta {
    display: flex;
    gap: var(--trusty-space-3);
    align-items: center;
    margin-top: var(--trusty-space-3);
  }
  .results {
    display: flex;
    flex-direction: column;
    gap: var(--trusty-space-3);
  }
  .result {
    background: var(--trusty-card-bg);
    border: 1px solid var(--trusty-border);
    border-radius: var(--trusty-radius);
    padding: var(--trusty-space-4);
  }
  .result-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: var(--trusty-space-3);
    gap: var(--trusty-space-3);
  }
  .result-path {
    display: flex;
    align-items: center;
    gap: var(--trusty-space-2);
    min-width: 0;
    flex: 1;
  }
  .result-score {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    flex-shrink: 0;
  }
  .score-label {
    font-size: var(--trusty-fs-xs);
    color: var(--trusty-text-muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }
  .score-value {
    font-family: var(--trusty-mono);
    font-weight: 600;
    color: var(--trusty-text-primary);
  }
  .snippet {
    margin: 0;
    padding: var(--trusty-space-3);
    background: var(--trusty-content-bg);
    border-radius: var(--trusty-radius);
    overflow-x: auto;
    white-space: pre-wrap;
    word-break: break-word;
    font-size: var(--trusty-fs-xs);
    color: var(--trusty-text-secondary);
    line-height: 1.5;
  }
</style>
