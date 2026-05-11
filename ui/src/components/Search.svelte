<script>
  import { onMount } from 'svelte';
  import { indexes } from '../lib/stores.svelte.js';
  import { api, searchAcross } from '../lib/api.js';

  let selectedIndex = $state('__all__');
  let query = $state('');
  let topK = $state(10);
  let results = $state([]);
  let loading = $state(false);
  let error = $state('');
  let intent = $state('');
  let latencyMs = $state(0);
  let showFull = $state(false);

  onMount(() => {
    indexes.refresh().then(() => {
      // Auto-select first index if there's only one.
      if (indexes.names.length === 1) selectedIndex = indexes.names[0];
    });
  });

  async function runSearch(e) {
    e?.preventDefault();
    if (!query.trim()) return;
    loading = true;
    error = '';
    results = [];
    try {
      if (selectedIndex === '__all__') {
        const t0 = performance.now();
        const merged = await searchAcross(indexes.names, query, topK);
        latencyMs = Math.round(performance.now() - t0);
        results = merged;
        intent = '(cross-collection)';
      } else {
        const body = await api.search(selectedIndex, query, topK);
        results = (body.results || []).map((r) => ({ ...r, _index_id: selectedIndex }));
        intent = body.intent || '';
        latencyMs = body.latency_ms || 0;
      }
    } catch (err) {
      error = err.message || String(err);
    } finally {
      loading = false;
    }
  }

  function reasonClass(reason) {
    if (!reason) return 'unknown';
    if (reason.includes('hybrid')) return 'hybrid';
    if (reason.includes('bm25')) return 'bm25';
    if (reason.includes('vector')) return 'vector';
    if (reason.includes('fallback')) return 'fallback';
    return 'unknown';
  }
</script>

<h1>Search</h1>

<div class="card">
  <form onsubmit={runSearch}>
    <div class="row">
      <div class="field">
        <label for="idx">Collection</label>
        <select id="idx" bind:value={selectedIndex}>
          <option value="__all__">All Collections</option>
          {#each indexes.names as id (id)}
            <option value={id}>{id}</option>
          {/each}
        </select>
      </div>
      <div class="field" style="flex: 3;">
        <label for="q">Query</label>
        <input id="q" type="text" bind:value={query} placeholder="fn authenticate, error handling, …" />
      </div>
      <div class="field shrink" style="min-width: 160px;">
        <label for="k">Top-K: {topK}</label>
        <input id="k" type="range" min="5" max="50" bind:value={topK} />
      </div>
      <div class="shrink">
        <button class="primary" type="submit" disabled={loading}>
          {#if loading}<span class="spinner"></span>{/if}
          Search
        </button>
      </div>
    </div>
    <div style="margin-top: 8px;">
      <label style="font-size: 12px; color: var(--trusty-text-muted);">
        <input type="checkbox" bind:checked={showFull} /> Show full chunk
      </label>
    </div>
  </form>
</div>

{#if error}
  <div class="card" style="border-color: var(--trusty-danger); color: var(--trusty-danger);">
    {error}
  </div>
{/if}

{#if results.length > 0 || (loading === false && query)}
  <div class="card">
    <h2>
      Results
      <span class="muted" style="font-weight: normal; font-size: 12px;">
        — {results.length} hits
        {#if intent} · intent: {intent}{/if}
        {#if latencyMs} · {latencyMs}ms{/if}
      </span>
    </h2>
    {#if results.length === 0 && !loading}
      <div class="empty">No matches.</div>
    {/if}
    {#each results as r, i (r.id || `${r.file}:${r.start_line}:${i}`)}
      <div class="result-item">
        <div class="result-head">
          <span class="result-file">{r.file}:{r.start_line}–{r.end_line}</span>
          <span class="badge {reasonClass(r.match_reason)}">{r.match_reason || '?'}</span>
          {#if r._index_id && selectedIndex === '__all__'}
            <span class="badge unknown">{r._index_id}</span>
          {/if}
          <span class="result-score mono">score {(r.score ?? 0).toFixed(3)}</span>
        </div>
        <pre class="snippet">{showFull ? (r.content || '') : (r.compact_snippet || r.content || '')}</pre>
      </div>
    {/each}
  </div>
{/if}
