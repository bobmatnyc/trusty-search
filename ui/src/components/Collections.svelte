<script>
  import { onMount, onDestroy } from 'svelte';
  import { indexes } from '../lib/stores.svelte.js';
  import { api } from '../lib/api.js';

  let newId = $state('');
  let newPath = $state('');
  let creating = $state(false);
  let createMsg = $state('');
  let busyIds = $state(new Set());
  let pollTimer = null;

  // Poll statuses every 2s so reindex progress is visible without manual refresh.
  // Why: index_status doesn't currently expose `indexing_in_progress`, so we
  // approximate by re-fetching status while any reindex was started in this
  // session. We always refresh anyway — it's cheap.
  function startPolling() {
    if (pollTimer) return;
    pollTimer = setInterval(() => {
      indexes.refresh();
    }, 2000);
  }
  function stopPolling() {
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
  }

  onMount(() => { indexes.refresh(); });
  onDestroy(() => stopPolling());

  async function handleCreate(e) {
    e.preventDefault();
    if (!newId.trim() || !newPath.trim()) return;
    creating = true;
    createMsg = '';
    try {
      const r = await api.createIndex(newId.trim(), newPath.trim());
      createMsg = r.created ? `Created '${r.id}'` : `'${r.id}' already exists`;
      newId = '';
      newPath = '';
      await indexes.refresh();
    } catch (err) {
      createMsg = `Failed: ${err.message}`;
    } finally {
      creating = false;
    }
  }

  async function handleReindex(id) {
    busyIds = new Set([...busyIds, id]);
    try {
      await api.reindex(id);
      startPolling();
      // Stop polling for this index after 60s as a safety net.
      setTimeout(() => {
        busyIds = new Set([...busyIds].filter((x) => x !== id));
        if (busyIds.size === 0) stopPolling();
      }, 60_000);
    } catch (err) {
      alert(`Reindex failed: ${err.message}`);
      busyIds = new Set([...busyIds].filter((x) => x !== id));
    }
  }

  async function handleDelete(id) {
    if (!confirm(`Delete index '${id}'? This cannot be undone.`)) return;
    try {
      await api.deleteIndex(id);
      await indexes.refresh();
    } catch (err) {
      alert(`Delete failed: ${err.message}`);
    }
  }

  function fmtPath(p) {
    if (!p) return '—';
    if (typeof p === 'string') return p;
    return JSON.stringify(p);
  }
</script>

<h1>Collections</h1>

<div class="card">
  <h2>Add Collection</h2>
  <p class="card-sub">Register a new index with the daemon.</p>
  <form onsubmit={handleCreate}>
    <div class="row">
      <div class="field">
        <label for="new-id">Index name</label>
        <input id="new-id" type="text" bind:value={newId} placeholder="myproject" required />
      </div>
      <div class="field">
        <label for="new-path">Root path</label>
        <input id="new-path" type="text" bind:value={newPath} placeholder="/Users/me/Projects/myproject" required />
      </div>
      <div class="shrink">
        <button class="primary" type="submit" disabled={creating}>
          {#if creating}<span class="spinner"></span>{/if}
          Add
        </button>
      </div>
    </div>
    {#if createMsg}
      <div class="muted" style="margin-top: 8px;">{createMsg}</div>
    {/if}
  </form>
</div>

<div class="card">
  <div style="display: flex; justify-content: space-between; align-items: center;">
    <h2 style="margin: 0;">Registered Collections</h2>
    <button class="ghost" onclick={() => indexes.refresh()}>↻ Refresh</button>
  </div>

  {#if indexes.loading && indexes.names.length === 0}
    <div class="empty"><span class="spinner"></span> Loading…</div>
  {:else if indexes.error}
    <div class="empty">Error: {indexes.error}</div>
  {:else if indexes.names.length === 0}
    <div class="empty">No collections registered. Add one above.</div>
  {:else}
    <table>
      <thead>
        <tr>
          <th>Name</th>
          <th>Root Path</th>
          <th style="text-align: right;">Chunks</th>
          <th style="text-align: right;">Files</th>
          <th>Last Updated</th>
          <th style="text-align: right;">Actions</th>
        </tr>
      </thead>
      <tbody>
        {#each indexes.names as id (id)}
          {@const status = indexes.statuses[id] || {}}
          <tr>
            <td><strong>{id}</strong></td>
            <td class="mono muted" style="font-size: 12px;">{fmtPath(status.root_path)}</td>
            <td style="text-align: right;" class="mono">{status.chunk_count ?? '—'}</td>
            <td style="text-align: right;" class="mono">{status.file_count ?? '—'}</td>
            <td class="muted" style="font-size: 12px;">{status.last_updated ?? '—'}</td>
            <td style="text-align: right; white-space: nowrap;">
              <button onclick={() => handleReindex(id)} disabled={busyIds.has(id)}>
                {#if busyIds.has(id)}<span class="spinner"></span>{/if}
                Reindex
              </button>
              <button class="danger" onclick={() => handleDelete(id)}>Delete</button>
            </td>
          </tr>
        {/each}
      </tbody>
    </table>
  {/if}
</div>
