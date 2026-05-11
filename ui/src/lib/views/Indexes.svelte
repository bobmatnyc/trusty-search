<script>
  /*
   * Why: The Indexes view is the admin pane for the daemon — list every
   * registered collection, register a new one, kick off a reindex, or evict
   * a stale one. All actions hit the existing flat REST surface.
   * What: Header card with a create form, then a table of every index
   * with per-row Reindex / Delete buttons.
   * Test: Click "Create", enter `demo` + a path, confirm the row appears
   * and Reindex returns a `queued: true` toast.
   */
  import { api } from '../api.js';
  import { getIndexes, getLoading, getError, refreshIndexes } from '../state.svelte.js';

  let indexes = $derived(getIndexes());
  let loading = $derived(getLoading());
  let error = $derived(getError());

  let newId = $state('');
  let newPath = $state('');
  let creating = $state(false);
  let createError = $state(null);
  let rowError = $state(null);
  let busyId = $state(null);

  async function createIndex(e) {
    e?.preventDefault?.();
    if (!newId.trim() || !newPath.trim()) return;
    creating = true;
    createError = null;
    try {
      await api.createIndex(newId.trim(), newPath.trim());
      newId = '';
      newPath = '';
      await refreshIndexes();
    } catch (err) {
      createError = err.message || String(err);
    } finally {
      creating = false;
    }
  }

  async function reindex(id) {
    rowError = null;
    busyId = id;
    try {
      await api.reindex(id);
      await refreshIndexes();
    } catch (err) {
      rowError = `Reindex ${id}: ${err.message || err}`;
    } finally {
      busyId = null;
    }
  }

  async function remove(id) {
    if (!confirm(`Delete index "${id}"? On-disk data is preserved.`)) return;
    rowError = null;
    busyId = id;
    try {
      await api.deleteIndex(id);
      await refreshIndexes();
    } catch (err) {
      rowError = `Delete ${id}: ${err.message || err}`;
    } finally {
      busyId = null;
    }
  }
</script>

<h1 class="page-title">Indexes</h1>

<div class="card mb-4">
  <div class="card-header">Register a new index</div>
  <div class="card-body">
    <form class="create-row" onsubmit={createIndex}>
      <input
        type="text"
        class="input"
        placeholder="Index id (e.g. my-project)"
        bind:value={newId}
      />
      <input
        type="text"
        class="input"
        placeholder="Absolute root path (e.g. /Users/me/code/my-project)"
        bind:value={newPath}
      />
      <button
        type="submit"
        class="btn btn-primary"
        disabled={creating || !newId.trim() || !newPath.trim()}
      >
        {creating ? 'Creating…' : 'Create'}
      </button>
    </form>
    {#if createError}
      <p class="text-sm mt-3" style="color: var(--trusty-danger)">{createError}</p>
    {/if}
  </div>
</div>

<div class="card">
  <div class="card-header flex-between">
    <span>Registered indexes</span>
    <button class="btn btn-sm" onclick={refreshIndexes} disabled={loading}>
      {loading ? 'Refreshing…' : 'Refresh'}
    </button>
  </div>
  <div class="card-body" style="padding: 0">
    {#if rowError}
      <div class="row-error">{rowError}</div>
    {/if}
    {#if error}
      <div class="row-error">{error}</div>
    {/if}
    {#if indexes.length === 0}
      <div class="empty">
        {#if loading}
          Loading…
        {:else}
          No indexes registered yet.
        {/if}
      </div>
    {:else}
      <table class="table">
        <thead>
          <tr>
            <th>Name</th>
            <th>Documents</th>
            <th>Root path</th>
            <th>Status</th>
            <th style="width: 220px; text-align: right">Actions</th>
          </tr>
        </thead>
        <tbody>
          {#each indexes as ix (ix.id)}
            <tr>
              <td><strong>{ix.id}</strong></td>
              <td>{(ix.chunk_count ?? 0).toLocaleString()}</td>
              <td class="text-mono text-xs text-muted truncate" style="max-width: 360px">
                {ix.root_path || '—'}
              </td>
              <td>
                {#if ix.error}
                  <span class="badge badge-danger">error</span>
                {:else}
                  <span class="badge badge-success">ready</span>
                {/if}
              </td>
              <td style="text-align: right">
                <button
                  class="btn btn-sm"
                  disabled={busyId === ix.id}
                  onclick={() => reindex(ix.id)}
                >
                  {busyId === ix.id ? 'Working…' : 'Reindex'}
                </button>
                <button
                  class="btn btn-sm btn-danger"
                  disabled={busyId === ix.id}
                  onclick={() => remove(ix.id)}
                >
                  Delete
                </button>
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
    {/if}
  </div>
</div>

<style>
  .page-title {
    font-size: var(--trusty-fs-xl);
    margin: 0 0 var(--trusty-space-5) 0;
    font-weight: 600;
  }
  .create-row {
    display: grid;
    grid-template-columns: 1fr 2fr auto;
    gap: var(--trusty-space-2);
    align-items: stretch;
  }
  @media (max-width: 720px) {
    .create-row {
      grid-template-columns: 1fr;
    }
  }
  .row-error {
    padding: var(--trusty-space-3) var(--trusty-space-5);
    color: var(--trusty-danger);
    background: var(--trusty-danger-soft);
    border-bottom: 1px solid var(--trusty-border);
    font-size: var(--trusty-fs-sm);
  }
</style>
