<script>
  import { onMount } from 'svelte';
  import { health, indexes } from '../lib/stores.svelte.js';
  import { api } from '../lib/api.js';

  let selectedIndex = $state('');
  let filePath = $state('');
  let busy = $state(false);
  let msg = $state('');

  onMount(() => {
    indexes.refresh().then(() => {
      if (!selectedIndex && indexes.names.length > 0) selectedIndex = indexes.names[0];
    });
  });

  function port() {
    return typeof window !== 'undefined' ? window.__DAEMON_PORT__ : '?';
  }

  async function indexOne() {
    if (!selectedIndex || !filePath.trim()) return;
    busy = true; msg = '';
    try {
      // Server expects {path, content}; for one-shot indexing the content
      // can be empty — the daemon will read the file off disk via the watcher
      // pipeline. If the daemon requires content here, the user should use
      // the watcher instead.
      const r = await api.indexFile(selectedIndex, filePath.trim(), '');
      msg = `Indexed: ${r.path || filePath}`;
    } catch (e) { msg = `Failed: ${e.message}`; }
    finally { busy = false; }
  }

  async function removeOne() {
    if (!selectedIndex || !filePath.trim()) return;
    busy = true; msg = '';
    try {
      const r = await api.removeFile(selectedIndex, filePath.trim());
      msg = `Removed ${r.removed_chunks ?? '?'} chunks for ${r.path || filePath}`;
    } catch (e) { msg = `Failed: ${e.message}`; }
    finally { busy = false; }
  }

  async function deleteAll() {
    if (!confirm(`Delete ALL ${indexes.names.length} indexes? This cannot be undone.`)) return;
    busy = true; msg = '';
    let ok = 0, err = 0;
    for (const id of indexes.names) {
      try { await api.deleteIndex(id); ok++; } catch (_e) { err++; }
    }
    await indexes.refresh();
    msg = `Deleted ${ok} indexes (${err} errors)`;
    busy = false;
  }
</script>

<h1>Admin</h1>

<div class="card">
  <h2>Daemon</h2>
  <table>
    <tbody>
      <tr><th style="width: 200px;">Status</th><td>{health.online ? 'running' : 'unreachable'}</td></tr>
      <tr><th>Version</th><td class="mono">{health.value.version || '—'}</td></tr>
      <tr><th>Port</th><td class="mono">{port()}</td></tr>
      <tr><th>Indexes</th><td>{indexes.names.length}</td></tr>
    </tbody>
  </table>
</div>

<div class="card">
  <h2>Per-File Operations</h2>
  <p class="card-sub">Add or remove a single file from a specific index.</p>
  <div class="row">
    <div class="field">
      <label for="adm-idx">Index</label>
      <select id="adm-idx" bind:value={selectedIndex}>
        {#each indexes.names as id (id)}<option>{id}</option>{/each}
      </select>
    </div>
    <div class="field" style="flex: 3;">
      <label for="adm-file">File path</label>
      <input id="adm-file" type="text" bind:value={filePath} placeholder="/abs/path/to/file.rs" />
    </div>
    <div class="shrink">
      <button onclick={indexOne} disabled={busy || !selectedIndex || !filePath}>Index file</button>
    </div>
    <div class="shrink">
      <button class="danger" onclick={removeOne} disabled={busy || !selectedIndex || !filePath}>Remove file</button>
    </div>
  </div>
  {#if msg}<div class="muted" style="margin-top: 8px;">{msg}</div>{/if}
</div>

<div class="card danger-zone">
  <h2 style="color: var(--danger);">Danger Zone</h2>
  <p class="card-sub">Irreversible operations.</p>
  <button class="danger" onclick={deleteAll} disabled={busy || indexes.names.length === 0}>
    Delete all indexes ({indexes.names.length})
  </button>
</div>
