<script>
  /*
   * Why: Operators need a single pane to verify the daemon's configuration —
   * OpenRouter availability, the listening port, uptime, version, and the
   * total chunk count across all indexes. Without this view, those signals are
   * scattered across CLI subcommands.
   * What: Read-only stat grid + key/value table reflecting whatever the
   * daemon advertised via /health and the injected window globals.
   * Test: Open #/config, confirm the version cell matches GET /health.version
   * and the OpenRouter row reads "enabled" when OPENROUTER_API_KEY is set.
   */
  import { getHealth, getIndexes } from '../state.svelte.js';

  let health = $derived(getHealth());
  let indexes = $derived(getIndexes());

  let totalChunks = $derived(
    indexes.reduce((sum, ix) => sum + (ix.chunk_count || 0), 0)
  );

  let openrouterEnabled = $derived(
    typeof window !== 'undefined' && !!window.__OPENROUTER_ENABLED__
  );

  let daemonPort = $derived(
    (typeof window !== 'undefined' && window.__DAEMON_PORT__) || null
  );

  /**
   * Why: Raw seconds are unfriendly; reuse the same humaniser shape used on
   * the Dashboard so both panes agree.
   * What: Returns "Xs / Xm / Xh / Xd" or "—" when not a number.
   * Test: humanUptime(3600) === "1h".
   */
  function humanUptime(secs) {
    if (typeof secs !== 'number' || secs < 0) return '—';
    if (secs < 60) return `${secs}s`;
    const m = Math.floor(secs / 60);
    if (m < 60) return `${m}m`;
    const h = Math.floor(m / 60);
    if (h < 24) return `${h}h`;
    const d = Math.floor(h / 24);
    return `${d}d`;
  }
</script>

<h1 class="page-title">Configuration</h1>

<div class="stat-grid">
  <div class="stat">
    <div class="stat-label">Status</div>
    <div class="stat-value" style="font-size: var(--trusty-fs-lg)">
      {#if health?.status === 'ok'}
        <span class="badge badge-success">healthy</span>
      {:else}
        <span class="badge badge-danger">offline</span>
      {/if}
    </div>
    <div class="stat-meta">daemon</div>
  </div>
  <div class="stat">
    <div class="stat-label">Uptime</div>
    <div class="stat-value">{humanUptime(health?.uptime_secs)}</div>
    <div class="stat-meta">since start</div>
  </div>
  <div class="stat">
    <div class="stat-label">Version</div>
    <div class="stat-value text-mono" style="font-size: var(--trusty-fs-lg)">
      {health?.version || '—'}
    </div>
    <div class="stat-meta">trusty-search</div>
  </div>
  <div class="stat">
    <div class="stat-label">Chunks</div>
    <div class="stat-value">{totalChunks.toLocaleString()}</div>
    <div class="stat-meta">across {indexes.length} index{indexes.length === 1 ? '' : 'es'}</div>
  </div>
</div>

<div class="card">
  <div class="card-header">Daemon details</div>
  <div class="card-body" style="padding: 0">
    <table class="table">
      <tbody>
        <tr>
          <th style="width: 240px">OpenRouter chat</th>
          <td>
            {#if openrouterEnabled}
              <span class="badge badge-success">enabled</span>
              <span class="text-muted text-sm">OPENROUTER_API_KEY detected</span>
            {:else}
              <span class="badge badge-muted">disabled</span>
              <span class="text-muted text-sm"
                >Set <code>OPENROUTER_API_KEY</code> and restart the daemon to
                enable <code>/chat</code>.</span
              >
            {/if}
          </td>
        </tr>
        <tr>
          <th>Daemon port</th>
          <td class="text-mono">{daemonPort ?? '—'}</td>
        </tr>
        <tr>
          <th>API base URL</th>
          <td class="text-mono">{typeof window !== 'undefined' ? window.location.origin : '—'}</td>
        </tr>
        <tr>
          <th>Indexes registered</th>
          <td>{indexes.length}</td>
        </tr>
        <tr>
          <th>Total chunks</th>
          <td>{totalChunks.toLocaleString()}</td>
        </tr>
        <tr>
          <th>Data directory</th>
          <td class="text-muted text-sm">
            Managed by the daemon — see <code>trusty-search doctor</code> for the
            resolved path on this machine.
          </td>
        </tr>
      </tbody>
    </table>
  </div>
</div>

<style>
  .page-title {
    font-size: var(--trusty-fs-xl);
    margin: 0 0 var(--trusty-space-5) 0;
    font-weight: 600;
  }
  .table th {
    background: var(--trusty-content-bg);
    text-transform: none;
    letter-spacing: 0;
    font-size: var(--trusty-fs-sm);
    color: var(--trusty-text-secondary);
  }
</style>
