<script>
  /*
   * Why: Breadcrumb + daemon-status header that mirrors trusty-memory's
   * Topbar. The version badge doubles as a health indicator — green when
   * the daemon is reachable, muted while connecting.
   * What: Renders crumbs derived from the current route, then a right-side
   * cluster with the version badge.
   * Test: Navigate to /search, confirm crumb text reads "Search".
   */
  import { getHealth } from '../state.svelte.js';
  import { getRoute } from '../router.svelte.js';

  let health = $derived(getHealth());
  let route = $derived(getRoute());

  let crumbs = $derived.by(() => {
    const segs = route.segments;
    if (segs.length === 0) return ['Dashboard'];
    if (segs[0] === 'search') return ['Search'];
    if (segs[0] === 'indexes' || segs[0] === 'index') {
      const parts = ['Indexes'];
      if (segs.length > 1) parts.push(segs[1]);
      return parts;
    }
    if (segs[0] === 'config') return ['Config'];
    return ['Dashboard'];
  });

  let healthy = $derived(health && health.status === 'ok');
</script>

<header class="topbar">
  <div class="crumbs">
    {#each crumbs as crumb, i}
      {#if i > 0}<span class="sep">/</span>{/if}
      <span class="crumb">{crumb}</span>
    {/each}
  </div>
  <div class="actions">
    {#if health && healthy}
      <span class="badge badge-success">v{health.version || '?'}</span>
    {:else if health}
      <span class="badge badge-danger">unreachable</span>
    {:else}
      <span class="badge badge-muted">connecting…</span>
    {/if}
  </div>
</header>

<style>
  .topbar {
    height: var(--trusty-topbar-height);
    background: var(--trusty-card-bg);
    border-bottom: 1px solid var(--trusty-border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 var(--trusty-space-6);
    position: sticky;
    top: 0;
    z-index: 10;
  }
  .crumbs {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: var(--trusty-fs-sm);
    color: var(--trusty-text-secondary);
  }
  .crumb {
    font-weight: 500;
  }
  .crumb:last-child {
    color: var(--trusty-text-primary);
    font-weight: 600;
  }
  .sep {
    color: var(--trusty-text-muted);
  }
  .actions {
    display: flex;
    align-items: center;
    gap: 12px;
  }
</style>
