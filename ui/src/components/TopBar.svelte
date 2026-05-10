<script>
  import { health } from '../lib/stores.svelte.js';
</script>

<header class="topbar">
  <div class="brand">
    <span class="logo">🔍</span>
    <span>trusty-search</span>
    <span class="version">v{health.value.version || '?'}</span>
  </div>
  <div class="health">
    <span class="dot" class:ok={health.online} class:bad={!health.online && health.lastChecked > 0}></span>
    <span>
      {#if health.online}
        Daemon healthy
      {:else if health.lastChecked === 0}
        Connecting…
      {:else}
        Daemon unreachable
      {/if}
    </span>
  </div>
</header>
