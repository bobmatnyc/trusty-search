<script>
  let { view, openrouterEnabled, onnavigate } = $props();

  const items = $derived([
    { id: 'collections', label: 'Collections', icon: '📚' },
    { id: 'search', label: 'Search', icon: '🔎' },
    ...(openrouterEnabled ? [{ id: 'chat', label: 'Chat', icon: '💬' }] : []),
    { id: 'admin', label: 'Admin', icon: '⚙️' },
  ]);
</script>

<nav class="sidebar">
  {#each items as item (item.id)}
    <div
      class="nav-item"
      class:active={view === item.id}
      onclick={() => onnavigate(item.id)}
      role="button"
      tabindex="0"
      onkeydown={(e) => { if (e.key === 'Enter') onnavigate(item.id); }}
    >
      <span class="nav-icon">{item.icon}</span>
      <span>{item.label}</span>
    </div>
  {/each}
</nav>
