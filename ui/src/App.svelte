<script>
  import { onMount } from 'svelte';
  import TopBar from './components/TopBar.svelte';
  import Sidebar from './components/Sidebar.svelte';
  import Collections from './components/Collections.svelte';
  import Search from './components/Search.svelte';
  import Chat from './components/Chat.svelte';
  import Admin from './components/Admin.svelte';
  import { health, indexes } from './lib/stores.svelte.js';

  // Why: Single-page navigation without a router — admin UIs are tiny and a
  // hash-based route avoids pulling in svelte-spa-router.
  // What: `view` is one of collections | search | chat | admin.
  // Test: Click sidebar nav items → main panel swaps without reload.
  let view = $state('collections');

  function setHashFromView(v) {
    if (typeof window !== 'undefined') {
      window.location.hash = v;
    }
  }
  function viewFromHash() {
    if (typeof window === 'undefined') return 'collections';
    const h = (window.location.hash || '').replace(/^#/, '');
    if (['collections', 'search', 'chat', 'admin'].includes(h)) return h;
    return 'collections';
  }

  function navigate(v) {
    view = v;
    setHashFromView(v);
  }

  onMount(() => {
    view = viewFromHash();
    health.refresh();
    indexes.refresh();
    // Poll daemon health every 10s — cheap and keeps the badge live.
    const t = setInterval(() => health.refresh(), 10_000);
    const onHash = () => { view = viewFromHash(); };
    window.addEventListener('hashchange', onHash);
    return () => {
      clearInterval(t);
      window.removeEventListener('hashchange', onHash);
    };
  });

  let openrouterEnabled = $derived(
    typeof window !== 'undefined' && window.__OPENROUTER_ENABLED__ === true
  );
</script>

<div class="app-shell">
  <TopBar />
  <Sidebar {view} {openrouterEnabled} onnavigate={navigate} />
  <main class="main">
    {#if view === 'collections'}
      <Collections />
    {:else if view === 'search'}
      <Search />
    {:else if view === 'chat' && openrouterEnabled}
      <Chat />
    {:else if view === 'admin'}
      <Admin />
    {:else}
      <Collections />
    {/if}
  </main>
</div>
