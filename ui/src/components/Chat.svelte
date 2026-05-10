<script>
  import { onMount } from 'svelte';
  import { indexes } from '../lib/stores.svelte.js';
  import { api } from '../lib/api.js';

  let selectedIndex = $state('');
  let input = $state('');
  let sending = $state(false);
  let history = $state([]); // { role: 'user' | 'assistant', content }
  let error = $state('');

  onMount(() => {
    indexes.refresh().then(() => {
      if (indexes.names.length > 0 && !selectedIndex) selectedIndex = indexes.names[0];
    });
  });

  async function send(e) {
    e?.preventDefault();
    if (!input.trim() || !selectedIndex) return;
    const message = input.trim();
    input = '';
    sending = true;
    error = '';
    history = [...history, { role: 'user', content: message }];
    try {
      const resp = await api.chat(selectedIndex, message, history.slice(0, -1));
      const reply = resp.reply ?? resp.message ?? resp.content ?? JSON.stringify(resp);
      history = [...history, { role: 'assistant', content: reply }];
    } catch (err) {
      error = err.message || String(err);
      history = [...history, { role: 'assistant', content: `[error] ${error}` }];
    } finally {
      sending = false;
    }
  }

  function clearChat() {
    history = [];
    error = '';
  }
</script>

<h1>Chat</h1>

<div class="card">
  <div class="row">
    <div class="field">
      <label for="chat-idx">Collection</label>
      <select id="chat-idx" bind:value={selectedIndex}>
        {#each indexes.names as id (id)}
          <option value={id}>{id}</option>
        {/each}
      </select>
    </div>
    <div class="shrink">
      <button class="ghost" onclick={clearChat}>Clear</button>
    </div>
  </div>

  <div class="chat-thread">
    {#if history.length === 0}
      <div class="empty">Ask a question — search results from <strong>{selectedIndex || '(none)'}</strong> will be passed to the model as context.</div>
    {/if}
    {#each history as msg, i (i)}
      <div class="chat-msg {msg.role}">
        <div class="role">{msg.role}</div>
        <div class="body">{msg.content}</div>
      </div>
    {/each}
  </div>

  <form onsubmit={send}>
    <div class="row">
      <div class="field" style="flex: 4;">
        <input type="text" bind:value={input} placeholder="Ask about the codebase…" disabled={sending} />
      </div>
      <div class="shrink">
        <button class="primary" type="submit" disabled={sending || !selectedIndex}>
          {#if sending}<span class="spinner"></span>{/if}
          Send
        </button>
      </div>
    </div>
  </form>
</div>
