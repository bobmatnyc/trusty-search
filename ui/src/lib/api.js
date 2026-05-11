/*
 * Why: All UI components hit the same daemon REST surface; centralizing fetch
 * logic gives us one place to handle errors, base URL, and JSON parsing.
 * The daemon serves the bundle at /ui and the API at flat paths
 * (/health, /indexes, /search, ...) so requests are always same-origin in
 * production. In `vite dev`, vite.config.js proxies the API paths through to
 * 127.0.0.1:7878.
 * What: Thin wrappers returning parsed JSON or throwing on non-2xx.
 * Test: Console-call api.health() and confirm shape matches /health.
 */

async function request(path, opts = {}) {
  const res = await fetch(path, {
    headers: { 'Content-Type': 'application/json', ...(opts.headers || {}) },
    ...opts
  });
  if (!res.ok) {
    let detail = '';
    try {
      detail = await res.text();
    } catch {
      /* ignore */
    }
    throw new Error(`${res.status} ${res.statusText}: ${detail}`);
  }
  if (res.status === 204) return null;
  const ct = res.headers.get('content-type') || '';
  if (ct.includes('application/json')) return res.json();
  return res.text();
}

export const api = {
  health: () => request('/health'),

  listIndexes: () => request('/indexes'),
  createIndex: (id, root_path) =>
    request('/indexes', {
      method: 'POST',
      body: JSON.stringify({ id, root_path })
    }),
  deleteIndex: (id) =>
    request(`/indexes/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  indexStatus: (id) => request(`/indexes/${encodeURIComponent(id)}/status`),

  /** Per-index hybrid search. */
  search: (id, text, top_k = 10) =>
    request(`/indexes/${encodeURIComponent(id)}/search`, {
      method: 'POST',
      body: JSON.stringify({ text, top_k })
    }),

  /** Cross-collection fan-out search across every registered index. */
  globalSearch: (query, top_k = 10, full_content = false) =>
    request('/search', {
      method: 'POST',
      body: JSON.stringify({ query, top_k, full_content })
    }),

  reindex: (id, root_path) =>
    request(`/indexes/${encodeURIComponent(id)}/reindex`, {
      method: 'POST',
      body: JSON.stringify(root_path ? { root_path } : {})
    }),

  chat: (index_id, message, history = []) =>
    request('/chat', {
      method: 'POST',
      body: JSON.stringify({ index_id, message, history })
    })
};
