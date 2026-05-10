// Why: Centralize fetch wrappers so components never hardcode URLs.
// What: Resolves the daemon base URL from window.__DAEMON_PORT__ (injected by
// the Rust server) and exposes typed-ish helpers for every endpoint we use.
// Test: in dev, vite proxy forwards /indexes etc. to 127.0.0.1:7878; in
// production the same-origin URL hits the embedded server directly.

function baseUrl() {
  // When served by the daemon, all requests are same-origin.
  // When running `vite dev`, vite.config.js proxies /indexes → daemon.
  // Both cases: empty string base works.
  return '';
}

async function request(path, opts = {}) {
  const url = baseUrl() + path;
  const headers = { 'Content-Type': 'application/json', ...(opts.headers || {}) };
  const resp = await fetch(url, { ...opts, headers });
  if (!resp.ok) {
    const text = await resp.text().catch(() => '');
    throw new Error(`${opts.method || 'GET'} ${path} → ${resp.status}: ${text}`);
  }
  // Accept empty bodies (e.g. some POSTs).
  const ct = resp.headers.get('content-type') || '';
  if (ct.includes('application/json')) return resp.json();
  return resp.text();
}

export const api = {
  health: () => request('/health'),

  listIndexes: () => request('/indexes'),

  createIndex: (id, root_path) =>
    request('/indexes', {
      method: 'POST',
      body: JSON.stringify({ id, root_path }),
    }),

  deleteIndex: (id) =>
    request(`/indexes/${encodeURIComponent(id)}`, { method: 'DELETE' }),

  indexStatus: (id) => request(`/indexes/${encodeURIComponent(id)}/status`),

  search: (id, text, top_k = 10) =>
    request(`/indexes/${encodeURIComponent(id)}/search`, {
      method: 'POST',
      body: JSON.stringify({ text, top_k }),
    }),

  indexFile: (id, path, content = '') =>
    request(`/indexes/${encodeURIComponent(id)}/index-file`, {
      method: 'POST',
      body: JSON.stringify({ path, content }),
    }),

  removeFile: (id, path) =>
    request(`/indexes/${encodeURIComponent(id)}/remove-file`, {
      method: 'POST',
      body: JSON.stringify({ path }),
    }),

  reindex: (id, root_path) =>
    request(`/indexes/${encodeURIComponent(id)}/reindex`, {
      method: 'POST',
      body: JSON.stringify(root_path ? { root_path } : {}),
    }),

  chat: (index_id, message, history) =>
    request('/chat', {
      method: 'POST',
      body: JSON.stringify({ index_id, message, history }),
    }),
};

// Why: Cross-collection search runs N parallel /search calls and merges the
// returned ranked lists by score (descending).
// What: Returns a flat list with `{ ...result, _index_id }` so the UI can
// label each row with its source index.
// Test: with two indexes, results from both appear sorted by score.
export async function searchAcross(indexIds, text, top_k = 10) {
  const all = await Promise.all(
    indexIds.map(async (id) => {
      try {
        const body = await api.search(id, text, top_k);
        return (body.results || []).map((r) => ({ ...r, _index_id: id }));
      } catch (e) {
        console.error('search failed for', id, e);
        return [];
      }
    })
  );
  const merged = all.flat();
  merged.sort((a, b) => (b.score || 0) - (a.score || 0));
  return merged.slice(0, top_k);
}
