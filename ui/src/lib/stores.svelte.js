// Why: Svelte 5 runes can't be exported as top-level `let` declarations from
// a plain .js module — `$state` only works inside .svelte / .svelte.js files.
// What: This file is `.svelte.js`-equivalent — it uses runes by being imported
// from .svelte components. We export factory closures that hold runes-backed
// state and getter accessors.
// Test: components call `health.refresh()` and read `health.value` reactively.

import { api } from './api.js';

function createHealthStore() {
  let value = $state({ status: 'unknown', version: '' });
  let online = $state(false);
  let lastChecked = $state(0);

  async function refresh() {
    try {
      const h = await api.health();
      value = h;
      online = h.status === 'ok';
    } catch (_e) {
      online = false;
      value = { status: 'unreachable', version: '' };
    }
    lastChecked = Date.now();
  }

  return {
    get value() { return value; },
    get online() { return online; },
    get lastChecked() { return lastChecked; },
    refresh,
  };
}

function createIndexesStore() {
  let names = $state([]);
  let statuses = $state({}); // { id: { chunk_count, root_path, ... } }
  let loading = $state(false);
  let error = $state(null);

  async function refresh() {
    loading = true;
    error = null;
    try {
      const body = await api.listIndexes();
      names = body.indexes || [];
      // Fetch statuses in parallel.
      const pairs = await Promise.all(
        names.map(async (id) => {
          try {
            const s = await api.indexStatus(id);
            return [id, s];
          } catch (_e) {
            return [id, { error: true }];
          }
        })
      );
      const next = {};
      for (const [id, s] of pairs) next[id] = s;
      statuses = next;
    } catch (e) {
      error = e.message || String(e);
      names = [];
      statuses = {};
    } finally {
      loading = false;
    }
  }

  return {
    get names() { return names; },
    get statuses() { return statuses; },
    get loading() { return loading; },
    get error() { return error; },
    refresh,
  };
}

export const health = createHealthStore();
export const indexes = createIndexesStore();
