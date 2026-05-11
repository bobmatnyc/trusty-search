/*
 * Why: Centralised reactive state for daemon health and the index catalogue,
 * so multiple views don't refetch on every mount.
 * What: Exports plain getters/setters backed by Svelte 5 runes, plus refresh
 * helpers. The shapes are intentionally flat so views can `$derived(getX())`
 * directly.
 * Test: Mount two views, call refreshIndexes() in one, observe the other
 * update its derived counters without a manual refresh.
 */

import { api } from './api.js';

let _health = $state(null);
let _indexes = $state([]); // [{ id, chunk_count, root_path }]
let _loading = $state(false);
let _error = $state(null);

export function getHealth() {
  return _health;
}

export function getIndexes() {
  return _indexes;
}

export function getLoading() {
  return _loading;
}

export function getError() {
  return _error;
}

export async function refreshHealth() {
  try {
    _health = await api.health();
  } catch (e) {
    _health = { status: 'unreachable', version: '', indexes: 0, uptime_secs: 0 };
    _error = e.message || String(e);
  }
  return _health;
}

/**
 * Why: The /indexes endpoint only returns names; the admin UI wants chunk
 * counts and root paths for every index. We fan out per-index /status calls
 * in parallel and merge into a single array.
 * What: Refreshes `_indexes` to a list of `{ id, chunk_count, root_path }`.
 * Indexes whose status call fails are still included with `error: true`.
 * Test: Register two indexes, call refreshIndexes(), assert length === 2.
 */
export async function refreshIndexes() {
  _loading = true;
  _error = null;
  try {
    const body = await api.listIndexes();
    const names = body?.indexes || [];
    const pairs = await Promise.all(
      names.map(async (id) => {
        try {
          const s = await api.indexStatus(id);
          return {
            id,
            chunk_count: s.chunk_count ?? 0,
            root_path: s.root_path ?? '',
            error: false
          };
        } catch (_e) {
          return { id, chunk_count: 0, root_path: '', error: true };
        }
      })
    );
    _indexes = pairs;
  } catch (e) {
    _error = e.message || String(e);
    _indexes = [];
  } finally {
    _loading = false;
  }
  return _indexes;
}
