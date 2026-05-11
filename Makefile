# trusty-search Makefile
#
# Why: `cargo publish` requires the embedded Svelte UI assets to live inside
# the `trusty-search-service` crate at `crates/trusty-search-service/ui-dist/`.
# The UI source lives at `ui/` (workspace root); without an explicit sync step
# a developer who rebuilds the UI can easily publish stale assets. These
# targets make the sync step a single, well-named command.
# What: `build-ui` runs the Svelte/Vite build; `sync-ui` mirrors `ui/dist/`
# into the service crate's `ui-dist/`; `release-prep` chains the two and is
# the documented prerequisite for `cargo publish`.
# Test: `make release-prep` populates
# `crates/trusty-search-service/ui-dist/index.html` and the `assets/` tree.

UI_DIR      := ui
UI_DIST     := $(UI_DIR)/dist
UI_EMBED    := crates/trusty-search-service/ui-dist

.PHONY: ui build-ui sync-ui release-prep check clippy test

## Build Svelte UI (pnpm preferred, npm fallback)
build-ui:
	@if command -v pnpm >/dev/null 2>&1; then \
		echo ">> pnpm install && pnpm build (in $(UI_DIR))"; \
		cd $(UI_DIR) && pnpm install --frozen-lockfile && pnpm build; \
	else \
		echo ">> npm ci && npm run build (in $(UI_DIR))"; \
		cd $(UI_DIR) && npm ci && npm run build; \
	fi

## Sync ui/dist → crates/trusty-search-service/ui-dist (no rebuild)
sync-ui:
	@test -d $(UI_DIST) || (echo "ERROR: $(UI_DIST) missing — run 'make build-ui' first" && exit 1)
	rm -rf $(UI_EMBED)
	cp -r $(UI_DIST) $(UI_EMBED)
	@echo ">> synced $(UI_DIST) → $(UI_EMBED)"

## Build UI + sync (required before cargo publish)
release-prep: build-ui sync-ui
	@echo ">> ui-dist synced. Ready for cargo publish."

## Convenience alias
ui: build-ui

## Quick local quality gate
check:
	cargo check --workspace

clippy:
	cargo clippy --workspace --all-targets -- -D warnings

test:
	cargo test --workspace
