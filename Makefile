# trusty-search Makefile
#
# Why: `cargo publish` requires the embedded Svelte UI assets to live inside
# the crate-root `ui-dist/` (post-consolidation single-crate layout).
# The UI source lives at `ui/`; without an explicit sync step a developer who
# rebuilds the UI can easily publish stale assets. These targets make the sync
# step a single, well-named command.
# What: `build-ui` runs the Svelte/Vite build; `sync-ui` mirrors `ui/dist/`
# into `ui-dist/`; `release-prep` chains the two and is the documented
# prerequisite for `cargo publish`.
# Test: `make release-prep` populates `ui-dist/index.html` and the `assets/` tree.

UI_DIR      := ui
UI_DIST     := $(UI_DIR)/dist
UI_EMBED    := ui-dist

.PHONY: ui build-ui sync-ui release-prep install patch reinstall check clippy test

## Build Svelte UI (pnpm preferred, npm fallback)
build-ui:
	@if command -v pnpm >/dev/null 2>&1; then \
		echo ">> pnpm install && pnpm build (in $(UI_DIR))"; \
		cd $(UI_DIR) && pnpm install --frozen-lockfile && pnpm build; \
	else \
		echo ">> npm ci && npm run build (in $(UI_DIR))"; \
		cd $(UI_DIR) && npm ci && npm run build; \
	fi

## Sync ui/dist → ui-dist (no rebuild)
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

## Install binary from source, stopping any running daemon first (closes #87).
## Why: replacing the binary while the daemon is running causes macOS 26.3+ to
## SIGKILL the process with "Code Signature Invalid"; stopping first ensures a
## clean handoff.  `|| true` makes the stop a no-op when no daemon is running.
install:
	trusty-search stop 2>/dev/null || true
	sleep 1
	cargo install --path . --locked

## Bump the patch version, stop the daemon, install, and start it again.
## Why: same macOS binary-replacement hazard as `install`; version bump and
## daemon restart are always paired during development patch cycles.
patch:
	trusty-search stop 2>/dev/null || true
	cargo set-version --bump patch
	cargo install --path . --locked
	trusty-search start

## Stop daemon, install new binary from source, restart (closes #87)
## Why: replacing the binary while the daemon is running causes macOS to
## SIGKILL the process; stopping first ensures a clean handoff.
reinstall:
	trusty-search stop 2>/dev/null || true
	sleep 2
	cargo install --path . --locked
	trusty-search start

## Quick local quality gate
check:
	cargo check --workspace

clippy:
	cargo clippy --workspace --all-targets -- -D warnings

test:
	cargo test --workspace
