#!/usr/bin/env bash
# publish.sh — Publish all trusty-* crates to crates.io in dependency order.
#
# PREREQUISITES:
#   1. cargo login <token> has been run
#   2. The crates.io account has a verified email address
#      (https://crates.io/settings/profile)
#   3. This script is run from the trusty-search workspace root
#   4. The trusty-common repo is available at /tmp/trusty-common-publish
#      (git clone https://github.com/bobmatnyc/trusty-common /tmp/trusty-common-publish)
#
# PUBLISH ORDER:
#   trusty-common  → trusty-embedder → trusty-mcp-core
#   → trusty-search-core → trusty-search-service → trusty-search-mcp
#   → trusty-search (root bin)

set -euo pipefail

WAIT=35  # seconds between publishes for crates.io index propagation

log() { echo "[$(date +%H:%M:%S)] $*"; }
wait_for_index() { log "Waiting ${WAIT}s for crates.io index propagation..."; sleep "$WAIT"; }

# ---------------------------------------------------------------------------
# 1. Shared crates from trusty-common
# ---------------------------------------------------------------------------

COMMON_DIR="/tmp/trusty-common-publish"
if [[ ! -d "$COMMON_DIR" ]]; then
  log "Cloning trusty-common..."
  git clone https://github.com/bobmatnyc/trusty-common "$COMMON_DIR"
fi

log "Publishing trusty-common v0.1.2..."
(cd "$COMMON_DIR" && cargo publish -p trusty-common)
wait_for_index

log "Publishing trusty-embedder v0.1.0..."
(cd "$COMMON_DIR" && cargo publish -p trusty-embedder)
wait_for_index

log "Publishing trusty-mcp-core v0.1.0..."
(cd "$COMMON_DIR" && cargo publish -p trusty-mcp-core)
wait_for_index

# ---------------------------------------------------------------------------
# 2. trusty-search workspace crates
#    The [patch] block in Cargo.toml must be removed so Cargo resolves the
#    shared crates from crates.io, not from a local path.
# ---------------------------------------------------------------------------

WORKSPACE_ROOT="$(cd "$(dirname "$0")" && pwd)"
CARGO_TOML="$WORKSPACE_ROOT/Cargo.toml"
CARGO_TOML_BACKUP="$WORKSPACE_ROOT/Cargo.toml.prepublish-backup"

log "Backing up Cargo.toml..."
cp "$CARGO_TOML" "$CARGO_TOML_BACKUP"

log "Removing [patch] section from Cargo.toml for crates.io publish..."
# Remove the [patch."https://github.com/bobmatnyc/trusty-common"] block
python3 - "$CARGO_TOML" <<'PYEOF'
import sys, re

path = sys.argv[1]
text = open(path).read()

# Remove the patch block and the comment above it
patched = re.sub(
    r'\n# Local development override.*?\n\[patch\."https://github\.com/bobmatnyc/trusty-common"\]\n(?:.*\n)*?(?=\n\[|\Z)',
    '\n',
    text,
    flags=re.MULTILINE
)
open(path, 'w').write(patched)
print("Patch section removed.")
PYEOF

log "Publishing trusty-search-core v0.1.7..."
cargo publish -p trusty-search-core
wait_for_index

log "Publishing trusty-search-service v0.1.7..."
cargo publish -p trusty-search-service
wait_for_index

log "Publishing trusty-search-mcp v0.1.7..."
cargo publish -p trusty-search-mcp
wait_for_index

log "Publishing trusty-search v0.1.57..."
cargo publish -p trusty-search

log "Restoring Cargo.toml with [patch] section..."
mv "$CARGO_TOML_BACKUP" "$CARGO_TOML"

log "Done! Published crates:"
log "  https://crates.io/crates/trusty-common"
log "  https://crates.io/crates/trusty-embedder"
log "  https://crates.io/crates/trusty-mcp-core"
log "  https://crates.io/crates/trusty-search-core"
log "  https://crates.io/crates/trusty-search-service"
log "  https://crates.io/crates/trusty-search-mcp"
log "  https://crates.io/crates/trusty-search"
