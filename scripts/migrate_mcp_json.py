#!/usr/bin/env python3
"""Migrate `.mcp.json` files from mcp-vector-search to trusty-search.

Why: ~62 projects on this machine reference the legacy `mcp-vector-search`
binary in their `.mcp.json` server entries. We want them to use the new
`trusty-search serve` stdio MCP server while preserving the existing
`"mcp-vector-search"` JSON key so skills calling `mcp__mcp-vector-search__*`
tools continue to work unchanged.

What: Walks the filesystem under --root, finds every `.mcp.json` that has
a `"mcp-vector-search"` entry, computes the index_id from the sibling
`.mcp-vector-search/config.json` (or directory basename as fallback), and
rewrites the entry to spawn `trusty-search serve` with `TRUSTY_INDEX` set.
Defaults to dry-run; use --apply to actually write changes.

Test: Create a temp dir with a `.mcp.json` containing an old mcp-vector-search
entry and a `.mcp-vector-search/config.json` with `{"project_root": "/x/foo"}`;
run with --root <tempdir> --apply; assert the rewritten `.mcp.json` has
`command: "trusty-search"`, `args: ["serve"]`, `env.TRUSTY_INDEX: "foo"`,
and the JSON key remains `"mcp-vector-search"`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterator

SKIP_DIRS = {".git", "node_modules", ".cargo", ".venv", "venv", "target", "__pycache__"}
MAX_DEPTH = 6
MCP_KEY = "mcp-vector-search"
TRUSTY_COMMAND = "trusty-search"
TRUSTY_ARGS = ["serve"]


def find_mcp_json_files(root: Path, max_depth: int = MAX_DEPTH) -> Iterator[Path]:
    """Why: avoid scanning unbounded depths and skip noisy vendored dirs.
    What: yields every `.mcp.json` under root up to max_depth, skipping SKIP_DIRS.
    Test: create nested .mcp.json at depth 7 — assert it is NOT yielded; at depth 3 — assert it IS.
    """
    root = root.resolve()
    root_depth = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root):
        depth = len(Path(dirpath).parts) - root_depth
        if depth >= max_depth:
            dirnames[:] = []
            continue
        dirnames[:] = [
            d
            for d in dirnames
            if d not in SKIP_DIRS and not d.startswith(".") or d == ".mcp-vector-search"
        ]
        # Re-allow .mcp-vector-search but keep .git etc. blocked
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        if ".mcp.json" in filenames:
            yield Path(dirpath) / ".mcp.json"


def derive_index_id(mcp_json_path: Path) -> tuple[str, str]:
    """Why: trusty-search routes by index_id; we must reproduce the same id mcp-vector-search used.
    What: returns (index_id, source) — prefers `project_root` basename from sibling
    `.mcp-vector-search/config.json`; falls back to parent-dir basename.
    Test: with sibling config containing `{"project_root": "/a/b/myproj"}` returns ("myproj", "config");
    without it, for `.mcp.json` in `/x/y/myproj/.mcp.json` returns ("myproj", "dirname").
    """
    project_dir = mcp_json_path.parent
    config_path = project_dir / ".mcp-vector-search" / "config.json"
    if config_path.exists():
        try:
            with config_path.open() as fh:
                cfg = json.load(fh)
            project_root = cfg.get("project_root")
            if project_root:
                return os.path.basename(project_root.rstrip("/")), "config"
        except (json.JSONDecodeError, OSError):
            pass
    return project_dir.name, "dirname"


def build_new_entry(index_id: str) -> dict:
    """Why: centralize the target schema so all rewrites are identical.
    What: returns the canonical trusty-search stdio MCP entry for the given index_id.
    Test: assert returned dict equals the schema in CLAUDE.md (type/command/args/env keys).
    """
    return {
        "type": "stdio",
        "command": TRUSTY_COMMAND,
        "args": list(TRUSTY_ARGS),
        "env": {"TRUSTY_INDEX": index_id},
    }


def is_already_migrated(entry: dict) -> bool:
    """Why: re-runs must be idempotent — never rewrite a file that already uses trusty-search.
    What: returns True iff entry.command is the trusty-search binary.
    Test: assert True for `{"command": "trusty-search", ...}`; False for `{"command": "mcp-vector-search", ...}`.
    """
    return entry.get("command") == TRUSTY_COMMAND


def process_file(mcp_json_path: Path, apply: bool) -> str:
    """Why: encapsulates per-file decision logic so we can report consistent status codes.
    What: reads the .mcp.json, decides update/skip/error, and (if apply) writes the new file.
    Returns one of: "updated", "skipped:already", "skipped:no-entry", "error:<reason>".
    Test: feed a temp file with old entry, call with apply=True, assert "updated" and file contents.
    """
    try:
        with mcp_json_path.open() as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        return f"error:read:{exc}"

    servers = data.get("mcpServers")
    if not isinstance(servers, dict):
        return "skipped:no-entry"

    entry = servers.get(MCP_KEY)
    if not isinstance(entry, dict):
        return "skipped:no-entry"

    if is_already_migrated(entry):
        return "skipped:already"

    index_id, source = derive_index_id(mcp_json_path)
    new_entry = build_new_entry(index_id)

    print(f"\n=== {mcp_json_path}")
    print(f"    index_id: {index_id}  (source: {source})")
    print(f"    OLD: {json.dumps(entry, indent=2)}")
    print(f"    NEW: {json.dumps(new_entry, indent=2)}")

    if apply:
        servers[MCP_KEY] = new_entry
        # Preserve trailing newline if present
        tmp = mcp_json_path.with_suffix(".json.tmp")
        with tmp.open("w") as fh:
            json.dump(data, fh, indent=2)
            fh.write("\n")
        tmp.replace(mcp_json_path)
    return "updated"


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument(
        "--root", default="/Users/masa", help="Search root (default: /Users/masa)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Print changes without writing (default)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually write changes (overrides --dry-run)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Limit number of files processed (0 = no limit)",
    )
    args = parser.parse_args()

    apply = bool(args.apply)
    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"error: --root {root} is not a directory", file=sys.stderr)
        return 2

    mode = "APPLY" if apply else "DRY-RUN"
    print(f"trusty-search migrate_mcp_json.py [{mode}]")
    print(f"  root: {root}")

    counts: dict[str, int] = {}
    processed = 0
    for mcp_path in find_mcp_json_files(root):
        if args.max_files and processed >= args.max_files:
            break
        status = process_file(mcp_path, apply=apply)
        key = status.split(":", 1)[0] if status.startswith("error") else status
        counts[key] = counts.get(key, 0) + 1
        processed += 1

    print("\n--- Summary ---")
    for status in ("updated", "skipped:already", "skipped:no-entry"):
        print(f"  {status}: {counts.get(status, 0)}")
    err = sum(v for k, v in counts.items() if k.startswith("error"))
    print(f"  errors: {err}")
    print(f"  total scanned: {processed}")
    if not apply:
        print("\n(dry-run — pass --apply to write changes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
