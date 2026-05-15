# trusty-search Dev Workflow

> **MANDATORY: All resolution commits MUST include `(closes #N)` in the
> message** so the referenced GitHub issue auto-closes when the commit lands
> on `main`. This applies to every phase below that produces a commit
> (implementation, test, patch bump, etc.) — not just the patch bump.
> Skipping the trailer leaves issues open and breaks the audit trail.

## Required Workflow Sequence

(prompt → ticket) OR (check tickets) → read ticket + comments → implement → test → build → patch bump → **deploy → restart daemon → smoke test** → verify CI/CD to crates.io

**Every commit produced in this sequence MUST end with `(closes #N)` where
`#N` is the ticket being resolved.** Use `(refs #N)` only for intermediate
commits that should NOT auto-close the issue (e.g. partial fixes still under
review).

## Phase Definitions

### Phase 0: Ticket
**Either:** user provides a prompt → create a GitHub issue capturing requirements and acceptance criteria
**Or:** check existing open tickets → pick the next item to work on

No work begins without a ticket reference.

### Phase 1: Read Ticket
- Read the full ticket body AND all comments
- Understand acceptance criteria completely before writing any code
- Agent: ticketing_agent (to fetch issue + comments)

### Phase 2: Implement
- Agent: rust-engineer
- Write code satisfying all acceptance criteria
- Follow coding rules: no `unwrap()` in library code, `thiserror` for crates, `anyhow` for binary
- Commit message format: `feat|fix|refactor(<scope>): <description> (closes #N)`
- The `(closes #N)` trailer is REQUIRED so the issue closes automatically on merge.

### Phase 3: Test
- Agent: rust-engineer (inline) or qa
- Run: `cargo test --workspace`
- Run: `cargo clippy --workspace --all-targets -- -D warnings`
- Must show raw test output before proceeding
- All tests green, clippy clean → proceed; else fix and re-run
- If new tests are committed separately, the commit message MUST include the
  `(closes #N)` trailer: `test(<scope>): <description> (closes #N)`.

### Phase 4: Build
- Agent: local-ops
- Run: `cargo build --release`
- Confirms release binary compiles cleanly
- If this phase produces a commit (rare — typically only for build-script
  fixes), use: `chore(build): <description> (closes #N)`.

### Phase 5: Patch Bump
- Agent: local-ops
- Run: `make patch`
- Verifies Cargo.toml version incremented and git tag created
- Commit message format: `feat|fix|chore|refactor|test|docs(<scope>): <description> (closes #N)`
- The `(closes #N)` trailer is REQUIRED — `make patch` produces the commit
  that lands on `main` and triggers issue auto-close.

### Phase 6: Deploy (MANDATORY — no skip)
- Agent: local-ops
- Run: `make deploy` (cargo install --path . --locked with CARGO_BUILD_JOBS=2)
- Verify binary version matches the patch bump: `trusty-search --version`
- Restart the daemon:
  ```bash
  launchctl unload ~/Library/LaunchAgents/com.bobmatnyc.trusty-search.plist
  launchctl load  ~/Library/LaunchAgents/com.bobmatnyc.trusty-search.plist
  ```
- Confirm daemon is running: `trusty-search status`
- Check RSS is reasonable (< 2 GB): `ps -o pid,rss -p $(pgrep -f "trusty-search start") | awk 'NR>1{printf "RSS: %.0f MB\n", $2/1024}'`

### Phase 7: Smoke Test (MANDATORY — no skip)
- Agent: local-ops or qa
- Run: `make smoke` (if recipe exists) OR manual HTTP probe:
  ```bash
  curl -s http://127.0.0.1:$(cat ~/Library/Application\ Support/trusty-search/port.lock)/health | jq .
  ```
- Must return `{"status":"ok"}` with the correct version
- WARN is acceptable for daemon-dependent checks; ERROR is a blocker

### Phase 8: Verify CI/CD
- Agent: local-ops or version-control
- Confirm GitHub Actions publish workflow triggered on the new tag
- Check workflow run status: `gh run list --repo bobmatnyc/trusty-search --limit 5`
- Confirm crates.io publish job passed (or dry-run passed if not a release tag)

## Skip Rules
- Phase 4 (build) may be skipped if Phase 3 already ran `cargo build` internally
- Phase 5 (patch) may be skipped for chore/docs-only changes with no binary impact
- Phase 6 (deploy + restart) is **NEVER skipped** — the local service must always run the latest version
- Phase 7 (smoke test) is **NEVER skipped** — must confirm the restarted daemon is healthy
- Phase 8 (CI/CD verify) may be skipped for non-tagged commits
- Phase 1 (ticket) is always required — no work without a ticket reference

## Commit Message Format
feat|fix|chore|refactor|test|docs(<scope>): <description> (closes #N)

## Success Criteria
All phases green → ticket closed on GitHub
