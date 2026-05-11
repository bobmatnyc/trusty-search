# trusty-search Dev Workflow

## Required Workflow Sequence

(prompt → ticket) OR (check tickets) → read ticket + comments → implement → test → build → deploy → smoke test → patch bump → verify CI/CD to crates.io

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

### Phase 3: Test
- Agent: rust-engineer (inline) or qa
- Run: `cargo test --workspace`
- Run: `cargo clippy --workspace --all-targets -- -D warnings`
- Must show raw test output before proceeding
- All tests green, clippy clean → proceed; else fix and re-run

### Phase 4: Build
- Agent: local-ops
- Run: `cargo build --release`
- Confirms release binary compiles cleanly

### Phase 5: Deploy
- Agent: local-ops
- Run: `make deploy` (cargo install --path . --locked)
- Verify binary installed: `which trusty-search`

### Phase 6: Smoke Test
- Agent: local-ops or qa
- Run: `make smoke`
- Must pass all checks; WARN is acceptable for daemon-dependent checks

### Phase 7: Patch Bump
- Agent: local-ops
- Run: `make patch`
- Verifies Cargo.toml version incremented and git tag created
- Commit message format: `feat|fix|chore|test(<scope>): <description> (closes #N)`

### Phase 8: Verify CI/CD
- Agent: local-ops or version-control
- Confirm GitHub Actions publish workflow triggered on the new tag
- Check workflow run status: `gh run list --repo bobmatnyc/trusty-search --limit 5`
- Confirm crates.io publish job passed (or dry-run passed if not a release tag)

## Skip Rules
- Phase 4 (build) may be skipped if Phase 3 already ran `cargo build` internally
- Phase 5+6 (deploy + smoke) may be skipped for library-only changes with no CLI surface
- Phase 7 (patch) may be skipped for chore/docs-only changes
- Phase 8 (CI/CD verify) may be skipped for non-tagged commits
- Phase 1 (ticket) is always required — no work without a ticket reference

## Commit Message Format
feat|fix|chore|refactor|test|docs(<scope>): <description> (closes #N)

## Success Criteria
All phases green → ticket closed on GitHub
