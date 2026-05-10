# trusty-search Project Workflow

## Standard Development Workflow

Every feature/fix follows this exact sequence:

1. **Ticket** — Start from a GitHub issue (e.g., #1 FastEmbedder, #2 UsearchStore)
2. **Implement + Test** — Write code, write tests, run `cargo test --workspace`
3. **Commit** — `git commit` with conventional commit message referencing issue (`feat: ... closes #N`)
4. **Patch bump** — Bump version in `Cargo.toml` (patch increment, e.g., 0.1.0 → 0.1.1)
5. **Deploy locally** — `cargo install --path .` to install binary machine-wide
6. **Smoke test** — Run `trusty-search --version` and basic CLI smoke tests

## Phase Skipping Rules

- Skip Research phase when: implementing a well-defined ticket with clear acceptance criteria
- Skip QA phase when: engineer ran full test suite (`cargo test --workspace`) and passes
- **Never skip**: Commit, Patch bump, or `cargo install --path .` (reinstall) steps
- Reinstall is mandatory after every patch bump — the installed binary must always match the source version

## Commit Message Format

```
<type>: <description> (closes #<issue-number>)

Types: feat, fix, refactor, test, chore, docs, perf
```

## Version Bumping

Edit `Cargo.toml` (root) version field. Patch bump only unless breaking API change.
All workspace crate versions should stay in sync.

## Local Deploy Command

```bash
cargo install --path .
```

## Smoke Test Commands

```bash
trusty-search --version
trusty-search daemon &
sleep 1
trusty-search status
```
