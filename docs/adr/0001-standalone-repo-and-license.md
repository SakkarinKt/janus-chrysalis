# ADR-0001: Standalone repo, MIT license, public from day 1

- **Status**: accepted (approved by @SakkarinKt in planning session, 2026-07-04)
- **Deciders**: @SakkarinKt (signoff), Claude (proposal)

## Decision

1. `janus-chrysalis` is a **standalone git repository**, not a subdirectory of the personal grab-bag repo. An open-source project needs its own history, issues, milestones, releases, and a clean clone URL. The parent repo gitignores this directory.
2. **License: MIT** — the norm for RL research code; zero-friction reuse maximizes the chance others build on the findings. Apache-2.0 was the runner-up (patent grant); not needed at this project's scale.
3. **Repo visibility** to be confirmed at creation time; recommendation is public from day 1 — working in the open is part of the OSS-practice learning goal, and rough early commits are normal.

## Consequences

- All work happens here; the parent repo never sees this history.
- The daily cloud loop can clone from GitHub independently of the human's machine.
- License headers are not required per-file; the root LICENSE governs.
