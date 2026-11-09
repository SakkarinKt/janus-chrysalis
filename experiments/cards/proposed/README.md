# Proposed cards

Drafts. Nothing in this directory can be run: `npm run exp` only sees `cards/NNNN-slug/`.

- Anyone may add a draft here as `NNNN-short-slug/{card.json,run.ts}`, starting from
  `../TEMPLATE/`. The number is provisional.
- The daily loop may add at most one draft per run, and never moves one out of here.
- **Approval is the human moving the directory up to `cards/`** with the next free number, and
  merging that PR. Until then a draft can be edited freely. Afterwards it cannot be edited at all.
- A draft that is rejected is deleted, with the reason in the PR that deletes it.

The approval checklist is at the end of `../README.md`.
