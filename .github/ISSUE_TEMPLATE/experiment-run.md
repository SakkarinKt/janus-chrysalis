---
name: Experiment run
about: Spec a training/experiment run before it burns compute (W-A3 discipline)
labels: experiment
---

## Spec (co-signed BEFORE the run starts)

- **Proposal**: docs/proposals/NN-…
- **Grid**: variants × seeds × steps
- **Metrics + analysis plan**:
- **Config**: `experiments/<name>.json`
- **Estimated wall-clock**:

## Sign-off

- [ ] Human approved
- [ ] Claude reviewed spec for confounds

## Results (fill after)

- Manifests: `artifacts/<run>/manifest.json`
- Summary CSVs:
- Verdict vs success/kill criteria:
