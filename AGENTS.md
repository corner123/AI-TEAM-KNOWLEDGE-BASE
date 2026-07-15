# Repository maintenance instructions

These instructions apply to the whole repository.

- Keep this RAG repository independent from Mini-Nanobot. Integration remains
  a read-only HTTP boundary; do not merge the two codebases.
- After an authorized code, test, data-contract, or documentation change,
  run the relevant tests and release checks, commit the completed change, and
  push it to the configured `origin` remote unless the user explicitly asks
  not to push.
- Never commit `.env`, API keys, tokens, local vector indexes, raw fetched
  corpora, build manifests, model weights, logs, databases, or cache files.
- Before every push, verify the staged diff, run a secret/host-path scan, and
  confirm that the remote commit matches the local commit after pushing.
- Keep formal engineering evaluation deterministic. Do not silently include
  online answer generation in frozen retrieval metrics.
- Do not overwrite the published first-run holdout report. A new final
  evaluation requires a new private holdout, snapshot, build ID, and report.
- Treat the published `dirty=true` snapshot as a historical local run. Do not
  claim exact cross-machine reproducibility until the corresponding
  Mini-Nanobot worktree changes exist in its own remote history.
