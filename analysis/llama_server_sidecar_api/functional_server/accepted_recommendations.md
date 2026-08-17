# Phase 1F Expert Recommendations

The advisory review was completed before implementation using the preserved
Step-28/29/30 source and the current llama-server integration.

## Accepted

- Port the working Step-30 token routine rather than extending the old
  post-sampling cvec approximation.
- Keep the current-token transaction slot/context-local. A sidecar request may
  temporarily reserve a slot; correctness takes priority over batching.
- Do not sample during the query-producing pass. Sample exactly once after
  replay, hidden interventions, and the final output residual.
- Keep portable bank state `P`, transient KV/context state, sampler state, and
  episode/write state separate.
- Keep explicit `/observe` as the only mutation boundary.
- Preserve the three runtime modes explicitly:
  `gguf`, `gguf_sparse_residual`, and `gguf_sidecar_plasticity`.
  The historical validated configuration is
  `gguf_sparse_residual_sidecar_plasticity`.
- Allow `gguf_sidecar_plasticity` without sparse residual as an engineering
  configuration, but mark it scientifically untested. No result from that
  mode is a Step-30 parity claim.
- Keep the residual path independent. Do not double-apply or silently alter
  sparse-residual behavior when a sidecar is absent.
- Retain lightweight debug counters and ordering traces, but do not make the
  scientific C2q/control suite part of normal benchmark requests.

## Rejected or Deferred

- C1/C1a/C2q, clean comparator contexts, per-token rollback hashes, first-step
  equality gates, and causal verdict controls are deferred to scientific parity
  validation, not functional benchmark readiness.
- Multi-slot sidecar batching, speculative decoding interaction, prompt-cache
  interaction, and throughput optimization are deferred.
- No new learning rule, addressing rule, profile, or sidecar format is added.
- No-residual sidecar behavior is not described as scientifically validated;
  it is only an explicit untested runtime mode.
