# Phase 1F Expert Consultation

Reviewer: Claude Opus 5, Anthropic subscription, medium reasoning
Scope: functional llama-server port of the validated Step-30 runtime

## Review

Normal sidecar inference requires this order for every generated token:

1. Tentatively evaluate the current token and capture the validated `result_norm` query representation.
2. Retrieve `r = Pz` and compute `B = 0.01 * min(1, ||r|| / 0.05)`.
3. Apply the validated hidden interventions at `l_out-23` and `l_out-27`, including local calibration if it is part of the runtime source.
4. Project the final direction through the model output projection.
5. Add the `0.89438`-scaled logits residual before sampling.
6. Sample exactly once from corrected logits.
7. Commit the intervened replay state so later tokens see the sidecar-influenced state.

The explicit `/observe` path remains the only writer. Generation must not mutate the bank or trigger decay/consolidation.

The following are validation machinery and should not block functional benchmark readiness: C1/C1a/C2q, clean comparator contexts, per-token rollback hashes, first-step gates, causal verdict/effect-size accounting, and ablation controls. Hooks may remain available for later parity validation, but they are not required in the production request path.

Main hazards are shared slot state, sampler side effects during the tentative pass, off-by-one position/KV errors during replay, and accidental double application when sparse residual and sidecar logits contributions are composed. A full serialized context transaction is acceptable for the first functional implementation if sidecar requests are structurally slot-exclusive/serialized and `/observe` uses the same synchronization. The cost must be documented as provisional throughput overhead.

Minimum local tests are mode isolation, exactly one sampler acceptance per emitted token, zero-coefficient replay identity, empty-bank `B=0` identity, non-empty retrieval reachability, explicit-observe-only mutation, lifecycle/serialization, and a short intermediate-value spot check against the research runtime where practical. Numerical parity and quality claims remain deferred.

Explicitly deferred are C1/C1a/C2q, comparator contexts, scientific rollback proofs, multi-slot sidecar batching, speculative decoding, prompt-cache/context-shift compatibility, performance optimization, and locked-runtime parity certification.

## Three runtime modes

The review also confirms that the implementation must keep these modes explicit and independent:

1. Plain GGUF / normal llama-server.
2. GGUF plus the existing sparse residual.
3. GGUF plus plastic sidecar, optionally composed with the sparse residual when both are explicitly configured.

The sidecar mechanism is architecturally independent of the sparse residual. GGUF plus sidecar without the residual is a supported engineering mode, but it is not covered by the historical scientific experiments and must not be reported as experimentally validated. The combined residual-plus-sidecar configuration is the historically relevant configuration and needs its later parity check.

## Accepted recommendations

- Port the working Step-30 token order rather than extending the existing post-sampling cvec approximation.
- Keep sidecar state `P`, sparse-residual state, transient request KV, sampler state, and episode state separate.
- Enforce sidecar slot/request serialization structurally for the first implementation.
- Do not sample during the tentative pass; sample once after replay and final residual.
- Add the zero-coefficient identity control as the highest-value runtime plumbing check.
- Preserve explicit no-sidecar and residual-only paths without hidden sidecar activation.
- Hard-disable unsupported interactions such as speculative decoding until separately validated.

## Rejected or deferred recommendations

- Requiring C2q, a clean comparator, per-token hash telemetry, or causal-control arms for benchmark readiness: deferred to scientific parity validation.
- Requiring multi-slot sidecar batching in this phase: deferred; correctness-first serialization is acceptable.
- Claiming numerical or quality parity for GGUF plus sidecar without sparse residual: rejected; this is an untested runtime variant.
- Changing the writer, authority, topology, or adding a new learning rule: rejected as outside scope.
