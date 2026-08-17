# Phase 1E Expert Consultation

## Consultation

Reviewer: Claude Opus 5, Anthropic subscription, medium effort.

Question supplied:

> Review whether this server orchestration exactly preserves the validated Step-29/30 semantics. Focus on slot ownership, token/KV transaction boundaries, sampler side effects, graph lifetime, local calibration capture, final residual timing, and clean-context alignment. Identify implementation hazards. Do not redesign the research mechanism.

Context supplied:

* The research implementation uses a full serialized context checkpoint before the current-token prepass.
* The prepass captures `result_norm` and `l_out-23`/`l_out-27`, retrieves from a frozen bank, restores the checkpoint before every calibration probe and replay, and adds the projected final residual before sampling.
* The current server owns one shared `ctx_tgt`, batches slots, samples in `post_decode`, and applies the existing cvec path after sampling for the next decode.
* The Phase 1D transaction API snapshots/restores the full context, not a sequence-local view.
* C2q requires a request-local clean comparator context aligned to the same visible token prefix.

## Advisory response

The review endorsed the design only if the server introduces a hard admission gate for the whole transaction. A best-effort check of `n_parallel == 1` is insufficient: the current transaction snapshot covers the full shared context, so any other active slot or scheduler mutation would make restore semantically invalid. The coordinator must own the context from checkpoint through prepass, every calibration probe, restore, replay, final-logit correction, sampling, and commit. New work on that context must be rejected or held until the transaction completes.

The review rejected a sequence-local approximation as a substitute for the current full-context transaction. Either the server must prove that the full context contains only the admitted request, or it must add a genuinely sequence-local snapshot primitive. Given the current core API, the safe engineering choice is a single active sidecar slot with an explicit server-level admission guard, not merely a request validation flag.

The prepass must not call the sampler and must not advance sampler state, RNG, grammar, penalties, or speculative state. Sampling must happen exactly once after injected replay and after the final output residual has been added to the sampler-visible logits. This avoids sampler rollback and makes the causal ordering auditable.

The callback collector must copy `result_norm` and the two boundary vectors to host memory immediately after synchronized graph execution. Device tensor pointers must not escape the graph lifetime. Every calibration probe must restore the same checkpoint before decoding, clear the temporary intervention after decoding, and leave only the final replay as committed state. The review recommended treating non-convergence as a hard engineering error for a new server implementation rather than silently accepting an unverified calibration.

The final correction must be applied at the exact logits stage consumed by the sampler. It must be computed by projecting the direction through the actual model output matrix, not by treating the final site as another hidden cvec. The existing core projection and residual APIs are appropriate only if the server invokes them before sampling on the replay output.

C2q requires a second request-local context. It must consume the identical visible token prefix and position IDs as the stateful context, must never receive sidecar-influenced KV, and must be destroyed with the request. The forced response must be copied from the clean comparator and checked for exact equality before it is used in the stateful replay.

The minimum local validation set should include: exact serialized restore on every probe and replay; rejection of competing slot work while a transaction is held; sampler state unchanged by prepass; collector lifetime checks; calibration convergence and failure behavior; residual timing; clean/stateful prefix alignment; one committed replay per token; and sequential independence across requests. A Qwen fixture can validate these engineering invariants locally, but it cannot establish the historical RTX 3090 numerical effect.

## Accepted recommendations

* Use a hard server admission gate, not a request-level advisory check.
* Keep the full-context transaction and require an effectively single-slot context.
* Do not sample during prepass or calibration.
* Copy collector values before graph tensors expire.
* Restore before every probe and replay; clear temporary intervention state.
* Apply the final residual before the one sampler call.
* Keep C2q comparator state request-local and prefix-aligned.
* Test restore, sampler isolation, calibration, residual timing, and slot contention before remote validation.

## Rejected or not adopted

* No sequence-local rollback approximation was adopted.
* No post-sampling steering was accepted as Step-30 parity.
* No new control condition or scientific threshold was added.
* No change was made to the validated reader, writer, authority, topology, or sidecar format.

## Consequence for Phase 1E

The current server cannot honestly receive a passing verdict without the admission gate, callback collector/calibration path, pre-sampling output correction, and clean comparator. Those are implementation blockers, not evidence against the scientific mechanism. This Phase 1E run therefore records `llama_server_step30_slot_path_local_not_validated` and leaves the existing partial cvec server behavior clearly non-Step-30-compatible.
