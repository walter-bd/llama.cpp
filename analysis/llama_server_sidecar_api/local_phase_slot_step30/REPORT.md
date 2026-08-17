# Phase 1E Report: Exact Slot-Local Step-30 Execution Path

## Scope

This was a local engineering audit and implementation gate. No remote RTX 3090 run, new scientific experiment, sidecar format change, or research-mechanism change was performed.

## Core verification

The Phase 1D APIs are present in the built shared library `build/bin/libllama.so.0.1.0` and were verified with `nm -D -C`:

* `llama_state_transaction_begin`
* `llama_state_transaction_restore`
* `llama_state_transaction_is_exact`
* `llama_state_transaction_free`
* `llama_context_project_output`
* `llama_context_add_logits_residual`

The local server and CLI were rebuilt successfully. The Qwen3.5-4B GGUF fixture loaded and exited successfully through `llama-cli`. These are engineering prerequisites only, not scientific parity results.

## Exact research mapping

The authoritative source is commit `f89d3896283721b947f086c24ee207f1ed1baea9`, primarily `examples/factor-cache-persistence/factor-cache-persistence.cpp`. The exact stateful function captures a full context, decodes the current token without intervention, captures `result_norm` and `l_out-23`/`l_out-27`, retrieves `P z_t`, calibrates hidden injections with restore-before-probe, restores again, replays the token, projects the final direction through the output matrix, adds the final residual to logits, and only then exposes the result for scoring/sampling.

The research callback copies tensors to host memory immediately after synchronized evaluation. It recognizes `result_norm` and `l_out-*` tensors. The research injection sites are internal layers 23 and 27, corresponding to the graph boundaries named `l_out-23` and `l_out-27`; the final site is an output projection/logits residual, not another hidden cvec.

The exact local calibration is six iterations with a `2e-6` local-rho tolerance. Each probe restores the current-token checkpoint before decoding and measures `captured_boundary - alpha * direction`. The research source includes a causal sequential fallback when simultaneous calibration does not converge; any future server implementation must preserve and audit that exact behavior rather than silently use a generic normalization.

## Current server mismatch

`llama-server` currently owns one shared `ctx_tgt` used by slots and batched decode. `server_slot::ctx_tgt` points to that shared context. The existing request validation limiting `n_parallel` to one is not a transaction-wide scheduler admission gate. The Phase 1D transaction serializes the full context, so it cannot be treated as slot-local while other work can mutate the context.

At the audited source revision, the shared batched decode is in `tools/server/server-context.cpp:3612`, sampling occurs in `post_decode()` at `:3786`, and the current sidecar query/cvec operations occur afterward at `:3818` and `:3830`. That is not the validated C2 order. It does not provide prepass-only query capture, restore-before-probe, committed replay, or a pre-sampling final logits residual. The server also has no request-local clean comparator context for C2q.

## Expert review

Claude Opus 5 reviewed the proposed orchestration before any server scheduling change. The review requires a hard server admission gate for the entire checkpoint/prepass/calibration/restore/replay/final-residual/sample/commit window. It rejects a sequence-local approximation of the full-context transaction, sampling during prepass, retaining device tensor pointers beyond graph execution, and treating the final site as a hidden cvec. It also requires a request-local prefix-aligned clean comparator for C2q. The full recorded consultation is in `expert_consultation_phase1e.md`.

## Local tests and limits

Passed:

* Phase 1D core symbol verification.
* Local CUDA build of `llama-server` and `llama-cli`.
* Qwen3.5-4B fixture load smoke.
* Preservation of existing `sidecar_v1` and historical research artifacts.

Not run as false evidence:

* exact server query capture;
* local-norm calibration through server callbacks;
* slot-exclusive transactional C2;
* pre-sampling final residual;
* exact zero-intervention transaction control;
* C2q clean comparator;
* recursive persistence through the server;
* full-C2 concurrency isolation;
* Phase 1E performance.

The partial cvec path was not relabeled as Step-30 compatibility.

## Verdict

```text
llama_server_step30_slot_path_local_not_validated
```

Failure tags:

```text
slot_transaction_not_exclusive
validated_local_calibration_unavailable
final_residual_after_sampling
c2q_clean_context_missing
stateful_next_token_not_from_committed_replay
```

This is an engineering non-validation, not a falsification of Steps 28-31B or of the sidecar mechanism. Do not start the remote instance until the missing local implementation is completed and revalidated.
