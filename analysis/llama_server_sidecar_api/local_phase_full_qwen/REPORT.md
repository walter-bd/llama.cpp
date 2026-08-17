# Phase 1C: Full Qwen Backend Local Audit

Date: 2026-08-17

## Scope

This phase was local engineering only. No remote GPU was used and no historical
Step 28-31B artifact was modified. The exact research source was recovered from
the preserved Git ref `research/step28-30-recursive-plasticity` at commit
`f89d3896283721b947f086c24ee207f1ed1baea9`.

## Validated-source mapping

The exact research implementation is in
`examples/factor-cache-persistence/factor-cache-persistence.cpp` and
`examples/integrated-recursive-plasticity/integrated-recursive-plasticity.cpp`.
The mapping is recorded in `implementation_mapping.json`.

The research path uses `run_stateful_step28`, `state_blob`, local-norm
calibration, and `output_projection`/`add_final_residual`. Step 29 uses the
stateful episode and a separate clean-context clamp. Step 30 reuses that reader
and adds `compute_gate_step30` and `update_bank_step30`.

## Local evidence

The local CUDA server build passed. The existing Phase 1 API smoke test passed.
The three preserved Step-31B `sidecar_v1` states loaded successfully through the
server and retained their expected component counts, width, profile, and
historical payload metadata. Learning-off immutability, explicit observe
lifecycle, ordinary no-sidecar requests, save/load/restart, and registry
isolation were covered by the existing local smoke path.

The current profile exposes the public cvec sites corresponding to research
layers 23 and 27 (server graph sites 24 and 28). This is a partial integration,
not Step-30 parity.

## Full-backend blockers

1. The current server cvec path runs after sampling and prepares the next decode;
   Step 29/30 requires a current-token prepass, exact rollback, calibrated
   injection, replay, and one committed state.
2. The current cvec path applies a direct scale. The research path calibrates
   each internal site using the local boundary norm, with up to six rollback
   probes and tolerance `2e-6`.
3. The validated final site is a post-decode output projection followed by a
   logit residual correction. The current server has no exact output projection
   hook for this operation.
4. The current context implementation does not expose the required slot-local
   current-token checkpoint/replay lifecycle.
5. The validated C2q diagnostic uses a separately generated clean-context
   clamp. A live server request does not currently provide that separate aligned
   clean context.

These are semantic gaps, not missing profile metadata. No approximate backend
was added.

## Expert review

Claude Opus 5 was consulted through the configured Anthropic subscription.
It confirmed that the final site is a projected logit correction, that internal
sites require local-norm calibration, that C2 requires prepass/rollback/replay,
and that the validated C2q control is a separate clean-context diagnostic.
The advice was accepted. The recommendation to stop short of claiming full
parity was followed.

## Verdict

`llama_server_full_qwen_sidecar_backend_local_not_validated`

Failure tags:

`final_site_hook_missing`, `recursive_path_not_integrated`,
`c2q_clean_context_missing`, `local_norm_calibration_not_exposed`.

## Next action

Do not start remote validation yet. Expose the exact output projection and
slot-local C2/C2q replay primitives locally, then rebuild and retest before
using the RTX 3090 for behavioral parity validation.
