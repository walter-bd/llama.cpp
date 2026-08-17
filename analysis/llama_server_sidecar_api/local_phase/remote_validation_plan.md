# Phase 2 Remote Validation Plan

Do not run this plan during Phase 1. It requires a supplied instance with the
validated research runtime and must not be substituted by local numerical
outputs.

## Minimal checks

1. Run the locked-runtime warmup and record the runtime epoch.
2. Start `llama-server` with the Phase 1 executable and the exact validated
   model/runtime configuration.
3. Verify an ordinary request without `sidecar_id` remains the baseline path.
4. Load `P_init`, `P_T_E4`, and `P_S_E4` through `sidecar_v1` and verify
   package checksums.
5. Run the explicit `T -> S -> T` state swap using fresh server processes.
6. Verify learning-off inference leaves the sidecar SHA unchanged.
7. Exercise learning-on inference followed by an external `/observe` and
   compare the write telemetry with the validated Step-30 writer.
8. If required, run the C2/C2q recursive diagnostic on the server path and
   document the current internal-cvec-only limitation versus the full research
   topology.
9. Test save, server restart, reload, and state-specific behavioral output.

## Required metadata

Record model, sidecar, source, executable, CUDA/backend, runtime epoch, API
request bodies, response hashes, sidecar SHA before/after inference, and all
known differences from the research runner. Do not claim historical numerical
identity merely because the API lifecycle succeeds.
