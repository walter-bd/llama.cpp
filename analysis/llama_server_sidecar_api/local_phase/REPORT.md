# llama-server Sidecar API: Local Phase 1

Status: local engineering integration validated.

This directory records the local engineering phase only. It intentionally does
not require `rtx3090_20260814` or any other scientific runtime epoch. Phase 1
checks compilation, HTTP lifecycle behavior, sidecar isolation, serialization
round trips, and the explicit observation boundary. Numerical parity with the
validated Step 30/31B experiments is deferred to Phase 2.

## Implemented boundary

- A `sidecar_id` registry is owned by the server context.
- Ordinary requests without `sidecar_id` use the existing path and do not
  allocate or retrieve sidecar state.
- Sidecar requests are restricted to one slot, one completion, and no
  speculative decoder because the public cvec hook is context-global.
- Retrieval is applied through the existing public internal cvec hook after a
  sampled token, affecting the next decode. This is a narrow recursive server
  integration, not a claim of full Step-30 final-logit topology parity.
- Writes occur only through `/v1/sidecar/observe` after a returned episode
  handle. Generation never updates the bank automatically.
- `sidecar_v1` packages are validated for magic/version/dtype/shape, checksums,
  reader constants, and component vector checksums.

## Validation status

The local CUDA-enabled `llama-server` target built successfully and the
HTTP smoke test passed. The smoke test covered ordinary no-sidecar chat,
create/load/save/reset/unload/delete, checksum rejection, duplicate and
unknown IDs, explicit observation, learning-off immutability, two-sidecar
isolation, and server restart/reload. The exact results are in
`summary.json`, `build_info.json`, and `test_results.json`.

The smoke test used the local Qwen model only to exercise server plumbing.
It did not attempt scientific numerical parity with Steps 30/31B.

## Known Phase 2 boundary

The server-side public API does not expose the research runner's final-logit
injection site or its target output-row construction. The API therefore marks
the current mode as `internal_cvec_only`; exact Step-30 behavioral parity,
full topology, and locked-runtime validation remain separate work. Package
base-model metadata is retained and dimension-checked; exact model-file hash
enforcement is deferred until the Phase 2 validation harness supplies that
model identity explicitly.
