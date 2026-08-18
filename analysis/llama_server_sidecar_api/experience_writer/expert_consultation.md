# Expert Consultation

Reviewer: Claude Opus 5, medium reasoning
Scope: Step-30 single-token experience-derived writer port.

## Recommendations

- The validated Step-30 direction is the normalized difference between the
  dequantized output rows for the observed target token and the clean,
  pre-injection greedy token. It is not the softmax `W_out^T(e_y-p)` helper
  found in another historical translation unit; do not substitute that helper.
- Capture the clean prepass greedy token in the episode. The query/source key,
  sidecar version, and that token are the minimum writer state. The sampled
  replay token and full logits are telemetry, not writer inputs.
- Add a narrow core output-row/backprojection primitive using the same
  dequantized `model.output` cache as the existing forward projection. Fail
  explicitly when the output matrix is unavailable or incompatible.
- Preserve `ETA=0.05`, the exact Step-30 gate formula, row/token indexing, L2
  normalization, component layout, and the positive-coefficient insertion
  guard. A zero target direction should be consumed as an observed episode
  with `updated=false`, not inserted as a zero component.
- Keep runtime experience counts internal. The historical Step-30 source uses
  an unconditional matching schedule rather than comparing the observed token
  to the attempted token; do not silently replace that with an accuracy-based
  count. Do not change `sidecar_v1` to serialize these counters in this task.
- Keep the raw vector route only as an explicitly named debug path, mutually
  exclusive with the derived token route. Do not accept the old ambiguous
  `write_direction` field as the normal API.
- Reject multi-token observed strings, stale/cross-sidecar episodes, invalid
  token IDs, and unavailable output matrices explicitly. Preserve the
  learning-off and no-sidecar invariants.

## Accepted Scope

The implementation will restore only the validated single observed-token
writer. Sequence-level supervision, embeddings, teacher forcing, and any new
plasticity rule remain out of scope.
