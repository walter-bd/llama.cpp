# Accepted Recommendations

Accepted before implementation:

- Use the source-level Step-30 hard row-difference direction, not the separate
  softmax target-error helper.
- Capture the clean prepass greedy token in the episode and derive the vector
  inside the server.
- Add one narrow output-row/backprojection core primitive and fail explicitly
  when it cannot serve the requested token.
- Compute the Step-30 gate from server-owned counters and preserve `ETA=0.05`.
- Skip positive-coefficient insertion for zero-gate updates and consume a
  zero-direction observation without mutating the bank.
- Separate the normal observed-token API from an explicitly named debug vector
  route.
- Keep `sidecar_v1` unchanged; writer counters are runtime/session metadata,
  not serialized portable sidecar state in this phase.

Rejected or deferred:

- Softmax `W_out^T(e_y-p)` as a replacement for the source-called
  `direction_step30` path: rejected because the cited Step-30 function does
  not call it.
- Accuracy-based matching counts: deferred because the historical Step-30
  source unconditionally increments its scripted matching count.
- Full logits, sequence targets, embeddings, teacher forcing, and new
  sequence-level writers: out of scope.
