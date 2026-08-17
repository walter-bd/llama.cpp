# Phase 1D Expert Consultation

## Consultation Context

Provider: Anthropic subscription through `prime-agent`
Model: `claude-opus-5`
Timing: pre-implementation; no Phase 1D results existed and no core edits had been made.

The review was given the exact Step-28/29/30 source mapping, the current
llama.cpp APIs, the Phase 1C blockers, the no-op requirement, the proposed
generic intervention hooks, the transaction design, the output-stage design,
and the request-local C2q clean comparator design.

## Complete Advisory Response

### Verified facts that change the design

1. `llama_context::state_write_data` currently writes only the architecture
   string plus `memory->state_write(io)`. Logits, embeddings, `output_ids`, and
   RNG are no longer serialized. A full-state snapshot is therefore a
   memory-only snapshot, and byte equality proves KV/recurrent identity, not
   every context field.
2. Qwen3.5(-MoE) uses `llama_memory_hybrid` / `llama_memory_hybrid_iswa`.
   Recurrent/linear-attention layer state is not revertible by
   `llama_memory_seq_rm` or by rewinding `n_past`; full state restore is
   mandatory.
3. `build_cvec` is applied at the per-layer `l_out` boundary and
   `llama_adapter_cvec::apply_to` inserts `ggml_add` only when the tensor for
   the layer is present. Enabling/disabling a site between decodes changes
   graph topology.
4. Layer tensors are named `"<name>-<il>"` by `llama_context::graph_get_cb()`,
   so `l_out-24` / `l_out-28` and `result_norm` are addressable from the
   existing scheduler evaluation callback.
5. `res->t_logits` is copied into the host logits buffer during decode, and a
   host-side logit residual applied before sampling is representable.

### Review of the proposed boundary

The proposed generic core/profile boundary is sufficient for the causal
mechanism only with three corrections:

- The transaction must support a loop, not only prepass -> restore -> replay.
  Calibration uses up to six rollback probes, and site 27 is measured after
  the causal site-23 injection. The sequence is snapshot, no-injection
  prepass, restore, probe/restore iterations, then final replay and commit.
  The six-probe cap and `2e-6` tolerance remain in the Qwen backend.
- Graph topology must be pinned for the whole transaction. Keep site 24 and
  site 28 cvec tensors present and zero-filled when inactive; update their
  data between probes. Toggling graph nodes between probes can invalidate
  graph reuse and alter numerics.
- Snapshot and verification must use the same serializer and state flags.
  With a shared/hybrid cache, whole-context restore can clobber other slots;
  use an exclusive context transaction or prove a safe per-sequence state
  restore. The lowest-risk local implementation is an exclusive solo
  single-token transaction.

### State that must be handled

Beyond the serialized llama state, preserve or prove invariants for:

- cvec adapter state, which is not serialized;
- server slot bookkeeping, token history, prompt-cache/context-shift state;
- sampler state, which must not advance during probes;
- output buffers, `output_ids`, and output counts, which are not restored by
  state serialization;
- hybrid/recurrent state and SWA cells, covered only if the correct full
  memory serializer is used;
- adapters/LoRA scaling, thread/eval callback settings, ubatch shape, and
  other numerical settings held fixed across probes.

Probes must not call the sampler. Sampling occurs exactly once using the
committed replay logits. Cancellation and exceptions need an abort path that
restores the checkpoint and cvec state.

### Final-output residual

A generic host `float *` logits-delta callback is sufficient and preferable,
but the backend still needs:

1. the committed replay final `result_norm` hidden state/norm; and
2. a read-only primitive to project the normalized direction through the
   actual `model.output` matrix using the same device/dtype path as the model.

Do not add a separate output graph hook or an fp32 shadow copy. Do not fall
back to tied token embeddings when `model.output` is absent. If output
projection is unavailable or tied in a way that changes semantics, stop.

### Graph hook and no-op guidance

Reuse existing scheduler evaluation infrastructure: observe only
`l_out-24`, `l_out-28`, and `result_norm` while a nullable per-context
intervention handle is active. Keep the ordinary path unchanged when the
handle is null. Compose with or explicitly reject an already-installed user
`cb_eval`; never silently overwrite it. Do not put Qwen constants in core.

### Server hazards

- A shared context and context-global cvec are unsafe when intervention is
  batched with other slots. Intervened decode must be a solo one-token batch.
- Whole-context state restore can destroy other slots. Use an exclusive server
  lock for the transaction or prove per-sequence restoration; the former is
  the lower-risk Phase 1D choice.
- Disable/defer prompt-cache reuse, context shift, defrag, speculative decode,
  multi-token draft acceptance, and continuous batching during a transaction.
- Cancellation and exceptions must restore state.

### Recommended incremental implementation

1. Transaction snapshot/restore and recapture byte verification, with no
   intervention; verify idempotent decode.
2. Read-only tensor observation through `cb_eval`; verify no-hook no-op.
3. Query capture and normalization.
4. Site 24 and site 28 calibration with rollback probes.
5. Output projection plus host residual.
6. Server wiring with exclusive solo transactions and abort paths.
7. C2q clean comparator as debug-only validation.

### Hard stops

Stop instead of degrading if byte equality or idempotent replay fails,
calibration does not converge within six probes at `2e-6`, required memory or
output projection is unavailable, a solo transaction cannot be guaranteed,
sidecar_v1 must change, or the result diverges from the research semantics.

## Accepted Recommendations

- Use nullable opt-in generic core hooks with no Qwen constants in core.
- Extend the transaction abstraction to support multiple probe restores and a
  single final commit.
- Capture `l_out-24`, `l_out-28`, and `result_norm` through existing graph
  evaluation infrastructure.
- Use existing cvec graph insertion for hidden sites, but pin topology and
  update data rather than toggling nodes.
- Add a read-only model-output projection primitive and apply the final
  residual to host logits before sampling.
- Keep C2q request-local and debug-only with a separate clean comparator.
- Keep the Qwen backend model-specific behind the generic interface.
- Make the transaction opt-in, solo, exclusive, and abort-safe.

## Rejected Recommendations

- Rewind using only `n_past` or `llama_memory_seq_rm`.
- Restore whole-context state while other slots are active.
- Use plain static cvec or post-sampling injection as Step-30 parity.
- Use a separate fp32 output matrix copy or silently substitute token
  embeddings.
- Add always-on hooks or change the ordinary no-sidecar path.
- Change sidecar_v1, constants, tolerance, or probe cap.
- Batch intervened tokens with other slots or speculative decoding.

## Engineering Decision

Proceed incrementally only if each primitive can be validated locally. The
scientific mechanism and constants remain frozen. If any hard stop is reached,
retain the partial API/profile implementation and report
`llama_core_sidecar_intervention_local_not_validated` with the precise blocker.
