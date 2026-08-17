# Phase 1D: Core Intervention API and Step-30 Server Path

## Scope

This phase was local engineering only. The target was to expose the smallest
opt-in llama.cpp core primitives needed by the validated Step-30 Qwen backend
without changing the research mechanism or the ordinary no-sidecar path.

The exact research semantics were recovered from commit
`f89d3896283721b947f086c24ee207f1ed1baea9` on
`research/step28-30-recursive-plasticity` and are recorded in
`validated_semantics_mapping.json`.

## Implemented core primitives

- Opt-in serialized llama memory transaction begin/restore/free APIs.
- Exact byte comparison of the serialized memory state after restore.
- Read-only projection of a direction through the loaded `model.output`
  tensor, including supported quantized output tensors.
- Host-side logits residual addition with exact vocabulary-size validation.
- No changes to `sidecar_v1`, Qwen weights, plasticity equations, or the
  ordinary graph path.

The core target `llama-server` builds successfully with these APIs.

## Blocking gaps

The full validated Step-30 server path is not implemented. The current server
still uses a shared target context, batches slots through that context, samples
in `post_decode()`, and installs a next-decode cvec after sampling. That path
cannot represent the required prepass, exact restore, calibrated replay, final
residual-before-sampling, and single commit sequence.

The remaining blockers are:

1. No slot-exclusive transactional server path. The new core snapshot covers
   serialized llama memory, but not cvec adapter state, sampler/output state,
   or server slot bookkeeping. Restoring it while other slots share the target
   context is unsafe.
2. The exact `l_out-23`, `l_out-27`, and `result_norm` observation callback is
   not installed with a request-lifetime collector, and the six-probe local
   calibration sequence is not wired into server replay.
3. The output projection and logits-residual primitives are not connected at
   the pre-sampling point. The existing server sidecar path remains
   post-sampling/next-decode cvec behavior.
4. The request-local clean context and forced-retrieval C2q diagnostic are not
   implemented.
5. No compatible local Qwen3.5 model fixture was available for numerical
   server smoke tests. Existing non-Qwen GGUF files are not valid substitutes.

## Verification

- Incremental CUDA/server build passed using the existing `build` tree.
- `llama-server --version` and `--help` passed.
- `git diff --check` passed.
- Existing generic server-side sidecar changes were not rewritten as Step-30
  parity.
- A recurrent rollback test invocation without its required generated Qwen
  fixture was not treated as a scientific pass; it terminated with SIGSEGV and
  is recorded as uninterpretable in `transaction_restore_test.json`.

## Verdict

```text
llama_core_sidecar_intervention_local_not_validated
```

This is an engineering implementation verdict, not a falsification of the
validated research mechanism. Remote RTX 3090 validation must not begin until
the missing local transaction, calibrated hooks, output-stage, and C2q paths
are implemented and locally smoke-tested.
