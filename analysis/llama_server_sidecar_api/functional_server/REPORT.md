# Phase 1F Functional Server

## Verdict

`llama_sidecar_benchmark_server_functional`

This is an engineering verdict. It means the local server exposes the
sidecar runtime through the HTTP lifecycle and can run the explicit residual
and sidecar modules. It does not claim numerical parity with the historical
locked-runtime Step-30 executable.

## Explicit modules

The implementation keeps these capabilities disjoint:

1. `GGUF`: ordinary llama.cpp inference.
2. `GGUF + sparse residual`: the existing `LLAMA_SPARSE_RESIDUAL` path.
3. `GGUF + plastic sidecar`: `sidecar_v1`, retrieval, Step-30 runtime
   injection, and explicit `/observe` writes.

When both optional modules are enabled, the result is explicit composition:
`GGUF + sparse residual + plastic sidecar`. The sparse residual is applied by
the model graph and the sidecar is applied by the request-local sidecar
runtime; neither module enables or mutates the other.

All four local engineering modes were exercised: ordinary GGUF, residual-only,
sidecar-only, and explicit residual-plus-sidecar composition. Historical
experiments did not isolate sidecar-only from the residual, so no sidecar-only
scientific claim is made.

## Runtime path

Sidecar requests use the Qwen profile and execute the current-token path:

```text
tentative decode -> result_norm query -> Pz retrieval -> authority
-> exact transaction restore -> calibrated hidden sites 24/28
-> output projection and final logits residual -> sample once -> commit
```

The ordinary request path does not allocate or invoke sidecar machinery.
Learning remains an explicit `chat -> episode_id -> /observe` operation;
generated tokens do not update the bank automatically.

## Verification

The local CUDA build succeeded. Lifecycle/runtime smoke passed for residual-only,
sidecar-only, and sparse-residual-plus-sidecar configurations. The smoke
covered ordinary inference, profile compatibility, bad checksum rejection,
create/load/save/reset/unload/delete, learning-off immutability, explicit
observation, invalid/consumed episodes, two-sidecar isolation, and
save/restart/reload.

Concurrency throughput and benchmark-scale latency were not measured. They
remain engineering follow-up, not evidence of scientific failure.

## Deferred scientific parity

The server has not been compared against the historical Step-30 binary for
locked-runtime logits/NLL, `T -> S -> T`, or C2/C2q causal effects. Those are
separate parity work. This phase only establishes a usable local benchmark
server around the existing runtime implementation.
