# Graph-Native Step-30 Sidecar Runtime

## Result

The optimized active-sidecar path reached 12.34 tokens/s for a 64-token local
generation on a GTX 1660 Ti Max-Q. A same-prompt 16-token comparison measured
12.13 tokens/s in fast mode and 1.106 tokens/s in reference mode, a 10.97x
speedup. The requested 10 tokens/s threshold is met on this machine.

## Implementation

`--sidecar-runtime fast` is now the default benchmark path. It performs the
clean current-token prepass on a private KV sequence, captures the validated
`result_norm` query, computes `P z` and authority, and then runs one committed
replay on the live slot. Qwen graph nodes calculate the site-local norms and
apply the direction at `l_out-23`, `l_out-27`, and through the actual output
matrix before sampling.

The graph adapter tensors are allocated once per context and updated with the
current scaled direction. The private sequence is request-local and removed
after every token. The normal request slot is exclusively advanced by the
committed replay, so the next token sees sidecar-influenced KV state.

`--sidecar-runtime reference` preserves the serialized transaction and tensor
capture path for diagnostics. Requests without `sidecar_id` do not select
either sidecar runtime.

## Why it is faster

The fast path removes full-context state serialization/restoration, host-side
hidden-boundary collection, and calibration-only replay passes from every
token. It retains one clean prepass and one committed replay, but local norm
calibration and all interventions execute in the replay graph on the GPU.

## Validation

- Release CUDA and CPU `llama-server` builds completed.
- The full `sidecar_api_smoke.py` lifecycle suite passed.
- A 10-component sidecar remained byte-identical with learning disabled:
  `3b8ae2f2bac45b1743cbb8aa08745d1ec20e2167b59b13cdc74b97ed86c45d27`.
- The fast and reference paths produced the same deterministic 16-token
  continuation and top-5 first-token ordering.
- Their selected-token log-probabilities differed by `3.58e-7`.
- Fast telemetry reported 16 query captures, 16 dispatches at each hidden
  site, 16 retrievals, and no writes for the 16-token parity request.
- Empty-sidecar fast output matched ordinary output in the frozen 64-token
  benchmark.

## Performance

| Mode | Components | Tokens | Tokens/s |
|---|---:|---:|---:|
| Ordinary GGUF + residual | none | 64 | 41.46 |
| Fast empty sidecar | 0 | 64 | 18.53 |
| Fast active sidecar | 10 | 64 | 12.34 |
| Fast active parity run | 10 | 16 | 12.13 |
| Reference active parity run | 10 | 16 | 1.106 |

The active fast path is still about 3.4x slower than ordinary inference
because exact Step-30 requires a clean prepass plus a committed replay and
host retrieval between them. Further optimization should profile graph
reservation, sequence copying, and retrieval upload, but is not required to
meet the current threshold.

## Claim boundary

This is an engineering throughput and local equivalence result. It does not
replace the deferred historical RTX 3090 scientific parity campaign.

