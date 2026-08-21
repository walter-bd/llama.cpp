# Fast Batched Multi-Token Learning Report

## Result

The new text-oriented learning API processed a 35-token experience in one operation while preserving token-local targets, baselines, keys, writer directions, and ordered state updates. On this GTX 1660 Ti runtime, full batch reached 149.80 observed tokens/s versus 17.35 tokens/s for exact-token, an 8.63x speedup.

Status: `batched_multitoken_learning_validated_as_fast_approximation`

## Throughput

| Mode | Decode calls | Time (ms) | Observed tok/s | Speedup |
|---|---:|---:|---:|---:|
| EXACT | 35 | 2017.222 | 17.35 | 1.00x |
| CHUNK4 | 9 | 362.151 | 96.64 | 5.57x |
| CHUNK16 | 3 | 255.242 | 137.12 | 7.90x |
| CHUNK32 | 2 | 249.824 | 140.10 | 8.07x |
| BATCH | 1 | 233.645 | 149.80 | 8.63x |

Every mode observed 35 tokens and realized 27 writes.

## Fidelity Against Exact

| Mode | Mean key cosine | Minimum key cosine | Baseline agreement | Writer cosine | Gate agreement |
|---|---:|---:|---:|---:|---:|
| CHUNK4 | 0.999839 | 0.999604 | 100% | 1.000000 | 100% |
| CHUNK16 | 0.999445 | 0.997604 | 100% | 1.000000 | 100% |
| CHUNK32 | 0.999476 | 0.997414 | 100% | 1.000000 | 100% |
| BATCH | 0.999430 | 0.996059 | 100% | 1.000000 | 100% |

All learned states contained 27 components. Serialized SHA values differed because fast acquisition omits within-acquisition reader recurrence and floating-point representations are not byte-identical.

## Behavioral Pilot

For the overwrite-like experience `The room costs five gold pieces.`, F generated ` $100 per night...`; all learned modes generated ` 1000 rubles...`. For the novel `Neral` experience, F and learned states also diverged, while Exact and every fast mode produced the same learned-state completion. No mode emitted either exact target phrase.

This shows that fast acquisition preserved the measured behavioral effect of Exact on these two probes. It does not establish reliable multi-token recall.

## Semantics

`y_hat_t` is the greedy token from logits at the causal predictor position immediately before observed `y_t`. The prefix includes the actual earlier observed tokens. `k_t` is captured from the validated per-position `result_norm` representation. The writer remains `normalize(W_out[y_t] - W_out[y_hat_t])`.

Fast modes collect all token events before mutating the sidecar, then apply events in deterministic token order. Gates and counters therefore advance in order, but earlier fast-mode writes do not affect later acquisition hidden states. Exact-token retains that recurrence.

## Recommendation

- Default interactive mode: `batch`.
- Memory-constrained/intermediate mode: `chunked`, default K=16.
- Scientific reference: `exact-token`.

Batch is a measured high-fidelity fast approximation on this fixed runtime and experience. It is not claimed semantically identical across arbitrary inputs or runtimes.

## Unmeasured Items

Peak VRAM, CUDA synchronization count, target NLL/rank, and retrieval-trajectory similarity matrices were not available from existing instrumentation. No values were inferred or fabricated.
