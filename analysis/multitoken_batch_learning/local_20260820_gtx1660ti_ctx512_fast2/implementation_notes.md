# Implementation Notes

## Alignment

For observed tokens `y_1...y_n`, predictor position 1 is the final explicit context token and subsequent predictor positions are `y_1...y_(n-1)`. Therefore logits at predictor position t score `y_t` without including `y_t` itself. The causal prefix includes the actual externally observed earlier tokens.

## Keys and Writes

Per-position keys come from the same `result_norm` source representation used by the one-token path. Baseline IDs are per-position greedy logits. Output-row differences use the existing numeric implementation. Prediction matches preserve the existing no-write behavior.

## State Semantics

Model acquisition is batched; state mutation is not parallel-reduced. Events are committed in deterministic token order so gate/counter/component side effects remain ordered. Episode and chunk metadata are diagnostic only. No sidecar format change was made.

## API and UI

The low-level `/v1/sidecar/observe` endpoint is unchanged. `/v1/sidecar/observe_sequence` is the text-oriented endpoint. The default web UI Plastic Sidecar panel uses it and exposes Batch, Chunked, and Exact token (research) modes without requiring token IDs.

## Current Constraints

- Fast acquisition requires `--sidecar-runtime fast` and `--parallel 1`.
- Batch size/context limits are validated before acquisition; text is not silently truncated.
- Peak VRAM and CUDA synchronization counts are not exposed by the current server instrumentation.
- Retrieval-trajectory similarity matrices were not collected in this engineering run because the existing API does not expose retrieval query vectors without further instrumentation.
