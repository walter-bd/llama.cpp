# Sidecar Throughput Optimization Plan

## Objective

Raise the exact Qwen3.5 Step-30 sidecar generation path above 10 tokens/s on
the local GTX 1660 Ti Max-Q without changing the reader, writer, authority,
injection coefficients, injection sites, final output projection, sampling
order, or `sidecar_v1`.

## Baseline bottleneck

The reference server serialized and restored the complete llama context for
every generated token. It also copied intermediate tensors to the host and
performed several one-token replays for local calibration. GPU work was split
by CPU synchronization and state-management boundaries, yielding about 1.1
tokens/s despite ordinary generation exceeding 40 tokens/s.

## Implementation stages

1. Keep the reference transaction path available behind an explicit runtime
   selector.
2. Reserve one private internal sequence for the clean current-token prepass.
   Copy the live sequence into it, decode the tentative token there, capture
   `result_norm`, and delete it after retrieval. The live sequence is never
   advanced by this prepass.
3. Store the three scaled directions in persistent backend tensors and inject
   them directly into the Qwen graph at `l_out-23`, `l_out-27`, and the final
   output projection.
4. Compute the validated site-local norms in the replay graph instead of
   collecting hidden tensors on the host or running calibration replays.
5. Replay the current token once on the live sequence, apply the final logits
   residual before sampling, sample once, and retain the replayed KV state.
6. Validate builds, lifecycle APIs, bank immutability, reference/fast output
   agreement, and throughput.

## Non-goals

- No approximation through cvec.
- No change to Step-30 coefficients or authority.
- No change to learning or `/observe`.
- No change to sparse residual composition.
- No change to sidecar serialization.
- No claim of historical cross-runtime numerical identity.

