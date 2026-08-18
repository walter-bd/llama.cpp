# Step-30 Experience-Derived Writer

## Verdict

```text
step30_experience_writer_server_restored
```

This result covers the validated single observed-token writer only. It does
not add a sequence-level writer for function calls, JSON, or arbitrary text.

## Research mapping

The authoritative source is commit `f89d3896283721b947f086c24ee207f1ed1baea9`.
`direction_step30` calls a clean prepass and constructs
`normalized(output_row_difference(output, target, baseline.metrics.greedy))`.
The server now captures that clean greedy token in the episode and derives the
same direction from dequantized `model.output` rows through
`llama_model_output_row`. The caller supplies only an observed token.

The separate conceptual `W_out^T(e_y-p)` helper is not substituted because it
is not the operation called by the recovered Step-30 writer. No sequence
aggregation, embedding, teacher forcing, or LLM-derived vector was added.

## API change

Normal learning is now:

```text
chat -> episode_id + clean prepass greedy -> observe(observed_token) -> server derives direction -> gate -> write
```

`observed_token_id` is preferred. `observed_token` is accepted only when the
active tokenizer produces exactly one token. A raw `debug_write_direction`
field remains separate for research probes; legacy ambiguous `write_direction`
is rejected.

## Evidence

- Local Release server build completed successfully.
- Existing HTTP smoke passed with exit code 0, including load/save/restart,
  learning-off immutability, invalid/multi-token observations, and isolation.
- Controlled old debug-vector versus new observed-token updates produced the
  same component payload, update norm, and final sidecar SHA.
- Server-owned gate probe produced `0.25`, `1/3`, `2/3`, and `1.0` while
  counts advanced one-for-one under the validated unconditional recurrence
  schedule.
- `sidecar_v1` manifests and payloads for `p_init`, `p_true_e4`, and
  `p_shuffled_e4` remain semantically unchanged.

## Gate and state caveat

The gate formula and ETA are source-matched. The recovered historical harness
pre-seeds true per-source counts at `{1,1}`, while a newly created server
sidecar starts runtime-only counters at `{0,0}` because those counters are not
part of `sidecar_v1`. This is recorded explicitly and is not silently encoded
into portable sidecar state.

## Boundaries

The server does not claim historical full-experiment numerical parity. It
restores the experience API boundary and the existing single-token writer
semantics. Full-call BFCL supervision remains blocked until a separately
validated structured experience mapping exists.

