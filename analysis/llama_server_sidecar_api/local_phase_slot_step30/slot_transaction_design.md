# Slot Transaction Design Audit

## Required lifecycle

The validated lifecycle is:

```text
exclusive slot admission
checkpoint full committed context
decode current token without intervention
capture result_norm and l_out-23/l_out-27
retrieve and calibrate
restore exact checkpoint
replay current token with hidden interventions
project final direction and add logits residual
sample once
commit replay state
```

## Current server boundary

`server_context_impl` owns one shared `ctx_tgt`. `server_slot::ctx_tgt` points at that shared context, and `decode()` can batch work from multiple slots into it. The current request validation requires one parallel slot and one completion, but it does not hold the shared context exclusively from checkpoint through commit. The core transaction API serializes the whole context, including all sequences, so a slot-local interpretation is not valid while any other slot can mutate that context.

## Required implementation before coding

1. Add a server-level admission guard covering the entire transaction.
2. Reject or defer every other decode, prompt fill, speculative step, context shift, and scheduler mutation touching the context while held.
3. Re-check ownership immediately before checkpoint.
4. Keep sampler calls outside the prepass and calibration probes.
5. Treat only the replay as the committed current-token decode.
6. Release the guard only after sampling and commit bookkeeping complete.

## Phase 1E result

The current code does not provide this coordinator. Adding only a mutex around a helper would be insufficient because the scheduler and shared context are owned by broader server loops. No partial transaction was added.
