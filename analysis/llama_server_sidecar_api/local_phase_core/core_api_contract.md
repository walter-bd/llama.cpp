# Core Intervention API Contract

## Opt-in behavior

The new APIs are disabled unless explicitly called. A null/unused intervention
path does not install graph callbacks, cvec state, transaction replay, query
capture, or output residuals.

## Transaction

`llama_state_transaction_begin()` captures the serialized llama memory state
after synchronization. `llama_state_transaction_restore()` clears the memory
sequence and restores the captured state, then recaptures it and exposes an
exact byte-equality result. The transaction is owned by one context.

The current primitive does not claim to restore sampler state, logits/output
buffers, cvec adapter data, server slot bookkeeping, prompt-cache state, or
other state not included by the existing llama state serializer. A server
caller must not use it concurrently with another slot sharing the context.

## Output operations

`llama_context_project_output()` reads the loaded model output tensor and
projects an exact-size direction into vocabulary space without changing model
weights. `llama_context_add_logits_residual()` adds an exact-size host residual
to the current logits buffer. The latter is only meaningful after the
corresponding committed decode and before sampling.

## Deliberate non-claims

These primitives do not by themselves implement Step-30 C2. They do not
provide a server slot transaction, hidden-site calibration collector, final
pre-sampling callback, or C2q clean comparator. Those remain backend/server
work and are currently incomplete.
