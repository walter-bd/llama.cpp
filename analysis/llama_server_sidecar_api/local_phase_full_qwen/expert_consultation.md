# Expert Consultation: Phase 1C

Provider: Anthropic subscription through `prime-agent`
Model: `claude-opus-5`
Scope: pre-execution engineering review

## Question

Can the current llama.cpp server expose the exact Step-30 Qwen backend without
changing the validated mechanism, including the final site, local-norm
calibration, stateful C2 replay, and C2q diagnostic?

## Advice summary

The final research site is not a last-layer graph mutation. It projects the
retrieval direction through the model output matrix and adds the scaled result
to logits after decode. A server implementation must therefore access the
output projection/logits correction path; treating it as another cvec site is
not faithful.

The internal sites require the research local-norm calibration. A direct cvec
scale is not equivalent to the validated `rho * local_norm` operation.

C2 requires a current-token prepass, exact rollback, calibrated probes, final
injection, replay, and one committed token state. The current server's
post-sampling cvec callback and shared context do not provide that lifecycle.

The validated C2q comparison uses a separately generated clean-context clamp.
It should not be relabeled as a live stateful server mode without that separate
aligned context.

## Decision

Accepted: preserve the generic/profile/API work and report the partial public
cvec path accurately.

Rejected: approximate final injection, calibration, C2, or C2q using the current
post-decode callback. Full Phase 1C remains not validated until the required
core/slot primitives are implemented and tested locally.
