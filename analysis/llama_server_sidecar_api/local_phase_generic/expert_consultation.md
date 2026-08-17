# Expert Consultation: Phase 1B Generic Injection Refactor

Provider: Anthropic subscription through `prime-agent`.
Model: `opus-5`.
Mode: focused advisory review, no tools, no repository context.
Timing: before the Phase 1B implementation was finalized.

## Question supplied

The local Phase 1 server already supports sidecar lifecycle, explicit observe,
and a narrow internal-cvec inference path. The validated research mechanism
uses Qwen3.5-4B internal sites at graph blocks 24 and 28 plus a final pre-head
logit residual, and its recursive C2 path requires checkpoint, current-token
prepass, rollback, injected replay, and commit. The proposed Phase 1B change
is to introduce a generic injection engine, a model-specific Qwen profile,
identity representation adapter, profile compatibility checks, and explicit
unsupported capabilities where llama.cpp public APIs are insufficient.

Does this refactor preserve the scientific boundary? Can the current public
cvec path represent the validated internal sites? Should final-logit and full
C2 replay be added now or rejected explicitly?

## Response recorded

The safe scope is a generic refactor around the existing cvec path. Public cvec
can express the Qwen per-layer internal sites at the post-layer/post-FFN
residual boundary without changing llama.cpp core. The final-logit residual,
which is applied after final RMSNorm and before the LM head, has no public hook
and must not be faked by treating the last internal layer as equivalent.

The C2 prepass/rollback/replay path is also not a simple refactor. It requires
KV checkpoint/restore and server sequencing. It should not be added through an
approximate implementation during this phase.

Keep `sidecar_v1` unchanged because it has no profile metadata. Use a registry
and compatibility layer first, with server-side profile binding. Use explicit
model architecture, hidden-size, layer-count, and adapter checks. Unknown
profiles and incompatible models must fail rather than guess.

The defensible hook description is a supported `residual.post_layer` boundary
backed by the public cvec index space, with the Qwen mapping graph layer 24 to
public layer 23 and graph layer 28 to public layer 27. The final/logits hook
and full recursive state path should remain reserved and explicitly
unsupported until a backend exposes them safely.

The claim boundary must say that the profile refactor is partial: it preserves
the validated internal cvec mapping but is not validated-equivalent to the
complete Step-30 topology or behavior. A Qwen-specific backend behind the
generic interface is acceptable if the generic abstraction cannot safely hide
the required context internals.

## Decisions

- Accepted: profile-driven public cvec dispatch for the two supported internal
  sites.
- Accepted: identity adapter with exact dimension checks.
- Accepted: explicit rejection of unknown/incompatible profiles and models.
- Accepted: keep `sidecar_v1` unchanged.
- Accepted: report final-logit and C2 replay as unsupported.
- Rejected: approximating final-logit injection with the last internal layer.
- Rejected: implementing a new recursive algorithm without the existing
  llama.cpp context checkpoint API.
