# llama-server Generic Sidecar Injection: Local Phase 1B

Status: `llama_server_generic_sidecar_injection_local_not_validated`.

This phase refactors the existing local sidecar server path around a generic
profile/adapter/dispatch boundary. It does not claim to reproduce the full
Step-30 research topology or the locked RTX 3090 behavior.

## Implemented

- Added a profile registry with the validated `qwen35-4b-step30-v1` profile.
- Moved authority and topology constants behind profile/authority helpers.
- Routed query capture through the profile query hook contract.
- Routed internal cvec dispatch through a profile-driven injection engine.
- Added identity adapter and hidden-dimension compatibility checks.
- Exposed profile ID, hook, supported/research site counts, dispatch counters,
  and unsupported capability flags in sidecar status.
- Preserved `sidecar_v1` serialization without adding profile fields or
  changing its on-disk format.
- Preserved the explicit `/observe` write boundary and existing sidecar API.

## Validated Qwen profile

```text
profile: qwen35-4b-step30-v1
architecture: qwen35
hidden size: 2560
adapter: identity/v1
query: result_norm
graph layer 24 -> public cvec layer 23, scale 0.23072
graph layer 28 -> public cvec layer 27, scale 0.38322
```

The research final pre-head/logit site, scale `0.89438`, is recorded as an
unsupported profile site. The current public server backend does not expose a
safe final-logit hook or the full checkpoint/prepass/rollback/replay sequence.
It fails explicitly instead of substituting a different hook.

## Verification

- Local CUDA `llama-server` build: pass.
- Existing Phase 1 API/lifecycle smoke with profile assertions: pass when run
  with the compatible local Qwen model.
- Static profile mapping and compatibility contract: pass by source/profile
  audit.
- Full recursive C2/C2q server path: not validated; backend capability is
  explicitly false.
- Remote scientific runtime: not used.

Because the full recursive path and final-logit site are not exposed by this
server backend, the Phase 1B hard verdict remains not validated. This is a
boundary report, not a claim that the validated research implementation was
changed or disproven.
