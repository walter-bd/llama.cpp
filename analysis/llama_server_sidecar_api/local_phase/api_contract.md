# llama-server Sidecar API (Phase 1)

Phase 1 is a local engineering integration. It does not claim numerical
parity with the RTX 3090 research epochs. The base model and llama.cpp remain
frozen; sidecar writes are explicit API operations only.

## Lifecycle

All JSON endpoints use the existing llama-server HTTP server. Sidecar package
paths are resolved below the configured `--sidecar-dir` root. Directory
packages use the existing `sidecar_v1` layout (`manifest.json`,
`components.jsonl`, and `vectors.f32`).

| Method | Route | Purpose |
| --- | --- | --- |
| GET | `/v1/sidecar/status?id=ID` | Inspect one sidecar; omit `id` for all sidecars |
| POST | `/v1/sidecar/create` | Create an empty isolated sidecar |
| POST | `/v1/sidecar/load` | Load and validate a `sidecar_v1` package |
| POST | `/v1/sidecar/save` | Save a deterministic package below the sidecar root |
| POST | `/v1/sidecar/reset` | Clear components and counters, retaining the ID |
| POST | `/v1/sidecar/unload` | Remove the in-memory sidecar without saving |
| DELETE | `/v1/sidecar/ID` | Delete the in-memory sidecar |
| POST | `/v1/sidecar/learning` | Enable or disable explicit writes |
| POST | `/v1/sidecar/observe` | Apply one validated forward-only write |

`create`, `load`, `save`, `reset`, `unload`, `learning`, and `observe` take an
`id` field. `load` and `save` take a relative package path; traversal and
absolute paths outside the configured root are rejected.

## Chat integration

`POST /v1/chat/completions` accepts an optional `sidecar_id`. Without it, the
ordinary llama-server path is used. With it, the request must use one slot,
one completion, and no speculative decoder. This is deliberate: the public
adapter cvec hook is context-global, so Phase 1 rejects unsafe multi-slot use
rather than risking cross-request state leakage.

When learning is enabled, a sidecar-backed response may include
`sidecar_episode_id`. The episode is consumed by a later `/observe` request.
Reset, unload, replacement, and model/context teardown invalidate pending
episodes. Normal generation never writes automatically.

The current server integration applies the validated authority and internal
cvec sites (`block24`, `block28`) through the public llama.cpp hook. The final
logit injection site is not exposed by the server API, so status reports
`injection_mode: internal_cvec_only` and Phase 2 must validate behavioral
parity with the full research topology.

## Write boundary

`learning_enabled=false` permits retrieval but guarantees no bank mutation.
`learning_enabled=true` only permits writes after a valid, unconsumed episode
is supplied to `/observe`. The current server API receives the validated
write direction explicitly because server inference does not expose the
research runner's target output-row construction. This is plumbing for Phase
1; exact Step-30 target-direction parity is deferred to the locked-runtime
Phase 2 validation.

The manager preserves the validated constants and checks them when loading a
package:

```text
B_max = 0.01
s_ref = 0.05
alpha_24 = 0.23072
alpha_28 = 0.38322
alpha_final = 0.89438 (metadata only in Phase 1)
eta = 0.05
```

## Concurrency and isolation

The registry is keyed by `sidecar_id`; each sidecar has its own mutex and
episode namespace. Requests never use a global "last active" sidecar.
Because cvec state is context-global, the Phase 1 request validator limits
sidecar requests to a single active slot and rejects speculative decoding.

## Client

`tools/server/sidecar_client.py` is a dependency-free convenience client. It
only transports lifecycle, chat, and explicit observation data; it does not
reimplement retrieval or plasticity.
