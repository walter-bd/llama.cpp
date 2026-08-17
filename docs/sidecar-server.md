# llama-server Sidecar API and Injection Profiles

This document describes the local functional server integration for the
external `sidecar_v1` state. It is separate from the locked-runtime research
reports; numerical parity with those reports remains a later validation task.

## Explicit runtime modules

The server has three independent opt-in capabilities:

1. **GGUF**: ordinary llama.cpp inference with no residual or sidecar state.
2. **GGUF + sparse residual**: the existing sparse-residual model correction,
   enabled independently through `LLAMA_SPARSE_RESIDUAL`.
3. **GGUF + plastic sidecar**: external `sidecar_v1` retrieval, validated
   Step-30 runtime injection, and explicit `/observe` writes.

The residual and sidecar are disjoint. Enabling a sidecar does not enable a
sparse residual, and enabling a sparse residual does not select a sidecar.
When both are configured, the result is explicit composition:

```text
GGUF + sparse residual + plastic sidecar
```

This combined mode is the historically relevant Step-30 configuration. The
server also permits sidecar-only operation as an engineering mode, but the
existing scientific experiments did not isolate sidecar-only behavior from
the residual and therefore do not establish a sidecar-only scientific result.

## Runtime boundary

The ordinary path is unchanged when a request omits `sidecar_id`. No sidecar
retrieval, sidecar injection, sidecar write, or sidecar-specific token
transaction is selected for that request.

Sidecar-backed requests use the validated Qwen profile and a slot-local
current-token routine:

```text
tentative current-token evaluation
  -> result_norm query capture
  -> P z retrieval and authority
  -> exact state restore
  -> hidden site 24 and site 28 injection
  -> final output projection and logits residual
  -> sample once
  -> commit the injected replay state
```

The final residual is applied before sampling. The committed state remains in
the request context and is used by later tokens. Scientific controls such as
C2q clean-comparator runs, per-token rollback hashing, and causal effect
analysis are not benchmark-runtime requirements and remain deferred.

Sidecar support is opt-in: start the server with `--sidecar-dir PATH`.
Without that option, sidecar requests fail explicitly as disabled.

## Lifecycle endpoints

All endpoints are under `/v1/sidecar/`:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/status` | List sidecars, or pass `?id=...` for one state. |
| `POST` | `/create` | Create an empty isolated sidecar. |
| `POST` | `/load` | Load a validated `sidecar_v1` directory. |
| `POST` | `/save` | Save the current state deterministically. |
| `POST` | `/reset` | Clear components and counters while retaining the ID. |
| `POST` | `/unload` | Remove the in-memory sidecar without saving. |
| `DELETE` | `/{id}` | Delete the in-memory sidecar. It never deletes files. |
| `POST` | `/learning` | Enable or disable explicit writes. |
| `POST` | `/observe` | Consume an episode and perform one forward-only write. |

Paths supplied to load/save are resolved below the configured
`--sidecar-dir` root. Absolute paths are accepted only when they remain inside
that root. Directory traversal is rejected.

## Chat integration

Add an optional `sidecar_id` to `/v1/chat/completions` or the completion
request. An unknown ID is an error; the server never falls back to the last
active sidecar. A sidecar request returns `sidecar_episode_id` when learning
is enabled. The handle is consumed by `/observe` and invalidated by reset,
unload, model reload, or expiry. It intentionally survives ordinary request
completion so an external outcome can be submitted afterward.

Example:

```json
{
  "messages": [{"role": "user", "content": "..."}],
  "sidecar_id": "agent",
  "max_tokens": 32
}
```

The server does not learn automatically from generated tokens. The only write
boundary is an explicit `/observe` call after an external outcome is
available. With learning disabled, inference may retrieve/inject but the
sidecar state cannot change.

## Injection profile

The server separates model-independent sidecar lifecycle from model-specific
graph locations. The only registered full runtime profile is:

```text
qwen35-4b-step30-v1
architecture: qwen35
hidden size: 2560
adapter: identity/v1
query: result_norm
hidden sites: l_out-23 / graph site 24, alpha 0.23072
               l_out-27 / graph site 28, alpha 0.38322
final output: actual model.output projection, alpha 0.89438
```

Authority is:

```text
r = P z
B = 0.01 * min(1, ||r|| / 0.05)
```

The profile does not silently adapt to other architectures. Unknown models,
missing profiles, incompatible dimensions, and unsupported hooks fail
explicitly. Generic server infrastructure does not imply cross-model sidecar
portability.

## Learning boundary

The forward-only writer remains separate from inference:

```text
chat/inference -> episode_id -> external outcome -> /observe -> writer
```

`learning=true` permits an explicit observation to write; it does not make
ordinary generation update the bank. The validated gate and `eta` remain in
the existing shared implementation.

## Portable state

`sidecar_v1` is a deterministic directory containing `manifest.json`,
`components.jsonl`, and `vectors.f32`. It stores bank coefficients, keys,
values, component metadata, reader metadata, and checksums only. It does not
store Qwen weights, llama context, KV cache, prompt cache, hidden states,
RNG/session state, or CUDA state.

## Concurrency and isolation

The registry owns independent sidecar objects keyed by ID. Each object has its
own mutex and episode table. Sidecar-backed requests are kept out of ordinary
batching because the current cvec/injection state is context-local. The
request's transient KV/transaction state is never stored in portable sidecar
state. This is a correctness restriction; throughput tuning is deferred.

## Python client and benchmark workflow

`tools/server/sidecar_client.py` provides a dependency-free transport helper.
It only calls the HTTP API and does not implement sidecar math or learning.
An external harness can repeatedly:

```python
sid = client.create("task-1")
episode = client.chat(messages, sidecar_id=sid)
client.observe(sid, episode["sidecar_episode_id"], outcome="D")
prediction = client.chat(test_messages, sidecar_id=sid)
client.delete(sid)
```

The model stays loaded while sidecars are created, reset, saved, unloaded,
reloaded, and deleted.

## Scientific validation deferred

The functional server result must not be described as exact historical
Step-29/30 numerical parity. A later locked-runtime validation should compare
the research runner and server for the combined residual-plus-sidecar mode,
including selected logits/NLL fingerprints, `T -> S -> T`, writer behavior,
restart/reload, and the C2/C2q diagnostic. C2q and other research controls do
not block normal benchmark operation.
