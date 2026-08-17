# Phase 1E State Ownership

| State | Correct owner | Current audit |
| --- | --- | --- |
| Portable bank `P` | Sidecar instance selected by `sidecar_id` | Existing manager; no change in Phase 1E |
| Qwen KV/cache | Request/slot llama context | Current server uses a shared context; this blocks full-context transaction safety |
| Transaction checkpoint | Slot/request temporary state | Core API exists, but snapshots the whole context |
| Sampler/RNG/grammar | Request generation state | Must not be touched by prepass; current post-decode ordering is incompatible |
| Query and boundary collector | Request/token temporary host copies | Research callback exists; server does not attach an equivalent collector |
| Episode/write eligibility | Server episode handle | Existing explicit `/observe` boundary remains correct |
| C2q comparator | Request-local clean llama context | Missing from current server |

The portable sidecar must not contain KV, sampler state, graph tensor pointers, prompt cache, or episode context. Phase 1E made no serialization changes.
