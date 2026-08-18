# Experience-Derived Step-30 Writer API

## Normal path

Inference creates an episode only when sidecar learning is enabled. The episode
contains the sidecar identity, lifecycle generation, query/source key, and the
clean prepass greedy token. The environment then supplies exactly one observed
token:

```json
{
  "id": "agent-1",
  "episode_id": "agent-1-episode-1",
  "observed_token_id": 32,
  "observed_target": "optional provenance label"
}
```

`observed_token` is also accepted when it tokenizes to exactly one token.
Zero-token and multi-token strings are rejected. The server obtains the
output rows, derives and normalizes the Step-30 direction, computes the
server-owned gate, and appends the component. The caller never sends a
2560-dimensional vector or gate/count values.

## Debug path

`debug_write_direction` is retained on the same endpoint as an explicitly
named, mutually exclusive research/debug field. The legacy ambiguous
`write_direction` field is rejected. This route is not the normal learning
contract.

## Invariants

- Chat alone never mutates a sidecar.
- Learning disabled returns `updated: false` and does not mutate state.
- A successful episode is consumed once.
- Episodes are invalidated by sidecar lifecycle changes.
- `sidecar_v1` remains unchanged.
- Only single-token observations are supported in this phase.

