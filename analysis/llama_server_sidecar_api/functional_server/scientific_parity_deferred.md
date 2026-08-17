# Scientific Parity Deferred

Phase 1F validates an engineering path, not historical numerical parity.

The later locked-runtime comparison must use the combined configuration that
was historically relevant:

```text
GGUF + sparse residual + plastic sidecar
```

It should compare the research runner and server for selected logits/NLL
fingerprints, `T -> S -> T`, explicit Step-30 writer behavior, save/restart/
reload, and the C2/C2q diagnostic. The sidecar-only server mode is useful for
engineering and modularity, but the existing experiments did not test it
without the residual and therefore do not support a sidecar-only scientific
claim.

Research-only controls such as C1/C1a/C2q clean comparators, first-step gates,
per-token serialized rollback hashes, and causal verdict generation do not
block functional benchmark use.
