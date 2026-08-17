# Phase 2 Qwen Profile Behavioral Validation

This plan is not executed by local Phase 1B.

Use the locked research/runtime package supplied for the remote validation and
record its own runtime epoch. Do not require the local GTX 1660 Ti output to
match historical RTX 3090 values.

1. Run `warmupinstance.md` and verify model, executable, backend, and profile
   fingerprints.
2. Verify ordinary llama-server requests without `sidecar_id` remain on the
   normal path.
3. Load `P_init`, `P_T_E4`, and `P_S_E4` through unchanged `sidecar_v1`.
4. Check T -> S -> T behavior using fresh server processes.
5. Check learning-off inference leaves the sidecar SHA unchanged.
6. Exercise explicit `/observe` and compare writer telemetry with the
   validated Step-30 implementation.
7. Save, restart the server, reload, and compare state-specific fingerprints.
8. If the server backend gains the required context-local state API, run a
   minimal C2/C2q diagnostic. Until then, report the public-cvec limitation.

The Phase 2 report must distinguish the supported internal profile sites from
the unsupported final-logit and recursive research sites.
