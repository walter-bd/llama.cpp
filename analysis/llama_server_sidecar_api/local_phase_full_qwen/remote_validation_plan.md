# Remote Phase 2 Validation Plan

Remote validation is not schedulable until the exact full backend is
implemented and locally smoke-tested. The RTX 3090 is for behavioral parity,
not architecture development.

1. Register and verify the current deterministic runtime epoch.
2. Run the ordinary no-sidecar server baseline.
3. Load `P_init` and run the preserved fingerprint prompts.
4. Load `P_T_E4` and `P_S_E4` independently.
5. Run `T -> S -> T` with fresh server/context state.
6. Verify full C2 reads leave the sidecar byte-identical when learning is off.
7. Run a known Step-30 episode and verify explicit `/observe` writer parity.
8. Save, restart the server, reload, and compare state-specific behavior.
9. Run the minimal C2/C2q retrieval-mediated diagnostic.
10. Record Qwen, sidecar, process, and runtime immutability/provenance.

No remote architecture development or numerical tuning is permitted.
