# Remote Phase 2 Plan

Remote validation is not authorized by this Phase 1E result. It becomes appropriate only after the local implementation closes the blockers recorded in `REPORT.md`.

Required validation-only sequence after implementation:

1. Lock and record the remote runtime epoch and model/executable/source hashes.
2. Run an ordinary no-sidecar server baseline.
3. Load `P_T_E4` through the server and compare the existing fingerprint.
4. Load `P_S_E4` through the server.
5. Run `T -> S -> T` with fresh server/session state and verify behavior follows the selected sidecar.
6. Verify full C2 learning-off sidecar immutability.
7. Run explicit `/observe` and compare the Step-30 writer telemetry.
8. Save, restart the server, reload, and repeat the state-specific fingerprint.
9. Run the minimal C2/C2q retrieval-mediated recursion diagnostic.
10. Record Qwen and sidecar immutability/provenance.

The remote instance must perform validation only; it must not be used to finish missing orchestration code.
