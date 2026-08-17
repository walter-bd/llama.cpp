# Remote Phase 2 Status

Remote validation is blocked until the local implementation passes Phase 1D.
The RTX 3090 must not be used to finish missing core/server functionality.

After local completion, the locked runtime plan is validation-only:

1. V0: register and verify the current runtime epoch and source/executable/model
   fingerprints.
2. V1: ordinary no-sidecar server baseline.
3. V2: load `P_T_E4` through the server and run the preserved fingerprints.
4. V3: load `P_S_E4`.
5. V4: verify T -> S -> T follows the loaded sidecar.
6. V5: full C2 with learning disabled leaves the sidecar byte-identical.
7. V6: explicit `/observe` matches the validated Step-30 writer boundary.
8. V7: save -> server restart -> reload preserves state-specific behavior.
9. V8: minimal C2/C2q retrieval-mediated diagnostic.
10. V9: sidecar and Qwen immutability/provenance checks.

This plan is not executable as validation-only work yet because the current
server still lacks the exact C2, final-output-before-sampling, calibrated site,
and C2q implementations.
