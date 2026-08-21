# Plastic Sidecar External Test Report

```text
Tester:
Date:
Runtime epoch:

OS:
Kernel:
CPU:
System RAM:

GPU:
VRAM:
NVIDIA driver:
CUDA:

Compiler:
CMake:
Node:
npm:

Repository commit:
Server binary SHA256:

Model filename:
Model SHA256:
Quantization:

Residual enabled: yes / no
Residual filename:
Residual SHA256:

Sidecar profile:
Context size:
Batch:
Ubatch:
Parallel slots:
GPU layers/offload:

Server command:

UI available: yes / no
Plastic Sidecar panel available: yes / no
UI Batch teaching available: yes / no

Experience text:
Learning mode:
Chunk size, if used:
Tokens observed:
Writes realized:
Observed learning tok/s:

Sidecar initial SHA:
Sidecar learned SHA:
Sidecar saved SHA:
Sidecar reloaded SHA:

State changed after writes: PASS / FAIL
Save/reload SHA match: PASS / FAIL
Second-sidecar isolation: PASS / FAIL
Learning OFF immutability: PASS / FAIL

Reasoning-time plasticity: NOT TESTED / PASS / FAIL
Persistent state unchanged after ephemeral reasoning mutation: NOT TESTED / PASS / FAIL

Approximate BASE generation tok/s:
Approximate SIDECAR generation tok/s:
Approximate Batch learning tok/s:
Approximate Exact learning tok/s:

Build problems:
Runtime problems:
Unexpected behavior:
Behavioral differences noticed:
Comments:
```

Attach the exact startup command, startup/error logs, smoke-test output, and
screenshots if useful. Do not attach model weights, `.env`, credentials, API
tokens, or private sidecar data that should not leave the machine.
