# Plastic Sidecar Friend Quickstart

This repository runs a frozen quantized Qwen model with a detachable,
forward-only plastic sidecar. The sidecar is separate from Qwen weights: it
can be explicitly taught, selected, saved, unloaded, and reloaded.

The feature is experimental. Current tests establish detachable state and
causal inference changes, not reliable arbitrary factual recall or general
continual learning.

## 1. User Modes

- **Normal frozen Qwen:** select `No sidecar`. No sidecar is read or written.
- **Persistent plastic sidecar:** select a sidecar for inference. Persistent
  learning occurs only through `Teach / Observe` or the explicit API.
- **Experimental reasoning-time plasticity:** a request-local copy may mutate
  during one reasoning request and is discarded afterward. It is off by
  default and does not persist internal writes.

The important boundary is:

```text
chat with sidecar = read persistent state
Teach / Observe   = explicitly update persistent state
```

Ordinary chat never silently trains the sidecar.

## 2. Hardware

Linux is the tested path. Install Git, CMake, a C++ compiler, Python 3,
`curl`, `jq`, and Node.js 20 or newer. NVIDIA builds also require a compatible
CUDA toolkit and driver.

A GTX 1660 Ti 6 GB can run the validated 4B model with conservative context
and batch settings, but it is much slower than an RTX 3090-class GPU. Reducing
context or GPU layers reduces VRAM pressure. Different hardware is a new
runtime epoch; exact token sequences and logits need not match old runs.

Useful diagnostics:

```bash
nvidia-smi
watch -n 1 nvidia-smi
```

## 3. Validated Model Contract

Model weights are not included. The research-validated artifact is:

```text
filename: Qwen3.5-4B-UD-Q3_K_XL.gguf
architecture: qwen35
hidden width: 2560
layers: 32
SHA256: 1874727e4bb6ac23fd0cd134ac6562d2759fe44f37fa68c55197b36526da4e37
sidecar profile: qwen35-4b-step30-v1
query source: result_norm
topology coefficients: [0.23072, 0.38322, 0.89438]
```

Verify the independently obtained model:

```bash
sha256sum /absolute/path/Qwen3.5-4B-UD-Q3_K_XL.gguf
```

The implementation can load other GGUFs, but other models are not validated
for this sidecar profile. Incompatible architecture/width combinations fail.

### Optional Sparse Residual

The sparse residual is independent of the sidecar. Historical combined runs
used a residual with SHA256:

```text
63202298b1c797aec1bf2a5c666fa36e62d8b142a800bf4583a53fe10aefdee6
```

Enable only the matching artifact:

```bash
export LLAMA_SPARSE_RESIDUAL=/absolute/path/Qwen3.5-4B-UD-Q3_K_XL.residual-0.25pct.srs
export LLAMA_SPARSE_RESIDUAL_MODEL_SHA256=1874727e4bb6ac23fd0cd134ac6562d2759fe44f37fa68c55197b36526da4e37
sha256sum "$LLAMA_SPARSE_RESIDUAL"
```

Unset `LLAMA_SPARSE_RESIDUAL` for sidecar-only engineering tests.

## 4. Build

The modified default UI must be built locally before CMake embeds it. The
repository was verified with these Node/npm binaries:

```bash
export PATH=/home/walt/.nvm/versions/node/v24.18.0/bin:$PATH
cd tools/ui
/home/walt/.nvm/versions/node/v24.18.0/bin/npm ci
LLAMA_UI_OUT_DIR="$PWD/dist" /home/walt/.nvm/versions/node/v24.18.0/bin/npm run build
cd ../..
```

Recommended CUDA build:

```bash
cmake -S . -B build \
  -DGGML_CUDA=ON \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_UI=ON \
  -DLLAMA_USE_PREBUILT_UI=OFF
cmake --build build --config Release --parallel "$(nproc)" --target llama-server
```

CPU-capable build:

```bash
cmake -S . -B build \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_UI=ON \
  -DLLAMA_USE_PREBUILT_UI=OFF
cmake --build build --config Release --parallel "$(nproc)" --target llama-server
```

Expected binary: `build/bin/llama-server`.

If CMake reports an npm engine error and downloads upstream UI assets, it used
an older system Node. Rebuild `tools/ui/dist` with Node 20+ as above, then
rebuild `llama-server`. The build log must say:

```text
UI: using pre-built assets from .../tools/ui/dist
```

## 5. Start the Server

```bash
mkdir -p sidecars
./build/bin/llama-server \
  -m /absolute/path/Qwen3.5-4B-UD-Q3_K_XL.gguf \
  --ctx-size 4096 \
  --n-gpu-layers 999 \
  --sidecar-dir "$PWD/sidecars" \
  --sidecar-runtime fast \
  --parallel 1 \
  --port 8080
```

`--parallel 1` is required by the current exact sidecar transaction/runtime
contract. Use `--n-gpu-layers 0` for CPU execution.

Verify:

```bash
curl -fsS http://127.0.0.1:8080/health
curl -fsS http://127.0.0.1:8080/v1/sidecar/status | jq .
```

Open `http://127.0.0.1:8080`.

## 6. Default Web UI

Open `Settings -> Chat -> Plastic Sidecar`.

The panel supports:

- create, load, select, save, reset, and unload;
- explicit persistent-learning enable/disable;
- active sidecar ID, profile, component/write counts, and abbreviated hash;
- multi-line `Teach / Observe` with Batch, Chunked, or Exact token mode;
- request-local reasoning-time plasticity under the experimental control.

The active sidecar also appears above the chat composer. Selecting it adds the
sidecar ID to subsequent chat requests. Selection does not itself enable
learning.

## 7. Teaching Multi-Token Experience

In `Teach / Observe`:

1. Select or create a sidecar.
2. Enable persistent learning.
3. Enter a complete experience, for example:

   ```text
   The room costs five gold pieces.
   ```

4. Choose a mode.
5. Click `Learn`.

Modes:

- **Batch (fast):** one causal acquisition pass where feasible, followed by
  ordered token-local writes. This is the recommended interactive default.
- **Chunked:** causal acquisition in configurable model chunks. It preserves
  token ordering but currently commits learned state after acquisition, so it
  is not token-recurrent exact mode.
- **Exact token (research):** the slow reference path. Sidecar writes can
  affect later acquisition positions.

The UI reports tokens observed, writes, chunks, elapsed time, observed tok/s,
and state hash before/after. A changed hash proves persistent sidecar state
changed; it does not guarantee exact recall.

## 8. Direct API

```bash
export BASE_URL=http://127.0.0.1:8080
export SIDECAR_ID=friend-test
```

Create a sidecar:

```bash
curl -fsS "$BASE_URL/v1/sidecar/create" \
  -H 'Content-Type: application/json' \
  -d '{"id":"friend-test","learning_enabled":true}' | jq .
curl -fsS "$BASE_URL/v1/sidecar/status?id=friend-test" | jq .
```

Teach the sentence in Batch mode:

```bash
curl -fsS "$BASE_URL/v1/sidecar/observe_sequence" \
  -H 'Content-Type: application/json' \
  -d '{
    "id":"friend-test",
    "text":"The room costs five gold pieces.",
    "mode":"batch"
  }' | jq .
```

Chunked mode:

```bash
curl -fsS "$BASE_URL/v1/sidecar/observe_sequence" \
  -H 'Content-Type: application/json' \
  -d '{
    "id":"friend-test",
    "text":"The room costs five gold pieces.",
    "mode":"chunked",
    "chunk_size":16
  }' | jq .
```

Exact reference mode:

```bash
curl -fsS "$BASE_URL/v1/sidecar/observe_sequence" \
  -H 'Content-Type: application/json' \
  -d '{
    "id":"friend-test",
    "text":"The room costs five gold pieces.",
    "mode":"exact-token"
  }' | jq .
```

The server tokenizes text internally. Normal callers never supply a 2560-D
vector. The low-level `/v1/sidecar/observe` endpoint remains available for
single-token research workflows using `episode_id` and `observed_token_id`.

Chat with the selected sidecar without writing:

```bash
curl -fsS "$BASE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "messages":[{"role":"user","content":"How much does the room cost?"}],
    "sidecar_id":"friend-test",
    "max_tokens":32,
    "temperature":0
  }' | jq .
```

Disable writes while retaining reads:

```bash
curl -fsS "$BASE_URL/v1/sidecar/learning" \
  -H 'Content-Type: application/json' \
  -d '{"id":"friend-test","enabled":false}' | jq .
```

## 9. Verify, Save, Restart, Reload

Record the hash before and after teaching:

```bash
curl -fsS "$BASE_URL/v1/sidecar/status?id=friend-test" | jq .
```

When writes occur, require:

```text
hash_before != hash_after_learning
```

Save below the configured `--sidecar-dir`:

```bash
curl -fsS "$BASE_URL/v1/sidecar/save" \
  -H 'Content-Type: application/json' \
  -d '{"id":"friend-test","path":"friend-test-package"}' | jq .
```

After restarting the server with the same model/profile/runtime:

```bash
curl -fsS "$BASE_URL/v1/sidecar/load" \
  -H 'Content-Type: application/json' \
  -d '{
    "id":"friend-test",
    "path":"friend-test-package",
    "learning_enabled":true
  }' | jq .
curl -fsS "$BASE_URL/v1/sidecar/status?id=friend-test" | jq .
```

Require `hash_after_learning == hash_after_reload`.

## 10. Sidecar Isolation

Create `friend-A` and `friend-B`, teach only `friend-A`, then inspect both:

```bash
curl -fsS "$BASE_URL/v1/sidecar/create" -H 'Content-Type: application/json' \
  -d '{"id":"friend-A","learning_enabled":true}' | jq .
curl -fsS "$BASE_URL/v1/sidecar/create" -H 'Content-Type: application/json' \
  -d '{"id":"friend-B","learning_enabled":true}' | jq .
curl -fsS "$BASE_URL/v1/sidecar/status" | jq '.sidecars[] | {id,sha256,components,writes}'
```

Teach only `friend-A`. Require `A changed` and `B unchanged`. Switching UI
selection changes the detachable state used by the same frozen model.

## 11. Experimental Reasoning-Time Plasticity

Start a reasoning-enabled server with the supported settings, for example:

```text
--reasoning on --reasoning-format deepseek --reasoning-budget 128
```

The budget is a ceiling, not a forced minimum. The UI experimental checkbox
sends `sidecar_debug_internal_plasticity: true` with the next chat request.
The server clones the selected persistent sidecar, mutates only the working
copy during reasoning, and discards it after the request. Verify the persistent
SHA before and after; it must be unchanged.

This is `EXPERIMENTAL / RESEARCH ONLY` and is not persistent self-learning.

## 12. Reasoning Policy Seed

`seed_prompt.md` is not present in this branch. There is no normal UI/startup
operation that silently creates a seed sidecar. If the frozen file and a
compatible `sidecar_v1` package are supplied separately, use the research
harness to build it once and load it explicitly. Never auto-teach it at server
startup.

## 13. Troubleshooting

- **Sidecar disabled:** restart with `--sidecar-dir` and `--sidecar-runtime fast`.
- **Profile mismatch:** verify model SHA, Qwen3.5 architecture, and width 2560.
- **CUDA OOM:** reduce context/batch or GPU layers; record the new runtime.
- **UI missing sidecar controls:** build `tools/ui/dist` with Node 20+, rebuild
  the server, restart, and hard-refresh the browser.
- **Learning rejected:** select the intended sidecar and enable persistent
  learning. Check the returned error; empty input and over-context input are
  rejected rather than silently truncated.
- **No hash change:** inspect `writes_realized`, `prediction_matches`, selected
  sidecar ID, and learning state.
- **Save/load rejected:** paths must remain below `--sidecar-dir`.
- **Exact mode is slow:** use Batch for interaction; retain Exact for research
  comparisons.

## 14. What To Send Back

Fill in `FRIEND_TEST_REPORT.md` and `FRIEND_REPRO_CHECKLIST.md`. Include exact
commands, commit and binary hash, model/residual hashes, runtime settings,
before/after/reload sidecar hashes, startup/error logs, and screenshots if
useful. Do not send model weights or credentials.

Run `scripts/friend_sidecar_smoke_test.sh` for the shortest API lifecycle check.
The API reference is `docs/sidecar-server.md`.
