# Plastic Sidecar Reproduction Checklist

## Build And Runtime

- [ ] Repository cloned or copied.
- [ ] Model obtained independently and SHA256 verified.
- [ ] Node 20+ UI build completed.
- [ ] CMake build completed.
- [ ] `llama-server` launched with `--sidecar-dir`, `--sidecar-runtime fast`, and `--parallel 1`.
- [ ] Model loaded and `/health` passed.
- [ ] Default web UI opened.
- [ ] `Settings -> Chat -> Plastic Sidecar` is visible.
- [ ] Normal no-sidecar chat works.

## Persistent Sidecar

- [ ] Fresh sidecar created.
- [ ] Sidecar selected in the UI.
- [ ] Persistent learning explicitly enabled.
- [ ] Initial sidecar SHA recorded.
- [ ] Multi-token Batch observation accepted.
- [ ] Response reports more than one observed token.
- [ ] Sidecar SHA changed when writes occurred.
- [ ] Chat with the selected sidecar works.
- [ ] Learned sidecar saved.
- [ ] Server stopped and restarted.
- [ ] Learned sidecar reloaded.
- [ ] Saved and reloaded SHA values match.

## Isolation And Controls

- [ ] A second sidecar was created.
- [ ] Teaching sidecar A did not change sidecar B.
- [ ] Switching UI sidecar selection changes the attached sidecar ID.
- [ ] Learning disabled prevents persistent mutation while reads still work.
- [ ] Exact-token mode remains available as the research reference.
- [ ] Chunked mode reports the configured chunk size and continuous token count.

## Optional Research Mode

- [ ] Reasoning-time plasticity remained OFF by default.
- [ ] Optional request-local internal plasticity executed.
- [ ] Persistent sidecar SHA remained unchanged after ephemeral mutation.

## Evidence To Return

- [ ] `FRIEND_TEST_REPORT.md` completed.
- [ ] Startup command and log attached.
- [ ] Build/runtime/model hashes attached.
- [ ] Observation response attached.
- [ ] Before/after/reload sidecar hashes attached.
- [ ] Error or crash logs attached if applicable.
- [ ] No model weights, `.env`, tokens, or credentials included.
