# Benchmark Readiness

An external harness can keep one Qwen model loaded while creating isolated
sidecars, submitting explicit observations, evaluating, saving/reloading, and
deleting sidecars:

```python
sid = client.create("task-1")
episode = client.chat(messages, sidecar_id=sid)
client.observe(sid, episode["sidecar_episode_id"], outcome="D")
prediction = client.chat(test_messages, sidecar_id=sid)
client.delete(sid)
```

The Python client transports requests only; it does not implement retrieval,
authority, or plasticity. Requests without `sidecar_id` use normal GGUF
inference. Setting `LLAMA_SPARSE_RESIDUAL` adds the independent residual
module. Supplying both that environment variable and `--sidecar-dir` enables
explicit residual-plus-sidecar composition.

Before production-scale throughput claims, run a dedicated concurrency and
latency benchmark. Phase 1F only established the functional lifecycle and
single-slot engineering path.
