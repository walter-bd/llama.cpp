# Multi-Token Batch Learning Contract

The writer, reader, gate, eta, authority, topology, and sidecar serialization remain unchanged. The intervention changes only model-forward execution granularity.

The fixed experience is processed by `EXACT`, `CHUNK4`, `CHUNK16`, `CHUNK32`, and `BATCH`. Each mode must preserve causal teacher forcing, one event opportunity per observed token, and ordered gate/state mutation. Fast modes are not assumed equivalent to exact-token recurrence.

Primary engineering measurements are acquisition throughput, model decode calls, per-position key cosine, baseline-token agreement, writer-direction cosine, gate agreement, component count, and short downstream generation. Exact recall is secondary and is not required for engineering success.

The overwrite-like and novel behavioral examples are frozen in `tools/server/tests/multitoken_learning_benchmark.py`. No authority or learning-rule parameters are tuned from results.
