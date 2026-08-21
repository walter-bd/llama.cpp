# Architecture Before and After

## Before

The reference path forced each observed token through the existing generation transaction. For an N-token experience it performed N model steps, allowing the sidecar state written at token t to affect acquisition at token t+1. This remains available as `exact-token`.

## After

`POST /v1/sidecar/observe_sequence` accepts text, tokenizes it with the loaded model tokenizer, and offers:

- `exact-token`: existing recurrent reference behavior, N decode calls.
- `chunked`: causal acquisition in chunks, approximately `ceil(N/K)` decode calls.
- `batch`: one causal acquisition decode where capacity permits.

For fast modes, the server decodes the explicit context and observed predictor sequence using normal causal KV semantics. At each observed position it captures the validated `result_norm` source key and the logits predicting the next externally observed token. It computes the unchanged writer direction:

```text
q_t = normalize(W_out[y_t] - W_out[y_hat_t])
```

All captured token-local events are then applied in token order, preserving ordered counters, gates, and component mutation. A transaction snapshot restores the original sidecar on fatal failure.

Fast acquisition deliberately does not feed writes created inside a batch/chunk back into later acquisition hidden states. Current `chunked` mode reduces decode granularity but, like `batch`, commits writes only after the complete acquisition. The API reports this explicitly. It is a fast approximation, not exact-token recurrence.
