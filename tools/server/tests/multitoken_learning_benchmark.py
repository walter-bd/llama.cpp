#!/usr/bin/env python3
"""Compact engineering benchmark for multi-token sidecar acquisition modes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any

from utils import ServerProcess

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/server"))
from sidecar_client import SidecarClient

DEFAULT_MODEL = Path("/home/walt/models/qwen35-q3-sparse/Qwen3.5-4B-UD-Q3_K_XL.gguf")
DEFAULT_EXPERIENCE = (
    "The room costs five gold pieces, includes breakfast, overlooks the old harbor, "
    "and remains available to quiet travelers who arrive before sunset with a written "
    "reservation during the winter market."
)
MODES = (
    ("EXACT", "exact-token", None),
    ("CHUNK4", "chunked", 4),
    ("CHUNK16", "chunked", 16),
    ("CHUNK32", "chunked", 32),
    ("BATCH", "batch", None),
)
BEHAVIOR_CASES = (
    {
        "id": "overwrite-room-price",
        "regime": "overwrite",
        "experience": "The room costs five gold pieces.",
        "probe": "The room costs",
        "target": "five gold pieces",
    },
    {
        "id": "novel-neral-seal",
        "regime": "novel",
        "experience": "The Neral seal is cobalt amber.",
        "probe": "The Neral seal is",
        "target": "cobalt amber",
    },
)


def cosine(left: list[float], right: list[float]) -> float | None:
    if not left or len(left) != len(right):
        return None
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    return dot / (left_norm * right_norm)


def summarize(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "minimum": None}
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "minimum": min(values),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def normalize_result(
    label: str,
    mode: str,
    chunk_size: int | None,
    response: dict[str, Any],
    token_count: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    events = response.get("events", response.get("sidecar_experience_events", []))
    if len(events) != token_count:
        raise AssertionError(f"{label}: expected {token_count} events, got {len(events)}")

    for position, event in enumerate(events):
        event_position = event.get("token_position", event.get("sequence_index"))
        if event_position != position:
            raise AssertionError(f"{label}: event position {event_position} != {position}")
        if mode == "chunked" and event.get("chunk_index") != position // int(chunk_size):
            raise AssertionError(f"{label}: incorrect chunk index at position {position}")

    if mode == "exact-token":
        timings = response["timings"]
        elapsed_ms = float(timings["prompt_ms"]) + float(timings["predicted_ms"])
        throughput = token_count / max(elapsed_ms / 1000.0, 1.0e-9)
        decode_calls = token_count
        writes = sum(bool(event.get("updated")) for event in events)
        state_before = events[0].get("sha256_before")
        state_after = events[-1].get("sha256_after")
    else:
        elapsed_ms = float(response["elapsed_ms"])
        throughput = float(response["observed_tokens_per_second"])
        decode_calls = int(response["llama_decode_calls"])
        writes = int(response["writes_realized"])
        state_before = response.get("sha256_before")
        state_after = response.get("sha256_after")

    compact = {
        "label": label,
        "learning_mode": mode,
        "chunk_size": chunk_size,
        "tokens_observed": token_count,
        "writes_realized": writes,
        "elapsed_ms": elapsed_ms,
        "observed_tokens_per_second": throughput,
        "llama_decode_calls": decode_calls,
        "sha256_before": state_before,
        "sha256_after": state_after,
        "fast_approximation": mode != "exact-token",
        "plastic_state_recurs_between_chunks": response.get(
            "plastic_state_recurs_between_chunks", mode == "exact-token"
        ),
    }
    return compact, events


def run(model: Path, server_path: Path, output: Path, experience: str) -> dict[str, Any]:
    if not model.is_file():
        raise RuntimeError(f"model does not exist: {model}")
    if not server_path.is_file():
        raise RuntimeError(f"server does not exist: {server_path}")

    server = ServerProcess()
    server.server_path = str(server_path)
    server.model_file = str(model)
    server.model_hf_repo = None
    server.model_hf_file = None
    server.offline = True
    server.no_ui = True
    server.n_slots = 1
    server.n_ctx = 512
    server.n_batch = 64
    server.n_predict = 8
    server.temperature = 0.0

    with tempfile.TemporaryDirectory(prefix="llama-multitoken-benchmark-") as temp_dir:
        server.sidecar_dir = str(Path(temp_dir) / "sidecars")
        server.start(timeout_seconds=180)
        try:
            client = SidecarClient(
                f"http://{server.server_host}:{server.server_port}", timeout=180
            )
            token_count = len(client.tokenize(experience)["tokens"])
            if token_count <= 32:
                raise AssertionError(f"benchmark experience is only {token_count} tokens")

            mode_results: dict[str, dict[str, Any]] = {}
            captured_events: dict[str, list[dict[str, Any]]] = {}
            generations: list[dict[str, Any]] = []

            baseline_generation = client._request(
                "POST",
                "/completion",
                {
                    "prompt": "The room costs",
                    "n_predict": 8,
                    "stream": False,
                    "temperature": 0.0,
                    "cache_prompt": False,
                },
            )
            generations.append({
                "label": "F",
                "content": baseline_generation.get("content", ""),
                "tokens_predicted": baseline_generation.get("tokens_predicted"),
            })

            for label, mode, chunk_size in MODES:
                sidecar_id = f"benchmark-{label.lower()}"
                client.create(sidecar_id, learning_enabled=True)
                response = client.observe_sequence(
                    sidecar_id,
                    experience,
                    mode=mode,
                    chunk_size=chunk_size,
                    debug_capture_vectors=True,
                )
                compact, events = normalize_result(
                    label, mode, chunk_size, response, token_count
                )
                compact["components_after"] = client.status(sidecar_id)["components"]
                mode_results[label] = compact
                captured_events[label] = events

                client.set_learning(sidecar_id, False)
                generation = client._request(
                    "POST",
                    "/completion",
                    {
                        "prompt": "The room costs",
                        "sidecar_id": sidecar_id,
                        "n_predict": 8,
                        "stream": False,
                        "temperature": 0.0,
                        "cache_prompt": False,
                    },
                )
                generations.append({
                    "label": label,
                    "content": generation.get("content", ""),
                    "tokens_predicted": generation.get("tokens_predicted"),
                })

            exact_events = captured_events["EXACT"]
            key_rows: list[dict[str, Any]] = []
            writer_rows: list[dict[str, Any]] = []
            baseline_rows: list[dict[str, Any]] = []
            gate_rows: list[dict[str, Any]] = []
            equivalence_summary: dict[str, Any] = {}
            for label, _, _ in MODES[1:]:
                key_values: list[float] = []
                writer_values: list[float] = []
                baseline_matches = 0
                gate_matches = 0
                for position, (exact, fast) in enumerate(
                    zip(exact_events, captured_events[label])
                ):
                    key_cosine = cosine(exact["debug_key"], fast["debug_key"])
                    writer_cosine = cosine(
                        exact.get("debug_write_direction", []),
                        fast.get("debug_write_direction", []),
                    )
                    same_baseline = (
                        exact["baseline_greedy_token_id"]
                        == fast["baseline_greedy_token_id"]
                    )
                    same_gate = math.isclose(
                        float(exact["G"]), float(fast["G"]), abs_tol=1.0e-12
                    )
                    if key_cosine is not None:
                        key_values.append(key_cosine)
                    if writer_cosine is not None:
                        writer_values.append(writer_cosine)
                    baseline_matches += int(same_baseline)
                    gate_matches += int(same_gate)
                    key_rows.append({
                        "mode": label,
                        "token_position": position,
                        "cosine": key_cosine,
                    })
                    writer_rows.append({
                        "mode": label,
                        "token_position": position,
                        "cosine": writer_cosine,
                    })
                    baseline_rows.append({
                        "mode": label,
                        "token_position": position,
                        "exact_token": exact["baseline_greedy_token_id"],
                        "fast_token": fast["baseline_greedy_token_id"],
                        "same": same_baseline,
                    })
                    gate_rows.append({
                        "mode": label,
                        "token_position": position,
                        "exact_gate": exact["G"],
                        "fast_gate": fast["G"],
                        "same": same_gate,
                    })
                equivalence_summary[label] = {
                    "key_cosine": summarize(key_values),
                    "writer_cosine": summarize(writer_values),
                    "baseline_token_agreement": baseline_matches / token_count,
                    "gate_agreement": gate_matches / token_count,
                }

            throughput = {
                label: {
                    key: value
                    for key, value in result.items()
                    if key in {
                        "learning_mode",
                        "chunk_size",
                        "tokens_observed",
                        "writes_realized",
                        "elapsed_ms",
                        "observed_tokens_per_second",
                        "llama_decode_calls",
                    }
                }
                for label, result in mode_results.items()
            }
            exact_speed = mode_results["EXACT"]["observed_tokens_per_second"]
            for label, result in throughput.items():
                result["speedup_vs_exact"] = (
                    result["observed_tokens_per_second"] / exact_speed
                )

            write_json(output / "benchmarks/throughput.json", throughput)
            write_json(output / "benchmarks/performance_by_mode.json", mode_results)
            write_jsonl(output / "equivalence/key_cosines.jsonl", key_rows)
            write_jsonl(output / "equivalence/writer_cosines.jsonl", writer_rows)
            write_jsonl(
                output / "equivalence/baseline_token_agreement.jsonl", baseline_rows
            )
            write_jsonl(output / "equivalence/gate_comparison.jsonl", gate_rows)
            write_json(
                output / "equivalence/sidecar_state_comparison.json",
                {
                    label: {
                        "components_after": result["components_after"],
                        "sha256_after": result["sha256_after"],
                    }
                    for label, result in mode_results.items()
                },
            )
            write_jsonl(output / "behavior/generation_samples.jsonl", generations)

            behavior_rows: list[dict[str, Any]] = []
            for case in BEHAVIOR_CASES:
                baseline = client._request(
                    "POST",
                    "/completion",
                    {
                        "prompt": case["probe"],
                        "n_predict": 12,
                        "stream": False,
                        "temperature": 0.0,
                        "cache_prompt": False,
                    },
                )
                baseline_content = baseline.get("content", "")
                behavior_rows.append({
                    **case,
                    "arm": "F",
                    "content": baseline_content,
                    "target_occurs": case["target"].casefold() in baseline_content.casefold(),
                })

                for label, mode, chunk_size in MODES:
                    sidecar_id = f"behavior-{case['id']}-{label.lower()}"
                    client.create(sidecar_id, learning_enabled=True)
                    learned = client.observe_sequence(
                        sidecar_id,
                        case["experience"],
                        mode=mode,
                        chunk_size=chunk_size,
                    )
                    client.set_learning(sidecar_id, False)
                    generated = client._request(
                        "POST",
                        "/completion",
                        {
                            "prompt": case["probe"],
                            "sidecar_id": sidecar_id,
                            "n_predict": 12,
                            "stream": False,
                            "temperature": 0.0,
                            "cache_prompt": False,
                        },
                    )
                    content = generated.get("content", "")
                    behavior_rows.append({
                        **case,
                        "arm": label,
                        "content": content,
                        "target_occurs": case["target"].casefold() in content.casefold(),
                        "tokens_observed": learned.get("tokens_observed"),
                        "writes_realized": learned.get(
                            "writes_realized",
                            sum(bool(event.get("updated")) for event in learned.get("sidecar_experience_events", [])),
                        ),
                    })

            write_jsonl(
                output / "behavior/overwrite_results.jsonl",
                [row for row in behavior_rows if row["regime"] == "overwrite"],
            )
            write_jsonl(
                output / "behavior/novel_results.jsonl",
                [row for row in behavior_rows if row["regime"] == "novel"],
            )

            summary = {
                "experience": experience,
                "tokens_observed": token_count,
                "throughput": throughput,
                "equivalence": equivalence_summary,
                "generation_samples": generations,
                "behavioral_pilot": behavior_rows,
                "status": "batch_learning_fast_approximation_measured",
            }
            write_json(output / "summary.json", summary)
            write_json(
                output / "runtime.json",
                {
                    "runtime_epoch": output.name,
                    "model_path": str(model),
                    "model_sha256": sha256(model),
                    "server_path": str(server_path),
                    "server_sha256": sha256(server_path),
                    "context_size": server.n_ctx,
                    "batch_size": server.n_batch,
                    "parallel": server.n_slots,
                    "temperature": server.temperature,
                },
            )
            return summary
        finally:
            server.stop()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--server", type=Path, default=ROOT / "build/bin/llama-server"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experience", default=DEFAULT_EXPERIENCE)
    args = parser.parse_args()
    result = run(args.model, args.server, args.output, args.experience)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
