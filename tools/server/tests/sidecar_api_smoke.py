#!/usr/bin/env python3
"""Local Phase 1 lifecycle smoke test for llama-server sidecars.

This is deliberately an engineering test.  It checks the HTTP lifecycle,
explicit observation boundary, isolation, and restart/load behavior.  It does
not assert scientific numerical parity with the locked research runtime.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

from utils import ServerError, ServerProcess

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/server"))
from sidecar_client import SidecarClient, SidecarError

DEFAULT_MODEL = Path("/home/walt/models/qwen35-q3-sparse/Qwen3.5-4B-UD-Q3_K_XL.gguf")
DEFAULT_SIDECAR = (
    ROOT.parent
    / "evo-prunning"
    / "analysis/qwen35_detachable_sidecar/sidecars/p_init"
)


def expect_error(client, method, *args, **kwargs) -> dict:
    try:
        method(*args, **kwargs)
    except Exception as error:
        if isinstance(error, (ServerError, SidecarError)):
            if isinstance(error, SidecarError):
                return {"status": error.status, "body": error.payload}
            return {"status": error.code, "body": error.body}
        raise
    raise AssertionError("request unexpectedly succeeded")


def run(model: Path, sidecar: Path, server_path: Path) -> dict:
    if not model.is_file():
        raise RuntimeError(f"model does not exist: {model}")
    if not (sidecar / "manifest.json").is_file():
        raise RuntimeError(f"sidecar package does not exist: {sidecar}")

    server = ServerProcess()
    server.server_path = str(server_path)
    server.model_file = str(model)
    server.model_hf_repo = None
    server.model_hf_file = None
    server.offline = True
    server.no_ui = True
    server.n_slots = 1
    server.n_ctx = 512
    server.n_batch = 32
    server.n_predict = 4
    server.temperature = 0.0

    with tempfile.TemporaryDirectory(prefix="llama-sidecar-smoke-") as temp_dir:
        root = Path(temp_dir)
        server.sidecar_dir = str(root / "packages")
        shutil.copytree(sidecar, root / "packages" / "p_init")
        client = None
        try:
            server.start(timeout_seconds=180)
            client = SidecarClient(
                f"http://{server.server_host}:{server.server_port}", timeout=60
            )

            initial = client.status()
            assert initial["sidecars"] == []

            # The local Qwen profile is selected at model bind time.  These
            # checks exercise the public profile contract without asserting
            # locked-runtime numerical parity.
            client.load("profile-reference", "p_init")
            profile_status = client.status("profile-reference")
            assert profile_status["injection_profile_id"] == "qwen35-4b-step30-v1"
            assert profile_status["profile_supported"] is True
            assert profile_status["query_hook"] == "result_norm"
            assert profile_status["supported_injection_sites"] == 2
            assert profile_status["research_injection_sites"] == 3
            assert profile_status["final_logit_injection_supported"] is True
            assert profile_status["recursive_state_path_supported"] is True
            assert profile_status["injection_mode"] == "validated_step30_runtime"
            client.unload("profile-reference")

            ordinary = client.chat(
                [{"role": "user", "content": "Say one word."}],
                max_tokens=1,
                stream=False,
            )
            assert "sidecar_episode_id" not in ordinary

            bad_package = root / "packages" / "bad-checksum"
            shutil.copytree(root / "packages" / "p_init", bad_package)
            bad_vectors = bad_package / "vectors.f32"
            corrupted = bytearray(bad_vectors.read_bytes())
            corrupted[0] ^= 1
            bad_vectors.write_bytes(corrupted)
            bad_load = expect_error(client, client.load, "bad", "bad-checksum")
            assert bad_load["status"] == 400

            created = client.create("agent")
            assert created["components"] == 0
            duplicate = expect_error(client, client.create, "agent")

            client.set_learning("agent", True)
            first = client.chat(
                [{"role": "user", "content": "Say one word."}],
                sidecar_id="agent",
                sidecar_capture_episode=True,
                max_tokens=1,
                stream=False,
            )
            episode_id = first.get("sidecar_episode_id")
            assert episode_id, first
            tokenized = client.tokenize("D")
            assert len(tokenized["tokens"]) == 1, tokenized

            legacy_vector = expect_error(
                client,
                client._request,
                "POST",
                "/v1/sidecar/observe",
                {
                    "id": "agent",
                    "episode_id": episode_id,
                    "observed_target": "D",
                    "write_direction": [0.0],
                },
            )
            assert legacy_vector["status"] == 400

            multi_token_observe = expect_error(
                client,
                client.observe,
                "agent",
                episode_id,
                observed_token="D D",
            )
            assert multi_token_observe["status"] == 400

            observed = client.observe(
                "agent",
                episode_id,
                observed_token="D",
                observed_target="D",
            )
            assert observed["updated"] is True
            written_sha = observed["sha256_after"]
            written_components = observed["components"]

            client.set_learning("agent", False)
            no_write_before = client.status("agent")["sha256"]
            no_write_response = client.chat(
                [{"role": "user", "content": "Say one word."}],
                sidecar_id="agent",
                max_tokens=1,
                stream=False,
            )
            no_write_after = client.status("agent")["sha256"]
            assert no_write_before == no_write_after == written_sha
            assert "sidecar_episode_id" not in no_write_response
            agent_after_read = client.status("agent")
            assert agent_after_read["query_captures"] >= 2
            disabled_observe = client.observe(
                "agent",
                "missing-episode",
                observed_token_id=tokenized["tokens"][0],
            )
            assert disabled_observe["updated"] is False

            client.set_learning("agent", True)
            invalid_observe = expect_error(
                client,
                client.observe,
                "agent",
                "missing-episode",
                observed_token_id=tokenized["tokens"][0],
            )
            assert invalid_observe["status"] == 400

            chat_before = client.status("agent")["sha256"]
            multi_token_generation = client.chat(
                [{"role": "user", "content": "Say two words."}],
                sidecar_id="agent",
                max_tokens=2,
                stream=False,
            )
            assert "sidecar_episode_id" not in multi_token_generation
            assert client.status("agent")["sha256"] == chat_before

            invalid_episode_capture = expect_error(
                client,
                client.chat,
                [{"role": "user", "content": "Say two words."}],
                sidecar_id="agent",
                sidecar_capture_episode=True,
                max_tokens=2,
                stream=False,
            )
            assert invalid_episode_capture["status"] == 400
            client.set_learning("agent", False)

            # The fast sequence API tokenizes internally, captures one causal
            # event per observed token, and commits writes transactionally.
            client.create("sequence", learning_enabled=True)
            sequence_before = client.status("sequence")["sha256"]
            batch_observation = client.observe_sequence(
                "sequence", "The room costs five gold pieces.", mode="batch"
            )
            assert batch_observation["learning_mode"] == "batch"
            assert batch_observation["tokens_observed"] > 1
            assert batch_observation["write_opportunities"] == batch_observation["tokens_observed"]
            assert len(batch_observation["events"]) == batch_observation["tokens_observed"]
            assert [event["token_position"] for event in batch_observation["events"]] == list(
                range(batch_observation["tokens_observed"])
            )
            sequence_after_batch = client.status("sequence")["sha256"]
            if batch_observation["writes_realized"]:
                assert sequence_after_batch != sequence_before

            chunked_observation = client.observe_sequence(
                "sequence",
                "The forge takes three hours.",
                mode="chunked",
                chunk_size=4,
            )
            assert chunked_observation["learning_mode"] == "chunked"
            assert chunked_observation["chunk_size"] == 4
            assert chunked_observation["chunks_processed"] >= 1
            assert [event["token_position"] for event in chunked_observation["events"]] == list(
                range(chunked_observation["tokens_observed"])
            )

            # The sequence endpoint must retain the historical forced-token
            # transaction as the exact reference path.
            exact_text = "The room costs five gold."
            exact_token_count = len(client.tokenize(exact_text)["tokens"])
            client.create("sequence-exact", learning_enabled=True)
            exact_before = client.status("sequence-exact")["sha256"]
            exact_observation = client.observe_sequence(
                "sequence-exact", exact_text, mode="exact-token"
            )
            exact_events = exact_observation.get("sidecar_experience_events", [])
            assert exact_observation["tokens_predicted"] == exact_token_count
            assert len(exact_events) == exact_token_count
            if any(event.get("updated") for event in exact_events):
                assert client.status("sequence-exact")["sha256"] != exact_before
            client.delete("sequence-exact")

            client.set_learning("sequence", False)
            disabled_sequence_before = client.status("sequence")["sha256"]
            disabled_sequence = client.observe_sequence(
                "sequence", "Learning is disabled here.", mode="batch"
            )
            assert disabled_sequence["updated"] is False
            assert disabled_sequence["reason"] == "learning_disabled"
            assert client.status("sequence")["sha256"] == disabled_sequence_before
            client.delete("sequence")

            saved = client.save("agent", "saved")
            assert saved["sha256"] == written_sha
            saved_by_id = client.save("agent")
            assert saved_by_id["sha256"] == written_sha
            client.unload("agent")
            client.load("agent", "saved")
            assert client.status("agent")["sha256"] == written_sha

            client.create("sidecar-a")
            client.create("sidecar-b")
            client.set_learning("sidecar-a", True)
            a_response = client.chat(
                [{"role": "user", "content": "Say one word."}],
                sidecar_id="sidecar-a",
                sidecar_capture_episode=True,
                max_tokens=1,
                stream=False,
            )
            assert a_response.get("sidecar_episode_id")
            assert client.status("sidecar-b")["retrievals"] == 0
            b_before = client.status("sidecar-b")["sha256"]
            client.chat(
                [{"role": "user", "content": "Say one word."}],
                sidecar_id="sidecar-b",
                max_tokens=1,
                stream=False,
            )
            assert client.status("sidecar-b")["sha256"] == b_before
            assert client.status("sidecar-a")["retrievals"] >= 1

            client.delete("sidecar-a")
            client.delete("sidecar-b")
            client.reset("agent")
            assert client.status("agent")["components"] == 0
            client.unload("agent")
            unknown = expect_error(client, client.status, "agent")

            # Exercise save -> process restart -> load through HTTP.
            server.stop()
            server.start(timeout_seconds=180)
            client = SidecarClient(
                f"http://{server.server_host}:{server.server_port}", timeout=60
            )
            client.load("restarted", "saved")
            restart_status = client.status("restarted")
            assert restart_status["components"] == written_components
            assert restart_status["sha256"] == written_sha
            client.delete("restarted")

            return {
                "server": str(server_path),
                "model": str(model),
                "sidecar": str(sidecar),
                "duplicate_create": duplicate,
                "bad_checksum_load": bad_load,
                "legacy_vector": legacy_vector,
                "multi_token_observe": multi_token_observe,
                "multi_token_generation": multi_token_generation,
                "batch_observation": batch_observation,
                "chunked_observation": chunked_observation,
                "exact_observation": exact_observation,
                "disabled_sequence": disabled_sequence,
                "disabled_observe": disabled_observe,
                "invalid_observe": invalid_observe,
                "unknown_status": unknown,
                "written_sha": written_sha,
                "written_components": written_components,
                "restart_components": restart_status["components"],
                "profile": {
                    "id": profile_status["injection_profile_id"],
                    "supported_sites": profile_status["supported_injection_sites"],
                    "research_sites": profile_status["research_injection_sites"],
                    "final_logit_supported": profile_status["final_logit_injection_supported"],
                    "recursive_supported": profile_status["recursive_state_path_supported"],
                },
            }
        finally:
            server.stop()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument(
        "--server",
        type=Path,
        default=ROOT / "build/bin/llama-server",
    )
    args = parser.parse_args()
    result = run(args.model, args.sidecar, args.server)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
