"""Small HTTP client for llama-server's Phase 1 sidecar API.

The client only transports lifecycle and observation data.  It deliberately
does not implement retrieval, plasticity, or any other sidecar mathematics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from http.client import HTTPResponse
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass
class SidecarError(RuntimeError):
    """HTTP or protocol error returned by llama-server."""

    status: int
    payload: Any

    def __str__(self) -> str:
        return f"sidecar request failed ({self.status}): {self.payload}"


class SidecarClient:
    """Dependency-free client for the ``/v1/sidecar`` endpoints."""

    def __init__(self, base_url: str = "http://127.0.0.1:8080", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, body: Mapping[str, Any] | None = None) -> Any:
        data = None if body is None else json.dumps(body, separators=(",", ":")).encode("utf-8")
        request = Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"} if data is not None else {},
            method=method,
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return self._decode(response)
        except HTTPError as error:
            raise SidecarError(error.code, self._decode(error)) from error
        except URLError as error:
            raise SidecarError(0, str(error.reason)) from error

    @staticmethod
    def _decode(response: HTTPResponse) -> Any:
        raw = response.read()
        if not raw:
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return raw.decode("utf-8", errors="replace")

    def status(self, sidecar_id: str | None = None) -> Any:
        path = "/v1/sidecar/status"
        if sidecar_id:
            path += f"?id={sidecar_id}"
        return self._request("GET", path)

    def create(self, sidecar_id: str, learning_enabled: bool = False) -> Any:
        return self._request("POST", "/v1/sidecar/create", {
            "id": sidecar_id,
            "learning_enabled": learning_enabled,
        })

    def load(self, sidecar_id: str, path: str, learning_enabled: bool = False) -> Any:
        return self._request("POST", "/v1/sidecar/load", {
            "id": sidecar_id,
            "path": path,
            "learning_enabled": learning_enabled,
        })

    def save(self, sidecar_id: str, path: str | None = None) -> Any:
        body: dict[str, Any] = {"id": sidecar_id}
        if path is not None:
            body["path"] = path
        return self._request("POST", "/v1/sidecar/save", body)

    def reset(self, sidecar_id: str) -> Any:
        return self._request("POST", "/v1/sidecar/reset", {"id": sidecar_id})

    def unload(self, sidecar_id: str) -> Any:
        return self._request("POST", "/v1/sidecar/unload", {"id": sidecar_id})

    def delete(self, sidecar_id: str) -> Any:
        return self._request("DELETE", f"/v1/sidecar/{sidecar_id}")

    def set_learning(self, sidecar_id: str, enabled: bool) -> Any:
        return self._request("POST", "/v1/sidecar/learning", {
            "id": sidecar_id,
            "enabled": enabled,
        })

    def observe(
        self,
        sidecar_id: str,
        episode_id: str,
        observed_target: str,
        *,
        source: str = "",
        write_direction: list[float] | None = None,
        novelty: float = 0.0,
        surprise: float = 0.0,
        recurrence: float = 0.0,
        consistency: float = 0.0,
    ) -> Any:
        body: dict[str, Any] = {
            "id": sidecar_id,
            "episode_id": episode_id,
            "observed_target": observed_target,
            "source": source,
            "novelty": novelty,
            "surprise": surprise,
            "recurrence": recurrence,
            "consistency": consistency,
        }
        if write_direction is not None:
            body["write_direction"] = write_direction
        return self._request("POST", "/v1/sidecar/observe", body)

    def chat(self, messages: list[Mapping[str, Any]], *, sidecar_id: str | None = None, **options: Any) -> Any:
        body: dict[str, Any] = {"messages": messages, **options}
        if sidecar_id is not None:
            body["sidecar_id"] = sidecar_id
        return self._request("POST", "/v1/chat/completions", body)
