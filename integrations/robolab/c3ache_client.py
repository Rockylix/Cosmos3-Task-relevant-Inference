# SPDX-License-Identifier: Apache-2.0
"""RoboLab client with explicit per-environment planning/episode identities.

Observation processing, open-loop chunking, retry behavior and action conversion
are inherited unchanged from RoboLab's Cosmos3 baseline client.
"""

from __future__ import annotations

import logging
import uuid

from policies.cosmos3.client import Cosmos3Client

logger = logging.getLogger(__name__)


class C3acheCosmos3Client(Cosmos3Client):
    """Keep different envs and repeated trials of the same task out of each other's cache."""

    def __init__(self, remote_host: str = "localhost", remote_port: int = 8000):
        """Create a unique client session and lazily allocate per-env episode IDs."""
        self._session_id = uuid.uuid4().hex
        self._episode_ids: dict[int, str] = {}
        self._planning_chunks: dict[int, int] = {}
        self._instructions: dict[int, str] = {}
        self.last_cache_stats: dict[int, dict] = {}
        super().__init__(remote_host=remote_host, remote_port=remote_port)

    def infer(self, obs, instruction: str, *, env_id: int = 0) -> dict:
        """Start a fresh episode when the task changes, even inside an open-loop chunk."""
        if env_id in self._instructions and self._instructions[env_id] != instruction:
            self.reset(env_id=env_id)
        self._instructions[env_id] = instruction
        return super().infer(obs, instruction, env_id=env_id)

    def _extract_observation(self, raw_obs: dict, *, env_id: int = 0) -> dict:
        """Preserve baseline images/state and carry env_id to the request hook."""
        extracted = super()._extract_observation(raw_obs, env_id=env_id)
        extracted["_c3ache_env_id"] = env_id
        return extracted

    def _pack_request(self, extracted_obs: dict, instruction: str) -> dict:
        """Add identities only when the base client requests a new action chunk."""
        env_id = extracted_obs["_c3ache_env_id"]
        episode = self._episode_ids.setdefault(env_id, uuid.uuid4().hex)
        request = super()._pack_request(extracted_obs, instruction)
        request.update(
            session_id=f"{self._session_id}:{env_id}",
            episode_id=episode,
            chunk_id=self._planning_chunks.get(env_id, 0),
        )
        return request

    def _query_server(self, request: dict) -> dict:
        """Advance chunk_id only after success; retries reuse the original identity.

        If the server completed a request whose reply was lost, its duplicate
        chunk check forces a dense refresh rather than applying stale residuals.
        """
        response = super()._query_server(request)
        env_id = int(request["session_id"].rsplit(":", 1)[1])
        self._planning_chunks[env_id] = request["chunk_id"] + 1
        self.last_cache_stats[env_id] = dict(response.get("c3ache", {}))
        return response

    def reset(self, *, env_id: int | None = None) -> None:
        """Notify the server and rotate episode IDs even if the connection is lost.

        A fresh UUID on the next observation provides isolation independently
        of reset delivery. Reset control messages never request an action.
        """
        env_ids = list(self._episode_ids) if env_id is None else [env_id]
        super().reset(env_id=env_id)
        for index in env_ids:
            episode = self._episode_ids.pop(index, None)
            self._planning_chunks.pop(index, None)
            self._instructions.pop(index, None)
            self.last_cache_stats.pop(index, None)
            if episode is None:
                continue
            try:
                response = self._infer_with_retry(
                    {
                        "c3ache_control": "reset",
                        "session_id": f"{self._session_id}:{index}",
                        "episode_id": episode,
                    }
                )
                if response.get("reset") is not True:
                    raise RuntimeError("Server did not acknowledge the episode reset")
            except Exception as exc:
                logger.warning("C3ache reset delivery failed; next request uses a fresh episode ID: %s", exc)
