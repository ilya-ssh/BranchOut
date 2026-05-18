# src/progress.py
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProgressManager:
    def __init__(self) -> None:
        self._states: Dict[str, Dict[str, Any]] = {}
        self._listeners: Dict[str, List[asyncio.Queue]] = {}
        self._lock = asyncio.Lock()

    def _public_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "generation_id": state["generation_id"],
            "status": state["status"],
            "stage": state["stage"],
            "message": state["message"],
            "percent": state["percent"],
            "created_at": state["created_at"],
            "updated_at": state["updated_at"],
            "error": state.get("error"),
            "has_result": state.get("result") is not None,
            "extra": dict(state.get("extra") or {}),
            "history": list(state.get("history") or []),
        }

    async def create(
        self,
        generation_id: str,
        *,
        message: str = "Queued",
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return await self.update(
            generation_id,
            status="queued",
            stage="queued",
            message=message,
            percent=0,
            extra=extra or {},
        )

    async def update(
        self,
        generation_id: str,
        *,
        status: Optional[str] = None,
        stage: Optional[str] = None,
        message: Optional[str] = None,
        percent: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        async with self._lock:
            now = _now_iso()

            state = self._states.setdefault(
                generation_id,
                {
                    "generation_id": generation_id,
                    "status": "queued",
                    "stage": "queued",
                    "message": "",
                    "percent": 0,
                    "created_at": now,
                    "updated_at": now,
                    "error": None,
                    "result": None,
                    "extra": {},
                    "history": [],
                },
            )

            if status is not None:
                state["status"] = status
            if stage is not None:
                state["stage"] = stage
            if message is not None:
                state["message"] = message
            if percent is not None:
                bounded = max(0, min(int(percent), 100))
                if state["status"] == "failed":
                    state["percent"] = bounded
                else:
                    state["percent"] = max(int(state.get("percent") or 0), bounded)
            if extra is not None:
                state["extra"] = extra

            state["updated_at"] = now

            event = {
                "type": "progress",
                "generation_id": generation_id,
                "status": state["status"],
                "stage": state["stage"],
                "message": state["message"],
                "percent": state["percent"],
                "ts": now,
                "error": state.get("error"),
                "has_result": state.get("result") is not None,
                "extra": dict(state.get("extra") or {}),
            }

            state["history"].append(event)
            state["history"] = state["history"][-200:]

            listeners = list(self._listeners.get(generation_id, []))
            public_state = self._public_state(state)

        for q in listeners:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass

        return public_state

    async def from_orchestrator(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        generation_id = str(payload["generation_id"])
        return await self.update(
            generation_id,
            status=payload.get("status"),
            stage=payload.get("stage"),
            message=payload.get("message"),
            percent=payload.get("percent"),
            extra=payload.get("extra") or {},
        )

    async def complete(
        self,
        generation_id: str,
        *,
        result: Dict[str, Any],
        message: str = "Generation completed",
    ) -> Dict[str, Any]:
        async with self._lock:
            now = _now_iso()
            state = self._states.setdefault(
                generation_id,
                {
                    "generation_id": generation_id,
                    "status": "queued",
                    "stage": "queued",
                    "message": "",
                    "percent": 0,
                    "created_at": now,
                    "updated_at": now,
                    "error": None,
                    "result": None,
                    "extra": {},
                    "history": [],
                },
            )
            state["result"] = result
            state["error"] = None

        return await self.update(
            generation_id,
            status="completed",
            stage="completed",
            message=message,
            percent=100,
        )

    async def fail(
        self,
        generation_id: str,
        *,
        error: str,
        message: str = "Generation failed",
    ) -> Dict[str, Any]:
        async with self._lock:
            now = _now_iso()
            state = self._states.setdefault(
                generation_id,
                {
                    "generation_id": generation_id,
                    "status": "queued",
                    "stage": "queued",
                    "message": "",
                    "percent": 0,
                    "created_at": now,
                    "updated_at": now,
                    "error": None,
                    "result": None,
                    "extra": {},
                    "history": [],
                },
            )
            state["error"] = error

        return await self.update(
            generation_id,
            status="failed",
            stage="failed",
            message=message,
            percent=100,
            extra={"error": error},
        )

    async def get(self, generation_id: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            state = self._states.get(generation_id)
            if state is None:
                return None
            return self._public_state(state)

    async def get_result(self, generation_id: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            state = self._states.get(generation_id)
            if state is None:
                return None
            return state.get("result")

    async def subscribe(self, generation_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        async with self._lock:
            self._listeners.setdefault(generation_id, []).append(q)
        return q

    async def unsubscribe(self, generation_id: str, queue: asyncio.Queue) -> None:
        async with self._lock:
            listeners = self._listeners.get(generation_id) or []
            if queue in listeners:
                listeners.remove(queue)
            if not listeners and generation_id in self._listeners:
                self._listeners.pop(generation_id, None)


progress_manager = ProgressManager()