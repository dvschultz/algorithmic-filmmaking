"""Serial owner-thread execution for retained MCP project sessions."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Callable

from core.spine.project_sessions import ProjectSessions


class SessionRuntime:
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="project-session"
        )
        self._sessions = ProjectSessions()
        self._closed = False

    async def call(self, operation: Callable, *args, **kwargs) -> dict:
        if self._closed:
            raise RuntimeError("Project session runtime is closed")
        future = self._executor.submit(
            partial(operation, self._sessions, *args, **kwargs)
        )
        return await asyncio.wrap_future(future)

    async def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await asyncio.shield(
                asyncio.wrap_future(self._executor.submit(self._sessions.close_all))
            )
        finally:
            await asyncio.to_thread(self._executor.shutdown, wait=True)
