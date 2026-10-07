"""让 CLI 的同步入口与 FastAPI 的异步入口共用一个持久事件循环。

SQLite 异步连接、Semaphore、会话锁都属于同一个循环，避免每次 ask 都
asyncio.run() 导致跨循环使用资源。退出时先清理业务资源，再关闭循环。
"""

import asyncio
import threading
from collections.abc import Coroutine
from concurrent.futures import Future
from typing import Any


class AsyncRunner:
    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="customer-agent-loop", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()
        self.loop.run_until_complete(self.loop.shutdown_asyncgens())
        self.loop.run_until_complete(self.loop.shutdown_default_executor())
        self.loop.close()

    def submit(self, coroutine: Coroutine[Any, Any, Any]) -> Future:
        if self._closed:
            coroutine.close()
            raise RuntimeError("Agent 已关闭")
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self.loop.call_soon_threadsafe(self.loop.stop)
            self._thread.join()
