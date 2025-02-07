import asyncio
import json
import threading
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Coroutine, Iterator, TypeVar
import textwrap

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Import protocols from inspect_ai
from ..core.display import (
    Display,
    Progress,
    TaskScreen,
    TaskDisplay,
    TaskSpec,
    TaskProfile,
    TaskResult,
    TaskDisplayMetric,
)

TR = TypeVar("TR")

class ServerDisplay(Display):
    """
    This display does not do rendering itself, but instead acts as a server that a client can connect to.

    Endpoints available:
     - GET /          : Returns a status message with available endpoints.
     - GET /logs      : Returns the logs and current progress.
     - GET /progress  : Returns only the current progress.
     - GET /metrics   : Returns the stored metrics.
     - WS  /ws       : WebSocket endpoint for streaming updates.
    """
    def __init__(self, host: str = "localhost", port: int = 8080) -> None:
        print(f"Initializing ServerDisplay with host: {host}, port: {port}")
        self.host: str = host
        self.port: int = port
        self.logs: list[str] = []
        self.log_lock = threading.Lock()
        # We simply keep one progress state; you could extend this to support many.
        self.current_progress: dict[str, Any] = {}

        # New: Initialize metrics storage with its own lock.
        self.metrics: dict[str, list[Any]] = {}
        self.metrics_lock = threading.Lock()

        # Create FastAPI app and attach this instance to its state.
        self.app = FastAPI()
        self.app.state.server_display = self

        # Define endpoints.
        self.app.add_api_route(path="/", endpoint=self.index, methods=["GET"])
        self.app.add_api_route(path="/logs", endpoint=self.get_logs, methods=["GET"])
        self.app.add_api_route(path="/progress", endpoint=self.get_progress, methods=["GET"])
        # New endpoint for querying metrics.
        self.app.add_api_route(path="/metrics", endpoint=self.get_metrics, methods=["GET"])
        self.app.websocket("/ws")(self.websocket_endpoint)

        # Start the server in a background thread.
        thread = threading.Thread(target=self._run_uvicorn, daemon=True)
        thread.start()

        # Log the URL.
        self.print(message=f"Inspect ServerDisplay running at http://{self.host}:{self.port}")

    def _run_uvicorn(self) -> None:
        uvicorn.run(self.app, host=self.host, port=self.port, log_level="debug")

    async def index(self) -> dict[str, Any]:
        # A simple JSON status message indicating the available endpoints.
        return {
            "message": "Inspect Server running",
            "endpoints": ["/logs", "/progress", "/metrics", "/ws"]
        }

    async def get_logs(self) -> dict[str, Any]:
        with self.log_lock:
            return {"logs": self.logs, "progress": self.current_progress}

    async def get_progress(self) -> dict[str, Any]:
        with self.log_lock:
            return {"progress": self.current_progress}

    async def get_metrics(self) -> dict[str, Any]:
        with self.metrics_lock:
            return {"metrics": self.metrics}

    async def websocket_endpoint(self, websocket: WebSocket) -> None:
        print("Incoming WebSocket connection.")
        await websocket.accept()
        print("WebSocket connection accepted.")
        try:
            while True:
                await asyncio.sleep(1)
                with self.log_lock, self.metrics_lock:
                    data = {
                        "logs": self.logs,
                        "progress": self.current_progress,
                        "metrics": self.metrics,
                    }
                await websocket.send_text(json.dumps(data))
        except Exception as e:
            print(f"Exception in WebSocket endpoint: {e}")
        finally:
            print("Closing WebSocket connection.")

    def print(self, message: str) -> None:
        with self.log_lock:
            self.logs.append(message)

    @contextmanager
    def progress(self, total: int) -> Iterator[Progress]:
        prog = ServerProgress(display=self, total=total)
        try:
            yield prog
        finally:
            prog.complete()

    def run_task_app(self, main: Coroutine[Any, Any, TR]) -> TR:
        return asyncio.run(main)

    @contextmanager
    def suspend_task_app(self) -> Iterator[None]:
        # For simplicity, nothing is suspended in this server display.
        yield

    @asynccontextmanager
    async def task_screen(self, tasks: list[TaskSpec], parallel: bool) -> AsyncIterator[TaskScreen]:
        screen = ServerTaskScreen(display=self)
        try:
            yield screen
        finally:
            # Any cleanup for the task screen if necessary.
            pass

    @contextmanager
    def task(self, profile: TaskProfile) -> Iterator[TaskDisplay]:
        task_display = ServerTaskDisplay(display=self, profile=profile)
        try:
            yield task_display
        finally:
            task_display.complete(result=None)


class ServerProgress(Progress):
    def __init__(self, display: ServerDisplay, total: int) -> None:
        self.display = display
        self.total = total
        self.current = 0
        self.done = False
        self._update_display()

    def update(self, n: int = 1) -> None:
        self.current += n
        self._update_display()

    def complete(self) -> None:
        self.done = True
        self.current = self.total
        self._update_display()

    def _update_display(self) -> None:
        progress_info = {
            "current": self.current,
            "total": self.total,
            "done": self.done,
        }
        with self.display.log_lock:
            self.display.current_progress = progress_info
            # self.display.logs.append(f"Progress update: {progress_info}")


class ServerTaskScreen(TaskScreen):
    def __init__(self, display: ServerDisplay) -> None:
        self.display = display
        # self.display.print(message="Entering server task screen.")

    def __exit__(self, *excinfo: Any) -> None:
        # self.display.print(message="Exiting server task screen.")
        pass

    @contextmanager
    def input_screen(self, header: str | None = None, transient: bool | None = None, width: int | None = None) -> Iterator[Any]:
        if header:
            # self.display.print(message=f"Input screen header: {header}")
            pass
        yield None  # No actual input collection in this minimal version.


class ServerTaskDisplay(TaskDisplay):
    def __init__(self, display: ServerDisplay, profile: TaskProfile) -> None:
        self.display = display
        self.profile = profile
        self.display.print(message=f"Starting task: {profile.name}")
        self.progress_value = 0

    @contextmanager
    def progress(self) -> Iterator[Progress]:
        total_steps = getattr(self.profile, "steps", 100)
        prog = ServerProgress(display=self.display, total=total_steps)
        try:
            yield prog
        finally:
            prog.complete()

    def sample_complete(self, complete: int, total: int) -> None:
        self.progress_value = complete
        # self.display.print(message=f"Task {self.profile.name} sample complete: {complete}/{total}")

    def update_metrics(self, scores: list[TaskDisplayMetric]) -> None:
        # self.display.print(message=f"Task {self.profile.name} metrics update: {scores}")
        with self.display.metrics_lock:
            # Store the latest scores for this task.
            self.display.metrics[self.profile.name] = [
                {
                    "scorer": metric.scorer,
                    "name": metric.name,
                    "value": metric.value,
                    "reducer": metric.reducer,
                }
                for metric in scores
            ]

    def complete(self, result: TaskResult | None) -> None:
        self.display.print(message=f"Task {self.profile.name} complete with result: {result}")