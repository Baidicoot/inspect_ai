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


class WebDisplay(Display):
    def __init__(self, host: str = "localhost", port: int = 8000) -> None:
        self.host: str = host
        self.port: int = port
        self.logs: list[str] = []
        self.log_lock = threading.Lock()
        # We simply keep one progress state; you could extend this to support many.
        self.current_progress: dict[str, Any] = {}

        # Create FastAPI app and attach this instance to its state.
        self.app = FastAPI()
        self.app.state.web_display = self

        # Define endpoints.
        self.app.add_api_route(path="/", endpoint=self.index, methods=["GET"])
        self.app.add_api_route(path="/logs", endpoint=self.get_logs, methods=["GET"])
        self.app.websocket("/ws")(self.websocket_endpoint)

        # Start the server in a background thread.
        thread = threading.Thread(target=self._run_uvicorn, daemon=True)
        thread.start()

        # Log the URL.
        self.print(message=f"WebDisplay server running at http://{self.host}:{self.port}")

    def _run_uvicorn(self) -> None:
        uvicorn.run(self.app, host=self.host, port=self.port, log_level="debug")

    async def index(self) -> HTMLResponse:
        # A minimal HTML page with a bona fide progress bar above the logs.
        html_content = textwrap.dedent("""\
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>Inspect Web Display</title>
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        padding: 20px;
                    }
                    #progressContainer {
                        margin-bottom: 20px;
                        padding: 10px;
                        border: 1px solid #ccc;
                        background: #e9e9e9;
                    }
                    #progressBar {
                        width: 100%;
                        height: 20px;
                    }
                    #logsContainer {
                        background: #f9f9f9;
                        border: 1px solid #ccc;
                        padding: 10px;
                    }
                </style>
            </head>
            <body>
                <h1>Inspect Web Display</h1>
                <div id="progressContainer">
                    <h2>Progress</h2>
                    <progress id="progressBar" value="0" max="100"></progress>
                    <span id="progressText">0%</span>
                </div>
                <div id="logsContainer">
                    <h2>Logs</h2>
                    <pre id="logs"></pre>
                </div>
                <script>
                    const ws = new WebSocket("ws://" + location.host + "/ws");
                    ws.onmessage = function(event) {
                        const data = JSON.parse(event.data);
                        // Update the progress bar if progress data is available.
                        if (data.progress) {
                            const progressBar = document.getElementById("progressBar");
                            const progressText = document.getElementById("progressText");
                            const current = data.progress.current;
                            const total = data.progress.total;
                            progressBar.max = total;
                            progressBar.value = current;
                            const percent = Math.floor((current / total) * 100);
                            progressText.textContent = percent + "%";
                        }
                        // Update the logs.
                        if (data.logs) {
                            const logsElem = document.getElementById("logs");
                            logsElem.textContent = data.logs.join("\\n");
                        }
                    };
                </script>
            </body>
            </html>
        """)
        return HTMLResponse(content=html_content)

    async def get_logs(self) -> Any:
        with self.log_lock:
            return {"logs": self.logs, "progress": self.current_progress}

    async def websocket_endpoint(self, websocket: WebSocket) -> None:
        self.print(message="Incoming WebSocket connection.")
        await websocket.accept()
        self.print(message="WebSocket connection accepted.")
        try:
            while True:
                await asyncio.sleep(1)
                with self.log_lock:
                    data = {"logs": self.logs, "progress": self.current_progress}
                await websocket.send_text(json.dumps(data))
        except Exception as e:
            self.print(message=f"Exception in WebSocket endpoint: {e}")
        finally:
            self.print(message="Closing WebSocket connection.")

    def print(self, message: str) -> None:
        with self.log_lock:
            self.logs.append(message)

    @contextmanager
    def progress(self, total: int) -> Iterator[Progress]:
        prog = WebProgress(display=self, total=total)
        try:
            yield prog
        finally:
            prog.complete()

    def run_task_app(self, main: Coroutine[Any, Any, TR]) -> TR:
        return asyncio.run(main)

    @contextmanager
    def suspend_task_app(self) -> Iterator[None]:
        # For simplicity, nothing is suspended in this web display.
        yield

    @asynccontextmanager
    async def task_screen(self, tasks: list[TaskSpec], parallel: bool) -> AsyncIterator[TaskScreen]:
        screen = WebTaskScreen(display=self)
        try:
            yield screen
        finally:
            # Any cleanup for the task screen if necessary.
            pass

    @contextmanager
    def task(self, profile: TaskProfile) -> Iterator[TaskDisplay]:
        task_display = WebTaskDisplay(display=self, profile=profile)
        try:
            yield task_display
        finally:
            task_display.complete(result=None)


class WebProgress(Progress):
    def __init__(self, display: WebDisplay, total: int) -> None:
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
            self.display.logs.append(f"Progress update: {progress_info}")


class WebTaskScreen(TaskScreen):
    def __init__(self, display: WebDisplay) -> None:
        self.display = display
        self.display.print(message="Entering task screen.")

    def __exit__(self, *excinfo: Any) -> None:
        self.display.print(message="Exiting task screen.")

    @contextmanager
    def input_screen(self, header: str | None = None, transient: bool | None = None, width: int | None = None) -> Iterator[Any]:
        if header:
            self.display.print(message=f"Input screen header: {header}")
        yield None  # No actual input collection in this minimal version.


class WebTaskDisplay(TaskDisplay):
    def __init__(self, display: WebDisplay, profile: TaskProfile) -> None:
        self.display = display
        self.profile = profile
        self.display.print(message=f"Starting task: {profile.name}")
        self.progress_value = 0

    @contextmanager
    def progress(self) -> Iterator[Progress]:
        total_steps = getattr(self.profile, "steps", 100)
        prog = WebProgress(display=self.display, total=total_steps)
        try:
            yield prog
        finally:
            prog.complete()

    def sample_complete(self, complete: int, total: int) -> None:
        self.progress_value = complete
        self.display.print(message=f"Task {self.profile.name} sample complete: {complete}/{total}")

    def update_metrics(self, scores: list[TaskDisplayMetric]) -> None:
        self.display.print(message=f"Task {self.profile.name} metrics update: {scores}")

    def complete(self, result: TaskResult | None) -> None:
        self.display.print(message=f"Task {self.profile.name} complete with result: {result}")