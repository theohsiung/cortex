"""Cortex FastAPI microservice."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
import warnings

# Filter Pydantic warnings
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from app.config import CortexConfig
from cortex import Cortex

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Cortex Microservice")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store task events for SSE (history + real-time)
# task_id -> List[dict]
event_store: dict[str, list] = {}


class TaskRequest(BaseModel):
    """Request body for creating a new task."""

    query: str


class TaskResponse(BaseModel):
    """Response body returned when a task is created."""

    task_id: str
    status: str


@app.post("/api/tasks", response_model=TaskResponse)
async def create_task(request: TaskRequest, background_tasks: BackgroundTasks):
    """Create a new Cortex task and start it in the background."""
    task_id = str(uuid.uuid4())
    event_store[task_id] = []

    # Start execution in background
    background_tasks.add_task(run_cortex_task, task_id, request.query)

    return TaskResponse(task_id=task_id, status="accepted")


@app.get("/api/tasks/{task_id}/events")
async def stream_events(task_id: str, request: Request):
    """Stream events for a specific task via SSE with history support."""
    if task_id not in event_store:
        return JSONResponse(status_code=404, content={"error": "Task not found"})

    async def event_generator():
        # Wrap in "data" so it sends as a default "message" event
        # that evtSource.onmessage can catch.
        # MUST json.dumps the data dict, otherwise it sends python string (single quotes)
        yield {"data": json.dumps({"event": "connected", "data": "Connected to event stream"})}

        current_idx = 0
        events = event_store[task_id]

        try:
            while True:
                if await request.is_disconnected():
                    break

                # Check for new events
                if current_idx < len(events):
                    event = events[current_idx]
                    current_idx += 1
                    # Yield as data payload
                    yield {"data": json.dumps(event)}

                    if event["event"] in ["execution_complete", "error"]:
                        # Close stream after final event
                        await asyncio.sleep(0.5)
                        break
                else:
                    # Wait for new events
                    await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            logger.info("Client disconnected from task %s", task_id)

    return EventSourceResponse(event_generator())


async def run_cortex_task(task_id: str, query: str) -> None:
    """Run a Cortex task and store events for SSE streaming."""
    logger.info("Starting task %s: %s", task_id, query)

    if task_id not in event_store:
        logger.error("No event store found for task %s", task_id)
        return

    try:
        # Initialize Cortex
        config = CortexConfig()  # type: ignore[call-arg]
        cortex = Cortex(config)

        # Define event handler
        async def on_event(event_type: str, data: dict):
            if task_id in event_store:
                event_store[task_id].append({"event": event_type, "data": data})

        # Execute
        await cortex.execute(query, on_event=on_event)

    except Exception as e:
        logger.error("Task %s failed: %s", task_id, e)
        if task_id in event_store:
            event_store[task_id].append({"event": "error", "data": {"message": str(e)}})


# Mount static files for frontend (if exists)
if os.path.exists("frontend/dist"):
    app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
