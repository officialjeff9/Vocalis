"""Vocalis — Your AI communication coach.

FastAPI server that serves the product UI and exposes a realtime WebSocket
(``/ws/stream``). The WebSocket receives raw audio chunks from the browser
mic and streams back live delivery metrics, live coach nudges, and a
before/after adaptive analysis per practice attempt.

The audio pipeline (librosa metrics -> Strands/Ollama coaching) is shared
with the CLI mic client and remains unchanged in behavior.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from agent import (
    SCENARIOS,
    derive_trigger,
    get_attempt_coaching,
    get_coaching_feedback,
    get_interview_question,
    warm_model,
)
from metrics import (
    PAUSE_QUALITY_LABELS,
    _pause_quality,
    compute_metrics,
    get_filler_detector,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

ANALYSIS_WINDOW_S = 3.0
DEFAULT_BASELINE_PACE_WPM = 150.0
COACHING_COOLDOWN_S = 20.0
MIN_ATTEMPT_S = 1.5


class SessionState:
    """Per-connection state for the scenario -> attempt -> retry flow."""

    def __init__(self) -> None:
        self.scenario: str | None = None
        self.question: str | None = None
        self.attempt: int = 0
        self.phase: str | None = None
        self.before_summary: dict | None = None
        self.analysis_in_flight: bool = False
        self.analysis_task: asyncio.Task | None = None
        self.question_task: asyncio.Task | None = None
        self.coach_tasks: set = set()
        self.background_tasks: set = set()
        self.asked_questions: list[str] = []
        self._acc: dict | None = None

    def track(self, task: asyncio.Task) -> None:
        """Register a socket-scoped background task so it is cancelled on disconnect."""
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)

    def start_attempt(self) -> str:
        self.phase = "before" if self.attempt == 0 else "after"
        self._acc = {
            "pace": [],
            "energy": [],
            "pause_ratio": [],
            "presence": [],
            "fillers_max": 0,
            "duration": 0.0,
            "windows": 0,
        }
        try:
            get_filler_detector().reset()
        except Exception:  # noqa: BLE001
            pass
        return self.phase

    def record(self, metrics: dict) -> None:
        if self._acc is None:
            return
        acc = self._acc
        acc["pace"].append(metrics.get("pace_wpm", 0.0))
        acc["energy"].append(metrics.get("energy", 0))
        acc["pause_ratio"].append(metrics.get("pause_ratio", 0.0))
        acc["presence"].append(metrics.get("presence", 0))
        acc["fillers_max"] = max(acc["fillers_max"], int(metrics.get("fillers", 0)))
        acc["duration"] += float(metrics.get("duration_sec", 0.0))
        acc["windows"] += 1

    def finalize(self, baseline: float) -> dict:
        acc = self._acc or {}
        self._acc = None
        duration = acc.get("duration", 0.0)
        windows = acc.get("windows", 0)

        def avg(key: str, digits: int = 1) -> float:
            values = acc.get(key, []) or []
            return round(sum(values) / len(values), digits) if values else 0.0

        avg_pace = avg("pace")
        pause_ratio = avg("pause_ratio", 4)
        quality = _pause_quality(pause_ratio)
        return {
            "attempt": self.attempt + 1,
            "avg_pace_wpm": avg_pace,
            "baseline_pace_wpm": round(baseline, 1),
            "pace_pct_of_baseline": round(avg_pace / max(baseline, 1.0) * 100, 0)
            if avg_pace
            else 0,
            "total_fillers": acc.get("fillers_max", 0),
            "fillers_per_minute": round(
                acc.get("fillers_max", 0) / max(duration, 0.001), 1
            ),
            "avg_energy": avg("energy", 0),
            "pause_ratio": pause_ratio,
            "pause_quality": quality,
            "pause_quality_label": PAUSE_QUALITY_LABELS[quality],
            "avg_presence": avg("presence", 0),
            "duration_s": round(duration, 1),
            "windows": windows,
        }


@asynccontextmanager
async def lifespan(_: FastAPI):
    asyncio.create_task(warm_model())
    yield


app = FastAPI(title="Vocalis — Your AI communication coach", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "vocalis"}


async def _streaming_analyzer(
    metrics_queue: asyncio.Queue,
    audio_queue: asyncio.Queue,
    baseline_holder: dict[str, float],
    state: SessionState,
    window_reset: dict[str, bool],
) -> None:
    """Consume audio chunks, analyze windows, and produce metric payloads."""
    window: list[bytes] = []
    window_start = time.monotonic()

    while True:
        chunk = await audio_queue.get()
        if chunk is None:
            metrics_queue.put_nowait(None)
            return

        if window_reset["reset"]:
            window = []
            window_start = time.monotonic()
            window_reset["reset"] = False

        window.append(chunk)
        elapsed = time.monotonic() - window_start
        if elapsed < ANALYSIS_WINDOW_S:
            continue

        audio_bytes = b"".join(window)
        window = []
        window_start = time.monotonic()

        try:
            metrics = compute_metrics(
                audio_bytes,
                baseline_pace_wpm=baseline_holder.get("baseline"),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Metrics analysis failed: %s", exc)
            metrics = {
                "pace_wpm": 0.0,
                "baseline_pace_wpm": baseline_holder.get(
                    "baseline", DEFAULT_BASELINE_PACE_WPM
                ),
                "pace_ratio": 1.0,
                "fillers": 0,
                "fillers_per_minute": 0.0,
                "pause_count": 0,
                "pause_ratio": 0.0,
                "silence_ratio": 0.0,
                "duration_sec": elapsed,
                "energy": 0,
                "pause_quality": "balanced",
                "pause_quality_label": PAUSE_QUALITY_LABELS["balanced"],
                "presence": 0,
            }
        state.record(metrics)
        metrics_queue.put_nowait(metrics)


async def _emit_coaching(websocket: WebSocket, metrics: dict, trigger: str) -> None:
    """Run one Strands coach turn and push the live nudge to the client."""
    try:
        coaching = await get_coaching_feedback(metrics)
        coaching["trigger"] = trigger
    except Exception as exc:  # noqa: BLE001
        logger.warning("Coaching generation failed: %s", exc)
        return
    try:
        await websocket.send_json(
            {"type": "coach_nudge", "coaching": coaching, "metrics": metrics}
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Coaching frame send failed: %s", exc)


async def _coach_loop(
    metrics_queue: asyncio.Queue, websocket: WebSocket, state: SessionState
) -> None:
    """Emit metric frames immediately; generate live nudges in background.

    At most one Strands invocation is in flight at a time and a cooldown
    bounds how often the (slow) local LLM is queried. Live nudges yield
    while a full attempt analysis is running.
    """
    last_coaching_at = 0.0
    pending: set[asyncio.Task] = set()

    while True:
        metrics = await metrics_queue.get()
        if metrics is None:
            for task in pending:
                task.cancel()
            return

        now = time.monotonic()
        trigger = derive_trigger(metrics)
        status = {"trigger": trigger, "feedback": None, "pending": False}

        if (
            trigger != "none"
            and not pending
            and not state.analysis_in_flight
            and now - last_coaching_at >= COACHING_COOLDOWN_S
        ):
            last_coaching_at = now
            status["pending"] = True
            task = asyncio.create_task(_emit_coaching(websocket, metrics, trigger))
            pending.add(task)
            state.coach_tasks.add(task)
            task.add_done_callback(pending.discard)
            task.add_done_callback(state.coach_tasks.discard)

        try:
            await websocket.send_json(
                {"type": "update", "metrics": metrics, "coaching": status}
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Update frame send failed: %s", exc)
            return


async def _emit_question(
    websocket: WebSocket, state: SessionState, scenario: str
) -> None:
    question = await get_interview_question(
        scenario, avoid=list(state.asked_questions)
    )
    state.question = question
    if question not in state.asked_questions:
        state.asked_questions.append(question)
    try:
        await websocket.send_json(
            {
                "type": "question",
                "scenario": scenario,
                "title": SCENARIOS.get(scenario, {}).get("title", scenario),
                "question": question,
            }
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Question frame send failed: %s", exc)


async def _emit_attempt_result(
    websocket: WebSocket,
    state: SessionState,
    summary: dict,
    phase: str,
    after_phase: bool,
) -> None:
    """Run the adaptive analysis agent and push the attempt result.

    The in-flight flag is owned by ``_attempt_result_task`` so the busy
    state is set synchronously (before the task can start) and only the
    task that still owns the slot clears it.
    """
    try:
        if after_phase and state.before_summary is not None:
            coaching = await get_attempt_coaching(
                before=state.before_summary,
                after=summary,
                question=state.question,
                scenario=state.scenario,
            )
        else:
            coaching = await get_attempt_coaching(
                before=summary,
                after=None,
                question=state.question,
                scenario=state.scenario,
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Attempt analysis failed: %s", exc)
        coaching = {}

    try:
        await websocket.send_json(
            {
                "type": "attempt_result",
                "attempt": state.attempt,
                "phase": phase,
                "after_phase": after_phase,
                "before_summary": state.before_summary,
                "after_summary": summary,
                "coaching": coaching,
                "scenario": state.scenario,
                "question": state.question,
            }
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Attempt result send failed: %s", exc)


async def _emit_question_task(
    websocket: WebSocket, state: SessionState, scenario: str
) -> None:
    """Draft and send one question; clear the slot when finished."""
    try:
        await _emit_question(websocket, state, scenario)
    finally:
        if state.question_task is asyncio.current_task():
            state.question_task = None


def _spawn_question(websocket: WebSocket, state: SessionState, scenario: str) -> None:
    """Replace any in-flight question draft so only ONE question is ever sent."""
    prev = state.question_task
    if prev is not None and not prev.done():
        prev.cancel()
    task = asyncio.create_task(_emit_question_task(websocket, state, scenario))
    state.question_task = task
    state.track(task)


async def _attempt_result_task(
    websocket: WebSocket,
    state: SessionState,
    summary: dict,
    phase: str,
    after_phase: bool,
) -> None:
    """Run the adaptive analysis; clear the busy flag when it finishes."""
    try:
        await _emit_attempt_result(websocket, state, summary, phase, after_phase)
    finally:
        if state.analysis_task is asyncio.current_task():
            state.analysis_in_flight = False
            state.analysis_task = None


@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    logger.info("WebSocket connected: %s", websocket.client.host)

    audio_queue: asyncio.Queue = asyncio.Queue()
    metrics_queue: asyncio.Queue = asyncio.Queue()
    baseline_holder: dict[str, float] = {}
    state = SessionState()
    window_reset: dict[str, bool] = {"reset": False}

    producer = asyncio.create_task(
        _streaming_analyzer(
            metrics_queue, audio_queue, baseline_holder, state, window_reset
        )
    )
    consumer = asyncio.create_task(_coach_loop(metrics_queue, websocket, state))

    def drain_audio() -> None:
        window_reset["reset"] = True
        while not audio_queue.empty():
            audio_queue.get_nowait()

    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                break

            if "bytes" in message and message["bytes"]:
                audio_queue.put_nowait(message["bytes"])
                continue

            if "text" in message and message["text"]:
                try:
                    control = json.loads(message["text"])
                except json.JSONDecodeError:
                    await websocket.send_json(
                        {"type": "error", "message": "control messages must be JSON"}
                    )
                    continue

                control_type = control.get("type")

                if control_type == "set_baseline":
                    pace = control.get("pace_wpm")
                    if isinstance(pace, (int, float)) and pace > 0:
                        baseline_holder["baseline"] = float(pace)
                        await websocket.send_json(
                            {"type": "baseline", "pace_wpm": float(pace)}
                        )

                elif control_type == "start_session":
                    scenario = control.get("scenario", "job_interview")
                    if scenario not in SCENARIOS:
                        await websocket.send_json(
                            {"type": "error", "message": f"unknown scenario: {scenario}"}
                        )
                        continue
                    state.scenario = scenario
                    state.question = None
                    state.attempt = 0
                    state.phase = None
                    state.before_summary = None
                    state.asked_questions = []
                    drain_audio()
                    await websocket.send_json(
                        {
                            "type": "session_started",
                            "scenario": scenario,
                            "title": SCENARIOS[scenario]["title"],
                            "blurb": SCENARIOS[scenario]["blurb"],
                        }
                    )
                    _spawn_question(websocket, state, scenario)

                elif control_type == "next_question":
                    if state.scenario is None:
                        await websocket.send_json(
                            {"type": "error", "message": "no active session"}
                        )
                        continue
                    state.question = None
                    state.attempt = 0
                    state.phase = None
                    state.before_summary = None
                    drain_audio()
                    _spawn_question(websocket, state, state.scenario)

                elif control_type == "start_attempt":
                    if state.analysis_in_flight:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "message": "Your previous attempt is still being analyzed — please wait a moment.",
                            }
                        )
                        continue
                    if state.question is None:
                        await websocket.send_json(
                            {"type": "error", "message": "waiting for a question"}
                        )
                        continue
                    phase = state.start_attempt()
                    await websocket.send_json(
                        {
                            "type": "attempt_started",
                            "attempt": state.attempt + 1,
                            "phase": phase,
                        }
                    )

                elif control_type == "end_attempt":
                    if state.analysis_in_flight:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "message": "Your previous attempt is still being analyzed — please wait a moment.",
                            }
                        )
                        continue
                    if state.coach_tasks:
                        for task in list(state.coach_tasks):
                            task.cancel()
                        try:
                            await asyncio.gather(
                                *list(state.coach_tasks), return_exceptions=True
                            )
                        except Exception:  # noqa: BLE001
                            pass
                        state.coach_tasks.clear()
                    baseline = baseline_holder.get("baseline")
                    summary = state.finalize(
                        baseline if baseline else DEFAULT_BASELINE_PACE_WPM
                    )
                    phase = state.phase or "before"
                    if (
                        summary["windows"] == 0
                        or summary["duration_s"] < MIN_ATTEMPT_S
                    ):
                        await websocket.send_json(
                            {
                                "type": "attempt_result",
                                "short": True,
                                "message": "We didn't catch enough audio. Press Start and speak for a few seconds.",
                            }
                        )
                        continue
                    after_phase = phase == "after"
                    if not after_phase:
                        state.before_summary = summary
                    state.attempt += 1
                    state.phase = None
                    drain_audio()
                    state.analysis_in_flight = True
                    state.analysis_task = asyncio.create_task(
                        _attempt_result_task(
                            websocket, state, summary, phase, after_phase
                        )
                    )
                    state.track(state.analysis_task)

                else:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": f"unknown control type: {control_type}",
                        }
                    )
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: %s", websocket.client.host)
    finally:
        audio_queue.put_nowait(None)
        for task in list(state.coach_tasks):
            task.cancel()
        for task in list(state.background_tasks):
            task.cancel()
        try:
            await asyncio.wait_for(asyncio.gather(producer, consumer), timeout=3.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            producer.cancel()
            consumer.cancel()


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        loop="asyncio",
    )