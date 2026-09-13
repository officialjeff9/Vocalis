"""Strands agents wrapper: live-metrics to coaching-intervention agent.

Uses the Strands Agents SDK with an ``OllamaModel`` pointed at a local
Ollama server running the Qwen3 model family. Each invocation feeds the
latest live metrics and returns one short, actionable intervention.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re

from strands import Agent
from strands.models.ollama import OllamaModel

logger = logging.getLogger(__name__)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.6")
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.4"))

REFERENCE_CUES = {
    "pace": "You're rushing. Slow down and pause before your main point.",
    "fillers": "Replace 'um' with a deliberate pause before your next word.",
    "pauses": "Tighten your pauses — connect your points with a transition.",
    "energy": "Raise your energy: open your voice and commit to every word.",
    "strength": "Strong delivery. Hold that presence and add one confident pause.",
}

FOCUS_LABELS = {
    "pace": "PACE",
    "fillers": "FILLERS",
    "pauses": "PAUSES",
    "energy": "ENERGY",
    "strength": "DELIVERY STRENGTH",
    "none": "ON TRACK",
}

SYSTEM_PROMPT = """You are an elite high-stakes communication coach embedded in a real-time speaking app, coaching people through investor pitches, negotiations, and presentations.

You receive a JSON blob of live, machine-generated delivery metrics. Produce exactly ONE short, actionable intervention.

Rules:
- Output a single imperative sentence, 8-16 words, no preamble or markdown.
- Address only the single most urgent issue, prioritized: dangerous pacing speed -> excessive fillers -> disruptive pauses -> everything is fine.
- Be specific and prescriptive, e.g. "Slow down before your next point" or "Pause two beats after the word 'Q3' instead of saying 'um'."
- Never give generic praise or multiple suggestions at once.
- If delivery is on track, give a brief reinforcing nudge.

Prescribed go-to cues (paraphrase them, don't repeat verbatim):
- Pace too fast: "Slow down before your next point."
- Fillers high: "Replace 'um' with a deliberate pause."
- Pauses too long: "Restart with a stronger connecting word."
"""

QUESTION_SYSTEM = """You are a senior hiring manager designing tough, realistic practice questions for "Vocalis", an AI communication coach.

Given a scenario and brief, write ONE question that puts the speaker under real workplace pressure: a direct conflict, a skeptical stakeholder, a missed deadline, or an exchange where they must respond on the spot.

Pressure is the point — hard, fast questions make speakers stumble naturally into filler words, rushed pacing, and pauses, which is exactly what Vocalis needs to analyze.

Rules:
- A single question only, up to 24 words.
- Use direct, confrontational phrasing that names the accusation or the stake, then pin the speaker with a follow-up: e.g. "Your manager says your project isn't moving fast enough. How would you respond?"
- Keep it second-person ("you") so the pressure lands on the speaker right now.
- No markdown, quotes, numbering, or labels.

Output ONLY the question text.
"""

COACH_ANALYSIS_SYSTEM = """You are the adaptive communication coach behind "Vocalis". Users practice a live scenario and you analyze measured delivery.

You are given JSON summaries of real speech metrics for one speech attempt, or two attempts ("before" then "after").

Decide the SINGLE highest-impact problem on the most recent attempt and give ONE concise, immediate coaching cue.

Rules:
- Prioritize: rushing pace -> excessive fillers -> poor pauses -> low energy -> already solid.
- Focus on exactly one thing. Never list multiple.
- cue: one imperative sentence, 7-15 words, user-facing, e.g. "You're rushing. Slow down and pause before your key point."
- explanation: ONE short sentence naming the measured number and the fix. Never include reasoning steps or chain-of-thought.
- When a "before" and "after" attempt were both given, first recognize whether the after improved using the real numbers, then pick the NEXT focus for the after attempt.

Return STRICT JSON only (no markdown) with keys:
{"focus": "pace"|"fillers"|"pauses"|"energy"|"strength",
 "cue": "<string>",
 "explanation": "<string>",
 "improvement": {"recognized": <bool>, "note": "<string>", "deltas": {<metric>: <pct change>}}}
"""

SCENARIOS = {
    "job_interview": {
        "title": "Job Interview",
        "blurb": "Behavioral questions from real tech hiring managers.",
        "hint": (
            "a high-pressure moment a senior candidate faces: a skeptical manager "
            "pushing back, a missed deadline, or an angry stakeholder"
        ),
        "fallback_questions": [
            "Your manager says your project isn't moving fast enough. How would you respond?",
            "A stakeholder says your work is behind the plan. What do you tell them?",
            "Your manager says they're losing confidence in your delivery. How do you handle it?",
            "A teammate calls your decision a costly mistake. How would you respond?",
        ],
    },
    "startup_pitch": {
        "title": "Startup Pitch",
        "blurb": "Convince an investor your product is the future.",
        "hint": (
            "a hostile seed-stage investor pushing back on valuation, burn, or "
            "why customers will leave"
        ),
        "fallback_questions": [
            "Your valuation is ridiculous and your burn is worse. Convince me otherwise.",
            "Why will customers churn off your product in six months? Be honest.",
            "If I pass on this round, what will you do with no money left?",
            "Your own data shows users are leaving. Why should I still invest?",
        ],
    },
    "town_hall": {
        "title": "Leadership Q&A",
        "blurb": "Handle a skeptical employee at an all-hands.",
        "hint": (
            "a skeptical employee challenging leadership on slipped promises, "
            "layoffs, or broken trust at an all-hands"
        ),
        "fallback_questions": [
            "Our pay hasn't kept up and you've said nothing. What do you tell us?",
            "You promised no layoffs two years ago, yet cutbacks are here. What changed?",
            "The roadmap you sold us has slipped twice already. Why trust the next one?",
            "You announced a bonus while this team is being cut. Defend that.",
        ],
    },
}

_TIMEOUT_S = 300
_QUESTION_TIMEOUT_S = 120
_COACHING_TIMEOUT_S = 120

# Strands/Ollama serves one generation at a time; overlapping invokes
# overrun the local model and stall the pipeline. Every agent turn is
# serialized through this process-wide lock so incoming requests wait
# cleanly instead of ever running the model concurrently.
_AGENT_LOCK = asyncio.Lock()

_TRIGGER_ORDER = ("pace", "fillers", "pauses")

_model: OllamaModel | None = None
_agent: Agent | None = None
_question_agent: Agent | None = None
_analysis_agent: Agent | None = None


def _get_model() -> OllamaModel:
    """Return the process-wide Ollama model instance (kept warm)."""
    global _model
    if _model is None:
        _model = OllamaModel(
            host=OLLAMA_HOST,
            model_id=OLLAMA_MODEL,
            temperature=OLLAMA_TEMPERATURE,
            max_tokens=512,
            keep_alive="10m",
            additional_args={"think": False},
        )
    return _model


def build_agent() -> Agent:
    """Construct the Strands live-nudge coach agent backed by local Ollama."""
    return Agent(
        model=_get_model(), system_prompt=SYSTEM_PROMPT, callback_handler=None
    )


def get_agent() -> Agent:
    """Return the lazily-created singleton live coach agent."""
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent


def get_question_agent() -> Agent:
    """Return the lazily-created interviewer agent."""
    global _question_agent
    if _question_agent is None:
        _question_agent = Agent(
            model=_get_model(),
            system_prompt=QUESTION_SYSTEM,
            callback_handler=None,
        )
    return _question_agent


def get_analysis_agent() -> Agent:
    """Return the lazily-created adaptive analysis agent."""
    global _analysis_agent
    if _analysis_agent is None:
        _analysis_agent = Agent(
            model=_get_model(),
            system_prompt=COACH_ANALYSIS_SYSTEM,
            callback_handler=None,
        )
    return _analysis_agent


async def _run_agent_locked(agent: Agent, prompt: str, timeout: float) -> str:
    """Serialize one Strands turn behind the process-wide lock.

    Queueing every invoke on ``_AGENT_LOCK`` guarantees no two agent runs
    ever overlap (whether within a session or across sockets). The timeout
    applies to lock acquisition + generation so one hung call cannot pin
    the lock (and therefore the whole pipeline) forever.
    """
    async with _AGENT_LOCK:
        result = await asyncio.wait_for(agent.invoke_async(prompt), timeout=timeout)
    return _extract_text(result)


def derive_trigger(metrics: dict) -> str:
    """Pick the single most urgent coaching trigger from live metrics."""
    pace_ratio = metrics.get("pace_ratio", 1.0)
    pace_wpm = metrics.get("pace_wpm", 0.0)
    if pace_wpm > 0 and (pace_ratio >= 1.15 or pace_wpm >= 190):
        return "pace"
    fillers_per_min = metrics.get("fillers_per_minute", 0.0)
    if fillers_per_min >= 1.0:
        return "fillers"
    if metrics.get("pause_ratio", 0.0) >= 0.30:
        return "pauses"
    return "none"


def _extract_text(result) -> str:
    """Best-effort extraction of the final text from an AgentResult."""
    message = getattr(result, "message", None)
    if message is None:
        return str(result)
    if isinstance(message, dict):
        content = message.get("content")
    else:
        content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text") or block.get("content") or "")
            else:
                parts.append(getattr(block, "text", "") or "")
        return "".join(parts)
    if isinstance(message, dict):
        return str(message.get("text") or message)
    text = getattr(message, "text", None)
    return str(text or message)


def format_metrics_prompt(metrics: dict, trigger: str) -> str:
    """Render live metrics + derived trigger into an agent prompt."""
    payload = json.dumps(metrics, indent=2)
    return (
        f"Latest speaker metrics:\n{payload}\n\n"
        f"Flags:\n"
        f"- Urgent trigger: {trigger}\n"
        f"- Baseline pace: {metrics.get('baseline_pace_wpm')} wpm\n"
        f"- Current pace: {metrics.get('pace_wpm')} wpm\n"
        f"- Fillers observed: {metrics.get('fillers')}\n"
        f"- Pause count: {metrics.get('pause_count')}\n\n"
        "Reply with one actionable intervention following the coach rules."
    )


async def get_coaching_feedback(metrics: dict) -> dict:
    """Run one agentic loop turn against Ollama and return the intervention."""
    trigger = derive_trigger(metrics)
    if trigger == "none":
        return {"trigger": trigger, "feedback": "Keep it up — delivery is on track.", "model": OLLAMA_MODEL}

    agent = get_agent()
    try:
        feedback = (
            await _run_agent_locked(
                agent, format_metrics_prompt(metrics, trigger), _COACHING_TIMEOUT_S
            )
        ).strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Strands agent call failed: %s", exc)
        feedback = ""
    if not feedback:
        feedback = {
            "pace": "Slow down before your next point.",
            "fillers": "Replace the filler with a deliberate pause.",
            "pauses": "Restart with a stronger connecting word.",
        }[trigger]
    return {"trigger": trigger, "feedback": feedback, "model": OLLAMA_MODEL}


def _safe_json(text: str):
    """Best-effort JSON.parse with brace-slice recovery."""
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def _norm_coaching(data: dict, model_key: str = "ollama") -> dict | None:
    focus = str(data.get("focus", "")).strip().lower()
    if focus not in FOCUS_LABELS:
        return None
    cue = str(data.get("cue", "")).strip().strip('"')
    explanation = str(data.get("explanation", "")).strip().strip('"')
    if not cue or len(cue) > 160:
        return None
    improvement = data.get("improvement")
    if isinstance(improvement, dict):
        recognized = bool(improvement.get("recognized"))
        note = str(improvement.get("note", "")).strip()
        deltas = improvement.get("deltas")
        if not isinstance(deltas, dict):
            deltas = {}
        improvement = {"recognized": recognized, "note": note, "deltas": deltas}
    else:
        improvement = None
    return {
        "focus": focus,
        "focus_label": FOCUS_LABELS[focus],
        "cue": cue,
        "explanation": explanation,
        "improvement": improvement,
        "source": model_key,
        "model": OLLAMA_MODEL,
    }


def _derive_focus(summary: dict) -> str:
    pace = summary.get("avg_pace_wpm", 0.0)
    baseline = summary.get("baseline_pace_wpm", 150.0) or 150.0
    ratio = pace / max(baseline, 1.0) if pace else 1.0
    if pace > 0 and (ratio >= 1.15 or pace >= 190):
        return "pace"
    if summary.get("fillers_per_minute", 0.0) >= 1.0:
        return "fillers"
    if summary.get("pause_ratio", 0.0) >= 0.30:
        return "pauses"
    if summary.get("avg_energy", 60) < 45:
        return "energy"
    return "strength"


def _delta_pct(before: dict, after: dict, key: str) -> float | None:
    b, a = before.get(key), after.get(key)
    if isinstance(b, (int, float)) and isinstance(a, (int, float)) and b and abs(b) > 1e-6:
        return round((a - b) / abs(b) * 100.0, 1)
    return None


def _rule_based_coaching(
    before: dict, after: dict | None, question: str | None, scenario: str | None
) -> dict:
    """Deterministic fallback derived entirely from the real measured numbers."""
    target = after if after is not None else before
    focus = _derive_focus(target)
    improvement = None

    if after is not None:
        deltas = {k: _delta_pct(before, after, k) for k in ("avg_pace_wpm", "avg_energy", "avg_presence", "pause_ratio", "fillers_per_minute")}
        deltas = {k: v for k, v in deltas.items() if v is not None}
        pace_a = after.get("avg_pace_wpm", 0)
        desired = "down" if pace_a > before.get("baseline_pace_wpm", 150) else "up"
        recognized = (
            desired == "down" and _delta_pct(before, after, "avg_pace_wpm") is not None and _delta_pct(before, after, "avg_pace_wpm") < 0
        ) or (
            desired == "up" and _delta_pct(before, after, "avg_pace_wpm") is not None and _delta_pct(before, after, "avg_pace_wpm") > 0
        ) or (
            _delta_pct(before, after, "avg_energy") or 0
        ) > 0 or (
            _delta_pct(before, after, "fillers_per_minute") or 99
        ) < 0
        note = _build_improvement_note(before, after)
        improvement = {"recognized": bool(recognized), "note": note, "deltas": deltas}

    explanation = _build_explanation(focus, target)
    return {
        "focus": focus,
        "focus_label": FOCUS_LABELS[focus],
        "cue": REFERENCE_CUES[focus],
        "explanation": explanation,
        "improvement": improvement,
        "source": "rules",
        "model": OLLAMA_MODEL,
    }


def _build_improvement_note(before: dict, after: dict) -> str:
    parts = []
    pace_delta = _delta_pct(before, after, "avg_pace_wpm")
    if pace_delta is not None and abs(pace_delta) >= 3:
        parts.append(f"pace {pace_delta:+.0f}%")
    energy_delta = _delta_pct(before, after, "avg_energy")
    if energy_delta is not None and abs(energy_delta) >= 3:
        parts.append(f"energy {energy_delta:+.0f}%")
    fillers_delta = _delta_pct(before, after, "fillers_per_minute")
    if fillers_delta is not None and abs(fillers_delta) >= 10:
        parts.append(f"fillers {fillers_delta:+.0f}%")
    if parts:
        return "Your second attempt shows real progress: " + ", ".join(parts) + "."
    return "Noticeable change between the two attempts — your focus is working."


def _build_explanation(focus: str, summary: dict) -> str:
    pacing = (
        summary.get("avg_pace_wpm", 0)
        and f"Pacing ran at {summary.get('avg_pace_wpm')} WPM versus {summary.get('baseline_pace_wpm')} target"
    ) or "Pacing drifted from your target"
    return {
        "pace": f"{pacing} — rushing costs clarity and trust.",
        "fillers": f"Fillers hit {summary.get('fillers_per_minute')} per minute — they chip away at authority.",
        "pauses": f"Gaps made up {round(summary.get('pause_ratio', 0) * 100)}% of the attempt — breaks the flow.",
        "energy": f"Energy sat at {summary.get('avg_energy')}/100 — flat delivery weakens the message.",
        "strength": f"Presence averaged {summary.get('avg_presence')}/100 — hold this and refine one detail.",
    }[focus]


def _summary_block(summary: dict, tag: str) -> str:
    return f"{tag} attempt metrics (measured live):\n{json.dumps(summary, indent=2)}"


def build_attempt_prompt(
    before: dict, after: dict | None, question: str | None, scenario: str | None
) -> str:
    scenario_title = SCENARIOS.get(scenario, {}).get("title", str(scenario or "practice"))
    lines = [f"Scenario: {scenario_title}"]
    if question:
        lines.append(f"Question: {question}")
    if after is None:
        lines.append(_summary_block(before, "Before"))
        lines.append(
            "This is the first attempt. Set improvement.recognized=false and deltas to {}."
        )
    else:
        lines.append(_summary_block(before, "Before"))
        lines.append(_summary_block(after, "After"))
        lines.append(
            "Compare the two attempts. If the After attempt improved on any measured "
            "metric, set improvement.recognized=true, write a factual note about the real "
            "delta, and list measured deltas. Then pick the NEXT most important focus for "
            "the After attempt."
        )
    lines.append("Return strict JSON only.")
    return "\n".join(lines)


async def get_interview_question(
    scenario: str, avoid: list[str] | None = None
) -> str:
    """Ask the interviewer agent for a question; fall back to a curated one.

    ``avoid`` lists previously asked questions so the agent produces variety.
    """
    scenario_meta = SCENARIOS.get(scenario, {})
    avoid = avoid or []

    def _prompt() -> str:
        base = (
            f"Scenario: {scenario_meta.get('title', scenario)}.\n"
            f"Brief: {scenario_meta.get('hint', 'a realistic practice question')}.\n"
            "Ask one question."
        )
        if avoid:
            base += (
                "\nDo NOT repeat any of these previously asked questions:\n- "
                + "\n- ".join(avoid[:5])
            )
        return base

    for _ in range(2):
        try:
            question = (
                await _run_agent_locked(
                    get_question_agent(), _prompt(), _QUESTION_TIMEOUT_S
                )
            ).strip().strip('"').strip()
            if (
                8 <= len(question) <= 260
                and any(ch in question for ch in "?")
                and question not in avoid
            ):
                return question
        except Exception as exc:  # noqa: BLE001
            logger.warning("Question generation failed: %s", exc)

    fallbacks = scenario_meta.get("fallback_questions") or [
        "Tell me about a time you convinced someone to change their mind."
    ]
    pool = [q for q in fallbacks if q not in avoid] or fallbacks
    return random.choice(pool)


async def get_attempt_coaching(
    before: dict,
    after: dict | None = None,
    question: str | None = None,
    scenario: str | None = None,
) -> dict:
    """Run the adaptive analysis agent over one or two attempts."""
    prompt = build_attempt_prompt(before, after, question, scenario)
    try:
        data = _safe_json(
            await _run_agent_locked(get_analysis_agent(), prompt, _TIMEOUT_S)
        )
        if data is None:
            raise ValueError("unparsable agent response")
        coaching = _norm_coaching(data)
        if coaching is None:
            raise ValueError("invalid coaching payload")
        logger.info("Analysis agent produced focus=%s", coaching["focus"])
        return coaching
    except Exception as exc:  # noqa: BLE001
        logger.warning("Analysis agent failed (%s); using rule fallback", exc)
        return _rule_based_coaching(before, after, question, scenario)


async def warm_model() -> None:
    """Issue a lightweight generation so the shared model stays hot."""
    try:
        await _run_agent_locked(get_agent(), "Say 'ready'.", timeout=100)
        logger.info("Strands agent warmed")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Warmup failed: %s", exc)