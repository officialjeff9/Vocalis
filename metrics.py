"""Real-time speech metrics computed from raw audio chunks.

Pace and pause detection use librosa on raw PCM/WAV bytes.
Filler-word detection is optional and, when an AssemblyAI API key is
present in ``ASSEMBLYAI_API_KEY``, streams the same audio through
AssemblyAI's realtime transcription and scans it for filler words.
"""

from __future__ import annotations

import io
import logging
import os
from collections import deque

import librosa
import numpy as np
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)

TARGET_SR = 16_000
AVG_SYLLABLES_PER_WORD = 1.5
MIN_PEAK_GAP_S = 0.12
MIN_PAUSE_S = 0.25

FILLER_WORDS = frozenset(
    {
        "um",
        "uh",
        "uhm",
        "umm",
        "er",
        "erm",
        "hmm",
        "like",
        "you know",
        "kind of",
        "sort of",
        "i mean",
        "basically",
        "actually",
    }
)

PAUSE_QUALITY_LABELS = {
    "rushing": "Rushing — very few pauses",
    "brisk": "A bit brisk",
    "balanced": "Well-paced",
    "excessive": "Too many pauses",
}


def _pause_quality(pause_ratio: float) -> str:
    if pause_ratio >= 0.30:
        return "excessive"
    if pause_ratio >= 0.12:
        return "balanced"
    if pause_ratio >= 0.03:
        return "brisk"
    return "rushing"


def _presence_score(
    pace_ratio: float,
    energy: float,
    fillers_per_minute: float,
    pause_ratio: float,
) -> int:
    score = 100.0
    score -= min(30.0, abs(pace_ratio - 1.0) * 60.0)
    score -= min(30.0, max(0.0, 55.0 - energy))
    score -= min(25.0, fillers_per_minute * 8.0)
    score -= min(25.0, max(0.0, pause_ratio - 0.25) * 80.0)
    return int(max(0, min(100, round(score))))

_filler_detector: "FillerDetector | None" = None


def bytes_to_audio(audio_bytes: bytes) -> np.ndarray:
    """Decode raw audio bytes into a mono float32 array at ``TARGET_SR``.

    Tries a WAV/container decode first; falls back to interpreting the
    buffer as raw 16-bit little-endian PCM (the format mic clients and
    AssemblyAI realtime streaming send).
    """
    try:
        audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=TARGET_SR, mono=True)
    except Exception:  # noqa: BLE001
        raw = np.frombuffer(audio_bytes, dtype="<i2").astype(np.float32) / 32768.0
        audio = raw.copy()
    if len(audio) == 0:
        raise ValueError("no audio samples decoded")
    return audio


def _syllable_nuclei_and_pauses(
    audio: np.ndarray,
) -> tuple[int, float, int, float]:
    """Estimate syllable count and pause statistics from the RMS envelope."""
    duration = len(audio) / TARGET_SR
    rms = librosa.feature.rms(y=audio, hop_length=512)[0]
    if rms.size == 0:
        return 0, 0.0, 0, duration

    threshold = max(np.percentile(rms, 40), float(np.max(rms)) * 0.05)
    rms_fps = TARGET_SR / 512.0
    min_gap_frames = max(1, int(MIN_PEAK_GAP_S * rms_fps))

    peaks, _ = find_peaks(rms, height=threshold, distance=min_gap_frames)
    syllables = int(len(peaks))

    voiced = rms >= threshold
    pause_frames = 0
    pause_events = 0
    in_pause = False
    min_pause_frames = int(MIN_PAUSE_S * rms_fps)
    for frame in range(voiced.size):
        if not voiced[frame]:
            pause_frames += 1
            in_pause = True
        elif in_pause:
            if frame > min_pause_frames:
                pause_events += 1
            in_pause = False
    if in_pause and voiced.size > min_pause_frames:
        pause_events += 1

    pause_ratio = pause_frames / voiced.size if voiced.size else 0.0
    return syllables, duration, pause_events, pause_ratio


def count_fillers_from_text(transcript: str) -> int:
    """Count filler-word occurrences in a transcript (case-insensitive)."""
    lowered = transcript.lower()
    total = 0
    for filler in FILLER_WORDS:
        total += lowered.count(filler)
    return total


class FillerDetector:
    """Optional realtime filler-word counter backed by AssemblyAI streaming."""

    def __init__(self) -> None:
        self._transcriber = None
        self._transcripts: deque[str] = deque(maxlen=40)
        self._sample_rate = TARGET_SR
        self._explicitly_closed = False

        api_key = os.getenv("ASSEMBLYAI_API_KEY")
        if not api_key:
            logger.info("ASSEMBLYAI_API_KEY not set; filler detection disabled")
            return

        try:
            import assemblyai as aai

            aai.settings.api_key = api_key

            def on_data(transcript: aai.RealtimeTranscript) -> None:
                if transcript.text:
                    self._transcripts.append(transcript.text)

            def on_error(error: Exception) -> None:
                logger.warning("AssemblyAI streaming error: %s", error)

            self._transcriber = aai.RealtimeTranscriber(
                sample_rate=self._sample_rate,
                on_data=on_data,
                on_error=on_error,
            )
            self._transcriber.connect()
            logger.info("AssemblyAI realtime transcriber connected")
        except Exception as exc:  # noqa: BLE001
            logger.warning("AssemblyAI unavailable; fillers default to 0: %s", exc)
            self._transcriber = None

    def feed(self, audio_bytes: bytes) -> None:
        if self._transcriber is not None:
            self._transcriber.stream(audio_bytes)

    @property
    def filler_count(self) -> int:
        if self._transcriber is None:
            return 0
        return count_fillers_from_text(" ".join(self._transcripts))

    def reset(self) -> None:
        """Clear accumulated transcripts so counts reflect the current attempt."""
        self._transcripts.clear()

    def close(self) -> None:
        if self._explicitly_closed:
            return
        self._explicitly_closed = True
        if self._transcriber is not None:
            try:
                self._transcriber.close()
            except Exception as exc:  # noqa: BLE001
                logger.debug("AssemblyAI close failed: %s", exc)
            self._transcriber = None


def get_filler_detector() -> FillerDetector:
    """Return the process-wide filler detector, creating it lazily."""
    global _filler_detector
    if _filler_detector is None:
        _filler_detector = FillerDetector()
    return _filler_detector


def compute_metrics(
    audio_bytes: bytes,
    baseline_pace_wpm: float | None = None,
) -> dict:
    """Analyze one audio buffer and return JSON-serializable metrics.

    Keys produced: ``pace_wpm``, ``baseline_pace_wpm``, ``pace_ratio``,
    ``fillers``, ``fillers_per_minute``, ``pause_count``, ``pause_ratio``,
    ``silence_ratio``, ``duration_sec``.
    """
    audio = bytes_to_audio(audio_bytes)
    duration = len(audio) / TARGET_SR
    syllables, _duration, pause_events, pause_ratio = _syllable_nuclei_and_pauses(
        audio
    )
    pace_wpm = round(
        syllables / duration * 60.0 / AVG_SYLLABLES_PER_WORD, 1
    )
    baseline = baseline_pace_wpm if baseline_pace_wpm else 150.0

    detector = get_filler_detector()
    detector.feed(audio_bytes)
    fillers = detector.filler_count

    rms_all = librosa.feature.rms(y=audio, hop_length=512)[0]
    mean_rms = float(np.mean(rms_all)) if rms_all.size else 0.0
    db = 20.0 * np.log10(mean_rms + 1e-9)
    energy = int(max(0.0, min(100.0, (db + 60.0) / 60.0 * 100.0)))

    fillers_per_min = round(fillers / max(duration, 0.001), 1)
    quality = _pause_quality(pause_ratio)
    silence_ratio = round(pause_ratio, 4)
    pace_ratio = round(pace_wpm / max(baseline, 1.0), 2)
    presence = _presence_score(pace_ratio, energy, fillers_per_min, pause_ratio)

    return {
        "pace_wpm": pace_wpm,
        "baseline_pace_wpm": round(baseline, 1),
        "pace_ratio": pace_ratio,
        "fillers": fillers,
        "fillers_per_minute": fillers_per_min,
        "pause_count": pause_events,
        "pause_ratio": round(pause_ratio, 4),
        "silence_ratio": silence_ratio,
        "duration_sec": round(duration, 2),
        "energy": energy,
        "pause_quality": quality,
        "pause_quality_label": PAUSE_QUALITY_LABELS[quality],
        "presence": presence,
    }