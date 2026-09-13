"""Stream live microphone audio to the voice-coach WebSocket.

Captures 16 kHz mono PCM from the default (or ``--device``) microphone
with ``sounddevice``, buffers ``--window`` seconds of raw PCM per chunk,
and prints the JSON (metrics + coaching) frames received back.

Usage:
    python client.py --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import argparse
import json
import queue
import time

import numpy as np
import sounddevice as sd
from websockets.sync.client import connect

DEFAULT_SAMPLE_RATE = 16_000
CHANNELS = 1
DTYPE = "int16"

_CONTROL = {"type": "set_baseline", "pace_wpm": 150.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mic -> voice-coach WebSocket")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", type=int, default=None, help="input device index")
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--blocksize", type=int, default=1600)
    parser.add_argument("--window", type=float, default=3.0, help="seconds per chunk")
    parser.add_argument("--baseline", type=float, default=150.0, help="baseline pace wpm")
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="print available audio devices and exit",
    )
    return parser.parse_args()


def print_received(message: str | bytes) -> None:
    if isinstance(message, bytes):
        print("<< raw bytes (control frames are string JSON), len:", len(message))
        return
    try:
        payload = json.loads(message)
    except json.JSONDecodeError:
        print("<<", message)
        return
    print("<<", json.dumps(payload, indent=2))


def run(args: argparse.Namespace) -> None:
    uri = f"ws://{args.host}:{args.port}/ws/stream"
    audio_queue: queue.Queue = queue.Queue(maxsize=128)
    last_flush = time.monotonic()

    def callback(indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            print("sounddevice status:", status)
        audio_queue.put(indata[:, 0].copy())

    with connect(uri, max_size=None) as ws:
        _CONTROL["pace_wpm"] = args.baseline
        ws.send(json.dumps(_CONTROL))
        print("Connected:", uri)

        with sd.InputStream(
            samplerate=args.sample_rate,
            blocksize=args.blocksize,
            device=args.device,
            channels=CHANNELS,
            dtype=DTYPE,
            callback=callback,
        ):
            print(
                "Streaming mic ->", uri,
                f"(sr={args.sample_rate}, window={args.window}s). Ctrl+C to stop.",
            )
            while True:
                now = time.monotonic()
                if now - last_flush >= args.window:
                    chunks: list[bytes] = []
                    while True:
                        try:
                            chunks.append(audio_queue.get_nowait().tobytes())
                        except queue.Empty:
                            break
                    if chunks:
                        ws.send(b"".join(chunks))
                        last_flush = now

                try:
                    message = ws.recv(timeout=0.2)
                except TimeoutError:
                    continue
                print_received(message)


def main() -> None:
    args = parse_args()
    if args.list_devices:
        print(sd.query_devices())
        return
    try:
        run(args)
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()