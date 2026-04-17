"""
omiGlass voice + vision script.

Hold Cmd+Shift to record voice → release to transcribe with Whisper medium
→ triggers single photo capture → sends image + transcript to gemma4.

Usage:
    conda run -n omi python voice_capture.py

Controls:
    Hold Cmd+Shift  — record voice
    Release         — transcribe + capture photo + analyze
    Ctrl+C          — quit
"""

import asyncio
import base64
import os
import struct
import time
import threading
import numpy as np
import httpx

# Ensure libopus is findable on macOS (Homebrew).
# Must be set before the process starts; if missing, re-exec with it set.
_OPUS_LIB_PATH = "/opt/homebrew/lib"
if _OPUS_LIB_PATH not in os.environ.get("DYLD_LIBRARY_PATH", ""):
    import sys
    os.environ["DYLD_LIBRARY_PATH"] = _OPUS_LIB_PATH + ":" + os.environ.get("DYLD_LIBRARY_PATH", "")
    os.execv(sys.executable, [sys.executable] + sys.argv)

from bleak import BleakClient, BleakScanner
from pynput import keyboard

# ── BLE ──────────────────────────────────────────────────────────────────────
SERVICE_UUID       = "19b10000-e8f2-537e-4f6c-d104768a1214"
PHOTO_DATA_UUID    = "19b10005-e8f2-537e-4f6c-d104768a1214"
PHOTO_CONTROL_UUID = "19b10006-e8f2-537e-4f6c-d104768a1214"
AUDIO_DATA_UUID    = "19b10001-e8f2-537e-4f6c-d104768a1214"
AUDIO_CODEC_UUID   = "19b10002-e8f2-537e-4f6c-d104768a1214"
DEVICE_NAME        = "OMI Glass"

# ── Audio ─────────────────────────────────────────────────────────────────────
SAMPLE_RATE    = 16000
OPUS_FRAME_MS  = 20
HEADER_BYTES   = 3  # 2-byte index + 1-byte sub-index

# ── omlx ─────────────────────────────────────────────────────────────────────
OMLX_API_BASE = os.environ.get("OMLX_API_BASE", "http://localhost:8000")
OMLX_MODEL    = os.environ.get("OMLX_MODEL",    "gemma-4-26b-a4b-it-4bit")
OMLX_API_KEY  = os.environ.get("OMLX_API_KEY",  "omlx")

OUTPUT_DIR = "./captures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── State ─────────────────────────────────────────────────────────────────────
_recording        = False
_opus_frames: list[bytes] = []
_opus_lock        = threading.Lock()

# Photo reassembly
_prev_chunk  = -1
_buf         = bytearray()
_photo_queue: asyncio.Queue = asyncio.Queue()

# BLE client reference (set after connect)
_ble_client: BleakClient | None = None
_loop: asyncio.AbstractEventLoop | None = None

# Hotkey state
_cmd_held   = False
_shift_held = False

# ── Opus decoder ──────────────────────────────────────────────────────────────
try:
    import opuslib
    _decoder = opuslib.Decoder(SAMPLE_RATE, 1)
    print("[*] Opus decoder ready (opuslib)")
except Exception as e:
    _decoder = None
    print(f"[!] opuslib not available: {e}. Audio decoding disabled.")


def decode_opus_frames(frames: list[bytes]) -> np.ndarray:
    """Decode list of Opus frames to float32 PCM."""
    if _decoder is None or not frames:
        return np.array([], dtype=np.float32)
    pcm_chunks = []
    frame_size = SAMPLE_RATE * OPUS_FRAME_MS // 1000  # 320 samples
    for frame in frames:
        try:
            pcm = _decoder.decode(bytes(frame), frame_size)
            samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            pcm_chunks.append(samples)
        except Exception:
            pass
    return np.concatenate(pcm_chunks) if pcm_chunks else np.array([], dtype=np.float32)


# ── Whisper ───────────────────────────────────────────────────────────────────
_whisper_model = None

def load_whisper():
    global _whisper_model
    print("[*] Loading Whisper medium (mlx)...")
    import mlx_whisper
    _whisper_model = "mlx-community/whisper-medium-mlx"
    print("[*] Whisper ready.")

def transcribe(audio: np.ndarray) -> str:
    if _whisper_model is None or len(audio) == 0:
        return ""
    import mlx_whisper
    result = mlx_whisper.transcribe(audio, path_or_hf_repo=_whisper_model)
    return result.get("text", "").strip()


# ── BLE callbacks ─────────────────────────────────────────────────────────────
def on_audio(sender, data: bytearray):
    if not _recording:
        return
    # Strip 3-byte header, extract raw Opus frame
    if len(data) <= HEADER_BYTES:
        return
    # First 2 bytes: packet index, 3rd byte: sub-index, rest: opus frame
    frame = bytes(data[HEADER_BYTES:])
    with _opus_lock:
        _opus_frames.append(frame)


def on_photo(sender, data: bytearray):
    global _prev_chunk, _buf
    if data[0] == 0xFF and data[1] == 0xFF:
        if _prev_chunk >= 0 and len(_buf) > 0:
            _photo_queue.put_nowait(bytes(_buf))
        _prev_chunk = -1
        _buf = bytearray()
        return
    pid     = data[0] + (data[1] << 8)
    payload = data[2:]
    if _prev_chunk == -1:
        if pid != 0:
            return
        _prev_chunk = 0
        _buf        = bytearray()
        payload     = payload[1:]  # skip orientation byte
    else:
        if pid != _prev_chunk + 1:
            _prev_chunk = -1
            _buf = bytearray()
            return
        _prev_chunk = pid
    _buf.extend(payload)


# ── Gemma4 vision ─────────────────────────────────────────────────────────────
async def analyze(jpeg: bytes, prompt: str) -> str:
    b64 = base64.b64encode(jpeg).decode()
    payload = {
        "model": OMLX_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text",      "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }],
        "max_tokens": 512,
        "temperature": 0.7,
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{OMLX_API_BASE}/v1/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {OMLX_API_KEY}"},
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


# ── Processing pipeline ────────────────────────────────────────────────────────
async def process(frames: list[bytes]):
    t_total = time.perf_counter()
    print("[*] Processing...")

    # 1. Decode audio + transcribe
    print("    Transcribing audio...")
    t0 = time.perf_counter()
    audio = decode_opus_frames(frames)
    t_decode = time.perf_counter() - t0

    t0 = time.perf_counter()
    prompt = transcribe(audio) if len(audio) > 0 else ""
    t_whisper = time.perf_counter() - t0

    if prompt:
        print(f"    [Whisper] {prompt}  ({t_decode*1000:.0f}ms decode, {t_whisper:.1f}s transcribe)")
    else:
        prompt = "请描述这张图片里有什么？"
        print(f"    [Whisper] (silent) using default prompt  ({t_decode*1000:.0f}ms decode, {t_whisper:.1f}s transcribe)")

    # 2. Trigger single photo capture
    # Pause audio subscription so firmware can prioritize photo upload
    if _ble_client and _ble_client.is_connected:
        t0 = time.perf_counter()
        await _ble_client.stop_notify(AUDIO_DATA_UUID)
        await _ble_client.write_gatt_char(PHOTO_CONTROL_UUID, bytes([0xFF]))  # -1 as uint8
        print(f"    Photo triggered.  ({(time.perf_counter()-t0)*1000:.0f}ms)")
    else:
        print("    [!] BLE not connected, cannot trigger photo.")
        return

    # 3. Wait for photo (timeout 15s)
    t0 = time.perf_counter()
    try:
        jpeg = await asyncio.wait_for(_photo_queue.get(), timeout=15.0)
    except asyncio.TimeoutError:
        print("    [!] Photo timeout.")
        await _ble_client.start_notify(AUDIO_DATA_UUID, on_audio)
        return
    t_photo = time.perf_counter() - t0

    # Resume audio subscription
    await _ble_client.start_notify(AUDIO_DATA_UUID, on_audio)

    # Save photo
    ts   = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(OUTPUT_DIR, f"photo_{ts}.jpg")
    with open(path, "wb") as f:
        f.write(jpeg)
    print(f"    Saved {path} ({len(jpeg)} bytes, {t_photo:.1f}s transfer)")

    # 4. Send to gemma4
    print(f"    Sending to gemma4: \"{prompt}\"")
    t0 = time.perf_counter()
    result = await analyze(jpeg, prompt)
    t_gemma = time.perf_counter() - t0

    t_total = time.perf_counter() - t_total
    print(f"\n{'─'*60}")
    print(f"Prompt : {prompt}")
    print(f"gemma4 : {result}")
    print(f"{'─'*60}")
    print(f"Timing : decode={t_decode*1000:.0f}ms  whisper={t_whisper:.1f}s  photo={t_photo:.1f}s  gemma4={t_gemma:.1f}s  total={t_total:.1f}s")
    print(f"{'─'*60}\n")


# ── Hotkey (Cmd + Shift) ──────────────────────────────────────────────────────
def on_press(key):
    global _recording, _cmd_held, _shift_held, _opus_frames

    if key in (keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r):
        _cmd_held = True
    if key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r):
        _shift_held = True

    if _cmd_held and _shift_held and not _recording:
        _recording = True
        with _opus_lock:
            _opus_frames = []
        print("\n[●] Recording... (release Cmd+Shift to stop)")


def on_release(key):
    global _recording, _cmd_held, _shift_held, _opus_frames

    if key in (keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r):
        _cmd_held = False
    if key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r):
        _shift_held = False

    if _recording and not (_cmd_held and _shift_held):
        _recording = False
        with _opus_lock:
            frames = list(_opus_frames)
            _opus_frames = []
        duration = len(frames) * OPUS_FRAME_MS / 1000
        print(f"[■] Stopped ({duration:.1f}s, {len(frames)} frames)")
        if _loop:
            asyncio.run_coroutine_threadsafe(process(frames), _loop)


# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    global _ble_client, _loop
    _loop = asyncio.get_running_loop()

    # Load Whisper in background thread
    t = threading.Thread(target=load_whisper, daemon=True)
    t.start()

    print(f"[*] Scanning for '{DEVICE_NAME}'...")
    device = await BleakScanner.find_device_by_name(DEVICE_NAME, timeout=15)
    if device is None:
        print(f"[!] '{DEVICE_NAME}' not found.")
        return

    print(f"[*] Found: {device.name} ({device.address})")

    async with BleakClient(device) as client:
        _ble_client = client
        print("[*] Connected.")

        # Subscribe to audio and photo streams
        await client.start_notify(AUDIO_DATA_UUID,  on_audio)
        await client.start_notify(PHOTO_DATA_UUID,  on_photo)
        print("[*] Subscribed to audio + photo streams.")

        # Stop any auto-capture
        await client.write_gatt_char(PHOTO_CONTROL_UUID, bytes([0x00]))

        # Start hotkey listener
        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()

        print("\n[*] Whisper medium loading (first time downloads ~1.5GB)...")
        print("[*] Hold Cmd+Shift to record, release to analyze.")
        print("[*] Ctrl+C to quit.\n")

        try:
            while True:
                await asyncio.sleep(0.1)
        except KeyboardInterrupt:
            print("\n[*] Quit.")

        listener.stop()
        await client.stop_notify(AUDIO_DATA_UUID)
        await client.stop_notify(PHOTO_DATA_UUID)


if __name__ == "__main__":
    asyncio.run(main())
