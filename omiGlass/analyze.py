"""
omiGlass real-time camera analysis script.

Connects to OMI Glass via BLE, captures photos every 5 seconds,
and sends each photo to the local gemma4 model (via omlx) for analysis.
"""

import asyncio
import base64
import os
import time

import httpx
from bleak import BleakClient, BleakScanner

# BLE UUIDs
SERVICE_UUID       = "19b10000-e8f2-537e-4f6c-d104768a1214"
PHOTO_DATA_UUID    = "19b10005-e8f2-537e-4f6c-d104768a1214"
PHOTO_CONTROL_UUID = "19b10006-e8f2-537e-4f6c-d104768a1214"

DEVICE_NAME = "OMI Glass"
OUTPUT_DIR  = "./captures"

# omlx config
OMLX_API_BASE = os.environ.get("OMLX_API_BASE", "http://localhost:8000")
OMLX_MODEL    = os.environ.get("OMLX_MODEL", "gemma-4-26b-a4b-it-4bit")
OMLX_API_KEY  = os.environ.get("OMLX_API_KEY", "omlx")
PROMPT        = os.environ.get("VISION_PROMPT", "请简洁描述这张图片里有什么。")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Photo reassembly state
_prev_chunk  = -1
_buffer      = bytearray()
_orientation = 0
_photo_queue: asyncio.Queue = asyncio.Queue()


def on_photo_chunk(sender, data: bytearray):
    global _prev_chunk, _buffer, _orientation

    if data[0] == 0xFF and data[1] == 0xFF:
        if _prev_chunk >= 0 and len(_buffer) > 0:
            _photo_queue.put_nowait(bytes(_buffer))
        _prev_chunk = -1
        _buffer = bytearray()
        return

    packet_id = data[0] + (data[1] << 8)
    payload   = data[2:]

    if _prev_chunk == -1:
        if packet_id != 0:
            return
        _prev_chunk  = 0
        _buffer      = bytearray()
        _orientation = payload[0]
        payload      = payload[1:]
    else:
        if packet_id != _prev_chunk + 1:
            print(f"  [!] Chunk gap {_prev_chunk+1}→{packet_id}, dropping frame")
            _prev_chunk = -1
            _buffer = bytearray()
            return
        _prev_chunk = packet_id

    _buffer.extend(payload)


async def analyze_photo(jpeg_bytes: bytes) -> str:
    b64 = base64.b64encode(jpeg_bytes).decode()
    payload = {
        "model": OMLX_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }],
        "max_tokens": 256,
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


async def analysis_worker():
    """Consume photos from queue and analyze them."""
    while True:
        jpeg_bytes = await _photo_queue.get()
        ts = time.strftime("%Y%m%d_%H%M%S")

        # Save photo
        path = os.path.join(OUTPUT_DIR, f"photo_{ts}.jpg")
        with open(path, "wb") as f:
            f.write(jpeg_bytes)

        print(f"\n[Photo] {path} ({len(jpeg_bytes)} bytes)")
        print("[Analyzing...]")

        try:
            result = await analyze_photo(jpeg_bytes)
            print(f"[gemma4] {result}")
        except Exception as e:
            print(f"[Error] Analysis failed: {e}")

        _photo_queue.task_done()


async def main():
    print(f"[*] Scanning for '{DEVICE_NAME}'...")
    device = await BleakScanner.find_device_by_name(DEVICE_NAME, timeout=15)
    if device is None:
        print(f"[!] '{DEVICE_NAME}' not found. Make sure it's powered on and nearby.")
        return

    print(f"[*] Found: {device.name} ({device.address})")

    async with BleakClient(device) as client:
        print("[*] Connected.")

        # Start analysis worker
        worker = asyncio.create_task(analysis_worker())

        # Subscribe to photo notifications
        await client.start_notify(PHOTO_DATA_UUID, on_photo_chunk)
        print("[*] Subscribed to photo stream.")

        # Trigger capture every 5 seconds
        await client.write_gatt_char(PHOTO_CONTROL_UUID, bytes([0x05]))
        print(f"[*] Capture started (5s interval). Model: {OMLX_MODEL}")
        print(f"[*] Prompt: {PROMPT}")
        print("[*] Ctrl+C to stop.\n")

        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\n[*] Stopped.")

        worker.cancel()
        await client.stop_notify(PHOTO_DATA_UUID)


if __name__ == "__main__":
    asyncio.run(main())
