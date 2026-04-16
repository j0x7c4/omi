"""
omiGlass BLE camera capture script.

Connects to the OMI Glass device, triggers photo capture every 5 seconds,
and saves received JPEG images to ./captures/
"""

import asyncio
import os
import time
from bleak import BleakClient, BleakScanner

# BLE UUIDs
SERVICE_UUID         = "19b10000-e8f2-537e-4f6c-d104768a1214"
PHOTO_DATA_UUID      = "19b10005-e8f2-537e-4f6c-d104768a1214"
PHOTO_CONTROL_UUID   = "19b10006-e8f2-537e-4f6c-d104768a1214"

DEVICE_NAME = "OMI Glass"
OUTPUT_DIR  = "./captures"

# Photo reassembly state
_prev_chunk = -1
_buffer     = bytearray()
_orientation = 0

os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_photo(data: bytearray, orientation: int):
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(OUTPUT_DIR, f"photo_{ts}.jpg")
    with open(path, "wb") as f:
        f.write(data)
    print(f"[+] Saved {path} ({len(data)} bytes, orientation={orientation})")


def on_photo_chunk(sender, data: bytearray):
    global _prev_chunk, _buffer, _orientation

    # End-of-photo marker: 0xFF 0xFF
    if data[0] == 0xFF and data[1] == 0xFF:
        if _prev_chunk >= 0 and len(_buffer) > 0:
            save_photo(bytes(_buffer), _orientation)
        _prev_chunk = -1
        _buffer = bytearray()
        return

    packet_id = data[0] + (data[1] << 8)
    payload   = data[2:]

    if _prev_chunk == -1:
        if packet_id != 0:
            return  # Wait for first packet
        _prev_chunk  = 0
        _buffer      = bytearray()
        _orientation = payload[0]
        payload      = payload[1:]
    else:
        if packet_id != _prev_chunk + 1:
            print(f"[!] Chunk gap: expected {_prev_chunk + 1}, got {packet_id} — dropping frame")
            _prev_chunk = -1
            _buffer = bytearray()
            return
        _prev_chunk = packet_id

    _buffer.extend(payload)


async def main():
    print(f"[*] Scanning for '{DEVICE_NAME}'...")
    device = await BleakScanner.find_device_by_name(DEVICE_NAME, timeout=15)
    if device is None:
        print(f"[!] Device '{DEVICE_NAME}' not found. Make sure it's powered on and nearby.")
        return

    print(f"[*] Found: {device.name} ({device.address})")

    async with BleakClient(device) as client:
        print("[*] Connected.")

        # Subscribe to photo data notifications
        await client.start_notify(PHOTO_DATA_UUID, on_photo_chunk)
        print("[*] Subscribed to photo notifications.")

        # Trigger photo capture every 5 seconds
        await client.write_gatt_char(PHOTO_CONTROL_UUID, bytes([0x05]))
        print("[*] Capture triggered (every 5s). Saving to ./captures/ — Ctrl+C to stop.\n")

        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\n[*] Stopped.")

        await client.stop_notify(PHOTO_DATA_UUID)


if __name__ == "__main__":
    asyncio.run(main())
