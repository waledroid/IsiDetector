import asyncio
import json
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()


def _get_handler():
    """Lazy import to avoid circular dependency with app.py."""
    from isitec_api.app import stream_handler
    return stream_handler


@router.websocket("/ws/video")
async def ws_video(websocket: WebSocket):
    """Stream JPEG frames over WebSocket — replaces MJPEG /video_feed.

    Sends binary messages (raw JPEG bytes). If the client can't keep up,
    frames are silently dropped rather than buffered.
    """
    await websocket.accept()
    handler = _get_handler()

    try:
        while True:
            # Wait for a new frame (non-blocking async)
            frame_ready = await asyncio.to_thread(handler.frame_ready.wait, 0.1)
            if frame_ready:
                handler.frame_ready.clear()

            frame_bytes = handler.latest_annotated
            if not handler.running or frame_bytes is None:
                frame_bytes = handler._standby_frame

            try:
                await websocket.send_bytes(frame_bytes)
            except (WebSocketDisconnect, RuntimeError):
                break

            # Throttle to ~30fps max to avoid flooding
            await asyncio.sleep(0.033)

    except WebSocketDisconnect:
        pass


@router.websocket("/ws/stats")
async def ws_stats(websocket: WebSocket):
    """Push live stats JSON over WebSocket — replaces 2s polling of /api/stats.

    Sends JSON text messages at ~2Hz (every 500ms).
    """
    await websocket.accept()
    handler = _get_handler()

    try:
        while True:
            stats = handler.get_stats()
            try:
                await websocket.send_json(stats)
            except (WebSocketDisconnect, RuntimeError):
                break

            # 500ms interval — matches the old 2s polling but with 4x responsiveness
            await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        pass
