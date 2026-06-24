"""
WebSocket endpoint: /ws/{camera_id}
Clients receive JSON messages:
  { camera_id, frame (base64 JPEG), metrics, alerts, ts }
"""
import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query

from app.core.security import decode_token
from app.services import camera_manager

router = APIRouter(tags=["stream"])


@router.websocket("/ws/{camera_id}")
async def stream(websocket: WebSocket, camera_id: int, token: str = Query(...)):
    user_id = decode_token(token)
    if not user_id:
        await websocket.close(code=4001)
        return

    await websocket.accept()
    q = camera_manager.subscribe(camera_id)

    try:
        while True:
            try:
                payload = await asyncio.wait_for(q.get(), timeout=5.0)
                await websocket.send_text(json.dumps(payload))
            except asyncio.TimeoutError:
                # Send keep-alive ping
                await websocket.send_text(json.dumps({"type": "ping"}))
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        camera_manager.unsubscribe(camera_id, q)
