"""
CameraManager: spins up one worker thread per camera.
Each worker runs the detection pipeline and broadcasts JPEG frames
+ metrics + alerts to all WebSocket subscribers via asyncio queues.
"""
import asyncio
import base64
import logging
import threading
import time
from typing import Dict, List, Optional, Set

import cv2
import numpy as np

from app.core.config import settings
from app.services.detection.pipeline import DetectionPipeline
from app.services.model_loader import get_models

log = logging.getLogger(__name__)

TARGET_FPS = 10
FRAME_INTERVAL = 1.0 / TARGET_FPS

# Maps camera_id → set of asyncio.Queue (one per connected WebSocket client)
_subscribers: Dict[int, Set[asyncio.Queue]] = {}
_subscribers_lock = threading.Lock()

# Maps camera_id → (thread, stop_event)
_workers: Dict[int, tuple] = {}

# Maps camera_id → live DetectionPipeline instance (for hot zone updates)
_pipelines: Dict[int, DetectionPipeline] = {}
_pipelines_lock = threading.Lock()


class _FrameGrabber:
    """
    Background thread that continuously reads from a capture source
    and stores only the most recent frame. Prevents stale-frame lag
    when the detection pipeline is slower than the source frame rate.
    Auto-reconnects when the stream drops.
    """
    def __init__(self, source):
        self._source = source
        self._cap = self._open()
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._running = True
        self._thread = threading.Thread(target=self._grab, daemon=True)
        self._thread.start()

    def _open(self):
        cap = cv2.VideoCapture(self._source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _grab(self):
        consecutive_failures = 0
        while self._running:
            ret, frame = self._cap.read()
            if ret:
                consecutive_failures = 0
                with self._lock:
                    self._frame = frame
            else:
                consecutive_failures += 1
                time.sleep(0.05)
                if consecutive_failures >= 10:
                    log.warning("FrameGrabber: stream lost, reconnecting...")
                    self._cap.release()
                    time.sleep(1.0)
                    self._cap = self._open()
                    consecutive_failures = 0

    def read(self):
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def release(self):
        self._running = False
        self._thread.join(timeout=2)
        self._cap.release()


def subscribe(camera_id: int) -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue(maxsize=2)
    with _subscribers_lock:
        if camera_id not in _subscribers:
            _subscribers[camera_id] = set()
        _subscribers[camera_id].add(q)
    return q


def unsubscribe(camera_id: int, q: asyncio.Queue):
    with _subscribers_lock:
        if camera_id in _subscribers:
            _subscribers[camera_id].discard(q)


def _broadcast(camera_id: int, payload: dict, loop: asyncio.AbstractEventLoop):
    with _subscribers_lock:
        queues = list(_subscribers.get(camera_id, []))
    for q in queues:
        try:
            # Drop oldest frame if queue is full to stay real-time
            if q.full():
                try:
                    q.get_nowait()
                except Exception:
                    pass
            loop.call_soon_threadsafe(q.put_nowait, payload)
        except Exception:
            pass


def _worker(camera_id: int, source: str, zones: List[dict],
            stop_event: threading.Event, loop: asyncio.AbstractEventLoop,
            alert_callback):
    models = get_models()
    pipeline = DetectionPipeline(models, zones, settings)
    with _pipelines_lock:
        _pipelines[camera_id] = pipeline

    grabber = _FrameGrabber(source if source != "0" else 0)
    log.info(f"Camera {camera_id} worker started (source={source})")

    # Wait up to 5s for first frame
    for _ in range(100):
        ret, _ = grabber.read()
        if ret:
            break
        time.sleep(0.05)

    while not stop_event.is_set():
        t0 = time.time()

        ret, frame = grabber.read()
        if not ret or frame is None:
            time.sleep(0.05)
            continue

        annotated, metrics, new_alerts = pipeline.process(frame)

        # JPEG encode at lower quality for faster WebSocket transmission
        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 65])
        frame_b64 = base64.b64encode(buf).decode("ascii")

        payload = {
            "camera_id": camera_id,
            "frame": frame_b64,
            "metrics": metrics,
            "alerts": new_alerts,
            "ts": time.time(),
        }
        _broadcast(camera_id, payload, loop)

        # Fire alert callbacks for persistence
        if new_alerts and alert_callback:
            for alert in new_alerts:
                _, snap_buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
                snapshot = snap_buf.tobytes()
                loop.call_soon_threadsafe(
                    asyncio.ensure_future,
                    alert_callback(camera_id, alert, snapshot),
                )

        # Pace to TARGET_FPS; if inference was slower, skip immediately
        elapsed = time.time() - t0
        sleep_time = FRAME_INTERVAL - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    grabber.release()
    with _pipelines_lock:
        _pipelines.pop(camera_id, None)
    log.info(f"Camera {camera_id} worker stopped")


def start_camera(camera_id: int, source: str, zones: List[dict],
                 loop: asyncio.AbstractEventLoop, alert_callback=None):
    if camera_id in _workers:
        stop_camera(camera_id)
    stop_event = threading.Event()
    t = threading.Thread(
        target=_worker,
        args=(camera_id, source, zones, stop_event, loop, alert_callback),
        daemon=True,
    )
    _workers[camera_id] = (t, stop_event)
    t.start()


def stop_camera(camera_id: int):
    if camera_id in _workers:
        t, stop_event = _workers.pop(camera_id)
        stop_event.set()
        t.join(timeout=3)


def update_zones(camera_id: int, zones: List[dict]):
    with _pipelines_lock:
        pipeline = _pipelines.get(camera_id)
    if pipeline:
        pipeline.update_zones(zones)


def is_running(camera_id: int) -> bool:
    return camera_id in _workers and _workers[camera_id][0].is_alive()


def stop_all():
    for cid in list(_workers.keys()):
        stop_camera(cid)
