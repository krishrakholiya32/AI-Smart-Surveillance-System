# 🎥 AI Smart Surveillance System

Real-time, multi-camera threat detection — person tracking, weapon detection, restricted-zone intrusion, running/loitering/crowd detection — running entirely on CPU, with a live web dashboard.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react)
![YOLO](https://img.shields.io/badge/YOLOv11-Ultralytics-purple)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)
![License](https://img.shields.io/badge/License-MIT-green)

> Originally a single-file Streamlit prototype, rebuilt as a full-stack app: FastAPI backend running the detection pipeline, React dashboard streaming live annotated video over WebSockets, Postgres-backed persistence for cameras/zones/alerts.

**Demo:** see [`assets/demo.mp4`](assets/demo.mp4)

---

## Features

- **Person detection** — YOLOv11s (COCO-pretrained)
- **Pose estimation** — YOLOv11n-pose, used for keypoint-aware zone intrusion checks
- **Weapon detection** — custom-trained YOLOv11 (gun & knife), runs on high-res crops around each detected person rather than the full downscaled frame, for better recall at distance
- **Restricted zone intrusion** — draw arbitrary polygon zones directly on the live feed; alerts fire when a person's keypoints enter the zone
- **Running detection** — speed-based, normalized by person height so it works regardless of distance from the camera
- **Loitering detection** — dwell-time based, per tracked person
- **Crowd detection** — alerts when the number of people in frame crosses a configurable limit
- **Multi-camera support** — webcam, DroidCam (phone over Wi-Fi), or any RTSP/HTTP IP camera; each camera runs its own independent detection worker
- **Live dashboard** — WebSocket-streamed annotated video, per-camera and aggregate metrics, all in a multi-camera grid
- **Adjustable detection thresholds** — tune confidence/speed/dwell/crowd limits live from the Settings page, with a one-click reset to factory defaults — no restart needed
- **Alerts & events** — snapshot capture on detection, confirm/dismiss workflow, filterable history
- **Login lockout** — 5 failed attempts locks that account for 15 minutes

---

## Architecture

```
                    ┌──────────────┐
   Camera sources → │   FastAPI    │ → Postgres (cameras, zones, alerts, events)
 (webcam / DroidCam │  detection   │
   / RTSP)          │   pipeline   │ → WebSocket → React dashboard (live video + metrics)
                    └──────────────┘
                           │
                    nginx reverse proxy
                    (single entrypoint: :80)
```

Each camera gets its own worker thread running all three YOLO models per frame, tracked centroids for re-identification across frames, and a temporal smoother to debounce flickering detections before anything is shown or alerted on.

---

## Tech Stack

| Layer | Stack |
|---|---|
| Detection | Ultralytics YOLOv11 (person/pose/weapon), OpenCV, ONNX Runtime (auto-preferred over PyTorch for ~2-3x faster CPU inference) |
| Backend | FastAPI, SQLAlchemy (async), PostgreSQL, WebSockets, JWT auth |
| Frontend | React 18, TypeScript, Vite, Tailwind CSS, Zustand |
| Infra | Docker Compose, nginx |

---

## Quickstart (local)

**1. Get the model files** — not committed to git (binary, large). Place these three files in the project root:
- `yolo11s.pt` — [download](https://github.com/ultralytics/assets/releases/latest/download/yolo11s.pt)
- `yolo11n-pose.pt` — [download](https://github.com/ultralytics/assets/releases/latest/download/yolo11n-pose.pt)
- `weapon.pt` — your own custom-trained weapon detector (see `weapon_detection_training.ipynb` for the training pipeline used here)

Then run:
```bash
python scripts/setup_models.py
```
This copies them into `models/`, which is what the backend actually mounts.

**2. Configure environment**
```bash
cp .env.example .env
```
Edit `.env` — at minimum set `SECRET_KEY` (`openssl rand -hex 32`) and `ADMIN_PASSWORD`. The app refuses to start without both.

**3. Start it**
```bash
bash scripts/start-demo.sh
```
This launches Docker Desktop (if needed), a local webcam-to-MJPEG bridge (`scripts/webcam_server.py`) so the container can see your laptop's webcam, then the full stack. First boot takes 1-2 minutes longer — the backend exports the `.pt` models to ONNX once and reuses them after.

Open **http://localhost**, log in with `admin` / whatever you set as `ADMIN_PASSWORD`, go to **Cameras** and hit **Start**.

When you're done:
```bash
bash scripts/stop-demo.sh
```

*(Or skip the helper scripts and run `docker compose up -d` / `down` directly — the scripts just also handle Docker Desktop and the webcam bridge for you.)*

---

## Adding Cameras

From the **Cameras** page, add a source:

| Source | Value |
|---|---|
| Laptop webcam | `http://host.docker.internal:8765/video` — requires `scripts/webcam_server.py` running on the host (containers can't see host hardware directly) |
| Phone (DroidCam) | `http://<phone-ip>:4747/video` — phone and PC must be on the same Wi-Fi/hotspot |
| RTSP/IP camera | `rtsp://user:pass@<camera-ip>/stream` |

Add as many as you like — each runs its own independent detection worker. Note: running multiple cameras splits the same CPU budget, so per-camera FPS drops roughly proportionally — this is sized for one camera at a time on a typical laptop/free-tier cloud CPU.

---

## Detection Thresholds

All five tunable values (person/weapon confidence, run speed, loiter seconds, crowd limit) live in `.env` as startup defaults, and are editable live from **Settings** afterward without restarting anything — the backend mutates the same shared config object every camera worker reads each frame.

---

## Performance (CPU-only, no GPU)

| Mode | FPS (single camera) |
|---|---|
| `.pt` (PyTorch) | 5-10 |
| `.onnx` (ONNX Runtime, used automatically when present) | 10-18 |

The pipeline is paced to a 10 FPS target — anything above that is headroom, not wasted capacity.

---

## Project Structure

```
backend/            FastAPI app — API, detection pipeline, camera worker manager
frontend/            React dashboard
nginx/                Reverse proxy config
models/                .pt / .onnx files (gitignored — see Quickstart)
scripts/
  setup_models.py       copies root .pt files into models/
  export_onnx.py         exports .pt → .onnx (runs automatically on first boot)
  webcam_server.py        MJPEG bridge so Docker can see your host webcam
  start-demo.sh / stop-demo.sh   local run/stop helpers
weapon_detection_training.ipynb   training pipeline for the custom weapon model
```

---

## License

MIT
